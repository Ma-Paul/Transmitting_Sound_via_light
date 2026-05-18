/*
 * NAF Converter  –  Normalized Audio Format
 * ==========================================
 * Compile:  gcc -O2 -o naf naf_converter.c -lm
 *
 * Usage:
 *   ./naf encode [-S|-Q|-Q2] input.wav  output.naf
 *   ./naf decode              input.naf  output.wav
 *   ./naf info                input.naf
 *
 * Encode modes
 * ------------
 *   -S   Size      – small files, acceptable quality
 *   -Q   Quality   – good quality  (default)
 *   -Q2  Ultra     – best quality, larger files, slower encode
 *
 * NAF Binary Format
 * -----------------
 * Header (40 bytes, little-endian):
 *   [0]   4 bytes  char[4]   Magic "NAF1"
 *   [4]   4 bytes  float32   Global minimum frequency stored (Hz)
 *   [8]   4 bytes  float32   Global maximum frequency stored (Hz)
 *   [12]  4 bytes  uint32    Original sample rate
 *   [16]  4 bytes  float32   Analysis window length (seconds)
 *   [20]  4 bytes  uint32    Total number of slices
 *   [24]  4 bytes  uint32    Number of channels in original WAV
 *   [28]  4 bytes  float32   (unused, was amplitude threshold – kept for
 * compat) [32]  4 bytes  float32   Hop size between slices (seconds) [36]  4
 * bytes  uint8[4]  Reserved (zeros)
 *
 * Per slice (variable length):
 *   2 bytes  uint16   Number of frequency/loudness pairs N
 *   N * 8 bytes:
 *     4 bytes  float32  Normalised frequency  0.0 (min_hz) … 1.0 (max_hz)
 *     4 bytes  float32  Normalised loudness   0.0 (silence) … 1.0 (loudest)
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════ */

#define MAGIC "NAF1"
#define PI 3.14159265358979323846

/* Maximum peaks storable per slice (hard ceiling for malloc sizing) */
#define MAX_PEAKS_PER_SLICE 1024

/* ═══════════════════════════════════════════════════════════════════
 * Encode parameter sets  (one per quality mode)
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct
{
    const char *name;
    float window_sec;   /* FFT analysis window length          */
    float hop_sec;      /* hop between consecutive slices      */
    int fft_size;       /* zero-padded FFT size (power of 2)   */
    float max_freq_hz;  /* highest frequency to store          */
    int peaks_per_band; /* local-maxima peaks kept per band  */
    /* frequency bands used for peak picking [lo, hi] Hz */
    int n_bands;
    double bands[12][2];
} EncodeParams;

/* -S  Size mode */
static const EncodeParams PARAMS_S = {
    .name = "Size (-S)",
    .window_sec = 0.04f, /* 40ms window  */
    .hop_sec = 0.02f,    /* 20ms hop (50% overlap) */
    .fft_size = 4096,    /* 48000/4096 ≈ 11.7 Hz/bin */
    .max_freq_hz = 8000.0f,
    .peaks_per_band = 6,
    .n_bands = 5,
    .bands =
        {
            {20, 300},
            {300, 900},
            {900, 2500},
            {2500, 5000},
            {5000, 8000},
        },
};

/* -Q  Quality mode (default) */
static const EncodeParams PARAMS_Q = {
    .name = "Quality (-Q)",
    .window_sec = 0.04f, /* 40ms window  */
    .hop_sec = 0.01f,    /* 10ms hop (75% overlap) */
    .fft_size = 16384,   /* 48000/16384 ≈ 2.93 Hz/bin */
    .max_freq_hz = 10000.0f,
    .peaks_per_band = 16,
    .n_bands = 6,
    .bands =
        {
            {20, 150},
            {150, 500},
            {500, 1500},
            {1500, 4000},
            {4000, 7000},
            {7000, 10000},
        },
};

/* -Q9  Maximum quality mode — absolute best, no compromises
 * FFT: 65536 bins → 0.73 Hz/bin resolution at 48kHz
 * 10 bands covering full human hearing range (20 Hz – 20 kHz)
 * 64 peaks per band → up to 640 peaks per slice
 * Same 10ms hop as Q/Q2 (5ms causes OLA distortion)             */
static const EncodeParams PARAMS_Q9 = {
    .name = "Maximum Quality (-Q9)",
    .window_sec = 0.04f,
    .hop_sec = 0.010f,
    .fft_size = 65536, /* 48000/65536 ≈ 0.73 Hz/bin */
    .max_freq_hz = 20000.0f,
    .peaks_per_band = 64,
    .n_bands = 10,
    .bands =
        {
            {20, 80},       /* sub-bass     */
            {80, 200},      /* bass         */
            {200, 500},     /* upper bass   */
            {500, 1000},    /* low-mid      */
            {1000, 2000},   /* mid          */
            {2000, 4000},   /* upper-mid    */
            {4000, 6000},   /* presence     */
            {6000, 10000},  /* brilliance   */
            {10000, 16000}, /* air          */
            {16000, 20000}, /* ultra-high   */
        },
};

/* -Q2  Ultra quality mode
 * Same hop as -Q (10ms, 75% overlap) but with:
 *  - 4x larger FFT  → 1.46 Hz/bin resolution (vs 2.93 Hz in -Q)
 *  - More peaks per band (32 vs 16)
 *  - More bands (8 vs 6) covering higher frequencies
 *  - Higher max frequency (12 kHz vs 10 kHz)
 * This gives much finer pitch accuracy and more harmonic content
 * without the overlap-distortion that 5ms hops cause.             */
static const EncodeParams PARAMS_Q2 = {
    .name = "Ultra Quality (-Q2)",
    .window_sec = 0.04f, /* 40ms window                      */
    .hop_sec = 0.010f,   /* 10ms hop (75% overlap, same as Q) */
    .fft_size = 32768,   /* 48000/32768 ≈ 1.46 Hz/bin        */
    .max_freq_hz = 12000.0f,
    .peaks_per_band = 32,
    .n_bands = 8,
    .bands =
        {
            {20, 100},
            {100, 300},
            {300, 800},
            {800, 2000},
            {2000, 4000},
            {4000, 7000},
            {7000, 10000},
            {10000, 12000},
        },
};

/* ═══════════════════════════════════════════════════════════════════
 * NAF header
 * ═══════════════════════════════════════════════════════════════════ */

#pragma pack(push, 1)
typedef struct
{
    char magic[4];
    float min_hz;
    float max_hz;
    uint32_t sample_rate;
    float window_sec;
    uint32_t n_slices;
    uint32_t n_channels;
    float _unused; /* was threshold_db – kept for file compat */
    float hop_sec;
    uint8_t reserved[4];
} NafHeader;
#pragma pack(pop)

/* Sanity check: header must be exactly 40 bytes */
typedef char _header_size_check[(sizeof(NafHeader) == 40) ? 1 : -1];

/* ═══════════════════════════════════════════════════════════════════
 * WAV chunk scanner
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct
{
    int sample_rate;
    int channels;
    int bps;
    long data_offset;
    uint32_t data_size;
} WavInfo;

static int
read_wav_info(FILE *f, WavInfo *out)
{
    char id[4];
    uint32_t size;

    if (fread(id, 1, 4, f) != 4 || memcmp(id, "RIFF", 4)) {
	fprintf(stderr, "Not a RIFF file\n");
	return 0;
    }
    if (fread(&size, 4, 1, f) != 1) { /* RIFF chunk size – ignored */
	fprintf(stderr, "Truncated RIFF header\n");
	return 0;
    }
    if (fread(id, 1, 4, f) != 4 || memcmp(id, "WAVE", 4)) {
	fprintf(stderr, "Not a WAVE file\n");
	return 0;
    }

    int found_fmt = 0, found_data = 0;
    memset(out, 0, sizeof(*out));

    while (!feof(f)) {
	if (fread(id, 1, 4, f) != 4)
	    break;
	if (fread(&size, 4, 1, f) != 1)
	    break;

	if (memcmp(id, "fmt ", 4) == 0) {
	    uint16_t fmt, ch, ba, bps;
	    uint32_t sr, br;
	    /* Check each fmt field read individually */
	    if (fread(&fmt, 2, 1, f) != 1 || fread(&ch, 2, 1, f) != 1 ||
	        fread(&sr, 4, 1, f) != 1 || fread(&br, 4, 1, f) != 1 ||
	        fread(&ba, 2, 1, f) != 1 || fread(&bps, 2, 1, f) != 1) {
		fprintf(stderr, "Truncated fmt chunk\n");
		return 0;
	    }
	    if (fmt != 1) {
		fprintf(stderr, "Only PCM WAV supported (format=%u)\n", fmt);
		return 0;
	    }
	    out->sample_rate = (int)sr;
	    out->channels = (int)ch;
	    out->bps = (int)bps;
	    /* Skip any extra fmt bytes (extended PCM headers) */
	    if (size > 16)
		fseek(f, (long)(size - 16), SEEK_CUR);
	    found_fmt = 1;

	} else if (memcmp(id, "data", 4) == 0) {
	    out->data_size = size;
	    out->data_offset = ftell(f);
	    found_data = 1;
	    break; /* file pointer is now at the start of audio data */

	} else {
	    /* Skip unknown chunks: LIST, id3, bext, smpl, JUNK, PAD … */
	    fseek(f, (long)size, SEEK_CUR);
	}
    }

    if (!found_fmt) {
	fprintf(stderr, "No fmt chunk in WAV\n");
	return 0;
    }
    if (!found_data) {
	fprintf(stderr, "No data chunk in WAV\n");
	return 0;
    }
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════
 * FFT  (Cooley-Tukey, in-place, power-of-2)
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct
{
    double r, i;
} Complex;

static void
fft(Complex *x, int n)
{
    /* Bit-reversal permutation */
    for (int i = 1, j = 0; i < n; i++) {
	int bit = n >> 1;
	for (; j & bit; bit >>= 1)
	    j ^= bit;
	j ^= bit;
	if (i < j) {
	    Complex tmp = x[i];
	    x[i] = x[j];
	    x[j] = tmp;
	}
    }
    /* Butterfly passes */
    for (int len = 2; len <= n; len <<= 1) {
	double ang = -2.0 * PI / len;
	Complex wlen = {cos(ang), sin(ang)};
	for (int i = 0; i < n; i += len) {
	    Complex w = {1.0, 0.0};
	    for (int j = 0; j < len / 2; j++) {
		Complex u = x[i + j];
		Complex v = {
		    x[i + j + len / 2].r * w.r - x[i + j + len / 2].i * w.i,
		    x[i + j + len / 2].r * w.i + x[i + j + len / 2].i * w.r};
		x[i + j] = (Complex){u.r + v.r, u.i + v.i};
		x[i + j + len / 2] = (Complex){u.r - v.r, u.i - v.i};
		/* Twiddle: compute both new values from OLD w before
		 * writing either, to avoid corrupting the next iteration */
		double wr = w.r * wlen.r - w.i * wlen.i;
		double wi = w.r * wlen.i + w.i * wlen.r;
		w.r = wr;
		w.i = wi;
	    }
	}
    }
}

static int
next_pow2(int n)
{
    int p = 1;
    while (p < n)
	p <<= 1;
    return p;
}

/* ═══════════════════════════════════════════════════════════════════
 * Frequency/loudness pair (used in both encode and decode)
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct
{
    float hz;
    float amp;
} Peak;

/* ═══════════════════════════════════════════════════════════════════
 * analyze_slice  –  FFT a mono frame, return true spectral peaks
 *
 * Returns the number of peaks written into `out` (at most
 * params->n_bands * params->peaks_per_band).
 *
 * Strategy:
 *  1. Zero-pad to params->fft_size for fine frequency resolution.
 *  2. Find local maxima (bins higher than both neighbours) per band.
 *  3. Keep the top peaks_per_band local maxima per band.
 *
 * This avoids the "cluster" problem where adjacent bins from the
 * same note all get stored, causing beating artefacts on decode.
 * ═══════════════════════════════════════════════════════════════════ */

static int
analyze_slice(const double *frame, int n, int sr, const EncodeParams *p,
              Peak *out)
{
    int fft_n = p->fft_size;
    /* Safety: fft_n must be at least as large as the actual input */
    if (fft_n < next_pow2(n))
	fft_n = next_pow2(n);

    Complex *x = calloc((size_t)fft_n, sizeof(Complex));
    if (!x)
	return 0;

    /* Hann window applied to the real samples; zero-padding fills the rest */
    for (int i = 0; i < n; i++) {
	double w = 0.5 * (1.0 - cos(2.0 * PI * i / (n - 1)));
	x[i].r = frame[i] * w;
    }

    fft(x, fft_n);

    int bins = fft_n / 2 + 1;
    double *mags = malloc((size_t)bins * sizeof(double));
    double *freqs = malloc((size_t)bins * sizeof(double));
    if (!mags || !freqs) {
	free(x);
	free(mags);
	free(freqs);
	return 0;
    }

    for (int i = 0; i < bins; i++) {
	freqs[i] = (double)i * sr / fft_n;
	mags[i] = sqrt(x[i].r * x[i].r + x[i].i * x[i].i) / fft_n;
    }
    free(x);

    /* --- Per-band local-maxima peak picking --- */

    /* Candidate buffer: worst case = every bin is a local max */
    typedef struct
    {
	double freq;
	double mag;
    } FM;
    FM *cands = malloc((size_t)bins * sizeof(FM));
    if (!cands) {
	free(mags);
	free(freqs);
	return 0;
    }

    int count = 0;
    int max_peaks = p->n_bands * p->peaks_per_band;
    if (max_peaks > MAX_PEAKS_PER_SLICE)
	max_peaks = MAX_PEAKS_PER_SLICE;

    for (int b = 0; b < p->n_bands && count < max_peaks; b++) {
	double lo = p->bands[b][0];
	double hi = p->bands[b][1];

	/* Collect true local maxima in this band */
	int nc = 0;
	for (int i = 1; i < bins - 1; i++) {
	    if (freqs[i] < lo || freqs[i] > hi)
		continue;
	    if (mags[i] > mags[i - 1] && mags[i] > mags[i + 1] && mags[i] > 0.0)
		cands[nc++] = (FM){freqs[i], mags[i]};
	}
	if (nc == 0)
	    continue;

	/* Partial selection sort: bring top peaks_per_band to front */
	int take = nc < p->peaks_per_band ? nc : p->peaks_per_band;
	for (int i = 0; i < take && count < max_peaks; i++) {
	    int best = i;
	    for (int j = i + 1; j < nc; j++)
		if (cands[j].mag > cands[best].mag)
		    best = j;
	    FM tmp = cands[i];
	    cands[i] = cands[best];
	    cands[best] = tmp;
	    out[count].hz = (float)cands[i].freq;
	    out[count].amp = (float)cands[i].mag;
	    count++;
	}
    }

    free(cands);
    free(mags);
    free(freqs);
    return count;
}

/* ═══════════════════════════════════════════════════════════════════
 * ENCODE
 * ═══════════════════════════════════════════════════════════════════ */

static int
encode(const char *wav_path, const char *naf_path, const EncodeParams *p)
{
    FILE *wf = fopen(wav_path, "rb");
    if (!wf) {
	fprintf(stderr, "Cannot open %s\n", wav_path);
	return 1;
    }

    WavInfo wi;
    if (!read_wav_info(wf, &wi)) {
	fclose(wf);
	return 1;
    }

    int sr = wi.sample_rate;
    int channels = wi.channels;
    int bps = wi.bps;
    int bytes_per_sample = bps / 8;
    /* Use long to avoid int overflow for large files */
    long total_frames =
        (long)(wi.data_size / (uint32_t)(bytes_per_sample * channels));

    printf("Encoding %s  [%s]\n", wav_path, p->name);
    printf("  Sample rate : %d Hz\n", sr);
    printf("  Channels    : %d\n", channels);
    printf("  Bit depth   : %d\n", bps);
    printf("  Duration    : %.2f s\n", (double)total_frames / sr);

    fseek(wf, wi.data_offset, SEEK_SET);

    int window_samples = (int)(p->window_sec * sr);
    int hop_samples = (int)(p->hop_sec * sr);

    if (hop_samples < 1)
	hop_samples = 1;

    /* Guard: need at least one full window */
    if (total_frames < window_samples) {
	fprintf(stderr, "Audio too short (need at least %.0f ms)\n",
	        p->window_sec * 1000.0f);
	fclose(wf);
	return 1;
    }

    long n_slices = (total_frames - window_samples) / hop_samples + 1;

    printf("  Window      : %.0f ms\n", p->window_sec * 1000.0f);
    printf("  Hop         : %.0f ms\n", p->hop_sec * 1000.0f);
    printf("  FFT size    : %d bins\n", p->fft_size);
    printf("  Slices      : %ld\n", n_slices);

    /* Read all audio into a mono double array */
    double max_val = (double)(1L << (bps - 1));
    double *mono = malloc((size_t)total_frames * sizeof(double));
    if (!mono) {
	fprintf(stderr, "Out of memory\n");
	fclose(wf);
	return 1;
    }

    for (long i = 0; i < total_frames; i++) {
	double sum = 0.0;
	for (int c = 0; c < channels; c++) {
	    int32_t s = 0;
	    if (fread(&s, (size_t)bytes_per_sample, 1, wf) != 1) {
		fprintf(stderr, "Unexpected end of WAV data\n");
		free(mono);
		fclose(wf);
		return 1;
	    }
	    /* Sign-extend */
	    if (bps == 8)
		s = (int8_t)s;
	    else if (bps == 16)
		s = (int16_t)s;
	    /* 32-bit already sign-extended */
	    sum += (double)s / max_val;
	}
	mono[i] = sum / channels;
    }
    fclose(wf);

    /* ── Pass 1: FFT every slice, collect raw Hz/amp ── */
    printf("Analyzing frequencies (pass 1/2)...\n");

    int max_per_slice = p->n_bands * p->peaks_per_band;
    if (max_per_slice > MAX_PEAKS_PER_SLICE)
	max_per_slice = MAX_PEAKS_PER_SLICE;

    Peak **slices = malloc((size_t)n_slices * sizeof(Peak *));
    if (!slices) {
	free(mono);
	return 1;
    }
    int *counts = malloc((size_t)n_slices * sizeof(int));
    if (!counts) {
	free(mono);
	free(slices);
	return 1;
    }

    float global_min_hz = 1e10f;
    float global_max_hz = 0.0f;
    float global_max_amp = 0.0f;

    for (long i = 0; i < n_slices; i++) {
	slices[i] = malloc((size_t)max_per_slice * sizeof(Peak));
	if (!slices[i]) {
	    fprintf(stderr, "Out of memory at slice %ld\n", i);
	    for (long k = 0; k < i; k++)
		free(slices[k]);
	    free(slices);
	    free(counts);
	    free(mono);
	    return 1;
	}
	counts[i] = analyze_slice(mono + i * hop_samples, window_samples, sr, p,
	                          slices[i]);
	for (int j = 0; j < counts[i]; j++) {
	    float hz = slices[i][j].hz;
	    float amp = slices[i][j].amp;
	    if (hz < global_min_hz)
		global_min_hz = hz;
	    if (hz > global_max_hz)
		global_max_hz = hz;
	    if (amp > global_max_amp)
		global_max_amp = amp;
	}
	/* Simple progress indicator every 5% */
	if (n_slices >= 20 && i % (n_slices / 20) == 0)
	    printf("\r  %3ld%%", i * 100 / n_slices), fflush(stdout);
    }
    printf("\r  100%%\n");
    free(mono);

    /* Clamp global range */
    if (global_min_hz > 9e9f)
	global_min_hz = 20.0f;
    if (global_max_hz == 0.0f)
	global_max_hz = p->max_freq_hz;
    if (global_max_amp == 0.0f)
	global_max_amp = 1.0f;
    float hz_range = global_max_hz - global_min_hz;
    if (hz_range < 1.0f)
	hz_range = 1.0f;

    printf("  Freq range  : %.1f Hz – %.1f Hz\n", global_min_hz, global_max_hz);

    /* ── Pass 2: normalise and write ── */
    printf("Writing NAF (pass 2/2)...\n");

    FILE *nf = fopen(naf_path, "wb");
    if (!nf) {
	fprintf(stderr, "Cannot write %s\n", naf_path);
	for (long i = 0; i < n_slices; i++)
	    free(slices[i]);
	free(slices);
	free(counts);
	return 1;
    }

    NafHeader nh;
    memcpy(nh.magic, MAGIC, 4);
    nh.min_hz = global_min_hz;
    nh.max_hz = global_max_hz;
    nh.sample_rate = (uint32_t)sr;
    nh.window_sec = p->window_sec;
    nh.n_slices = (uint32_t)n_slices;
    nh.n_channels = (uint32_t)channels;
    nh._unused = 0.0f;
    nh.hop_sec = p->hop_sec;
    memset(nh.reserved, 0, 4);
    fwrite(&nh, sizeof(NafHeader), 1, nf);

    for (long i = 0; i < n_slices; i++) {
	uint16_t cnt = (uint16_t)counts[i];
	fwrite(&cnt, sizeof(uint16_t), 1, nf);
	for (int j = 0; j < counts[i]; j++) {
	    float fn = (slices[i][j].hz - global_min_hz) / hz_range;
	    float ln = slices[i][j].amp / global_max_amp;
	    /* Clamp to [0,1] */
	    fn = fn < 0.0f ? 0.0f : (fn > 1.0f ? 1.0f : fn);
	    ln = ln < 0.0f ? 0.0f : (ln > 1.0f ? 1.0f : ln);
	    fwrite(&fn, sizeof(float), 1, nf);
	    fwrite(&ln, sizeof(float), 1, nf);
	}
	free(slices[i]);
    }
    free(slices);
    free(counts);
    fclose(nf);

    printf("Done! Written to %s\n", naf_path);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * DECODE
 * ═══════════════════════════════════════════════════════════════════ */

static int
decode(const char *naf_path, const char *wav_path)
{
    FILE *nf = fopen(naf_path, "rb");
    if (!nf) {
	fprintf(stderr, "Cannot open %s\n", naf_path);
	return 1;
    }

    NafHeader nh;
    if (fread(&nh, sizeof(NafHeader), 1, nf) != 1) {
	fprintf(stderr, "Failed to read NAF header (file too short?)\n");
	fclose(nf);
	return 1;
    }
    if (memcmp(nh.magic, MAGIC, 4)) {
	fprintf(stderr, "Not a valid NAF file (bad magic)\n");
	fclose(nf);
	return 1;
    }

    int sr = (int)nh.sample_rate;
    int window_n = (int)(nh.window_sec * sr);
    /* Read hop from header; fall back to 50% of window for old files
     * that were encoded before hop_sec was added to the header.      */
    int hop_n = (nh.hop_sec > 0.0f) ? (int)(nh.hop_sec * sr) : window_n / 2;
    float hz_range = nh.max_hz - nh.min_hz;

    /* Total output length in mono samples */
    long total_samples = (long)(nh.n_slices - 1) * hop_n + window_n;

    printf("Decoding %s\n", naf_path);
    printf("  Sample rate : %u Hz\n", nh.sample_rate);
    printf("  Channels    : %u\n", nh.n_channels);
    printf("  Slices      : %u\n", nh.n_slices);
    printf("  Window      : %.0f ms\n", nh.window_sec * 1000.0f);
    printf("  Hop         : %.0f ms\n", nh.hop_sec * 1000.0f);
    printf("  Freq range  : %.1f – %.1f Hz\n", nh.min_hz, nh.max_hz);
    printf("  Duration    : %.2f s\n", (double)total_samples / sr);

    double *output = calloc((size_t)total_samples, sizeof(double));
    double *frame = malloc((size_t)window_n * sizeof(double));
    if (!output || !frame) {
	fprintf(stderr, "Out of memory\n");
	free(output);
	free(frame);
	fclose(nf);
	return 1;
    }

    /* Pre-compute Hann window coefficients scaled by the OLA correction
     * factor.  With a hop of H and window of W, the Hann OLA sum at any
     * steady-state sample equals W/(2*H) (provable analytically).
     * Dividing each window coefficient by that factor makes the OLA sum
     * unity everywhere, so slices with many overlapping neighbours don't
     * get boosted relative to sparse regions.                           */
    double overlap_factor = (double)window_n / (2.0 * hop_n);
    double *hann = malloc((size_t)window_n * sizeof(double));
    if (!hann) {
	free(output);
	free(frame);
	fclose(nf);
	return 1;
    }
    for (int k = 0; k < window_n; k++)
	hann[k] = (0.5 * (1.0 - cos(2.0 * PI * k / window_n))) / overlap_factor;

    printf("Reconstructing audio...\n");

    for (uint32_t i = 0; i < nh.n_slices; i++) {
	uint16_t count;
	if (fread(&count, sizeof(uint16_t), 1, nf) != 1)
	    break;

	long start = (long)i * hop_n;

	/* Synthesise this slice as a sum of sine waves.
	 * Each sine is weighted by its stored loudness (ln) directly.
	 * The stored ln values encode relative energy correctly; the
	 * Hann OLA window + final peak normalisation handles the
	 * absolute level.  No per-slice scaling is applied here so
	 * that the loudness relationships between slices are preserved. */
	memset(frame, 0, (size_t)window_n * sizeof(double));

	for (int j = 0; j < (int)count; j++) {
	    float fn, ln;
	    if (fread(&fn, sizeof(float), 1, nf) != 1)
		goto decode_done;
	    if (fread(&ln, sizeof(float), 1, nf) != 1)
		goto decode_done;

	    double hz = nh.min_hz + fn * hz_range;

	    for (int k = 0; k < window_n; k++) {
		/* Global sample index keeps sine phase continuous
		 * across slice boundaries — no phase resets.          */
		double t = (double)(start + k) / sr;
		frame[k] += (double)ln * sin(2.0 * PI * hz * t);
	    }
	}

	/* Apply Hann window and overlap-add into output */
	for (int k = 0; k < window_n; k++) {
	    long idx = start + k;
	    if (idx >= total_samples)
		break;
	    output[idx] += frame[k] * hann[k];
	}

	/* Progress every 5% */
	if (nh.n_slices >= 20 && i % (nh.n_slices / 20) == 0)
	    printf("\r  %3u%%", i * 100 / nh.n_slices), fflush(stdout);
    }
    printf("\r  100%%\n");

decode_done:
    free(frame);
    free(hann);
    fclose(nf);

    /* Normalise to full scale */
    double peak = 0.0;
    for (long i = 0; i < total_samples; i++)
	if (fabs(output[i]) > peak)
	    peak = fabs(output[i]);
    if (peak > 0.0)
	for (long i = 0; i < total_samples; i++)
	    output[i] /= peak;

    /* Write WAV – field-by-field to avoid any struct padding surprises */
    printf("Writing %s...\n", wav_path);
    FILE *wf = fopen(wav_path, "wb");
    if (!wf) {
	fprintf(stderr, "Cannot write %s\n", wav_path);
	free(output);
	return 1;
    }

    uint32_t data_size = (uint32_t)(total_samples * (long)nh.n_channels * 2);
    uint32_t byte_rate =
        (uint32_t)((uint64_t)nh.sample_rate * nh.n_channels * 2);
    uint32_t u32;
    uint16_t u16;

    fwrite("RIFF", 1, 4, wf);
    u32 = 36 + data_size;
    fwrite(&u32, 4, 1, wf);
    fwrite("WAVE", 1, 4, wf);
    fwrite("fmt ", 1, 4, wf);
    u32 = 16;
    fwrite(&u32, 4, 1, wf);
    u16 = 1;
    fwrite(&u16, 2, 1, wf); /* PCM */
    u16 = (uint16_t)nh.n_channels;
    fwrite(&u16, 2, 1, wf);
    u32 = nh.sample_rate;
    fwrite(&u32, 4, 1, wf);
    u32 = byte_rate;
    fwrite(&u32, 4, 1, wf);
    u16 = (uint16_t)(nh.n_channels * 2);
    fwrite(&u16, 2, 1, wf);
    u16 = 16;
    fwrite(&u16, 2, 1, wf);
    fwrite("data", 1, 4, wf);
    fwrite(&data_size, 4, 1, wf);

    /* Expand mono reconstruction to n_channels (same sample both channels) */
    for (long i = 0; i < total_samples; i++) {
	int16_t s = (int16_t)(output[i] * 32767.0);
	for (uint32_t c = 0; c < nh.n_channels; c++)
	    fwrite(&s, sizeof(int16_t), 1, wf);
    }

    fclose(wf);
    free(output);
    printf("Done!\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * INFO
 * ═══════════════════════════════════════════════════════════════════ */

static int
info(const char *naf_path)
{
    FILE *nf = fopen(naf_path, "rb");
    if (!nf) {
	fprintf(stderr, "Cannot open %s\n", naf_path);
	return 1;
    }

    NafHeader nh;
    if (fread(&nh, sizeof(NafHeader), 1, nf) != 1) {
	fprintf(stderr, "Failed to read header\n");
	fclose(nf);
	return 1;
    }
    fclose(nf);

    if (memcmp(nh.magic, MAGIC, 4)) {
	fprintf(stderr, "Not a valid NAF file\n");
	return 1;
    }

    float hop = (nh.hop_sec > 0.0f) ? nh.hop_sec : nh.window_sec;
    float duration = (float)(nh.n_slices - 1) * hop + nh.window_sec;

    printf("NAF File      : %s\n", naf_path);
    printf("  Magic       : %.4s\n", nh.magic);
    printf("  Sample rate : %u Hz\n", nh.sample_rate);
    printf("  Channels    : %u\n", nh.n_channels);
    printf("  Window      : %.0f ms\n", nh.window_sec * 1000.0f);
    printf("  Hop         : %.0f ms\n", hop * 1000.0f);
    printf("  Total slices: %u\n", nh.n_slices);
    printf("  Duration    : %.2f s\n", duration);
    printf("  Freq range  : %.2f – %.2f Hz\n", nh.min_hz, nh.max_hz);

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════ */

static void
usage(void)
{
    printf("NAF Converter – Normalized Audio Format\n\n");
    printf("Usage:\n");
    printf("  naf encode [-S|-Q|-Q2|-Q9] input.wav  output.naf\n");
    printf("  naf decode                 input.naf  output.wav\n");
    printf("  naf info                   input.naf\n\n");
    printf("Encode modes:\n");
    printf("  -S   Size mode    – small files, fast encode\n");
    printf("  -Q   Quality mode – good quality  (default)\n");
    printf("  -Q2  Ultra mode   – best quality, larger files\n");
    printf("  -Q9  Maximum mode – absolute best, very large, slow\n");
}

int
main(int argc, char *argv[])
{
    if (argc < 3) {
	usage();
	return 1;
    }

    const char *cmd = argv[1];

    if (strcmp(cmd, "encode") == 0) {
	/* encode [-S|-Q|-Q2] input.wav output.naf */
	const EncodeParams *p = &PARAMS_Q; /* default */
	const char *wav = NULL;
	const char *naf = NULL;

	for (int i = 2; i < argc; i++) {
	    if (strcmp(argv[i], "-S") == 0)
		p = &PARAMS_S;
	    else if (strcmp(argv[i], "-Q") == 0)
		p = &PARAMS_Q;
	    else if (strcmp(argv[i], "-Q2") == 0)
		p = &PARAMS_Q2;
	    else if (strcmp(argv[i], "-Q9") == 0)
		p = &PARAMS_Q9;
	    else if (argv[i][0] == '-') {
		fprintf(stderr, "Unknown flag: %s\n", argv[i]);
		usage();
		return 1;
	    } else if (!wav)
		wav = argv[i];
	    else if (!naf)
		naf = argv[i];
	    else {
		fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
		usage();
		return 1;
	    }
	}
	if (!wav || !naf) {
	    usage();
	    return 1;
	}
	return encode(wav, naf, p);

    } else if (strcmp(cmd, "decode") == 0 && argc == 4) {
	return decode(argv[2], argv[3]);

    } else if (strcmp(cmd, "info") == 0 && argc == 3) {
	return info(argv[2]);

    } else {
	usage();
	return 1;
    }
}
