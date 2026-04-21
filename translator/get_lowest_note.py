import numpy as np
import scipy.io.wavfile as wav
def get_lowest_note_fn(filename):
    sample_rate, data = wav.read(filename)

    # If stereo, take one channel
    if data.ndim > 1:
        data = data[:, 0]

    # Normalize to float
    data = data.astype(np.float32)

    # FFT
    fft_result = np.abs(np.fft.rfft(data))
    frequencies = np.fft.rfftfreq(len(data), d=1/sample_rate)

    # Threshold: ignore anything below 1% of the peak amplitude
    threshold = np.max(fft_result) * 0.01

    # Highest frequency above threshold
    significant = frequencies[fft_result > threshold]
    lowest = significant.min()
    return lowest