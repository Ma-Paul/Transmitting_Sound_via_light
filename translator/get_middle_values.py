import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import time
import serial
from get_lowest_note import get_lowest_note_fn
from get_highest_note import get_highest_note_fn

adress = False
adress = str(input())
high = get_highest_note_fn(rf"{adress}")
low = get_lowest_note_fn(rf"{adress}")
print(f"High: {high}, Low: {low}")
arduino = serial.Serial(port="COM4", baudrate=115200, timeout=0.1)


def write_read(x):
    arduino.write(bytes(x, "utf-8"))
    time.sleep(0.05)
    data = arduino.readline()
    return data


def get_current_frequency(filename):
    sample_rate, data = wav.read(filename)

    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)

    WINDOW_SIZE = 2048  # How many samples per analysis chunk
    THRESHOLD = 0.01  # Noise threshold (tune if needed)

    current_frequency = 0.0  # <-- this is your variable

    def get_dominant_frequency(chunk):
        fft_result = np.abs(np.fft.rfft(chunk))
        frequencies = np.fft.rfftfreq(len(chunk), d=1 / sample_rate)
        threshold = np.max(fft_result) * THRESHOLD
        significant = frequencies[fft_result > threshold]
        if len(significant) == 0:
            return 0.0
        return significant.max()  # Change to .min() for lowest

    sd.play(data, sample_rate)
    start_time = time.time()

    for i in range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE):
        chunk = data[i : i + WINDOW_SIZE]
        current_frequency = get_dominant_frequency(chunk)
        print(f"Current frequency: {current_frequency:.2f} Hz")

        chunk_time = i / sample_rate
        elapsed = time.time() - start_time
        sleep_time = chunk_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    sd.wait()
    return current_frequency


atm = get_current_frequency(
    r"C:\Users\mantz\OneDrive\Desktop\SFZ Projekte\Transmitting_Sound_via_light\translator\Coldplay - Viva La Vida (Official Video).wav"
)
if atm == low:
    percent = 0
else:
    percent = atm / high


while True:
    num = percent
    value = write_read(num)
    print(value)  # printing the value
