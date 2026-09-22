import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Load your speech
audio, sample_rate = sf.read("voice_original.wav")

# Convert stereo to mono
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Normalize
audio = audio / np.max(np.abs(audio))

# ------------------------------------------------
# Filter design
# ------------------------------------------------

# Low-pass filter
low_b, low_a = butter(
    4,
    1000,
    btype="low",
    fs=sample_rate
)

# High-pass filter
high_b, high_a = butter(
    4,
    300,
    btype="high",
    fs=sample_rate
)

# Band-pass filter
band_b, band_a = butter(
    4,
    [300, 3000],
    btype="band",
    fs=sample_rate
)

# ------------------------------------------------
# Apply filters
# ------------------------------------------------

low_pass = filtfilt(
    low_b,
    low_a,
    audio
)

high_pass = filtfilt(
    high_b,
    high_a,
    audio
)

band_pass = filtfilt(
    band_b,
    band_a,
    audio
)

# ------------------------------------------------
# Save filtered signals
# ------------------------------------------------

sf.write(
    "voice_lowpass.wav",
    low_pass,
    sample_rate
)

sf.write(
    "voice_highpass.wav",
    high_pass,
    sample_rate
)

sf.write(
    "voice_bandpass.wav",
    band_pass,
    sample_rate
)

# ------------------------------------------------
# Time axis
# ------------------------------------------------

time = np.arange(len(audio)) / sample_rate

# ------------------------------------------------
# Plot comparison
# ------------------------------------------------

plt.figure(figsize=(12, 9))

plt.subplot(4, 1, 1)

plt.plot(time, audio)

plt.title("Original Speech")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4, 1, 2)

plt.plot(time, low_pass)

plt.title("Low-Pass Filtered Speech")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4, 1, 3)

plt.plot(time, high_pass)

plt.title("High-Pass Filtered Speech")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4, 1, 4)

plt.plot(time, band_pass)

plt.title("Band-Pass Filtered Speech")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()

plt.show()

print("\n===== DIGITAL FILTER BANK =====")

print("Low-pass cutoff   : 1000 Hz")
print("High-pass cutoff  : 300 Hz")
print("Band-pass range   : 300 - 3000 Hz")

print("\nFiltered audio files created:")
print("voice_lowpass.wav")
print("voice_highpass.wav")
print("voice_bandpass.wav")