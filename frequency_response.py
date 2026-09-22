import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# --------------------------------
# Load your recorded speech
# --------------------------------
audio, sample_rate = sf.read("voice_original.wav")

# Make sure audio is 1-D
if audio.ndim > 1:
    audio = audio[:, 0]

N = len(audio)

# --------------------------------
# DTFT using FFT
# --------------------------------
X = np.fft.fft(audio)

# Frequency axis
frequency = np.fft.fftfreq(N, 1 / sample_rate)

# Keep only positive frequencies
positive = frequency >= 0

frequency = frequency[positive]
X = X[positive]

# --------------------------------
# Magnitude response
# --------------------------------
magnitude = np.abs(X)

# Normalize magnitude
magnitude = magnitude / np.max(magnitude)

# --------------------------------
# Phase response
# --------------------------------
phase = np.unwrap(np.angle(X))

# --------------------------------
# Magnitude Plot
# --------------------------------
plt.figure(figsize=(12, 4))

plt.plot(frequency, magnitude)

plt.title("Speech Signal - Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X(e^jω)|")
plt.grid()

plt.tight_layout()
plt.show()

# --------------------------------
# Phase Plot
# --------------------------------
plt.figure(figsize=(12, 4))

plt.plot(frequency, phase)

plt.title("Speech Signal - Phase Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.grid()

plt.tight_layout()
plt.show()

print("================================")
print("Speech Frequency Analysis")
print("================================")
print("Sampling Frequency:", sample_rate, "Hz")
print("Number of Samples:", N)
print("Duration:", N / sample_rate, "seconds")
print("Maximum Frequency:", sample_rate / 2, "Hz")
print("================================")