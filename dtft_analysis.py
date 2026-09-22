import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Load recorded speech
audio, sample_rate = sf.read("voice_original.wav")

if audio.ndim > 1:
    audio = audio[:, 0]

# Take a smaller section for DTFT calculation
N = min(4000, len(audio))
x = audio[:N]

# Frequency range
w = np.linspace(-np.pi, np.pi, 1000)

# DTFT
X = np.zeros(len(w), dtype=complex)

n = np.arange(N)

for k in range(len(w)):
    X[k] = np.sum(x * np.exp(-1j * w[k] * n))

# Magnitude
magnitude = np.abs(X)

# Phase
phase = np.unwrap(np.angle(X))

# Convert angular frequency to Hz
frequency = w * sample_rate / (2 * np.pi)

# -----------------------------
# DTFT Magnitude
# -----------------------------
plt.figure(figsize=(12, 4))

plt.plot(frequency, magnitude)

plt.title("DTFT Magnitude Spectrum of Speech")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X(e^jω)|")
plt.grid()

plt.tight_layout()
plt.show()

# -----------------------------
# DTFT Phase
# -----------------------------
plt.figure(figsize=(12, 4))

plt.plot(frequency, phase)

plt.title("DTFT Phase Spectrum of Speech")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.grid()

plt.tight_layout()
plt.show()

print("================================")
print("DTFT Analysis Completed")
print("================================")
print("Sampling Frequency:", sample_rate, "Hz")
print("Samples used for DTFT:", N)
print("Frequency range:",
      round(frequency[0], 2), "to",
      round(frequency[-1], 2), "Hz")
print("================================")