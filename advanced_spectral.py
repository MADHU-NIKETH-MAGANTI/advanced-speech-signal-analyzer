import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Load your recorded speech
audio, sample_rate = sf.read("voice_original.wav")

# Convert stereo to mono
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Remove DC component
audio = audio - np.mean(audio)

# FFT
N = len(audio)

spectrum = np.fft.rfft(audio)
magnitude = np.abs(spectrum)

frequencies = np.fft.rfftfreq(
    N,
    1 / sample_rate
)

# Avoid DC component
magnitude[0] = 0

# ------------------------------------------------
# 1. Dominant Frequency
# ------------------------------------------------

dominant_index = np.argmax(magnitude)

dominant_frequency = frequencies[dominant_index]

# ------------------------------------------------
# 2. Spectral Centroid
# ------------------------------------------------

total_magnitude = np.sum(magnitude)

spectral_centroid = (
    np.sum(frequencies * magnitude)
    / total_magnitude
)

# ------------------------------------------------
# 3. Spectral Bandwidth
# ------------------------------------------------

spectral_bandwidth = np.sqrt(
    np.sum(
        ((frequencies - spectral_centroid) ** 2)
        * magnitude
    )
    / total_magnitude
)

# ------------------------------------------------
# 4. Spectral Roll-off (85%)
# ------------------------------------------------

cumulative_energy = np.cumsum(magnitude)

rolloff_threshold = 0.85 * cumulative_energy[-1]

rolloff_index = np.where(
    cumulative_energy >= rolloff_threshold
)[0][0]

spectral_rolloff = frequencies[rolloff_index]

# ------------------------------------------------
# Display Results
# ------------------------------------------------

print("\n===== ADVANCED SPECTRAL ANALYSIS =====")

print(f"Sampling Frequency : {sample_rate} Hz")
print(f"Dominant Frequency : {dominant_frequency:.2f} Hz")
print(f"Spectral Centroid  : {spectral_centroid:.2f} Hz")
print(f"Spectral Bandwidth : {spectral_bandwidth:.2f} Hz")
print(f"Spectral Roll-off  : {spectral_rolloff:.2f} Hz")

# ------------------------------------------------
# Plot Spectrum
# ------------------------------------------------

normalized_magnitude = magnitude / np.max(magnitude)

plt.figure(figsize=(12, 6))

plt.plot(
    frequencies,
    normalized_magnitude
)

plt.axvline(
    dominant_frequency,
    linestyle="--",
    label=f"Dominant = {dominant_frequency:.1f} Hz"
)

plt.axvline(
    spectral_centroid,
    linestyle="--",
    label=f"Centroid = {spectral_centroid:.1f} Hz"
)

plt.axvline(
    spectral_rolloff,
    linestyle="--",
    label=f"Roll-off = {spectral_rolloff:.1f} Hz"
)

plt.title("Speech Frequency Spectrum")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Magnitude")

plt.xlim(0, sample_rate / 2)

plt.grid()

plt.legend()

plt.tight_layout()

plt.show()