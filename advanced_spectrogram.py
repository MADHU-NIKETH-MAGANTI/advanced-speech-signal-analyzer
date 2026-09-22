import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Load your original speech
audio, sample_rate = sf.read("voice_original.wav")

# Convert stereo to mono
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Remove DC component
audio = audio - np.mean(audio)

# -----------------------------------------
# Spectrogram
# -----------------------------------------

frequencies, times, Sxx = spectrogram(
    audio,
    fs=sample_rate,
    window="hann",
    nperseg=512,
    noverlap=384
)

# Convert magnitude to dB
Sxx_dB = 10 * np.log10(
    Sxx + 1e-12
)

# -----------------------------------------
# Plot
# -----------------------------------------

plt.figure(figsize=(12, 6))

plt.pcolormesh(
    times,
    frequencies,
    Sxx_dB,
    shading="gouraud"
)

plt.colorbar(
    label="Power (dB)"
)

plt.title(
    "Time-Frequency Analysis of Original Speech"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")

# Focus on useful speech frequencies
plt.ylim(0, 4000)

plt.tight_layout()
plt.show()

# -----------------------------------------
# Display information
# -----------------------------------------

print("\n===== TIME-FREQUENCY ANALYSIS =====")

print(f"Sampling Frequency : {sample_rate} Hz")
print(f"Frequency Resolution: "
      f"{frequencies[1] - frequencies[0]:.2f} Hz")

print(f"Time Resolution: "
      f"{times[1] - times[0]:.4f} sec")