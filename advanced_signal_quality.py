import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# -----------------------------------------
# Load original speech
# -----------------------------------------

audio, sample_rate = sf.read("voice_original.wav")

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Convert to float
audio = audio.astype(float)

# -----------------------------------------
# Signal measurements
# -----------------------------------------

peak_amplitude = np.max(np.abs(audio))

rms = np.sqrt(
    np.mean(audio ** 2)
)

signal_energy = np.sum(
    audio ** 2
)

signal_power = np.mean(
    audio ** 2
)

# Avoid division by zero
if rms > 0:
    crest_factor = peak_amplitude / rms
else:
    crest_factor = 0

# Dynamic range in dB
if peak_amplitude > 0 and rms > 0:
    dynamic_range = (
        20 * np.log10(
            peak_amplitude / rms
        )
    )
else:
    dynamic_range = 0

# -----------------------------------------
# Results
# -----------------------------------------

duration = len(audio) / sample_rate

print("\n======================================")
print("       SIGNAL QUALITY ANALYSIS")
print("======================================")

print(f"Duration             : {duration:.2f} sec")
print(f"Sampling Frequency   : {sample_rate} Hz")
print(f"Peak Amplitude       : {peak_amplitude:.4f}")
print(f"RMS Level            : {rms:.4f}")
print(f"Signal Energy        : {signal_energy:.4f}")
print(f"Signal Power         : {signal_power:.6f}")
print(f"Crest Factor         : {crest_factor:.4f}")
print(f"Dynamic Range        : {dynamic_range:.2f} dB")

# -----------------------------------------
# Frame-wise RMS
# -----------------------------------------

frame_size = int(0.025 * sample_rate)
hop_size = int(0.010 * sample_rate)

rms_frames = []
time = []

for start in range(
    0,
    len(audio) - frame_size,
    hop_size
):

    frame = audio[
        start:start + frame_size
    ]

    frame_rms = np.sqrt(
        np.mean(frame ** 2)
    )

    rms_frames.append(frame_rms)
    time.append(start / sample_rate)

# -----------------------------------------
# Plot RMS
# -----------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    rms_frames
)

plt.title(
    "Frame-wise RMS Energy of Original Speech"
)

plt.xlabel("Time (seconds)")
plt.ylabel("RMS")

plt.grid()

plt.tight_layout()
plt.show()