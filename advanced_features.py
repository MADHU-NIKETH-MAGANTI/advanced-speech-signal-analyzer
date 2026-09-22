import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Load recorded speech
audio, sample_rate = sf.read("voice_original.wav")

# Convert stereo to mono
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Frame settings
frame_size = int(0.025 * sample_rate)   # 25 ms
hop_size = int(0.010 * sample_rate)     # 10 ms

rms_values = []
zcr_values = []
time = []

# Process speech frame-by-frame
for start in range(0, len(audio) - frame_size, hop_size):

    frame = audio[start:start + frame_size]

    # -------------------------
    # RMS Energy
    # -------------------------
    rms = np.sqrt(np.mean(frame ** 2))

    # -------------------------
    # Zero Crossing Rate
    # -------------------------
    signs = np.sign(frame)

    zero_crossings = np.sum(
        np.abs(np.diff(signs))
    ) / 2

    zcr = zero_crossings / (len(frame) - 1)

    rms_values.append(rms)
    zcr_values.append(zcr)

    time.append(start / sample_rate)

rms_values = np.array(rms_values)
zcr_values = np.array(zcr_values)
time = np.array(time)

# Overall values
average_rms = np.mean(rms_values)
average_zcr = np.mean(zcr_values)

print("\n===== ADVANCED SPEECH FEATURES =====")

print(f"Sampling Frequency : {sample_rate} Hz")
print(f"Average RMS Energy : {average_rms:.4f}")
print(f"Average ZCR        : {average_zcr:.4f}")

# -------------------------
# Plot RMS Energy
# -------------------------

plt.figure(figsize=(12, 5))

plt.plot(time, rms_values)

plt.title("Speech RMS Energy")
plt.xlabel("Time (seconds)")
plt.ylabel("RMS Energy")
plt.grid()

plt.tight_layout()
plt.show()

# -------------------------
# Plot ZCR
# -------------------------

plt.figure(figsize=(12, 5))

plt.plot(time, zcr_values)

plt.title("Speech Zero-Crossing Rate")
plt.xlabel("Time (seconds)")
plt.ylabel("ZCR")
plt.grid()

plt.tight_layout()
plt.show()