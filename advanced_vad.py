import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Load recorded speech
audio, sample_rate = sf.read("voice_original.wav")

# Convert stereo to mono if required
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# -------------------------------
# 1. Calculate short-time energy
# -------------------------------

frame_size = int(0.025 * sample_rate)      # 25 ms
hop_size = int(0.010 * sample_rate)        # 10 ms

energy = []
time = []

for start in range(0, len(audio) - frame_size, hop_size):

    frame = audio[start:start + frame_size]

    rms = np.sqrt(np.mean(frame ** 2))

    energy.append(rms)
    time.append(start / sample_rate)

energy = np.array(energy)
time = np.array(time)

# -------------------------------
# 2. Determine speech threshold
# -------------------------------

threshold = 0.15 * np.max(energy)

speech_frames = energy > threshold

# -------------------------------
# 3. Find speech region
# -------------------------------

speech_indices = np.where(speech_frames)[0]

if len(speech_indices) > 0:

    start_time = time[speech_indices[0]]

    end_time = time[speech_indices[-1]] + frame_size / sample_rate

    active_duration = end_time - start_time

else:

    start_time = 0
    end_time = 0
    active_duration = 0

# -------------------------------
# 4. Display results
# -------------------------------

total_duration = len(audio) / sample_rate

speech_percentage = (
    active_duration / total_duration
) * 100

print("\n===== ADVANCED SPEECH ANALYSIS =====")

print(f"Total Duration       : {total_duration:.2f} sec")
print(f"Speech Start         : {start_time:.2f} sec")
print(f"Speech End           : {end_time:.2f} sec")
print(f"Active Speech        : {active_duration:.2f} sec")
print(f"Speech Activity      : {speech_percentage:.2f}%")

# -------------------------------
# 5. Plot waveform + VAD
# -------------------------------

time_audio = np.arange(len(audio)) / sample_rate

plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)

plt.plot(time_audio, audio)

plt.axvline(start_time, linestyle="--")
plt.axvline(end_time, linestyle="--")

plt.title("Speech Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)

plt.plot(time, energy)

plt.axhline(threshold, linestyle="--")

plt.fill_between(
    time,
    0,
    energy,
    where=speech_frames,
    alpha=0.3
)

plt.title("Voice Activity Detection")
plt.xlabel("Time (seconds)")
plt.ylabel("RMS Energy")
plt.grid()

plt.tight_layout()

plt.show()