import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# -----------------------------------------
# Load original speech
# -----------------------------------------

audio, sample_rate = sf.read("voice_original.wav")

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# -----------------------------------------
# Frame parameters
# -----------------------------------------

frame_size = int(0.025 * sample_rate)   # 25 ms
hop_size = int(0.010 * sample_rate)     # 10 ms

# -----------------------------------------
# Calculate RMS energy
# -----------------------------------------

energy = []
time = []

for start in range(
    0,
    len(audio) - frame_size,
    hop_size
):

    frame = audio[start:start + frame_size]

    rms = np.sqrt(np.mean(frame ** 2))

    energy.append(rms)
    time.append(start / sample_rate)

energy = np.array(energy)
time = np.array(time)

# -----------------------------------------
# Speech threshold
# -----------------------------------------

threshold = 0.15 * np.max(energy)

speech = energy > threshold

# -----------------------------------------
# Detect transitions
# -----------------------------------------

changes = np.diff(
    speech.astype(int)
)

speech_starts = np.where(changes == 1)[0]
speech_ends = np.where(changes == -1)[0]

# Handle recording beginning with speech
if speech[0]:
    speech_starts = np.insert(
        speech_starts, 0, 0
    )

# Handle recording ending with speech
if speech[-1]:
    speech_ends = np.append(
        speech_ends, len(speech) - 1
    )

# -----------------------------------------
# Speech segments
# -----------------------------------------

speech_segments = []

for start, end in zip(
    speech_starts,
    speech_ends
):

    start_time = time[start]

    end_time = (
        time[end] +
        frame_size / sample_rate
    )

    speech_segments.append(
        (start_time, end_time)
    )

# -----------------------------------------
# Pause calculation
# -----------------------------------------

pauses = []

for i in range(
    len(speech_segments) - 1
):

    pause_start = speech_segments[i][1]

    pause_end = speech_segments[i + 1][0]

    pause_duration = pause_end - pause_start

    if pause_duration > 0:
        pauses.append(pause_duration)

# -----------------------------------------
# Results
# -----------------------------------------

total_duration = len(audio) / sample_rate

active_speech = np.sum(speech) * (
    hop_size / sample_rate
)

total_pause = total_duration - active_speech

if len(pauses) > 0:
    average_pause = np.mean(pauses)
    longest_pause = np.max(pauses)
else:
    average_pause = 0
    longest_pause = 0

speech_activity = (
    active_speech / total_duration
) * 100

# -----------------------------------------
# Display
# -----------------------------------------

print("\n======================================")
print("       SPEECH PAUSE ANALYSIS")
print("======================================")

print(f"Total Duration       : {total_duration:.2f} sec")
print(f"Speech Segments      : {len(speech_segments)}")
print(f"Number of Pauses     : {len(pauses)}")
print(f"Active Speech        : {active_speech:.2f} sec")
print(f"Total Pause Time     : {total_pause:.2f} sec")
print(f"Average Pause        : {average_pause:.2f} sec")
print(f"Longest Pause        : {longest_pause:.2f} sec")
print(f"Speech Activity      : {speech_activity:.2f}%")

# -----------------------------------------
# Plot
# -----------------------------------------

audio_time = np.arange(len(audio)) / sample_rate

plt.figure(figsize=(12, 6))

plt.plot(
    audio_time,
    audio,
    label="Original Speech"
)

# Mark speech segments
for start, end in speech_segments:

    plt.axvspan(
        start,
        end,
        alpha=0.25
    )

plt.title(
    "Speech Segmentation and Pause Detection"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid()
plt.tight_layout()

plt.show()