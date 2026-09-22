import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# SETTINGS
# ==============================
duration = 5
sample_rate = 16000
gain = 2.0

# ==============================
# RECORD AUDIO
# ==============================
print("Recording will start in 2 seconds...")
sd.sleep(2000)

print("🎤 Recording started... Speak now!")

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="float32"
)

sd.wait()

print("✅ Recording completed!")

# Convert from 2D to 1D
audio = audio.flatten()

# ==============================
# SAVE ORIGINAL
# ==============================
sf.write("speech_original.wav", audio, sample_rate)

# ==============================
# GAIN ADJUSTMENT
# ==============================
gain_audio = audio * gain
gain_audio = np.clip(gain_audio, -1.0, 1.0)

sf.write("speech_gain.wav", gain_audio, sample_rate)

# ==============================
# TIME SHIFTING
# ==============================
shift_seconds = 1

shift_samples = int(shift_seconds * sample_rate)

# Delay the signal by 1 second
shifted_audio = np.concatenate(
    (
        np.zeros(shift_samples),
        audio
    )
)

# Keep same duration
shifted_audio = shifted_audio[:len(audio)]

sf.write("speech_shifted.wav", shifted_audio, sample_rate)

# ==============================
# TIME REVERSAL
# ==============================
reversed_audio = audio[::-1]

sf.write("speech_reversed.wav", reversed_audio, sample_rate)

# ==============================
# TIME AXIS
# ==============================
time = np.arange(len(audio)) / sample_rate

# ==============================
# ORIGINAL
# ==============================
plt.figure(figsize=(12, 4))
plt.plot(time, audio)

plt.title("Original Speech Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

# ==============================
# GAIN
# ==============================
plt.figure(figsize=(12, 4))
plt.plot(time, gain_audio)

plt.title("Gain Adjusted Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

# ==============================
# TIME SHIFT
# ==============================
plt.figure(figsize=(12, 4))
plt.plot(time, shifted_audio)

plt.title("Time Shifted Signal - Delay = 1 Second")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

# ==============================
# TIME REVERSAL
# ==============================
plt.figure(figsize=(12, 4))
plt.plot(time, reversed_audio)

plt.title("Time Reversed Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

# ==============================
# PLAY ORIGINAL
# ==============================
print("🔊 Playing original speech...")
sd.play(audio, sample_rate)
sd.wait()

# ==============================
# PLAY GAIN
# ==============================
print("🔊 Playing gain-adjusted speech...")
sd.play(gain_audio, sample_rate)
sd.wait()

# ==============================
# PLAY SHIFTED
# ==============================
print("🔊 Playing time-shifted speech...")
sd.play(shifted_audio, sample_rate)
sd.wait()

# ==============================
# PLAY REVERSED
# ==============================
print("🔊 Playing reversed speech...")
sd.play(reversed_audio, sample_rate)
sd.wait()

print("✅ Module 1 signal operations completed!")