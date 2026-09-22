import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

# ==========================================
# SETTINGS
# ==========================================

duration = 5
sample_rate = 16000

print("Recording will start in 2 seconds...")
sd.sleep(2000)

print("🎤 Speak now...")

# Record voice
audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="float32"
)

sd.wait()

audio = audio.flatten()

print("✅ Recording completed!")

# ==========================================
# SAVE ORIGINAL
# ==========================================

sf.write("voice_original.wav", audio, sample_rate)

# ==========================================
# IMPULSE RESPONSE
# ==========================================

# Simple moving-average impulse response
h = np.ones(5) / 5

# ==========================================
# CONVOLUTION
# ==========================================

processed_audio = np.convolve(audio, h, mode="same")

# Prevent clipping
processed_audio = np.clip(processed_audio, -1, 1)

# Save processed audio
sf.write(
    "voice_convolved.wav",
    processed_audio,
    sample_rate
)

# ==========================================
# TIME AXIS
# ==========================================

time = np.arange(len(audio)) / sample_rate

# ==========================================
# PLOT ORIGINAL
# ==========================================

plt.figure(figsize=(12, 4))

plt.plot(time, audio)

plt.title("Original Speech Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

# ==========================================
# PLOT CONVOLVED
# ==========================================

plt.figure(figsize=(12, 4))

plt.plot(time, processed_audio)

plt.title("Speech Signal After LTI Convolution")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

# ==========================================
# PLAY ORIGINAL
# ==========================================

print("🔊 Playing original speech...")

sd.play(audio, sample_rate)
sd.wait()

# ==========================================
# PLAY PROCESSED
# ==========================================

print("🔊 Playing convolved speech...")

sd.play(processed_audio, sample_rate)
sd.wait()

print("✅ Audio convolution completed!")