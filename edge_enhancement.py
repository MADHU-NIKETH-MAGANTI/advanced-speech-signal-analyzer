import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

# -----------------------------
# Recording parameters
# -----------------------------
duration = 5
sample_rate = 16000

print("Recording will start in 2 seconds...")
sd.sleep(2000)

print("🎤 Speak now...")

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="float32"
)

sd.wait()
audio = audio.flatten()

# Save original
sf.write("voice_original.wav", audio, sample_rate)

# -----------------------------
# Edge-enhancement impulse response
# -----------------------------
h = np.array([-1, 1])

# -----------------------------
# Discrete convolution
# -----------------------------
edge_audio = np.convolve(audio, h, mode="same")

# Prevent clipping
edge_audio = np.clip(edge_audio, -1, 1)

# Save processed audio
sf.write("voice_edge_enhanced.wav", edge_audio, sample_rate)

# Time axis
time = np.arange(len(audio)) / sample_rate

# -----------------------------
# Plot original speech
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(time, audio)
plt.title("Original Speech Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()
plt.tight_layout()
plt.show()

# -----------------------------
# Plot edge-enhanced speech
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(time, edge_audio)
plt.title("Speech Signal After Edge Enhancement")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()
plt.tight_layout()
plt.show()

# -----------------------------
# Play audio
# -----------------------------
print("🔊 Playing original speech...")
sd.play(audio, sample_rate)
sd.wait()

print("🔊 Playing edge-enhanced speech...")
sd.play(edge_audio, sample_rate)
sd.wait()

print("✅ Edge enhancement completed!")