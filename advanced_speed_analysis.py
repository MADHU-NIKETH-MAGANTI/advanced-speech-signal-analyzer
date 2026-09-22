import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr

# -----------------------------------------
# Settings
# -----------------------------------------

SAMPLE_RATE = 16000
DURATION = 10

AUDIO_FILE = "advanced_speech.wav"

# -----------------------------------------
# Record speech
# -----------------------------------------

print("\n================================")
print("   ADVANCED SPEECH ANALYZER")
print("================================")

print(f"\nSpeak for {DURATION} seconds...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

audio = audio.flatten()

sf.write(
    AUDIO_FILE,
    audio,
    SAMPLE_RATE
)

print("Recording completed.")

# -----------------------------------------
# Calculate RMS-based speech activity
# -----------------------------------------

frame_size = int(0.025 * SAMPLE_RATE)
hop_size = int(0.010 * SAMPLE_RATE)

energy = []

for start in range(
    0,
    len(audio) - frame_size,
    hop_size
):

    frame = audio[
        start:start + frame_size
    ]

    rms = np.sqrt(
        np.mean(frame ** 2)
    )

    energy.append(rms)

energy = np.array(energy)

# -----------------------------------------
# Speech threshold
# -----------------------------------------

threshold = 0.15 * np.max(energy)

speech_frames = energy > threshold

speech_indices = np.where(
    speech_frames
)[0]

# -----------------------------------------
# Active speech duration
# -----------------------------------------

if len(speech_indices) > 0:

    active_frames = np.sum(
        speech_frames
    )

    active_duration = (
        active_frames * hop_size
    ) / SAMPLE_RATE

else:

    active_duration = 0

# -----------------------------------------
# Speech-to-text
# -----------------------------------------

recognizer = sr.Recognizer()

with sr.AudioFile(AUDIO_FILE) as source:

    recorded_audio = recognizer.record(
        source
    )

try:

    text = recognizer.recognize_google(
        recorded_audio
    )

    words = text.split()

    word_count = len(words)

except sr.UnknownValueError:

    text = ""

    word_count = 0

except sr.RequestError:

    text = ""

    word_count = 0

# -----------------------------------------
# Calculate WPM
# -----------------------------------------

if active_duration > 0:

    wpm = (
        word_count /
        active_duration
    ) * 60

else:

    wpm = 0

# -----------------------------------------
# Speech activity percentage
# -----------------------------------------

activity_percentage = (
    active_duration /
    DURATION
) * 100

# -----------------------------------------
# Display results
# -----------------------------------------

print("\n================================")
print("        ANALYSIS RESULTS")
print("================================")

print(f"\nTranscript:")
print(text)

print(f"\nWord Count           : {word_count}")
print(f"Total Duration       : {DURATION:.2f} sec")
print(f"Active Speech        : {active_duration:.2f} sec")
print(f"Speech Activity      : {activity_percentage:.2f}%")
print(f"Speaking Rate        : {wpm:.2f} WPM")

print("\n================================")