import speech_recognition as sr
import soundfile as sf

# --------------------------------
# Load recorded speech
# --------------------------------
audio, sample_rate = sf.read("voice_original.wav")

duration = len(audio) / sample_rate

# --------------------------------
# Speech recognition
# --------------------------------
recognizer = sr.Recognizer()

with sr.AudioFile("voice_original.wav") as source:
    recorded_audio = recognizer.record(source)

print("Recognizing your speech...")

try:
    text = recognizer.recognize_google(recorded_audio)

    # Count words
    words = len(text.split())

    # Calculate WPM
    wpm = (words / duration) * 60

    print("\n================================")
    print("AUTOMATIC SPEECH SPEED ANALYSIS")
    print("================================")

    print("Detected Speech:")
    print(text)

    print("\nNumber of Words :", words)
    print("Duration        :", round(duration, 2), "seconds")
    print("Sampling Rate   :", sample_rate, "Hz")
    print("Speech Speed    :", round(wpm, 2), "WPM")

    print("================================")

except sr.UnknownValueError:
    print("❌ Could not understand the speech.")

except sr.RequestError:
    print("❌ Speech recognition service unavailable.")