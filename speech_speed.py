import soundfile as sf

# --------------------------------
# Load recorded speech
# --------------------------------
audio, sample_rate = sf.read("voice_original.wav")

# Duration
duration = len(audio) / sample_rate

# --------------------------------
# Enter number of words spoken
# --------------------------------
words = int(input("Enter the number of words you spoke: "))

# --------------------------------
# Calculate Words Per Minute
# --------------------------------
wpm = (words / duration) * 60

# --------------------------------
# Display results
# --------------------------------
print("\n================================")
print("SPEECH SPEED ANALYSIS")
print("================================")

print("Number of words :", words)
print("Duration        :", round(duration, 2), "seconds")
print("Sampling rate   :", sample_rate, "Hz")
print("Speech speed    :", round(wpm, 2), "WPM")

print("================================")