import tkinter as tk
from tkinter import messagebox
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# SETTINGS
# ============================================================

DURATION = 5
SAMPLE_RATE = 16000
AUDIO_FILE = "voice_original.wav"


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()
root.title("Speech Speed Analyzer - Signals and Systems")
root.geometry("1250x950")
root.configure(bg="white")


# ============================================================
# TITLE
# ============================================================

tk.Label(
    root,
    text="SPEECH SPEED ANALYZER",
    font=("Arial", 24, "bold"),
    bg="white"
).pack(pady=12)

tk.Label(
    root,
    text="Signals and Systems Based Speech Analysis",
    font=("Arial", 12),
    bg="white"
).pack()


# ============================================================
# RECORD BUTTON
# ============================================================

record_button = tk.Button(
    root,
    text="🎤  RECORD SPEECH",
    font=("Arial", 14, "bold"),
    padx=30,
    pady=10
)

record_button.pack(pady=12)


status_label = tk.Label(
    root,
    text="Press Record Speech and speak for 5 seconds",
    font=("Arial", 11),
    bg="white"
)

status_label.pack()


# ============================================================
# SIGNAL INFORMATION
# ============================================================

info_frame = tk.Frame(root, bg="white")
info_frame.pack(pady=8)

duration_label = tk.Label(
    info_frame,
    text="Duration: -- sec",
    font=("Arial", 11),
    bg="white"
)
duration_label.grid(row=0, column=0, padx=30)

samples_label = tk.Label(
    info_frame,
    text="Samples: --",
    font=("Arial", 11),
    bg="white"
)
samples_label.grid(row=0, column=1, padx=30)

sampling_label = tk.Label(
    info_frame,
    text="Sampling Frequency: -- Hz",
    font=("Arial", 11),
    bg="white"
)
sampling_label.grid(row=0, column=2, padx=30)


# ============================================================
# SPEECH SPEED
# ============================================================

speed_frame = tk.LabelFrame(
    root,
    text="SPEECH SPEED",
    font=("Arial", 12, "bold"),
    bg="white",
    padx=15,
    pady=8
)

speed_frame.pack(fill="x", padx=30, pady=5)

speech_text_label = tk.Label(
    speed_frame,
    text="Detected Speech: --",
    font=("Arial", 11),
    bg="white",
    wraplength=1100,
    justify="left"
)
speech_text_label.pack(anchor="w")

words_label = tk.Label(
    speed_frame,
    text="Words: --",
    font=("Arial", 11),
    bg="white"
)
words_label.pack(anchor="w")

wpm_label = tk.Label(
    speed_frame,
    text="Speech Speed: -- WPM",
    font=("Arial", 13, "bold"),
    bg="white"
)
wpm_label.pack(anchor="w")


# ============================================================
# Z-TRANSFORM INFORMATION
# ============================================================

z_frame = tk.LabelFrame(
    root,
    text="Z-TRANSFORM / LTI ANALYSIS",
    font=("Arial", 12, "bold"),
    bg="white",
    padx=15,
    pady=8
)

z_frame.pack(fill="x", padx=30, pady=5)

z_label = tk.Label(
    z_frame,
    text=(
        "Impulse Response: h[n] = [1/5, 1/5, 1/5, 1/5, 1/5]\n"
        "H(z) = (1/5)(1 + z⁻¹ + z⁻² + z⁻³ + z⁻⁴)\n"
        "System: FIR LTI System    |    Causal: Yes    |    BIBO Stable: Yes\n"
        "Poles: z = 0 (multiplicity 4)    |    Zeros: 4 non-zero roots\n"
        "ROC: |z| > 0"
    ),
    font=("Arial", 10),
    bg="white",
    justify="left"
)

z_label.pack(anchor="w")


# ============================================================
# MATPLOTLIB FIGURE
# ============================================================

figure = plt.Figure(
    figsize=(12, 9),
    dpi=85
)

ax1 = figure.add_subplot(321)
ax2 = figure.add_subplot(322)
ax3 = figure.add_subplot(323)
ax4 = figure.add_subplot(324)
ax5 = figure.add_subplot(325)
ax6 = figure.add_subplot(326)

figure.tight_layout(pad=3)


canvas = FigureCanvasTkAgg(
    figure,
    master=root
)

canvas.get_tk_widget().pack(
    fill=tk.BOTH,
    expand=True,
    padx=20,
    pady=5
)


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def record_and_analyze():

    record_button.config(state="disabled")

    status_label.config(
        text="🎤 Recording... Speak now!"
    )

    root.update()

    # --------------------------------------------------------
    # RECORD SPEECH
    # --------------------------------------------------------

    try:

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

    except Exception as e:

        messagebox.showerror(
            "Recording Error",
            str(e)
        )

        record_button.config(state="normal")
        return


    status_label.config(
        text="Processing speech signal..."
    )

    root.update()


    # ========================================================
    # BASIC SIGNAL INFORMATION
    # ========================================================

    N = len(audio)

    duration = N / SAMPLE_RATE

    time = np.arange(N) / SAMPLE_RATE

    duration_label.config(
        text=f"Duration: {duration:.2f} sec"
    )

    samples_label.config(
        text=f"Samples: {N}"
    )

    sampling_label.config(
        text=f"Sampling Frequency: {SAMPLE_RATE} Hz"
    )


    # ========================================================
    # LTI FILTER
    # ========================================================

    h = np.ones(5) / 5


    # ========================================================
    # CONVOLUTION - SMOOTHING
    # ========================================================

    convolved_audio = np.convolve(
        audio,
        h,
        mode="same"
    )


    # ========================================================
    # EDGE ENHANCEMENT
    # ========================================================

    edge_filter = np.array([-1, 1])

    edge_audio = np.convolve(
        audio,
        edge_filter,
        mode="same"
    )


    # ========================================================
    # SPEECH DTFT USING FFT
    # ========================================================

    X = np.fft.rfft(audio)

    frequencies = np.fft.rfftfreq(
        N,
        1 / SAMPLE_RATE
    )

    magnitude = np.abs(X)

    if np.max(magnitude) > 0:
        magnitude = magnitude / np.max(magnitude)

    # Wrapped phase: -π to +π
    phase = np.angle(X)


    # ========================================================
    # FILTER FREQUENCY RESPONSE
    # ========================================================

    H = np.fft.rfft(
        h,
        4096
    )

    filter_frequency = np.fft.rfftfreq(
        4096,
        1 / SAMPLE_RATE
    )

    filter_magnitude = np.abs(H)

    filter_phase = np.angle(H)


    # ========================================================
    # POLES AND ZEROS
    # ========================================================

    # H(z) =
    # (1/5)(z^4 + z^3 + z^2 + z + 1) / z^4

    zeros = np.roots(
        [1, 1, 1, 1, 1]
    )

    poles = np.zeros(4)


    # ========================================================
    # SPEECH RECOGNITION
    # ========================================================

    recognizer = sr.Recognizer()

    detected_text = "Could not understand speech"
    word_count = 0

    try:

        with sr.AudioFile(AUDIO_FILE) as source:

            recorded_audio = recognizer.record(source)

        detected_text = recognizer.recognize_google(
            recorded_audio
        )

        word_count = len(
            detected_text.split()
        )

    except sr.UnknownValueError:

        detected_text = "Could not understand speech"

    except sr.RequestError:

        detected_text = "Speech recognition service unavailable"


    # ========================================================
    # WPM
    # ========================================================

    if word_count > 0:

        wpm = (word_count / duration) * 60

    else:

        wpm = 0


    speech_text_label.config(
        text=f"Detected Speech: {detected_text}"
    )

    words_label.config(
        text=f"Words: {word_count}"
    )

    wpm_label.config(
        text=f"Speech Speed: {wpm:.2f} WPM"
    )


    # ========================================================
    # CLEAR OLD PLOTS
    # ========================================================

    for ax in [
        ax1, ax2, ax3,
        ax4, ax5, ax6
    ]:
        ax.clear()


    # ========================================================
    # 1. SPEECH WAVEFORM
    # ========================================================

    ax1.plot(
        time,
        audio
    )

    ax1.set_title(
        "TIME DOMAIN - Speech Waveform"
    )

    ax1.set_xlabel(
        "Time (seconds)"
    )

    ax1.set_ylabel(
        "Amplitude"
    )

    ax1.grid()


    # ========================================================
    # 2. SPEECH DTFT MAGNITUDE
    # ========================================================

    ax2.plot(
        frequencies,
        magnitude
    )

    ax2.set_title(
        "DTFT - Speech Magnitude"
    )

    ax2.set_xlabel(
        "Frequency (Hz)"
    )

    ax2.set_ylabel(
        "|X(e^jω)|"
    )

    ax2.set_xlim(
        0,
        SAMPLE_RATE / 2
    )

    ax2.grid()


    # ========================================================
    # 3. SPEECH DTFT PHASE
    # ========================================================

    ax3.plot(
        frequencies,
        phase
    )

    ax3.set_title(
        "DTFT - Speech Phase"
    )

    ax3.set_xlabel(
        "Frequency (Hz)"
    )

    ax3.set_ylabel(
        "Phase (rad)"
    )

    ax3.set_ylim(
        -np.pi,
        np.pi
    )

    ax3.grid()


    # ========================================================
    # 4. CONVOLUTION / SMOOTHING
    # ========================================================

    ax4.plot(
        time,
        convolved_audio
    )

    ax4.set_title(
        "LTI Convolution - Smoothed Speech"
    )

    ax4.set_xlabel(
        "Time (seconds)"
    )

    ax4.set_ylabel(
        "Amplitude"
    )

    ax4.grid()


    # ========================================================
    # 5. EDGE ENHANCEMENT
    # ========================================================

    ax5.plot(
        time,
        edge_audio
    )

    ax5.set_title(
        "LTI Convolution - Edge Enhancement"
    )

    ax5.set_xlabel(
        "Time (seconds)"
    )

    ax5.set_ylabel(
        "Amplitude"
    )

    ax5.grid()


    # ========================================================
    # 6. FILTER FREQUENCY RESPONSE
    # ========================================================

    ax6.plot(
        filter_frequency,
        filter_magnitude,
        label="Magnitude"
    )

    ax6.plot(
        filter_frequency,
        filter_phase / np.pi,
        label="Phase / π"
    )

    ax6.set_title(
        "LTI Filter - Magnitude & Phase"
    )

    ax6.set_xlabel(
        "Frequency (Hz)"
    )

    ax6.set_ylabel(
        "Response"
    )

    ax6.set_xlim(
        0,
        SAMPLE_RATE / 2
    )

    ax6.grid()
    ax6.legend()


    # ========================================================
    # UPDATE
    # ========================================================

    figure.tight_layout(
        pad=3
    )

    canvas.draw()


    status_label.config(
        text="✅ Analysis completed successfully!"
    )

    record_button.config(
        state="normal"
    )


# ============================================================
# BUTTON COMMAND
# ============================================================

record_button.config(
    command=record_and_analyze
)


# ============================================================
# START
# ============================================================

root.mainloop()