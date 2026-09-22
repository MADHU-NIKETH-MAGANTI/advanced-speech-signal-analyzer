import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr

from scipy.signal import butter, filtfilt, spectrogram

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
DURATION = 5
AUDIO_FILE = "voice_original.wav"


# ============================================================
# MAIN APPLICATION
# ============================================================

class SpeechAnalyzer:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Advanced Speech Signal Processing & Speed Analyzer"
        )

        self.root.geometry("1400x900")

        self.root.configure(bg="#f2f2f2")

        # Data
        self.audio = None
        self.sample_rate = SAMPLE_RATE

        self.total_duration = 0
        self.active_duration = 0

        self.word_count = 0
        self.wpm = 0

        self.transcript = ""

        # -----------------------------
        # Title
        # -----------------------------

        title = tk.Label(
            root,
            text="ADVANCED SPEECH SIGNAL PROCESSING & SPEED ANALYZER",
            font=("Arial", 20, "bold"),
            bg="#f2f2f2"
        )

        title.pack(pady=12)

        subtitle = tk.Label(
            root,
            text="DSP • LTI Systems • DTFT/FFT • Z-Transform • Speech Analysis",
            font=("Arial", 11),
            bg="#f2f2f2"
        )

        subtitle.pack()

        # -----------------------------
        # Control section
        # -----------------------------

        control_frame = tk.Frame(
            root,
            bg="#f2f2f2"
        )

        control_frame.pack(pady=12)

        self.record_button = tk.Button(
            control_frame,
            text="🎤  RECORD & ANALYZE",
            font=("Arial", 13, "bold"),
            command=self.record_and_analyze,
            padx=20,
            pady=10
        )

        self.record_button.grid(
            row=0,
            column=0,
            padx=10
        )

        self.status_label = tk.Label(
            control_frame,
            text="Ready",
            font=("Arial", 11),
            bg="#f2f2f2"
        )

        self.status_label.grid(
            row=0,
            column=1,
            padx=20
        )

        # -----------------------------
        # Notebook
        # -----------------------------

        self.notebook = ttk.Notebook(root)

        self.notebook.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # Create tabs

        self.time_tab = ttk.Frame(self.notebook)
        self.frequency_tab = ttk.Frame(self.notebook)
        self.spectrogram_tab = ttk.Frame(self.notebook)
        self.filter_tab = ttk.Frame(self.notebook)
        self.z_tab = ttk.Frame(self.notebook)
        self.speech_tab = ttk.Frame(self.notebook)
        self.summary_tab = ttk.Frame(self.notebook)

        self.notebook.add(
            self.time_tab,
            text="Time Domain"
        )

        self.notebook.add(
            self.frequency_tab,
            text="Frequency Domain"
        )

        self.notebook.add(
            self.spectrogram_tab,
            text="Spectrogram"
        )

        self.notebook.add(
            self.filter_tab,
            text="LTI Filters"
        )

        self.notebook.add(
            self.z_tab,
            text="Z-Transform"
        )

        self.notebook.add(
            self.speech_tab,
            text="Speech Analysis"
        )

        self.notebook.add(
            self.summary_tab,
            text="Final Results"
        )


    # ========================================================
    # RECORD SPEECH
    # ========================================================

    def record_and_analyze(self):

        try:

            self.status_label.config(
                text="Recording..."
            )

            self.root.update()

            audio = sd.rec(
                int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )

            sd.wait()

            audio = audio.flatten()

            # Remove DC component
            audio = audio - np.mean(audio)

            # Normalize
            peak = np.max(
                np.abs(audio)
            )

            if peak > 0:

                audio = audio / peak

            self.audio = audio
            self.sample_rate = SAMPLE_RATE

            sf.write(
                AUDIO_FILE,
                audio,
                SAMPLE_RATE
            )

            self.status_label.config(
                text="Analyzing..."
            )

            self.root.update()

            self.perform_analysis()

            self.status_label.config(
                text="Analysis completed ✓"
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )

            self.status_label.config(
                text="Error"
            )


    # ========================================================
    # MAIN ANALYSIS
    # ========================================================

    def perform_analysis(self):

        audio = self.audio
        fs = self.sample_rate

        self.total_duration = (
            len(audio) / fs
        )

        # ----------------------------------------------------
        # TIME FEATURES
        # ----------------------------------------------------

        self.peak_amplitude = np.max(
            np.abs(audio)
        )

        self.rms = np.sqrt(
            np.mean(audio ** 2)
        )

        self.signal_energy = np.sum(
            audio ** 2
        )

        self.signal_power = np.mean(
            audio ** 2
        )

        if self.rms > 0:

            self.crest_factor = (
                self.peak_amplitude /
                self.rms
            )

            self.dynamic_range = (
                20 *
                np.log10(
                    self.peak_amplitude /
                    self.rms
                )
            )

        else:

            self.crest_factor = 0
            self.dynamic_range = 0

        # ----------------------------------------------------
        # FRAME ANALYSIS
        # ----------------------------------------------------

        frame_size = int(
            0.025 * fs
        )

        hop_size = int(
            0.010 * fs
        )

        self.rms_frames = []
        self.zcr_frames = []

        frame_times = []

        for start in range(
            0,
            len(audio) - frame_size,
            hop_size
        ):

            frame = audio[
                start:
                start + frame_size
            ]

            # RMS
            frame_rms = np.sqrt(
                np.mean(frame ** 2)
            )

            # ZCR
            signs = np.sign(frame)

            crossings = np.sum(
                np.abs(
                    np.diff(signs)
                )
            ) / 2

            zcr = crossings / (
                len(frame) - 1
            )

            self.rms_frames.append(
                frame_rms
            )

            self.zcr_frames.append(
                zcr
            )

            frame_times.append(
                start / fs
            )

        self.rms_frames = np.array(
            self.rms_frames
        )

        self.zcr_frames = np.array(
            self.zcr_frames
        )

        self.frame_times = np.array(
            frame_times
        )

        # ----------------------------------------------------
        # VAD
        # ----------------------------------------------------

        threshold = (
            0.15 *
            np.max(self.rms_frames)
        )

        self.speech_frames = (
            self.rms_frames > threshold
        )

        speech_indices = np.where(
            self.speech_frames
        )[0]

        if len(speech_indices) > 0:

            self.speech_start = (
                self.frame_times[
                    speech_indices[0]
                ]
            )

            self.speech_end = (
                self.frame_times[
                    speech_indices[-1]
                ]
                +
                frame_size / fs
            )

        else:

            self.speech_start = 0
            self.speech_end = 0

        self.active_duration = (
            np.sum(self.speech_frames)
            *
            hop_size / fs
        )

        self.speech_activity = (
            self.active_duration /
            self.total_duration
        ) * 100

        # ----------------------------------------------------
        # PAUSE ANALYSIS
        # ----------------------------------------------------

        changes = np.diff(
            self.speech_frames.astype(int)
        )

        speech_starts = np.where(
            changes == 1
        )[0]

        speech_ends = np.where(
            changes == -1
        )[0]

        if self.speech_frames[0]:

            speech_starts = np.insert(
                speech_starts,
                0,
                0
            )

        if self.speech_frames[-1]:

            speech_ends = np.append(
                speech_ends,
                len(self.speech_frames) - 1
            )

        self.speech_segments = []

        for start, end in zip(
            speech_starts,
            speech_ends
        ):

            start_time = (
                self.frame_times[start]
            )

            end_time = (
                self.frame_times[end]
                +
                frame_size / fs
            )

            self.speech_segments.append(
                (start_time, end_time)
            )

        self.pauses = []

        for i in range(
            len(self.speech_segments) - 1
        ):

            pause_start = (
                self.speech_segments[i][1]
            )

            pause_end = (
                self.speech_segments[i + 1][0]
            )

            pause_duration = (
                pause_end - pause_start
            )

            if pause_duration > 0:

                self.pauses.append(
                    pause_duration
                )

        if len(self.pauses) > 0:

            self.average_pause = np.mean(
                self.pauses
            )

            self.longest_pause = np.max(
                self.pauses
            )

        else:

            self.average_pause = 0
            self.longest_pause = 0

        self.total_pause = (
            self.total_duration -
            self.active_duration
        )

        # ----------------------------------------------------
        # FFT / DTFT-BASED ANALYSIS
        # ----------------------------------------------------

        N = len(audio)

        spectrum = np.fft.rfft(
            audio
        )

        self.frequencies = (
            np.fft.rfftfreq(
                N,
                1 / fs
            )
        )

        self.magnitude = np.abs(
            spectrum
        )

        self.phase = np.angle(
            spectrum
        )

        # Remove DC for feature calculations
        magnitude_features = (
            self.magnitude.copy()
        )

        magnitude_features[0] = 0

        if np.max(
            magnitude_features
        ) > 0:

            dominant_index = np.argmax(
                magnitude_features
            )

            self.dominant_frequency = (
                self.frequencies[
                    dominant_index
                ]
            )

        else:

            self.dominant_frequency = 0

        total_magnitude = np.sum(
            magnitude_features
        )

        if total_magnitude > 0:

            self.spectral_centroid = (
                np.sum(
                    self.frequencies *
                    magnitude_features
                )
                /
                total_magnitude
            )

            self.spectral_bandwidth = np.sqrt(
                np.sum(
                    (
                        self.frequencies -
                        self.spectral_centroid
                    ) ** 2
                    *
                    magnitude_features
                )
                /
                total_magnitude
            )

            cumulative = np.cumsum(
                magnitude_features
            )

            rolloff_value = (
                0.85 *
                cumulative[-1]
            )

            rolloff_index = np.where(
                cumulative >= rolloff_value
            )[0][0]

            self.spectral_rolloff = (
                self.frequencies[
                    rolloff_index
                ]
            )

        else:

            self.spectral_centroid = 0
            self.spectral_bandwidth = 0
            self.spectral_rolloff = 0

        # ----------------------------------------------------
        # SPECTROGRAM
        # ----------------------------------------------------

        (
            self.spec_frequencies,
            self.spec_times,
            self.spec_power
        ) = spectrogram(
            audio,
            fs=fs,
            window="hann",
            nperseg=512,
            noverlap=384
        )

        self.spec_db = (
            10 *
            np.log10(
                self.spec_power +
                1e-12
            )
        )

        # ----------------------------------------------------
        # FILTER BANK
        # ----------------------------------------------------

        self.low_b, self.low_a = butter(
            4,
            1000,
            btype="low",
            fs=fs
        )

        self.high_b, self.high_a = butter(
            4,
            300,
            btype="high",
            fs=fs
        )

        self.band_b, self.band_a = butter(
            4,
            [300, 3000],
            btype="band",
            fs=fs
        )

        self.low_pass = filtfilt(
            self.low_b,
            self.low_a,
            audio
        )

        self.high_pass = filtfilt(
            self.high_b,
            self.high_a,
            audio
        )

        self.band_pass = filtfilt(
            self.band_b,
            self.band_a,
            audio
        )

        # ----------------------------------------------------
        # FILTER FREQUENCY RESPONSES
        # ----------------------------------------------------

        n_freq = 4096

        self.filter_frequencies = np.fft.rfftfreq(
            n_freq,
            1 / fs
        )

        self.low_response = np.fft.rfft(
            self.low_b,
            n_freq
        ) / (
            np.fft.rfft(
                self.low_a,
                n_freq
            )
        )

        self.high_response = np.fft.rfft(
            self.high_b,
            n_freq
        ) / (
            np.fft.rfft(
                self.high_a,
                n_freq
            )
        )

        self.band_response = np.fft.rfft(
            self.band_b,
            n_freq
        ) / (
            np.fft.rfft(
                self.band_a,
                n_freq
            )
        )

        # ----------------------------------------------------
        # Z-TRANSFORM
        # Actual smoothing filter used earlier
        # ----------------------------------------------------

        self.h = np.ones(5) / 5

        # H(z) = 1/5(1+z^-1+z^-2+z^-3+z^-4)
        #
        # Numerator after multiplying by z^4:
        # z^4 + z^3 + z^2 + z + 1
        #
        # Four poles at z = 0

        self.zeros = np.roots(
            [1, 1, 1, 1, 1]
        )

        self.poles = np.zeros(4)

        self.roc = "|z| > 0"

        self.stability = "BIBO Stable"

        # ----------------------------------------------------
        # SPEECH RECOGNITION
        # ----------------------------------------------------

        self.recognize_speech()

        # ----------------------------------------------------
        # UPDATE ALL GUI TABS
        # ----------------------------------------------------

        self.update_time_tab()
        self.update_frequency_tab()
        self.update_spectrogram_tab()
        self.update_filter_tab()
        self.update_z_tab()
        self.update_speech_tab()
        self.update_summary_tab()


    # ========================================================
    # SPEECH RECOGNITION
    # ========================================================

    def recognize_speech(self):

        recognizer = sr.Recognizer()

        try:

            with sr.AudioFile(
                AUDIO_FILE
            ) as source:

                recorded_audio = (
                    recognizer.record(source)
                )

            self.transcript = (
                recognizer.recognize_google(
                    recorded_audio
                )
            )

            self.word_count = len(
                self.transcript.split()
            )

        except:

            self.transcript = (
                "Speech could not be recognized."
            )

            self.word_count = 0

        if self.active_duration > 0:

            self.wpm = (
                self.word_count /
                self.active_duration
            ) * 60

        else:

            self.wpm = 0


    # ========================================================
    # CLEAR TAB
    # ========================================================

    def clear_tab(self, tab):

        for widget in tab.winfo_children():

            widget.destroy()


    # ========================================================
    # TIME DOMAIN TAB
    # ========================================================

    def update_time_tab(self):

        self.clear_tab(
            self.time_tab
        )

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 9)
        )

        time = (
            np.arange(len(self.audio))
            /
            self.sample_rate
        )

        # Waveform
        axes[0].plot(
            time[::20],
            self.audio[::20]
        )

        axes[0].set_title(
            "Original Speech Waveform"
        )

        axes[0].set_xlabel(
            "Time (seconds)"
        )

        axes[0].set_ylabel(
            "Amplitude"
        )

        axes[0].grid()

        # RMS
        axes[1].plot(
            self.frame_times,
            self.rms_frames
        )

        axes[1].set_title(
            "RMS Energy"
        )

        axes[1].set_xlabel(
            "Time (seconds)"
        )

        axes[1].set_ylabel(
            "RMS"
        )

        axes[1].grid()

        # ZCR
        axes[2].plot(
            self.frame_times,
            self.zcr_frames
        )

        axes[2].set_title(
            "Zero-Crossing Rate"
        )

        axes[2].set_xlabel(
            "Time (seconds)"
        )

        axes[2].set_ylabel(
            "ZCR"
        )

        axes[2].grid()

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            self.time_tab
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

        plt.close(fig)


    # ========================================================
    # FREQUENCY TAB
    # ========================================================

    def update_frequency_tab(self):

        self.clear_tab(
            self.frequency_tab
        )

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 8)
        )

        magnitude = (
            self.magnitude /
            np.max(self.magnitude)
        )

        # Magnitude
        axes[0].plot(
            self.frequencies,
            magnitude
        )

        axes[0].set_title(
            "Speech FFT / DTFT Magnitude"
        )

        axes[0].set_xlabel(
            "Frequency (Hz)"
        )

        axes[0].set_ylabel(
            "Normalized Magnitude"
        )

        axes[0].set_xlim(
            0,
            4000
        )

        axes[0].grid()

        # Phase
        axes[1].plot(
            self.frequencies,
            self.phase
        )

        axes[1].set_title(
            "Speech Phase Spectrum"
        )

        axes[1].set_xlabel(
            "Frequency (Hz)"
        )

        axes[1].set_ylabel(
            "Phase (radians)"
        )

        axes[1].set_xlim(
            0,
            4000
        )

        axes[1].grid()

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            self.frequency_tab
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

        plt.close(fig)

        # Feature information

        info = tk.Label(
            self.frequency_tab,
            text=(
                f"Dominant Frequency: "
                f"{self.dominant_frequency:.2f} Hz     |     "
                f"Spectral Centroid: "
                f"{self.spectral_centroid:.2f} Hz     |     "
                f"Bandwidth: "
                f"{self.spectral_bandwidth:.2f} Hz     |     "
                f"Roll-off: "
                f"{self.spectral_rolloff:.2f} Hz"
            ),
            font=("Arial", 11, "bold")
        )

        info.pack(
            pady=8
        )


    # ========================================================
    # SPECTROGRAM TAB
    # ========================================================

    def update_spectrogram_tab(self):

        self.clear_tab(
            self.spectrogram_tab
        )

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        image = ax.pcolormesh(
            self.spec_times,
            self.spec_frequencies,
            self.spec_db,
            shading="gouraud"
        )

        fig.colorbar(
            image,
            ax=ax,
            label="Power (dB)"
        )

        ax.set_title(
            "Time-Frequency Spectrogram"
        )

        ax.set_xlabel(
            "Time (seconds)"
        )

        ax.set_ylabel(
            "Frequency (Hz)"
        )

        ax.set_ylim(
            0,
            4000
        )

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            self.spectrogram_tab
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

        plt.close(fig)


    # ========================================================
    # FILTER TAB
    # ========================================================

    def update_filter_tab(self):

        self.clear_tab(
            self.filter_tab
        )

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 9)
        )

        time = (
            np.arange(len(self.audio))
            /
            self.sample_rate
        )

        # Original
        axes[0, 0].plot(
            time[::20],
            self.audio[::20]
        )

        axes[0, 0].set_title(
            "Original Speech"
        )

        # Low-pass
        axes[0, 1].plot(
            time[::20],
            self.low_pass[::20]
        )

        axes[0, 1].set_title(
            "Low-Pass Filtered Speech"
        )

        # High-pass
        axes[1, 0].plot(
            time[::20],
            self.high_pass[::20]
        )

        axes[1, 0].set_title(
            "High-Pass Filtered Speech"
        )

        # Band-pass
        axes[1, 1].plot(
            time[::20],
            self.band_pass[::20]
        )

        axes[1, 1].set_title(
            "Band-Pass Filtered Speech"
        )

        for ax in axes.flat:

            ax.set_xlabel(
                "Time (seconds)"
            )

            ax.set_ylabel(
                "Amplitude"
            )

            ax.grid()

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            self.filter_tab
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

        plt.close(fig)

        # Filter responses

        fig2, axes2 = plt.subplots(
            2,
            1,
            figsize=(12, 7)
        )

        f = self.filter_frequencies

        # Magnitude
        axes2[0].plot(
            f,
            20 * np.log10(
                np.maximum(
                    np.abs(
                        self.low_response
                    ),
                    1e-10
                )
            ),
            label="Low-pass"
        )

        axes2[0].plot(
            f,
            20 * np.log10(
                np.maximum(
                    np.abs(
                        self.high_response
                    ),
                    1e-10
                )
            ),
            label="High-pass"
        )

        axes2[0].plot(
            f,
            20 * np.log10(
                np.maximum(
                    np.abs(
                        self.band_response
                    ),
                    1e-10
                )
            ),
            label="Band-pass"
        )

        axes2[0].set_xlim(
            0,
            5000
        )

        axes2[0].set_title(
            "Filter Magnitude Response"
        )

        axes2[0].set_xlabel(
            "Frequency (Hz)"
        )

        axes2[0].set_ylabel(
            "Magnitude (dB)"
        )

        axes2[0].grid()
        axes2[0].legend()

        # Phase
        axes2[1].plot(
            f,
            np.unwrap(
                np.angle(
                    self.low_response
                )
            ),
            label="Low-pass"
        )

        axes2[1].plot(
            f,
            np.unwrap(
                np.angle(
                    self.high_response
                )
            ),
            label="High-pass"
        )

        axes2[1].plot(
            f,
            np.unwrap(
                np.angle(
                    self.band_response
                )
            ),
            label="Band-pass"
        )

        axes2[1].set_xlim(
            0,
            5000
        )

        axes2[1].set_title(
            "Filter Phase Response"
        )

        axes2[1].set_xlabel(
            "Frequency (Hz)"
        )

        axes2[1].set_ylabel(
            "Phase (radians)"
        )

        axes2[1].grid()
        axes2[1].legend()

        fig2.tight_layout()

        canvas2 = FigureCanvasTkAgg(
            fig2,
            self.filter_tab
        )

        canvas2.draw()

        canvas2.get_tk_widget().pack(
            fill="both",
            expand=True
        )

        plt.close(fig2)


    # ========================================================
    # Z-TRANSFORM TAB
    # ========================================================

    def update_z_tab(self):

        self.clear_tab(
            self.z_tab
        )

        fig, ax = plt.subplots(
            figsize=(9, 8)
        )

        # Unit circle

        theta = np.linspace(
            0,
            2 * np.pi,
            500
        )

        ax.plot(
            np.cos(theta),
            np.sin(theta),
            linestyle="--",
            label="Unit Circle"
        )

        # Zeros

        ax.scatter(
            np.real(self.zeros),
            np.imag(self.zeros),
            marker="o",
            s=100,
            label="Zeros"
        )

        # Poles

        ax.scatter(
            np.real(self.poles),
            np.imag(self.poles),
            marker="x",
            s=100,
            label="Poles"
        )

        ax.axhline(
            0,
            linewidth=0.8
        )

        ax.axvline(
            0,
            linewidth=0.8
        )

        ax.set_xlabel(
            "Real"
        )

        ax.set_ylabel(
            "Imaginary"
        )

        ax.set_title(
            "Pole-Zero Plot of Smoothing LTI Filter"
        )

        ax.grid()
        ax.legend()

        ax.set_aspect(
            "equal"
        )

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            self.z_tab
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

        plt.close(fig)

        info = tk.Label(
            self.z_tab,
            text=(
                "H(z) = (1/5)(1 + z⁻¹ + z⁻² + z⁻³ + z⁻⁴)\n\n"
                f"Poles: 4 poles at z = 0\n"
                f"ROC: {self.roc}\n"
                f"Causality: Causal FIR system\n"
                f"Stability: {self.stability}"
            ),
            font=("Arial", 12, "bold"),
            justify="center"
        )

        info.pack(
            pady=10
        )


    # ========================================================
    # SPEECH ANALYSIS TAB
    # ========================================================

    def update_speech_tab(self):

        self.clear_tab(
            self.speech_tab
        )

        frame = tk.Frame(
            self.speech_tab
        )

        frame.pack(
            fill="both",
            expand=True,
            padx=40,
            pady=30
        )

        values = [

            ("Transcript", self.transcript),

            ("Word Count", str(
                self.word_count
            )),

            ("Total Duration",
             f"{self.total_duration:.2f} sec"),

            ("Active Speech",
             f"{self.active_duration:.2f} sec"),

            ("Speech Activity",
             f"{self.speech_activity:.2f}%"),

            ("Speech Segments",
             str(len(
                 self.speech_segments
             ))),

            ("Number of Pauses",
             str(len(
                 self.pauses
             ))),

            ("Total Pause Time",
             f"{self.total_pause:.2f} sec"),

            ("Average Pause",
             f"{self.average_pause:.2f} sec"),

            ("Longest Pause",
             f"{self.longest_pause:.2f} sec"),

            ("Speaking Rate",
             f"{self.wpm:.2f} WPM")
        ]

        for row, (name, value) in enumerate(
            values
        ):

            tk.Label(
                frame,
                text=name + ":",
                font=("Arial", 12, "bold"),
                anchor="w"
            ).grid(
                row=row,
                column=0,
                sticky="w",
                pady=7
            )

            tk.Label(
                frame,
                text=value,
                font=("Arial", 12),
                anchor="w",
                wraplength=900,
                justify="left"
            ).grid(
                row=row,
                column=1,
                sticky="w",
                padx=20,
                pady=7
            )


    # ========================================================
    # FINAL RESULTS TAB
    # ========================================================

    def update_summary_tab(self):

        self.clear_tab(
            self.summary_tab
        )

        frame = tk.Frame(
            self.summary_tab
        )

        frame.pack(
            fill="both",
            expand=True,
            padx=40,
            pady=25
        )

        title = tk.Label(
            frame,
            text="FINAL SPEECH ANALYSIS",
            font=("Arial", 18, "bold")
        )

        title.pack(
            pady=10
        )

        summary = [

            f"Sampling Frequency : {self.sample_rate} Hz",

            f"Samples            : {len(self.audio)}",

            f"Duration           : {self.total_duration:.2f} sec",

            f"Active Speech      : {self.active_duration:.2f} sec",

            f"Speech Activity    : {self.speech_activity:.2f}%",

            f"Word Count         : {self.word_count}",

            f"Speaking Rate      : {self.wpm:.2f} WPM",

            f"RMS Energy         : {self.rms:.5f}",

            f"Signal Energy      : {self.signal_energy:.5f}",

            f"Signal Power       : {self.signal_power:.7f}",

            f"Peak Amplitude     : {self.peak_amplitude:.5f}",

            f"Crest Factor       : {self.crest_factor:.4f}",

            f"Dynamic Range      : {self.dynamic_range:.2f} dB",

            f"Dominant Frequency : "
            f"{self.dominant_frequency:.2f} Hz",

            f"Spectral Centroid  : "
            f"{self.spectral_centroid:.2f} Hz",

            f"Spectral Bandwidth : "
            f"{self.spectral_bandwidth:.2f} Hz",

            f"Spectral Roll-off  : "
            f"{self.spectral_rolloff:.2f} Hz",

            f"Pause Count        : {len(self.pauses)}",

            f"Average Pause      : "
            f"{self.average_pause:.2f} sec",

            f"Longest Pause      : "
            f"{self.longest_pause:.2f} sec",

            f"Z-Transform ROC    : {self.roc}",

            f"LTI Stability      : {self.stability}"
        ]

        for text in summary:

            tk.Label(
                frame,
                text=text,
                font=("Arial", 11),
                anchor="w"
            ).pack(
                anchor="w",
                pady=3
            )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = SpeechAnalyzer(root)

    root.mainloop()