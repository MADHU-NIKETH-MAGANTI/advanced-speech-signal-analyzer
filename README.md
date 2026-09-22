# 🎙️ Advanced Speech Signal Processing & Speed Analyzer

An integrated Python-based Digital Signal Processing (DSP) platform for
analyzing real speech signals recorded through a microphone.

The project combines time-domain, frequency-domain, spectral, filtering,
Z-transform, speech activity, pause and speaking-rate analysis into a
single interactive dashboard.

---

## 📌 Project Overview

Speech is a real-world signal that contains important information in both
the time and frequency domains.

This project applies fundamental Signals and Systems and DSP concepts to
an actual recorded speech signal and provides a unified analysis platform.

The system can analyze:

- How the speech signal varies with time
- Frequency components present in speech
- Speech activity and pauses
- Signal strength and quality
- Spectral characteristics
- Effects of digital filters
- Z-transform properties of an LTI system
- Speaking rate using speech-to-text

---

## ✨ Key Features

### 🎤 Speech Recording
- Records speech directly from a microphone
- Sampling frequency: 16 kHz
- 5-second recording
- Automatic preprocessing and normalization

### 📈 Time-Domain Analysis
- Speech waveform
- RMS analysis
- Zero-Crossing Rate (ZCR)
- Signal energy
- Signal power
- Peak amplitude
- Crest factor
- Dynamic range

### 📊 Frequency-Domain Analysis
- FFT-based frequency spectrum
- Frequency magnitude
- Phase spectrum
- Dominant frequency
- Spectral centroid
- Spectral bandwidth
- Spectral roll-off

### 🌈 Spectrogram Analysis
- Time-frequency representation of speech
- Visualization of changing frequency components over time

### 🗣️ Speech Activity Detection
- Voice Activity Detection (VAD)
- Active speech duration
- Speech activity percentage
- Speech start and end detection

### ⏸️ Pause Analysis
- Number of pauses
- Total pause duration
- Average pause duration
- Longest pause

### 🎚️ Digital Filtering
The system demonstrates digital filtering using:

- Low-pass filter
- High-pass filter
- Band-pass filter

The effect of filtering on the speech signal can be visualized and analyzed.

### 🔄 Convolution
Demonstrates convolution-based speech processing using:

- Smoothing filter
- Edge-enhancement filter

### 🔢 Z-Transform Analysis
The project demonstrates Z-transform concepts using an LTI FIR smoothing system.

Includes:

- Transfer function
- Poles
- Zeros
- Region of Convergence (ROC)
- Causality
- BIBO stability

### 📝 Speech Speed Analysis
Speech-to-text is used to calculate:

- Transcript
- Word count
- Speaking duration
- Words Per Minute (WPM)

### 🖥️ Integrated Dashboard

All major analyses are organized into a single Tkinter-based dashboard.

---

## 🧠 Signals & Systems Concepts Demonstrated

This project connects theoretical concepts with a real-world speech signal.

- Digital signal representation
- Sampling
- Signal operations
- LTI systems
- Convolution
- Fourier analysis
- DTFT
- Digital filtering
- Z-transform
- Pole-zero analysis
- Region of Convergence
- BIBO stability
- Time-frequency analysis
- Signal statistics

---

## 🏗️ System Architecture

```text
             🎤 Microphone
                  │
                  ▼
          Speech Recording
                  │
                  ▼
        Signal Preprocessing
        (DC Removal/Scaling)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Time-Domain Analysis   Frequency Analysis
        │                   │
        ▼                   ▼
 RMS / ZCR              FFT / Spectrum
 Energy / Power         Spectral Features
        │                   │
        └─────────┬─────────┘
                  ▼
          Speech Processing
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
   VAD        Filtering      Convolution
     │            │            │
     ▼            ▼            ▼
  Pauses       LP/HP/BP      Smoothing
     │                         │
     └────────────┬────────────┘
                  ▼
          Z-Transform Analysis
                  │
                  ▼
        Speech Speed Analysis
                  │
                  ▼
       ┌──────────────────────┐
       │  Integrated Dashboard │
       └──────────────────────┘