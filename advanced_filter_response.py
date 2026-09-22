import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, freqz

SAMPLE_RATE = 16000

# -----------------------------------------
# Design filters
# -----------------------------------------

low_b, low_a = butter(
    4,
    1000,
    btype="low",
    fs=SAMPLE_RATE
)

high_b, high_a = butter(
    4,
    300,
    btype="high",
    fs=SAMPLE_RATE
)

band_b, band_a = butter(
    4,
    [300, 3000],
    btype="band",
    fs=SAMPLE_RATE
)

# -----------------------------------------
# Frequency responses
# -----------------------------------------

w_low, h_low = freqz(
    low_b,
    low_a,
    worN=4096,
    fs=SAMPLE_RATE
)

w_high, h_high = freqz(
    high_b,
    high_a,
    worN=4096,
    fs=SAMPLE_RATE
)

w_band, h_band = freqz(
    band_b,
    band_a,
    worN=4096,
    fs=SAMPLE_RATE
)

# -----------------------------------------
# Magnitude responses
# -----------------------------------------

plt.figure(figsize=(12, 7))

plt.plot(
    w_low,
    20 * np.log10(np.maximum(np.abs(h_low), 1e-10)),
    label="Low-pass"
)

plt.plot(
    w_high,
    20 * np.log10(np.maximum(np.abs(h_high), 1e-10)),
    label="High-pass"
)

plt.plot(
    w_band,
    20 * np.log10(np.maximum(np.abs(h_band), 1e-10)),
    label="Band-pass"
)

plt.axhline(
    -3,
    linestyle="--",
    label="-3 dB"
)

plt.title("Digital Filter Magnitude Responses")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")

plt.xlim(0, 5000)

plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------------------
# Phase responses
# -----------------------------------------

plt.figure(figsize=(12, 7))

plt.plot(
    w_low,
    np.unwrap(np.angle(h_low)),
    label="Low-pass"
)

plt.plot(
    w_high,
    np.unwrap(np.angle(h_high)),
    label="High-pass"
)

plt.plot(
    w_band,
    np.unwrap(np.angle(h_band)),
    label="Band-pass"
)

plt.title("Digital Filter Phase Responses")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")

plt.xlim(0, 5000)

plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

print("\n===== FILTER FREQUENCY RESPONSE =====")

print("Low-pass cutoff  : 1000 Hz")
print("High-pass cutoff : 300 Hz")
print("Band-pass range  : 300 - 3000 Hz")

print("\nMagnitude and phase responses calculated.")