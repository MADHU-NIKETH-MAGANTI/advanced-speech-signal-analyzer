import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# LTI smoothing filter used
# on our recorded speech
# --------------------------------
h = np.ones(5) / 5

# --------------------------------
# Find zeros of the filter
# --------------------------------
zeros = np.roots(h)

# No finite poles for FIR filter
poles = []

print("================================")
print("Z-TRANSFORM OF SPEECH FILTER")
print("================================")

print("Impulse Response h[n]:")
print(h)

print("\nZ-Transform:")
print("H(z) = (1/5)(1 + z^-1 + z^-2 + z^-3 + z^-4)")

print("\nZeros:")
for z in zeros:
    print(z)

print("\nPoles:")
print("No finite poles (FIR system)")

print("\nROC:")
print("Entire z-plane except z = 0")

print("\nStability:")
print("STABLE")
print("Reason: FIR system has finite impulse response")

# --------------------------------
# Pole-Zero Plot
# --------------------------------
plt.figure(figsize=(7, 7))

# Unit circle
theta = np.linspace(0, 2 * np.pi, 500)

plt.plot(
    np.cos(theta),
    np.sin(theta),
    "--",
    label="Unit Circle"
)

# Zeros
plt.scatter(
    np.real(zeros),
    np.imag(zeros),
    marker="o",
    s=100,
    label="Zeros"
)

# Axes
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)

plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.title("Pole-Zero Plot of Speech Smoothing Filter")

plt.grid()
plt.axis("equal")
plt.legend()

plt.tight_layout()
plt.show()

print("================================")
print("Analysis Completed")
print("================================")