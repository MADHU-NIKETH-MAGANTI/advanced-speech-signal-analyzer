import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --------------------------------
# Define a simple digital LTI system
# --------------------------------
# H(z) = (1 - 0.5z^-1) / (1 - 0.8z^-1)

b = [1, -0.5]      # Numerator
a = [1, -0.8]      # Denominator

# --------------------------------
# Find zeros and poles
# --------------------------------
zeros = np.roots(b)
poles = np.roots(a)

print("================================")
print("Z-TRANSFORM ANALYSIS")
print("================================")

print("Zeros:")
print(zeros)

print("\nPoles:")
print(poles)

# --------------------------------
# Stability check
# --------------------------------
stable = np.all(np.abs(poles) < 1)

print("\nPole Magnitudes:")
for p in poles:
    print(abs(p))

if stable:
    print("\nSystem Stability: STABLE")
else:
    print("\nSystem Stability: UNSTABLE")

# --------------------------------
# Pole-Zero Plot
# --------------------------------
plt.figure(figsize=(7, 7))

# Unit circle
theta = np.linspace(0, 2 * np.pi, 500)

plt.plot(
    np.cos(theta),
    np.sin(theta),
    linestyle="--",
    label="Unit Circle"
)

# Plot zeros
plt.scatter(
    np.real(zeros),
    np.imag(zeros),
    marker="o",
    s=100,
    label="Zeros"
)

# Plot poles
plt.scatter(
    np.real(poles),
    np.imag(poles),
    marker="x",
    s=100,
    label="Poles"
)

# Axes
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)

plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.title("Pole-Zero Plot")
plt.grid()
plt.axis("equal")
plt.legend()

plt.tight_layout()
plt.show()

print("================================")
print("Z-Transform Analysis Completed")
print("================================")