import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# Input signals
# --------------------------------

x = np.array([1, 2, 3, 4])
h = np.array([1, 1, 1])

# --------------------------------
# Convolution from scratch
# --------------------------------

y = np.zeros(len(x) + len(h) - 1)

for n in range(len(y)):
    for k in range(len(x)):

        if 0 <= n - k < len(h):
            y[n] += x[k] * h[n - k]

# --------------------------------
# Display results
# --------------------------------

print("Input signal x[n]:")
print(x)

print("\nImpulse response h[n]:")
print(h)

print("\nConvolution result y[n]:")
print(y)

# --------------------------------
# Verify with NumPy
# --------------------------------

y_numpy = np.convolve(x, h)

print("\nNumPy convolution:")
print(y_numpy)

print("\nResults match:",
      np.allclose(y, y_numpy))

# --------------------------------
# Plot
# --------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.stem(x)
plt.title("Input Signal x[n]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 1, 2)
plt.stem(h)
plt.title("Impulse Response h[n]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 1, 3)
plt.stem(y)
plt.title("Convolution Output y[n] = x[n] * h[n]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()