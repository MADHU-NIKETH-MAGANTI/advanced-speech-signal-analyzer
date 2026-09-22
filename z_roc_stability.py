import numpy as np

# Actual smoothing filter used for our speech
h = np.ones(5) / 5

print("================================")
print("ROC, CAUSALITY AND STABILITY")
print("================================")

print("\nImpulse Response:")
print(h)

# FIR system
print("\nSystem Type: FIR")

# Causality
print("Causality: CAUSAL")
print("Reason: h[n] exists only for n >= 0")

# ROC
print("\nROC:")
print("Entire z-plane")

# Stability
print("\nBIBO Stability: STABLE")
print("Reason: Sum of |h[n]| is finite")

# Calculate absolute sum
absolute_sum = np.sum(np.abs(h))

print("\nSum of |h[n]| =", absolute_sum)

if absolute_sum < np.inf:
    print("Therefore, the system is BIBO STABLE.")

print("================================")
print("Analysis Completed")
print("================================")