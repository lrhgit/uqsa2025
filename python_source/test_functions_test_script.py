"""
Test script for test_functions_numpy.py
"""

import numpy as np
import matplotlib.pyplot as plt
from test_functions_numpy import functions

# Generate test input: 100 samples, 4 variables
np.random.seed(0)
Z = np.random.rand(100, 4)
a2 = np.array([0.1, 0.2, 0.3, 0.4])
b3 = np.array([0.4, 0.3, 0.2, 0.1])

# Evaluate all functions
results = {}
for name, func in functions.items():
    if name == "A2":
        results[name] = func(Z, a2)
    elif name == "B3":
        results[name] = func(Z, b3)
    else:
        results[name] = func(Z)

# Plot results
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
axs = axs.flatten()

for i, (name, values) in enumerate(results.items()):
    axs[i].plot(values, label=name)
    axs[i].set_title(name)
    axs[i].set_xlabel("Sample index")
    axs[i].set_ylabel("Output")
    axs[i].legend()

# Hide unused subplots
for j in range(i + 1, len(axs)):
    axs[j].axis("off")

fig.suptitle("Test Function Outputs", fontsize=16)
plt.tight_layout()
plt.show()
