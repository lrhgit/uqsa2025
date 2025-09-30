"""
Test functions for sensitivity analysis
Author: Leif Rune Hellevik
Date: 2025-09-29
"""

import numpy as np

# Improved versions of the original functions, using only numpy

def A1(sm):
    """Product of increasing subsets of columns, alternating sign."""
    k = sm.shape[1]
    result = np.zeros(sm.shape[0])
    for j in range(k):
        prod = np.prod(sm[:, :j+1], axis=1)
        result += (-1)**(j+1) * prod
    return result

def A2(sm, a2):
    """Smooth function depending on parameter vector a2."""
    out = (np.abs(4 * sm - 2) + a2) / (1 + a2)
    return np.prod(out, axis=1)

def B1(sm):
    """Normalized linear decrease."""
    k = sm.shape[1]
    out = (k - sm) / (k - 0.5)
    return np.prod(out, axis=1)

def B2(sm):
    """Power function scaled to sum to 1."""
    k = sm.shape[1]
    out = sm ** (1 / k)
    return np.prod(out, axis=1) * (1 + 1 / k)**k

def B3(sm, b3):
    """Smooth function similar to A2, depending on b3."""
    out = (np.abs(4 * sm - 2) + b3) / (1 + b3)
    return np.prod(out, axis=1)

def C1(sm):
    """Product of transformed inputs."""
    out = np.abs(4 * sm - 2)
    return np.prod(out, axis=1)

def C2(sm):
    """Simple product scaled by 2^k."""
    k = sm.shape[1]
    return np.prod(sm, axis=1) * 2**k

# Convenience dictionary for lookup
functions = {
    "A1": A1,
    "A2": A2,
    "B1": B1,
    "B2": B2,
    "B3": B3,
    "C1": C1,
    "C2": C2,
}
