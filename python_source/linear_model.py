# --- cell: linear_model ---

import numpy as np

def linear_model(w, z):
    """
    Linear model used across the entire notebook:
        Y = sum_i (w_i * z_i)

    Parameters
    ----------
    w : array-like, shape (Nrv,)
        Weights for each random variable.

    z : array-like, shape (Ns, Nrv)
        Sample matrix where rows are samples and columns correspond to random variables.

    Returns
    -------
    Y : array-like, shape (Ns,)
        Model outputs for each sample.
    """
    return np.sum(w * z, axis=1)

# --- endcell: linear_model ---
