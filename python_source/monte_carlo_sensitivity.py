import numpy as np
from linear_model import linear_model
from monte_carlo import generate_sample_matrices_mc, calculate_sensitivity_indices_mc


def generate_sample_matrices_mc(Ns, Nrv, jpdf):
    """Generate the A, B, and C sample matrices for the Sobol MC method."""
    # A and B
    A = jpdf.sample(Ns).T
    B = jpdf.sample(Ns).T

    # C-matrices: shape (Nrv, Ns, Nrv)
    C = np.zeros((Nrv, Ns, Nrv))
    for i in range(Nrv):
        C[i] = A.copy()
        C[i, :, i] = B[:, i]

    return A, B, C


def evaluate_model_mc(A, B, C, w):
    """Evaluate model on A, B, and all C matrices."""
    Y_A = linear_model(w, A)
    Y_B = linear_model(w, B)
    Nrv = C.shape[0]
    Y_C = np.zeros((A.shape[0], Nrv))
    for i in range(Nrv):
        Y_C[:, i] = linear_model(w, C[i])
    return Y_A, Y_B, Y_C


def sobol_indices_mc(Y_A, Y_B, Y_C):
    """Compute first-order and total-order Sobol indices."""
    return calculate_sensitivity_indices_mc(Y_A, Y_B, Y_C)
