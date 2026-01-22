"""
Module with Monte Carlo methods for uncertainty and  sensitivity analysis using the chaospy package for sampling
"""
import numpy as np

def uq_measures_dummy_func():
    if __name__ == '__main__':
        from sensitivity_examples_nonlinear import generate_distributions
        from sensitivity_examples_nonlinear import linear_model

    # start uq
    # generate the distributions for the problem
    Nrv = 4
    c = 0.5
    zm = np.array([[0., i] for i in range(1, Nrv + 1)])
    wm = np.array([[i * c, i] for i in range(1, Nrv + 1)])
    jpdf = generate_distributions(zm, wm)

    # 1. Generate a set of Xs
    Ns = 20000
    Xs = jpdf.sample(Ns, rule='R').T  # <- transform the sample matrix

    # 2. Evaluate the model
    Zs = Xs[:, :Nrv]
    Ws = Xs[:, Nrv:]
    Ys = linear_model(Ws, Zs)

    # 3. Calculate expectation and variance
    EY = np.mean(Ys)
    VY = np.var(Ys, ddof=1)  # NB: use ddof=1 for unbiased variance estimator, i.e /(Ns - 1)

    print('E(Y): {:2.5f} and  Var(Y): {:2.5f}'.format(EY, VY))
    # end uq


# sample matrices
def generate_sample_matrices_mc(Ns, number_of_parameters, jpdf, sample_method='R'):

    Xtot = jpdf.sample(2*Ns, sample_method).transpose()
    A = Xtot[0:Ns, :]
    B = Xtot[Ns:, :]

    C = np.empty((number_of_parameters, Ns, number_of_parameters))
    # create C sample matrices
    for i in range(number_of_parameters):
        C[i, :, :] = B.copy()
        C[i, :, i] = A[:, i].copy()

    return A, B, C
# end sample matrices



import numpy as np

import numpy as np

def calculate_sensitivity_indices_mc(y_a, y_b, y_c, ddof=1, clip=True):
    """
    First-order (S) and total-order (ST) Sobol indices from Saltelli sampling.

    Inputs
    ------
    y_a : array, shape (Ns,)
        Model evaluations on matrix A.
    y_b : array, shape (Ns,)
        Model evaluations on matrix B.
    y_c : array, shape (Ns, P) or (P, Ns)
        Model evaluations on C_i matrices.
        Either:
          - y_c[:, i] is Y(C_i)  -> shape (Ns, P)
          - y_c[i, :] is Y(C_i)  -> shape (P, Ns)

    Parameters
    ----------
    ddof : int
        Degrees of freedom for variance (1 gives unbiased sample variance).
    clip : bool
        If True, clips S and ST to [0, 1] (helpful for small Ns / noise).

    Returns
    -------
    S  : array, shape (P,)
        First-order Sobol indices.
    ST : array, shape (P,)
        Total-order Sobol indices.
    """
    y_a = np.asarray(y_a).reshape(-1)
    y_b = np.asarray(y_b).reshape(-1)
    y_c = np.asarray(y_c)

    Ns = y_a.size
    if y_b.size != Ns:
        raise ValueError(f"y_b has length {y_b.size}, expected {Ns}")
    if y_c.ndim != 2:
        raise ValueError(f"y_c must be 2D, got shape {y_c.shape}")

    # Normalize y_c to shape (P, Ns), so row i corresponds to C_i
    if y_c.shape[0] == Ns:
        # (Ns, P) -> transpose to (P, Ns)
        y_c = y_c.T
    elif y_c.shape[1] == Ns:
        # already (P, Ns)
        pass
    else:
        raise ValueError(
            f"y_c has incompatible shape {y_c.shape}. "
            f"Expected (Ns,P) or (P,Ns) with Ns={Ns}."
        )

    P = y_c.shape[0]

    # Variance of Y (use A; you can also use np.r_[y_a, y_b] if you prefer)
    VY = np.var(y_a, ddof=ddof)
    if not np.isfinite(VY) or VY <= 0:
        raise ValueError(f"Non-positive or invalid variance VY={VY}")

    # Saltelli estimators (common, robust form)
    #
    # First order:
    #   S_i = (1/N) * sum( y_b * (y_c_i - y_a) ) / V(Y)
    #
    # Total order:
    #   ST_i = (1/(2N)) * sum( (y_a - y_c_i)^2 ) / V(Y)
    #
    # Notes:
    # - These assume C_i is A with column i replaced by B (or equivalent).
    # - If your C_i uses the opposite convention (B with one col from A),
    #   you can swap y_a and y_b in the S-estimator.

    S = np.empty(P, dtype=float)
    ST = np.empty(P, dtype=float)

    for i in range(P):
        y_ci = y_c[i, :].reshape(-1)
        if y_ci.size != Ns:
            raise ValueError(f"y_c row {i} has length {y_ci.size}, expected {Ns}")

        S[i]  = np.mean(y_b * (y_ci - y_a)) / VY
        ST[i] = 0.5 * np.mean((y_a - y_ci) ** 2) / VY

    if clip:
        S = np.clip(S, 0.0, 1.0)
        ST = np.clip(ST, 0.0, 1.0)

    return S, ST
