# ============================================================
# Monte Carlo notebook – extracted code cells (7–9)
# For DocOnce inclusion via @@@CODE ... fromto:
# ============================================================




# --- cell: uq_problem_setup ---
# start uq
# generate the distributions for the problem

Nrv = 4

zm = np.array([[0., i] for i in range(1, Nrv + 1)])

c = 0.5
wm = np.array([[i * c, i] for i in range(1, Nrv + 1)])

jpdf = generate_distributions(zm, wm)

# sensitivity analytical values
Sa, Szw, Sta = analytic_sensitivity_coefficients(zm, wm)

# Monte Carlo
# Ns_mc = 1000000  # Number of samples mc
Ns_mc = 10000      # Number of samples mc

# calculate sensitivity indices with mc
A_s, B_s, C_s, f_A, f_B, f_C, Smc, Stmc = monte_carlo_sens_nonlin(Ns_mc, jpdf)



# --- endcell: uq_problem_setup ---


# --- cell: expectation_and_variance ---
# 1. Generate a set of Xs
Ns = 20000
Xs = jpdf.sample(Ns, rule='R').T  # <- transform the sample matrix

# 2. Evaluate the model
Zs = Xs[:, :Nrv]
Ws = Xs[:, Nrv:]
Ys = linear_model(Ws, Zs)

# 3. Calculate expectation and variance
EY = np.mean(Ys)
VY = np.var(Ys, ddof=1)  # NB: ddof=1 for unbiased variance estimator, i.e /(Ns - 1)

# --- endcell: expectation_and_variance ---


# --- cell: load_problem_helpers ---
import numpy as np
import chaospy as cp

from sensitivity_examples_nonlinear import (
    analytic_sensitivity_coefficients,
)

# --- endcell: load_problem_helpers ---


# --- cell: define_problem_and_jpdf ---
# Define the (Z, W) non-additive linear demo problem and construct the joint distribution explicitly

Nrv = 4  # number of Z-variables (and also number of W-variables)

# Each row is [mu, sigma] for a Normal(mu, sigma)
zm = np.array([[0.0, i] for i in range(1, Nrv + 1)])

c = 0.5
wm = np.array([[i * c, i] for i in range(1, Nrv + 1)])

# --- Explicit construction of the joint PDF (independent marginals) ---
# Stack Z and W parameter definitions -> total parameter vector X = (Z1..ZNrv, W1..WNrv)
x_params = np.vstack([zm, wm])  # shape (2*Nrv, 2)

marginals = [cp.Normal(mu, sigma) for (mu, sigma) in x_params]
jpdf = cp.J(*marginals)  # independent joint distribution

# Analytic sensitivity values (reference)
Sa, Szw, Sta = analytic_sensitivity_coefficients(zm, wm)

print("Nrv =", Nrv, " -> total parameters =", len(jpdf))
# --- endcell: define_problem_and_jpdf ---


# --- cell: mc_setup ---
# Monte Carlo sample size and basic dimensions

Ns_mc = 1000
P = len(jpdf)  # total number of stochastic inputs (e.g., Z and W stacked)

print("Ns_mc =", Ns_mc)
print("P     =", P)
# --- endcell: mc_setup ---


# --- cell: saltelli_matrices ---
# Step 1: create Saltelli sampling matrices A, B, and C_i

from monte_carlo import generate_sample_matrices_mc

A_s, B_s, C_s = generate_sample_matrices_mc(
    Ns_mc,
    P,
    jpdf,
    sample_method="R",
)

print("A_s:", np.shape(A_s))
print("B_s:", np.shape(B_s))
print("C_s:", np.shape(C_s))
# --- endcell: saltelli_matrices ---


# --- cell: model_evaluation_saltelli ---
# Step 2: evaluate the model for A, B, and each C_i (glass box)

def eval_model(X):
    """
    Evaluate the non-additive linear demo model using the convention
    X = [Z1..Z_Nrv, W1..W_Nrv] (so total P = 2*Nrv).
    """
    Z = X[:, :Nrv]
    W = X[:, Nrv:]
    return linear_model(W, Z)


# Evaluate A and B
f_A = eval_model(A_s)
f_B = eval_model(B_s)

# Evaluate C_i. We support two common storage conventions for C_s:
#   (i)  C_s shape = (P, Ns, P)  -> C_s[i] is matrix C_i of shape (Ns, P)
#   (ii) C_s shape = (Ns, P, P)  -> C_s[:, i, :] is matrix C_i of shape (Ns, P)
C_shape = np.shape(C_s)

if len(C_shape) != 3:
    raise ValueError(f"Expected C_s to be 3D, got shape {C_shape}")

if C_shape[0] == P and C_shape[1] == Ns_mc and C_shape[2] == P:
    # convention (i): (P, Ns, P)
    f_C = np.zeros((P, Ns_mc))
    for i in range(P):
        f_C[i, :] = eval_model(C_s[i])
elif C_shape[0] == Ns_mc and C_shape[1] == P and C_shape[2] == P:
    # convention (ii): (Ns, P, P)
    f_C = np.zeros((P, Ns_mc))
    for i in range(P):
        f_C[i, :] = eval_model(C_s[:, i, :])
else:
    raise ValueError(
        "Unrecognized C_s layout. "
        f"Got C_s shape {C_shape}, expected (P, Ns_mc, P) or (Ns_mc, P, P)."
    )

print("f_A:", np.shape(f_A))
print("f_B:", np.shape(f_B))
print("f_C:", np.shape(f_C))
# --- endcell: model_evaluation_saltelli ---


# --- cell: sobol_estimators ---
# Step 3: compute first-order (S) and total-order (ST) indices from model evaluations

from monte_carlo import calculate_sensitivity_indices_mc

Smc, Stmc = calculate_sensitivity_indices_mc(f_A, f_B, f_C)

labels = (
    [f"Z_{i}" for i in range(1, Nrv+1)] +
    [f"W_{i}" for i in range(1, Nrv+1)]
)

pretty_print_sobol_mc(Smc, Stmc, labels)


# --- endcell: sobol_estimators ---
