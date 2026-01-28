import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt

# imports for gpc
import math
# end imports for gpc



# === The useful help function ===
# show help for uniform distributions
cp.Uniform?

# show help for sample generation
cp.Distribution.sample?
# end help

# === Distributions ===
# simple distributions
rv1 = cp.Uniform(0, 1)
rv2 = cp.Normal(0, 1)
rv3 = cp.LogNormal(0, 1, 0.2, 0.8)
print(rv1, rv2, rv3)
# end simple distributions

# joint distributions
joint_distribution = cp.J(rv1, rv2, rv3)
print(joint_distribution)
# end joint distributions

# creating iid variables
X = cp.Normal()
Y = cp.Iid(X, 4)
print(Y)
# end creating iid variables

# === Sampling ===
# sampling in chaospy
u = cp.Uniform(0,1)
u.sample?
# end sampling chaospy

# example sampling with plots

import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

u1 = cp.Uniform(0, 1)
u2 = cp.Uniform(0, 1)
joint = cp.J(u1, u2)

def plot_sampling(N=200):
    s_r = joint.sample(size=N, rule="random")
    s_h = joint.sample(size=N, rule="hammersley")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True, constrained_layout=True)

    ax[0].scatter(*s_r, s=10)
    ax[0].set_title(f"Random (N={N})")
    ax[0].set_xlabel("Uniform 1"); ax[0].set_ylabel("Uniform 2")
    ax[0].set_aspect("equal", adjustable="box")

    ax[1].scatter(*s_h, s=10)
    ax[1].set_title(f"Hammersley (N={N})")
    ax[1].set_xlabel("Uniform 1")
    ax[1].set_aspect("equal", adjustable="box")

    plt.show()
    plt.close(fig)

widgets.interact(
    plot_sampling,
    N=widgets.IntSlider(value=200, min=20, max=1000, step=20, description="N", continuous_update=False)
);

# end example sampling with plots

# example save samples to file
# Creates a csv file where each row corresponds to the sample number and each column with teh variables in the joint distribution
csv_file = "csv_samples.csv"
sep = '\t'
header = ["u1", "u2"]
header = sep.join(header)
np.savetxt(csv_file, samples_random, delimiter=sep, header=header)
# end example save samples to file

# generate external data
ext_data = np.array([sample[0] + sample[1] + sample[0]*sample[1] for sample in samples_random.T])
header = ['y0']
header = sep.join(header)
filepath = "external_evaluations.csv"
np.savetxt(filepath, ext_data, delimiter=sep, header=header)
# end generate external data

# example load samples from file
# loads a csv file where the samples/or model evaluations for each sample are saved
# with one sample per row. Multiple components ofoutput can be stored as separate columns 
filepath = "external_evaluations.csv"
data = np.loadtxt(filepath)
# end example load samples from file

# === quadrature ===
# quadrature in polychaos
cp.generate_quadrature?
# end quadrature in polychaos

# example quadrature
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

order = 5


nodes_gaussian, weights_gaussian = cp.generate_quadrature(
    order=order,
    dist=joint_distribution,
    rule="G",
)

nodes_clenshaw, weights_clenshaw = cp.generate_quadrature(
    order=order,
    dist=joint_distribution,
    rule="C",
)


print('Number of nodes gaussian quadrature: {}'.format(len(nodes_gaussian[0])))
print('Number of nodes clenshaw-curtis quadrature: {}'.format(len(nodes_clenshaw[1])))


fig1, ax1 = plt.subplots()
ax1.scatter(*nodes_gaussian, marker='o', color='b')
ax1.scatter(*nodes_clenshaw, marker= 'x', color='r')
ax1.set_xlabel("Uniform 1")
ax1.set_ylabel("Uniform 2")
ax1.axis('equal')
# end example quadrature

# example sparse grid quadrature
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

order = 2
# sparse grid has exponential growth, thus a smaller order results in more points
nodes_clenshaw, weights_clenshaw = cp.generate_quadrature(
    order=order, dist=joint_distribution, rule="C"
)
nodes_clenshaw_sparse, weights_clenshaw_sparse = cp.generate_quadrature(
    order=order, dist=joint_distribution, rule="C", sparse=True
)


print('Number of nodes normal clenshaw-curtis quadrature: {}'.format(len(nodes_clenshaw[0])))
print('Number of nodes clenshaw-curtis quadrature with sparse grid : {}'.format(len(nodes_clenshaw_sparse[0])))

fig1, ax1 = plt.subplots()
ax1.scatter(*nodes_clenshaw, marker= 'x', color='r')
ax1.scatter(*nodes_clenshaw_sparse, marker= 'o', color='b')
ax1.set_xlabel("Uniform 1")
ax1.set_ylabel("Uniform 2")
ax1.axis('equal')
# end example sparse grid quadrature

# example orthogonalization schemes
from pretty_polynomial import poly_overview_cell
import chaospy as cp
import numpy as np

dist = cp.Normal(0, 1)
order = 3
x = np.linspace(-4, 4, 400)

poly_overview_cell(cp, dist, order, x=x, normed=True, var="q", decimals=3)
# end example orthogonalization schemes

# _Linear Regression_
# linear regression in chaospy
cp.fit_regression?
# end linear regression in chaospy


# example linear regression
# 1. define marginal and joint distributions
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

# 2. generate orthogonal polynomials
polynomial_order = 3
poly = cp.expansion.stieltjes(polynomial_order, joint_distribution)

# 3.1 generate samples

number_of_samples = math.comb(
    polynomial_order + len(joint_distribution),
    len(joint_distribution),
)
samples = joint_distribution.sample(size=number_of_samples, rule='R')

# 3.2 evaluate the simple model for all samples
model_evaluations = samples[0]+samples[1]*samples[0]

# 3.3 use regression to generate the polynomial chaos expansion
gpce_regression = cp.fit_regression(poly, samples, model_evaluations)
print("Success")
# end example linear regression


# _Spectral Projection_
# spectral projection in chaospy
cp.fit_quadrature?
# end spectral projection in chaospy


# example spectral projection
# 1. define marginal and joint distributions
u1 = cp.Uniform(0,1)
u2 = cp.Uniform(0,1)
joint_distribution = cp.J(u1, u2)

# 2. generate orthogonal polynomials
polynomial_order = 3
poly = cp.expansion.stieltjes(polynomial_order, joint_distribution)

# 4.1 generate quadrature nodes and weights
order = 5
nodes, weights = cp.generate_quadrature(
    order=order,
    dist=joint_distribution,   # <-- was: domain=joint_distribution
    rule="G"                   # gaussian quadrature (can also use "C", "E", etc.)
)


# 4.2 evaluate the simple model for all nodes
model_evaluations = nodes[0]+nodes[1]*nodes[0]

# 4.3 use quadrature to generate the polynomial chaos expansion
gpce_quadrature = cp.fit_quadrature(poly, nodes, weights, model_evaluations)
print("Success")
# end example spectral projection

# example uq
# Expected value
exp_reg = cp.E(gpce_regression, joint_distribution)
exp_ps  = cp.E(gpce_quadrature,  joint_distribution)

# Standard deviation
std_reg = cp.Std(gpce_regression, joint_distribution)
std_ps  = cp.Std(gpce_quadrature,  joint_distribution)

# Prediction intervals (90%)
pred_reg = cp.Perc(gpce_regression, [5, 95], joint_distribution)
pred_ps  = cp.Perc(gpce_quadrature,  [5, 95], joint_distribution)

# Assemble table
df_stats = pd.DataFrame(
    {
        "E (regression)": [f"{exp_reg:.3f}"],
        "E (projection)": [f"{exp_ps:.3f}"],
        "Std (regression)": [f"{std_reg:.3f}"],
        "Std (projection)": [f"{std_ps:.3f}"],
        "PI 5–95% (regression)": [f"[{pred_reg[0]:.3f}, {pred_reg[1]:.3f}]"],
        "PI 5–95% (projection)": [f"[{pred_ps[0]:.3f}, {pred_ps[1]:.3f}]"],
    },
    index=["Y"]
)

display(section_title("Uncertainty statistics from PCE"))
pretty_table(df_stats)
# end example uq

# example sens
sensFirst_reg = cp.Sens_m(gpce_regression, joint_distribution)
sensFirst_ps = cp.Sens_m(gpce_quadrature, joint_distribution)

sensT_reg = cp.Sens_t(gpce_regression, joint_distribution)
sensT_ps = cp.Sens_t(gpce_quadrature, joint_distribution)

# Assemble table
df_sens = pd.DataFrame(
    {
        "S (regression)": sensFirst_reg,
        "S (projection)": sensFirst_ps,
        "ST (regression)": sensT_reg,
        "ST (projection)": sensT_ps,
    },
    index=[f"X{i+1}" for i in range(len(sensFirst_reg))]
).round(3)

display(section_title("Sensitivity indices from PCE"))
pretty_table(df_sens)

# end example sens

# example exact solution
import sympy as sp
import sympy.stats
from sympy.utilities.lambdify import lambdify, implemented_function

pdf_beta = lambda b: 1
support_beta = (pdf_beta,0,1)
         
pdf_chi = lambda x:  1
support_chi = (pdf_chi,0, 1)
x, b = sp.symbols("x, b")
y = x + x*b

support_beta = (b,0,1)
support_chi = (x,0,1)
mean_g_beta = sp.Integral(y*pdf_chi(x), support_chi)
mean_g_chi =  sp.Integral(y*pdf_beta(b), support_beta)
mean = sp.Integral(mean_g_beta*pdf_beta(b), support_beta)
print("Expected value {}".format(mean.doit()))
variance = sp.Integral(pdf_beta(b)*sp.Integral(pdf_chi(x)*(y-mean)**2,support_chi), support_beta)
print("Variance: {}".format(variance.doit()))
var_E_g_beta = sp.Integral(pdf_beta(b)*(mean_g_beta-mean)**2, support_beta)
var_E_g_chi = sp.Integral(pdf_chi(x)*(mean_g_chi-mean)**2, support_chi)

S_chi =  var_E_g_chi/variance
S_beta = var_E_g_beta/variance


print("S_beta {}".format(S_beta.doit()))
print("S_chi {}".format(S_chi.doit()))
# end example exact solution
