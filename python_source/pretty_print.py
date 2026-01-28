# python_source/pretty_print.py

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from pretty_printing import show_poly_basis


def poly_overview_cell(cp, dist, order, x=None, normed=True, var="q", decimals=3):
    """
    One-stop helper for a notebook cell:
      - builds 3 orthogonal polynomial bases with different methods
      - prints them nicely as formulas (rounded)
      - plots them side-by-side
    """
    if x is None:
        x = np.linspace(-4, 4, 400)

    methods = [
        ("Cholesky decomposition", cp.expansion.cholesky),
        ("Discretized Stieltjes / three terms recursion", cp.expansion.stieltjes),
        ("Modified Gram–Schmidt", cp.expansion.gram_schmidt),
    ]

    display(Markdown("### Example: Orthogonalization schemes in Chaospy\n"
                     "Below we compare three common orthogonalization methods in `chaospy` for a standard normal distribution.\n"
                     "Increase the polynomial order to see that numerical instabilities may appear for some methods.\n"))

    # ---- pretty printed basis ----
    polys = []
    for title, method in methods:
        poly = method(order, dist, normed=normed)
        polys.append((title, poly))
        show_poly_basis(poly, title=f"{title} (normed={normed})", var=var, decimals=decimals)

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)

    for ax, (title, poly) in zip(axes, polys):
        y = poly(x).T  # shape: (order+1, len(x)) typically
        for k in range(y.shape[0]):
            ax.plot(x, y[k], label=f"$\\phi_{k}$")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(var)
        if ax is axes[0]:
            ax.set_ylabel("value")
        ax.legend(fontsize=8, ncol=2, frameon=False)

    plt.tight_layout()
    plt.show()
