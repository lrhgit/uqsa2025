"""
Pretty visualization and printing of orthogonal polynomial bases.

Focused on:
- clear mathematical presentation
- robust plotting with chaospy/numpoly
- minimal notebook boilerplate
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML


# -------------------------------------------------
# helpers for mathematical pretty printing
# -------------------------------------------------

def _poly_to_latex(poly, k, var="q", decimals=3):
    """
    Convert the k-th polynomial in a chaospy expansion to LaTeX.
    """
    s = str(poly[k])
    s = s.replace("**", "^")
    s = s.replace("*", "")
    s = s.replace("q0", var)

    # round floating coefficients
    import re
    def _round(match):
        return f"{float(match.group()):.{decimals}f}"

    s = re.sub(r"-?\d+\.\d+", _round, s)
    return rf"\phi_{{{k}}}({var}) = {s}"


def _show_poly_basis(poly, title, var="q", decimals=3):
    """
    Display a polynomial basis as readable math.
    """
    display(HTML(f"<h4>{title}</h4>"))
    for k in range(len(poly)):
        latex = _poly_to_latex(poly, k, var=var, decimals=decimals)
        display(HTML(rf"<p>${latex}$</p>"))


# -------------------------------------------------
# main high-level cell function
# -------------------------------------------------

def poly_overview_cell(
    cp,
    dist,
    order,
    x=None,
    normed=True,
    var="q",
    decimals=3,
):
    """
    Plot and pretty-print orthogonal polynomial bases generated
    by different orthogonalization schemes in chaospy.
    """

    methods = [
        ("Cholesky decomposition", cp.expansion.cholesky),
        ("Discretized Stieltjes / three terms recursion", cp.expansion.stieltjes),
        ("Modified Gram–Schmidt", cp.expansion.gram_schmidt),
    ]

    if x is None:
        x = np.linspace(-4, 4, 400)

    x = np.asarray(x)
    x_eval = x[None, :]  # shape (1, N) is safe for 1D chaospy distributions

    fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharey=True)

    for ax, (title, method) in zip(axes, methods):

        poly = method(order, dist, normed=normed)

        # -------- plotting (robust) --------
        vals = np.asarray(poly(x_eval))

        # Typical shapes to handle:
        # (n_basis, N)          -> OK
        # (n_basis, 1, N)       -> squeeze -> (n_basis, N)
        # (N, n_basis)          -> transpose
        # (1, N)                -> one curve -> (1, N)
        # (N,)                  -> one curve -> (1, N)
        vals = np.squeeze(vals)

        if vals.ndim == 1:
            vals = vals[None, :]  # (N,) -> (1, N)
        elif vals.ndim == 2:
            # if first dimension matches N, assume it's (N, n_basis) and transpose
            if vals.shape[0] == x.size and vals.shape[1] != x.size:
                vals = vals.T
        else:
            raise ValueError(
                f"Unexpected poly(x) shape {vals.shape}; expected 1D/2D after squeeze."
            )

        if vals.shape[1] != x.size:
            raise ValueError(
                f"x and y mismatch after shaping: x={x.shape}, vals={vals.shape}"
            )

        for k in range(vals.shape[0]):
            ax.plot(x, vals[k, :], lw=1.8)

        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # -------- printing --------
        _show_poly_basis(
            poly,
            f"{title} (normed={normed})",
            var=var,
            decimals=decimals,
        )

    axes[0].set_ylabel("Polynomial value")
    plt.tight_layout()


    axes[0].set_ylabel("Polynomial value")
    plt.tight_layout()
    plt.show()


