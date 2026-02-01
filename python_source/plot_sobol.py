# python_source/plot_sobol.py
import numpy as np

def plot_sobol_bars(ax, Si, ST=None, labels=None, title="Sobol indices"):
    Si = np.asarray(Si, float).ravel()
    k = len(Si)
    x = np.arange(1, k + 1)
    if labels is None: labels = [f"X{i}" for i in x]
    w = 0.28 if ST is not None else 0.45
    ax.bar(x - (w/2 if ST is not None else 0), Si, width=w, label="Sᵢ")
    if ST is not None:
        ST = np.asarray(ST, float).ravel()
        ax.bar(x + w/2, ST, width=w, label="Sᵀ")
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1), ylabel="Sensitivity index", title=title)
    ax.legend()




# python_source/plot_sobol.py

import numpy as np
import matplotlib.pyplot as plt


def plot_sobol_mc_vs_pce(
    sobol_mc: dict,
    sobol_pce: dict,
    labels: list[str],
    col_order: list[str] | None = None,
    width: float = 0.35,
    figsize: tuple[float, float] = (9, 4.5),
    ylimit: tuple[float, float] = (0.0, 1.0),
    title_prefix: str = "",
):
    """
    Plot variance-weighted Sobol indices (S and ST) for MC (top row) vs PCE (bottom row),
    with models side-by-side in columns.

    Expected input:
        sobol_mc[name]["S"], sobol_mc[name]["ST"]  -> arrays of length len(labels)
        sobol_pce[name]["S"], sobol_pce[name]["ST"] -> arrays of length len(labels)

    Parameters
    ----------
    sobol_mc, sobol_pce : dict
        Dictionaries keyed by model name (e.g. "Quadratic model", "Logarithmic model").
    labels : list[str]
        Parameter labels shown on x-axis (must match length of S/ST arrays).
    col_order : list[str] | None
        Column ordering for models. Default: sorted intersection of keys.
    width : float
        Bar width.
    figsize : tuple
        Figure size.
    ylimit : tuple
        y-axis limits.
    title_prefix : str
        Optional prefix for subplot titles.
    """

    # Determine model columns
    if col_order is None:
        keys = sorted(set(sobol_mc.keys()).intersection(set(sobol_pce.keys())))
        col_order = keys

    N = len(labels)
    ind = np.arange(N)

    fig, axes = plt.subplots(2, len(col_order), figsize=figsize, sharey=True)

    # Handle the case len(col_order)==1 (axes becomes 1D)
    if len(col_order) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, name in enumerate(col_order):

        # --- MC (top row) ---
        ax = axes[0, col]
        ax.bar(ind,       sobol_mc[name]["S"],  width, alpha=0.5)
        ax.bar(ind+width, sobol_mc[name]["ST"], width, hatch=".")
        ax.set_title(f"{title_prefix}{name}" if title_prefix else name)
        ax.set_xticks(ind + width/2)
        ax.set_xticklabels(labels)
        ax.set_ylim(*ylimit)

        # --- PCE (bottom row) ---
        ax = axes[1, col]
        ax.bar(ind,       sobol_pce[name]["S"],  width, alpha=0.5)
        ax.bar(ind+width, sobol_pce[name]["ST"], width, hatch=".")
        ax.set_xticks(ind + width/2)
        ax.set_xticklabels(labels)
        ax.set_ylim(*ylimit)

    axes[0, 0].set_ylabel("MC: Sobol index")
    axes[1, 0].set_ylabel("PCE: Sobol index")

    # Legend once
    axes[0, -1].legend(["S", "ST"], frameon=False, loc="upper right")

    # Clean style
    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, axes



