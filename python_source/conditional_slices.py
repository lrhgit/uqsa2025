import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

import ipywidgets as widgets
from ipywidgets import VBox, HBox, IntSlider, Dropdown, Checkbox, Output
from IPython.display import display, clear_output

# ====================================================================
# 1) Core plotting function (clean, no duplicated output)
# ====================================================================

def plot_slices_core(Z, Y, w, Ndz, show_model=True,
                     ymin=None, ymax=None):
    """
    Compute equal-width slices in Z_k, compute slice means,
    optionally overlay linear model, return (fig, df_spoor).
    """

    Nrv = Z.shape[0]

    # Determine y-axis limits if not given
    if ymin is None or ymax is None:
        span = float(np.max(Y) - np.min(Y))
        pad = 0.05 * (span if span > 0 else 1.0)
        ymin = float(np.min(Y) - pad)
        ymax = float(np.max(Y) + pad)

    # Layout: 2 columns, enough rows
    ncols = 2
    nrows = math.ceil(Nrv / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(10, 4.6 * nrows),
                             squeeze=False)

    YsliceMean = np.full((Nrv, Ndz), np.nan)

    for k in range(Nrv):
        ax = axes[k // ncols, k % ncols]

        # Sort by Z_k
        sidx = np.argsort(Z[k, :])
        Zsorted_k = Z[k, sidx]
        Ysorted = Y[sidx]

        # Slice boundaries
        zmin, zmax = float(np.min(Zsorted_k)), float(np.max(Zsorted_k))
        if zmax == zmin:     # avoid degeneracy
            zmin -= 0.5
            zmax += 0.5

        ZB = np.linspace(zmin, zmax, Ndz + 1)
        Zmid = 0.5 * (ZB[:-1] + ZB[1:])

        # Vertical lines
        for edge in ZB:
            ax.axvline(edge, linestyle='--', color='.75')

        # Slice means
        for i in range(Ndz):
            if i < Ndz - 1:
                mask = (Zsorted_k >= ZB[i]) & (Zsorted_k < ZB[i+1])
            else:
                mask = (Zsorted_k >= ZB[i]) & (Zsorted_k <= ZB[i+1])

            if np.any(mask):
                YsliceMean[k, i] = np.mean(Ysorted[mask])

        # Plot slice means
        ax.plot(Zmid, YsliceMean[k, :], 'o-', label="slice mean")

        # Linear model overlay (vary dimension k only)
        if show_model:
            zvals = np.linspace(zmin, zmax, 30)
            Zinp = np.zeros((len(zvals), Nrv))
            Zinp[:, k] = zvals
            Ymodel = np.sum(Zinp * w, axis=1)
            ax.plot(zvals, Ymodel, label="linear model")

        ax.set_xlabel(f"Z{k+1}")
        ax.set_ylim([ymin, ymax])
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    # Remove unused axes
    for idx in range(Nrv, nrows * ncols):
        fig.delaxes(axes[idx // ncols, idx % ncols])

    fig.tight_layout()

    # SpoorMan-like metric: var(E[Y|Zi]) / var(Y)
    VarY = np.var(Y)
    if VarY == 0:
        spoor = [np.nan] * Nrv
    else:
        spoor = [np.nanvar(YsliceMean[k, :]) / VarY for k in range(Nrv)]

    df_spoor = pd.DataFrame({"Ssl": np.round(spoor, 3)},
                            index=[f"Z{i+1}" for i in range(Nrv)])
    return fig, df_spoor


# ====================================================================
# 2) Sampling helper (method = "Chaospy" or "NumPy")
# ====================================================================

def sample_data(N, method, zm, jpdf, w):
    """
    Generate (Z, Y) using chosen sampling method.
    method ∈ {"Chaospy", "NumPy"}.
    Returns Z (Nrv x N) and Y (length N).
    """

    Nrv = zm.shape[0]

    if method == "Chaospy":
        # jpdf is a Joint distribution from chaospy
        Z = jpdf.sample(N)
        Y = np.sum(Z.T * w, axis=1)
        return Z, Y

    elif method == "NumPy":
        # Independent Gaussians using zm[:,1] as std deviations
        Z = np.array([
            np.random.normal(loc=0.0, scale=zm[i, 1], size=N)
            for i in range(Nrv)
        ])
        Y = np.sum(Z.T * w, axis=1)
        return Z, Y

    else:
        raise ValueError(f"Unknown method: {method}")


# ====================================================================
# 3) Interactive demo widget
# ====================================================================

def conditional_slices_interactive(zm, w, jpdf):
    """
    Interactive demo:
      - Slider for N (samples)
      - Slider for Ndz (slices)
      - Dropdown: sampling method
      - Dropdown: which rule to use for Ndz_max (N//20, sqrt(N), manual)
      - Checkbox for model overlay

    Displays:
      - slice figure
      - SpoorMan DataFrame
    """

    out = Output()

    # --- Widgets ---

    slider_N = IntSlider(
        value=500, min=50, max=4000, step=50,
        description="N", continuous_update=False
    )

    slider_Ndz = IntSlider(
        value=10, min=2, max=25, step=1,
        description="Ndz", continuous_update=False
    )

    dropdown_limit = Dropdown(
        options=[
            ("N//20 (stable)", "N20"),
            ("sqrt(N)", "sqrt"),
            ("manual", "manual")
        ],
        value="N20",
        description="Limit rule"
    )

    dropdown_method = Dropdown(
        options=["Chaospy", "NumPy"],
        value="Chaospy",
        description="Sample via"
    )

    cb_model = Checkbox(
        value=True,
        description="Show model"
    )

    # ==============================================================
    # Helper: compute Ndz_max according to rule
    # ==============================================================

    def compute_Ndz_max(N, rule):
        if rule == "N20":
            return max(2, N // 20)
        elif rule == "sqrt":
            return max(2, int(np.sqrt(N)))
        elif rule == "manual":
            return 200  # allow large values, manual override
        else:
            return 20

    # ==============================================================
    # Update function
    # ==============================================================

    def _update(change=None):
        with out:
            clear_output(wait=True)

            N = slider_N.value
            rule = dropdown_limit.value
            method = dropdown_method.value

            # Recompute Ndz_max and adjust slider
            Ndz_max = compute_Ndz_max(N, rule)
            if rule != "manual":
                slider_Ndz.max = Ndz_max
                if slider_Ndz.value > Ndz_max:
                    slider_Ndz.value = Ndz_max

            Ndz = slider_Ndz.value

            # --- Sampling ---
            Z, Y = sample_data(N, method, zm, jpdf, w)

            # --- Plot slices ---
            fig, df_spoor = plot_slices_core(
                Z, Y, w,
                Ndz=Ndz,
                show_model=cb_model.value
            )

            display(fig)
            display(df_spoor)

    # ==============================================================
    # Link widgets to update
    # ==============================================================

    slider_N.observe(_update, "value")
    slider_Ndz.observe(_update, "value")
    dropdown_limit.observe(_update, "value")
    dropdown_method.observe(_update, "value")
    cb_model.observe(_update, "value")

    # Initial call
    _update()

    # Layout UI
    ui = VBox([
        HBox([slider_N, slider_Ndz]),
        HBox([dropdown_limit, dropdown_method, cb_model]),
        out
    ])

    return ui

