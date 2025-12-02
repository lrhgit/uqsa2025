# ------------------------------------------------------------
#  conditional_slices.py
#  Full interactive conditional-variance slice demo
#  Clean output: no double figures, no intrusive prints
# ------------------------------------------------------------

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

import ipywidgets as widgets
from ipywidgets import VBox, HBox, Dropdown, IntSlider, Checkbox, Output
from IPython.display import display, clear_output

try:
    import chaospy as cp
    CHAOSPY_AVAILABLE = True
except Exception:
    CHAOSPY_AVAILABLE = False


# ============================================================
#  Core slice-plotting function
#  ------------------------------------------------------------
#  - No display() and no plt.show() here!
#  - Returns (fig, df_spoor)
# ============================================================

def plot_slices_core(Z, Y, w, Ndz, show_model=True, ymin=None, ymax=None):
    """
    Core logic:
      - compute equal-width slices for each Z_i
      - compute slice means
      - return figure + dataframe with conditional-variance metric
    """

    Nrv = Z.shape[0]

    # y-limits
    if ymin is None or ymax is None:
        span = float(np.max(Y) - np.min(Y))
        pad  = 0.05 * (span if span > 0 else 1.0)
        ymin = float(np.min(Y) - pad)
        ymax = float(np.max(Y) + pad)

    # subplot grid
    ncols = 2
    nrows = math.ceil(Nrv / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.5 * nrows), squeeze=False)

    # slice mean array
    YsliceMean = np.full((Nrv, Ndz), np.nan)

    for k in range(Nrv):
        ax = axes[k // ncols, k % ncols]

        sidx = np.argsort(Z[k, :])
        Zs   = Z[k, sidx]
        Ys   = Y[sidx]

        zmin, zmax = np.min(Zs), np.max(Zs)
        if zmax == zmin:
            zmin -= 0.5
            zmax += 0.5

        Zbndry = np.linspace(zmin, zmax, Ndz + 1)
        Zslice = 0.5 * (Zbndry[:-1] + Zbndry[1:])

        # vertical boundaries
        for z in Zbndry:
            ax.axvline(z, linestyle='--', color='.75')

        # compute slice means
        for i in range(Ndz):
            left, right = Zbndry[i], Zbndry[i+1]
            if i < Ndz - 1:
                mask = (Zs >= left) & (Zs < right)
            else:
                mask = (Zs >= left) & (Zs <= right)

            if np.any(mask):
                YsliceMean[k, i] = np.mean(Ys[mask])

        ax.plot(Zslice, YsliceMean[k, :], '.', label='slice mean')

        # linear model line
        if show_model:
            zvals = np.linspace(zmin, zmax, 3)
            Zin   = np.zeros((len(zvals), Nrv))
            Zin[:, k] = zvals
            Ymod  = np.sum(w * Zin, axis=1)
            ax.plot(zvals, Ymod, label='linear model')

        ax.set_xlabel(f"Z{k+1}")
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc='best')

    # remove unused axes
    for idx in range(Nrv, nrows*ncols):
        fig.delaxes(axes[idx // ncols, idx % ncols])

    fig.tight_layout()

    # conditional variance metric
    varY = np.var(Y) if np.var(Y) != 0 else np.nan
    spoor = [np.nanvar(YsliceMean[k, :]) / varY for k in range(Nrv)]
    df_spoor = pd.DataFrame({"Ssl": np.round(spoor, 3)},
                             index=[f"Ssl_{i+1}" for i in range(Nrv)])

    return fig, df_spoor


# ============================================================
#  Sampling function
# ============================================================

def sample_Z(jpdf, N, method):
    """N-sample generator using either numpy or chaospy."""
    if method == "Numpy":
        # Normal( mean, std )
        dims = len(jpdf)
        Z = np.zeros((dims, N))
        for i, pdf in enumerate(jpdf):
            mu, sd = float(pdf.mean()), float(pdf.std())
            Z[i, :] = np.random.normal(mu, sd, N)
        return Z

    # Chaospy
    return jpdf.sample(N)


# ============================================================
#  Interactive UI
# ============================================================

def conditional_slices_interactive(zm, w, jpdf):
    """
    Full interactive conditional-slice demo with:
        - Sampling selector
        - Slice rule selector (N/20, sqrt(N), Unlimited)
        - Sliders (N, Ndz)
        - Show model checkbox
    """
    out = Output()

    # ---- Widgets ----
    sampling = Dropdown(
        options=["Chaospy", "Numpy"],
        value="Chaospy",
        description="Sampling:"
    )

    slice_rule = Dropdown(
        options=["N/20", "sqrt(N)", "Unlimited"],
        value="N/20",
        description="Slices rule:"
    )

    slider_N = IntSlider(
        value=1000,
        min=100,
        max=20000,
        step=100,
        description="N",
        continuous_update=False
    )

    slider_Ndz = IntSlider(
        value=20,
        min=2,
        max=100,
        step=1,
        description="Ndz",
        continuous_update=False
    )

    show_model = Checkbox(
        value=True,
        description="Show model"
    )

    # ---- Rule logic ----
    def update_Ndz_max(*args):
        """Update max Ndz based on chosen rule + current N."""
        N = slider_N.value
        rule = slice_rule.value

        if rule == "N/20":
            Ndz_max = max(2, N // 20)
        elif rule == "sqrt(N)":
            Ndz_max = max(2, int(math.sqrt(N)))
        else:
            Ndz_max = max(2, N)  # unlimited-ish

        slider_Ndz.max = Ndz_max
        if slider_Ndz.value > Ndz_max:
            slider_Ndz.value = Ndz_max

    slider_N.observe(update_Ndz_max, names="value")
    slice_rule.observe(update_Ndz_max, names="value")
    update_Ndz_max()

    # ---- Main update ----
    def update_plot(*args):
        with out:
            clear_output(wait=True)

            # sample Z
            Z = sample_Z(jpdf, slider_N.value, sampling.value)

            # evaluate model
            Y = np.sum(w * Z, axis=0)

            # compute plot
            fig, df = plot_slices_core(Z, Y, w,
                                       slider_Ndz.value,
                                       show_model=show_model.value)

            display(fig)
            display(df)

    # bind triggers
    sampling.observe(update_plot, names="value")
    slice_rule.observe(update_plot, names="value")
    slider_N.observe(update_plot, names="value")
    slider_Ndz.observe(update_plot, names="value")
    show_model.observe(update_plot, names="value")

    # initial draw
    update_plot()

    # UI layout
    controls = VBox([
        HBox([sampling, slice_rule]),
        HBox([slider_N, slider_Ndz]),
        show_model
    ])

    return VBox([controls, out])
