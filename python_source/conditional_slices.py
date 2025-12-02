import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import chaospy as cp

from ipywidgets import (
    VBox, HBox, IntSlider, Dropdown, Checkbox, Output
)
from IPython.display import display, clear_output


# -------------------------------------------------------------
#  Core computation (no display)
# -------------------------------------------------------------

def plot_slices_core(Z, Y, w, Ndz, show_model=True,
                     ymin=None, ymax=None):
    """
    Returns (fig, df_spoor).
    Does NOT display the figure (wrapper does that).
    """

    Nrv = Z.shape[0]

    # y-limits auto
    if ymin is None or ymax is None:
        span = float(np.max(Y) - np.min(Y))
        pad  = 0.05 * (span if span > 0 else 1.0)
        ymin = float(np.min(Y) - pad)
        ymax = float(np.max(Y) + pad)

    # layout
    ncols = 2
    nrows = math.ceil(Nrv / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(10, 4.5 * nrows),
                             squeeze=False)

    YsliceMean = np.full((Nrv, Ndz), np.nan)

    # For each variable Z_k
    for k in range(Nrv):
        ax = axes[k // ncols, k % ncols]

        # sort samples
        sidx = np.argsort(Z[k, :])
        Zs = Z[k, sidx]
        Ys = Y[sidx]

        # slice boundaries
        zmin, zmax = np.min(Zs), np.max(Zs)
        if zmax == zmin:
            zmin -= 0.5
            zmax += 0.5

        ZB = np.linspace(zmin, zmax, Ndz + 1)
        Zmid = 0.5 * (ZB[:-1] + ZB[1:])

        # vertical boundaries
        for edge in ZB:
            ax.axvline(edge, linestyle='--', color='.75')

        # slice means
        for i in range(Ndz):
            if i < Ndz - 1:
                mask = (Zs >= ZB[i]) & (Zs <  ZB[i+1])
            else:
                mask = (Zs >= ZB[i]) & (Zs <= ZB[i+1])
            if np.any(mask):
                YsliceMean[k, i] = np.mean(Ys[mask])

        ax.plot(Zmid, YsliceMean[k, :], 'o-', label="slice mean")

        # optional linear model
        if show_model:
            zvals = np.linspace(zmin, zmax, 40)
            Ztmp = np.zeros((len(zvals), Nrv))
            Ztmp[:, k] = zvals
            Ymodel = np.sum(Ztmp * w, axis=1)
            ax.plot(zvals, Ymodel, label="linear model")

        ax.set_xlabel(f"Z{k+1}")
        ax.set_ylim([ymin, ymax])
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    # remove empty axes
    for idx in range(Nrv, nrows*ncols):
        fig.delaxes(axes[idx // ncols, idx % ncols])

    fig.tight_layout()

    # SpoorMan metric
    VarY = np.var(Y)
    spoor = [np.nanvar(YsliceMean[k, :]) / VarY for k in range(Nrv)]
    df_spoor = pd.DataFrame(
        {"Ssl": np.round(spoor, 3)},
        index=[f"Z{i+1}" for i in range(Nrv)]
    )

    return fig, df_spoor


# -------------------------------------------------------------
# Helper functions for sampling
# -------------------------------------------------------------

def sample_inputs(zm, jpdf, N, method):
    """
    Return Z (Nrv x N) and Y (N values)
    """
    w = np.ones(zm.shape[0]) * 2  # same as notebook

    if method == "Chaospy":
        Z = jpdf.sample(N)
        Y = np.sum(Z.T * w, axis=1)
        return Z, Y

    elif method == "NumPy":
        means = zm[:, 0]
        stds  = zm[:, 1]
        Z = np.random.normal(means[:, None], stds[:, None], size=(len(means), N))
        Y = np.sum(Z.T * w, axis=1)
        return Z, Y

    else:
        raise ValueError("Unknown sampling method")


# -------------------------------------------------------------
# Interactive UI
# -------------------------------------------------------------

def conditional_slices_interactive(zm, w, jpdf):

    out = Output()

    # Top controls
    ui_method = Dropdown(
        options=["Chaospy", "NumPy"],
        value="Chaospy",
        description="Sampling:"
    )

    ui_ndzrule = Dropdown(
        options=["N//20", "sqrt(N)", "Manual"],
        value="N//20",
        description="Slices rule:"
    )

    ui_showmodel = Checkbox(
        value=True,
        description="Show model"
    )

    # Sample size slider
    ui_N = IntSlider(
        value=500, min=50, max=5000, step=50,
        description="N"
    )

    # Ndz slider will be updated dynamically
    ui_Ndz = IntSlider(
        value=10, min=2, max=50,
        description="Ndz"
    )

    # ---------------------------------------------------------
    # Update logic
    # ---------------------------------------------------------

    def update(*args):
        with out:
            clear_output(wait=True)

            # 1. sample Z and Y
            Z, Y = sample_inputs(zm, jpdf, ui_N.value, ui_method.value)

            # 2. update Ndz.max depending on rule
            if ui_ndzrule.value == "N//20":
                ui_Ndz.max = max(2, ui_N.value // 20)

            elif ui_ndzrule.value == "sqrt(N)":
                ui_Ndz.max = max(2, int(np.sqrt(ui_N.value)))

            # else Manual: keep slider as-is

            # 3. ensure slider value fits new max
            if ui_Ndz.value > ui_Ndz.max:
                ui_Ndz.value = ui_Ndz.max

            # 4. compute figure
            fig, df_spoor = plot_slices_core(
                Z, Y, w,
                Ndz=ui_Ndz.value,
                show_model=ui_showmodel.value
            )

            display(fig)
            display(df_spoor)

    # ---------------------------------------------------------
    # Bind widget change events
    # ---------------------------------------------------------

    ui_method.observe(update, "value")
    ui_ndzrule.observe(update, "value")
    ui_showmodel.observe(update, "value")
    ui_N.observe(update, "value")
    ui_Ndz.observe(update, "value")

    # initial draw
    update()

    # layout group
    controls_top = VBox([ui_method, ui_ndzrule, ui_showmodel])
    controls_bottom = VBox([ui_N, ui_Ndz])

    ui = VBox([
        controls_top,
        controls_bottom,
        out
    ])

    return ui
