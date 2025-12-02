import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

import ipywidgets as widgets
from ipywidgets import VBox, HBox
from IPython.display import display, clear_output

# ----------------------------------------------------
#  Core plot function
# ----------------------------------------------------
def plot_slices_core(Z, Y, w, Ndz, show_model=True):
    """
    Compute per-Z_i slices and plot scatter of slice means + optional model line.
    Z: (N_samples, Nrv)
    Y: (N_samples,)
    w: (Nrv,)
    """
    N_samples, Nrv = Z.shape

    # Compute global y-limits with padding
    span = float(np.max(Y) - np.min(Y))
    pad = 0.05 * (span if span > 0 else 1.0)
    ymin = float(np.min(Y) - pad)
    ymax = float(np.max(Y) + pad)

    # Layout: 2 columns
    ncols = 2
    nrows = math.ceil(Nrv / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(10, 4.5 * nrows),
                             squeeze=False)

    slice_means = np.full((Nrv, Ndz), np.nan)

    for k in range(Nrv):
        ax = axes[k // ncols, k % ncols]

        # Sort by Z_k
        sidx = np.argsort(Z[:, k])
        Zk = Z[sidx, k]
        Yk = Y[sidx]

        zmin, zmax = np.min(Zk), np.max(Zk)
        if zmax == zmin:
            zmin -= 0.5
            zmax += 0.5

        # Slice boundaries
        Zb = np.linspace(zmin, zmax, Ndz + 1)
        Zmid = 0.5 * (Zb[:-1] + Zb[1:])

        # Vertical lines
        for edge in Zb:
            ax.axvline(edge, linestyle='--', color='.75')

        # Slice means
        for i in range(Ndz):
            mask = (Zk >= Zb[i]) & (Zk <= Zb[i+1] if i < Ndz - 1 else Zk <= Zb[i+1])
            if np.any(mask):
                slice_means[k, i] = np.mean(Yk[mask])

        ax.plot(Zmid, slice_means[k, :], '.', label="slice mean")

        # Model line (vary only factor k)
        if show_model:
            zvals = np.linspace(zmin, zmax, 30)
            Z_input = np.zeros((len(zvals), Nrv))
            Z_input[:, k] = zvals
            Ymodel = np.sum(w * Z_input, axis=1)
            ax.plot(zvals, Ymodel, label="linear model")

        ax.set_xlabel(f"Z{k+1}")
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    # Remove unused axes
    for idx in range(Nrv, nrows * ncols):
        fig.delaxes(axes[idx // ncols, idx % ncols])

    fig.tight_layout()
    return fig, slice_means


# ----------------------------------------------------
#  Interactive UI
# ----------------------------------------------------
def conditional_slices_interactive(zm, w, jpdf):

    # Precompute global quantities
    Nrv = len(zm)
    Z = jpdf.sample(2000)
    Z = np.asarray(Z)
    w = np.asarray(w)
    Y = np.sum(w[:,None] * Z, axis=0)

    # Widgets
    Ndz_slider = widgets.IntSlider(
        value=4, min=2, max=20, step=1, description="Slices:"
    )
    show_model_chk = widgets.Checkbox(
        value=True, description="Show model mean"
    )

    # --- IMPORTANT: create an Output widget ---
    out = widgets.Output()

    # Update function
    def update(change=None):
        with out:
            out.clear_output(wait=True)
            fig, df_spoor = plot_slices_core(Z, Y, w, Ndz_slider.value, show_model_chk.value)
            display(fig)
            display(df_spoor)

    # Trigger update on changes
    Ndz_slider.observe(update, names='value')
    show_model_chk.observe(update, names='value')

    # Initial plot
    update(None)

    return widgets.VBox([Ndz_slider, show_model_chk, out])



def conditional_slices_interactive(zm, w, jpdf):

    # Precompute global quantities
    Nrv = len(zm)
    Z = jpdf.sample(2000)
    Z = np.asarray(Z)
    w = np.asarray(w)
    Y = np.sum(w[:,None] * Z, axis=0)

    # Widgets
    Ndz_slider = widgets.IntSlider(
        value=4, min=2, max=20, step=1, description="Slices:"
    )
    show_model_chk = widgets.Checkbox(
        value=True, description="Show model mean"
    )

    # --- IMPORTANT: create an Output widget ---
    out = widgets.Output()

    # Update function
    def update(change=None):
        with out:
            out.clear_output(wait=True)
            fig, df_spoor = plot_slices_core(Z, Y, w, Ndz_slider.value, show_model_chk.value)
            display(fig)
            display(df_spoor)

    # Trigger update on changes
    Ndz_slider.observe(update, names='value')
    show_model_chk.observe(update, names='value')

    # Initial plot
    update(None)

    return widgets.VBox([Ndz_slider, show_model_chk, out])


