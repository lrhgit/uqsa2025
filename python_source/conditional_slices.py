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
    """
    Builds a UI that:
      - Generates samples (Chaospy or Standard)
      - Computes Y
      - Calls plot_slices_core
      - Avoids duplicate plotting
    """
    
    out_table = widgets.Output()
    out_plot = widgets.Output()


    # Default values
    Nrv = len(w)
    default_N = 1000
    default_Ndz = min(20, default_N // 10)

    # Widgets
    sampling_dd = widgets.Dropdown(
        options=["Standard", "Chaospy"],
        value="Chaospy",
        description="Sampling:",
    )

    N_slider = widgets.IntSlider(
        min=100,
        max=3000,
        value=default_N,
        step=50,
        description="N",
        continuous_update=False,
        readout=True,
    )

    Ndz_slider = widgets.IntSlider(
        min=2,
        max=default_N // 10,
        value=default_Ndz,
        step=1,
        description="Slices:",
        continuous_update=False,
    )

    show_model_cb = widgets.Checkbox(
        value=True,
        description="Show model",
    )

    # ------------------------------------------------
    #  Update function (redraws plot)
    # ------------------------------------------------
    def update(*args):
        N = N_slider.value
        Ndz_slider.max = max(2, N // 10)  # automatic adjustment

        # Regenerate samples
        if sampling_dd.value == "Chaospy":
            Z = jpdf.sample(N).T  # Chaospy gives (Nrv, N); transpose → (N, Nrv)
        else:
            Z = np.zeros((N, Nrv))
            for i, (mu, sigma) in enumerate(zm):
                Z[:, i] = np.random.normal(mu, sigma, size=N)

        # Compute Y
        Y = np.sum(w * Z, axis=1)

        # Plot

        with out:
            clear_output(wait=True)
            fig, slice_means = plot_slices_core(
                Z, Y, w,
                Ndz=Ndz_slider.value,
                show_model=show_model_cb.value
            )
            display(fig)
            plt.close(fig)

            df = pd.DataFrame({
                f"Z{i+1}": [round(np.nanvar(slice_means[i]) / np.var(Y), 3)]
                for i in range(Nrv)
            })
            display(df)

    # Attach callbacks
    sampling_dd.observe(update, names="value")
    N_slider.observe(update, names="value")
    Ndz_slider.observe(update, names="value")
    show_model_cb.observe(update, names="value")

    # Initial draw
    update()

    controls = VBox([
        HBox([sampling_dd, show_model_cb]),
        HBox([N_slider, Ndz_slider]),
    ])


    return VBox([controls, out_table, out_plot])


