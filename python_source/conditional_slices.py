import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, Checkbox, HBox, VBox, Output
from IPython.display import display
import pandas as pd


# ============================================================
# Core plotting function (pure computation – no widgets)
# ============================================================

def plot_slices_core(Z, Y, w, Ndz, show_model=True):
    """
    Compute equidistant slices along each Zi axis and plot Y distributions.

    Z: array shape (Nrv, Ns)
    Y: array shape (Ns,)
    Ndz: number of slices
    """

    Nrv, Ns = Z.shape

    # --- Compute slice edges: equidistant in Zi ---
    slice_edges = np.zeros((Nrv, Ndz + 1))
    slice_centers = np.zeros((Nrv, Ndz))

    for i in range(Nrv):
        zmin, zmax = Z[i, :].min(), Z[i, :].max()
        slice_edges[i, :] = np.linspace(zmin, zmax, Ndz + 1)
        slice_centers[i, :] = 0.5 * (slice_edges[i, :-1] + slice_edges[i, 1:])

    # --- Compute slice means table ---
    df_vals = {}
    for i in range(Nrv):
        means = []
        for j in range(Ndz):
            idx = (Z[i, :] >= slice_edges[i, j]) & (Z[i, :] < slice_edges[i, j + 1])
            if np.sum(idx) > 0:
                means.append(np.mean(Y[idx]))
            else:
                means.append(np.nan)
        df_vals[f"Z{i+1}"] = means

    df = pd.DataFrame(df_vals).T
    df.columns = [f"Slice {k+1}" for k in range(Ndz)]

    # --- Plotting ---
    fig, axs = plt.subplots(1, Nrv, figsize=(4*Nrv, 5))

    if Nrv == 1:
        axs = [axs]

    for i in range(Nrv):
        ax = axs[i]
        ax.scatter(Z[i, :], Y, s=8, alpha=0.35)

        # vertical slice lines
        for edge in slice_edges[i, :]:
            ax.axvline(edge, color='gray', alpha=0.35, linestyle='--')

        # model curve: linear model Y = sum(w_j Z_j)
        if show_model:
            # compute E[Z_not_i] for each sample
            Z_not_i = np.delete(Z, i, axis=0)
            E_rest = np.sum(np.delete(w, i)[:, None] * Z_not_i, axis=0)
            # compute model prediction with Zi varying
            Yi_model = w[i] * Z[i, :] + E_rest
            # Sort for display
            order = np.argsort(Z[i, :])
            ax.plot(Z[i, :][order], Yi_model[order], color='black')

        ax.set_title(f"Z{i+1}")
        ax.set_xlabel(f"Z{i+1}")
        ax.set_ylabel("Y")

    fig.tight_layout()
    return fig, df


# ============================================================
# Interactive wrapper
# ============================================================

def conditional_slices_interactive(zm, w, jpdf):
    """
    Full widget interface for conditional slices.
    Returns a VBox UI object.
    """

    # -----------------------
    # Monte Carlo sampling
    # -----------------------
    Ns = 800  # fixed cost
    Z = jpdf.sample(Ns)
    if Z.ndim == 1:
        Z = Z.reshape(1, -1)
    Y = np.sum(w[:, None] * Z, axis=0)

    Nrv, Ns = Z.shape

    # Widgets
    s_slider = IntSlider(
        min=3, max=20, value=10,
        description="Slices:", continuous_update=False
    )
    chk_model = Checkbox(value=True, description="Show model curve")

    # Outputs
    out_table = Output()
    out_fig = Output()

    # -----------------------
    # Update routine
    # -----------------------
    def update(_=None):
        Ndz = s_slider.value
        show_model = chk_model.value

        # Compute plots and table
        fig, df = plot_slices_core(Z, Y, w, Ndz, show_model)

        # Render table
        with out_table:
            out_table.clear_output(wait=True)
            display(df.round(6))

        # Render figure
        with out_fig:
            out_fig.clear_output(wait=True)
            display(fig)

    # trigger once
    s_slider.observe(update, "value")
    chk_model.observe(update, "value")

    update()

    # -----------------------
    # Final layout
    # -----------------------
    ui = VBox([
        HBox([s_slider, chk_model]),
        out_table,       # table appears directly under slider – as you wanted
        out_fig
    ])
    return ui
