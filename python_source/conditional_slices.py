import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output


# -----------------------------------------------------------
# CORE plotting function (no widgets, no display)
# -----------------------------------------------------------
def plot_slices_core(Z, Y, w, Ndz, show_model=True):
    """
    Z  : (Nrv, Ns)   samples
    Y  : (Ns,)       output
    w  : (Nrv,)      weights
    Ndz: number of slices
    """
    Nrv, Ns = Z.shape

    # Compute slice boundaries
    zvals = np.linspace(0, 1, Ndz+1)

    # Output figure
    fig, axs = plt.subplots(1, Nrv, figsize=(4*Nrv, 4), sharey=True)

    if Nrv == 1:
        axs = [axs]

    # DataFrame to show slice means
    df_rows = []

    # Loop over each variable Z_i
    for i in range(Nrv):
        Zi = Z[i, :]
        ax = axs[i]

        # plot scatter
        ax.scatter(Zi, Y, s=8, alpha=0.3)
        ax.set_title(f"Z{i+1}")
        ax.set_xlabel(f"Z{i+1}")
        if i == 0:
            ax.set_ylabel("Y")

        # Compute slice means
        slice_means = []
        for k in range(Ndz):
            z_low = np.quantile(Zi, zvals[k])
            z_high = np.quantile(Zi, zvals[k+1])
            idx = (Zi >= z_low) & (Zi < z_high)
            Yslice = Y[idx]

            m = np.mean(Yslice) if len(Yslice) > 0 else np.nan
            slice_means.append(m)

            # vertical slice boundary markers
            ax.axvline(z_low, linestyle="--", color="gray", alpha=0.4)

        # final right boundary
        ax.axvline(np.quantile(Zi, zvals[-1]), linestyle="--", color="gray", alpha=0.4)

        # Optional model line
        if show_model:
            # Create model curve along Z_i, fixing others at mean
            Z_input = np.zeros((Ndz, Nrv))
            Z_input[:, i] = [np.quantile(Zi, zvals[k]) for k in range(Ndz)]
            Ymodel = np.sum(w * Z_input, axis=1)

            ax.plot([np.quantile(Zi, zvals[k]) for k in range(Ndz)],
                    Ymodel,
                    '-k', lw=2, label="Linear model")
            ax.legend()

        # Store row for dataframe
        df_rows.append(slice_means)

    # Build DataFrame with shape (Nrv, Ndz)
    df = pd.DataFrame(df_rows,
                      index=[f"Z{i+1}" for i in range(Nrv)],
                      columns=[f"Slice {k+1}" for k in range(Ndz)])

    return fig, df


# -----------------------------------------------------------
# INTERACTIVE WRAPPER (widgets)
# -----------------------------------------------------------
def conditional_slices_interactive(zm, w, jpdf):
    """
    zm   : array of shape (Nrv, 2)
    w    : array (Nrv,)
    jpdf : chaospy joint distribution
    """

    Nrv = len(zm)

    # --- Precompute sample data ---
    Ns = 2000
    Z = jpdf.sample(Ns)   # shape (Nrv, Ns)
    Z = np.asarray(Z)
    w = np.asarray(w)
    Y = np.sum(w[:, None] * Z, axis=0)    # shape (Ns,)

    # --- Widgets ---
    Ndz_slider = widgets.IntSlider(min=2, max=20, value=4, description="Slices:")
    show_model_chk = widgets.Checkbox(value=True, description="Show model curve")

    out = widgets.Output()
    table_out = widgets.Output()   # <--- To show table right BELOW sliders

    # update function
    def update(change=None):
        with out:
            clear_output(wait=True)
            fig, df = plot_slices_core(Z, Y, w, Ndz_slider.value, show_model_chk.value)
            display(fig)

        with table_out:
            clear_output(wait=True)
            display(df)

    # bind widget events
    Ndz_slider.observe(update, names="value")
    show_model_chk.observe(update, names="value")

    # initial draw
    update(None)

    # return layout: sliders stacked, table right below them, figure under that
    controls = widgets.VBox([
        Ndz_slider,
        show_model_chk,
        table_out
    ])

    ui = widgets.VBox([controls, out])
    return ui

