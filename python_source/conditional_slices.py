import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import VBox, HBox


# --------------------------------------------------------------
# Core plotting function (no interactivity)
# --------------------------------------------------------------
def plot_slices_core(Z, Y, w, Ndz, show_model=True):
    """
    Plots 4 scatterplots and vertical equidistant slice lines.
    Returns: figure, slice_midpoints (as DataFrame-like dict)
    """

    import pandas as pd

    Nrv = Z.shape[0]          # number of random variables
    Ns  = Z.shape[1]          # number of samples

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    axs = axs.flatten()

    slice_info = {}

    for i in range(Nrv):

        zi = Z[i, :]
        ax = axs[i]

        # ---- Equidistant slicing (simple and clean) ----
        zmin, zmax = np.min(zi), np.max(zi)
        edges = np.linspace(zmin, zmax, Ndz + 1)
        mids  = 0.5 * (edges[:-1] + edges[1:])

        slice_info[f"Z{i+1}"] = mids

        # ---- Scatter plot ----
        ax.scatter(zi, Y, alpha=0.5, s=12)

        # ---- Linear model line (optional) ----
        if show_model:
            Ymodel = np.mean(Y) + (zi - np.mean(zi)) * (w[i])
            ax.plot(zi, Ymodel, 'r-', lw=2)

        # ---- Draw vertical slice lines ----
        for e in edges:
            ax.axvline(e, color='gray', lw=1, ls='--', alpha=0.5)

        ax.set_xlabel(f"Z{i+1}")
        ax.set_ylabel("Y")

    fig.tight_layout()

    df = pd.DataFrame(slice_info)
    return fig, df


# --------------------------------------------------------------
# Interactive wrapper
# --------------------------------------------------------------
def conditional_slices_interactive(zm, w, jpdf):
    """
    Creates sliders:
      - Ndz: number of slices
      - Ns:  number of samples (also limits Ndz)
      - factor index (1..4)
    """

    import pandas as pd

    # Initial sampling
    Nrv = len(zm)
    Ns_default = 500

    Z = jpdf.sample(Ns_default)
    Z = np.asarray(Z)
    w = np.asarray(w)
    Y = np.sum(w[:, None] * Z, axis=0)

    # Widgets
    Ndz_slider = widgets.IntSlider(min=2, max=20, value=6,
                                  description="Slices:", continuous_update=False)
    Ns_slider = widgets.IntSlider(min=100, max=2000, value=Ns_default,
                                 description="Samples:", continuous_update=False)
    show_model_check = widgets.Checkbox(value=True, description="Show model line")

    out = widgets.Output()

    def update(_):
        with out:
            clear_output(wait=True)

            # Resample if Ns changed
            Ns = Ns_slider.value
            Znew = jpdf.sample(Ns)
            Znew = np.asarray(Znew)
            Ynew = np.sum(w[:, None] * Znew, axis=0)

            fig, df = plot_slices_core(Znew, Ynew, w,
                                       Ndz_slider.value,
                                       show_model_check.value)

            display(fig)
            display(df)

    # Connect widgets
    Ndz_slider.observe(update, "value")
    Ns_slider.observe(update, "value")
    show_model_check.observe(update, "value")

    # Initial update
    update(None)

    ui = VBox([
        HBox([Ndz_slider, Ns_slider, show_model_check]),
        out
    ])

    return ui
