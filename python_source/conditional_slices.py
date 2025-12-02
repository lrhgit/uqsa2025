import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import VBox, HBox
from IPython.display import display, clear_output


# ------------------------------------------------------------------
# Compute slice boundaries and slice means for a single variable Zi
# ------------------------------------------------------------------
def compute_slice_means(Zi, Y, Ndz):
    # equidistant slice edges
    edges = np.linspace(Zi.min(), Zi.max(), Ndz + 1)

    slice_centers = 0.5 * (edges[:-1] + edges[1:])
    slice_means = np.zeros(Ndz)

    for k in range(Ndz):
        mask = (Zi >= edges[k]) & (Zi < edges[k+1])
        if mask.sum() > 0:
            slice_means[k] = Y[mask].mean()
        else:
            slice_means[k] = np.nan

    return edges, slice_centers, slice_means


# ------------------------------------------------------------------
# Build one complete 2×2 grid of slice plots
# ------------------------------------------------------------------
def plot_all_slices(Z, Y, Ndz):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for i in range(4):
        row = i // 2
        col = i % 2
        ax = axs[row, col]

        Zi = Z[i, :]

        # Scatter points
        ax.scatter(Zi, Y, s=15, alpha=0.25)

        # Slices and slice means
        edges, centers, means = compute_slice_means(Zi, Y, Ndz)

        # vertical slice boundaries
        for e in edges:
            ax.axvline(e, color='gray', linestyle='--', alpha=0.25)

        # slice mean line
        ax.plot(centers, means, '-o', color='red', label="Slice mean")

        ax.set_xlabel(f"Z{i+1}")
        ax.set_ylabel("Y")
        ax.legend()

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# MAIN INTERACTIVE FUNCTION
# ------------------------------------------------------------------
def conditional_slices_interactive(zm, w, jpdf):
    out = widgets.Output()

    # sliders
    Ns_slider = widgets.IntSlider(
        value=200, min=50, max=2000, step=50,
        description="Samples", continuous_update=False
    )
    Ndz_slider = widgets.IntSlider(
        value=10, min=4, max=30, step=1,
        description="N slices", continuous_update=True
    )

    # --------------------------------------------------------------
    # update function
    # --------------------------------------------------------------
    def update(*args):
        with out:
            clear_output(wait=True)

            # (re)sample only when Ns changes
            Ns = Ns_slider.value
            Z = jpdf.sample(Ns)         # shape (4, Ns)
            Y = np.sum(w[:, None] * Z, axis=0)

            fig = plot_all_slices(Z, Y, Ndz_slider.value)
            display(fig)

    # Trigger update on both sliders
    Ns_slider.observe(update, 'value')
    Ndz_slider.observe(update, 'value')

    # initial plot
    update()

    return VBox([Ns_slider, Ndz_slider, out])
