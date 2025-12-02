import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import VBox, HBox, Output


# ------------------------------------------------------------
# Compute slices and slice means
# ------------------------------------------------------------
def compute_slices(Z, Y, Nd):
    """
    Z : array shape (N,)
    Y : array shape (N,)
    Nd : number of slices
    """

    # Compute quantile-based slice boundaries
    qs = np.linspace(0, 1, Nd + 1)
    edges = np.quantile(Z, qs)

    slice_means = []
    for i in range(Nd):
        lo, hi = edges[i], edges[i+1]
        mask = (Z >= lo) & (Z <= hi)
        if np.sum(mask) > 0:
            slice_means.append((np.mean([lo, hi]), np.mean(Y[mask])))
        else:
            slice_means.append((np.mean([lo, hi]), np.nan))

    return edges, np.array(slice_means)


# ------------------------------------------------------------
# Plot ONE variable Zi as a scatter + slice means
# ------------------------------------------------------------
def plot_one_axis(Zi, Y, Nd, ax, label):
    """
    Draw scatter + equidistant slices + slice mean points.
    """

    # Scatter
    ax.scatter(Zi, Y, color="steelblue", alpha=0.35, s=12)

    # Compute slices + slice means
    edges, slice_means = compute_slices(Zi, Y, Nd)

    # Vertical lines
    for x in edges:
        ax.axvline(x, color="lightgray", linestyle="--", linewidth=1)

    # Slice mean line
    ax.plot(slice_means[:, 0], slice_means[:, 1], "-o",
            color="red", markersize=4, label="Slice mean")

    ax.set_xlabel(label)
    ax.set_ylabel("Y")
    ax.legend()


# ------------------------------------------------------------
# Main UI function
# ------------------------------------------------------------
def conditional_slices_interactive(zm, w, jpdf):
    """
    zm  : (Nrv, 2)   means and std dev
    w   : (Nrv,)     weights
    jpdf: chaospy joint distribution
    """

    Nrv = zm.shape[0]

    # Widgets
    Ns_slider = widgets.IntSlider(
        value=400, min=50, max=2000, step=50,
        description="Samples"
    )

    Nd_slider = widgets.IntSlider(
        value=10, min=2, max=25, step=1,
        description="N slices"
    )

    out = Output()

    # --------------------------------------------------------
    # Update function
    # --------------------------------------------------------
    def update(*args):
        with out:
            out.clear_output(wait=True)

            # Draw new samples
            Z = jpdf.sample(Ns_slider.value)         # shape (Nrv, Ns)
            Z = np.asarray(Z)
            Y = np.sum(w[:, None] * Z, axis=0)

            # Make the grid figure
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            axs = axs.flatten()

            for i in range(4):
                plot_one_axis(Z[i, :], Y, Nd_slider.value,
                              axs[i], f"Z{i+1}")

            fig.tight_layout()
            plt.show()

    # Trigger updates
    Ns_slider.observe(update, "value")
    Nd_slider.observe(update, "value")

    # Initial call
    update()

    # UI layout
    ui = VBox([
        HBox([Ns_slider, Nd_slider]),
        out
    ])
    return ui
