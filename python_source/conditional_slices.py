import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# ------------------------------------------------------------
#  Core computation: compute slice means for Zi vs Y
# ------------------------------------------------------------

def compute_slice_stats(Z, Y, Ndz):
    """
    Return:
        slice_centers: shape (Ndz,)
        slice_means:   shape (Ndz,)
    """

    zi = Z
    ymin = np.min(Y)
    ymax = np.max(Y)

    # Equidistant slice boundaries
    zmin, zmax = np.min(zi), np.max(zi)
    edges = np.linspace(zmin, zmax, Ndz+1)
    centers = 0.5*(edges[:-1] + edges[1:])

    slice_means = np.zeros(Ndz)
    for k in range(Ndz):
        mask = (zi >= edges[k]) & (zi < edges[k+1])
        if np.any(mask):
            slice_means[k] = np.mean(Y[mask])
        else:
            slice_means[k] = np.nan

    return centers, slice_means, edges

# ------------------------------------------------------------
#  Plotting for one variable Zi
# ------------------------------------------------------------

def plot_one_slice(Z, Y, i, Ndz):
    zi = Z[i, :]
    centers, slice_means, edges = compute_slice_stats(zi, Y, Ndz)

    fig, ax = plt.subplots(figsize=(6,3))

    # Raw scatter
    ax.scatter(zi, Y, s=12, alpha=0.25)

    # Vertical slice boundaries
    for e in edges:
        ax.axvline(e, color="gray", ls="--", lw=1, alpha=0.5)

    # Mean in each slice
    ax.plot(centers, slice_means, "ro-", lw=2, label="Slice mean")

    ax.set_xlabel(f"Z{i+1}")
    ax.set_ylabel("Y")
    ax.set_title(f"Conditional slices for Z{i+1}")
    ax.legend()

    fig.tight_layout()
    return fig

# ------------------------------------------------------------
#  Interactive UI
# ------------------------------------------------------------

def conditional_slices_interactive(zm, w, jpdf, Ns=500):
    """
    Compute samples, evaluate model and provide interactive slicing UI.
    """

    # Sampling
    Z = jpdf.sample(Ns)      # shape (Nrv, Ns)
    Y = np.sum(w[:,None] * Z, axis=0)

    Nrv = Z.shape[0]

    # Widgets
    slider_var = widgets.IntSlider(
        min=1, max=Nrv, value=1,
        description="Variable", continuous_update=False
    )

    slider_ndz = widgets.IntSlider(
        min=2, max=20, value=5,
        description="N slices", continuous_update=False
    )

    out = widgets.Output()

    # Update function
    def update(*args):
        with out:
            clear_output(wait=True)
            fig = plot_one_slice(Z, Y, slider_var.value-1, slider_ndz.value)
            display(fig)

    slider_var.observe(update, "value")
    slider_ndz.observe(update, "value")

    update()

    ui = widgets.VBox([
        widgets.HBox([slider_var, slider_ndz]),
        out
    ])
    return ui
