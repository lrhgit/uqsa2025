import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, HBox, VBox, Output
from IPython.display import display, clear_output


def _compute_slices(Zi, Y, Nslices):
    """Return slice positions and slice means for one Z_i."""
    # Determine slice boundaries (equidistant)
    zmin, zmax = np.min(Zi), np.max(Zi)
    edges = np.linspace(zmin, zmax, Nslices + 1)

    slice_means = []
    slice_centers = 0.5 * (edges[:-1] + edges[1:])

    for a, b in zip(edges[:-1], edges[1:]):
        mask = (Zi >= a) & (Zi < b)
        if np.any(mask):
            slice_means.append(np.mean(Y[mask]))
        else:
            slice_means.append(np.nan)

    return slice_centers, slice_means, edges


def _plot_all(Z, Y, w, Nslices):
    """Create figure with 4 scatter plots + slice means."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for i in range(4):
        Zi = Z[i, :]
        ax = axs[i]

        ax.scatter(Zi, Y, s=8, alpha=0.4)

        xc, ym, edges = _compute_slices(Zi, Y, Nslices)
        ax.plot(xc, ym, "ro-", ms=6)

        for e in edges:
            ax.axvline(e, color="gray", lw=1, ls="--", alpha=0.6)

        ax.set_title(f"Z{i+1}")
        ax.set_xlabel(f"Z{i+1}")
        ax.set_ylabel("Y")

    fig.tight_layout()
    return fig


def conditional_slices_interactive(zm, w, jpdf):
    """Main interactive UI."""

    # Persistent output area—prevents multiple figures
    out = Output()

    # Sliders
    Ns_slider = IntSlider(
        value=400,
        min=50, max=3000, step=50,
        description="Samples Ns",
        continuous_update=False
    )

    slices_slider = IntSlider(
        value=5,
        min=2, max=20, step=1,
        description="Slices",
        continuous_update=False
    )

    # Internal state
    state = {"Z": None, "Y": None}

    def resample(Ns):
        """Sample new Z and Y only when Ns changes."""
        pdfs = [jpdf[i] for i in range(len(zm))]
        Z = jpdf.sample(Ns)
        Y = np.sum(w * Z, axis=0)

        state["Z"] = Z
        state["Y"] = Y

    def update(_=None):
        """Redraw full figure."""
        with out:
            clear_output(wait=True)

            Z = state["Z"]
            Y = state["Y"]
            Nslices = slices_slider.value

            fig = _plot_all(Z, Y, w, Nslices)
            display(fig)

    def update_Ns(change):
        resample(change["new"])
        update()

    # Connect
    Ns_slider.observe(update_Ns, names="value")
    slices_slider.observe(update, names="value")

    # Initial sampling + plot
    resample(Ns_slider.value)
    update()

    ui = VBox([HBox([Ns_slider, slices_slider]), out])
    return ui
