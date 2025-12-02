# ---------------------------------------------------------
# conditional_slices.py  — CLEAN, CONSISTENT, REFACTORED
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import VBox, HBox, IntSlider, Dropdown, Output
from IPython.display import display

# ---------------------------------------------------------
# Helper: compute slice means for a single Z_i
# ---------------------------------------------------------
def compute_slice_stats(zi, Y, N_slices):
    """
    zi: array of length Ns
    Y:  array of length Ns
    """
    # Equidistant slice boundaries
    zmin, zmax = zi.min(), zi.max()
    edges = np.linspace(zmin, zmax, N_slices + 1)

    slice_means = []
    slice_centers = []

    for k in range(N_slices):
        lo, hi = edges[k], edges[k + 1]
        mask = (zi >= lo) & (zi < hi)
        if np.sum(mask) > 0:
            slice_means.append(np.mean(Y[mask]))
            slice_centers.append(0.5 * (lo + hi))
        else:
            slice_means.append(np.nan)
            slice_centers.append(0.5 * (lo + hi))

    return np.array(slice_centers), np.array(slice_means), edges


# ---------------------------------------------------------
# Main interactive function
# ---------------------------------------------------------
def conditional_slices_interactive(zm, w, jpdf):
    """
    zm   : mean/std matrix (shape r × 2)
    w    : weights (length r)
    jpdf : Chaospy joint distribution
    """

    Nrv = zm.shape[0]

    # --- Widgets ---
    Ns_slider = IntSlider(
        value=500,
        min=50,
        max=5000,
        step=50,
        description="N samples"
    )

    Nslice_slider = IntSlider(
        value=6,
        min=2,
        max=20,
        step=1,
        description="N slices"
    )

    out = Output()

    # -----------------------------------------------------
    # Inner update function
    # -----------------------------------------------------
    def update(*args):

        with out:
            out.clear_output(wait=True)

            Ns = Ns_slider.value
            N_slices = Nslice_slider.value

            # --- Resample Z only when Ns changes ---
            Z = jpdf.sample(Ns)              # shape: (Nrv, Ns)
            Z = np.asarray(Z)
            Y = np.sum(w * Z, axis=1)        # correct shape: (Ns,)

            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            axs = axs.flatten()

            for i in range(Nrv):
                ax = axs[i]

                zi = Z[i, :]

                centers, slice_means, edges = compute_slice_stats(zi, Y, N_slices)

                # Scatter
                ax.scatter(zi, Y, s=4, alpha=0.4)

                # Vertical slice boundaries
                for x in edges:
                    ax.axvline(x, color='gray', linestyle='--', linewidth=1)

                # Slice mean points
                ax.scatter(centers, slice_means, s=40, color='red', zorder=4)

                ax.set_title(f"Z{i+1}")
                ax.set_xlabel("Z")
                ax.set_ylabel("Y")

            plt.tight_layout()
            display(fig)

    # -----------------------------------------------------
    # Wire up sliders
    # -----------------------------------------------------
    Ns_slider.observe(update, names="value")
    Nslice_slider.observe(update, names="value")

    # Initial draw
    update()

    return VBox([HBox([Ns_slider, Nslice_slider]), out])
