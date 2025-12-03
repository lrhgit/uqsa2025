import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import VBox, HBox
from linear_model import linear_model


def conditional_slices_interactive(zm, w, jpdf, slices=8, Ns=500):
    """
    Interactive conditional slices viewer.

    Parameters
    ----------
    zm : array (Nrv, 2)
        mean/std for each parameter
    w : array (Nrv,)
        weights
    jpdf : chaospy joint distribution
    slices : int
        number of vertical slices per RV
    Ns : int
        number of samples
    """

    Nrv = zm.shape[0]

    # --- sample Z and compute Y once ---
    Z = jpdf.sample(Ns)     # shape (Nrv, Ns)
    Y = linear_model(w, Z.T)

    # --- Create figure and axes once (IMPORTANT) ---
    fig, axes = plt.subplots(Nrv, 1, figsize=(6, 2.5*Nrv))
    if Nrv == 1:
        axes = [axes]

    # --- Create sliders (slice index per RV) ---
    sliders = []
    for i in range(Nrv):
        sliders.append(
            widgets.IntSlider(
                value=slices // 2,
                min=0,
                max=slices - 1,
                step=1,
                description=f"Z{i+1} slice"
            )
        )

    # --- define update function ---
    def update(*args):
        # For each RV
        for i in range(Nrv):
            ax = axes[i]
            ax.clear()

            zi = Z[i, :]  # samples for RV i
            ax.scatter(zi, Y, s=8, alpha=0.35)

            # vertical slice lines
            zmin, zmax = np.min(zi), np.max(zi)
            grid = np.linspace(zmin, zmax, slices+1)
            for g in grid:
                ax.axvline(g, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

            # choose active slice
            k = sliders[i].value
            mask = (zi >= grid[k]) & (zi < grid[k+1])
            if np.sum(mask) > 0:
                ym = np.mean(Y[mask])
                ax.scatter(
                    (grid[k] + grid[k+1]) / 2,
                    ym,
                    s=120,
                    color='red'
                )

            ax.set_ylabel("Y")
            ax.set_xlabel("Z")
            ax.set_title(f"Z{i+1}")

        fig.tight_layout()

    # --- connect sliders ---
    for s in sliders:
        s.observe(update, "value")

    # --- initial draw ---
    update()

    return VBox(sliders + [fig.canvas])
