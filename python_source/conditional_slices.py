import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import VBox, HBox, Output
from linear_model import linear_model


def conditional_slices_interactive(zm, w, jpdf, slices=8, Ns=500):
    """
    Interactive conditional slices viewer.
    """

    Nrv = zm.shape[0]

    # --- Sample once ---
    Z = jpdf.sample(Ns)
    Y = linear_model(w, Z.T)

    # --- Output widget for all plots ---
    out = Output()

    # --- Sliders ---
    sliders = []
    for i in range(Nrv):
        sliders.append(
            widgets.IntSlider(
                value=slices // 2,
                min=0,
                max=slices - 1,
                step=1,
                description=f"Z{i+1}"
            )
        )

    def update(*args):
        with out:
            out.clear_output(wait=True)

            fig, axes = plt.subplots(Nrv, 1, figsize=(6, 2.5*Nrv))
            if Nrv == 1:
                axes = [axes]

            for i in range(Nrv):
                ax = axes[i]
                zi = Z[i, :]
                ax.scatter(zi, Y, s=8, alpha=0.35)

                # Slice grid
                zmin, zmax = np.min(zi), np.max(zi)
                grid = np.linspace(zmin, zmax, slices + 1)
                for g in grid:
                    ax.axvline(g, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

                # Highlight slice
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

            plt.show()

    # Connect sliders
    for s in sliders:
        s.observe(update, "value")

    # Initial draw
    update()

    return VBox(sliders + [out])
