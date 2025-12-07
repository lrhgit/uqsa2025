import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import VBox, HBox, Output
from linear_model import linear_model


def conditional_slices_interactive(zm, w, jpdf, Ns=500, n_slices=8):
    """
    Interactive conditional slices viewer.

    Shows:
      • Scatter plot of (Zi, Y)
      • Vertical slice boundaries
      • Conditional mean in each slice (red dots + line)
    Controls:
      • Ns slider (total samples)
      • n_slices slider (number of slices)
    """

    Nrv = zm.shape[0]

    # --- widgets ---
    Ns_slider = widgets.IntSlider(
        value=Ns, min=100, max=5000, step=100,
        description="Ns"
    )

    n_slices_slider = widgets.IntSlider(
        value=n_slices, min=3, max=25, step=1,
        description="Slices"
    )

    out = Output()

    # --- Update function ---
    def update(*_):
        Ns_val = Ns_slider.value
        n_val = n_slices_slider.value

        # sample fresh each time
        Z = jpdf.sample(Ns_val)        # shape (Nrv, Ns)
        Y = linear_model(w, Z.T)       # shape (Ns,)

        with out:
            out.clear_output(wait=True)
            
            rows, cols = 2, 2
            fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
            axes = axes.flatten()

            
            if Nrv == 1:
                axes = [axes]

            for i in range(Nrv):
                ax = axes[i]
                zi = Z[i, :]

                # scatter
                ax.scatter(zi, Y, s=8, alpha=0.35)

                # slice boundaries
                zmin, zmax = float(np.min(zi)), float(np.max(zi))
                grid = np.linspace(zmin, zmax, n_val + 1)

                for g in grid:
                    ax.axvline(g, color="gray", linestyle="--",
                               linewidth=0.8, alpha=0.6)

                # conditional means (one per slice)
                centers = []
                means = []
                for k in range(n_val):
                    mask = (zi >= grid[k]) & (zi < grid[k+1])
                    if np.any(mask):
                        centers.append(0.5 * (grid[k] + grid[k+1]))
                        means.append(np.mean(Y[mask]))

                if centers:
                    ax.plot(centers, means, "o-", color="red")

                ax.set_ylabel("Y")
                ax.set_xlabel(f"Z{i+1}")

            fig.tight_layout()
            plt.show()

    # connect sliders
    Ns_slider.observe(update, "value")
    n_slices_slider.observe(update, "value")

    # initial render
    update()


    # --- stable Colab layout ---
    controls = VBox(
        [Ns_slider, n_slices_slider],
        layout=widgets.Layout(
            min_height="80px",
            padding="5px"
        )
    )

  
    return VBox([controls, out])





