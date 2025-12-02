import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, HBox, VBox, interactive_output

def linear_model(w, z):
    return np.sum(w * z, axis=1)

def scatter_demo(Z):
    """
    Creates a clean 4-slider + 2x2 scatterplot dynamic visualization.
    Z must be shaped (4, Ns).
    """
    Nrv = Z.shape[0]
    assert Nrv == 4, "This demo currently expects 4 variables."

    # --- Sliders ---
    sliders = [
        FloatSlider(
            value=2, min=0.5, max=5, step=0.1,
            description=f"Ω{i+1}", continuous_update=False, readout_format=".1f"
        )
        for i in range(Nrv)
    ]

    # --- Figure setup ---
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    def update(**weights):
        w = np.array([weights[f"w{i}"] for i in range(Nrv)])
        Y = linear_model(w, Z.T)

        # shared y-limits
        ymin, ymax = float(np.min(Y)), float(np.max(Y))
        dy = 0.05 * (ymax - ymin)
        ymin -= dy
        ymax += dy

        for i in range(Nrv):
            axs[i].clear()
            axs[i].scatter(Z[i, :], Y, alpha=0.5)
            axs[i].set_xlabel(f"Z{i+1}")
            axs[i].set_ylabel("Y")
            axs[i].set_ylim([ymin, ymax])
            axs[i].set_title(f"Y vs Z{i+1}")
            axs[i].grid(True, alpha=0.3)

        fig.tight_layout()
        display(fig)
        plt.close(fig)

    # Bind sliders → update()
    control_dict = {f"w{i}": sliders[i] for i in range(Nrv)}
    out = interactive_output(update, control_dict)

    return VBox([HBox(sliders), out])
