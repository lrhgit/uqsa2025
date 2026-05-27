import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import FloatSlider, HBox, VBox, interactive_output


def scatter_demo(Z):
    """
    Interactive 4-slider + 2x2 scatterplot visualization.

    Preferred input:
        Z.shape == (Ns, 4)

    Backward compatible:
        Z.shape == (4, Ns)
    """
    Z = np.asarray(Z)

    if Z.shape[1] == 4:
        Z_plot = Z
    elif Z.shape[0] == 4:
        Z_plot = Z.T
    else:
        raise ValueError("scatter_demo expects Z with shape (Ns, 4) or (4, Ns).")

    Ns, Nrv = Z_plot.shape
    assert Nrv == 4, "This demo currently expects 4 variables."

    sliders = [
        FloatSlider(
            value=2.0,
            min=0.5,
            max=5.0,
            step=0.1,
            description=f"Ω{i+1}",
            continuous_update=False,
            readout_format=".1f",
        )
        for i in range(Nrv)
    ]

    def update(**weights):
        omega = np.array([weights[f"w{i}"] for i in range(Nrv)])
        Y = Z_plot @ omega

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()

        ymin, ymax = float(np.min(Y)), float(np.max(Y))
        dy = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
        ymin -= dy
        ymax += dy

        for i in range(Nrv):
            axs[i].scatter(Z_plot[:, i], Y, alpha=0.5)
            axs[i].set_xlabel(f"Z{i+1}")
            axs[i].set_ylabel("Y")
            axs[i].set_ylim([ymin, ymax])
            axs[i].set_title(f"Y vs Z{i+1}")
            axs[i].grid(True, alpha=0.3)

        fig.tight_layout()
        display(fig)
        plt.close(fig)

    control_dict = {f"w{i}": sliders[i] for i in range(Nrv)}
    out = interactive_output(update, control_dict)

    return VBox([HBox(sliders), out])
