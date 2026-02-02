import math
import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from ipywidgets import HBox, VBox, HTML
from IPython.display import display


def _pie(N: int):
    vol = math.pi ** (N / 2) / math.gamma(N / 2 + 1)
    cube = 2 ** N
    rem = max(cube - vol, 0)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.pie(
        [vol, rem],
        labels=["Hypersphere", "Remaining"],
        autopct="%1.1f%%",
        explode=[0.05, 0],
        startangle=140,
    )
    ax.set_title(f"Hypersphere–Hypercube ratio in {N}D")
    return fig


def hypercube_demo():
    slider = widgets.IntSlider(
        min=1, max=20, value=4,
        description="",
        continuous_update=False,
        layout=widgets.Layout(width="220px"),
    )
    label = HTML("Number of dimensions (N):")
    out = widgets.Output()

    def _update(N: int):
        with out:
            out.clear_output(wait=True)
            fig = _pie(N)
            plt.show()
            plt.close(fig)  # avoids accumulating figures in some backends

    slider.observe(lambda c: _update(c["new"]), names="value")
    _update(slider.value)

    ui = VBox(
        [
            out,
            HBox(
                [label, slider],
                layout=widgets.Layout(
                    justify_content="flex-start",
                    align_items="center",
                    width="100%",
                    padding="0px 20px",
                ),
            ),
        ]
    )
    display(ui)
