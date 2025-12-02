import numpy as np, math, matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from matplotlib.figure import Figure
from ipywidgets import HBox, VBox, HTML

def _pie(N):
    vol = math.pi**(N/2) / math.gamma(N/2 + 1)
    cube = 2**N
    rem = max(cube - vol, 0)

    fig = Figure(figsize=(3.5,3.5))
    ax = fig.subplots()
    ax.pie([vol, rem], labels=["Hypersphere", "Remaining"],
           autopct='%1.1f%%', explode=[0.05,0], startangle=140)
    ax.set_title(f"Hypersphere–Hypercube ratio in {N}D")
    return fig

def hypercube_demo():
    slider = widgets.IntSlider(
        min=1, max=20, value=4,
        description='', continuous_update=False,
        layout=widgets.Layout(width='220px')
    )

    label = HTML("<b>Number of dimensions (N):</b>")

    out = widgets.Output()

    def _update(N):
        with out:
            out.clear_output(wait=True)
            display(_pie(N))

    slider.observe(lambda c: _update(c["new"]), names='value')
    _update(slider.value)


    ui = VBox([
        out,
        HBox(
            [label, slider],
            layout=widgets.Layout(
                justify_content='flex-start',    # left align
                align_items='center',
                width='100%',                    # take full available width
                padding='0px 20px'               # optional left padding
            )
        )
    ])


    
    display(ui)
    
