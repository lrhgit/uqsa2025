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
        description='', continuous_upda_
