# --- cell: gstar_statistics ---

# Gstar-statistics
# import modules

import numpy as np


def Vi(ai, alphai):
    return alphai**2 / ((1 + 2 * alphai) * (1 + ai) ** 2)


def V(a_prms, alpha):
    D = 1.0
    for ai, alphai in zip(a_prms, alpha):
        D *= (1.0 + Vi(ai, alphai))
    return D - 1.0


def S_i(a, alpha):
    S_i = np.zeros_like(a)
    Vtot = V(a, alpha)
    for i, (ai, alphai) in enumerate(zip(a, alpha)):
        S_i[i] = Vi(ai, alphai) / Vtot
    return S_i


def S_T(a, alpha):
    S_T = np.zeros_like(a)
    Vtot = V(a, alpha)
    for i, (ai, alphai) in enumerate(zip(a, alpha)):
        S_T[i] = (Vtot + 1.0) / (Vi(ai, alphai) + 1.0) * Vi(ai, alphai) / Vtot
    return S_T

# --- endcell: gstar_statistics ---


# --- cell: gstar_sliders_colab ---

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

plt.rcParams["figure.figsize"] = (7, 4)

out = widgets.Output()

def mk_sliders(prefix, n, v0, vmin, vmax, step, desc=None):
    if desc is None:
        desc = [f"{prefix}{i}" for i in range(1, n+1)]
    return [
        widgets.FloatSlider(
            value=v0[i-1] if isinstance(v0, (list, tuple)) else v0,
            min=vmin, max=vmax, step=step,
            description=desc[i-1],
            continuous_update=False
        )
        for i in range(1, n+1)
    ]

# Parameters used in G*
a = mk_sliders("a", 4, [0.75, 0.80, 0.20, 0.20], 0.0, 2.0, 0.05)
alpha = mk_sliders("alpha", 4, [0.75, 0.20, 0.20, 0.20],
                   0.0, 1.0, 0.05,
                   desc=["α1", "α2", "α3", "α4"])

# delta sliders kept for UI consistency (not used in G*)
delta = mk_sliders("delta", 4, [0.60, 0.50, 0.20, 0.20],
                   0.0, 1.0, 0.05,
                   desc=["δ1", "δ2", "δ3", "δ4"])

def redraw(*_):
    with out:
        clear_output(wait=True)

        aval = [s.value for s in a]
        alphaval = [s.value for s in alpha]

        Si = S_i(aval, alphaval)
        ST = S_T(aval, alphaval)

        fig, ax = plt.subplots()
        x = np.arange(1, len(Si) + 1)

        ax.bar(x - 0.15, Si, width=0.3, label="Sᵢ")
        ax.bar(x + 0.15, ST, width=0.3, label="Sᵀ")

        ax.set_xticks(x)
        ax.set_xticklabels([f"X{i}" for i in x])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Sensitivity index")
        ax.set_title("Sobol G* – sensitivities")
        ax.legend()

        plt.show()

for s in (*a, *alpha, *delta):
    s.observe(redraw, names="value")

ui = widgets.VBox([
    widgets.HBox(a),
    widgets.HBox(alpha),
    widgets.HBox(delta),
    out
])

display(ui)
redraw()

# --- endcell: gstar_sliders_colab ---
