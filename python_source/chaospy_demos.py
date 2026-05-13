# python_source/chaospy_demos.py

import chaospy as cp
import matplotlib.pyplot as plt
import ipywidgets as widgets


SAMPLING_RULES = [
    "random",
    "latin_hypercube",
    "halton",
    "hammersley",
    "sobol",
    "korobov",
    "additive_recursion",
]


GRID_RULES = [
    "grid",
    "nested_grid",
    "chebyshev",
    "nested_chebyshev",
]


def sampling_rule_demo(N_default=200):
    """
    Interactive comparison of Chaospy sampling schemes.
    """
    u1 = cp.Uniform(0, 1)
    u2 = cp.Uniform(0, 1)
    joint = cp.J(u1, u2)

    N_slider = widgets.IntSlider(
        value=N_default,
        min=20,
        max=1000,
        step=20,
        description="N",
        continuous_update=False,
        layout=widgets.Layout(width="45%"),
    )

    left_dropdown = widgets.Dropdown(
        options=SAMPLING_RULES,
        value="random",
        description="Left",
        layout=widgets.Layout(width="25%"),
    )

    right_dropdown = widgets.Dropdown(
        options=SAMPLING_RULES,
        value="hammersley",
        description="Right",
        layout=widgets.Layout(width="25%"),
    )

    def plot_sampling(N, rule_left, rule_right):
        s_left = joint.sample(size=N, rule=rule_left)
        s_right = joint.sample(size=N, rule=rule_right)

        fig, ax = plt.subplots(
            1, 2,
            figsize=(10, 4),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        for axis, samples, rule in zip(
            ax,
            [s_left, s_right],
            [rule_left, rule_right],
        ):
            axis.scatter(*samples, s=10)
            axis.set_title(f"{rule} (N={N})")
            axis.set_xlabel("Uniform 1")
            axis.set_aspect("equal", adjustable="box")

        ax[0].set_ylabel("Uniform 2")

        plt.show()
        plt.close(fig)

    controls = widgets.HBox(
        [N_slider, left_dropdown, right_dropdown],
        layout=widgets.Layout(
            width="100%",
            justify_content="space-between",
            align_items="center",
        ),
    )

    output = widgets.interactive_output(
        plot_sampling,
        {
            "N": N_slider,
            "rule_left": left_dropdown,
            "rule_right": right_dropdown,
        },
    )

    return widgets.VBox([controls, output])


def orthogonality_demo(order=4):
    import chaospy as cp
    import numpy as np
    import pandas as pd
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    distributions = {
        "Uniform(-1, 1)": cp.Uniform(-1, 1),
        "Gaussian N(0, 1)": cp.Normal(0, 1),
        "Gamma(2, 1)": cp.Gamma(2, 1),
        "Beta(2, 2)": cp.Beta(2, 2),
    }

    basis_distributions = {
        "Legendre-like basis": cp.Uniform(-1, 1),
        "Hermite-like basis": cp.Normal(0, 1),
        "Laguerre-like basis": cp.Gamma(2, 1),
        "Jacobi-like basis": cp.Beta(2, 2),
    }

    dist_dropdown = widgets.Dropdown(
        options=list(distributions.keys()),
        value="Uniform(-1, 1)",
        description="Distribution:",
        layout=widgets.Layout(width="45%"),
    )

    basis_dropdown = widgets.Dropdown(
        options=list(basis_distributions.keys()),
        value="Legendre-like basis",
        description="Basis:",
        layout=widgets.Layout(width="45%"),
    )

    output = widgets.Output()

    def update(change=None):
        with output:
            clear_output(wait=True)

            integration_dist = distributions[dist_dropdown.value]
            basis_dist = basis_distributions[basis_dropdown.value]

            poly = cp.expansion.stieltjes(order, basis_dist, normed=True)

            G = np.zeros((order + 1, order + 1))

            for i in range(order + 1):
                for j in range(order + 1):
                    G[i, j] = cp.E(poly[i] * poly[j], integration_dist)

            df = pd.DataFrame(
                np.round(G, 3),
                index=[f"Φ{i}" for i in range(order + 1)],
                columns=[f"Φ{j}" for j in range(order + 1)],
            )

            display(df)

            if np.allclose(G, np.diag(np.diag(G)), atol=1e-8):
                print("✓ This basis is orthogonal with respect to the selected distribution.")
            else:
                print("✗ This basis is not orthogonal with respect to the selected distribution.")

    dist_dropdown.observe(update, names="value")
    basis_dropdown.observe(update, names="value")

    update()

    controls = widgets.HBox(
        [dist_dropdown, basis_dropdown],
        layout=widgets.Layout(width="100%", justify_content="space-between"),
    )

    return widgets.VBox([controls, output])

