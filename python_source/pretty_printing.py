"""
Pretty-print helpers used across UQSA notebooks.
Keep notebooks clean and consistent.
"""

from IPython.display import HTML, display


def section_title(title: str):
    """Display a section title in notebook output."""
    return HTML(f"<h3>{title}</h3>")


def pretty_table(df):
    """Display a pandas DataFrame nicely in a notebook."""
    display(df)


def pretty_print_sobol_mc(S, ST, Nrv: int, digits: int = 3, title: str = "Monte Carlo Sobol indices"):
    """
    Pretty-print first-order (S) and total-order (ST) Sobol indices from MC.

    Assumes parameter order: [Z1..Z_Nrv, W1..W_Nrv]  (so P = 2*Nrv).
    """
    import numpy as np
    import pandas as pd

    S = np.asarray(S).ravel()
    ST = np.asarray(ST).ravel()

    P_expected = 2 * int(Nrv)
    if S.size != P_expected or ST.size != P_expected:
        raise ValueError(
            f"Expected S and ST to have length {P_expected} (=2*Nrv), "
            f"got len(S)={S.size}, len(ST)={ST.size}."
        )

    labels = [f"Z{i+1}" for i in range(Nrv)] + [f"W{i+1}" for i in range(Nrv)]

    df = pd.DataFrame(
        {"S (MC)": S, "ST (MC)": ST},
        index=labels,
    ).round(digits)

    display(section_title(title))
    pretty_table(df)


