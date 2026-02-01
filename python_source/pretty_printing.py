# python_source/pretty_printing.py

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from IPython.display import Math

# -----------------------------
# Pretty print helpers
# -----------------------------

def section_title(text: str, level: int = 3):
    """Nice section title as HTML in notebooks."""
    level = int(level)
    level = max(2, min(level, 6))
    return HTML(f"<h{level} style='margin-top:1.2em; font-weight:normal'>{text}</h{level}>")


def pretty_table(df: pd.DataFrame, floatfmt: str = None, decimals: int = None):
    if floatfmt is None and decimals is not None:
        floatfmt = f".{decimals}f"
    if floatfmt is None:
        floatfmt = ".3f"

    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].map(
                lambda x: f"{x:{floatfmt}}" if pd.notnull(x) else x
            )
    display(HTML(df2.to_html(escape=False)))




def pretty_print_sobol_mc(S, ST, labels=None, title="Monte Carlo Sobol indices", decimals=3):
    """Small helper for printing Sobol indices from MC in a compact table."""
    if labels is None:
        labels = [f"X{i+1}" for i in range(len(S))]
    df = pd.DataFrame(
        {"S (MC)": np.round(S, decimals), "ST (MC)": np.round(ST, decimals)},
        index=labels,
    )
    display(section_title(title))
    pretty_table(df)


def _poly_to_latex_rounded(poly_1d, var="q", decimals=3):
    """
    Convert a 1D numpoly/chaospy polynomial (in one variable) to LaTeX
    with rounded coefficients and reasonably compact formatting.
    """
    # poly_1d is typically something like 0.408*q0**3 - 1.225*q0 etc.
    # We'll evaluate coefficients by extracting terms from its string/latex
    # but numpoly has stable repr for simple 1D basis.
    # Easiest robust approach: use poly_1d.tostr() and parse lightly.
    s = str(poly_1d)

    # Replace q0 with desired variable name
    s = s.replace("q0", var)

    # Remove "*"
    s = s.replace("*", "")

    # Round float-like tokens
    import re
    def repl(m):
        val = float(m.group(0))
        return f"{val:.{decimals}f}"

    s = re.sub(r"(?<![A-Za-z0-9_])[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", repl, s)

    # Make powers LaTeX-ish: var**n -> var^{n}
    s = re.sub(rf"{re.escape(var)}\*\*(\d+)", rf"{var}^{{\1}}", s)

    # Clean up: "1.000q" -> "q"
    s = s.replace(f"1.{ '0'*decimals }{var}", var)

    return s


def show_poly_basis(poly, title="", var="q", decimals=3):
    """
    Display a chaospy/numpoly basis as LaTeX:
      φ_k(q) = ...
    with rounding.
    """
    if title:
        display(Math(rf"\text{{{title}}}"))
    for k, pk in enumerate(poly):
        latex = _poly_to_latex_rounded(pk, var=var, decimals=decimals)
        display(Math(rf"\phi_{{{k}}}({var}) = {latex}"))


import pandas as pd


def sobol_summary_table(sobol_mc, sobol_pce, model_name, labels):
    """
    Create a comparison table of Sobol indices (MC vs PCE)
    for a given model.
    """
    df = pd.DataFrame(
        {
            "S (MC)": sobol_mc[model_name]["S"],
            "ST (MC)": sobol_mc[model_name]["ST"],
            "S (PCE)": sobol_pce[model_name]["S"],
            "ST (PCE)": sobol_pce[model_name]["ST"],
        },
        index=labels,
    )
    return df


def uncertainty_summary_table(sobol_mc, sobol_pce):
    """
    Create a table summarizing mean and std (variance-weighted)
    for MC and PCE.
    """
    rows = []
    for name in sobol_mc.keys():
        rows.append(
            dict(
                Model=name,
                Method="MC",
                **{"E(Y)": sobol_mc[name]["E"], "Std(Y)": sobol_mc[name]["Std"]},
            )
        )
        rows.append(
            dict(
                Model=name,
                Method="PCE",
                **{"E(Y)": sobol_pce[name]["E"], "Std(Y)": sobol_pce[name]["Std"]},
            )
        )

    return pd.DataFrame(rows).set_index(["Model", "Method"])
