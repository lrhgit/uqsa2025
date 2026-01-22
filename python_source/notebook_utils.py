from IPython.display import HTML

def pretty_table(df, decimals=3, width="100%"):
    """
    Pretty, Colab/Jupyter-friendly table for DataFrame display.
    """
    return (
        df.style
        .format(f"{{:.{decimals}f}}")
        .set_table_styles([
            {"selector": "th",
             "props": [("font-weight", "normal"),
                       ("text-align", "left"),
                       ("padding", "6px")]},
            {"selector": "td",
             "props": [("text-align", "center"),
                       ("padding", "6px")]},
            {"selector": "table",
             "props": [("width", width)]},
        ])
    )


def section_title(text, level=4):
    size = {3: "1.25em", 4: "1.15em", 5: "1.05em"}.get(level, "1.15em")
    return HTML(
        f"<h{level} style='margin-top:1em; margin-bottom:0.3em; "
        f"color:#2c3e50; font-weight:normal; font-size:{size};'>"
        f"{text}</h{level}>"
    )
