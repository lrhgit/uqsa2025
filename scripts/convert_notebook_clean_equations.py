#!/usr/bin/env python3
import nbformat
import re
import sys
from pathlib import Path


def clean_equations_in_text(text: str) -> str:
    import re

    # 1. Remove \label{...}
    text = re.sub(r"\\label\{.*?\}", "", text)

    # 2. Remove DocOnce-style \begin{equation} ... \end{equation}
    text = re.sub(r"\\begin\{equation\}", "", text)
    text = re.sub(r"\\end\{equation\}", "", text)

    # 3. Remove empty or misplaced $$ before <p>, <img>, etc.
    text = re.sub(r"\$\$\s*\n\s*(?=<)", "", text)
    text = re.sub(r"(</?p>|</?img>)\s*\$\$", r"\1", text)


    # 4. Ensure lines containing \tag{n} are inside $$ ... $$
    text = re.sub(
        r"(?<!\$)\s*\n\s*([A-Za-z0-9\\].*?)\n\\tag\{(\d+)\}(?!\$)",
        r"\n$$\n\1\n\\tag{\2}\n$$",
        text,
        flags=re.DOTALL,
    )

    # 5. Collapse duplicate or empty $$ blocks
    text = re.sub(r"\$\$\s*\$\$", "$$", text)
    text = re.sub(r"\$\$\s*\n\s*\$\$", "$$", text)

    # 6. Remove $$ that wrap around HTML or are otherwise misplaced
    text = re.sub(r"\$\$\s*(<[^>]+>)", r"\1", text)

    # 7. Compact excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 8. Fjern tomlinjer etter åpning $$ og før lukking $$
    text = re.sub(r"\$\$\s*\n\s*\n", "$$\n", text)  # etter åpning
    text = re.sub(r"\n\s*\n\s*\$\$", "\n$$", text)  # før lukking

    # 9. Fjern tom linje rett før \tag{...}
    text = re.sub(r"\n\s*\n(?=\\tag\{)", "\n", text)

    return text


def clean_notebook_equations(path):
    nb = nbformat.read(path, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = clean_equations_in_text(cell.source)

    out_path = Path(path).with_name(Path(path).stem + "_clean.ipynb")
    nbformat.write(nb, out_path)
    print(f"✅ Cleaned notebook saved as: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_notebook_clean_equations.py notebook.ipynb")
        sys.exit(1)

    notebook_path = sys.argv[1]
    clean_notebook_equations(notebook_path)
