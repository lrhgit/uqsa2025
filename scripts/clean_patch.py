import nbformat, re

src = "sensitivity_introduction_interactive_clean.ipynb"
dst = src.replace(".ipynb", "_nogold.ipynb")

nb = nbformat.read(src, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "markdown":
        s = cell.source
        # Fjern unødige $$ rundt \begin{equation}
        s = re.sub(r'\$\$\s*\\begin\{equation\}', r'\\begin{equation}', s)
        s = re.sub(r'\\end\{equation\}\s*\$\$', r'\\end{equation}', s)
        cell.source = s

nbformat.write(nb, dst)
print(f"✅ Lagret uten gule ligninger som {dst}")

