import nbformat
import re
import sys

source = sys.argv[1] if len(sys.argv) > 1 else "sensitivity_introduction_interactive.ipynb"
target = source.replace(".ipynb", "_clean.ipynb")

print(f"🔧 Rensing av {source} → {target}")

nb = nbformat.read(source, as_version=4)

def remove_duplicates(text):
    # Fjern dupliserte equation-blokker med identisk innhold
    blocks = re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', text, flags=re.S)
    seen = set()
    for b in blocks:
        if b in seen:
            text = text.replace(b, "")
        else:
            seen.add(b)
    return text

for cell in nb.cells:
    if cell.cell_type == "markdown":
        s = cell.source

        # Fjern HTML/DocOnce-kommentarer og div
        s = re.sub(r'<!--.*?-->', '', s, flags=re.S)
        s = re.sub(r'<div.*?>|</div>', '', s)
        s = re.sub(r'<span.*?>|</span>', '', s)

        # Fjern tomme <p> og <a>-lenker fra DocOnce-formatet
        s = re.sub(r'<p>|</p>', '', s)
        s = re.sub(r'\[.*?\]\(#.*?\)', '', s)

        # Fjern dupliserte LaTeX-blokker
        s = remove_duplicates(s)

        # Rydd opp ubalanserte $$ (DocOnce kan duplisere inline)
        s = re.sub(r'\${3,}', '$$', s)

        # Rydd opp overflødige blanklinjer
        s = re.sub(r'\n{3,}', '\n\n', s)

        # Fjern whitespace på slutten
        s = s.strip()

        cell.source = s

nbformat.write(nb, target)
print(f"✅ Ferdig: {target}")
