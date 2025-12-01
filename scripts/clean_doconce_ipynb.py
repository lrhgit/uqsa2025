import nbformat
import re
import sys

# Hvis du vil kunne kjøre på ulike filer: python clean_doconce_ipynb.py myfile.ipynb
source = sys.argv[1] if len(sys.argv) > 1 else "sensitivity_introduction_interactive.ipynb"
target = source.replace(".ipynb", "_clean.ipynb")

print(f"🔧 Renser {source} → {target}")

nb = nbformat.read(source, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "markdown":
        s = cell.source

        # Fjern DocOnce HTML-kommentarer, div, span og figure metadata
        s = re.sub(r'<!--.*?-->', '', s, flags=re.S)
        s = re.sub(r'<div.*?>|</div>', '', s)
        s = re.sub(r'<span.*?>|</span>', '', s)

        # Fjern tomme <p> og <a>-lenker fra DocOnce-formatet
        s = re.sub(r'<p>|</p>', '', s)
        s = re.sub(r'\[.*?\]\(#.*?\)', '', s)   # fjerner hyperlenker, men lar teksten stå

        # Rydd opp overflødige blanklinjer
        s = re.sub(r'\n{3,}', '\n\n', s)

        # Fjern eventuelle dupliserte ligningsblokker (samme latex flere ganger)
        s = re.sub(
            r'(\${2}.*?\${2})(?:\s*\1)+',
            r'\1',
            s,
            flags=re.S
        )

        s = re.sub(
            r'(\\begin\{equation\}.*?\\end\{equation\})(?:\s*\1)+',
            r'\1',
            s,
            flags=re.S
        )

        cell.source = s.strip()

nbformat.write(nb, target)
print(f"✅ Ferdig! Lagret som {target}")
