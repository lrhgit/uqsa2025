# python_source/ipynb_header.py
from IPython.display import HTML, display

display(HTML(r"""
<script>
(function () {
  // Guard: don't redefine macros multiple times if the cell is re-run
  if (window.__UQSA_MATHJAX_MACROS_SET__) return;
  window.__UQSA_MATHJAX_MACROS_SET__ = true;

  // Ensure MathJax config objects exist
  window.MathJax = window.MathJax || {};
  window.MathJax.tex = window.MathJax.tex || {};
  window.MathJax.tex.macros = Object.assign({}, window.MathJax.tex.macros, {
    dd: "\\,\\mathrm{d}",
    EE: "\\mathbb{E}",
    VV: "\\mathbb{V}",
    Var: "\\operatorname{Var}",
    E: "\\operatorname{E}"
  });

  // Re-typeset after macros are set (JupyterLab + Colab)
  function typeset() {
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise();
    }
  }
  setTimeout(typeset, 0);
})();
</script>
"""))
