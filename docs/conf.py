# Configuration file for the Sphinx documentation builder.
#
# The package must be installed (e.g. `pip install -e .[docs]`) for autodoc
# to import it; no sys.path manipulation is used.

import importlib.metadata

project = "rspt2spectra"
copyright = "2019-2026, Johan Jönsson"  # noqa: A001 - Sphinx configuration name
author = "Johan Jönsson"

release = importlib.metadata.version("rspt2spectra")
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"

html_theme = "furo"
html_title = f"rspt2spectra {release}"
