# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Mock validate_data before any imports
def _mock_validate_data(estimator, X, y=None, reset=True, **kwargs):
    """Mock validate_data for documentation build."""
    if y is not None:
        return X, y
    return X

# Patch sklearn.utils.validation before importing thefittest
try:
    import sklearn.utils.validation
    if not hasattr(sklearn.utils.validation, 'validate_data'):
        sklearn.utils.validation.validate_data = _mock_validate_data
except ImportError:
    pass

path_list = os.getcwd().split("\\")
path = "\\".join(path_list[:-2]) + "\\src"
sys.path.insert(0, path)
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../../src"))
# -- Project information -----------------------------------------------------

project = "Thefittest"
copyright = "2023, Pavel Sherstnev"
author = "Pavel Sherstnev"

# The full version, including alpha/beta/rc tags
release = "0.2.3"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.doctest",  # Disabled - examples are for documentation only
    "numpydoc",
]

# Numpydoc configuration - disable doctest
numpydoc_show_class_members = False
numpydoc_use_plots = False

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "inherited-members": True,
    "show-inheritance": True,
}

# Suppress warnings during import
import warnings
warnings.filterwarnings('ignore')

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_rtd_theme"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Add meta tags to prevent browser caching
html_context = {"default_mode": "light"}

# Add cache-busting meta tags
html_meta = {
    "cache-control": "no-cache, no-store, must-revalidate",
    "pragma": "no-cache",
    "expires": "0",
}

# Path to the build directory
html_build_dir = os.path.join(os.path.dirname(__file__), "..", "docs")

# Output directory for the build html
html_output = os.path.join(html_build_dir, "html")

autosummary_generate = True
