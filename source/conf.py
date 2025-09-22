# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath("/exports/csce/datastore/geos/users/s1857688/Coarse_Graining/pysammos/"))  # adjust if your package is not one level up
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pysammos'
copyright = '2025, Claudia Elijas-Parra'
author = 'Claudia Elijas-Parra'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints", 
    "sphinx.ext.viewcode"
]


napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = []

language = 'english'

# Highlight style
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Mock heavy dependencies that cause import errors during doc build
autodoc_mock_imports = [
    "numba",
    "llvmlite",
    #"numpy",  # optional, if Numba triggers numpy import problems
]
