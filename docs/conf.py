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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import logging
import jinja2
import pandas_sphinx_theme
from sphinx.ext.autosummary import _import_by_name
logger = logging.getLogger(__name__)
# -- Project information -----------------------------------------------------

project = 'EMA'
copyright = '2020, Claudio Perez'
author = 'Claudio Perez'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    'sphinx.ext.napoleon',
    "sphinx.ext.todo",
    "nbsphinx",
    # "numpydoc", 
    'recommonmark',
    # 'sphinx_gallery.gen_gallery',
    ]
# sphinx_gallery_conf = {
#      'examples_dirs': '../examples',   # path to your example scripts
#      'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output

#     'filename_pattern': r'\d\d\d_',
#     'ignore_pattern': r'__init__\.py',
# }

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','**.ipynb_checkpoints']
try:
    import nbconvert
except ImportError:
    logger.warn("nbconvert not installed. Skipping notebooks.")
    exclude_patterns.append("**/*.ipynb")
else:
    try:
        nbconvert.utils.pandoc.get_pandoc_version()
    except nbconvert.utils.pandoc.PandocMissing:
        logger.warn("Pandoc not installed. Skipping notebooks.")
        exclude_patterns.append("**/*.ipynb")


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/claudioperezii/ema/tree/master/{}".format(filename)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# #
# html_theme_path = ["./_other/"]
# html_theme = 'pandas_sphinx_theme'
# html_theme_options = {
#     "external_links": [],
#     }

html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

html_theme_options = {
    'pytorch_project': 'docs',
    'canonical_url': 'https://pytorch.org/docs/stable/',
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}
# html_theme = 'bootstrap'
# html_theme_options = {
# #   'bootswatch_theme': "yeti",
#   'bootswatch_theme': "simplex",
#   'navbar_sidebarrel': "false",
#   'source_link_position': "footer",
# }

html_logo = "./_static/emtec-1.jpg"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']