# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import importlib

sys.path.insert(0, os.path.abspath(".."))

# mock dependencies to avoid installing them all just to build the configuration

from unittest.mock import Mock

MOCK_MODULES = [
    "tqdm",
    "tqdm.autonotebook",
    "yaml",
    "numpy",
    "PIL",
    "PIL.Image",
    "chainer",
    "chainer.iterators",
    "chainer.dataset",
    "pandas",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.gridspec",
    "tensorflow",
    "tensorflow.contrib.distributions",
    "tensorflow.contrib.framework.python.ops",
    "tensorflow.contrib.keras.api.keras",
    "tensorflow.contrib.keras.api.keras.models",
    "tensorflow.contrib.keras.api.keras.applications.vgg19",
    "tensorflow.python",
    "tensorflow.python.ops",
    "tensorflow.python.framework",
    "torch",
    "skimage",
    "skimage.measure",
    "tensorboard",
    "streamlit",
    "wandb",
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()


def run_apidoc(app):
    """Generate API documentation"""
    import better_apidoc

    better_apidoc.APP = app
    better_apidoc.main(
        [
            "better-apidoc",
            "-a",
            "-M",
            "-t",
            os.path.join(".", "templates"),
            "--force",
            "--no-toc",
            "--separate",
            "--ext-autodoc",
            "--ext-coverage",
            "-o",
            os.path.join(".", "source", "source_files/"),
            os.path.join("..", "edflow/"),
        ]
    )


def setup(app):
    app.connect("builder-inited", run_apidoc)


# -- Project information -----------------------------------------------------

project = "EDFlow"
copyright = "2019, Mimo Tilbich"
author = "Mimo Tilbich"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# html_static_path = ['_static']

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
}

exclude_patterns = ["_build"]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

## apidoc settings
# apidoc_module_dir = "../edflow"
# apidoc_output_dir = "source/source_files"
## apidoc_excluded_paths = ['tests']
## apidoc_separate_modules = True
