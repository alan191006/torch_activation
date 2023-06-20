import os
import numpy as np

project = "torch-activation"
copyright = "2023, Alan, H."
author = "Alan H."

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinxcontrib.jupyter",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "nbsphinx"
]

api_url = os.environ.get("API_DOC_URL", "/api/latest")
intersphinx_mapping = {"python": (api_url, "../build_api/html/objects.inv")}

templates_path = ["_templates"]
locale_dirs = ["locale/"]
gettext_compact = False

exclude_patterns = []
pygments_style = "sphinx"
todo_include_todos = True

html_theme = "pydata_sphinx_theme"
html_title = "Torch Activation"

jupyter_kernels = {
    "python3": {
        "kernelspec": {
            "display_name": "Python",
            "language": "python3",
            "name": "python3"
        },
        "file_extension": ".py"
    },
}

skip_blacklist = frozenset([
    "__weakref__", "__module__", "__doc__", "__abstractmethods__",
    "__hyperparam_spec__", "__hyperparam_trans_dict__", "__param_init_spec__"
])
skip_whitelist = frozenset()

def handle_skip(app, what, name, obj, skip, options):
    if name.startswith("_abc_") or name in skip_blacklist:
        return True
    if name.startswith("__") and name.endswith("__"):
        return False
    return skip

def setup(app):
    app.connect("autodoc-skip-member", handle_skip)

doctest_global_setup = """
np.random.seed(0)
import megengine as mge
np.set_printoptions(precision=4)
"""

# Customizations
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": True,
}

autosummary_generate = True

# Additional Sphinx configuration
html_static_path = ["_static"]

