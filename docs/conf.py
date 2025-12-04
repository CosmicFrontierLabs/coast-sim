# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "COASTSim"
copyright = "2025, Jamie A. Kennea"
author = "Jamie A. Kennea"

# The version info for the project you're documenting
try:
    from conops._version import __version__

    release = __version__
    version = ".".join(__version__.split(".")[:2])
except ImportError:
    release = "0.0.0"
    version = "0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "myst_parser",
]

# Enable autosummary to generate stub pages
autosummary_generate = False

# Napoleon settings for Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Logo
html_logo = "_static/coast-sim-logo.png"

# Theme options
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
}

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "imported-members": False,
    "inherited-members": False,
}


# Exclude imported names from documentation - these are imports from other modules
# that appear as module members but aren't part of that module's actual API
def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip imported members that aren't defined in the module."""
    # List of common imports to always exclude
    excluded_names = {
        # Pydantic imports
        "field_validator",
        "model_validator",
        "Field",
        "BaseModel",
        "ConfigDict",
        "PrivateAttr",
        "computed_field",
        "validator",
        # Typing imports
        "Any",
        "Dict",
        "List",
        "Optional",
        "Tuple",
        "Union",
        "Callable",
        "TypeVar",
        "Generic",
        "Literal",
        "ClassVar",
        "TYPE_CHECKING",
        # Dataclass imports
        "dataclass",
        "field",
        # Common stdlib imports
        "Path",
        "datetime",
        "timedelta",
        "Enum",
        "auto",
        # NumPy
        "np",
        "numpy",
        "ndarray",
    }
    if name in excluded_names:
        return True

    # Skip objects that are imported from other modules (not defined here)
    # This prevents showing e.g. MissionConfig under conops.ditl.ditl when it's
    # only imported there, not defined in that module.
    if what in ("class", "function", "exception"):
        try:
            obj_module = getattr(obj, "__module__", None)
            if obj_module:
                # Get the module we're documenting from the fully qualified name
                # The 'name' parameter is like 'MissionConfig' and we need parent
                parent = getattr(options, "objpath", [])
                if not parent:
                    # Try to get from __globals__ if it's a function
                    pass
                # Check the object's actual module
                if obj_module.startswith("conops."):
                    # Get documented module from env if available
                    env = getattr(app, "env", None)
                    if env:
                        docname = getattr(env, "docname", "")
                        if docname.startswith("api/conops."):
                            # Extract module name from docname like "api/conops.ditl.ditl"
                            documented_module = docname.replace("api/", "")
                            if obj_module != documented_module:
                                return True
        except Exception:
            pass

    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)


# Mock imports for packages that might not be available during doc build
autodoc_mock_imports: list = [
    "conops.battery",
    "conops.constraint",
    "conops.constants",
    "conops.emergency_charging",
    "conops.ephemeris",
    "conops.groundstation",
    "conops.passes",
    "conops.saa",
    "conops.slew",
    "conops.solar_panel",
    "conops.vector",
]

# -- MyST-Parser configuration -----------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# Suppress warnings from matplotlib docstrings and unknown roles
suppress_warnings = [
    "ref.python",
    "ref.doc",
    "ref.role",
]
