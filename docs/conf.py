#!/usr/bin/env python
#
# wrap_glmnet documentation build configuration file, created by sphinx-quickstart.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import wrap_glmnet

# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    #    'sphinx.ext.viewcode',  # create HTML file of source code and link to it
    "sphinx.ext.linkcode",  # link to github, see linkcode_resolve() below
    "recommonmark",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
# source_suffix = '.rst'
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document.
master_doc = "contents"

# General information about the project.
project = "wrap-glmnet"
copyright = '2024, <a href="https://www.maximzaslavsky.com">Maxim Zaslavsky</a>'
author = "Maxim Zaslavsky"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = wrap_glmnet.__version__
# The full version, including alpha/beta/rc tags.
release = wrap_glmnet.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "modules.rst"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# Resolve function for the linkcode extension.
# From https://github.com/Lasagne/Lasagne/blob/5d3c63cb315c50b1cbd27a6bc8664b406f34dd99/docs/conf.py#L113
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(wrap_glmnet.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "wrap_glmnet/%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"

    # TODO: attempt to read branch from CI config?
    tag = "master"  # if 'dev' in release else ('v' + release)

    return "https://github.com/maximz/wrap-glmnet/blob/%s/%s" % (tag, filename)


# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
# see https://alabaster.readthedocs.io/en/latest/customization.html
#
# see also requests docs conf, which uses alabaster theme too: https://github.com/psf/requests/blob/main/docs/conf.py
html_theme_options = {
    ## Alabaster options:
    # "show_powered_by": False,
    # "github_user": "maximz",
    # "github_repo": "wrap-glmnet",
    # "github_banner": True,
    # "github_type": "star",
    # "show_related": False,
    # "note_bg": "#FFF59C",
    ## Furo options:
    "top_of_page_button": None,
    # Since Furo doesn't allow us to disable dark mode, we make dark mode
    # equivalent to light mode by overriding all colors back to their light value.
    # From https://github.com/pradyunsg/furo/issues/28#issuecomment-1178947861
    "dark_css_variables": {
        # Taken from: https://github.com/pradyunsg/furo/blob/c682d5d3502f3fa713c909eebbf9f3afa0f469d9/src/furo/assets/styles/variables/_colors.scss
        "color-problematic": "#b30000",
        # Base Colors
        "color-foreground-primary": "black",  # for main text and headings
        "color-foreground-secondary": "#5a5c63",  # for secondary text
        "color-foreground-muted": "#646776",  # for muted text
        "color-foreground-border": "#878787",  # for content borders
        "color-background-primary": "white",  # for content
        "color-background-secondary": "#f8f9fb",  # for navigation + ToC
        "color-background-hover": "#efeff4ff",  # for navigation-item hover
        "color-background-hover--transparent": "#efeff400",
        "color-background-border": "#eeebee",  # for UI borders
        "color-background-item": "#ccc",  # for "background" items (eg: copybutton)
        # Announcements
        "color-announcement-background": "#000000dd",
        "color-announcement-text": "#eeebee",
        # Brand colors
        "color-brand-primary": "#2962ff",
        "color-brand-content": "#2a5adf",
        # Highlighted text (search)
        "color-highlighted-background": "#ddeeff",
        # GUI Labels
        "color-guilabel-background": "#ddeeff80",
        "color-guilabel-border": "#bedaf580",
        # API documentation
        "color-api-keyword": "var(--color-foreground-secondary)",
        "color-highlight-on-target": "#ffffcc",
        # Admonitions
        "color-admonition-background": "transparent",
        # Cards
        "color-card-border": "var(--color-background-secondary)",
        "color-card-background": "transparent",
        "color-card-marginals-background": "var(--color-background-hover)",
        # Code blocks
        "color-code-foreground": "black",
        "color-code-background": "#f8f9fb",
    },
}
# Force pygments style in dark mode back to the light variant
pygments_dark_style = "tango"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# html_extra_path = ["tests/baseline"]

# These paths are either relative to html_static_path or fully qualified paths (eg. https://...)
html_css_files = [
    "custom_furo.css",
]

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "wrap_glmnetdoc"


# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "wrap_glmnet.tex",
        "wrap-glmnet Documentation",
        author,
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        master_doc,
        "wrap_glmnet",
        "wrap-glmnet Documentation",
        [author],
        1,
    )
]

# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "wrap_glmnet",
        "wrap-glmnet Documentation",
        author,
        "wrap_glmnet",
        "Wrap Glmnet",
        "Miscellaneous",
    ),
]
