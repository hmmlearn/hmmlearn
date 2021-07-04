import hmmlearn

needs_sphinx = '2.0'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery',
]

project = 'hmmlearn'
copyright = '2010-present, hmmlearn developers (BSD License)'
version = release = hmmlearn.__version__
default_role = 'any'
pygments_style = 'sphinx'
language = 'en'

# -- Options for extensions --------------------------------------------------

autodoc_default_options = {
    'members': None,
    'inherited-members': None,
    'special-members': '__init__',
}

intersphinx_mapping = {
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

napoleon_use_ivar = True

sphinx_gallery_conf = {
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples'
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'github_url': 'https://github.com/hmmlearn/hmmlearn',
}
htmlhelp_basename = 'hmmlearn_doc'

# -- Options for LaTeX output ------------------------------------------------

latex_documents = [('index', 'user_guide.tex', 'hmmlearn user guide',
                    'hmmlearn developers', 'manual'), ]
