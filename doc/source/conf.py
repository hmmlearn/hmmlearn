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

autodoc_default_options = {'members': None, 'inherited-members': None}

intersphinx_mapping = {
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

sphinx_gallery_conf = {
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples'
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_sidebars = {'**': ['about.html', 'navigation.html', 'localtoc.html']}
html_theme_options = {
    'description': 'Unsupervised learning and inference of Hidden Markov Models',
    'github_user': 'hmmlearn',
    'github_repo': 'hmmlearn',
    'github_banner': True,
    'github_button': False,
    'code_font_size': '80%',
}
htmlhelp_basename = 'hmmlearn_doc'

# -- Options for LaTeX output ------------------------------------------------

latex_documents = [('index', 'user_guide.tex', 'hmmlearn user guide',
                    'hmmlearn developers', 'manual'), ]
