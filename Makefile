# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= py.test

all: clean inplace test

clean:
	$(PYTHON) setup.py clean
	rm -rf dist

inplace:
	$(PYTHON) setup.py build_ext -i

test: inplace
	$(PYTEST) -s -v --doctest-modules hmmlearn

trailing-spaces:
	find hmmlearn -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

cython:
	find hmmlearn -name "*.pyx" | xargs $(CYTHON)

doc: inplace
	$(MAKE) -C doc html

doc-noplot: inplace
	$(MAKE) -C doc html-noplot

code-analysis:
	flake8 hmmlearn | grep -v __init__ | grep -v external
	pylint -E -i y hmmlearn/ -d E1103,E0611,E1101
