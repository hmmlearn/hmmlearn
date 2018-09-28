# simple makefile to simplify repetitive build env management tasks under posix

# caution: testing won't work on windows

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= py.test

all: clean inplace test

clean:
	$(PYTHON) setup.py clean
	rm -rf dist
	rm -rf build

inplace:
	$(PYTHON) setup.py build_ext -i

test: inplace
	$(PYTEST) -s -v --durations=10 --doctest-modules hmmlearn

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
