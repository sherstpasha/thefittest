# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= python3 -m sphinx
PYTHON        ?= python3
SOURCEDIR     = source
DOC_VERSION   ?= 0.3.0
DOC_VERSIONS  ?= $(DOC_VERSION),0.2.7
BUILDDIR      = $(CURDIR)/$(DOC_VERSION)
DOCTREEDIR    = $(CURDIR)/.doctrees/$(DOC_VERSION)
SPHINXENV     = DOC_VERSION="$(DOC_VERSION)" DOC_VERSIONS="$(DOC_VERSIONS)"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXENV) $(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(CURDIR)/_build" $(SPHINXOPTS) $(O)

.PHONY: help html versioned-html Makefile

html:
	@$(SPHINXENV) $(PYTHON) build.py $(SPHINXOPTS) $(O)

versioned-html: html

# Catch-all target: route all other targets to Sphinx using the new
# "make mode" option. $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXENV) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(CURDIR)/_build" $(SPHINXOPTS) $(O)
