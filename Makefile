PACKAGE = anabel
VRNPATN = '__version__ = "([^"]+)"'
VERSION = $(shell sed -nE 's:__version__ = "([^"]+)":\1:p' ./src/$(PACKAGE)/__init__.py)
# Documentation
DOCDIR = docs
STYDIR = style
TESTDIR = ~/stdy

PDOC = pdoc --template-dir $(STYDIR) -o $(DOCDIR)

test:
	ln $(TESTDIR)/elle-0020/src/elle-0020.ipynb tests/elle-0020.ipynb

install:
	pip install -e .

api:
	$(PDOC) --config show_source_code=False \
	--config latex_math=True \
	--html \
	--force \
	anabel.assemblers anabel.elements anabel.matrices anabel.graphics
	#rm $(DOCDIR)/$(PACKAGE)/index.html
	mv $(DOCDIR)/$(PACKAGE)/*.html $(DOCDIR)/api/
	for item in $(DOCDIR)/api/*.html; do mv $$item $${item%.html}.md; done

gallery:

publish:
	python setup.py clean --all sdist bdist_wheel
	twine upload --skip-existing dist/*

.PHONY: api

