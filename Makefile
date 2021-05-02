PACKAGE = emme
VRNPATN = '__version__ = "([^"]+)"'
VERSION = $(shell sed -nE 's:__version__ = "([^"]+)":\1:p' ./src/$(PACKAGE)/__init__.py)
# Documentation
DOCDIR = docs
STYDIR = style

PDOC = pdoc --template-dir $(STYDIR) -o $(DOCDIR)

test:
	echo $(VERSION)

install:
	pip install -e .

docs:
	$(PDOC) --config show_source_code=False \
	--config latex_math=True \
	--html \
	--force \
	emme.objects emme.elements emme.matrices
	#rm $(DOCDIR)/$(PACKAGE)/index.html
	mv $(DOCDIR)/$(PACKAGE)/*.html $(DOCDIR)/api/
	for item in $(DOCDIR)/api/*.html; do mv $$item $${item%.html}.md; done


publish:
	python setup.py clean --all sdist bdist_wheel
	twine upload --skip-existing dist/*

.PHONY: docs
