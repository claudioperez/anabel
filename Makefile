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
	python3 setup.py install
	#pip install -e .

api:
	$(PDOC) --config show_source_code=False \
	--config latex_math=True \
	--html --force \
	anabel.assemble \
	anabel.matrices \
	anabel.graphics \
	anabel.template \
	anabel.sections
	#rm $(DOCDIR)/$(PACKAGE)/index.html
	mv $(DOCDIR)/$(PACKAGE)/*.html $(DOCDIR)/api/latest/
	for item in $(DOCDIR)/api/latest/*.html; do mv $$item $${item%.html}.md; done

gallery:

publish:
	python setup.py clean --all sdist bdist_wheel
	twine upload --skip-existing dist/*
	git tag -a $(VERSION) -m 'version $(VERSION)'
	git push --tags

.PHONY: api

