PACKAGE = emme
VRNPATN = '__version__ = "([^"]+)"'
VERSION = $(shell sed -nE 's:__version__ = "([^"]+)":\1:p' ./src/$(PACKAGE)/__init__.py)

PDOC = pdoc --template-dir ../elle/templates -o doc/

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
	#for item in doc/emme/**/*.html; do mv $$item $${item%.html}.md; done
	for item in doc/emme/*.html; do mv $$item $${item%.html}.md; done



publish:
	python setup.py clean --all sdist bdist_wheel
	twine upload --skip-existing dist/*

.PHONY: docs
