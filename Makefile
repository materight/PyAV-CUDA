LDFLAGS ?= ""
CFLAGS ?= "-O0"
PYTHON ?= python

.PHONY: default build dist publish clean

default: build

build:
	CFLAGS=$(CFLAGS) LDFLAGS=$(LDFLAGS) $(PYTHON) setup.py build_ext --inplace --debug

dist:
	- rm -rf dist
	- $(PYTHON) -m build --sdist

publish:
	$(PYTHON) -m twine upload dist/*

clean:
	- find avhardware -name '*.so' -delete
	- rm -rf build
	- rm -rf sandbox
