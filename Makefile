LDFLAGS ?= ""
CFLAGS ?= "-O0"
PYTHON ?= python

.PHONY: default build clean

default: build

build:
	CFLAGS=$(CFLAGS) LDFLAGS=$(LDFLAGS) $(PYTHON) setup.py build_ext --inplace --debug

clean:
	- find avhw -name '*.so' -delete
	- rm -rf build
	- rm -rf sandbox
