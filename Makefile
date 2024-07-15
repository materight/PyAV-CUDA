LDFLAGS ?= ""
CFLAGS ?= "-O0"
PYTHON ?= python

.PHONY: default build clean fate-suite lint test install build_nvcc

default: build

# build_nvcc: 
# 	mkdir -p build/libs && /usr/local/cuda/bin/nvcc -lib -Xcompiler -MD -O2 -o build/libs/nv12_to_rgb.lib include/libavcuda/nv12_to_rgb.cu
# 	# mkdir -p build/libs && /usr/local/cuda/bin/nvcc -shared -o build/libs/libnv12_to_rgb.so -Xcompiler -fPIC include/libavcuda/nv12_to_rgb.cu

build:
	CFLAGS=$(CFLAGS) LDFLAGS=$(LDFLAGS) $(PYTHON) setup.py build_ext --inplace --debug

clean:
	- find av_hw -name '*.so' -delete
	- rm -rf build
	- rm -rf sandbox
	- rm -rf src
