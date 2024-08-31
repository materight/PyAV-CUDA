import argparse
import os
import subprocess
import sys
from pathlib import Path

import av
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

CUDA_HOME = os.environ.get("CUDA_HOME", None)
NVCC_PATH = str(Path(CUDA_HOME) / "bin" / "nvcc") if CUDA_HOME else None
CUDA_ARCH = os.environ.get("CUDA_ARCH", "sm_75")

SKIP_LIBS_CHECKS = bool(int(os.environ.get("SKIP_LIBS_CHECKS", False)))
FFMPEG_LIBRARIES = [
    "avcodec",
    "avutil",
]


def get_include_dirs():
    """Get distutils-compatible extension arguments using pkg-config for libav and cuda."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-I", dest="include_dirs", action="append", default=[])
    parser.add_argument("-l", dest="libraries", action="append", default=[])
    parser.add_argument("-L", dest="library_dirs", action="append", default=[])
    parser.add_argument("-R", dest="runtime_library_dirs", action="append", default=[])

    # Get libav libraries
    try:
        raw_cflags = subprocess.check_output(
            ["pkg-config", "--cflags", "--libs"] + ["lib" + name for name in FFMPEG_LIBRARIES]  # noqa: S603
        )
    except subprocess.CalledProcessError as e:
        raw_cflags = b""
        if not SKIP_LIBS_CHECKS:
            raise RuntimeError(
                f"Couldn't find ffmpeg libs {FFMPEG_LIBRARIES}: {e.stderr}. "
                "Try specifying the ffmpeg dir with `export PKG_CONFIG_LIBDIR=[ffmpeg_dir]/lib/pkgconfig`"
            ) from e
    args, _ = parser.parse_known_args(raw_cflags.decode("utf-8").strip().split())

    # Get CUDA libraries
    if CUDA_HOME:
        args.include_dirs.extend([str(Path(CUDA_HOME) / "include")])
        args.libraries.extend(["cudart"])
        args.library_dirs.extend([str(Path(CUDA_HOME) / "lib64")])
        args.runtime_library_dirs.extend([str(Path(CUDA_HOME) / "lib64")])
    elif not SKIP_LIBS_CHECKS:
        raise RuntimeError("Couldn't find CUDA path. Please set $CUDA_HOME env variable.")
    return args


class CustomBuildExt(_build_ext):
    def build_extensions(self):
        if not NVCC_PATH:
            raise ValueError("Couldn't find nvcc compiler. Please set $CUDA_HOME env variable.")

        # Add support for .cu files compilation
        self.compiler.src_extensions.append(".cu")
        default_compile = self.compiler._compile

        # Redefine _compile to change compiler based on the source extension
        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            default_compiler_so = self.compiler.compiler_so
            if Path(src).suffix == ".cu":
                self.compiler.set_executable("compiler_so", NVCC_PATH)
                self.compiler.set_executable("compiler_cxx", NVCC_PATH)
                self.compiler.set_executable("compiler", NVCC_PATH)
                postargs = extra_postargs["nvcc"]
            else:
                postargs = extra_postargs["gcc"]
            default_compile(obj, src, ext, cc_args, postargs, pp_opts)
            self.compiler.compiler_so = default_compiler_so

        self.compiler._compile = _compile
        super().build_extensions()


extension_extras = get_include_dirs()

cuda_filepaths = [str(path) for path in Path("avhardware/cuda").glob("**/*.cu")]

ext_modules = []
for filepath in Path("avhardware").glob("**/*.pyx"):
    module_name = str(filepath.parent / filepath.stem).replace("/", ".").replace(os.sep, ".")
    ext_modules += cythonize(
        Extension(
            module_name,
            include_dirs=["avhardware"] + extension_extras.include_dirs,
            libraries=extension_extras.libraries,
            library_dirs=extension_extras.library_dirs,
            runtime_library_dirs=extension_extras.runtime_library_dirs,
            sources=[str(filepath), *cuda_filepaths],
            extra_compile_args={
                "gcc": [],
                "nvcc": [f"-arch={CUDA_ARCH}", "--ptxas-options=-v", "-c", "--compiler-options", "'-fPIC'"],
            },
        ),
        build_dir="build",
        include_path=[av.get_include()],
    )

setup(
    packages=find_packages(exclude=["build*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
)
