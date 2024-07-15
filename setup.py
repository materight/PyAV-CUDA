import argparse
import os
import subprocess
from pathlib import Path

import av
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

FFMPEG_LIBRARIES = [
    "avcodec",
    "avutil",
]
CUDA_HOME = os.environ.get("CUDA_HOME", None)


def get_include_dirs():
    """Get distutils-compatible extension arguments using pkg-config for libav and cuda."""
    # Get libav libraries
    try:
        raw_cflags = subprocess.check_output(
            ["pkg-config", "--cflags", "--libs"] + ["lib" + name for name in FFMPEG_LIBRARIES]  # noqa: S603
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Couldn't find ffmpeg libs {FFMPEG_LIBRARIES}: {e.stderr}. "
            "Try specifying the ffmpeg dir with `export PKG_CONFIG_LIBDIR=[ffmpeg_dir]/lib/pkgconfig`"
        ) from e
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-I", dest="include_dirs", action="append", default=[])
    parser.add_argument("-l", dest="libraries", action="append", default=[])
    parser.add_argument("-L", dest="library_dirs", action="append", default=[])
    args, _ = parser.parse_known_args(raw_cflags.decode("utf-8").strip().split())

    # Get CUDA libraries
    if CUDA_HOME is None:
        raise ValueError("Couldn't find cuda path. Pleae set $CUDA_HOME env variable.")
    args.include_dirs.extend([str(Path(CUDA_HOME) / "include")])
    args.libraries.extend(["cudart"])
    args.library_dirs.extend([str(Path(CUDA_HOME) / "lib64")])
    return args


extension_extras = get_include_dirs()

ext_modules = []
for filepath in Path("av_hw").glob("**/*.pyx"):
    module_name = str(filepath.parent / filepath.stem).replace("/", ".").replace(os.sep, ".")
    ext_modules += cythonize(
        Extension(
            module_name,
            include_dirs=extension_extras.include_dirs,
            libraries=extension_extras.libraries,
            library_dirs=extension_extras.library_dirs,
            sources=[str(filepath)],
        ),
        build_dir="build",
        include_path=[av.get_include()],
    )

print(ext_modules)
setup(
    name="av_hw",
    version="0.1.0",
    packages=find_packages(exclude=["build*"]),
    ext_modules=ext_modules,
    install_requires=["av", "torch"],  # TODO: move to pyproject.toml
)
