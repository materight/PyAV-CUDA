import argparse
import os
import subprocess
from pathlib import Path

import av
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

CUDA_HOME = os.environ.get("CUDA_HOME", None)
CUDA_ARCH_LIST = os.environ.get("CUDA_ARCH_LIST", "75,86")

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

    # Try to load CUDA libraries from the nvidia-* packages if available
    try:
        import nvidia

        nvidia_dir = Path(nvidia.__file__).parent
        for pkg_dir in nvidia_dir.iterdir():
            if pkg_dir.is_dir():
                if (include_dir := pkg_dir / "include").exists():
                    args.include_dirs.append(str(include_dir))
                if (lib_dir := pkg_dir / "lib").exists():
                    # Add missing symlinks
                    for lib_path in lib_dir.iterdir():
                        if lib_path.is_file() and ".so." in lib_path.name:
                            symlink_name = lib_path.name.split(".so.")[0] + ".so"
                            symlink_path = lib_dir / symlink_name
                            if not symlink_path.exists():
                                symlink_path.symlink_to(lib_path)
                    args.library_dirs.append(str(lib_dir))
                    args.runtime_library_dirs.append(str(lib_dir))
    except ImportError:
        pass

    # Try to load CUDA libraries from the CUDA_HOME environment variable
    if CUDA_HOME:
        args.include_dirs.extend([str(Path(CUDA_HOME) / "include")])
        args.libraries.extend(["nppicc"])
        args.library_dirs.extend([str(Path(CUDA_HOME) / "lib64")])
        args.runtime_library_dirs.extend([str(Path(CUDA_HOME) / "lib64")])

    return args


class CustomBuildExt(_build_ext):
    def build_extensions(self):
        if not CUDA_HOME:
            raise ValueError("Couldn't find nvcc compiler. Please set $CUDA_HOME env variable.")
        nvcc_path = str(Path(CUDA_HOME) / "bin" / "nvcc")

        # Add support for .cu files compilation
        self.compiler.src_extensions.append(".cu")
        default_compile = self.compiler._compile

        # Redefine _compile to change compiler based on the source extension
        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            default_compiler_so = self.compiler.compiler_so
            if Path(src).suffix == ".cu":
                self.compiler.set_executable("compiler_so", nvcc_path)
                self.compiler.set_executable("compiler_cxx", nvcc_path)
                self.compiler.set_executable("compiler", nvcc_path)
                postargs = extra_postargs["nvcc"]
            else:
                postargs = extra_postargs["gcc"]
            default_compile(obj, src, ext, cc_args, postargs, pp_opts)
            self.compiler.compiler_so = default_compiler_so

        self.compiler._compile = _compile
        super().build_extensions()


extension_extras = get_include_dirs()

cuda_filepaths = [str(path) for path in Path("avcuda/cuda").glob("**/*.cu")]
cuda_arch_flags = [f for arch in CUDA_ARCH_LIST.split(",") for f in ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]]
cuda_arch_flags.extend(["-gencode", "arch=compute_86,code=compute_86"])  # Add fallback for newer architectures

ext_modules = []
for filepath in Path("avcuda").glob("**/*.pyx"):
    module_name = str(filepath.parent / filepath.stem).replace("/", ".").replace(os.sep, ".")
    ext_modules += cythonize(
        Extension(
            module_name,
            include_dirs=["avcuda"] + extension_extras.include_dirs,
            libraries=extension_extras.libraries,
            library_dirs=extension_extras.library_dirs,
            runtime_library_dirs=extension_extras.runtime_library_dirs,
            sources=[str(filepath), *cuda_filepaths],
            extra_compile_args={
                "gcc": [],
                "nvcc": ["-c", *cuda_arch_flags, "-std=c++17", "--ptxas-options=-v", "--compiler-options", "'-fPIC'"],
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
