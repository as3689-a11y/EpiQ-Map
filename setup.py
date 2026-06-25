"""Native build configuration for the hklBen ctypes kernel.

Project metadata lives in pyproject.toml. This file only supplies the
platform-specific compiler flags that setuptools cannot express declaratively.
"""

import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    """Apply portable optimization and OpenMP flags per compiler family."""

    def get_export_symbols(self, extension):
        # This is a ctypes-loaded shared library, not a CPython extension
        # module, so it deliberately has no PyInit_* entry point.
        return []

    def build_extensions(self):
        compiler = self.compiler.compiler_type
        for extension in self.extensions:
            if compiler == "msvc":
                extension.extra_compile_args = ["/O2", "/openmp"]
            elif sys.platform == "darwin":
                # Apple Clang does not ship an OpenMP runtime. The pragmas are
                # safely ignored and the same kernel builds in serial mode.
                extension.extra_compile_args = ["-O3"]
            else:
                extension.extra_compile_args = ["-O3", "-fopenmp"]
                extension.extra_link_args = ["-fopenmp"]
                extension.libraries = ["m"]
        super().build_extensions()


setup(
    ext_modules=[
        Extension(
            "epiq_map.hkl_convert.libhklBen",
            sources=["src/epiq_map/hkl_convert/hkl_ben.c"],
        )
    ],
    cmdclass={"build_ext": BuildExt},
)
