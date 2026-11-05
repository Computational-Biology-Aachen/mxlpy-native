# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Build the solver as one stable-ABI extension, so a wheel covers all Python versions.

Wheels are published for the common platforms, so this only runs when pip has to build
from the sdist. That needs a C11 compiler and nothing else: no Fortran, BLAS or Python
headers. When it fails, the error says what to install instead of showing a raw
compiler traceback.
"""

from __future__ import annotations

import platform
import shutil
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import CCompilerError, ExecError, PlatformError

_NO_COMPILER_HINTS = {
    "linux": (
        "  Debian/Ubuntu:  sudo apt install build-essential\n"
        "  Fedora/RHEL:    sudo dnf install gcc\n"
        "  Alpine:         apk add build-base\n"
        "  conda/pixi:     conda install -c conda-forge c-compiler"
    ),
    "darwin": "  macOS:          xcode-select --install",
    "win32": (
        '  Windows:        install "Build Tools for Visual Studio" (2019 16.8 or newer)\n'
        '                  with the "Desktop development with C++" workload'
    ),
}
_ALTERNATIVE = (
    "If you can't install a compiler, use the WASM backend instead, which needs none:\n"
    '  pip install "mxlpy[native-wasm]"'
)


def _platform() -> str:
    return f"{sys.platform} {platform.machine()}, Python {platform.python_version()}"


def _no_compiler(detail: str) -> str:
    hints = _NO_COMPILER_HINTS.get(sys.platform, "  Install a C11 compiler such as gcc or clang.")
    return (
        f"mxlpy-native: no C compiler found ({detail}).\n\n"
        f"There is no prebuilt mxlpy-native wheel for {_platform()}, so pip is building it\n"
        "from source, which needs a C11 compiler (no Fortran, BLAS or Python headers):\n"
        f"{hints}\n\n{_ALTERNATIVE}"
    )


def _compile_failed(detail: str) -> str:
    return (
        f"mxlpy-native: compiling the solvers failed ({detail}).\n\n"
        f"There is no prebuilt mxlpy-native wheel for {_platform()}, so pip is building it\n"
        "from source. The solvers need C11 support: GCC 8+, Clang 7+ or MSVC 2019 16.8+.\n"
        "Check that your compiler is recent enough and that CC points to it. If it is, please\n"
        "report this at https://github.com/Computational-Biology-Aachen/mxlpy-native/issues\n"
        f"with the compiler output above.\n\n{_ALTERNATIVE}"
    )


class _BuildError(PlatformError):
    """A build error that already explains itself."""


class BuildExt(build_ext):
    """build_ext that explains missing or unsuitable compilers."""

    def run(self) -> None:
        """Explain a missing MSVC."""
        try:
            super().run()
        except _BuildError:
            raise
        except PlatformError as e:  # MSVC isn't installed
            raise _BuildError(_no_compiler(str(e))) from e

    def build_extension(self, ext: Extension) -> None:
        """Explain a missing or failing compiler."""
        compiler = getattr(self.compiler, "compiler_so", None)
        if compiler and shutil.which(compiler[0]) is None:
            raise _BuildError(_no_compiler(f"{compiler[0]!r} is not on PATH"))
        try:
            super().build_extension(ext)
        except (ExecError, FileNotFoundError) as e:  # the compiler couldn't be started
            raise _BuildError(_no_compiler(str(e))) from e
        except CCompilerError as e:  # it ran, but compiling or linking failed
            msg = _compile_failed("the compiler returned an error, see its output above")
            raise _BuildError(msg) from e


vendored = ["radau5.c", "dc_decsol.c", "decsol.c", "libf2c_stubs.c", "lsoda.c", "mxs_lapack.c"]
setup(
    ext_modules=[
        Extension(
            "mxlpy_native._solvers",
            sources=["c/mxn_solvers.c", *(f"c/vendor/{s}" for s in vendored)],
            include_dirs=["c/vendor"],
            py_limited_api=True,
            define_macros=[("Py_LIMITED_API", "0x030C0000")],
            # lsoda.c needs C11 (restrict, compound literals), which MSVC only enables on request.
            extra_compile_args=["/std:c11"]
            if sys.platform == "win32"
            else ["-std=gnu17", "-w", "-O2"],
            libraries=[] if sys.platform == "win32" else ["m"],
        )
    ],
    cmdclass={"build_ext": BuildExt},
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
