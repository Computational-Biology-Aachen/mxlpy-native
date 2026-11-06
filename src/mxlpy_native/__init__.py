# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Compiled solvers for MxlPy's Native integrator.

`load()` returns the solver library for ctypes. Everything that can go wrong with an
installation raises `NativeInstallError` with a message saying how to fix it.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

__all__ = ["ABI_VERSION", "NativeInstallError", "__version__", "library_path", "load"]

# Must match MXN_ABI_VERSION in c/mxn_solvers.c. MxlPy checks it before calling into
# the library, so a mismatched pair fails with an upgrade hint instead of crashing.
ABI_VERSION = 2

try:
    __version__ = version("mxlpy-native")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0+unknown"

_REINSTALL = 'pip install --force-reinstall --no-cache-dir "mxlpy-native"'


class NativeInstallError(ImportError):
    """mxlpy-native is installed, but its compiled solvers can't be used."""


def library_path() -> Path:
    """Return the path of the compiled solvers, for loading with ctypes."""
    here = Path(__file__).parent
    for suffix in EXTENSION_SUFFIXES:
        if (path := here / f"_solvers{suffix}").exists():
            return path
    found = sorted(p.name for p in here.glob("_solvers*"))
    detail = f"found only {found}, none built for this Python" if found else "none found"
    msg = (
        f"mxlpy-native {__version__} is installed without its compiled solvers in {here} "
        f"({detail}).\nThis usually means a source checkout was installed without building "
        f"it, or the wheel belongs to a different platform. Reinstall it with:\n  {_REINSTALL}"
    )
    raise NativeInstallError(msg)


@lru_cache(maxsize=1)
def load() -> ctypes.CDLL:
    """Load the compiled solvers and check that they match this package."""
    path = library_path()
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as e:
        msg = (
            f"mxlpy-native {__version__} can't load its compiled solvers {path}: {e}\n"
            "The file may be corrupt or built for another platform or architecture. "
            f"Reinstall it with:\n  {_REINSTALL}"
        )
        raise NativeInstallError(msg) from e
    try:
        lib.mxn_abi_version.restype = ctypes.c_int
        abi = lib.mxn_abi_version()
    except AttributeError:
        abi = 0  # built before the ABI was versioned
    if abi != ABI_VERSION:
        msg = (
            f"mxlpy-native {__version__}'s compiled solvers {path} have ABI {abi}, but its "
            f"Python code expects {ABI_VERSION}. The build is stale; rebuild it with:\n"
            f"  {_REINSTALL}"
        )
        raise NativeInstallError(msg)
    return lib
