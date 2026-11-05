# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Diagnose an installation: python -m mxlpy_native."""

from __future__ import annotations

import platform
import sys

import mxlpy_native


def main() -> int:
    """Print what's installed and whether the compiled solvers load."""
    print(f"mxlpy-native {mxlpy_native.__version__} (ABI {mxlpy_native.ABI_VERSION})")
    print(f"platform     {sys.platform} {platform.machine()}, Python {platform.python_version()}")
    ok = True
    try:
        print(f"solvers      {mxlpy_native.load()._name}: ok")  # noqa: SLF001
    except mxlpy_native.NativeInstallError as e:
        print(f"solvers      FAILED\n{e}")
        ok = False
    try:
        import llvmlite.binding as llvm  # noqa: PLC0415

        llvm.initialize_native_target()
        print(f"llvmlite     {llvm.llvm_version_info} targeting {llvm.get_default_triple()}: ok")
    except Exception as e:  # noqa: BLE001
        print(f"llvmlite     FAILED: {e}")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
