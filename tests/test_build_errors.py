# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Building from source without a working compiler explains what to install."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="MSVC ignores CC")


def _build(tmp_path: Path, cc: str) -> str:
    for name in ("setup.py", "pyproject.toml", "c", "src"):
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, tmp_path / name, ignore=shutil.ignore_patterns("*.so", "*.pyd"))
        else:
            shutil.copy(src, tmp_path / name)
    proc = subprocess.run(
        [sys.executable, "setup.py", "-q", "build_ext", "--inplace"],
        cwd=tmp_path,
        env={**os.environ, "CC": cc},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    return proc.stdout + proc.stderr


def test_missing_compiler(tmp_path: Path) -> None:
    out = _build(tmp_path, "/nonexistent/cc")
    assert "no C compiler found" in out
    assert "mxlpy[native-wasm]" in out


def test_failing_compiler(tmp_path: Path) -> None:
    out = _build(tmp_path, "false")
    assert "compiling the solvers failed" in out
    assert "C11" in out
