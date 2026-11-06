# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""The compiled solvers load, match the Python package and solve dy/dt = -k y."""

from __future__ import annotations

import ctypes as ct
import math

import pytest

import mxlpy_native

_DP = ct.POINTER(ct.c_double)
_MODEL_FN = ct.CFUNCTYPE(None, ct.c_int, ct.c_double, _DP, _DP, _DP)


@_MODEL_FN
def _decay(n: int, t: float, y: _DP, dydt: _DP, pars: _DP) -> None:  # noqa: ARG001
    for i in range(n):
        dydt[i] = -pars[0] * y[i]


def _arr(*values: float) -> ct.Array:
    return (ct.c_double * len(values))(*values)


def test_abi_matches() -> None:
    assert mxlpy_native.load().mxn_abi_version() == mxlpy_native.ABI_VERSION


@pytest.mark.parametrize("method", ["radau", "lsoda", "cvode-bdf", "cvode-adams"])
def test_solve_exponential_decay(method: str) -> None:
    lib = mxlpy_native.load()
    t_out = [0.0, 0.5, 1.0, 2.0]
    y, pars, t, out = _arr(1.0, 2.0), _arr(0.7), _arr(*t_out), _arr(*[0.0] * 8)
    lib.mxn_set_model.argtypes = [ct.c_void_p]
    lib.mxn_set_model(ct.cast(_decay, ct.c_void_p))
    head = (2, y, pars, ct.c_double(1e-10), ct.c_double(1e-10), 100_000)
    if method == "radau":
        rc = lib.mxn_radau(*head, 0, len(t_out), t, out)
        assert rc == 1
    elif method == "lsoda":
        rc = lib.mxn_lsoda(*head, len(t_out), t, out)
        assert rc == 2
    else:
        rc = lib.mxn_cvode(*head, int(method == "cvode-adams"), len(t_out), t, out)
        assert rc == 1
    for i, ti in enumerate(t_out):
        assert out[2 * i] == pytest.approx(math.exp(-0.7 * ti), rel=1e-7)
        assert out[2 * i + 1] == pytest.approx(2 * math.exp(-0.7 * ti), rel=1e-7)


def test_missing_library_explains_reinstall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mxlpy_native, "EXTENSION_SUFFIXES", [".nope"])
    with pytest.raises(mxlpy_native.NativeInstallError, match="force-reinstall"):
        mxlpy_native.library_path()


def test_diagnostics_pass() -> None:
    from mxlpy_native.__main__ import main  # noqa: PLC0415

    assert main() == 0
