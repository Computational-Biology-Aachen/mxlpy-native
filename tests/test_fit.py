# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Levenberg-Marquardt fits with forward sensitivities (mxn_fit_*) find known optima."""

from __future__ import annotations

import ctypes as ct
import math

import pytest

import mxlpy_native

_DP = ct.POINTER(ct.c_double)
_MODEL_FN = ct.CFUNCTYPE(None, ct.c_int, ct.c_double, _DP, _DP, _DP)
_RADAU5, _LSODA = 0, 3
_TIME_COURSE = 0


@_MODEL_FN
def _decay(n: int, t: float, y: _DP, dydt: _DP, pars: _DP) -> None:  # noqa: ARG001
    dydt[0] = -pars[0] * y[0]


@_MODEL_FN
def _decay_aug(n: int, t: float, y: _DP, dydt: _DP, pars: _DP) -> None:  # noqa: ARG001
    # The state x, then its sensitivity dx/dk
    dydt[0] = -pars[0] * y[0]
    dydt[1] = -pars[0] * y[1] - y[0]


def _doubles(*values: float) -> ct.Array:
    return (ct.c_double * max(len(values), 1))(*values)


def _ints(*values: int) -> ct.Array:
    return (ct.c_int * max(len(values), 1))(*values)


def _setup(lib: ct.CDLL) -> None:
    lib.mxn_fit_set_fns.argtypes = [ct.c_void_p] * 4
    lib.mxn_fit_residual_norm.restype = ct.c_double
    d = ct.c_double
    lib.mxn_fit_init.argtypes = [
        *[ct.c_int, _DP, ct.c_int, _DP, ct.c_int, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)],
        *[ct.c_int, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), _DP],
        *[ct.c_int, ct.POINTER(ct.c_int), ct.c_int, ct.POINTER(ct.c_int)],
        *[ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.c_int, _DP, _DP],
        *[ct.c_int, _DP, _DP, ct.c_int, ct.c_int, d, d, d, d],
    ]


@pytest.mark.parametrize("solver", [_RADAU5, _LSODA])
@pytest.mark.parametrize("log", [0, 1])
def test_fit_exponential_decay(solver: int, log: int) -> None:
    lib = mxlpy_native.load()
    _setup(lib)
    k_true = 0.7
    t = [0.5 * (j + 1) for j in range(6)]
    data = [2.0 * math.exp(-k_true * tj) for tj in t]
    lib.mxn_fit_set_fns(ct.cast(_decay, ct.c_void_p), ct.cast(_decay_aug, ct.c_void_p), None, None)
    rc = lib.mxn_fit_init(
        1, _doubles(2.0), 1, _doubles(1.5), 1, _ints(0), _ints(log),
        1, _ints(0), _ints(0), _doubles(1.0),
        0, _ints(), 1, _ints(_TIME_COURSE),
        _ints(1), _ints(len(t)), 1, _doubles(t[-1]), _doubles(),
        len(t), _doubles(*t), _doubles(*data), 0, solver, 1e-10, 1e-12, 1e-10, 1e6,
    )  # fmt: skip
    assert rc == 0
    try:
        info = 0
        for _ in range(20):
            info = lib.mxn_fit_chunk(50)
            if info != 5:
                break
        assert 1 <= info <= 4
        out = _doubles(0.0)
        lib.mxn_fit_params(out)
        assert out[0] == pytest.approx(k_true, rel=1e-8)
        assert lib.mxn_fit_residual_norm() < 1e-8
        assert lib.mxn_fit_nfev() > 0
        predictions = _doubles(*[0.0] * len(t))
        assert lib.mxn_fit_predictions(predictions) == 0
        assert list(predictions) == pytest.approx(data, rel=1e-8)
        jac = _doubles(*[0.0] * len(t))
        assert lib.mxn_fit_jacobian(jac) == 0
        # d/dk 2 exp(-k t) = -2 t exp(-k t), times k for log parameters
        for j, tj in enumerate(t):
            expected = -2 * tj * math.exp(-k_true * tj) * (k_true if log else 1.0)
            assert jac[j] == pytest.approx(expected, rel=1e-6)
    finally:
        lib.mxn_fit_free()


def test_fit_rejects_non_positive_log_parameter() -> None:
    lib = mxlpy_native.load()
    _setup(lib)
    lib.mxn_fit_set_fns(ct.cast(_decay, ct.c_void_p), ct.cast(_decay_aug, ct.c_void_p), None, None)
    rc = lib.mxn_fit_init(
        1, _doubles(2.0), 1, _doubles(-1.0), 1, _ints(0), _ints(1),
        1, _ints(0), _ints(0), _doubles(1.0),
        0, _ints(), 1, _ints(_TIME_COURSE),
        _ints(1), _ints(1), 1, _doubles(1.0), _doubles(),
        1, _doubles(1.0), _doubles(1.0), 0, _RADAU5, 1e-8, 1e-10, 1e-10, 1e6,
    )  # fmt: skip
    assert rc == -2
    lib.mxn_fit_free()
