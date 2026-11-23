# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Adjoint gradients (mxn_adjoint_*) match analytic ones."""

from __future__ import annotations

import ctypes as ct
import math

import pytest

import mxlpy_native

_DP = ct.POINTER(ct.c_double)
_IP = ct.POINTER(ct.c_int)
_MODEL_FN = ct.CFUNCTYPE(None, ct.c_int, ct.c_double, _DP, _DP, _DP)
_ADJOINT_FN = ct.CFUNCTYPE(None, ct.c_int, ct.c_double, _DP, _DP, _DP, _DP, _DP)
_RADAU5, _LSODA = 0, 3
_TIME_COURSE, _STATE = 0, 0


@_MODEL_FN
def _decay(n: int, t: float, y: _DP, dydt: _DP, pars: _DP) -> None:  # noqa: ARG001
    dydt[0] = -pars[0] * y[0]


@_ADJOINT_FN
def _decay_adjoint(  # noqa: PLR0913, PLR0917
    n: int,  # noqa: ARG001
    t: float,  # noqa: ARG001
    y: _DP,
    lam: _DP,
    pars: _DP,
    dlam: _DP,
    dtheta: _DP,
) -> None:
    # -(df/dx) lambda and -lambda df/dk for f = -k x
    dlam[0] = pars[0] * lam[0]
    dtheta[0] = lam[0] * y[0]


def _doubles(*values: float) -> ct.Array:
    return (ct.c_double * max(len(values), 1))(*values)


def _ints(*values: int) -> ct.Array:
    return (ct.c_int * max(len(values), 1))(*values)


@pytest.mark.parametrize("solver", [_RADAU5, _LSODA])
def test_vjp_exponential_decay(solver: int) -> None:
    lib = mxlpy_native.load()
    lib.mxn_adjoint_set_fns.argtypes = [ct.c_void_p] * 4
    d = ct.c_double
    lib.mxn_adjoint_vjp.argtypes = [
        *[ct.c_int, _DP, ct.c_int, _DP, ct.c_int, _IP],
        *[ct.c_int, _IP, _IP, ct.c_int, _IP],
        *[ct.c_int, _IP, _IP, _IP, ct.c_int, _DP, _DP],
        *[ct.c_int, _DP, _DP, ct.c_int, ct.c_int, d, d, d, d],
        *[_DP, _DP, _DP],
    ]
    k, x0 = 0.7, 2.0
    t = [0.5, 1.0, 2.0, 4.0]
    w = [1.0, -2.0, 0.5, 3.0]
    exact_k = sum(-wj * x0 * tj * math.exp(-k * tj) for wj, tj in zip(w, t, strict=True))
    exact_x0 = sum(wj * math.exp(-k * tj) for wj, tj in zip(w, t, strict=True))

    pred, grad, lam0 = _doubles(0, 0, 0, 0), _doubles(0), _doubles(0)
    lib.mxn_adjoint_set_fns(
        ct.cast(_decay, ct.c_void_p), ct.cast(_decay_adjoint, ct.c_void_p), None, None
    )
    rc = lib.mxn_adjoint_vjp(
        1, _doubles(x0), 1, _doubles(k), 1, _ints(0),
        1, _ints(_STATE), _ints(0), 0, _ints(),
        1, _ints(_TIME_COURSE), _ints(1), _ints(4), 1, _doubles(4.0), _doubles(),
        4, _doubles(*t), _doubles(*w), 0, solver, 1e-10, 1e-12, 1e-10, 1e6,
        pred, grad, lam0,
    )  # fmt: skip
    assert rc == 0
    assert grad[0] == pytest.approx(exact_k, rel=1e-6)
    assert lam0[0] == pytest.approx(exact_x0, rel=1e-6)
    assert pred[3] == pytest.approx(x0 * math.exp(-k * 4.0), rel=1e-8)


def test_vjp_needs_functions() -> None:
    lib = mxlpy_native.load()
    lib.mxn_adjoint_set_fns.argtypes = [ct.c_void_p] * 4
    lib.mxn_adjoint_set_fns(None, None, None, None)
    rc = lib.mxn_adjoint_vjp(
        1, _doubles(1.0), 1, _doubles(1.0), 1, _ints(0),
        1, _ints(_STATE), _ints(0), 0, _ints(),
        1, _ints(_TIME_COURSE), _ints(1), _ints(1), 1, _doubles(1.0), _doubles(),
        1, _doubles(1.0), _doubles(1.0), 0, _RADAU5, 1e-8, 1e-10, 1e-8, 1e6,
        _doubles(0), _doubles(0), _doubles(0),
    )  # fmt: skip
    assert rc == -7
