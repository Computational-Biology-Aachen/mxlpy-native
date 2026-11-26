# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Discrete adjoint gradients (mxn_da_*): exact for the recorded solve, for every method."""

from __future__ import annotations

import ctypes as ct
import math

import pytest

import mxlpy_native

_DP = ct.POINTER(ct.c_double)
_IP = ct.POINTER(ct.c_int)
_MODEL_FN = ct.CFUNCTYPE(None, ct.c_int, ct.c_double, _DP, _DP, _DP)
_ADJOINT_FN = ct.CFUNCTYPE(None, ct.c_int, ct.c_double, _DP, _DP, _DP, _DP, _DP)
_METHODS = {"radau5": 0, "lsoda": 1, "cvode_bdf": 2, "cvode_adams": 3}


# x' = -k1 x + u, y' = k1 x - k2 y; pars = [k1, k2, u], theta = [k1, k2]
@_MODEL_FN
def _chain(n: int, t: float, y: _DP, f: _DP, p: _DP) -> None:  # noqa: ARG001
    f[0] = -p[0] * y[0] + p[2]
    f[1] = p[0] * y[0] - p[1] * y[1]


@_MODEL_FN
def _chain_jac(n: int, t: float, y: _DP, j: _DP, p: _DP) -> None:  # noqa: ARG001
    j[0], j[1], j[2], j[3] = -p[0], p[0], 0.0, -p[1]  # column-major


@_ADJOINT_FN
def _chain_adjoint(  # noqa: PLR0913, PLR0917
    n: int,  # noqa: ARG001
    t: float,  # noqa: ARG001
    y: _DP,
    lam: _DP,
    p: _DP,
    dlam: _DP,
    dtheta: _DP,
) -> None:
    dlam[0] = -(-p[0] * lam[0] + p[0] * lam[1])
    dlam[1] = -(-p[1] * lam[1])
    dtheta[0] = -(-y[0] * lam[0] + y[0] * lam[1])
    dtheta[1] = -(-y[1] * lam[1])


def _doubles(*values: float) -> ct.Array:
    return (ct.c_double * max(len(values), 1))(*values)


def _ints(*values: int) -> ct.Array:
    return (ct.c_int * max(len(values), 1))(*values)


def _lib() -> ct.CDLL:
    lib = mxlpy_native.load()
    d, i, v = ct.c_double, ct.c_int, ct.c_void_p
    lib.mxn_da_forward.argtypes = [
        *[v, v, v, i, i, _DP, i, i, _DP, _DP, i, _DP, i, _IP, _IP, i],
        *[d, d, i, i, d, d, _DP, ct.POINTER(v)],
    ]
    lib.mxn_da_forward.restype = i
    lib.mxn_da_backward.argtypes = [v, v, v, i, _DP, _DP, _DP]
    lib.mxn_da_backward.restype = i
    lib.mxn_da_replay.argtypes = [v, _DP, _DP, _DP]
    lib.mxn_da_replay.restype = i
    lib.mxn_da_free.argtypes = [v]
    lib.mxn_da_free.restype = None
    lib.mxn_da_tape_steps.argtypes = [v]
    return lib


_SEG_END = (2.0, 5.0)
_TIMES = (0.0, 0.5, 2.0, 3.3, 5.0)


def _seg_pars(k1: float, k2: float) -> ct.Array:
    return _doubles(k1, k2, 0.0, k1, k2, 0.4)  # u switches on in the second segment


@pytest.mark.parametrize("method", list(_METHODS))
def test_gradient_matches_frozen_step_differences(method: str) -> None:
    lib = _lib()
    k1, k2, y0 = 0.9, 0.3, _doubles(1.0, 0.0)
    n_pts, n_tgt = len(_TIMES), 2
    pred = _doubles(*[0.0] * (n_pts * n_tgt))
    tape = ct.c_void_p()
    rc = lib.mxn_da_forward(
        ct.cast(_chain, ct.c_void_p),
        ct.cast(_chain_jac, ct.c_void_p),
        None,
        _METHODS[method],
        2,
        y0,
        3,
        2,
        _doubles(*_SEG_END),
        _seg_pars(k1, k2),
        n_pts,
        _doubles(*_TIMES),
        n_tgt,
        _ints(0, 0),
        _ints(0, 1),
        0,
        1e-6,
        1e-8,
        100_000,
        2,  # DA_FORWARD_JAC
        1e-3,
        0.0,
        pred,
        ct.byref(tape),
    )
    assert rc == 0
    try:
        assert lib.mxn_da_tape_steps(tape) > 0
        # x(t) = exp(-k1 t) before u switches on
        assert pred[1 * n_tgt] == pytest.approx(math.exp(-k1 * 0.5), rel=1e-5)
        cot = _doubles(*[(-1.0) ** j * (1 + j % 3) for j in range(n_pts * n_tgt)])
        grad, lam0 = _doubles(0.0, 0.0), _doubles(0.0, 0.0)
        rc = lib.mxn_da_backward(
            tape, ct.cast(_chain_adjoint, ct.c_void_p), None, 2, cot, grad, lam0
        )
        assert rc == 0

        def loss(a: float, b: float, x0: float = 1.0) -> float:
            out = _doubles(*[0.0] * (n_pts * n_tgt))
            assert lib.mxn_da_replay(tape, _seg_pars(a, b), _doubles(x0, 0.0), out) == 0
            return sum(cot[j] * out[j] for j in range(n_pts * n_tgt))

        h = 1e-6
        fd = [
            (loss(k1 + h, k2) - loss(k1 - h, k2)) / (2 * h),
            (loss(k1, k2 + h) - loss(k1, k2 - h)) / (2 * h),
            (loss(k1, k2, 1 + h) - loss(k1, k2, 1 - h)) / (2 * h),
        ]
        assert [grad[0], grad[1], lam0[0]] == pytest.approx(fd, rel=1e-6)
    finally:
        lib.mxn_da_free(tape)
