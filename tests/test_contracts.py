# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Contracts of mxn_radau / mxn_lsoda / mxn_cvode, ported from other ODE suites' tests.

From diffrax (test_saveat_solution, test_integrate), OrdinaryDiffEq.jl (InterfaceI,
check_error), scipy.integrate (test_integrate.py, test_ivp.py), ODEInterfaceDiffEq.jl
(jac_tests), SUNDIALS (cv_test_tstop, iterators) and the Bari test set. Errors are relative.
"""

from __future__ import annotations

import ctypes as ct
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

import mxlpy_native

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_DP = ct.POINTER(ct.c_double)
_MODEL_FN = ct.CFUNCTYPE(None, ct.c_int, ct.c_double, _DP, _DP, _DP)


@dataclass(frozen=True)
class Method:
    name: str
    success: int
    bad_input: int
    too_many_steps: int


METHODS = [
    Method("radau", 1, -1, -2),
    Method("lsoda", 2, -3, -1),
    Method("cvode-bdf", 1, -22, -1),
    Method("cvode-adams", 1, -22, -1),
]


@dataclass
class Calls:
    """What the right-hand side was called with."""

    n: int = 0
    t_min: float = math.inf
    t_max: float = -math.inf
    jac: int = 0
    times: list[float] = field(default_factory=list)


_calls = Calls()


def _model(f: Callable[[float, Sequence[float], Sequence[float]], Sequence[float]]) -> ct._CFuncPtr:
    @_MODEL_FN
    def rhs(n: int, t: float, y: _DP, dydt: _DP, pars: _DP) -> None:
        _calls.n += 1
        _calls.t_min = min(_calls.t_min, t)
        _calls.t_max = max(_calls.t_max, t)
        values = f(t, [y[i] for i in range(n)], pars)
        for i in range(n):
            dydt[i] = values[i]

    return rhs


DECAY = _model(lambda t, y, p: [-p[0] * v for v in y])  # noqa: ARG005
NAN_AFTER_HALF = _model(lambda t, y, p: [math.nan if t > 0.5 else -v for v in y])  # noqa: ARG005
SQUARE_BLOWUP = _model(lambda t, y, p: [v * v for v in y])  # noqa: ARG005
ROBERTSON = _model(
    lambda t, y, p: [  # noqa: ARG005
        -0.04 * y[0] + 1e4 * y[1] * y[2],
        0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
        3e7 * y[1] ** 2,
    ]
)


@_MODEL_FN
def ROBERTSON_JAC(n: int, t: float, y: _DP, jac: _DP, pars: _DP) -> None:  # noqa: ARG001, N802
    _calls.jac += 1
    values = [  # column-major
        -0.04,
        0.04,
        0.0,
        1e4 * y[2],
        -1e4 * y[2] - 6e7 * y[1],
        6e7 * y[1],
        1e4 * y[1],
        -1e4 * y[1],
        0.0,
    ]
    for i, v in enumerate(values):
        jac[i] = v


def _arr(values: Sequence[float]) -> ct.Array:
    return (ct.c_double * max(len(values), 1))(*values)


@dataclass
class Result:
    rc: int
    rows: list[list[float]]
    y: list[float]


def solve(  # noqa: PLR0913
    method: Method,
    model: ct._CFuncPtr,
    y0: Sequence[float],
    t_out: Sequence[float],
    pars: Sequence[float] = (0.0,),
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    nmax: int = 100_000,
    jac: ct._CFuncPtr | None = None,
) -> Result:
    global _calls  # noqa: PLW0603
    _calls = Calls()
    lib = mxlpy_native.load()
    lib.mxn_set_model.argtypes = [ct.c_void_p]
    lib.mxn_set_model(ct.cast(model, ct.c_void_p))
    n = len(y0)
    y, p, t = _arr(y0), _arr(pars), _arr(t_out)
    out = _arr([0.0] * max(n * len(t_out), 1))
    j = ct.cast(jac, ct.c_void_p) if jac is not None else ct.c_void_p(0)
    head = (n, y, p, j, ct.c_double(rtol), ct.c_double(atol), nmax)
    if method.name == "radau":
        rc = lib.mxn_radau(*head, 0, len(t_out), t, out)
    elif method.name == "lsoda":
        rc = lib.mxn_lsoda(*head, len(t_out), t, out)
    else:
        rc = lib.mxn_cvode(*head, int(method.name == "cvode-adams"), len(t_out), t, out)
    rows = [[out[k * n + i] for i in range(n)] for k in range(len(t_out))]
    return Result(rc, rows, [y[i] for i in range(n)])


methods = pytest.mark.parametrize("method", METHODS, ids=lambda m: m.name)


@methods
def test_equal_output_times_return_y0(method: Method) -> None:
    """diffrax test_t0_eq_t1: a zero-length span returns y0 exactly, for every repeat."""
    r = solve(method, DECAY, [2.0, -3.0], [1.0, 1.0, 1.0], [1.0])
    assert r.rc == method.success
    assert r.rows == [[2.0, -3.0]] * 3
    assert _calls.n == 0


@methods
def test_single_output_time(method: Method) -> None:
    r = solve(method, DECAY, [2.0], [1.0], [1.0])
    assert r.rc == method.success
    assert r.rows == [[2.0]]


@methods
def test_grid_values_and_first_row(method: Method) -> None:
    """diffrax test_saveat_solution, OrdinaryDiffEq ode_saveat_tests: y(t0) is y0 bitwise,
    closely spaced and repeated output times are all accurate."""
    t_out = [0.0, 0.125, 0.6, 0.61, 0.61, 0.8, 1.0]
    r = solve(method, DECAY, [2.1], t_out, [0.5])
    assert r.rc == method.success
    assert r.rows[0] == [2.1]
    for t, row in zip(t_out, r.rows, strict=True):
        assert row[0] == pytest.approx(2.1 * math.exp(-0.5 * t), rel=1e-8)
    assert r.rows[3] == r.rows[4]


@methods
def test_backward(method: Method) -> None:
    """diffrax test_reverse_time: decreasing output times integrate backward."""
    t_out = [4.0, 3.5, 0.7, 0.3]
    r = solve(method, DECAY, [1.0], t_out, [1.0])
    assert r.rc == method.success
    for t, row in zip(t_out, r.rows, strict=True):
        assert row[0] == pytest.approx(math.exp(4.0 - t), rel=1e-8)
    assert _calls.t_min >= 0.3
    assert _calls.t_max <= 4.0


@methods
@pytest.mark.parametrize(
    ("y0", "t_out"),
    [
        ([1.0], [0.0, 0.7, 0.3]),  # not monotone
        ([1.0], [0.0, math.nan, 1.0]),
        ([1.0], [0.0, math.inf]),
        ([math.nan], [0.0, 1.0]),
        ([1.0], []),  # no output time
    ],
    ids=["unsorted", "nan-time", "inf-time", "nan-y0", "no-times"],
)
def test_bad_input_is_refused(method: Method, y0: list[float], t_out: list[float]) -> None:
    """DiffEqBase solve_error_handling, scipy test_initial_state_finiteness / gh-8217."""
    r = solve(method, DECAY, y0, t_out, [1.0])
    assert r.rc == method.bad_input
    assert _calls.n == 0


@methods
def test_nan_rhs_fails_fast(method: Method) -> None:
    """OrdinaryDiffEq ode_initdt_tests (#1404): a NaN right-hand side is a clean failure, not a
    success with NaN and not a crawl through the whole step budget."""
    r = solve(method, NAN_AFTER_HALF, [1.0], [0.0, 1.0], rtol=1e-8, atol=1e-10)
    assert r.rc < 0
    assert _calls.n < 20_000


@methods
def test_blowup_fails_fast(method: Method) -> None:
    """OrdinaryDiffEq check_error: x' = x^2 blows up at t = 1."""
    r = solve(method, SQUARE_BLOWUP, [1.0], [0.0, 2.0], rtol=1e-8, atol=1e-8)
    assert r.rc < 0
    assert _calls.n < 50_000


@methods
def test_never_evaluates_past_the_last_output(method: Method) -> None:
    """scipy gh-17341 / 8848 / 9198: the model is never called outside [t0, t_end]."""
    r = solve(method, DECAY, [1.0], [0.0, 0.3, 0.77], [1.0])
    assert r.rc == method.success
    assert _calls.t_min >= 0.0
    assert _calls.t_max <= 0.77


@methods
def test_step_budget(method: Method) -> None:
    """LSODA.jl maxiters=1: an exhausted budget is the solver's too-many-steps code, and
    nmax <= 0 means the default budget."""
    assert (
        solve(method, ROBERTSON, [1.0, 0.0, 0.0], [0.0, 40.0], nmax=5).rc == method.too_many_steps
    )
    if method.name == "cvode-adams":
        return  # Robertson is too stiff for Adams' default budget
    assert solve(method, ROBERTSON, [1.0, 0.0, 0.0], [0.0, 40.0], nmax=0).rc == method.success


@methods
def test_empty_state(method: Method) -> None:
    """scipy test_empty: no state variables is a no-op success."""
    r = solve(method, DECAY, [], [0.0, 1.0], [1.0])
    assert r.rc == method.success


@methods
def test_repeated_solves_are_identical(method: Method) -> None:
    """Sundials.jl iterators (reinit): no state survives between solves."""
    first = solve(method, ROBERTSON, [1.0, 0.0, 0.0], [0.0, 0.4, 4.0], rtol=1e-6, atol=1e-10)
    solve(method, DECAY, [1.0], [0.0, 3.0], [1.0], rtol=1e-4, atol=1e-6)
    again = solve(method, ROBERTSON, [1.0, 0.0, 0.0], [0.0, 0.4, 4.0], rtol=1e-6, atol=1e-10)
    assert first.rows == again.rows


@methods
def test_user_jacobian_is_called_and_matches(method: Method) -> None:
    """ODEInterfaceDiffEq jac_tests: the user Jacobian is actually used, and the result matches
    the finite-difference one (Robertson at t = 40 against Hairer & Wanner's reference)."""
    if method.name == "cvode-adams":
        pytest.skip("Adams uses fixed-point iteration, no Jacobian")
    ref = [0.7158270687193941, 9.185534764557338e-6, 0.2841637457458413]
    r = solve(method, ROBERTSON, [1.0, 0.0, 0.0], [0.0, 40.0], jac=ROBERTSON_JAC)
    assert r.rc == method.success
    assert _calls.jac > 0
    for got, want in zip(r.y, ref, strict=True):
        assert got == pytest.approx(want, rel=1e-6)


@methods
def test_robertson_long_span(method: Method) -> None:
    """Bari test set ROBER to t = 1e11: positive, mass-conserving, 5 correct digits."""
    if method.name == "cvode-adams":
        pytest.skip("Robertson is stiff")
    ref = [2.083340149701255e-8, 8.333360770334713e-14, 0.9999999791665050]
    r = solve(
        method, ROBERTSON, [1.0, 0.0, 0.0], [0.0, 1e11], rtol=1e-10, atol=1e-20, nmax=1_000_000
    )
    assert r.rc == method.success
    for got, want in zip(r.y, ref, strict=True):
        assert got == pytest.approx(want, rel=1e-5)
    assert sum(r.y) == pytest.approx(1.0, abs=1e-8)
