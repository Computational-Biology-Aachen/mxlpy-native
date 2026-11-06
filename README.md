# mxlpy-native

The compiled Radau, LSODA and CVODE solvers behind [MxlPy](https://github.com/Computational-Biology-Aachen/MxlPy)'s `Native` integrator, plus `llvmlite`, which compiles models to machine code for them. Install it through MxlPy rather than on its own:

```bash
pip install "mxlpy[native]"
```

## Solvers and licences

| Solver | Origin | Licence |
|---|---|---|
| RADAU5 | E. Hairer and G. Wanner, via f2c | `c/vendor/HAIRER-LICENCE.txt` |
| LSODA | SciPy's C translation of ODEPACK | `c/vendor/SCIPY-LICENCE.txt` |
| CVODE (BDF and Adams) | [SUNDIALS](https://github.com/LLNL/sundials) v7.9.0, © Lawrence Livermore National Security and the SUNDIALS contributors | `c/vendor/SUNDIALS-LICENCE.txt` (BSD-3-Clause), `c/vendor/SUNDIALS-NOTICE.txt` |

All three are vendored from [mxl-solvers](https://github.com/Computational-Biology-Aachen/mxl-solvers), whose README lists the papers to cite for SUNDIALS. The licence files ship in every wheel.

## Platforms

Wheels are published for Linux (x86_64, aarch64; glibc 2.28+), macOS (arm64, 12+) and Windows (amd64, arm64), the platforms `llvmlite` has wheels for. On anything else, `mxlpy[native]` installs the WASM backend (`wasmtime`) instead, which runs the same solvers without a compiler, a little slower.

If pip has to build from the sdist anyway, it needs a C11 compiler (GCC 8+, Clang 7+ or MSVC 2019 16.8+), and nothing else: no Fortran, BLAS/LAPACK or Python headers. The build says what to install if the compiler is missing or fails.

## Troubleshooting

```bash
python -m mxlpy_native
```

prints the installed version, the solver ABI, and whether the compiled solvers and `llvmlite` load, with a fix for each failure.

## Development

```bash
uv sync           # builds the extension in place
uv run pytest
```

`c/vendor/` is copied from [mxl-solvers](https://github.com/Computational-Biology-Aachen/mxl-solvers) by its `scripts/vendor.sh`. `vendor/VENDORED` records the commit and file hashes, which `tests/test_vendor.py` checks. Fix solvers there, not here. `c/mxn_solvers.c` is the glue MxlPy calls, compiled unchanged into the wheel and into MxlPy's `solvers.wasm` (`scripts/build_native_wasm.sh` in MxlPy).

When an exported function's signature or meaning changes, bump `MXN_ABI_VERSION` in `c/mxn_solvers.c`, `ABI_VERSION` in `src/mxlpy_native/__init__.py` and the ABI MxlPy expects, then release both. MxlPy refuses a mismatched pair with an upgrade hint instead of crashing.

## Releasing

Bump `version` in `pyproject.toml` and publish a GitHub release `v<version>`. The release workflow builds the wheels with cibuildwheel plus an sdist and uploads them to PyPI via trusted publishing.
