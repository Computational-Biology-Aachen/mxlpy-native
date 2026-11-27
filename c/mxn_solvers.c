/* RADAU5, LSODA and CVODE for MxlPy's Native integrator, returning values at requested
 * time points. The same source builds the native wheel and solvers.wasm. The solvers
 * themselves are vendored from mxl-solvers (see vendor/VENDORED); CVODE is SUNDIALS'
 * (vendor/sundials, BSD-3-Clause, see vendor/SUNDIALS-LICENCE.txt). RADAU5 and this
 * file keep global state, so only one solve can run at a time in a process. */
#include "discrete_adjoint.h"
#include "f2c.h"
#include "finite_jacobian.h"
#include "lsoda.h"
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define MXN_API __declspec(dllexport)
#else
#define MXN_API
#endif

/* Bump whenever an exported signature or its meaning changes, together with ABI_VERSION
 * in mxlpy_native/__init__.py and the version MxlPy's Native integrator expects. */
#define MXN_ABI_VERSION 5
MXN_API int mxn_abi_version(void) { return MXN_ABI_VERSION; }

extern int radau5_(integer *n, U_fp fcn, doublereal *x, doublereal *y, doublereal *xend, doublereal *h,
                   doublereal *rtol, doublereal *atol, integer *itol, U_fp jac, integer *ijac, integer *mljac,
                   integer *mujac, U_fp mas, integer *imas, integer *mlmas, integer *mumas, U_fp solout,
                   integer *iout, doublereal *work, integer *lwork, integer *iwork, integer *liwork,
                   doublereal *rpar, integer *ipar, integer *idid);
extern doublereal contr5_(integer *i, doublereal *x, doublereal *cont, integer *lrc);
extern double radau5_jac_growth;

typedef void (*ModelFn)(int, double, double *, double *, double *);
static ModelFn g_model;
/* The model's Jacobian df/dy (ModelFn signature, n x n column-major), NULL = the solvers'
 * finite differences. Called through finite_jacobian, which differences its non-finite columns. */
static ModelFn g_jac;
static double *g_pars; /* LSODA's right-hand side takes no parameter array. */
static const double *g_t_out;
static double *g_y_out;
static int g_n_out, g_next;

/* Shared by mxn_radau, mxn_lsoda and mxn_cvode: 1 if the request is unusable — no output time,
 * a non-finite time, state or tolerance, a negative tolerance, or output times that aren't
 * monotone (repeats are fine) — so the solvers refuse it with their bad-input code instead of
 * reading out of bounds or integrating nonsense. */
static int bad_request(int n, const double *y, double rtol, double atol, int n_out, const double *t_out) {
    if (n < 0 || n_out < 1 || !isfinite(rtol) || !isfinite(atol) || rtol < 0 || atol < 0) return 1;
    for (int i = 0; i < n; i++)
        if (!isfinite(y[i])) return 1;
    const double dir = t_out[n_out - 1] >= t_out[0] ? 1.0 : -1.0;
    for (int k = 0; k < n_out; k++) {
        if (!isfinite(t_out[k])) return 1;
        if (k > 0 && (t_out[k] - t_out[k - 1]) * dir < 0) return 1;
    }
    return 0;
}

/* Nothing to integrate (no state, or all output times equal): every row is y. */
static int trivial_request(int n, const double *y, int n_out, const double *t_out, double *y_out) {
    if (n > 0 && t_out[n_out - 1] != t_out[0]) return 0;
    for (int k = 0; k < n_out; k++) memcpy(y_out + (size_t)k * (size_t)n, y, (size_t)n * sizeof(double));
    return 1;
}

#define DEFAULT_NMAX 100000

/* A function pointer natively, a table index in wasm. */
MXN_API void mxn_set_model(intptr_t model) { g_model = (ModelFn)model; }

static int fcn_dispatch(integer *n, doublereal *x, doublereal *y, doublereal *f, doublereal *rpar, integer *ipar) {
    (void)ipar;
    g_model((int)*n, *x, y, f, rpar);
    return 0;
}

static int radau_jac(integer *n, doublereal *x, doublereal *y, doublereal *dfy, integer *ldfy,
                     doublereal *rpar, integer *ipar) {
    (void)ldfy; /* full Jacobian: ldfy == n */
    (void)ipar;
    finite_jacobian(g_model, g_jac, (int)*n, *x, y, dfy, rpar);
    return 0;
}
static int no_mas(void) { return 0; }

static double g_dir; /* direction of integration, +1 or -1 */

static int solout(integer *nr, doublereal *xold, doublereal *x, doublereal *y, doublereal *cont, integer *lrc,
                  integer *n, doublereal *rpar, integer *ipar, integer *irtrn) {
    (void)xold; (void)rpar; (void)ipar;
    *irtrn = 1;
    while (g_next < g_n_out && (g_t_out[g_next] - *x) * g_dir <= 0) {
        double *row = g_y_out + (size_t)g_next * (size_t)*n;
        doublereal s = g_t_out[g_next];
        /* CONTR5 only exists from the second call on (nr == 1 comes before any step). */
        if (s == *x || *nr < 2) {
            memcpy(row, y, (size_t)*n * sizeof(double));
        } else {
            for (integer i = 1; i <= *n; i++) row[i - 1] = contr5_(&i, &s, cont, lrc);
        }
        g_next++;
    }
    return 0;
}

/* Integrates from t_out[0] to t_out[n_out - 1], writing rows of n values into y_out. The
 * output times are monotone in either direction (t_out[n_out - 1] < t_out[0] integrates
 * backward) and may repeat. y holds the initial state on entry and the final state on
 * return. jac is the model's Jacobian (see g_jac), 0 for finite differences. nmax <= 0 means
 * 100000 steps. Returns RADAU5's IDID: positive on success, -1 bad input (also no output
 * time, non-finite values, unsorted output times), -2 too many steps, -3 step too small
 * (also a non-finite right-hand side), -4 singular matrix. Returns -5 if allocation fails
 * or the result isn't finite. */
MXN_API int mxn_radau(int n, double *y, double *pars, intptr_t jac, double rtol, double atol, int nmax,
                      int reuse_jacobian, int n_out, const double *t_out, double *y_out) {
    if (bad_request(n, y, rtol, atol, n_out, t_out)) return -1;
    if (trivial_request(n, y, n_out, t_out, y_out)) return 1;
    if (nmax <= 0) nmax = DEFAULT_NMAX;
    g_jac = (ModelFn)jac;
    integer N = n, itol = 0, ijac = jac ? 1 : 0, mljac = n, mujac = n, imas = 0, mlmas = n, mumas = n, iout = 1, idid = 0, ipar = 0;
    integer lwork = 4 * N * N + 12 * N + 20, liwork = 3 * N + 20;
    doublereal *work = calloc((size_t)lwork, sizeof(doublereal));
    integer *iwork = calloc((size_t)liwork, sizeof(integer));
    if (!work || !iwork) {
        free(work);
        free(iwork);
        return -5;
    }
    doublereal x = t_out[0], xend = t_out[n_out - 1], h = 0.0, rt = rtol, at = atol;
    iwork[0] = 1;    /* IWORK(1): Hessenberg form */
    iwork[1] = nmax; /* IWORK(2): maximal number of steps */
    radau5_jac_growth = 0.0;
    if (reuse_jacobian) {
        /* Faster on larger stiff models, less accurate on some small ones, and too
         * rough for finite differences across parameters. See misc/mxl-perf. */
        iwork[2] = 10;
        work[2] = 0.05;
        work[5] = 1.5;
        radau5_jac_growth = 8.0;
    }
    g_t_out = t_out;
    g_y_out = y_out;
    g_n_out = n_out;
    g_next = 0;
    g_dir = xend >= x ? 1.0 : -1.0;
    radau5_(&N, (U_fp)fcn_dispatch, &x, y, &xend, &h, &rt, &at, &itol, (U_fp)radau_jac, &ijac, &mljac, &mujac,
            (U_fp)no_mas, &imas, &mlmas, &mumas, (U_fp)solout, &iout, work, &lwork, iwork, &liwork, pars, &ipar, &idid);
    free(work);
    free(iwork);
    if (idid > 0 && g_next != n_out) return -5;
    if (idid > 0) {
        for (int i = 0; i < n; i++) {
            if (!isfinite(y[i])) return -5;
        }
    }
    return (int)idid;
}

static void lsoda_fcn(int *n, double *t, double *y, double *dydt) { g_model(*n, *t, y, dydt, g_pars); }

static void lsoda_jac(int *n, double *t, double *y, int *ml, int *mu, double *pd, int *nrowpd) {
    (void)ml; (void)mu; (void)nrowpd; /* full Jacobian: nrowpd == n */
    finite_jacobian(g_model, g_jac, *n, *t, y, pd, g_pars);
}

/* Same contract as mxn_radau, with LSODA (SciPy's C translation), using jac or its
 * internal finite-difference Jacobian. nmax caps the total number of steps (<= 0: 100000).
 * Returns 2 on success, or LSODA's negative ISTATE: -1 too many steps, -2 too much accuracy
 * requested, -3 bad input, -4 repeated error test failures, -5 repeated convergence
 * failures, -6 an error weight became zero. Returns -7 if the step size became negligible
 * (a singularity or a non-finite right-hand side), -9 if allocation fails or the result
 * isn't finite. */
MXN_API int mxn_lsoda(int n, double *y, double *pars, intptr_t jac, double rtol, double atol, int nmax,
                      int n_out, const double *t_out, double *y_out) {
    if (bad_request(n, y, rtol, atol, n_out, t_out)) return -3;
    if (trivial_request(n, y, n_out, t_out, y_out)) return 2;
    if (nmax <= 0) nmax = DEFAULT_NMAX;
    g_jac = (ModelFn)jac;
    /* Work array sizes as in SciPy's odepack: Adams up to order 12, BDF up to 5. */
    const int lrn = 20 + n * 13 + 3 * n, lrs = 20 + n * 6 + 3 * n + n * n + 2;
    const int lrw = lrn > lrs ? lrn : lrs, liw = 20 + n;
    double *rwork = calloc((size_t)lrw, sizeof(double));
    int *iwork = calloc((size_t)liw, sizeof(int));
    if (!rwork || !iwork) {
        free(rwork);
        free(iwork);
        return -9;
    }
    /* One step at a time (ITASK = 5) and never past the last output time (TCRIT): models
     * needn't be defined beyond it. Rows come from LSODA's interpolant over each step. */
    const double t_end = t_out[n_out - 1], dir = t_end >= t_out[0] ? 1.0 : -1.0;
    int itask = 5, istate = 1, iopt = 1, k = 0;
    rwork[0] = t_end; /* RWORK(1): TCRIT */
    iwork[5] = nmax;  /* IWORK(6): steps allowed per call */
    lsoda_common_struct_t S = {0};
    double t = t_out[0];
    g_pars = pars;
    for (; k < n_out && t_out[k] == t; k++) memcpy(y_out + (size_t)k * (size_t)n, y, (size_t)n * sizeof(double));
    while (k < n_out && istate > 0) {
        double tout = t_end;
        lsoda(lsoda_fcn, n, y, &t, &tout, 1, &rtol, &atol, &itask, &istate, &iopt, rwork, lrw, iwork, liw,
              jac ? lsoda_jac : NULL, jac ? 1 : 2, &S);
        if (istate < 0) break;
        for (; k < n_out && (t_out[k] - t) * dir <= 0; k++) {
            double *row = y_out + (size_t)k * (size_t)n;
            int iflag = 0;
            if (t_out[k] == t) memcpy(row, y, (size_t)n * sizeof(double));
            else lsoda_intdy(t_out[k], 0, rwork, row, &iflag, &S);
        }
        if (iwork[10] > nmax) istate = -1; /* IWORK(11): steps so far */
        /* |h| below the spacing of doubles at t: LSODA only warns (MXHNIL = 10 times). */
        else if (S.nhnil >= 10) istate = -7;
    }
    free(rwork);
    free(iwork);
    if (istate < 0) return istate;
    for (int i = 0; i < n; i++) {
        if (!isfinite(y[i])) return -9;
    }
    return 2;
}

static int cvode_jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data, N_Vector tmp1,
                     N_Vector tmp2, N_Vector tmp3) {
    (void)fy; (void)user_data; (void)tmp1; (void)tmp2; (void)tmp3;
    finite_jacobian(g_model, g_jac, (int)N_VGetLength(y), t, N_VGetArrayPointer(y), SM_DATA_D(J),
                    g_pars); /* column-major */
    return 0;
}

static int cvode_rhs(sunrealtype t, N_Vector y, N_Vector dydt, void *user_data) {
    (void)user_data;
    int n = (int)N_VGetLength(y);
    double *f = N_VGetArrayPointer(dydt);
    g_model(n, t, N_VGetArrayPointer(y), f, g_pars);
    for (int i = 0; i < n; i++) {
        if (!isfinite(f[i])) return 1; /* recoverable: CVODE retries with a smaller step */
    }
    return 0;
}

/* mxn_cvode: the step size became negligible (see there). */
#define MXN_CVODE_STALLED (-98)

/* Same contract as mxn_lsoda, with SUNDIALS' CVODE: BDF with Newton iteration and jac or a
 * dense difference-quotient Jacobian, or with adams != 0, Adams-Moulton with fixed-point
 * iteration. nmax caps the total number of steps (<= 0: 100000). Returns 1 on success, or
 * CVODE's negative flag: -1 too many steps, -2 too much accuracy requested, -3 repeated
 * error test failures, -4 repeated convergence failures, -8 the right-hand side kept
 * failing (non-finite), -22 bad input (also no output time, non-finite values, unsorted
 * output times), ... (see cvode.h). Returns -98 if the step size became negligible (a
 * singularity, or a right-hand side that stays non-finite). Returns -99 if allocation fails or the
 * result isn't finite. */
MXN_API int mxn_cvode(int n, double *y, double *pars, intptr_t jac, double rtol, double atol, int nmax,
                      int adams, int n_out, const double *t_out, double *y_out) {
    if (bad_request(n, y, rtol, atol, n_out, t_out)) return CV_ILL_INPUT;
    if (trivial_request(n, y, n_out, t_out, y_out)) return 1;
    if (nmax <= 0) nmax = DEFAULT_NMAX;
    g_jac = (ModelFn)jac;
    SUNContext ctx = NULL;
    if (SUNContext_Create(SUN_COMM_NULL, &ctx) != 0) return -99;
    N_Vector v = N_VMake_Serial(n, y, ctx); /* works on y in place */
    void *mem = CVodeCreate(adams ? CV_ADAMS : CV_BDF, ctx);
    SUNMatrix A = NULL;
    SUNLinearSolver ls = NULL;
    SUNNonlinearSolver nls = NULL;
    int flag = (v && mem) ? CV_SUCCESS : -99;
    g_pars = pars;
    if (flag == CV_SUCCESS) flag = CVodeInit(mem, cvode_rhs, t_out[0], v);
    if (flag == CV_SUCCESS) flag = CVodeSStolerances(mem, rtol, atol);
    /* Never step past the last output time: models needn't be defined beyond it. */
    if (flag == CV_SUCCESS) flag = CVodeSetStopTime(mem, t_out[n_out - 1]);
    if (flag == CV_SUCCESS && adams) {
        nls = SUNNonlinSol_FixedPoint(v, 0, ctx);
        flag = nls ? CVodeSetNonlinearSolver(mem, nls) : -99;
    } else if (flag == CV_SUCCESS) {
        A = SUNDenseMatrix(n, n, ctx);
        ls = A ? SUNLinSol_Dense(v, A, ctx) : NULL;
        flag = ls ? CVodeSetLinearSolver(mem, ls, A) : -99;
        if (flag == CV_SUCCESS && jac) flag = CVodeSetJacFn(mem, cvode_jac);
    }
    /* One step at a time (CV_ONE_STEP, never past the stop time), rows from CVODE's
     * interpolant: the same steps as CV_NORMAL, but a step that no longer moves t (a
     * singularity, or a right-hand side that is non-finite beyond some t, which CVODE retries
     * with ever smaller steps) is caught after MXHNIL = 10 such steps instead of using up
     * nmax. */
    const double t_end = t_out[n_out - 1], dir = t_end >= t_out[0] ? 1.0 : -1.0;
    N_Vector row = flag == CV_SUCCESS ? N_VNew_Serial(n, ctx) : NULL;
    if (flag == CV_SUCCESS && !row) flag = -99;
    double t = t_out[0];
    int k = 0, tiny_steps = 0;
    for (; k < n_out && t_out[k] == t; k++) memcpy(y_out + (size_t)k * (size_t)n, y, (size_t)n * sizeof(double));
    while (k < n_out && flag >= 0) {
        long int steps = 0;
        CVodeGetNumSteps(mem, &steps);
        if (steps >= nmax) {
            flag = CV_TOO_MUCH_WORK;
            break;
        }
        flag = CVode(mem, t_end, v, &t, CV_ONE_STEP);
        if (flag < 0) break;
        for (; k < n_out && (t_out[k] - t) * dir <= 0; k++) {
            double *dst = y_out + (size_t)k * (size_t)n;
            if (t_out[k] == t) {
                memcpy(dst, y, (size_t)n * sizeof(double));
            } else {
                CVodeGetDky(mem, t_out[k], 0, row);
                memcpy(dst, N_VGetArrayPointer(row), (size_t)n * sizeof(double));
            }
        }
        double h = 0;
        CVodeGetLastStep(mem, &h);
        if (fabs(h) <= 4 * DBL_EPSILON * fabs(t) && ++tiny_steps >= 10) flag = MXN_CVODE_STALLED;
    }
    if (row) N_VDestroy(row);
    CVodeFree(&mem);
    SUNNonlinSolFree(nls);
    SUNLinSolFree(ls);
    SUNMatDestroy(A);
    N_VDestroy(v);
    SUNContext_Free(&ctx);
    if (flag < 0) return flag;
    for (int i = 0; i < n; i++) {
        if (!isfinite(y[i])) return -99;
    }
    return 1;
}

/* Evaluates fn once per row of y, writing n_out values per row into out. */
MXN_API void mxn_eval_rows(intptr_t fn, int n, int n_out, int rows, const double *t, double *y, double *out,
                           double *pars) {
    ModelFn f = (ModelFn)fn;
    for (int r = 0; r < rows; r++) {
        f(n, t[r], y + (size_t)r * (size_t)n, out + (size_t)r * (size_t)n_out, pars);
    }
}

/* ------------------------------------------------------------------ */
/* Levenberg-Marquardt fits with forward sensitivities                 */
/* ------------------------------------------------------------------ */
/* Exported wrappers around mxl-solvers' jacobian_wrapper.c (lmder over an augmented system of
 * states and sensitivities, see its header comment) and fit_experiments.h, which defines the
 * experiment layout. Like the solvers above, a fit keeps global state, so only one runs at a
 * time. Functions are function pointers natively and table indices in wasm. */
void set_derived_fn(intptr_t fn);
void set_derived_grad_fn(intptr_t fn);
void jacobian_set_plain_fn(intptr_t fn);
void jacobian_set_augmented_fn(intptr_t fn);
int jacobian_init(int n_y, double *y0, int n_pars, double *pars, int n_theta, int *theta_idx,
                  int *log_flags, int n_targets, int *target_kind, int *target_index, double *target_scale,
                  int n_ovr, int *ovr_idx, int n_exp, int *exp_kind, int *exp_n_seg, int *exp_n_points,
                  int n_seg_total, double *seg_t_end, double *seg_ovr, int n_points, double *data_t,
                  double *data_y, int n_derived, int solver_id, double rtol, double atol, double ss_tol,
                  double ss_t_max, double target_residual_norm, int progress_interval);
int jacobian_chunk(int maxfev);
int jacobian_get_nfev(void);
double jacobian_get_residual_norm(void);
void jacobian_get_params(double *out);
int jacobian_get_predictions(double *out);
int jacobian_get_jacobian(double *out);
void jacobian_free(void);

/* plain: the model (n_y outputs); augmented: states then sensitivities, parameter-major
 * (n_y * (1 + n_theta) outputs); derived / derived_grad: derived targets and their gradients
 * [d/dy, d/dtheta] per target, 0 if there are none. */
MXN_API void mxn_fit_set_fns(intptr_t plain, intptr_t augmented, intptr_t derived,
                             intptr_t derived_grad) {
    jacobian_set_plain_fn(plain);
    jacobian_set_augmented_fn(augmented);
    set_derived_fn(derived);
    set_derived_grad_fn(derived_grad);
}

/* Arguments as jacobian_init (fit_experiments.h): returns 0, -1 out of memory, -2 a log
 * parameter isn't positive, -3 inconsistent experiments, -4 a fitted parameter is overridden,
 * -5 derived targets without derived functions. */
MXN_API int mxn_fit_init(int n_y, double *y0, int n_pars, double *pars, int n_theta, int *theta_idx,
                         int *log_flags, int n_targets, int *target_kind, int *target_index,
                         double *target_scale, int n_ovr, int *ovr_idx, int n_exp, int *exp_kind,
                         int *exp_n_seg, int *exp_n_points, int n_seg_total, double *seg_t_end,
                         double *seg_ovr, int n_points, double *data_t, double *data_y, int n_derived,
                         int solver_id, double rtol, double atol, double ss_tol, double ss_t_max) {
    return jacobian_init(n_y, y0, n_pars, pars, n_theta, theta_idx, log_flags, n_targets, target_kind,
                         target_index, target_scale, n_ovr, ovr_idx, n_exp, exp_kind, exp_n_seg,
                         exp_n_points, n_seg_total, seg_t_end, seg_ovr, n_points, data_t, data_y,
                         n_derived, solver_id, rtol, atol, ss_tol, ss_t_max, -1.0, 0);
}

/* Runs lmder for at most maxfev residual evaluations. Returns MINPACK's info: 1-4 converged,
 * 5 out of evaluations (call again to continue), others failed. */
MXN_API int mxn_fit_chunk(int maxfev) { return jacobian_chunk(maxfev); }
MXN_API int mxn_fit_nfev(void) { return jacobian_get_nfev(); }
MXN_API double mxn_fit_residual_norm(void) { return jacobian_get_residual_norm(); }
/* All n_pars parameters, with the fitted ones at their current values. */
MXN_API void mxn_fit_params(double *out) { jacobian_get_params(out); }
/* Model values at the data points (n_targets * n_points, NaN for failed experiments). */
MXN_API int mxn_fit_predictions(double *out) { return jacobian_get_predictions(out); }
/* Residual Jacobian at the current parameters, column-major (n_targets * n_points, n_theta),
 * with respect to the natural logarithm of log-fitted parameters. */
MXN_API int mxn_fit_jacobian(double *out) { return jacobian_get_jacobian(out); }
MXN_API void mxn_fit_free(void) { jacobian_free(); }

/* ------------------------------------------------------------------ */
/* Adjoint gradients                                                   */
/* ------------------------------------------------------------------ */
/* Exported wrappers around mxl-solvers' adjoint_wrapper.c: one forward and one backward solve
 * per experiment give the gradient of sum(cotangent * model values) w.r.t. the parameters and
 * the initial state. Global state like the fits, so only one runs at a time. */
void set_forward_model_fn(intptr_t fn);
void set_adjoint_fn(intptr_t fn);
int adjoint_vjp(int n_y, double *y0, int n_pars, double *pars, int n_theta, int *theta_idx,
                int n_targets, int *target_kind, int *target_index, int n_ovr, int *ovr_idx,
                int n_exp, int *exp_kind, int *exp_n_seg, int *exp_n_points, int n_seg_total,
                double *seg_t_end, double *seg_ovr, int n_points, double *data_t,
                double *cotangent, int n_derived, int solver_id, double rtol, double atol,
                double ss_tol, double ss_t_max, double *out_pred, double *out_grad,
                double *out_lambda0);

/* model: the model; adjoint: void(n, t, y, lambda, pars, dlambda, dtheta) writing
 * -(df/dy)^T lambda and -lambda^T df/dtheta; derived / derived_grad as for mxn_fit_set_fns. */
MXN_API void mxn_adjoint_set_fns(intptr_t model, intptr_t adjoint, intptr_t derived,
                                 intptr_t derived_grad) {
    set_forward_model_fn(model);
    set_adjoint_fn(adjoint);
    set_derived_fn(derived);
    set_derived_grad_fn(derived_grad);
}

/* Arguments as adjoint_vjp (adjoint_wrapper.c, fit_experiments.h's experiment layout), with the
 * cotangent in data_y's layout. Returns 0, -1 out of memory, -3 inconsistent experiments, -4 a
 * differentiated parameter is overridden, -5 derived targets without derived functions, -7
 * functions not set, or for a failed forward or backward solve -1000 plus the solver's negative
 * code. */
MXN_API int mxn_adjoint_vjp(int n_y, double *y0, int n_pars, double *pars, int n_theta,
                            int *theta_idx, int n_targets, int *target_kind, int *target_index,
                            int n_ovr, int *ovr_idx, int n_exp, int *exp_kind, int *exp_n_seg,
                            int *exp_n_points, int n_seg_total, double *seg_t_end, double *seg_ovr,
                            int n_points, double *data_t, double *cotangent, int n_derived,
                            int solver_id, double rtol, double atol, double ss_tol, double ss_t_max,
                            double *out_pred, double *out_grad, double *out_lambda0) {
    return adjoint_vjp(n_y, y0, n_pars, pars, n_theta, theta_idx, n_targets, target_kind,
                       target_index, n_ovr, ovr_idx, n_exp, exp_kind, exp_n_seg, exp_n_points,
                       n_seg_total, seg_t_end, seg_ovr, n_points, data_t, cotangent, n_derived,
                       solver_id, rtol, atol, ss_tol, ss_t_max, out_pred, out_grad, out_lambda0);
}

/* ------------------------------------------------------------------ */
/* Discrete adjoint gradients                                          */
/* ------------------------------------------------------------------ */
/* Exported wrappers around mxl-solvers' discrete_adjoint.c (see its header for the
 * arguments): mxn_da_forward integrates a protocol time course with RADAU5, LSODA or CVODE
 * and records it on a tape (*out_tape, an opaque handle), mxn_da_backward returns the exact
 * gradient of sum(cotangent * predictions) of that solution for any number of cotangents,
 * mxn_da_free releases the tape. Tapes are independent, but a forward solve uses the
 * solvers' global state, so only one runs at a time. */
MXN_API int mxn_da_forward(intptr_t model, intptr_t jac, intptr_t derived, int method, int n_y,
                           const double *y0, int n_pars, int n_seg, const double *seg_t_end,
                           const double *seg_pars, int n_points, const double *t_points,
                           int n_targets, const int *target_kind, const int *target_index,
                           int n_derived, double rtol, double atol, int nmax, int options,
                           double newton_scale, double max_tape_bytes, double *out_pred,
                           DaTape **out_tape) {
    return da_forward(model, jac, derived, method, n_y, y0, n_pars, n_seg, seg_t_end, seg_pars,
                      n_points, t_points, n_targets, target_kind, target_index, n_derived, rtol,
                      atol, nmax, options, newton_scale, max_tape_bytes, out_pred, out_tape);
}

MXN_API int mxn_da_backward(DaTape *tape, intptr_t adjoint, intptr_t derived_grad, int n_theta,
                            const double *cotangent, double *out_grad, double *out_lambda0) {
    return da_backward(tape, adjoint, derived_grad, n_theta, cotangent, out_grad, out_lambda0);
}

MXN_API int mxn_da_replay(const DaTape *tape, const double *seg_pars, const double *y0,
                          double *out_pred) {
    return da_replay(tape, seg_pars, y0, out_pred);
}

MXN_API void mxn_da_free(DaTape *tape) { da_free(tape); }
MXN_API int mxn_da_tape_steps(const DaTape *tape) { return da_tape_steps(tape); }
MXN_API double mxn_da_tape_bytes(const DaTape *tape) { return da_tape_bytes(tape); }

#ifdef MXN_WASM
/* Model math stays inside wasm, rather than calling back into Python. */
#define F1(name) double mxn_##name(double x) { return name(x); }
F1(exp) F1(log) F1(sin) F1(cos) F1(tan) F1(asin) F1(acos) F1(atan)
F1(sinh) F1(cosh) F1(tanh) F1(asinh) F1(acosh) F1(atanh)
double mxn_pow(double a, double b) { return pow(a, b); }
#else
/* Loaded with ctypes, but setuptools builds it as an extension and Windows linkers
 * insist on this symbol. It's never imported. */
MXN_API void PyInit__solvers(void) {}
#endif
