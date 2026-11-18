#!/usr/bin/env /opt/miniconda3/bin/python3
"""Focused regression for the homogeneous-elasticity even-grid repair.

The defect: elastic.build_Omega / bulk3d.build_Omega_3d built the Green
operator Omega(k) = [C_ijkl k_j k_l]^-1 from the raw fftfreq/rfftfreq
wavevectors. On an even-length axis the Nyquist wavevectors +pi/h and -pi/h
coincide (the mode is self-conjugate), so the naive i*k first-derivative
symbol has no single well-defined sign there and the quadratic-in-k elastic
operator disagrees with the real-space energy. Independent principal repro
(bulk3d, n=4/5/6/7 cubes, dx=1, RNG37, P std 0.12, direction v, h=1e-6,
C=(275,179,54.3), Q=(.1,-.034,.029/4), stress_free): the FD mean elastic
energy vs mean(sum(elastic_force*v,axis=0)) had relative errors 0.12139594
(n=4) and 0.13563927 (n=6), while odd n=5/7 were <6e-12.

The repair: use real skew-adjoint first-derivative wavevectors for elasticity
(each component zero on its own even-axis Nyquist plane), construct BOTH
Omega and the Green operator from exactly those modified K, and regularize
every resulting zero-K mode. Odd grids are untouched bit-for-bit.

Run: /opt/miniconda3/bin/python3 -m pytest tests/test_elasticity_consistency.py -q
or:   /opt/miniconda3/bin/python3 tests/test_elasticity_consistency.py

Scope: elastic (2D, homogeneous + SPM) + bulk3d (3D homogeneous). This is a
targeted regression, NOT a CORE gate claim -- the full gates live in the
primary checkout's gates/ tree."""
import os

# Thread caps MUST precede the numpy/scipy imports (ENGINE_GUIDE 1.2).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np

from solvers import solver, bulk3d, elastic

solver.FFT_WORKERS = 1
bulk3d.FFT_WORKERS = 1
elastic.FFT_WORKERS = 1


# bulk3d FULL-TENSOR Q convention (Q1212 = 0.029/4); elastic.py's Voigt-flavoured
# Q44 = 4*Q1212 = 0.029 (SPEC S37.5's named trap).
C3 = (275.0, 179.0, 54.3)
Q3 = (0.1, -0.034, 0.029 / 4.0)
C2 = (275.0, 179.0, 54.3)
Q2 = (0.1, -0.034, 0.029)

H = 1e-6
TOL_FD = 1e-8
TOL_SPM = 1e-7


def _fd_rel_2d(el, P, v, dV):
    F, _ = elastic.elastic_force(P, el)

    def tot(Pv):
        return float(np.sum(elastic.elastic_energy_density(elastic.elastic_solve(Pv, el)))) * dV

    fd = (tot(P + H * v) - tot(P - H * v)) / (2 * H)
    fv = float(np.sum(F * v, axis=0).mean()) * (np.prod(P.shape[1:]) * dV)
    return abs(fd - fv) / max(abs(fd), abs(fv), 1e-300)


def _fd_rel_3d(el, P, v, dV):
    F, _ = bulk3d.elastic_force_3d(P, el)

    def tot(Pv):
        return float(np.sum(bulk3d.elastic_energy_density_3d(bulk3d.elastic_solve_3d(Pv, el)))) * dV

    fd = (tot(P + H * v) - tot(P - H * v)) / (2 * H)
    fv = float(np.sum(F * v, axis=0).mean()) * (np.prod(P.shape[1:]) * dV)
    return abs(fd - fv) / max(abs(fd), abs(fv), 1e-300)


def test_3d_stress_free_even_odd():
    for n in (4, 5, 6, 7):
        rng = np.random.default_rng(37)
        P = rng.standard_normal((3, n, n, n)) * 0.12
        v = rng.standard_normal((3, n, n, n))
        grid = bulk3d.make_grid_3d(n, n, n, Lx=n, Ly=n, Lz=n)
        el = bulk3d.setup_elastic_3d(grid, C3, Q3, k0="stress_free")
        dV = grid["hx"] * grid["hy"] * grid["hz"]
        rel = _fd_rel_3d(el, P, v, dV)
        assert rel < TOL_FD, f"3D stress_free n={n} relerr={rel:.3e}"


def test_3d_clamped_even_odd():
    for n in (4, 5, 6, 7):
        rng = np.random.default_rng(37)
        P = rng.standard_normal((3, n, n, n)) * 0.12
        v = rng.standard_normal((3, n, n, n))
        grid = bulk3d.make_grid_3d(n, n, n, Lx=n, Ly=n, Lz=n)
        el = bulk3d.setup_elastic_3d(grid, C3, Q3, k0="clamped")
        dV = grid["hx"] * grid["hy"] * grid["hz"]
        rel = _fd_rel_3d(el, P, v, dV)
        assert rel < TOL_FD, f"3D clamped n={n} relerr={rel:.3e}"


def test_2d_stress_free_even_odd():
    for n in (4, 5, 6, 7):
        rng = np.random.default_rng(37)
        P = rng.standard_normal((3, n, n)) * 0.12
        v = rng.standard_normal((3, n, n))
        grid = solver.make_grid(n, n, Lx=n, Ly=n)
        el = elastic.setup_elastic(grid, *C2, *Q2)
        dV = grid["hx"] * grid["hy"]
        rel = _fd_rel_2d(el, P, v, dV)
        assert rel < TOL_FD, f"2D stress_free n={n} relerr={rel:.3e}"


def test_2d_constrained_misfit_zero_even_odd():
    """elastic.py's constrained analog of 3D 'clamped': film misfit with
    eps_m=0 (ebar_11=ebar_22=0, ebar_12=0, ebar_33 from sigma_33=0)."""
    for n in (4, 5, 6, 7):
        rng = np.random.default_rng(37)
        P = rng.standard_normal((3, n, n)) * 0.12
        v = rng.standard_normal((3, n, n))
        grid = solver.make_grid(n, n, Lx=n, Ly=n)
        el = elastic.setup_elastic(grid, *C2, *Q2, eps_m=0.0)
        dV = grid["hx"] * grid["hy"]
        rel = _fd_rel_2d(el, P, v, dV)
        assert rel < TOL_FD, f"2D misfit(eps_m=0) n={n} relerr={rel:.3e}"


def test_uniform_eigenstrain_2d_3d():
    """Constant P => constant eps*: delta_eps = 0 exactly; ebar is the mean
    eigenstrain (stress_free) or 0 (3D clamped)."""
    P2 = np.full((3, 4, 4), 0.3)
    grid2 = solver.make_grid(4, 4, Lx=4.0, Ly=4.0)
    el2 = elastic.setup_elastic(grid2, *C2, *Q2)
    sol2 = elastic.elastic_solve(P2, el2)
    eps2 = elastic.eigenstrain(P2, el2["Q11"], el2["Q12"], el2["Q44"])
    assert float(np.max(np.abs(sol2["delta_eps"]))) == 0.0
    np.testing.assert_allclose(sol2["ebar"], eps2.mean(axis=(-2, -1)), rtol=1e-14, atol=1e-14)

    P3 = np.full((3, 4, 4, 4), 0.3)
    grid3 = bulk3d.make_grid_3d(4, 4, 4, Lx=4.0, Ly=4.0, Lz=4.0)
    for k0, ebar_target in (("stress_free", None), ("clamped", np.zeros((3, 3)))):
        el3 = bulk3d.setup_elastic_3d(grid3, C3, Q3, k0=k0)
        sol3 = bulk3d.elastic_solve_3d(P3, el3)
        assert float(np.max(np.abs(sol3["delta_eps"]))) == 0.0
        if ebar_target is None:
            eps3 = bulk3d.eigenstrain_3d(P3, el3["Q1111"], el3["Q1122"], el3["Q1212"])
            ebar_target = eps3.mean(axis=(-3, -2, -1))
        np.testing.assert_allclose(sol3["ebar"], ebar_target, rtol=1e-14, atol=1e-14)


def test_checkerboard_no_nan_2d_3d():
    """Pure alternating (Nyquist) modes on even grids must stay finite -- the
    zero-K regularizations must not produce NaN/inf in sigma/e/force."""
    for n in (4, 8):
        x = np.arange(n)[None, :]
        y = np.arange(n)[:, None]
        P = np.zeros((3, n, n))
        P[0] = 0.7 * (-1.0) ** x
        P[1] = 0.7 * (-1.0) ** y
        grid = solver.make_grid(n, n, Lx=n, Ly=n)
        el = elastic.setup_elastic(grid, *C2, *Q2)
        sol = elastic.elastic_solve(P, el)
        F, _ = elastic.elastic_force(P, el)
        assert np.isfinite(sol["sigma"]).all() and np.isfinite(F).all(), f"2D checkerboard n={n} NaN"

    for n in (4, 8):
        x = np.arange(n)[None, None, :]
        y = np.arange(n)[None, :, None]
        z = np.arange(n)[:, None, None]
        P = np.zeros((3, n, n, n))
        P[0] = 0.7 * (-1.0) ** x
        P[1] = 0.7 * (-1.0) ** y
        P[2] = 0.7 * (-1.0) ** z
        grid = bulk3d.make_grid_3d(n, n, n, Lx=n, Ly=n, Lz=n)
        el = bulk3d.setup_elastic_3d(grid, C3, Q3, k0="stress_free")
        sol = bulk3d.elastic_solve_3d(P, el)
        F, _ = bulk3d.elastic_force_3d(P, el)
        assert np.isfinite(sol["sigma"]).all() and np.isfinite(F).all(), f"3D checkerboard n={n} NaN"


def test_spm_2d_variational_consistency():
    """The 2D elastic SPM shares build_Omega, so the even-grid convention fix
    applies there too. Small 8x8 inhomogeneous-C(r) case, force is the
    variational derivative of the local-C elastic energy (within SPM tol)."""
    n = 8
    rng = np.random.default_rng(5)
    P = rng.standard_normal((3, n, n)) * 0.12
    v = rng.standard_normal((3, n, n))
    grid = solver.make_grid(n, n, Lx=n, Ly=n)
    yy = np.arange(n)[:, None]
    xx = np.arange(n)[None, :]
    C11 = 275.0 + 40.0 * np.sin(2 * np.pi * xx / n)
    C12 = 179.0 + 20.0 * np.cos(2 * np.pi * yy / n)
    C44 = 54.3 + 10.0 * np.sin(2 * np.pi * (xx + yy) / n)
    el = elastic.setup_elastic_spm(grid, C11, C12, C44, *Q2,
                                   spm_tol=1e-12, spm_max_iter=200,
                                   outer_tol=1e-12, outer_max_iter=100)
    dV = grid["hx"] * grid["hy"]
    rel = _fd_rel_2d(el, P, v, dV)
    assert rel < TOL_SPM, f"2D SPM n={n} relerr={rel:.3e}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    raise SystemExit(1 if failed else 0)
