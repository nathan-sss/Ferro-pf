#!/usr/bin/env /opt/miniconda3/bin/python3
"""Regression tests for the elastic / solver engine review fixes
(branch fix/engine-review-2026-11, items B1-B8).

Every test here FAILED on the pre-fix tree and passes after it; each one's
docstring names the defect it pins. Small grids only -- the byte baselines
for the default (converged / non-ensemble / same-parity) paths live in the
primary checkout's gates/ tree, not here.

Run: python3 -m pytest tests/test_regress_elastic_solver.py -q"""
import os

# Thread caps MUST precede the numpy/scipy imports (ENGINE_GUIDE 1.2).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pytest

from solvers import solver, bulk3d, bulk3d_spm, elastic

solver.FFT_WORKERS = 1
bulk3d.FFT_WORKERS = 1
elastic.FFT_WORKERS = 1

# bulk3d FULL-TENSOR Q convention (Q1212 = 0.029/4); elastic.py's Voigt-flavoured
# Q44 = 4*Q1212 = 0.029 (SPEC S37.5's named trap).
C3 = (275.0, 179.0, 54.3)
Q3 = (0.1, -0.034, 0.029 / 4.0)
Q2 = (0.1, -0.034, 0.029)


# --------------------------------------------------------------------------
# B1 -- elastic SPM non-convergence was silent
# --------------------------------------------------------------------------
def _inclusion_fields_3d(n, contrast, size=2):
    """Cubic C fields on an n^3 grid: matrix C3 everywhere, one size^3
    inclusion at `contrast` x C3."""
    fields = [np.full((n, n, n), c) for c in C3]
    for f in fields:
        f[:size, :size, :size] *= contrast
    return fields


def _sin_fields_2d(n, amp):
    """Cubic C fields on an n^2 grid, C3*(1 + amp*sin(2 pi x / n)): a
    smooth field of (1-amp)..(1+amp) contrast, mean C3."""
    prof = 1.0 + amp * np.sin(2 * np.pi * np.arange(n) / n)[None, :] * np.ones((n, 1))
    return [c * prof for c in C3]


@pytest.mark.parametrize("contrast", [2.25, 10.0])
def test_spm3d_divergence_raises(contrast):
    """B1: 8^3 grid, one 2^3 stiff inclusion, default C0 = <C>. At contrast
    2.25 the inner loop ended at the cap with resid_inner ~2 and a force
    off by orders of magnitude; at 10 it went NaN (resid_inner read 0.0).
    Both came back as a normal result with resid_outer ~1e-11. Now a
    RuntimeError naming resid_inner and the C0_override remedy, and the
    garbage is NOT written to the warm-start cache. The midpoint C0 the
    message suggests converges silently on the same field."""
    n = 8
    f = _inclusion_fields_3d(n, contrast)
    grid = bulk3d.make_grid_3d(n, n, n, float(n), float(n), float(n))
    P = np.random.default_rng(3).standard_normal((3, n, n, n)) * 0.2
    el = bulk3d_spm.setup_elastic_spm_3d(grid, dict(C11=f[0], C12=f[1], C44=f[2]), Q3)
    with np.errstate(all="ignore"), pytest.raises(RuntimeError, match="resid_inner.*C0_override"):
        bulk3d.elastic_force_3d(P, el)
    assert "_warm_ebar" not in el and "_warm_delta_eps" not in el

    mid = tuple(0.5 * (c + contrast * c) for c in C3)
    el_mid = bulk3d_spm.setup_elastic_spm_3d(grid, dict(C11=f[0], C12=f[1], C44=f[2]), Q3,
                                             C0_override=mid)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _F, sol = bulk3d.elastic_force_3d(P, el_mid)
    assert sol["resid_inner"] <= 1e-10 and sol["resid_outer"] <= 1e-10


def test_spm3d_general_tensor_divergence_raises():
    """B1, the full-tensor (form (iii)) contraction path of the same solve."""
    n = 8
    f = _inclusion_fields_3d(n, 2.4)
    grid = bulk3d.make_grid_3d(n, n, n, float(n), float(n), float(n))
    C_field = bulk3d_spm.cubic_tensor_field(f[0], f[1], f[2], (n, n, n))
    el = bulk3d_spm.setup_elastic_spm_3d(grid, C_field, Q3)
    P = np.random.default_rng(4).standard_normal((3, n, n, n)) * 0.2
    with np.errstate(all="ignore"), pytest.raises(RuntimeError, match="resid_inner"):
        bulk3d_spm.elastic_solve_spm_3d(P, el, warm_start=False)


@pytest.mark.parametrize("contrast", [2.4, 50.0])
def test_spm2d_divergence_raises(contrast):
    """B1, 2D elastic.elastic_solve_spm: 8^2 grid, one 2^2 stiff inclusion
    (1/16 area), default C0 -- diverges past contrast ~2.14 (finite garbage
    at 2.4, NaN at 50), previously returned silently."""
    n = 8
    grid = solver.make_grid(n, n, float(n), float(n))
    prof = np.ones((n, n))
    prof[:2, :2] = contrast
    el = elastic.setup_elastic_spm(grid, *(c * prof for c in C3), *Q2)
    P = np.random.default_rng(5).standard_normal((3, n, n)) * 0.12
    with np.errstate(all="ignore"), pytest.raises(RuntimeError, match="resid_inner.*C0_override"):
        elastic.elastic_force(P, el)
    assert "_warm_ebar" not in el


def test_spm2d_slow_convergence_only_warns_and_reports_returned_resid():
    """B1 (b)+(c): a 98%-contrast smooth 2D field converges, just slowly --
    the outer loop exhausts outer_max_iter=50 a decade above outer_tol, and
    with spm_max_iter=10 the inner loop also stops above spm_tol. Both must
    only WARN (result returned, finite), never raise. And resid_outer must
    be that of the RETURNED sigma: it used to be the one measured before the
    last ebar update (2.24e-9 reported vs 1.76e-9 actual here)."""
    n = 16
    grid = solver.make_grid(n, n, float(n), float(n))
    P = np.random.default_rng(5).standard_normal((3, n, n)) * 0.12
    for spm_max_iter in (100, 10):
        el = elastic.setup_elastic_spm(grid, *_sin_fields_2d(n, 0.98), *Q2,
                                       spm_max_iter=spm_max_iter)
        with pytest.warns(RuntimeWarning, match="did NOT converge"):
            sol = elastic.elastic_solve(P, el)
        assert sol["n_outer"] == el["outer_max_iter"]
        assert np.all(np.isfinite(sol["sigma"]))
        sigma = sol["sigma"]
        actual = (float(np.max(np.abs(sigma.mean(axis=(-2, -1)))))
                  / max(float(np.max(np.abs(sigma))), 1e-300))
        assert sol["resid_outer"] == actual
        if spm_max_iter == 10:
            assert 1e-10 < sol["resid_inner"] < 1e-3


def test_spm3d_outer_cap_warns_and_reports_returned_resid():
    """B1 (b)+(c), 3D: stress_free with outer_max_iter=2 cannot reach
    outer_tol; it must warn, and resid_outer must be recomputed from the
    returned sigma (it was the previous ebar's)."""
    n = 8
    f = _inclusion_fields_3d(n, 1.5)
    grid = bulk3d.make_grid_3d(n, n, n, float(n), float(n), float(n))
    el = bulk3d_spm.setup_elastic_spm_3d(grid, dict(C11=f[0], C12=f[1], C44=f[2]), Q3,
                                         outer_max_iter=2)
    P = np.random.default_rng(6).standard_normal((3, n, n, n)) * 0.2
    with pytest.warns(RuntimeWarning, match="outer ebar loop"):
        sol = bulk3d_spm.elastic_solve_spm_3d(P, el, warm_start=False)
    sigma = sol["sigma"]
    actual = (float(np.max(np.abs(sigma.mean(axis=(-3, -2, -1)))))
              / max(float(np.max(np.abs(sigma))), 1e-300))
    assert sol["n_outer"] == 2 and sol["resid_outer"] == actual > 1e-10


# --------------------------------------------------------------------------
# B2 -- per-realization fields against an ensemble (S) axis
# --------------------------------------------------------------------------
def _sym(a):
    return 0.5 * (a + np.swapaxes(a, 0, 1))


def _members(fn, Pb, axis=1):
    """fn applied to each member of a batched P, restacked on the S axis."""
    return np.stack([fn(Pb[:, s]) for s in range(Pb.shape[1])], axis=axis)


@pytest.mark.parametrize("S", [1, 2, 3])
def test_ensemble_static_eigenstrain_2d(S):
    """B2, elastic.elastic_solve: a (3,3,Ny,Nx) eps_star_static against a
    batched P (3,S,Ny,Nx) was left-padded onto the component axes (S=3:
    member s got static[j,s]; S=1: force grew to (3,3,Ny,Nx); S=2:
    error). Now shared by every member, bit-for-bit the per-member result;
    a (3,3,S,Ny,Nx) static is per member; any other shape raises."""
    n = 6
    rng = np.random.default_rng(10 + S)
    grid = solver.make_grid(n, n, float(n), float(n))
    static = _sym(rng.standard_normal((3, 3, n, n))) * 0.01
    el = elastic.setup_elastic(grid, *C3, *Q2, eps_star_static=static)
    Pb = rng.standard_normal((3, S, n, n)) * 0.2
    F, _ = elastic.elastic_force(Pb, el)
    np.testing.assert_array_equal(F, _members(lambda p: elastic.elastic_force(p, el)[0], Pb))

    per = _sym(rng.standard_normal((3, 3, S, n, n))) * 0.01
    el_per = elastic.setup_elastic(grid, *C3, *Q2, eps_star_static=per)
    F, _ = elastic.elastic_force(Pb, el_per)
    ref = np.stack([elastic.elastic_force(Pb[:, s], elastic.setup_elastic(
        grid, *C3, *Q2, eps_star_static=per[:, :, s]))[0] for s in range(S)], axis=1)
    np.testing.assert_array_equal(F, ref)

    # A mis-shaped static is refused -- at setup now (solvers.validate), and
    # still by elastic._ensemble_field at solve time if it gets past setup.
    with pytest.raises(ValueError, match="eps_star_static"):
        elastic.setup_elastic(grid, *C3, *Q2, eps_star_static=static[..., 0])
    el_bad = dict(el, eps_star_static=static[..., 0])
    with pytest.raises(ValueError, match="eps_star_static"):
        elastic.elastic_force(Pb, el_bad)


@pytest.mark.parametrize("S", [1, 3])
def test_ensemble_static_eigenstrain_3d(S):
    """B2, bulk3d.elastic_solve_3d (setup forces the static to
    (3,3,Nz,Ny,Nx)): batched stress_free homogeneous-C solves are
    bit-for-bit the per-member ones, as the ensemble section documents."""
    n = 5
    rng = np.random.default_rng(20 + S)
    grid = bulk3d.make_grid_3d(n, n, n, float(n), float(n), float(n))
    static = _sym(rng.standard_normal((3, 3, n, n, n))) * 0.01
    el = bulk3d.setup_elastic_3d(grid, C3, Q3, eps_star_static=static)
    Pb = rng.standard_normal((3, S, n, n, n)) * 0.2
    F, _ = bulk3d.elastic_force_3d(Pb, el)
    np.testing.assert_array_equal(F, _members(lambda p: bulk3d.elastic_force_3d(p, el)[0], Pb))


@pytest.mark.parametrize("S", [1, 3])
def test_ensemble_field_E_ext_2d(S):
    """B2, solver._reshape_vec3: a field-shaped E_ext (3,Ny,Nx) with a
    batched P (S=3 silently wrong, S=1 force grew to (3,3,Ny,Nx)). Covers
    elec_force, the energy density and one step (via _explicit_N)."""
    n = 6
    rng = np.random.default_rng(30 + S)
    grid = solver.make_grid(n, n, float(n), float(n))
    ops = solver.build_operators(grid, 1.0, 0.0, 0.5, 0.5, 0.1)
    E = rng.standard_normal((3, n, n)) * 0.1
    Pb = rng.standard_normal((3, S, n, n)) * 0.2
    np.testing.assert_array_equal(solver.elec_force(Pb, ops, E),
                                  _members(lambda p: solver.elec_force(p, ops, E), Pb))
    np.testing.assert_array_equal(solver.elec_energy_density(Pb, ops, E),
                                  _members(lambda p: solver.elec_energy_density(p, ops, E), Pb, 0))
    alpha = dict(a1=-0.3, a11=0.6, a12=0.15, a111=0.05, a112=0.02, a123=0.01)
    Minv = solver.build_Minv(grid, ops, 0.01, 1.0)
    np.testing.assert_array_equal(
        solver.step(Pb, alpha, ops, Minv, 0.01, 1.0, E),
        _members(lambda p: solver.step(p, alpha, ops, Minv, 0.01, 1.0, E), Pb))
    Eb = rng.standard_normal((3, S, n, n)) * 0.1          # already ensemble-shaped: as given
    np.testing.assert_array_equal(solver.elec_force(Pb, ops, Eb), -solver.elec_Ed(Pb, ops) - Eb)
    with pytest.raises(ValueError, match="E_ext"):
        solver.elec_force(Pb, ops, E[:, 0])


def test_ensemble_field_E_ext_3d():
    """B2, bulk3d: field-shaped E_ext (3,Nz,Ny,Nx) with a batched P."""
    n, S = 5, 3
    rng = np.random.default_rng(40)
    grid = bulk3d.make_grid_3d(n, n, n, float(n), float(n), float(n))
    ops = bulk3d.build_operators_3d(grid, 1.0, 0.0, 0.5, 0.5, 0.1)
    E = rng.standard_normal((3, n, n, n)) * 0.1
    Pb = rng.standard_normal((3, S, n, n, n)) * 0.2
    # A shared field must broadcast on the member axis exactly. Batched
    # and individual FFTs may round differently across SciPy builds.
    force = bulk3d.elec_force_3d(Pb, ops, E)
    explicit_field = np.broadcast_to(E[:, None], Pb.shape)
    np.testing.assert_array_equal(force, bulk3d.elec_force_3d(Pb, ops, explicit_field))
    tol = 8 * np.finfo(Pb.dtype).eps
    np.testing.assert_allclose(force, _members(lambda p: bulk3d.elec_force_3d(p, ops, E), Pb),
                               rtol=tol, atol=tol)
    alpha = dict(a1=-0.3, a11=0.6, a12=0.15, a111=0.05, a112=0.02, a123=0.01)
    Minv = bulk3d.build_Minv_3d(grid, ops, 0.01, 1.0)
    np.testing.assert_allclose(
        bulk3d.step_3d(Pb, alpha, ops, Minv, 0.01, 1.0, E),
        _members(lambda p: bulk3d.step_3d(p, alpha, ops, Minv, 0.01, 1.0, E), Pb), rtol=tol, atol=tol)


# --------------------------------------------------------------------------
# B3 -- solver.spectral_resample_2d: mixed parity and the coarse Nyquist bin
# --------------------------------------------------------------------------
def _band_limited(Ny, Nx):
    """A smooth real field with mean 0.3, band-limited to |k| <= 3 (per
    axis) -- exactly representable on every grid size >= 7 used below."""
    y = np.arange(Ny)[:, None] / Ny
    x = np.arange(Nx)[None, :] / Nx
    return (0.3 + 0.5 * np.cos(2 * np.pi * x) + 0.2 * np.sin(2 * np.pi * 2 * y)
            + 0.1 * np.cos(2 * np.pi * (x + y)) - 0.07 * np.sin(2 * np.pi * (2 * x - 3 * y)))


@pytest.mark.parametrize("src,dst", [((63, 63), (128, 128)), ((128, 128), (63, 63)),
                                     ((7, 7), (8, 8)), ((8, 8), (7, 7)),
                                     ((9, 16), (16, 9))])
def test_resample_mixed_parity_exact(src, dst):
    """B3(a): the crop/pad offsets were (n_src-n)//2, (n_dst-n)//2, but
    after fftshift k=0 sits at N//2 -- off by one bin when the two sizes
    differ in parity: the mean came back wrong (0.05 for 0.3 here) and
    pointwise errors were >1. Now exact for a band-limited field."""
    out = solver.spectral_resample_2d(_band_limited(*src), *dst)
    np.testing.assert_allclose(out, _band_limited(*dst), rtol=0, atol=1e-12)


def test_resample_even_downsample_keeps_nyquist():
    """B3(b): downsampling to an even size kept only src[-n/2] of the
    +-n/2 pair, so a coarse-Nyquist mode came back HALVED: (-1)^x through
    8->16->8 returned 0.5*(-1)^x although the docstring promises an exact
    round trip. Now src[+n/2] is folded into the coarse -n/2 bin (per axis,
    and at the corner), and the round trip is exact -- also through
    upsample_state."""
    x = np.arange(8)
    fields = [np.ones((8, 1)) * (-1.0) ** x[None, :],
              (-1.0) ** (x[None, :] + x[:, None]),
              (-1.0) ** x[:, None] * np.cos(2 * np.pi * x[None, :] / 8) + _band_limited(8, 8)]
    for fld in fields:
        back = solver.spectral_resample_2d(solver.spectral_resample_2d(fld, 16, 16), 8, 8)
        np.testing.assert_allclose(back, fld, rtol=0, atol=1e-13)
    P = np.stack(fields)
    back = solver.upsample_state(solver.upsample_state(P, 16, 16), 8, 8)
    np.testing.assert_allclose(back, P, rtol=0, atol=1e-13)
    # odd -> even downsampling folds too: a 9-point sample of cos(4*2*pi*x/L)
    xs = np.arange(9)[None, :] / 9 * np.ones((9, 1))
    out = solver.spectral_resample_2d(np.cos(2 * np.pi * 4 * xs), 9, 8)
    np.testing.assert_allclose(out, np.ones((9, 1)) * (-1.0) ** x[None, :], rtol=0, atol=1e-13)


# --------------------------------------------------------------------------
# B4 -- a label missing from a per-label dict silently became Q=0 / C=0
# --------------------------------------------------------------------------
def test_per_label_dict_missing_label_raises():
    """B4: bulk3d._resolve_Q_field_3d (setup_elastic_3d) and
    bulk3d_spm._resolve_C_field_3d start their np.where chains from zeros,
    so a label present in `labels` but absent from the dict silently got
    Q=0 (no electrostriction) or C=0 (a void). Now a ValueError naming the
    missing label(s); a complete dict still resolves as before."""
    n = 4
    grid = bulk3d.make_grid_3d(n, n, n, float(n), float(n), float(n))
    labels = np.zeros((n, n, n), dtype=int)
    labels[:2] = 1
    labels[0, 0, :2] = 7
    Qd = {0: Q3, 1: (0.08, -0.03, 0.006)}
    with pytest.raises(ValueError, match=r"\[7\]"):
        bulk3d.setup_elastic_3d(grid, C3, Qd, labels=labels)
    el = bulk3d.setup_elastic_3d(grid, C3, {**Qd, 7: Q3}, labels=labels)
    assert el["Q1111"][0, 0, 0] == Q3[0] and el["Q1111"][1, 3, 3] == 0.08

    Cd = {0: elastic.build_C_tensor(*C3), 7: elastic.build_C_tensor(*C3)}
    with pytest.raises(ValueError, match=r"\[1\]"):
        bulk3d_spm.setup_elastic_spm_3d(grid, Cd, Q3, labels=labels)
    with pytest.raises(ValueError, match=r"\[7\]"):    # the SPM setup's own Q resolution
        bulk3d_spm.setup_elastic_spm_3d(grid, {**Cd, 1: Cd[0]}, Qd, labels=labels)


# --------------------------------------------------------------------------
# B5 -- film_full=True without eps_m silently ran stress-free mode (a)
# --------------------------------------------------------------------------
def test_film_full_without_eps_m_raises():
    """B5: elastic.setup_elastic(..., film_full=True) with eps_m=None was
    accepted, and elastic_solve's `eps_m is None` branch then ran mode (a)
    stress-free -- the full film constraint the caller asked for was
    silently dropped. Now ValueError at setup, and at solve time for an el
    dict that was edited after setup. With eps_m the mode (c) path runs."""
    n = 6
    grid = solver.make_grid(n, n, float(n), float(n))
    with pytest.raises(ValueError, match="film_full"):
        elastic.setup_elastic(grid, *C3, *Q2, film_full=True)
    P = np.random.default_rng(50).standard_normal((3, n, n)) * 0.2
    el = dict(elastic.setup_elastic(grid, *C3, *Q2), film_full=True)
    with pytest.raises(ValueError, match="film_full"):
        elastic.elastic_force(P, el)
    sol = elastic.elastic_solve(P, elastic.setup_elastic(grid, *C3, *Q2, eps_m=-0.003,
                                                         film_full=True))
    assert sol["ebar"][0, 0] == sol["ebar"][1, 1] == -0.003


# --------------------------------------------------------------------------
# B6 -- timestep: engine=grad4, and H11 on the film adapter
# --------------------------------------------------------------------------
_ALPHA = dict(a1=-0.3, a11=0.6, a12=0.15, a111=0.05, a112=0.02, a123=0.01)


def test_timestep_engine_grad4_runs_as_solver():
    """B6(a): the docstrings said engine=grad4 works through the default
    adapter, but grad4 has no probe_stable_dt / build_Minv2 / step_sbdf2, so
    verified_dt(grad4, ...) and trajectory(grad4, ..., scheme="sbdf2") died
    with AttributeError. grad4.build_Minv/step ARE solver's, so grad4 now
    maps to solver -- and must give exactly the engine=solver answer."""
    from solvers import grad4, timestep
    n = 8
    grid = solver.make_grid(n, n, float(n), float(n))
    ops = grad4.build_operators_hog(grid, -0.2, 0.0, -0.1, -0.1, 0.1, 0.05)
    P0 = solver.random_smooth_field(grid, 3, 0.1, seed=3)
    for scheme in ("sbdf1", "sbdf2"):
        kw = dict(T_total=0.5, scheme=scheme, n_probe=10, max_doublings=6)
        dt_g, info_g = timestep.verified_dt(grad4, P0, _ALPHA, ops, grid, **kw)
        dt_s, info_s = timestep.verified_dt(solver, P0, _ALPHA, ops, grid, **kw)
        assert dt_g == dt_s and info_g == info_s
        tg = timestep.trajectory(grad4, P0, _ALPHA, ops, grid, dt=0.01, n_steps=4, scheme=scheme)
        ts = timestep.trajectory(solver, P0, _ALPHA, ops, grid, dt=0.01, n_steps=4, scheme=scheme)
        np.testing.assert_array_equal(tg[0], ts[0])
        np.testing.assert_array_equal(tg[1], ts[1])


def test_timestep_film_adapter_passes_H11(monkeypatch):
    """B6(b): the film adapter built its grad factors (SBDF1 and SBDF2) and
    ran film3d.probe_stable_dt WITHOUT H11, so an H11 film was probed,
    stepped and certified as H11=0 (a trajectory with ops["H11"]=50 was
    bit-identical to H11=None). Now ops.get("H11") goes to all three."""
    from solvers import film3d, timestep
    monkeypatch.setattr(film3d, "FFT_WORKERS", 1)
    grid = film3d.make_grid_3d(7, 5, 4.9, 4.0, 5, 2.4)
    P0 = np.random.default_rng(41).normal(scale=0.12, size=(3, 5, 5, 7))
    ops0 = {"G11": 0.5, "kappa_b": 0.885}
    ops = dict(ops0, H11=50.0)
    dt, L = 0.01, 1.0
    pf = film3d.poisson_factor(grid, ops["kappa_b"])
    gf1 = film3d.build_grad_factors(grid, ops["G11"], dt, L, H11=50.0)
    gf2 = film3d.build_grad_factors_sbdf2(grid, ops["G11"], dt, L, H11=50.0)
    ref1 = film3d.step(P0, _ALPHA, grid, gf1, pf, dt, L)
    ref2 = film3d.step_sbdf2(ref1, P0, _ALPHA, grid, gf2, pf, dt, L)
    t1 = timestep.trajectory(film3d, P0, _ALPHA, ops, grid, dt=dt, n_steps=1)[0]
    t2 = timestep.trajectory(film3d, P0, _ALPHA, ops, grid, dt=dt, n_steps=2, scheme="sbdf2")[0]
    np.testing.assert_array_equal(t1, ref1)
    np.testing.assert_array_equal(t2, ref2)
    assert not np.array_equal(t2, timestep.trajectory(film3d, P0, _ALPHA, ops0, grid, dt=dt,
                                                      n_steps=2, scheme="sbdf2")[0])

    seen = []
    real_probe = film3d.probe_stable_dt

    def probe(*a, **kw):
        seen.append(kw.get("H11"))
        return real_probe(*a, **kw)

    monkeypatch.setattr(film3d, "probe_stable_dt", probe)
    timestep.verified_dt(film3d, P0, _ALPHA, ops, grid, T_total=0.05, n_probe=4,
                         max_doublings=3, dt0=0.005)
    assert seen == [50.0]


# --------------------------------------------------------------------------
# B7 -- grad4.build_operators_hog_3d dropped the kappa(r)-SPM knobs
# --------------------------------------------------------------------------
def test_hog_3d_passes_kappa0_override_and_spm_warm():
    """B7: build_operators_hog_3d did not accept/pass kappa0_override or
    spm_warm, so bulk3d's documented remedy for a DIVERGING electrostatic
    SPM (kappa0 near the min/max midpoint) was unreachable on the H11 lane.
    A 6^3 box with one kappa=10 plane in kappa=1 diverges at the default
    mean (2.5) and converges at the override 5.5 -- now through grad4 too,
    with ops otherwise exactly bulk3d's + H11*(kx^4+ky^4+kz^4)."""
    from solvers import grad4
    n = 6
    grid = bulk3d.make_grid_3d(n, n, n, float(n), float(n), float(n))
    kappa = np.ones((n, n, n))
    kappa[0] = 10.0
    ops = grad4.build_operators_hog_3d(grid, 0.5, 0.0, 0.25, 0.25, kappa, 0.05,
                                       kappa0_override=5.5, spm_warm=True)
    ref = bulk3d.build_operators_3d(grid, 0.5, 0.0, 0.25, 0.25, kappa,
                                    kappa0_override=5.5, spm_warm=True)
    k4 = grid["KXr"] ** 4 + grid["KYr"] ** 4 + grid["KZr"] ** 4
    assert ops["kappa"] == ops["kappa0"] == 5.5 and ops["spm_warm"] is True
    np.testing.assert_array_equal(ops["A11"], ref["A11"] + 0.05 * k4)
    np.testing.assert_array_equal(ops["dkappa"], ref["dkappa"])
    P = np.random.default_rng(2).standard_normal((3, n, n, n)) * 0.1
    _Ed, _Edu, _n, resid = bulk3d._spm_iterate_3d(P, ops)
    assert resid <= 1e-10
    bulk3d.elec_Ed_3d(P, ops)                      # spm_warm: the cache is filled
    assert ops["_spm_warm"] is not None


def _sector_explicit_N_cases():
    """The q/theta sectors' and polycrystal's own P drivers each call
    _reshape_vec3 themselves, so they need the same ensemble= flag as the
    solver/bulk3d ones (follow-up to B2)."""
    from solvers import kittel_afe, octa_tilt, polycrystal

    def sector(mod, fn):
        coupling = {k: 0.3 for k in mod.COUPLE_KEYS}
        return lambda P, aux, alpha, E: fn(P, aux, alpha, coupling, {}, E, None)

    poly = lambda P, aux, alpha, E: polycrystal._explicit_N(P, alpha, {}, E, None)
    return [("kittel_afe_2d", 2, sector(kittel_afe, kittel_afe._explicit_N_P)),
            ("octa_tilt_2d", 2, sector(octa_tilt, octa_tilt._explicit_N_P)),
            ("kittel_afe_3d", 3, sector(kittel_afe, kittel_afe._explicit_N_P_3d)),
            ("octa_tilt_3d", 3, sector(octa_tilt, octa_tilt._explicit_N_P_3d)),
            ("polycrystal_2d", 2, poly)]


@pytest.mark.parametrize("name,dim,fn", _sector_explicit_N_cases(),
                         ids=[c[0] for c in _sector_explicit_N_cases()])
@pytest.mark.parametrize("S", [1, 3])
def test_sector_explicit_N_field_E_ext_ensemble(name, dim, fn, S):
    """A field-shaped E_ext with a batched P is shared by every member in
    the sector/polycrystal P drivers too (S=3 used to hand member s the
    field's component s; S=1 grew the output's shape)."""
    from solvers import materials
    alpha = materials.get("bto_lcc2005_8th", T=298.0)["landau"]
    rng = np.random.default_rng(70 + S + dim)
    shape = (5, 4, 6)[3 - dim:]
    Pb = rng.standard_normal((3, S) + shape) * 0.2
    aux = rng.standard_normal((3, S) + shape) * 0.1
    E = rng.standard_normal((3,) + shape) * 0.05
    N = fn(Pb, aux, alpha, E)
    ref = np.stack([fn(Pb[:, s], aux[:, s], alpha, E) for s in range(S)], axis=1)
    assert N.shape == Pb.shape
    np.testing.assert_allclose(N, ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("S", [1, 2, 3])
def test_step_langevin_2d_ensemble_noise_per_member(S):
    """solver.step_langevin drew a fixed (3,Ny,Nx) noise array, which numpy
    broadcast onto the ensemble axis of a batched P (S=3: member s got noise
    component s on all three components; S=1: the output grew to
    (3,3,Ny,Nx); S=2: error). The draw now follows P's shape, as in bulk3d:
    S=1 is bit-identical to the unbatched step and members get independent
    noise."""
    from solvers import materials
    n, dt, L, kT = 8, 0.05, 1.0, 0.01
    alpha = materials.get("bto_lcc2005_8th", T=298.0)["landau"]
    grid = solver.make_grid(n, n, 0.4 * n, 0.4 * n)
    ops = solver.build_operators(grid, 0.51, -0.02, 0.02, 0.0, 0.0885)
    ou = solver.build_ou_operators(ops, dt, L, kT)
    P = np.random.default_rng(5).standard_normal((3, n, n)) * 0.05
    Pb = np.repeat(P[:, None], S, axis=1)
    out = solver.step_langevin(Pb, alpha, ops, ou, dt, L, (0.0, 0.0, 0.0),
                               np.random.default_rng(11))
    assert out.shape == Pb.shape
    if S == 1:
        ref = solver.step_langevin(P, alpha, ops, ou, dt, L, (0.0, 0.0, 0.0),
                                   np.random.default_rng(11))
        np.testing.assert_array_equal(out[:, 0], ref)
    else:
        assert not np.allclose(out[:, 0], out[:, 1])   # independent noise per member
        # From P = 0 the step is pure noise. The old broadcast gave every
        # component of member s the SAME noise (component s's), so the three
        # components of a member came out identical.
        Z = solver.step_langevin(np.zeros_like(Pb), alpha, ops, ou, dt, L, (0.0, 0.0, 0.0),
                                 np.random.default_rng(11))
        for s in range(S):
            assert not np.allclose(Z[0, s], Z[1, s])
