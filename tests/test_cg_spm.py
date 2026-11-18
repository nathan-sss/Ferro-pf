#!/usr/bin/env /opt/miniconda3/bin/python3
"""Opt-in conjugate-gradient solvers for the inhomogeneous C(r) / kappa(r)
problems (docs/CG_SPM.md):

  elastic   setup_elastic_spm(..., method="cg")      2D (elastic.py)
            setup_elastic_spm_3d(..., method="cg")   3D (bulk3d_spm.py)
  electro   build_operators(..., spm_method="cg")    2D (solver.py)
            build_operators_3d(..., spm_method="cg") 3D (bulk3d.py)
            poisson_factor_spm(..., method="cg")     film (film3d_spm.py)

What is pinned here: CG reaches the basic SPM's own fixed point (same
discrete equations) to ~1e-9 or better wherever basic converges, in sigma/e/
ebar/energy/force and E_d; the FD energy-force identity holds with CG; CG
converges at contrasts 10/100/1000 where the basic scheme with its default
reference diverges; stress_free/clamped, static eigenstrain, ensemble axis,
warm starts and the iteration cap behave as documented; and a "basic" setup
dict gains no key. The default paths' bit-identity is checked against the
pre-change tree outside this suite (it needs that tree).

Run: python3 -m pytest tests/test_cg_spm.py -q"""
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

from solvers import solver, bulk3d, bulk3d_spm, elastic, film3d, film3d_spm

solver.FFT_WORKERS = 1
bulk3d.FFT_WORKERS = 1
elastic.FFT_WORKERS = 1

# bulk3d FULL-TENSOR Q convention (Q1212 = 0.029/4); elastic.py's Voigt-flavoured
# Q44 = 4*Q1212 = 0.029 (SPEC S37.5's named trap).
C3 = (275.0, 179.0, 54.3)
Q3 = (0.1, -0.034, 0.029 / 4.0)
Q2 = (0.1, -0.034, 0.029)
G = (1.0, 0.1, 0.5, 0.4)           # gradient coefficients (unused by the SPMs)
TIGHT = dict(spm_tol=1e-13, outer_tol=1e-13)


def _rel(a, b):
    return float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-300)


def _fields_2d(n, contrast, size=4):
    prof = np.ones((n, n))
    prof[:size, :size] = contrast
    return [c * prof for c in C3]


def _fields_3d(n, contrast, size=2):
    f = [np.full((n, n, n), c) for c in C3]
    for a in f:
        a[:size, :size, :size] *= contrast
    return f


def _two_phase_2d(n, contrast, seed):
    rng = np.random.default_rng(seed)
    prof = np.where(rng.random((n, n)) < 0.3, contrast, 1.0)
    return [c * prof for c in C3]


def _mid(contrast):
    return tuple(0.5 * (c + contrast * c) for c in C3)


def _sym(a):
    return 0.5 * (a + np.swapaxes(a, 0, 1))


# --------------------------------------------------------------------------
# elastic, 2D
# --------------------------------------------------------------------------
@pytest.mark.parametrize("contrast,c0", [(1.8, None), (10.0, "mid")])
def test_elastic_cg_2d_matches_basic(contrast, c0):
    """Same fixed point as the basic SPM (default C0 where it converges,
    the midpoint C0 at contrast 10): sigma, e, ebar, energy and force to
    ~1e-9, with and without the reference-medium preconditioner."""
    n = 16
    grid = solver.make_grid(n, n, float(n), float(n))
    F = _fields_2d(n, contrast)
    P = np.random.default_rng(1).standard_normal((3, n, n)) * 0.2
    elb = elastic.setup_elastic_spm(grid, *F, *Q2, C0_override=None if c0 is None else _mid(contrast),
                                    spm_max_iter=5000, outer_max_iter=500, **TIGHT)
    sb = elastic.elastic_solve_spm(P, elb, warm_start=False)
    fb = elastic.elastic_force_explicit(P, sb["sigma"], *Q2)
    eb = float(elastic.elastic_energy_density(sb).mean())
    for pre in (True, False):
        elc = elastic.setup_elastic_spm(grid, *F, *Q2, method="cg", cg_precond=pre, **TIGHT)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sc = elastic.elastic_solve_spm(P, elc, warm_start=False)
        assert sc["n_outer"] == 1 and sc["resid_inner"] <= 1e-13
        assert sc["resid_outer"] <= 1e-13
        assert set(sc) == set(sb)
        assert _rel(sc["sigma"], sb["sigma"]) < 1e-9
        assert _rel(sc["e"], sb["e"]) < 1e-9
        assert _rel(sc["ebar"], sb["ebar"]) < 1e-9
        assert _rel(elastic.elastic_force_explicit(P, sc["sigma"], *Q2), fb) < 1e-9
        assert abs(float(elastic.elastic_energy_density(sc).mean()) - eb) < 1e-9 * abs(eb)


def _fd_2d(el, P, v, h=1e-6):
    def tot(Pv):
        return float(elastic.elastic_energy_density(
            elastic.elastic_solve_spm(Pv, el, warm_start=False)).mean())
    sol = elastic.elastic_solve_spm(P, el, warm_start=False)
    force = elastic.elastic_force_explicit(P, sol["sigma"], *Q2)
    fd = (tot(P + h * v) - tot(P - h * v)) / (2 * h)
    fv = float(np.sum(force * v, axis=0).mean())
    return abs(fd - fv) / max(abs(fd), abs(fv))


@pytest.mark.parametrize("n", [10, 11])
def test_elastic_cg_2d_fd_energy_force(n):
    """Central FD of the mean elastic energy vs the envelope force
    -sigma:d(eps*)/dP with sigma from CG, contrast 50 two-phase field (the
    basic scheme diverges there at its default C0)."""
    grid = solver.make_grid(n, n, float(n), float(n))
    rng = np.random.default_rng(7)
    el = elastic.setup_elastic_spm(grid, *_two_phase_2d(n, 50.0, 3), *Q2, method="cg", **TIGHT)
    P = rng.standard_normal((3, n, n)) * 0.15
    v = rng.standard_normal((3, n, n))
    assert _fd_2d(el, P, v) < 1e-7


@pytest.mark.parametrize("contrast", [10.0, 100.0, 1000.0])
def test_elastic_cg_2d_high_contrast_converges(contrast):
    """Where basic (default C0 = <C>) diverges, CG converges silently: the
    equilibrium residual measured with the IDENTITY-reference projection
    (independent of the preconditioner) and <sigma> vanish, the plain and
    preconditioned CG agree, and the iteration count grows ~sqrt(contrast)."""
    n = 16
    grid = solver.make_grid(n, n, float(n), float(n))
    F = _two_phase_2d(n, contrast, 5)
    P = np.random.default_rng(2).standard_normal((3, n, n)) * 0.2
    sols = {}
    for pre in (True, False):
        el = elastic.setup_elastic_spm(grid, *F, *Q2, method="cg", cg_precond=pre)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sols[pre] = elastic.elastic_solve_spm(P, el, warm_start=False)
    s = sols[True]
    assert s["resid_inner"] <= 1e-10 and s["resid_outer"] <= 1e-10
    assert s["n_inner"] < 40 * np.sqrt(contrast) + 40
    elI = elastic.setup_elastic_spm(grid, *F, *Q2, method="cg", cg_precond=False)
    G_sigma = elastic._green_delta_eps(s["sigma"], elI["cg_K"], elI["cg_Omega"], elI["cg_zero_k"],
                                       (n, n), pre=elI["cg_green_pre"])
    assert float(np.max(np.abs(G_sigma))) < 1e-8 * float(np.max(np.abs(s["sigma"])))
    assert _rel(sols[False]["sigma"], s["sigma"]) < 1e-8
    if contrast == 10.0:
        elb = elastic.setup_elastic_spm(grid, *F, *Q2)
        with np.errstate(all="ignore"), pytest.raises(RuntimeError, match="DIVERGED"):
            elastic.elastic_solve_spm(P, elb, warm_start=False)


def test_elastic_cg_2d_static_ensemble_warm():
    """eps_star_static, a batched P (3,S,Ny,Nx) sharing it, and the warm
    start: every ensemble member equals its own unbatched CG solve (members
    of very different size converge independently), the result matches the
    basic SPM, and a warm re-solve at a nearby P needs fewer iterations and
    lands on the cold answer."""
    n = 16
    grid = solver.make_grid(n, n, float(n), float(n))
    rng = np.random.default_rng(9)
    F = _fields_2d(n, 1.8)
    stat = _sym(rng.standard_normal((3, 3, n, n))) * 2e-3
    elc = elastic.setup_elastic_spm(grid, *F, *Q2, eps_star_static=stat, method="cg", **TIGHT)
    elb = elastic.setup_elastic_spm(grid, *F, *Q2, eps_star_static=stat, spm_max_iter=3000,
                                    outer_max_iter=300, **TIGHT)
    PS = rng.standard_normal((3, 2, n, n)) * 0.2
    PS[:, 1] *= 1e-2
    sS = elastic.elastic_solve_spm(PS, elc, warm_start=False)
    for m in range(2):
        Pm = np.ascontiguousarray(PS[:, m])
        s1 = elastic.elastic_solve_spm(Pm, elc, warm_start=False)
        assert _rel(sS["sigma"][:, :, m], s1["sigma"]) < 1e-10
        assert _rel(sS["ebar"][:, :, m], s1["ebar"]) < 1e-10
        sb = elastic.elastic_solve_spm(Pm, elb, warm_start=False)
        assert _rel(s1["sigma"], sb["sigma"]) < 1e-9

    P = np.ascontiguousarray(PS[:, 0])
    cold = elastic.elastic_solve_spm(P, elc)                 # fills the warm cache
    P2 = P + 1e-3 * rng.standard_normal(P.shape)
    warm = elastic.elastic_solve_spm(P2, elc)
    ref = elastic.elastic_solve_spm(P2, dict(elc), warm_start=False)
    assert warm["n_inner"] < cold["n_inner"]
    assert _rel(warm["sigma"], ref["sigma"]) < 1e-10


def test_elastic_cg_setup_contract_and_cap():
    """A "basic" el gains no key; spm_max_iter=None resolves to 100 (basic)
    / CG_DEFAULT_MAX_ITER (cg); an unknown method is refused; a CG cap far
    too small to converge raises the CG-worded RuntimeError and does not
    touch the warm cache."""
    n = 8
    grid = solver.make_grid(n, n, float(n), float(n))
    F = _fields_2d(n, 100.0, size=2)
    elb = elastic.setup_elastic_spm(grid, *F, *Q2)
    assert "spm_method" not in elb and not any(k.startswith("cg_") for k in elb)
    assert elb["spm_max_iter"] == 100
    elc = elastic.setup_elastic_spm(grid, *F, *Q2, method="cg")
    assert elc["spm_max_iter"] == elastic.CG_DEFAULT_MAX_ITER and elc["spm_method"] == "cg"
    with pytest.raises(ValueError, match="method"):
        elastic.setup_elastic_spm(grid, *F, *Q2, method="pcg")
    el3 = elastic.setup_elastic_spm(grid, *F, *Q2, method="cg", spm_max_iter=2)
    P = np.random.default_rng(4).standard_normal((3, n, n)) * 0.2
    with pytest.raises(RuntimeError, match="CG solve FAILED"):
        elastic.elastic_solve_spm(P, el3)
    assert "_warm_ebar" not in el3


def test_elastic_cg_uniform_eigenstrain_is_immediate():
    """A single-domain (uniform) P in an inhomogeneous C(r): the exact
    stress-free answer is eps = eps*, sigma = 0. CG starts at <eps*> (the
    basic start), finds sigma at roundoff at once and does NOT chase the
    meaningless roundoff ratio max|<sigma>|/max|sigma| (which keeps the
    basic outer loop busy until outer_max_iter) -- no iterations, no
    warning, 2D and 3D."""
    n = 16
    grid = solver.make_grid(n, n, float(n), float(n))
    P = np.zeros((3, n, n))
    P[0], P[1], P[2] = 0.31, -0.17, 0.05
    el = elastic.setup_elastic_spm(grid, *_fields_2d(n, 1.8), *Q2, method="cg")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = elastic.elastic_solve_spm(P, el, warm_start=False)
    assert s["n_inner"] == 0 and float(np.max(np.abs(s["sigma"]))) < 1e-13
    n = 6
    grid = bulk3d.make_grid_3d(n, n, n, float(n), float(n), float(n))
    f = _fields_3d(n, 5.0)
    el3 = bulk3d_spm.setup_elastic_spm_3d(grid, dict(C11=f[0], C12=f[1], C44=f[2]), Q3,
                                          method="cg")
    P3 = np.broadcast_to(np.array([0.2, 0.1, -0.3])[:, None, None, None], (3, n, n, n)).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s3 = bulk3d_spm.elastic_solve_spm_3d(P3, el3, warm_start=False)
    assert s3["n_inner"] == 0 and float(np.max(np.abs(s3["sigma"]))) < 1e-13


# --------------------------------------------------------------------------
# elastic, 3D
# --------------------------------------------------------------------------
@pytest.mark.parametrize("form", ["cubic", "tensor"])
@pytest.mark.parametrize("k0", ["stress_free", "clamped"])
def test_elastic_cg_3d_matches_basic(form, k0):
    """8^3, one 2^3 inclusion of contrast 1.6 (basic converges at its
    default C0), plus a static eigenstrain: CG equals basic in sigma/e/ebar/
    force/energy for both C(r) input paths and both k=0 conventions;
    clamped keeps ebar = 0 exactly and only REPORTS <sigma>."""
    n = 8
    grid = bulk3d.make_grid_3d(n, n, n, float(n), float(n), float(n))
    rng = np.random.default_rng(6)
    f = _fields_3d(n, 1.6)
    Cin = dict(C11=f[0], C12=f[1], C44=f[2]) if form == "cubic" else \
        bulk3d_spm.cubic_tensor_field(*f, (n, n, n))
    stat = _sym(rng.standard_normal((3, 3, n, n, n))) * 1e-3
    P = rng.standard_normal((3, n, n, n)) * 0.2
    kw = dict(k0=k0, eps_star_static=stat, **TIGHT)
    elb = bulk3d_spm.setup_elastic_spm_3d(grid, Cin, Q3, spm_max_iter=3000, outer_max_iter=300, **kw)
    elc = bulk3d_spm.setup_elastic_spm_3d(grid, Cin, Q3, method="cg", **kw)
    sb = bulk3d_spm.elastic_solve_spm_3d(P, elb, warm_start=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sc = bulk3d_spm.elastic_solve_spm_3d(P, elc, warm_start=False)
    assert set(sc) == set(sb) and sc["n_outer"] == 1 and sc["n_inner_total"] == sc["n_inner"]
    for key in ("sigma", "e", "delta_eps"):
        assert _rel(sc[key], sb[key]) < 1e-9
    assert float(np.max(np.abs(sc["ebar"] - sb["ebar"]))) < 1e-9 * float(np.max(np.abs(sb["e"])))
    Fb = bulk3d.elastic_force_explicit_3d(P, sb["sigma"], *Q3)
    Fc = bulk3d.elastic_force_explicit_3d(P, sc["sigma"], *Q3)
    assert _rel(Fc, Fb) < 1e-9
    eb = float(elastic.elastic_energy_density(sb).mean())
    assert abs(float(elastic.elastic_energy_density(sc).mean()) - eb) < 1e-9 * abs(eb)
    if k0 == "clamped":
        assert np.all(sc["ebar"] == 0.0) and sc["resid_outer"] > 1e-3
    else:
        assert sc["resid_outer"] <= 1e-13
    # a basic el switched by hand (docs/CG_SPM.md §1) runs the same CG
    el_sw = dict(elb, spm_method="cg", spm_max_iter=1000)
    s_sw = bulk3d_spm.elastic_solve_spm_3d(P, el_sw, warm_start=False)
    assert s_sw["n_inner"] == sc["n_inner"] and _rel(s_sw["sigma"], sb["sigma"]) < 1e-9


@pytest.mark.parametrize("k0", ["stress_free", "clamped"])
def test_elastic_cg_3d_fd_energy_force(k0):
    """Central FD of the 3D mean elastic energy vs the CG-sigma force,
    contrast 100 two-phase cubic field (basic diverges at default C0)."""
    n = 6
    grid = bulk3d.make_grid_3d(n, n, n, float(n), float(n), float(n))
    rng = np.random.default_rng(8)
    prof = np.where(rng.random((n, n, n)) < 0.3, 100.0, 1.0)
    el = bulk3d_spm.setup_elastic_spm_3d(grid, dict(C11=C3[0] * prof, C12=C3[1] * prof,
                                                    C44=C3[2] * prof), Q3, k0=k0,
                                         method="cg", **TIGHT)
    P = rng.standard_normal((3, n, n, n)) * 0.15
    v = rng.standard_normal((3, n, n, n))
    h = 1e-6

    def tot(Pv):
        return float(elastic.elastic_energy_density(
            bulk3d_spm.elastic_solve_spm_3d(Pv, el, warm_start=False)).mean())
    sol = bulk3d_spm.elastic_solve_spm_3d(P, el, warm_start=False)
    force = bulk3d.elastic_force_explicit_3d(P, sol["sigma"], *Q3)
    fd = (tot(P + h * v) - tot(P - h * v)) / (2 * h)
    fv = float(np.sum(force * v, axis=0).mean())
    assert abs(fd - fv) / max(abs(fd), abs(fv)) < 1e-7


@pytest.mark.parametrize("contrast", [10.0, 1000.0])
def test_elastic_cg_3d_high_contrast_and_ensemble(contrast):
    """Stiff 2^3 inclusion at contrast 10 / 1000 in 8^3 (basic diverges at
    default C0): CG converges for both k0 conventions, and a batched P
    (3,S,...) reproduces each member's unbatched solve -- on the cubic path
    and on the full-tensor path (whose C_ref^-1 is the batched 6x6 solve)."""
    n = 8
    grid = bulk3d.make_grid_3d(n, n, n, float(n), float(n), float(n))
    rng = np.random.default_rng(10)
    f = _fields_3d(n, contrast)
    PS = rng.standard_normal((3, 2, n, n, n)) * 0.2
    for Cin in (dict(C11=f[0], C12=f[1], C44=f[2]), bulk3d_spm.cubic_tensor_field(*f, (n, n, n))):
        for k0 in ("stress_free", "clamped"):
            el = bulk3d_spm.setup_elastic_spm_3d(grid, Cin, Q3, k0=k0, method="cg")
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                sS = bulk3d_spm.elastic_solve_spm_3d(PS, el, warm_start=False)
            assert sS["resid_inner"] <= 1e-10
            for m in range(2):
                s1 = bulk3d_spm.elastic_solve_spm_3d(np.ascontiguousarray(PS[:, m]), el,
                                                     warm_start=False)
                assert _rel(sS["sigma"][:, :, m], s1["sigma"]) < 1e-8
                if k0 == "stress_free":
                    assert s1["resid_outer"] <= 1e-10


# --------------------------------------------------------------------------
# electrostatic, 2D / 3D / film
# --------------------------------------------------------------------------
def test_elec_cg_2d_matches_basic_and_dispatch():
    """Same kappa0 = (min+max)/2 (SPEC's 2D reference) for both: E_d and
    Ed_uniform agree; elec_Ed / elec_force / total_energy all route through
    the CG; a basic ops dict gains no key; an ensemble P reproduces each
    member."""
    n = 16
    grid = solver.make_grid(n, n, float(n), float(n))
    rng = np.random.default_rng(12)
    kap = np.full((n, n), 2.0)
    kap[:5, :6] = 20.0
    ob = solver.build_operators(grid, *G, kap)
    oc = solver.build_operators(grid, *G, kap, spm_method="cg")
    assert "spm_method" not in ob and oc["spm_method"] == "cg"
    assert "spm_method" not in solver.build_operators(grid, *G, 2.0, spm_method="cg")
    with pytest.raises(ValueError, match="spm_method"):
        solver.build_operators(grid, *G, kap, spm_method="fast")
    P = rng.standard_normal((3, n, n))
    Eb, Ebu, nb, _ = solver._spm_iterate(P, ob, tol=1e-14, max_iter=10000, warn_iter=10**9)
    Ec, Ecu, nc, rc = solver._spm_iterate(P, oc, tol=1e-13)
    assert rc <= 1e-13 and nc < nb
    assert _rel(Ec, Eb) < 1e-10 and np.array_equal(Ecu, Ebu) and np.all(Ec[2] == 0.0)
    assert _rel(solver.elec_Ed(P, oc, spm_tol=1e-13), Eb) < 1e-10
    E_ext = (0.1, -0.2, 0.0)
    assert _rel(solver.elec_force(P, oc, E_ext, spm_tol=1e-13),
                solver.elec_force(P, ob, E_ext, spm_tol=1e-14)) < 1e-10
    PS = rng.standard_normal((3, 2, n, n))
    PS[:, 1] *= 1e-3
    ES = solver._spm_iterate(PS, oc, tol=1e-12)[0]
    for m in range(2):
        assert _rel(ES[:, m], solver._spm_iterate(np.ascontiguousarray(PS[:, m]), oc,
                                                  tol=1e-12)[0]) < 1e-10


@pytest.mark.parametrize("contrast", [10.0, 1000.0])
def test_elec_cg_2d_fd_energy_force(contrast):
    """f_elec = -1/2 P.E_d - P.E_ext vs force -E_d - E_ext with the CG E_d
    (the P -> E_d map stays linear and self-adjoint): central FD, two-phase
    kappa at contrast 10 / 1000."""
    n = 12
    grid = solver.make_grid(n, n, float(n), float(n))
    rng = np.random.default_rng(13)
    kap = np.where(rng.random((n, n)) < 0.3, contrast, 1.0)
    oc = solver.build_operators(grid, *G, kap, spm_method="cg")
    P = rng.standard_normal((3, n, n))
    v = rng.standard_normal((3, n, n))
    E_ext = (0.3, -0.1, 0.2)
    h = 1e-5

    def tot(Pv):
        return float(solver.elec_energy_density(Pv, oc, E_ext, spm_tol=1e-13).mean())
    fd = (tot(P + h * v) - tot(P - h * v)) / (2 * h)
    fv = float(np.sum(solver.elec_force(P, oc, E_ext, spm_tol=1e-13) * v, axis=0).mean())
    assert abs(fd - fv) / max(abs(fd), abs(fv)) < 1e-8


def test_elec_cg_3d_thin_layer_and_warm():
    """bulk3d's documented divergence: one high-kappa plane (10x) in an odd
    9^3 box puts max|dkappa|/kappa0 > 1 at the default (mean) kappa0, so the
    basic SPM diverges; CG converges at that same default and matches the
    basic solve at the midpoint reference (odd grid: the discrete answer is
    kappa0-independent, docs/CG_SPM.md). spm_warm=True works with CG."""
    n = 9
    g3 = bulk3d.make_grid_3d(n, n, n, float(n), float(n), float(n))
    rng = np.random.default_rng(14)
    kap = np.ones((n, n, n))
    kap[0] = 10.0
    P = rng.standard_normal((3, n, n, n))
    ob_mean = bulk3d.build_operators_3d(g3, *G, kap)
    assert float(np.max(np.abs(ob_mean["dkappa"]))) > ob_mean["kappa0"]
    with np.errstate(all="ignore"):
        assert bulk3d._spm_iterate_3d(P, ob_mean)[3] > 1e-3      # basic: diverges (and warns)
    ob_mid = bulk3d.build_operators_3d(g3, *G, kap, kappa0_override=5.5)
    oc = bulk3d.build_operators_3d(g3, *G, kap, spm_method="cg", spm_warm=True)
    Eb = bulk3d._spm_iterate_3d(P, ob_mid, tol=1e-14, max_iter=10000, warn_iter=10**9)[0]
    Ed, _Edu, n1, r1 = bulk3d._spm_solve_3d(P, oc, tol=1e-13)
    assert r1 <= 1e-13 and _rel(Ed, Eb) < 1e-10
    P2 = P + 1e-3 * rng.standard_normal(P.shape)
    Ed2, _, n2, r2 = bulk3d._spm_solve_3d(P2, oc, tol=1e-13)          # warm from the cache
    Eref = bulk3d._spm_iterate_3d(P2, ob_mid, tol=1e-14, max_iter=10000, warn_iter=10**9)[0]
    assert n2 < n1 and _rel(Ed2, Eref) < 1e-10
    assert _rel(bulk3d.elec_Ed_3d(P2, oc, spm_tol=1e-13), Eref) < 1e-10


def test_elec_cg_3d_even_grid_same_kappa0():
    """On an even grid D(k) is not a projector on the Nyquist planes, so the
    basic discrete answer depends on kappa0 there; CG solves the SAME
    equations as basic at the SAME kappa0 (here a kappa0_override)."""
    n = 8
    g3 = bulk3d.make_grid_3d(n, n, n, float(n), float(n), float(n))
    rng = np.random.default_rng(15)
    kap = np.ones((n, n, n))
    kap[:3, :3, :3] = 4.0
    P = rng.standard_normal((3, n, n, n))
    ob = bulk3d.build_operators_3d(g3, *G, kap, kappa0_override=2.5)
    oc = bulk3d.build_operators_3d(g3, *G, kap, kappa0_override=2.5, spm_method="cg")
    Eb, Ebu, _, _ = bulk3d._spm_iterate_3d(P, ob, tol=1e-14, max_iter=10000, warn_iter=10**9)
    Ec, Ecu, _, rc = bulk3d._spm_iterate_3d(P, oc, tol=1e-13)
    assert rc <= 1e-13 and _rel(Ec, Eb) < 1e-10 and np.array_equal(Ecu, Ebu)


def test_film_cg_matches_basic_and_flags_indefinite():
    """film3d_spm with method="cg": a lateral inclusion with a z-uniform
    reference matches the basic fixed point (and runs through film3d's own
    solve_poisson_factored dispatch, warm cache included); a dead layer at
    contrast 10 makes the film's discrete operator indefinite (nodal dkappa
    against the face-harmonic kappa0(z)) -- CG refuses with a RuntimeError
    instead of returning garbage."""
    fg = film3d.make_grid_3d(12, 12, 12.0, 12.0, 7, 3.0)
    rng = np.random.default_rng(16)
    kf = np.ones((fg["Nz"], fg["Ny"], fg["Nx"]))
    kf[:, :4, :4] = 10.0
    mid = 0.5 * (kf.min(axis=(1, 2)) + kf.max(axis=(1, 2)))
    fb = film3d_spm.poisson_factor_spm(fg, kf, kappa0_profile=mid)
    fc = film3d_spm.poisson_factor_spm(fg, kf, kappa0_profile=mid, method="cg")
    assert "spm_method" not in fb and fc["spm_method"] == "cg"
    P = rng.standard_normal((3, fg["Nz"], fg["Ny"], fg["Nx"]))
    E_ext = (0.0, 0.1, 0.2)
    Eb, Xb, _, _ = film3d_spm.solve_poisson_spm(P, fg, fb, E_ext, tol=1e-14, max_iter=20000,
                                                warn_iter=10**9)
    Ec, Xc, nc, rc = film3d_spm.solve_poisson_spm(P, fg, fc, E_ext, tol=1e-13)
    assert rc <= 1e-13 and _rel(Ec, Eb) < 1e-10 and np.array_equal(Xc, Xb)
    fc["spm_tol"] = 1e-13
    E1, _ = film3d.solve_poisson_factored(P, fg, fc, E_ext)
    assert _rel(E1, Eb) < 1e-10 and fc["_warm_E"] is E1
    E2, _ = film3d.solve_poisson_factored(P, fg, fc, E_ext)             # warm: converged at once
    assert _rel(E2, Eb) < 1e-10

    # 16x16x9, 4x4 lateral inclusion, dead layer at z=0, midpoint profile:
    # the basic iteration stalls (resid ~6e-2 after 5000 steps) and CG meets
    # negative curvature. (Whether a given geometry is indefinite varies --
    # the 12x12x7 twin of this stack is positive definite and both converge.)
    fg = film3d.make_grid_3d(16, 16, 16.0, 16.0, 9, 4.0)
    kd = np.ones((fg["Nz"], fg["Ny"], fg["Nx"]))
    kd[:, :4, :4] = 10.0
    kd[0] = 0.5
    fd = film3d_spm.poisson_factor_spm(fg, kd, method="cg",
                                       kappa0_profile=0.5 * (kd.min(axis=(1, 2)) + kd.max(axis=(1, 2))))
    P = np.random.default_rng(16).standard_normal((3, fg["Nz"], fg["Ny"], fg["Nx"]))
    with pytest.raises(RuntimeError, match="not positive definite"):
        film3d_spm.solve_poisson_spm(P, fg, fd, tol=1e-12)
