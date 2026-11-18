"""Opt-in linear stabilization (`stab=`) and adaptive time stepping
(solvers/adaptive.py, casekit.anneal_adaptive) on the three engines.

What is pinned here:
  * stab=0 is the default code path (same bytes, same types, no tag);
  * builder and step must receive the same S (refused both ways);
  * S leaves fixed points unchanged and, at a dt where S=0 blows up,
    S >~ ell keeps SBDF1 bounded and energy-decreasing (SBDF2 bounded);
  * variable-step SBDF2 is SBDF2 at ratio 1 and second order on a
    non-uniform step sequence;
  * the adaptive controller: error tracks err_tol on a dynamics case,
    relaxes all three engines to a residual criterion with a monotone
    energy, reproduces the engine's fixed-dt SBDF2 bit-for-bit when pinned
    to one level, restarts bit-for-bit, and refuses Langevin.
Bit-identity of stab=0 against the pre-change tree is a separate snapshot
comparison (docs/ADAPTIVE_DT.md); here we pin stab=0.0 == omitted.

Run: python3 -m pytest tests/test_adaptive.py -q
"""
import os
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy.optimize import brentq

from solvers import (adaptive, bulk3d, casekit, elastic, film3d as f3, film_elastic as fe,
                     materials, solver, timestep)

for _m in (solver, bulk3d, f3, elastic):
    _m.FFT_WORKERS = 1

DECK = materials.get("bto_lcc2005_8th", T=298.0)
ALPHA = DECK["landau"]
FDECK = materials.get("bto_lcc2005_paper", T=298.15, dx_nm=0.4)
ELL = adaptive.landau_lipschitz(ALPHA, 0.4)          # ~25.8 for this deck


def _case(kind, elastic_on=True, n2=16):
    """(engine, alpha, ops, grid, el, P0) small enough for a unit test."""
    if kind == "solver":
        grid, ops = casekit.build_2d(DECK, Nx=n2, Ny=n2, h=0.4)
        el = None
        if elastic_on:
            C, Q = DECK["elastic"], DECK["electrostriction"]
            el = elastic.setup_elastic(grid, C["C11"], C["C12"], C["C44"],
                                       Q["Q11_phys"], Q["Q12_phys"], Q["Q44_phys"])
        P0 = solver.random_smooth_field(grid, 3, amplitude=0.05, seed=1)
        return solver, ALPHA, ops, grid, el, P0
    if kind == "bulk3d":
        grid, ops, el = casekit.build_3d(DECK, N=8, d=0.5, elastic=elastic_on)
        P0 = bulk3d.random_smooth_field_3d(grid, 3, amplitude=0.05, seed=2)
        return bulk3d, ALPHA, ops, grid, el, P0
    grid = f3.make_grid_3d(Nx=10, Ny=8, Lx=4.0, Ly=3.2, Nz=7, hf=2.4)
    el = None
    if elastic_on:
        stack = fe.make_stack_grid(grid, hs_dx=3)
        c, q = FDECK["elastic"], FDECK["electrostriction"]
        el = fe.setup_film_elastic(stack, c["C11"], c["C12"], c["C44"], q["Q11_phys"],
                                   q["Q12_phys"], q["Q44_phys"], e0=-0.005)
    g2 = solver.make_grid(Nx=10, Ny=8, Lx=4.0, Ly=3.2)
    P0 = np.stack([solver.random_smooth_field(g2, 3, amplitude=0.05, seed=10 + z)
                   for z in range(7)], axis=1)
    ops = {"G11": FDECK["gradient"]["G11"], "kappa_b": 0.0885}
    return f3, FDECK["landau"], ops, grid, el, P0


ENGINES = ("solver", "bulk3d", "film3d")


# --------------------------------------------------------------------------
# Part A: linear stabilization
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ENGINES)
def test_stab_zero_is_the_default_path(kind):
    eng, alpha, ops, grid, el, P0 = _case(kind)
    a = adaptive.Engine(eng, alpha, ops, grid, elastic_params=el)          # passes no stab kw
    dt = 0.07
    for build in ("build1", "build2"):
        default = getattr(a, build)(dt)
        if kind == "film3d":
            explicit = (f3.build_grad_factors if build == "build1" else
                        f3.build_grad_factors_sbdf2)(grid, ops["G11"], dt, 1.0, stab=0.0)
            assert "stab" not in default and default.keys() == explicit.keys()
            for key, value in default.items():
                if isinstance(value, np.ndarray):
                    assert value.tobytes() == explicit[key].tobytes()
        else:
            name = {"build1": "build_Minv", "build2": "build_Minv2"}[build]
            explicit = getattr(eng, name + ("_3d" if kind == "bulk3d" else ""))(
                grid, ops, dt, 1.0, stab=0.0)
            assert type(default) is np.ndarray and type(explicit) is np.ndarray
            assert default.tobytes() == explicit.tobytes()
    op1, op2 = a.build1(dt), a.build2(dt)
    P1 = a.step1(P0, op1, dt)
    P2 = a.step2(P1, P0, op2, dt)
    if kind == "film3d":
        pf = a.pf
        P1s = f3.step(P0, alpha, grid, op1, pf, dt, 1.0, el=el, stab=0.0)
        P2s = f3.step_sbdf2(P1, P0, alpha, grid, op2, pf, dt, 1.0, el=el, stab=0.0)
    elif kind == "bulk3d":
        P1s = bulk3d.step_3d(P0, alpha, ops, op1, dt, 1.0, (0, 0, 0), elastic_params=el, stab=0.0)
        P2s = bulk3d.step_sbdf2_3d(P1, P0, alpha, ops, op2, dt, 1.0, (0, 0, 0),
                                   elastic_params=el, stab=0.0)
    else:
        P1s = solver.step(P0, alpha, ops, op1, dt, 1.0, (0, 0, 0), elastic_params=el, stab=0.0)
        P2s = solver.step_sbdf2(P1, P0, alpha, ops, op2, dt, 1.0, (0, 0, 0),
                                elastic_params=el, stab=0.0)
    assert P1.tobytes() == P1s.tobytes() and P2.tobytes() == P2s.tobytes()


@pytest.mark.parametrize("kind", ENGINES)
def test_builder_and_step_must_share_stab(kind):
    eng, alpha, ops, grid, el, P0 = _case(kind, elastic_on=False)
    dt, S = 0.05, 3.0
    plain = adaptive.Engine(eng, alpha, ops, grid)
    stab = adaptive.Engine(eng, alpha, ops, grid, stab=S)
    other = adaptive.Engine(eng, alpha, ops, grid, stab=2 * S)
    for build, step in (("build1", "step1"), ("build2", "step2")):
        op_s, op_0 = getattr(stab, build)(dt), getattr(plain, build)(dt)
        assert adaptive.operator_stab(op_s) == S and adaptive.operator_stab(op_0) == 0.0
        args = (P0,) if step == "step1" else (P0, 0.9 * P0)
        getattr(stab, step)(*args, op_s, dt)                    # matching S: fine
        for runner, op in ((plain, op_s), (stab, op_0), (other, op_s)):
            with pytest.raises(ValueError, match="stab"):
                getattr(runner, step)(*args, op, dt)
    with pytest.raises(ValueError):
        adaptive.Engine(eng, alpha, ops, grid, stab=-1.0).build1(dt)
    if kind == "film3d":
        aniso = dict(G11=ops["G11"], G12=0.3 * ops["G11"], G44=0.6 * ops["G11"],
                     G44p=0.4 * ops["G11"])
        with pytest.raises(NotImplementedError):
            f3.build_grad_factors(grid, aniso, dt, 1.0, stab=S)
        with pytest.raises(NotImplementedError):
            f3.build_grad_factors_sbdf2(grid, aniso, dt, 1.0, stab=S)


def _uniform_fixed_point(alpha):
    """Tetragonal (0,0,Ps): a uniform state is an exact fixed point of all
    three engines without elasticity (zero gradient, zero depolarization)."""
    Ps = brentq(lambda p: solver.bulk_force(np.array([[0.0], [0.0], [p]]), alpha)[2, 0],
                0.1, 0.5, xtol=1e-15)
    return Ps


@pytest.mark.parametrize("kind", ENGINES)
def test_stab_leaves_fixed_points_unchanged(kind):
    eng, alpha, ops, grid, el, P0 = _case(kind, elastic_on=False)
    P = np.zeros_like(P0)
    P[2] = _uniform_fixed_point(alpha)
    for S in (0.0, ELL):
        a = adaptive.Engine(eng, alpha, ops, grid, stab=S)
        dt = 5.0
        P1 = a.step1(P, a.build1(dt), dt)
        P2 = a.step2(P1, P, a.build2(dt), dt)
        assert np.max(np.abs(P1 - P)) < 1e-12 and np.max(np.abs(P2 - P)) < 1e-12


def test_stab_relaxes_to_the_same_state_as_s0():
    eng, alpha, ops, grid, el, P0 = _case("solver", elastic_on=True)
    P_ref, _, why, st = adaptive.integrate(eng, P0, alpha, ops, grid, dt0=0.1,
                                           elastic_params=el, n_max=4000, sample_every=10,
                                           criterion="plateau", residual_tol=1e-9, n_windows=2)
    assert why == "stationary"
    kick = 1e-3 * solver.random_smooth_field(grid, 3, amplitude=1.0, seed=7)
    P_s, _, why_s, st_s = adaptive.integrate(eng, P_ref + kick, alpha, ops, grid, dt0=0.1,
                                             elastic_params=el, stab=ELL, n_max=4000,
                                             sample_every=10, criterion="plateau",
                                             residual_tol=1e-9, n_windows=2)
    assert why_s == "stationary"
    assert np.max(np.abs(P_s - P_ref)) < 1e-6
    assert abs(st_s["E"] - st["E"]) <= 1e-10 * abs(st["E"])


def test_stab_extends_stability_2d():
    eng, alpha, ops, grid, el, P0 = _case("solver", elastic_on=False, n2=32)
    dt = 16.0                                     # SBDF1 explicit limit is ~4 here

    def run(S, scheme, n):
        a = adaptive.Engine(eng, alpha, ops, grid, stab=S)
        op1, op2 = a.build1(dt), a.build2(dt)
        P, Pm = P0.copy(), None
        E = [solver.total_energy(P, alpha, ops, (0, 0, 0), grid)[1]["total"]]
        for _ in range(n):
            new = a.step1(P, op1, dt) if (scheme == "sbdf1" or Pm is None) else a.step2(P, Pm, op2, dt)
            Pm, P = P, new
            if not np.all(np.isfinite(P)) or np.max(np.abs(P)) > 10:
                return P, np.array(E), False
            E.append(solver.total_energy(P, alpha, ops, (0, 0, 0), grid)[1]["total"])
        return P, np.array(E), True

    for scheme in ("sbdf1", "sbdf2"):
        _, _, ok = run(0.0, scheme, 40)
        assert not ok, f"{scheme} at S=0 was expected to blow up at dt={dt}"
    for S in (ELL / 2, ELL):
        P, E, ok = run(S, "sbdf1", 120)
        assert ok and np.max(np.abs(P)) < 0.5
        assert np.all(np.diff(E) <= 1e-12 * np.abs(E[1:]))   # stabilized SBDF1: monotone
    P, E, ok = run(ELL, "sbdf2", 120)
    assert ok and np.max(np.abs(P)) < 0.5 and E[-1] < E[0]


def test_landau_lipschitz_quadratic_is_exact():
    alpha = {k: 0.0 for k in ALPHA}
    alpha["a1"] = -0.7
    assert adaptive.landau_lipschitz(alpha, 0.3, n=9) == pytest.approx(1.4, rel=1e-6)


def test_timestep_passes_stab_through():
    eng, alpha, ops, grid, el, P0 = _case("solver", elastic_on=False)
    S, dt = 5.0, 0.2
    a = adaptive.Engine(eng, alpha, ops, grid, stab=S)
    for scheme in ("sbdf1", "sbdf2"):
        P, Pm = P0.copy(), None
        for _ in range(6):
            new = a.step1(P, a.build1(dt), dt) if (scheme == "sbdf1" or Pm is None) else \
                a.step2(P, Pm, a.build2(dt), dt)
            Pm, P = P, new
        Pt, _ = timestep.trajectory(eng, P0, alpha, ops, grid, dt=dt, n_steps=6, scheme=scheme,
                                    stab=S)
        assert Pt.tobytes() == P.tobytes()
    dt_v, info = timestep.verified_dt(eng, P0, alpha, ops, grid, T_total=2.0, dt0=0.4,
                                      skip_probe=True, tol=0.05, stab=S)
    assert info["stab"] == S and dt_v <= 0.4
    with pytest.raises(ValueError):
        timestep.verified_dt(eng, P0, alpha, ops, grid, T_total=2.0, run_fn=lambda d: 1.0,
                             stab=S)


# --------------------------------------------------------------------------
# Part B: variable-step SBDF2 and the adaptive controller
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ENGINES)
@pytest.mark.parametrize("S", (0.0, 4.0))
def test_vsbdf2_is_sbdf2_at_unit_ratio(kind, S):
    eng, alpha, ops, grid, el, P0 = _case(kind)
    a = adaptive.Engine(eng, alpha, ops, grid, elastic_params=el, stab=S)
    dt = 0.05
    P1 = a.step1(P0, a.build1(dt), dt)
    ref = a.step2(P1, P0, a.build2(dt), dt)
    v, P_star = a.vstep2(P1, P0, a.build1(2 * dt / 3), dt, 1.0)
    assert np.max(np.abs(v - ref)) < 1e-13
    assert np.array_equal(P_star, 2 * P1 - P0)


def test_vsbdf2_is_second_order_on_a_nonuniform_mesh():
    eng, alpha, ops, grid, el, P0 = _case("solver", elastic_on=False)
    a = adaptive.Engine(eng, alpha, ops, grid)
    T = 2.4

    def run(h, pattern):
        steps = []
        while sum(steps) < T - 1e-12:
            steps.append(h * pattern[len(steps) % len(pattern)])
        steps[-1] -= sum(steps) - T
        P, Pm, prev = a.step1(P0, a.build1(steps[0]), steps[0]), P0, steps[0]
        for dt in steps[1:]:
            w = dt / prev
            new = (a.step2(P, Pm, a.build2(dt), dt) if w == 1.0 else
                   a.vstep2(P, Pm, a.build1(dt * (1 + w) / (1 + 2 * w)), dt, w)[0])
            Pm, P, prev = P, new, dt
        return P

    ref = run(0.0025, (1.0,))
    errs = [np.max(np.abs(run(h, (1.0, 2.0, 2.0, 0.5)) - ref)) for h in (0.04, 0.02, 0.01)]
    assert errs[0] / errs[1] > 3.0 and errs[1] / errs[2] > 3.0


def test_adaptive_error_tracks_tol():
    """Dynamics case (2D, quenched field, from noise): P(t_end) and E(t)
    against a fixed small-dt SBDF2 reference; the error falls with err_tol
    and stays below it."""
    eng, alpha, ops, grid, el, P0 = _case("solver", elastic_on=False, n2=24)
    E_ext = 0.05 * solver.random_smooth_field(grid, 3, amplitude=1.0, kmax_frac=0.3, seed=5)
    P0 = 0.1 * P0
    T, dt_ref = 12.0, 0.01
    M1, M2 = solver.build_Minv(grid, ops, dt_ref, 1.0), solver.build_Minv2(grid, ops, dt_ref, 1.0)
    P, Pm, t_ref, E_ref = P0.copy(), None, [0.0], [solver.total_energy(P0, alpha, ops, E_ext, grid)[1]["total"]]
    for n in range(int(round(T / dt_ref))):
        new = (solver.step(P, alpha, ops, M1, dt_ref, 1.0, E_ext) if Pm is None else
               solver.step_sbdf2(P, Pm, alpha, ops, M2, dt_ref, 1.0, E_ext))
        Pm, P = P, new
        t_ref.append((n + 1) * dt_ref)
        E_ref.append(solver.total_energy(P, alpha, ops, E_ext, grid)[1]["total"])
    errP, errE = [], []
    for tol in (1e-2, 1e-3, 1e-4):
        Pa, trace, why, st = casekit.anneal_adaptive(
            eng, P0, alpha, ops, grid, dt0=0.2, E_ext=E_ext, err_tol=tol, n_max=10 ** 5,
            t_end=T, sample_every=1, criterion=None, energy_check="off")
        assert why == "t_end" and st["t"] == pytest.approx(T, abs=1e-12)
        ta, Ea = np.array(trace).T[:2]
        errP.append(np.max(np.abs(Pa - P)))
        errE.append(np.max(np.abs(Ea - np.interp(ta, t_ref, E_ref))))
    assert errP[0] > errP[1] > errP[2] and errE[0] > errE[1] > errE[2]
    assert errP[2] < errP[0] / 20
    assert all(e <= tol for e, tol in zip(errP, (1e-2, 1e-3, 1e-4)))


@pytest.mark.parametrize("kind", ENGINES)
def test_adaptive_relaxes_with_monotone_energy(kind):
    eng, alpha, ops, grid, el, P0 = _case(kind)
    # the film's explicit electrostatics (1/kappa_b ~ 11) caps S=0 at dt ~ 0.1
    # (> 3000 steps here); S=8 lifts that cap and relaxes in < 1000
    S = 8.0 if kind == "film3d" else 0.0
    P, trace, why, st = casekit.anneal_adaptive(
        eng, P0, alpha, ops, grid, dt0=0.05, elastic_params=el, n_max=3000, sample_every=10,
        criterion="plateau", residual_tol=1e-6, n_windows=3, stab=S)
    assert why == "stationary", (why, st["n"], st["residual"])
    assert st["residual"] <= 1e-6
    E = np.array([x[1] for x in trace])
    assert np.all(np.diff(E) <= 1e-12 * np.abs(E[1:]) + 1e-15)
    assert max(st["log_dt"]) > 0.05                 # it did leave dt0
    _check_work_identity(st)


def test_energy_check_step_mode_is_monotone_per_step():
    eng, alpha, ops, grid, el, P0 = _case("solver")
    P, trace, why, st = casekit.anneal_adaptive(
        eng, P0, alpha, ops, grid, dt0=0.4, elastic_params=el, err_tol=1e-2, n_max=300,
        sample_every=1, criterion=None, energy_check="step", k_start=2)
    E = np.array([x[1] for x in trace])
    assert np.all(np.diff(E) <= 1e-12 * np.abs(E[1:]) + 1e-15)
    _check_work_identity(st)


def _check_work_identity(st):
    assert st["attempts"] == (st["n"] + st["rejected_err"] + st["rejected_energy"]
                              + st["rejected_nonfinite"] + st["rolled_back_steps"]
                              + st["bootstrap_redos"])


@pytest.mark.parametrize("kind", ENGINES)
def test_one_level_is_the_engines_fixed_dt_sbdf2(kind):
    eng, alpha, ops, grid, el, P0 = _case(kind)
    P, trace, why, st = adaptive.integrate(eng, P0, alpha, ops, grid, dt0=0.05, elastic_params=el,
                                           k_min=0, k_max=0, err_tol=np.inf, energy_check="off",
                                           n_max=25, criterion=None, sample_every=100)
    Pt, _ = timestep.trajectory(eng, P0, alpha, ops, grid, dt=0.05, n_steps=25, scheme="sbdf2",
                                elastic_params=el)
    assert why == "n_max" and P.tobytes() == Pt.tobytes()


def test_restart_is_bit_identical():
    eng, alpha, ops, grid, el, P0 = _case("solver")
    kw = dict(dt0=0.1, elastic_params=el, sample_every=7, criterion="legacy", tol=1e-9)
    P_full, tr_full, _, st_full = casekit.anneal_adaptive(eng, P0, alpha, ops, grid, n_max=90, **kw)
    P_a, _, _, st_a = casekit.anneal_adaptive(eng, P0, alpha, ops, grid, n_max=40, **kw)
    P_b, tr_b, _, st_b = casekit.anneal_adaptive(eng, P_a, alpha, ops, grid, n_max=50,
                                                 restart=st_a, **kw)
    assert P_b.tobytes() == P_full.tobytes() and tr_b == tr_full
    assert st_b["log_dt"] == st_full["log_dt"]


def test_controller_rejects_then_recovers_and_reports_dt_min():
    eng, alpha, ops, grid, el, P0 = _case("solver", elastic_on=False)
    # start far beyond the explicit limit: rejected, halved, then converges
    P, trace, why, st = casekit.anneal_adaptive(
        eng, P0, alpha, ops, grid, dt0=0.05, k_start=6, k_max=6, n_max=3000, sample_every=10,
        criterion="plateau", residual_tol=1e-6, n_windows=3)
    assert why == "stationary" and st["rejected_err"] + st["rejected_nonfinite"] > 0
    assert np.all(np.isfinite(P))
    # a floor that cannot meet the tolerance stops cleanly
    P, trace, why, st = casekit.anneal_adaptive(
        eng, P0, alpha, ops, grid, dt0=0.5, k_min=0, k_start=0, err_tol=1e-7, n_max=100)
    assert why == "dt_min" and np.all(np.isfinite(P))


def test_anneal_adaptive_refuses_langevin_and_bad_input():
    eng, alpha, ops, grid, el, P0 = _case("solver", elastic_on=False)
    with pytest.raises(ValueError, match="Langevin"):
        casekit.anneal_adaptive(eng, P0, alpha, ops, grid, dt0=0.1, n_max=5, kT=1e-3)
    with pytest.raises(ValueError, match="Langevin"):
        casekit.anneal_adaptive(eng, P0, alpha, ops, grid, dt0=0.1, n_max=5, noise_cutoff_nm=2.0)
    from solvers import film3d_langevin
    with pytest.raises(ValueError, match="unsupported engine"):
        casekit.anneal_adaptive(film3d_langevin, P0, alpha, ops, grid, dt0=0.1, n_max=5)
    with pytest.raises(ValueError, match="block_v2"):
        casekit.anneal_adaptive(eng, P0, alpha, ops, grid, dt0=0.1, n_max=5, criterion="block_v2")


def test_block_v2_criterion_matches_anneal_semantics():
    eng, alpha, ops, grid, el, P0 = _case("solver")
    kw = dict(criterion="block_v2", energy_scale=1e-3, polarization_scale=0.26, tol=0.01,
              residual_tol=1e-5, sample_every=5, n_max=4000, elastic_params=el)
    P, trace, why, st = casekit.anneal_adaptive(eng, P0, alpha, ops, grid, dt0=0.1, **kw)
    assert why == "stationary" and st["residual"] <= 1e-5
    force = solver.total_force(P, alpha, ops, (0, 0, 0), elastic_params=el)[-1]
    assert np.max(np.abs(force)) == pytest.approx(st["residual"])
