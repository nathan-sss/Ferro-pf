"""solvers/minimize.py + casekit.relax / relax_film: the direct free-energy
minimizer. Exactness of energy/gradient, monotone descent, stationarity of
the result, agreement with a long fixed-dt anneal on a unique minimum, and
refusal of what it does not support. Small grids; the whole file runs in
well under a minute."""
import os
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(name, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from solvers import (bulk3d, casekit, elastic, film3d as f3, film_elastic as fe, grad4,
                     materials, minimize as mz, solver, units)

for _m in (solver, elastic, bulk3d, f3):
    _m.FFT_WORKERS = 1

DECK = materials.get("bto_lcc2005_8th", T=298.0)
ALPHA = DECK["landau"]
C, Q = DECK["elastic"], DECK["electrostriction"]
ANISO = dict(G11=0.51, G12=-0.05, G44=0.08, G44p=0.03)


def _el2(grid, **kw):
    return elastic.setup_elastic(grid, C["C11"], C["C12"], C["C44"], Q["Q11_phys"],
                                 Q["Q12_phys"], Q["Q44_phys"], **kw)


def _el3(grid, k0="stress_free"):
    return bulk3d.setup_elastic_3d(grid, C=(C["C11"], C["C12"], C["C44"]),
                                   Q=(Q["Q11_phys"], Q["Q12_phys"], Q["Q44_phys"] / 4.0), k0=k0)


def _film(Nx=8, Ny=6, Nz=5, e0=-0.01):
    deck = materials.get("bto_lcc2005_paper", T=298.15, dx_nm=0.4)
    grid = f3.make_grid_3d(Nx, Ny, Nx * 0.4, Ny * 0.4, Nz, (Nz - 1) * 0.4)
    c, q = deck["elastic"], deck["electrostriction"]
    el = fe.setup_film_elastic(fe.make_stack_grid(grid, hs_dx=3), c["C11"], c["C12"], c["C44"],
                               q["Q11_phys"], q["Q12_phys"], q["Q44_phys"], e0=e0,
                               C_substrate=materials.get("ss316_rt", T=298.15)["elastic"])
    return deck, grid, el, units.relative_to_deck_permittivity(10.0)


def _periodic_cases():
    rng = np.random.default_rng(11)
    out = []
    for Nx, Ny in ((12, 10), (11, 9)):
        grid = solver.make_grid(Nx=Nx, Ny=Ny, Lx=Nx * 0.4, Ly=Ny * 0.4)
        ops = solver.build_operators(grid, **ANISO, kappa=DECK["kappa_b"])
        E = rng.normal(0, 0.02, (3, Ny, Nx))
        out += [(f"2d{Nx}_stressfree", solver, ops, grid, E, _el2(grid)),
                (f"2d{Nx}_misfit", solver, ops, grid, (0.01, 0., 0.02), _el2(grid, eps_m=-0.005)),
                (f"2d{Nx}_filmfull", solver, ops, grid, E,
                 _el2(grid, eps_m=-0.005, film_full=True)),
                (f"2d{Nx}_H11", solver,
                 grad4.build_operators_hog(grid, -0.3, 0., -0.15, -0.15, DECK["kappa_b"], 0.4),
                 grid, (0.01, 0., 0.02), _el2(grid))]
    for N in (8, 7):
        grid = bulk3d.make_grid_3d(N, N, N, N * 0.4, N * 0.4, N * 0.4)
        ops = bulk3d.build_operators_3d(grid, **ANISO, kappa=DECK["kappa_b"])
        out += [(f"3d{N}_stressfree", bulk3d, ops, grid, (0.01, -0.02, 0.03), _el3(grid)),
                (f"3d{N}_clamped", bulk3d, ops, grid, (0.0, 0.0, 0.01), _el3(grid, "clamped")),
                (f"3d{N}_H11", bulk3d,
                 grad4.build_operators_hog_3d(grid, 0.4, 0., 0.2, 0.2, DECK["kappa_b"], 0.3),
                 grid, (0., 0., 0.), _el3(grid))]
    return out


PERIODIC = _periodic_cases()


def _fd_check(problem, P, seed, h=1e-6, n_dir=2):
    rng = np.random.default_rng(seed)
    F, g = problem.evaluate(P)
    for _ in range(n_dir):
        d = rng.normal(size=P.shape)
        fd = (problem.energy(P + h * d) - problem.energy(P - h * d)) / (2 * h)
        an = float(np.vdot(g, d))
        assert abs(fd - an) <= 1e-7 * abs(an), (fd, an)
    return F, g


@pytest.mark.parametrize("case", PERIODIC, ids=[c[0] for c in PERIODIC])
def test_periodic_energy_identity_and_fd_gradient(case):
    name, engine, ops, grid, E, el = case
    P = np.random.default_rng(len(name)).normal(0, 0.15, (3,) + tuple(
        grid[k] for k in (("Nz", "Ny", "Nx") if engine is bulk3d else ("Ny", "Nx"))))
    exact = mz.PeriodicProblem(engine, ALPHA, ops, grid, E_ext=E, elastic_params=el,
                               exact_energy=True)
    # energy() IS the engine's number; the fused evaluate repeats it bit for bit
    if engine is solver:
        fn = grad4.total_energy_hog if "H11" in ops else solver.total_energy
    else:
        fn = grad4.total_energy_hog_3d if "H11" in ops else bulk3d.total_energy_3d
    ref = fn(P, ALPHA, ops, E, grid, elastic_params=el)[1]["total"]
    F, g = _fd_check(exact, P, seed=len(name))
    assert exact.energy(P) == ref and F == ref
    np.testing.assert_array_equal(g, exact.gradient(P))
    fast = mz.PeriodicProblem(engine, ALPHA, ops, grid, E_ext=E, elastic_params=el)
    Ff, gf = fast.evaluate(P)
    assert abs(Ff - ref) <= 1e-12 * abs(ref)
    np.testing.assert_array_equal(gf, g)


FILM_CASES = [("scalar", None, False), ("scalar", 0.002, True), ("aniso", None, True),
              ("aniso", 0.002, True), ("profile", None, True)]


@pytest.mark.parametrize("shape", [(8, 6, 5), (7, 9, 6)])
@pytest.mark.parametrize("gname,H11,with_el", FILM_CASES)
def test_film_energy_identity_and_fd_gradient(shape, gname, H11, with_el):
    deck, grid, el, kappa = _film(*shape)
    G = (dict(G11=0.006, G12=0.001, G44=0.004, G44p=0.002) if gname == "aniso"
         else deck["gradient"]["G11"])
    if gname == "profile":
        kappa = np.linspace(0.08, 0.2, grid["Nz"])
    el = el if with_el else None
    E = (0.01, 0.0, -0.02)
    P = np.random.default_rng(sum(shape)).normal(0, 0.15, (3, shape[2], shape[1], shape[0]))
    prob = mz.FilmProblem(deck["landau"], grid, G, kappa, E_ext=E, el=el, H11=H11,
                          exact_energy=True)
    F, g = _fd_check(prob, P, seed=sum(shape))
    ref = f3.total_energy_3d(P, deck["landau"], grid, kappa, G, E_ext=E, el=el, H11=H11)[1]
    assert F == ref["total"] == prob.energy(P)
    np.testing.assert_array_equal(g, prob.gradient(P))
    # the nodal weight is the trapezoidal z quadrature: the force density is the
    # engine's own (1/qw)-convention force (what bto_film_biaxial logs)
    force = f3.grad_force_3d(P, grid, G, H11=H11) + f3.explicit_N(P, deck["landau"], grid,
                                                                  prob.pf, E, el=el)
    np.testing.assert_allclose(prob.force_density(g), force, rtol=1e-12, atol=1e-14)


def test_bulk_energy_fast_matches_engine():
    rng = np.random.default_rng(2)
    P = rng.uniform(-0.5, 0.5, (3, 7, 9))
    for alpha in (ALPHA, dict(a1=-0.3, a11=0.6, a12=0.15, a111=0.05, a112=0.02, a123=0.01)):
        ref = solver.bulk_energy_density(P, alpha)
        np.testing.assert_allclose(mz.bulk_energy_density_fast(P, alpha), ref, rtol=1e-12,
                                   atol=1e-15 * np.abs(ref).max())


def test_preconditioner_is_one_sbdf1_step():
    """H0 g with tau = dt is exactly the SBDF1 increment P - step(P)."""
    grid, ops = casekit.build_2d(DECK, 12, 10, 0.4)
    el = _el2(grid)
    P = solver.random_smooth_field(grid, 3, 0.1, seed=4)
    prob = mz.PeriodicProblem(solver, ALPHA, ops, grid, E_ext=(0., 0., 0.01), elastic_params=el,
                              tau=0.07)
    new = solver.step(P, ALPHA, ops, solver.build_Minv(grid, ops, 0.07, 1.0), 0.07, 1.0,
                      (0., 0., 0.01), elastic_params=el)
    np.testing.assert_allclose(prob.precondition(prob.gradient(P)), P - new, rtol=0,
                               atol=1e-12 * np.abs(P - new).max())
    deck, fgrid, fel, kappa = _film()
    FP = np.random.default_rng(5).normal(0, 0.1, (3, 5, 6, 8))
    fprob = mz.FilmProblem(deck["landau"], fgrid, deck["gradient"]["G11"], kappa, el=fel,
                           tau=0.05, precondition="gradient")
    fnew = f3.step(FP, deck["landau"], fgrid,
                   f3.build_grad_factors(fgrid, deck["gradient"]["G11"], 0.05, 1.0),
                   f3.poisson_factor(fgrid, kappa), 0.05, 1.0, el=fel)
    np.testing.assert_allclose(fprob.precondition(fprob.gradient(FP)), FP - fnew, rtol=0,
                               atol=1e-11 * np.abs(FP - fnew).max())
    # the depolarization-aware film preconditioner is symmetric positive definite
    dprob = mz.FilmProblem(deck["landau"], fgrid, deck["gradient"]["G11"], kappa, el=fel)
    u, v = np.random.default_rng(6).normal(size=(2,) + FP.shape)
    a, b = np.vdot(u, dprob.precondition(v)), np.vdot(v, dprob.precondition(u))
    assert abs(a - b) <= 1e-12 * abs(a) and np.vdot(v, dprob.precondition(v)) > 0


@pytest.mark.parametrize("method", ["lbfgs", "fire"])
def test_monotone_and_stationary_2d(method):
    grid, ops = casekit.build_2d(DECK, 24, 24, 0.4)
    el = _el2(grid)
    P0 = solver.random_smooth_field(grid, 3, 0.02, seed=3)
    P, trace, why, info = casekit.relax(solver, P0, ALPHA, ops, grid, elastic_params=el,
                                        method=method, gtol=1e-6, max_iter=3000, sample_every=1)
    assert why == "converged"
    assert np.all(np.diff(trace) <= 0.0)                     # monotone, every iteration
    assert len(trace) == info["n_iter"] + 1 and info["trace_iterations"][-1] == info["n_iter"]
    force = solver.total_force(P, ALPHA, ops, (0., 0., 0.), elastic_params=el)[-1]
    assert np.max(np.abs(force)) < 1e-6 and abs(info["max_force"] - np.abs(force).max()) < 1e-12
    assert info["energy"] == solver.total_energy(P, ALPHA, ops, (0., 0., 0.), grid,
                                                 elastic_params=el)[1]["total"]
    assert info["n_energy_evals"] == info["n_grad_evals"] >= info["n_iter"]
    assert info["wall_time"] > 0 and info["energy"] < trace[0]


def test_unique_minimum_matches_long_anneal_2d():
    """Small uniform-start tetragonal BTO (+ elastic, + a small E_z): one
    minimum, so the minimizer and a long fixed-dt anneal must agree."""
    grid, ops = casekit.build_2d(DECK, 16, 16, 0.4)
    el, E = _el2(grid), (0., 0., 0.002)
    P0 = solver.random_smooth_field(grid, 3, 0.01, seed=1)
    P0[2] += 0.2
    Pa, trace_a, _ = casekit.anneal(solver, P0, ALPHA, ops, grid, dt=0.1, E_ext=E,
                                    elastic_params=el, n_max=3000, sample_every=100, tol=-1)
    Pm, trace_m, why, info = casekit.relax(solver, P0, ALPHA, ops, grid, E_ext=E,
                                           elastic_params=el, gtol=1e-9, ftol=1e-13)
    assert why == "converged" and info["n_iter"] < 100
    assert abs(info["energy"] - trace_a[-1]) <= 1e-8 * abs(trace_a[-1])
    assert np.max(np.abs(Pm - Pa)) < 1e-6
    assert abs(Pm[2].mean() - 0.26) < 0.05 and np.abs(Pm[:2]).max() < 1e-6     # tetragonal c


def test_unique_minimum_matches_long_anneal_3d():
    grid, ops, el = casekit.build_3d(DECK, 8, 0.4, elastic=True)
    P0 = bulk3d.random_smooth_field_3d(grid, 3, 0.01, seed=2)
    P0[2] += 0.2
    # t = 400: the slowest mode here (uniform transverse P1/P2, rate ~0.04) is
    # then relaxed to ~1e-9; at t = 150 it is still 2e-5 away.
    Pa, trace_a, _ = casekit.anneal(bulk3d, P0, ALPHA, ops, grid, dt=0.2, elastic_params=el,
                                    n_max=2000, sample_every=100, tol=-1, scheme="sbdf2")
    Pm, _, why, info = casekit.relax(bulk3d, P0, ALPHA, ops, grid, elastic_params=el,
                                     gtol=1e-9, ftol=1e-13)
    assert why == "converged"
    assert abs(info["energy"] - trace_a[-1]) <= 1e-8 * abs(trace_a[-1])
    assert np.max(np.abs(Pm - Pa)) < 1e-6


def test_film_relax_stationary_and_matches_stepper():
    deck, grid, el, kappa = _film(6, 6, 5)
    alpha, G = deck["landau"], deck["gradient"]["G11"]
    P0 = np.zeros((3, 5, 6, 6))
    P0[2] = 0.2
    P0 += np.random.default_rng(3).normal(0, 0.005, P0.shape)
    for mode in ("depolarization", "gradient"):
        # |F| ~ 0.06 here: an energy-monotone search resolves forces down to ~2e-9
        P, trace, why, info = casekit.relax_film(P0, alpha, grid, G, kappa, el=el, gtol=1e-8,
                                                 ftol=1e-13, precondition=mode)
        assert why == "converged" and np.all(np.diff(trace) <= 0)
        pf = f3.poisson_factor(grid, kappa)
        force = f3.grad_force_3d(P, grid, G) + f3.explicit_N(P, alpha, grid, pf, el=el)
        assert np.max(np.abs(force)) < 1e-8
    Pa, _, n, hist = f3.relax(P0, alpha, grid, G, kappa, 1.0, dt=0.05, el=el, n_max=3000,
                              tol_F=1e-14, scheme="sbdf2")
    assert abs(info["energy"] - hist[-1][1]) <= 1e-8 * abs(hist[-1][1])
    assert np.max(np.abs(P - Pa)) < 1e-5


def test_refusals():
    grid, ops = casekit.build_2d(DECK, 8, 8, 0.4)
    P = np.zeros((3, 8, 8))
    with pytest.raises(ValueError, match="Langevin"):
        casekit.relax(solver, P, ALPHA, ops, grid, kT=0.01)
    with pytest.raises(ValueError, match="Langevin"):
        casekit.relax(solver, P, ALPHA, ops, grid, noise_cutoff_nm=2.0)
    kmap = DECK["kappa_b"] * (1 + 0.1 * np.random.default_rng(0).random((8, 8)))
    spm_ops = solver.build_operators(grid, **DECK["gradient"], kappa=kmap)
    with pytest.raises(ValueError, match="SPM kappa"):
        casekit.relax(solver, P, ALPHA, spm_ops, grid)
    el_spm = elastic.setup_elastic_spm(grid, C["C11"] * np.ones((8, 8)), C["C12"], C["C44"],
                                       Q["Q11_phys"], Q["Q12_phys"], Q["Q44_phys"])
    with pytest.raises(ValueError, match="SPM elasticity"):
        casekit.relax(solver, P, ALPHA, ops, grid, elastic_params=el_spm)
    with pytest.raises(ValueError, match="ensemble"):
        casekit.relax(solver, np.zeros((3, 2, 8, 8)), ALPHA, ops, grid)
    with pytest.raises(ValueError, match="engine"):
        casekit.relax(f3, P, ALPHA, ops, grid)
    with pytest.raises(ValueError, match="method"):
        casekit.relax(solver, P + 0.01, ALPHA, ops, grid, method="newton")
    deck, fgrid, el, kappa = _film()
    from solvers import film3d_spm
    spm_pf = film3d_spm.poisson_factor_spm(fgrid, kappa * (1 + 0.1 * np.random.default_rng(1)
                                                           .random((5, 6, 8))))
    FP = np.zeros((3, 5, 6, 8))
    with pytest.raises(ValueError, match="film3d_spm"):
        casekit.relax_film(FP, deck["landau"], fgrid, deck["gradient"]["G11"], kappa,
                           poisson_factors=spm_pf)
    with pytest.raises(ValueError, match="Langevin"):
        casekit.relax_film(FP, deck["landau"], fgrid, deck["gradient"]["G11"], kappa, kT=0.1)
    with pytest.raises(ValueError, match="positive definite"):
        casekit.relax_film(FP, deck["landau"], fgrid, -1.0, kappa, precondition="gradient")
