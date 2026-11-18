"""Regression tests for the thin-film lane (film3d / film_elastic*).

Each test reproduces a confirmed bug and fails on the code before its fix:
  * anisotropic gradient A12 not masked on even-axis Nyquist lines, so the
    implicit block inverted a term the explicit force drops (and x<->y
    transposition symmetry of the energy broke);
  * fast/numba elastic force crashed without substrate layers (hs_dx=0);
  * C_substrate on a stack without substrate layers was silently accepted;
  * a 2-node film (Nz=2) could not build its Poisson factors;
  * physical_electric_field_3d was defined twice in film3d.py.
"""
import ast
import os
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(name, '1')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pytest

from solvers import film3d as f3, film_elastic as fe, film3d_spm as fspm
from solvers.film_elastic_fast import ForceSolver

f3.FFT_WORKERS = 1

ALPHA = dict(a1=-.3, a11=.6, a12=.15, a111=.05, a112=.02, a123=.01)
G_ANISO = dict(G11=.1, G12=.05, G44=.04, G44p=.01)      # A12 != 0
ELASTIC = (275., 179., 54.3, .1, -.034, .029)


@pytest.fixture(params=['numpy', 'numba'])
def backend(request):
    if request.param == 'numpy':
        return ForceSolver
    pytest.importorskip('numba')
    from solvers.film_elastic_numba import ForceSolver as NumbaForceSolver
    return NumbaForceSolver


# --------------------------------------------------------------------------
# Anisotropic gradient: Nyquist masking of the odd x odd symbol A12.
# --------------------------------------------------------------------------
@pytest.mark.parametrize('sbdf2', [False, True])
@pytest.mark.parametrize('H11', [None, 0.02])
@pytest.mark.parametrize('nx,ny', [(8, 6), (8, 5)])
def test_aniso_grad_implicit_solve_inverts_explicit_operator(nx, ny, H11, sbdf2):
    """c0*Pn + c1*grad_force_3d(Pn) == P for Pn = implicit solve of P."""
    dt, L = 0.7, 1.3
    g = f3.make_grid_3d(nx, ny, nx * .7, ny * .8, 5, 2.4)
    P = np.random.default_rng(0).normal(size=(3, 5, ny, nx))
    if sbdf2:
        gf = f3.build_grad_factors_sbdf2(g, G_ANISO, dt, L, H11=H11)
        c0, c1 = 3.0, 2 * dt * L
    else:
        gf = f3.build_grad_factors(g, G_ANISO, dt, L, H11=H11)
        c0, c1 = 1.0, dt * L
    Pn = f3._irfft2(f3._solve_grad_factored(gf, f3._rfft2(P)), (ny, nx))
    resid = c0 * Pn + c1 * f3.grad_force_3d(Pn, g, G_ANISO, H11=H11) - P
    assert np.max(np.abs(resid)) < 1e-12


@pytest.mark.parametrize('nx,ny', [(8, 6), (8, 5), (7, 6)])
def test_aniso_grad_energy_transposition_symmetric(nx, ny):
    """Cubic G: E[P on (nx,ny)] == E[swap_xy(P) with P1<->P2 on (ny,nx)]."""
    P = np.random.default_rng(0).normal(scale=.15, size=(3, 4, ny, nx))
    PT = np.swapaxes(P, -1, -2)[[1, 0, 2]].copy()
    g = f3.make_grid_3d(nx, ny, nx * .7, ny * .8, 4, 2.4)
    gT = f3.make_grid_3d(ny, nx, ny * .8, nx * .7, 4, 2.4)
    a = f3.grad_energy_3d(P, g, G_ANISO)
    b = f3.grad_energy_3d(PT, gT, G_ANISO)
    assert abs(a - b) <= 1e-12 * abs(a)


# --------------------------------------------------------------------------
# Elasticity without substrate layers (hs_dx=0).
# --------------------------------------------------------------------------
def _no_substrate_force_vs_reference(solver_cls, static, workers):
    nx, ny, nz = 8, 6, 5
    rng = np.random.default_rng(3)
    g = f3.make_grid_3d(nx, ny, nx * .7, ny * .8, nz, 2.4)
    stack = fe.make_stack_grid(g, hs_dx=0)
    ess = None
    if static:
        ess = np.zeros((3, 3, nz, ny, nx))
        v = rng.normal(scale=1e-3, size=(nz, ny, nx))
        ess[0, 1] = ess[1, 0] = v
    el = fe.setup_film_elastic(stack, *ELASTIC, e0=-.003, eps_star_static=ess)
    P = rng.normal(scale=.15, size=(3, nz, ny, nx))
    ref, _ = fe.elastic_force_film(P, el)
    solver = solver_cls(el, workers=workers)
    try:
        out = solver(P)
    finally:
        solver.close()
    assert np.max(np.abs(out - ref)) <= 1e-12 * np.max(np.abs(ref))


@pytest.mark.parametrize('static', [False, True])
def test_fast_elastic_force_without_substrate_matches_reference(backend, static):
    _no_substrate_force_vs_reference(backend, static, workers=1)


def test_fast_elastic_force_without_substrate_threaded_slabs():
    _no_substrate_force_vs_reference(ForceSolver, static=False, workers=3)


def test_c_substrate_without_substrate_layers_rejected():
    g = f3.make_grid_3d(7, 5, 4.9, 4.0, 5, 2.4)
    stack = fe.make_stack_grid(g, hs_dx=0)
    with pytest.raises(ValueError, match='substrate layers'):
        fe.setup_film_elastic(stack, *ELASTIC, e0=-.003, C_substrate=(220., 120., 80.))


# --------------------------------------------------------------------------
# Two-node film (Nz=2): no interior node between the electrodes.
# --------------------------------------------------------------------------
def _two_node_film(nx=7, ny=6):
    g = f3.make_grid_3d(nx, ny, nx * .7, ny * .8, 2, 0.6)
    rng = np.random.default_rng(1)
    return g, rng.normal(scale=.15, size=(3, 2, ny, nx)), rng


def test_two_node_film_depolarization_field_vanishes():
    g, P, rng = _two_node_film()
    for factors in (f3.poisson_factor(g, .885),
                    f3.poisson_factor(g, np.array([.5, 1.5])),
                    fspm.poisson_factor_spm(g, 2. + .4 * rng.random(P.shape[1:]))):
        E, Eext = f3.solve_poisson_factored(P, g, factors, (0.01, -0.02, 0.05))
        assert E.shape == P.shape and not np.any(E)
        assert np.array_equal(Eext.ravel(), [0.01, -0.02, 0.05])


@pytest.mark.parametrize('G,H11', [(0.1, None), (G_ANISO, 0.02)])
def test_two_node_film_energy_force_consistent(G, H11):
    g, P, rng = _two_node_film()
    d = rng.normal(size=P.shape)
    el = fe.setup_film_elastic(fe.make_stack_grid(g, hs_dx=2), *ELASTIC, e0=-.003)
    pf = f3.poisson_factor(g, .885)
    E_ext = (0.01, -0.02, 0.05)

    def energy(p):
        return f3.total_energy_3d(p, ALPHA, g, .885, G, poisson_factors=pf, E_ext=E_ext,
                                  el=el, H11=H11)[1]['total']

    force = f3.explicit_N(P, ALPHA, g, pf, E_ext, el=el) + f3.grad_force_3d(P, g, G, H11=H11)
    W = g['qw'][None, :, None, None] * g['hx'] * g['hy']
    analytic = float(np.sum(force * d * W))
    h = 1e-5
    numeric = (energy(P + h * d) - energy(P - h * d)) / (2 * h)
    assert abs(numeric - analytic) <= 1e-7 * abs(numeric)


def test_two_node_film_relax_and_probe_run():
    g, P, _ = _two_node_film()
    el = fe.setup_film_elastic(fe.make_stack_grid(g, hs_dx=2), *ELASTIC, e0=-.003)
    dt = f3.probe_stable_dt(P, ALPHA, g, G_ANISO, .885, 1.0, n_probe=10, max_doublings=3,
                            el=el)
    assert dt > 0
    P_out, _dt, n_run, hist = f3.relax(P, ALPHA, g, G_ANISO, .885, 1.0, dt=dt, n_max=20,
                                       check_every=5, el=el)
    assert n_run == 20 and len(hist) == 4 and np.all(np.isfinite(P_out))
    assert hist[-1][1] < hist[0][1]


# --------------------------------------------------------------------------
# Hygiene.
# --------------------------------------------------------------------------
def test_film3d_has_no_duplicate_top_level_definitions():
    with open(os.path.join(_ROOT, 'fftpf', 'film_engine.py')) as fh:
        tree = ast.parse(fh.read())
    names = [node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    assert sorted({n for n in names if names.count(n) > 1}) == []
