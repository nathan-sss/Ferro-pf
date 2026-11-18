"""Accelerated film forces must use the current coupled elastic source."""
import os
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(name, '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from solvers import film3d as f3, film_elastic as fe, kittel_afe, octa_tilt
from solvers.film_elastic_fast import ForceSolver

f3.FFT_WORKERS = 1


@pytest.fixture(params=['numpy', 'numba'])
def backend(request):
    if request.param == 'numpy':
        return ForceSolver
    pytest.importorskip('numba')
    from solvers.film_elastic_numba import ForceSolver as NumbaForceSolver
    return NumbaForceSolver


def setup(nx=7, ny=5):
    grid = f3.make_grid_3d(nx, ny, nx*0.7, ny*0.8, 5, 2.4)
    stack = fe.make_stack_grid(grid, hs_dx=2)
    el = fe.setup_film_elastic(stack, 275., 179., 54.3, .1, -.034, .029,
                               e0=-.003)
    P = np.random.default_rng(41).normal(scale=.12, size=(3, 5, ny, nx))
    return grid, el, P


COUPLED = [kittel_afe.with_q_eigenstrain_film,
           octa_tilt.with_theta_eigenstrain_film]


@pytest.mark.parametrize('coupled', COUPLED)
@pytest.mark.parametrize('nx,ny', [(7, 5), (8, 6)])
def test_current_coupled_source(backend, coupled, nx, ny):
    grid, el, P = setup(nx, ny)
    accelerator = backend(el, workers=1)
    el['_force_solver'] = accelerator
    field = np.random.default_rng(7).normal(scale=.2, size=P.shape)
    lam = dict(Lam11=.08, Lam12=-.02, Lam44=.04)
    alpha = dict(a1=-.3, a11=.6, a12=.15, a111=.05, a112=.02, a123=.01)
    poisson = f3.poisson_factor(grid, .885)
    try:
        bare = accelerator(P)
        for scale in (1., .5, 0.):
            current = coupled(el, scale*field, lam)
            expected, _ = fe.elastic_force_film(P, current)
            actual = accelerator(P, el=current)
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-12)
            if scale == 1.:
                assert np.max(np.abs(actual-bare)) > 1e-4
            # Exercise the production dispatcher with the shallow-copied owner.
            reference = dict(current)
            reference.pop('_force_solver')
            np.testing.assert_allclose(
                f3.explicit_N(P, alpha, grid, poisson, el=current),
                f3.explicit_N(P, alpha, grid, poisson, el=reference),
                rtol=2e-12, atol=1e-12)
    finally:
        accelerator.close()


@pytest.mark.parametrize('coupled', COUPLED)
def test_current_source_energy_derivative(backend, coupled):
    grid, el, P = setup()
    rng = np.random.default_rng(310)
    field = rng.normal(scale=.2, size=P.shape)
    current = coupled(el, field, dict(Lam11=.08, Lam12=-.02, Lam44=.04))
    # Q and imposed strain are source data, independent of factorizations.
    current['Q11'] *= 1.3
    current['ebar'] = current['ebar'] + np.diag([.001, -.002, .0005])
    direction = rng.normal(size=P.shape)
    volume = grid['hx']*grid['hy']*grid['hz']

    def energy(value):
        solution = fe.elastic_solve_film(value, current)
        # Include substrate energy when differentiating relaxed equilibrium.
        return .5*volume*np.sum(solution['sigma_face']*solution['e_face'])

    accelerator = backend(el, workers=1)
    try:
        force = accelerator(P, el=current)
        expected, _ = fe.elastic_force_film(P, current)
        np.testing.assert_allclose(force, expected, rtol=2e-12, atol=1e-12)
        h = 1e-6
        finite_difference = (energy(P+h*direction)-energy(P-h*direction))/(2*h)
        derivative = volume*np.sum(force*direction)
        np.testing.assert_allclose(derivative, finite_difference, rtol=1e-7, atol=1e-9)
    finally:
        accelerator.close()


@pytest.mark.parametrize('layout', ['readonly', 'fortran', 'strided'])
def test_input_and_output_ownership(backend, layout):
    _, el, P = setup()
    if layout == 'fortran':
        P = np.asfortranarray(P)
    elif layout == 'strided':
        storage = np.zeros(P.shape[:-1]+(2*P.shape[-1],))
        storage[..., ::2] = P
        P = storage[..., ::2]
    else:
        P.flags.writeable = False
    before = P.copy()
    accelerator = backend(el, workers=1)
    try:
        first = accelerator(P)
        retained = first.copy()
        second = accelerator(P*.7)
        expected, _ = fe.elastic_force_film(P, el)
        np.testing.assert_allclose(first, expected, rtol=2e-12, atol=1e-12)
        np.testing.assert_array_equal(first, retained)
        np.testing.assert_array_equal(P, before)
        assert not np.shares_memory(first, second)
        assert not np.shares_memory(first, P)
    finally:
        accelerator.close()


def test_incompatible_stack_and_source(backend):
    _, el, P = setup()
    _, other, _ = setup()
    accelerator = backend(el, workers=1)
    try:
        with pytest.raises(ValueError, match='stack changes'):
            accelerator(P, el=other)
        with pytest.raises(ValueError, match='homogeneous-layer'):
            accelerator(P, el=dict(el, spm=True))
        static = np.zeros((3, 3, el['Nz'], el['Ny'], el['Nx']))
        static[0, 1, el['film_lo']:] = .001
        with pytest.raises(ValueError, match='symmetric'):
            accelerator(P, el=dict(el, eps_star_static=static))
        static.fill(0.)
        static[0, 0, 0] = .001
        with pytest.raises(ValueError, match='zero in the substrate'):
            accelerator(P, el=dict(el, eps_star_static=static))
        with pytest.raises(ValueError, match='Polarization shape'):
            accelerator(P[:, :-1])
    finally:
        accelerator.close()


@pytest.mark.parametrize('fail', [False, True])
def test_numba_restores_thread_budget(monkeypatch, fail):
    pytest.importorskip('numba')
    from solvers import film_elastic_numba as compiled
    _, el, P = setup()
    accelerator = compiled.ForceSolver(el, workers=1)
    # Emulate a different caller budget without requesting extra CPU cores.
    settings = [4]
    monkeypatch.setattr(compiled.numba, 'get_num_threads', lambda: settings[-1])
    monkeypatch.setattr(compiled.numba, 'set_num_threads', settings.append)
    try:
        if fail:
            with pytest.raises(ValueError, match='Polarization shape'):
                accelerator(P[:, :-1])
        else:
            accelerator(P)
        assert settings == [4, 1, 4]
        with pytest.raises(ValueError, match='float64'):
            accelerator(P.astype(np.float32))
    finally:
        accelerator.close()
