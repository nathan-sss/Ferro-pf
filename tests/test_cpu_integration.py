"""Reference/CPU trajectories and checkpoint continuation, without timing."""
import os
import sys

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

pytest.importorskip('numba')

from cases.multicore_fixture import Fixture
from solvers import bulk3d, execution, film3d, film3d_langevin, io_utils, solver, units


def step(fixture, P, rng, ou):
    f = fixture
    if f.engine == 'film3d':
        if ou is None:
            return film3d.step(P, f.alpha, f.grid, f.first, f.poisson,
                               f.dt, 1., el=f.el)
        return film3d_langevin.step_langevin_film(P, f.alpha, f.grid, ou,
            f.poisson, f.dt, 1., (0., 0., 0.), rng, el=f.el)
    if f.engine == '2d':
        deterministic, stochastic = solver.step, solver.step_langevin
    else:
        deterministic, stochastic = bulk3d.step_3d, bulk3d.step_langevin_3d
    if ou is None:
        return deterministic(P, f.alpha, f.ops, f.first, f.dt, 1.,
                             (0., 0., 0.), elastic_params=f.el)
    return stochastic(P, f.alpha, f.ops, ou, f.dt, 1.,
                      (0., 0., 0.), rng, elastic_params=f.el)


@pytest.mark.parametrize('engine', ['2d', 'bulk3d', 'film3d'])
@pytest.mark.parametrize('scheme', ['sbdf1', 'langevin'])
def test_trajectory_and_restart(engine, scheme, tmp_path):
    execution.configure(1, 'numpy')
    fixture = Fixture(engine, 8, dt=.001)
    # Exercise broadcast material fields in the actual timesteppers.
    profile = 1. + .1*np.sin(2*np.pi*np.arange(8)/8)
    fixture.alpha = dict(fixture.alpha, a1=fixture.alpha['a1']*profile,
                         a11=np.full(fixture.P.shape[1:], fixture.alpha['a11']))
    ou = None
    if scheme == 'langevin':
        g = fixture.grid
        kT = units.kT_langevin(30., g['hx'], g['hy'], g.get('hz', g['hx']))
        if engine == 'film3d':
            ou = film3d_langevin.build_ou_operators_film(g,
                fixture.deck['gradient']['G11'], fixture.dt, 1., kT)
        else:
            build = solver.build_ou_operators if engine == '2d' else bulk3d.build_ou_operators_3d
            ou = build(fixture.ops, fixture.dt, 1., kT)
    initial = fixture.P.copy()
    path = tmp_path/'restart.npz'
    try:
        P, rng = initial.copy(), np.random.default_rng(2468)
        reference = []
        for _ in range(5):
            P = step(fixture, P, rng, ou)
            reference.append(P.copy())
        reference_rng = rng.bit_generator.state

        execution.configure(1, 'numba', el=fixture.el if engine == 'film3d' else None)
        P, rng = initial.copy(), np.random.default_rng(2468)
        for index in range(5):
            P = step(fixture, P, rng, ou)
            np.testing.assert_allclose(P, reference[index], rtol=1e-11, atol=1e-12)
            if index == 1:
                io_utils.save_state_npz(path, P, fixture.grid, 2, 2*fixture.dt,
                    alpha=fixture.alpha, restart=dict(scheme=scheme,
                    dt=fixture.dt, rng_state=rng.bit_generator.state))
        uninterrupted = P.copy()
        final_rng = rng.bit_generator.state
        assert final_rng == reference_rng

        # Restore both material fields and the generator through the public I/O.
        checkpoint = io_utils.load_state_npz(path)
        P = checkpoint['P']
        fixture.alpha = checkpoint['checkpoint']['alpha']
        rng = np.random.default_rng()
        rng.bit_generator.state = checkpoint['checkpoint']['restart']['rng_state']
        for _ in range(3):
            P = step(fixture, P, rng, ou)
        np.testing.assert_array_equal(P, uninterrupted)
        assert rng.bit_generator.state == final_rng
    finally:
        fixture.close()
