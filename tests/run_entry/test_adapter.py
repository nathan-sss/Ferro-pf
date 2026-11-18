"""run_adapter (SPEC § 16.5 package E0): bit-for-bit against direct engine calls
on cases/multicore_fixture.Fixture, energy / mean_P / refusal checks."""
import os
import sys

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from cases.multicore_fixture import Fixture
from fftpf import film_engine, film_langevin, pf2d_engine, pf3d_engine, units
from fftpf.run_adapter import Adapter
from solvers import execution

ENGINES = {'pf2d': '2d', 'pf3d': 'bulk3d', 'film': 'film3d'}
DT, NSTEP, E = .001, 5, (0.01, 0., 0.02)


@pytest.fixture(autouse=True)
def _numpy_backend():
    execution.configure(1, 'numpy')
    yield
    execution.configure(workers=1, backend='numpy')


def make(engine):
    f = Fixture(ENGINES[engine], 8, dt=DT)
    if engine == 'film':
        G = f.deck['gradient']['G11']
        film = dict(G=G, kappa_b=f.kappa, poisson_factors=f.poisson)
        return f, Adapter('film', f.grid, f.alpha, elastic=f.el, film=film)
    return f, Adapter(engine, f.grid, f.alpha, ops=f.ops, elastic=f.el)


def kT_of(f):
    g = f.grid
    return units.kT_langevin(30., g['hx'], g['hy'], g.get('hz', g['hx']))


def direct(f, engine, scheme, P, rng=None):
    """NSTEP steps by calling the engine functions as test_cpu_integration does."""
    a, g, ops, el = f.alpha, f.grid, f.ops if engine != 'film' else None, f.el
    G = f.deck['gradient']['G11']
    prev = None
    if scheme == 'langevin':
        if engine == 'pf2d':
            ou = pf2d_engine.build_ou_operators(ops, DT, 1., kT_of(f))
        elif engine == 'pf3d':
            ou = pf3d_engine.build_ou_operators_3d(ops, DT, 1., kT_of(f))
        else:
            ou = film_langevin.build_ou_operators_film(g, G, DT, 1., kT_of(f))
    for _ in range(NSTEP):
        if scheme == 'langevin':
            if engine == 'pf2d':
                new = pf2d_engine.step_langevin(P, a, ops, ou, DT, 1., E, rng, elastic_params=el)
            elif engine == 'pf3d':
                new = pf3d_engine.step_langevin_3d(P, a, ops, ou, DT, 1., E, rng,
                                                   elastic_params=el)
            else:
                new = film_langevin.step_langevin_film(P, a, g, ou, f.poisson, DT, 1., E, rng,
                                                       el=el)
        elif engine == 'pf2d':
            new = (pf2d_engine.step_sbdf2(P, prev, a, ops, f.second, DT, 1., E, elastic_params=el)
                   if scheme == 'sbdf2' and prev is not None else
                   pf2d_engine.step(P, a, ops, f.first, DT, 1., E, elastic_params=el))
        elif engine == 'pf3d':
            new = (pf3d_engine.step_sbdf2_3d(P, prev, a, ops, f.second, DT, 1., E,
                                             elastic_params=el)
                   if scheme == 'sbdf2' and prev is not None else
                   pf3d_engine.step_3d(P, a, ops, f.first, DT, 1., E, elastic_params=el))
        else:
            new = (film_engine.step_sbdf2(P, prev, a, g, f.second, f.poisson, DT, 1., E, el=el)
                   if scheme == 'sbdf2' and prev is not None else
                   film_engine.step(P, a, g, f.first, f.poisson, DT, 1., E, el=el))
        prev, P = P, new
    return P


@pytest.mark.parametrize('scheme', ['sbdf1', 'sbdf2', 'langevin'])
@pytest.mark.parametrize('engine', ['pf2d', 'pf3d', 'film'])
def test_bit_for_bit(engine, scheme):
    f, ad = make(engine)
    if scheme == 'langevin':
        ad.build(DT, kT=kT_of(f))
    else:
        ad.build(DT, scheme=scheme)
    P0 = f.P.copy()
    rng = np.random.default_rng(97) if scheme == 'langevin' else None
    P, prev = P0.copy(), None
    for _ in range(NSTEP):
        new = ad.step(P, E, rng=rng, P_prev=prev)
        prev, P = P, new
    rng_d = np.random.default_rng(97) if scheme == 'langevin' else None
    ref = direct(f, engine, scheme, P0.copy(), rng_d)
    assert np.array_equal(P, ref)
    assert not np.array_equal(P, P0)
    if rng is not None:
        assert rng.bit_generator.state == rng_d.bit_generator.state


@pytest.mark.parametrize('engine', ['pf2d', 'pf3d', 'film'])
def test_energy_equals_direct(engine):
    f, ad = make(engine)
    dens, tot = ad.energy(f.P, E)
    if engine == 'pf2d':
        rd, rt = pf2d_engine.total_energy(f.P, f.alpha, f.ops, E, f.grid, elastic_params=f.el)
    elif engine == 'pf3d':
        rd, rt = pf3d_engine.total_energy_3d(f.P, f.alpha, f.ops, E, f.grid, elastic_params=f.el)
    else:
        rd, rt = film_engine.total_energy_3d(f.P, f.alpha, f.grid, f.kappa,
                                             f.deck['gradient']['G11'],
                                             poisson_factors=f.poisson, E_ext=E, el=f.el)
    assert set(dens) == set(rd) and set(tot) == set(rt)
    for k in dens:
        assert np.array_equal(dens[k], rd[k])
    for k in tot:
        assert np.array_equal(tot[k], rt[k])


def test_mean_P():
    for engine in ('pf2d', 'pf3d'):
        f, ad = make(engine)
        nsp = 2 if engine == 'pf2d' else 3
        m = ad.mean_P(f.P)
        assert m.shape == (3,)
        assert np.array_equal(m, f.P.mean(axis=tuple(range(1, 1 + nsp))))
        ens = np.stack([f.P, 2 * f.P], axis=1)         # (3,S,...)
        me = ad.mean_P(ens)
        assert me.shape == (3, 2)
        assert np.array_equal(me[:, 1], ad.mean_P(2 * f.P))
    f, ad = make('film')
    qw = f.grid['qw']
    m = ad.mean_P(f.P)
    assert m.shape == (3,)
    # independent formula: explicit trapezoid over z of the in-plane means
    z = f.grid['z']
    layer = f.P.reshape(3, f.P.shape[1], -1).mean(axis=2)
    expect = np.array([((layer[c][1:] + layer[c][:-1]) / 2 * np.diff(z)).sum()
                       for c in range(3)]) / (z[-1] - z[0])
    np.testing.assert_allclose(m, expect, rtol=1e-12, atol=1e-18)
    assert not np.allclose(m, f.P.mean(axis=(1, 2, 3)), rtol=1e-6, atol=0)  # weights matter
    uniform = np.empty_like(f.P)
    for c, v in enumerate((0.11, -0.07, 0.23)):
        uniform[c] = v
    np.testing.assert_allclose(ad.mean_P(uniform), [0.11, -0.07, 0.23], rtol=1e-14, atol=0)
    assert np.isclose(qw.sum(), f.grid['hf'])


def test_random_start():
    f, ad = make('pf2d')
    assert np.array_equal(ad.random_start(.005, .5, 3),
                          pf2d_engine.random_smooth_field(f.grid, 3, .005, .5, 3))
    f, ad = make('pf3d')
    assert np.array_equal(ad.random_start(.005, .5, 3),
                          pf3d_engine.random_smooth_field_3d(f.grid, 3, .005, .5, 3))
    f, ad = make('film')
    P = ad.random_start(.005, .5, 3)
    assert P.shape == f.P.shape
    assert np.array_equal(P[:, 4], pf2d_engine.random_smooth_field(
        f.grid, 3, amplitude=.005, kmax_frac=.5, seed=3 * 100000 + 4))


def test_refusals():
    f, ad = make('pf2d')
    with pytest.raises(ValueError, match='first order'):
        ad.build(DT, scheme='sbdf2', kT=kT_of(f))
    with pytest.raises(ValueError, match='scheme'):
        ad.build(DT, scheme='rk4')
    with pytest.raises(ValueError, match='noise_cutoff_nm'):
        ad.build(DT, noise_cutoff_nm=2.)
    ad.build(DT)
    with pytest.raises(ValueError, match='kT=None'):
        ad.step(f.P, E, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match='engine'):
        Adapter('pf4d', f.grid, f.alpha, ops=f.ops)
    with pytest.raises(ValueError, match='film'):
        Adapter('pf2d', f.grid, f.alpha, ops=f.ops, film=dict(G=1., kappa_b=1.))
    with pytest.raises(ValueError, match='ops'):
        Adapter('pf2d', f.grid, f.alpha)
    with pytest.raises(ValueError, match='stab'):
        Adapter('pf2d', f.grid, f.alpha, ops=f.ops, stab=-1.)
    st = Adapter('pf2d', f.grid, f.alpha, ops=f.ops, stab=.1)
    with pytest.raises(ValueError, match='stab'):
        st.build(DT, kT=kT_of(f))
    with pytest.raises(ValueError, match='before|first'):
        Adapter('pf2d', f.grid, f.alpha, ops=f.ops).step(f.P, E)
    f, ad = make('film')
    ad.build(DT)
    ens = np.stack([f.P, f.P], axis=1)
    with pytest.raises(ValueError, match='ensemble'):
        ad.step(ens, E)
    with pytest.raises(ValueError, match='ensemble'):
        ad.mean_P(ens)
    # film Langevin: the engine's own refusal (G11 < 0) propagates
    with pytest.raises(ValueError, match='G11'):
        Adapter('film', f.grid, f.alpha, film=dict(G=-1., kappa_b=f.kappa)).build(DT, kT=1e-3)


def test_stab_passthrough():
    f = Fixture('2d', 8, dt=DT)
    ad = Adapter('pf2d', f.grid, f.alpha, ops=f.ops, elastic=f.el, stab=.5)
    ad.build(DT)
    op = pf2d_engine.build_Minv(f.grid, f.ops, DT, 1., stab=.5)
    ref = pf2d_engine.step(f.P, f.alpha, f.ops, op, DT, 1., E, elastic_params=f.el, stab=.5)
    assert np.array_equal(ad.step(f.P, E), ref)
