"""Opt-in fused step kernels (execution.configure(..., fused=True)) against the
NumPy reference: per-step agreement to round-off, coverage fallbacks, scratch
isolation and restart. No timing."""
import os
import sys

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

pytest.importorskip('numba')

from solvers import bulk3d, casekit, elastic, execution, io_utils, materials, solver, units

RTOL = 1e-12


@pytest.fixture(autouse=True)
def _reference_afterwards():
    yield
    execution.configure(1, 'numpy')


def _deck():
    return materials.get('bto_lcc2005_8th', T=298.15)


def _problem(engine, shape, *, elastic_mode='stress_free', static=False, ensemble=0, dt=0.004):
    deck = _deck()
    c, q = deck['elastic'], deck['electrostriction']
    rng = np.random.default_rng(97)
    if engine == '2d':
        Ny, Nx = shape
        grid, ops = casekit.build_2d(deck, Nx, Ny, .4)
        el = None
        if elastic_mode is not None:
            eps_m = -0.002 if elastic_mode == 'misfit' else None
            st = None
            if static:
                st = rng.normal(size=(3, 3, Ny, Nx)) * 1e-4
                st = .5 * (st + st.swapaxes(0, 1))
            el = elastic.setup_elastic(grid, c['C11'], c['C12'], c['C44'], q['Q11_phys'],
                                       q['Q12_phys'], q['Q44_phys'], eps_star_static=st, eps_m=eps_m)
        first = solver.build_Minv(grid, ops, dt, 1.)
        second = solver.build_Minv2(grid, ops, dt, 1.)
        space = (Ny, Nx)
    else:
        Nz, Ny, Nx = shape
        grid = bulk3d.make_grid_3d(Nx, Ny, Nz, Nx * .4, Ny * .4, Nz * .4)
        ops = bulk3d.build_operators_3d(grid, **deck['gradient'], kappa=deck['kappa_b'])
        el = None
        if elastic_mode is not None:
            st = None
            if static:
                st = rng.normal(size=(3, 3, Nz, Ny, Nx)) * 1e-4
                st = .5 * (st + st.swapaxes(0, 1))
            el = bulk3d.setup_elastic_3d(grid, (c['C11'], c['C12'], c['C44']),
                                         (q['Q11_phys'], q['Q12_phys'], q['Q44_phys'] / 4),
                                         k0=elastic_mode, eps_star_static=st)
        first = bulk3d.build_Minv_3d(grid, ops, dt, 1.)
        second = bulk3d.build_Minv2_3d(grid, ops, dt, 1.)
        space = (Nz, Ny, Nx)
    lead = (3, ensemble) if ensemble else (3,)
    P = .26 * np.tanh(rng.normal(size=lead + space))
    Pm = P + 1e-3 * rng.normal(size=P.shape)
    return dict(engine=engine, grid=grid, ops=ops, el=el, first=first, second=second,
                alpha=deck['landau'], P=P, Pm=Pm, dt=dt, deck=deck)


def _step(pr, kind, P, Pm, *, E=(0., 0., 0.), rng=None, ou=None, stab=0.0, first=None, second=None):
    mod2 = pr['engine'] == '2d'
    kw = {'stab': stab} if stab else {}
    first = pr['first'] if first is None else first
    second = pr['second'] if second is None else second
    args = (pr['alpha'], pr['ops'])
    if kind == 'sbdf1':
        fn = solver.step if mod2 else bulk3d.step_3d
        return fn(P, *args, first, pr['dt'], 1., E, elastic_params=pr['el'], **kw)
    if kind == 'sbdf2':
        fn = solver.step_sbdf2 if mod2 else bulk3d.step_sbdf2_3d
        return fn(P, Pm, *args, second, pr['dt'], 1., E, elastic_params=pr['el'], **kw)
    fn = solver.step_langevin if mod2 else bulk3d.step_langevin_3d
    return fn(P, *args, ou, pr['dt'], 1., E, rng, elastic_params=pr['el'])


def _ou(pr, cutoff=None):
    g = pr['grid']
    kT = units.kT_langevin(300., g['hx'], g['hy'], g.get('hz', g['hx']))
    build = solver.build_ou_operators if pr['engine'] == '2d' else bulk3d.build_ou_operators_3d
    if cutoff is None:
        return build(pr['ops'], pr['dt'], 1., kT)
    return build(pr['ops'], pr['dt'], 1., kT, noise_cutoff_nm=cutoff, grid=g)


def _trajectory(pr, kind, backend, steps=3, **kw):
    if backend == 'fused':
        execution.configure(1, 'numba', fused=True)
    else:
        execution.configure(1, backend)
    P, Pm = pr['P'].copy(), pr['Pm'].copy()
    rng = np.random.default_rng(4242) if kind == 'langevin' else None
    out = []
    for _ in range(steps):
        new = _step(pr, kind, P, Pm, rng=rng, **kw)
        out.append(new)
        Pm, P = P, new
    execution.configure(1, 'numpy')
    return out, (rng.bit_generator.state if rng is not None else None)


def _assert_close(a, b, rtol=RTOL):
    scale = max(np.max(np.abs(b)), 1e-300)
    assert np.max(np.abs(a - b)) <= rtol * scale, np.max(np.abs(a - b)) / scale


CASES = [('2d', (12, 16)), ('2d', (9, 11)), ('bulk3d', (6, 8, 10)), ('bulk3d', (5, 7, 6))]


@pytest.mark.parametrize('engine,shape', CASES)
@pytest.mark.parametrize('kind', ['sbdf1', 'sbdf2', 'langevin'])
def test_matches_reference_per_step(engine, shape, kind):
    pr = _problem(engine, shape)
    ou = _ou(pr) if kind == 'langevin' else None
    ref, ref_rng = _trajectory(pr, kind, 'numpy', ou=ou)
    got, got_rng = _trajectory(pr, kind, 'fused', ou=ou)
    for a, b in zip(got, ref):
        assert a.dtype == np.float64 and a.shape == b.shape
        _assert_close(a, b)
    assert got_rng == ref_rng


def test_batched_2d_inverse_transform_path(monkeypatch):
    from solvers import cpu_fused
    monkeypatch.setattr(cpu_fused, 'PER_COMPONENT_2D', False)   # the multi-worker choice
    pr = _problem('2d', (9, 11))
    for kind in ('sbdf1', 'sbdf2'):
        ref, _ = _trajectory(pr, kind, 'numpy')
        got, _ = _trajectory(pr, kind, 'fused')
        for a, b in zip(got, ref):
            _assert_close(a, b)


@pytest.mark.parametrize('engine,shape', [CASES[0], CASES[2]])
def test_stab_field_static_and_clamped(engine, shape):
    modes = ['stress_free'] + (['clamped'] if engine == 'bulk3d' else [])
    for mode in modes:
        pr = _problem(engine, shape, elastic_mode=mode, static=True)
        E = 0.01 * np.random.default_rng(3).normal(size=pr['P'].shape)
        if engine == '2d':
            f1 = solver.build_Minv(pr['grid'], pr['ops'], pr['dt'], 1., stab=1.5)
            f2 = solver.build_Minv2(pr['grid'], pr['ops'], pr['dt'], 1., stab=1.5)
        else:
            f1 = bulk3d.build_Minv_3d(pr['grid'], pr['ops'], pr['dt'], 1., stab=1.5)
            f2 = bulk3d.build_Minv2_3d(pr['grid'], pr['ops'], pr['dt'], 1., stab=1.5)
        for kind in ('sbdf1', 'sbdf2'):
            kw = dict(E=E, stab=1.5, first=f1, second=f2)
            ref, _ = _trajectory(pr, kind, 'numpy', **kw)
            got, _ = _trajectory(pr, kind, 'fused', **kw)
            for a, b in zip(got, ref):
                _assert_close(a, b)
        ou = _ou(pr, cutoff=1.0)
        ref, ref_rng = _trajectory(pr, 'langevin', 'numpy', ou=ou, E=(0., 0., .02))
        got, got_rng = _trajectory(pr, 'langevin', 'fused', ou=ou, E=(0., 0., .02))
        for a, b in zip(got, ref):
            _assert_close(a, b)
        assert got_rng == ref_rng


@pytest.mark.parametrize('backend', ['numpy', 'fused'])
@pytest.mark.parametrize('engine', ['2d', 'bulk3d'])
def test_casekit_temperature_seed_sweep_matches_separate_members(engine, backend):
    module = solver if engine == '2d' else bulk3d
    shape = (8, 10) if engine == '2d' else (4, 6, 6)
    pr = _problem(engine, shape)
    temperatures, seeds = (280., 300., 330.), (3, 4, 5)
    alphas = [materials.get('bto_lcc2005_8th', T=T)['landau'] for T in temperatures]
    alpha = casekit.ensemble_alpha(alphas, len(shape))
    P = casekit.ensemble_initial(module, pr['grid'], seeds, .05)
    single = (solver.random_smooth_field if engine == '2d' else bulk3d.random_smooth_field_3d)
    for s, seed in enumerate(seeds):
        np.testing.assert_array_equal(P[:, s], single(pr['grid'], 3, .05, .3, seed))
    if backend == 'fused':
        execution.configure(1, 'numba', fused=True)
    batch = dict(pr, alpha=alpha)
    out = _step(batch, 'sbdf2', _step(batch, 'sbdf1', P, None), P)
    for s in range(len(seeds)):
        member = dict(pr, alpha=alphas[s])
        ref = _step(member, 'sbdf2', _step(member, 'sbdf1', P[:, s], None), P[:, s])
        _assert_close(out[:, s], ref, rtol=1e-13)


@pytest.mark.parametrize('engine,shape', [('2d', (8, 10)), ('bulk3d', (4, 6, 6))])
def test_ensemble_axis_and_member_alpha(engine, shape):
    pr = _problem(engine, shape, ensemble=3)
    # A per-member a1 (temperature sweep) broadcast over the ensemble axis.
    a1 = np.asarray(pr['alpha']['a1']) * np.array([1., .9, 1.1]).reshape((3,) + (1,) * len(shape))
    pr['alpha'] = dict(pr['alpha'], a1=a1)
    E = 0.01 * np.random.default_rng(8).normal(size=(3,) + shape)    # shared field
    for kind in ('sbdf1', 'sbdf2'):
        ref, _ = _trajectory(pr, kind, 'numpy', E=E)
        got, _ = _trajectory(pr, kind, 'fused', E=E)
        for a, b in zip(got, ref):
            _assert_close(a, b)
    ou = _ou(pr)
    ref, _ = _trajectory(pr, 'langevin', 'numpy', ou=ou)
    got, _ = _trajectory(pr, 'langevin', 'fused', ou=ou)
    for a, b in zip(got, ref):
        _assert_close(a, b)


@pytest.mark.parametrize('engine,shape', [('2d', (12, 16)), ('bulk3d', (6, 8, 10))])
def test_uncovered_inputs_take_the_backend_path_unchanged(engine, shape):
    pr = _problem(engine, shape, elastic_mode='misfit' if engine == '2d' else 'stress_free')
    if engine == 'bulk3d':
        # An inhomogeneous background kappa (electrostatic SPM) is not fused.
        kappa = pr['deck']['kappa_b'] * (1 + .2 * np.random.default_rng(1).random(shape))
        pr['ops'] = bulk3d.build_operators_3d(pr['grid'], **pr['deck']['gradient'], kappa=kappa)
        pr['first'] = bulk3d.build_Minv_3d(pr['grid'], pr['ops'], pr['dt'], 1.)
        pr['second'] = bulk3d.build_Minv2_3d(pr['grid'], pr['ops'], pr['dt'], 1.)
    for kind in ('sbdf1', 'sbdf2'):
        plain, _ = _trajectory(pr, kind, 'numba', steps=2)
        fused, _ = _trajectory(pr, kind, 'fused', steps=2)
        for a, b in zip(fused, plain):
            np.testing.assert_array_equal(a, b)


def test_scratch_is_not_shared_between_trajectories():
    one = _problem('2d', (12, 16), static=True)
    two = _problem('bulk3d', (6, 8, 10))
    three = _problem('2d', (12, 16))
    three['P'] = three['P'][::-1].copy()
    alone = [_trajectory(pr, 'sbdf2', 'fused', steps=4)[0] for pr in (one, two, three)]
    execution.configure(1, 'numba', fused=True)
    states = [[pr['P'].copy(), pr['Pm'].copy()] for pr in (one, two, three)]
    outputs = [[], [], []]
    for _ in range(4):                       # interleave three trajectories step by step
        for i, pr in enumerate((one, two, three)):
            P, Pm = states[i]
            new = _step(pr, 'sbdf2', P, Pm)
            outputs[i].append(new)
            states[i] = [new, P]
    for got, ref in zip(outputs, alone):
        for a, b in zip(got, ref):
            np.testing.assert_array_equal(a, b)
    # Outputs never alias each other or the inputs; later steps leave them alone.
    for out in outputs:
        for a in range(len(out)):
            for b in range(a + 1, len(out)):
                assert not np.shares_memory(out[a], out[b])
    assert not np.shares_memory(outputs[0][0], one['P'])


def test_restart_continues_bit_for_bit(tmp_path):
    pr = _problem('bulk3d', (6, 8, 10))
    ou = _ou(pr)
    execution.configure(1, 'numba', fused=True)
    rng = np.random.default_rng(77)
    P = pr['P'].copy()
    for _ in range(2):
        P = _step(pr, 'langevin', P, None, rng=rng, ou=ou)
    io_utils.save_state_npz(tmp_path / 'r.npz', P, pr['grid'], 2, 2 * pr['dt'],
                            restart=dict(scheme='langevin', dt=pr['dt'],
                                         rng_state=rng.bit_generator.state))
    for _ in range(3):
        P = _step(pr, 'langevin', P, None, rng=rng, ou=ou)
    state = io_utils.load_state_npz(tmp_path / 'r.npz')
    rng2 = np.random.default_rng()
    rng2.bit_generator.state = state['checkpoint']['restart']['rng_state']
    Q = state['P']
    for _ in range(3):
        Q = _step(pr, 'langevin', Q, None, rng=rng2, ou=ou)
    np.testing.assert_array_equal(Q, P)


def test_configure_contract():
    with pytest.raises(ValueError):
        execution.configure(1, 'numpy', fused=True)
    with pytest.raises(ValueError):
        execution.configure(1, 'numba', fused='yes')
    assert 'fused' not in execution.configure(1, 'numba')
    assert solver._FUSED is None
    settings = execution.configure(1, 'numba', fused=True)
    assert settings['fused'] is True and solver._FUSED is not None
    execution.configure(1, 'numpy')
    assert solver._FUSED is None


@pytest.mark.parametrize('nx,ny,substrate', [(8, 8, True), (9, 7, True), (10, 6, False)])
def test_film_force_solver_matches_reference(nx, ny, substrate):
    from solvers import cpu_fused, film3d, film_elastic
    deck = materials.get('bto_lcc2005_paper', T=298.15, dx_nm=0.4)
    c, q = deck['elastic'], deck['electrostriction']
    grid = film3d.make_grid_3d(nx, ny, nx * .1, ny * .1, 9, 3.2)
    stack = film_elastic.make_stack_grid(grid, hs_dx=5 if substrate else 0)
    el = film_elastic.setup_film_elastic(
        stack, c['C11'], c['C12'], c['C44'], q['Q11_phys'], q['Q12_phys'], q['Q44_phys'], e0=-.01,
        C_substrate=materials.get('ss316_rt', T=298.15)['elastic'] if substrate else None)
    P = .2 * np.random.default_rng(nx).normal(size=(3, 9, ny, nx))
    reference, _ = film_elastic.elastic_force_film(P, el)
    force = cpu_fused.film_force_solver(el, 1)
    got = force(P)
    _assert_close(got, reference)
    again = force(P * .5)
    assert not np.shares_memory(got, again)
    _assert_close(again, film_elastic.elastic_force_film(P * .5, el)[0])


@pytest.mark.parametrize('kind', ['sbdf1', 'sbdf2', 'langevin'])
def test_film_steps_match_reference(kind):
    from cases.multicore_fixture import Fixture
    from solvers import film3d, film3d_langevin
    f = Fixture('film3d', 8, dt=.004)
    f.advance(2)
    g = f.grid
    G = f.deck['gradient']['G11']
    kT = units.kT_langevin(300., g['hx'], g['hy'], g['hz'])
    ou = film3d_langevin.build_ou_operators_film(g, G, f.dt, 1., kT)
    s1 = film3d.build_grad_factors(g, G, f.dt, 1., stab=1.5)
    s2 = film3d.build_grad_factors_sbdf2(g, G, f.dt, 1., stab=1.5)
    E = .01 * np.random.default_rng(6).normal(size=f.P.shape)

    def trajectory(mode, el, stab):
        if mode == 'fused':
            execution.configure(1, 'numba', el=f.el, fused=True)
        else:
            execution.configure(1, 'numpy', el=f.el)
        P, Pm, rng, out = f.P.copy(), f.previous.copy(), np.random.default_rng(9), []
        kw = {'stab': 1.5} if stab else {}
        for _ in range(3):
            if kind == 'sbdf1':
                new = film3d.step(P, f.alpha, g, s1 if stab else f.first, f.poisson, f.dt, 1.,
                                  E_ext=E, el=el, **kw)
            elif kind == 'sbdf2':
                new = film3d.step_sbdf2(P, Pm, f.alpha, g, s2 if stab else f.second, f.poisson,
                                        f.dt, 1., E_ext=(0., 0., .02), el=el, **kw)
            else:
                new = film3d_langevin.step_langevin_film(P, f.alpha, g, ou, f.poisson, f.dt, 1.,
                                                         (0., 0., 0.), rng, el=el)
            out.append(new)
            Pm, P = P, new
        return out, rng.bit_generator.state

    try:
        for el in (f.el, None):
            for stab in ((False, True) if kind != 'langevin' else (False,)):
                ref, ref_rng = trajectory('numpy', el, stab)
                got, got_rng = trajectory('fused', el, stab)
                for a, b in zip(got, ref):
                    _assert_close(a, b)
                assert got_rng == ref_rng
        if kind == 'sbdf2':
            # A pentadiagonal (H11) implicit operator keeps the reference solve.
            f.second = film3d.build_grad_factors_sbdf2(g, G, f.dt, 1., H11=0.0)
            ref, _ = trajectory('numpy', f.el, False)
            got, _ = trajectory('fused', f.el, False)
            for a, b in zip(got, ref):
                _assert_close(a, b)
    finally:
        f.close()
