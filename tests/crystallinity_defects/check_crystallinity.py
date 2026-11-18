"""Crystallinity acceptance; all fixtures are tiny and deterministic."""
import numpy as np
from solvers import crystallinity as crys, solver, bulk3d, film3d


def _alpha():
    # Well-scaled mathematical fixture, not a calibrated material.
    return dict(a1=-0.3, a11=0.5, a12=0.2, a111=0.03, a112=0.02,
                a123=0.01, a1111=0.01, a1112=0.003, a1122=0.002, a1123=0.001)


def _fixtures():
    E = (0., 0., 0.)
    dt = 0.001
    G = dict(G11=0.2, G12=0.05, G44=0.1, G44p=0.05)
    g = solver.make_grid(5, 5, 5., 5.)
    op = solver.build_operators(g, **G, kappa=2.)
    inv = solver.build_Minv(g, op, dt, 1.)
    yield ('2d', (3, 5, 5),
           lambda p, a: solver.total_energy(p, a, op, E, g)[1]['total'],
           lambda f: float(f.mean()) * g['Volume'],
           lambda p, a: solver.step(p, a, op, inv, dt, 1., E))
    g3 = bulk3d.make_grid_3d(Nx=5, Ny=5, Nz=3, Lx=5., Ly=5., Lz=3.)
    op3 = bulk3d.build_operators_3d(g3, **G, kappa=2.)
    inv3 = bulk3d.build_Minv_3d(g3, op3, dt, 1.)
    yield ('bulk3d', (3, 3, 5, 5),
           lambda p, a: bulk3d.total_energy_3d(p, a, op3, E, g3)[1]['total'],
           lambda f: float(f.mean()) * g3['Volume'],
           lambda p, a: bulk3d.step_3d(p, a, op3, inv3, dt, 1., E))
    gf = film3d.make_grid_3d(Nx=5, Ny=5, Lx=5., Ly=5., Nz=4, hf=3.)
    pf = film3d.poisson_factor(gf, 2.)
    grad = film3d.build_grad_factors(gf, G, dt, 1.)
    yield ('film3d', (3, 4, 5, 5),
           lambda p, a: film3d.total_energy_3d(p, a, gf, 2., G, poisson_factors=pf)[1]['total'],
           lambda f: 25. * float(np.sum(gf['qw'] * f.mean(axis=(-2, -1)))),
           lambda p, a: film3d.step(p, a, gf, grad, pf, dt, 1.))


def energy_force():
    rng = np.random.default_rng(720)
    numbers = {}
    for name, shape, energy, integrate, step in _fixtures():
        P = rng.normal(0., 0.2, shape)
        direction = rng.normal(size=shape)
        if name == 'film3d':
            direction[:, 1:-1] = 0.  # tests half-weighted surfaces explicitly
        for label, Xc in [('scalar', 0.6), ('field', rng.uniform(0.2, 0.8, shape[1:]))]:
            ref = _alpha()
            ref['a1'] = np.full(shape[1:], -0.3)
            alpha = crys.effective_alpha(ref, Xc, 0.8, shape=shape[1:])
            manual = dict(ref, a1=ref['a1'] + 0.4 * (1 - Xc))
            assert np.array_equal(step(P, alpha), step(P, manual))
            observed = energy(P, alpha) - energy(P, ref)
            expected = integrate(crys.energy_density(P, Xc, 0.8))
            assert np.isclose(observed, expected, rtol=1e-12, atol=1e-12)
            analytic = integrate(np.sum(crys.force(P, Xc, 0.8) * direction, axis=0))
            errors = []
            for eps in (1e-4, 1e-5, 1e-6):
                def delta(p):
                    return energy(p, alpha) - energy(p, ref)
                fd = (delta(P + eps*direction) - delta(P - eps*direction)) / (2*eps)
                errors.append(abs(fd-analytic) / max(abs(analytic), 1e-10))
            numbers[name+'_'+label+'_fd_relerr'] = min(errors)
            for index, error in enumerate(errors):
                numbers[name+'_'+label+'_fd_sweep_'+str(index)] = error
            assert min(errors) <= 1e-6
    return dict(ok=True, numbers=numbers)


def limits():
    rng = np.random.default_rng(721)
    count = 0
    for name, shape, energy, integrate, step in _fixtures():
        P = rng.normal(0., 0.2, shape)
        ref = _alpha()
        ref['a1'] = np.full(shape[1:], -0.3)
        before = ref['a1'].copy()
        baseline = step(P, ref)
        for Xc, A in [(None, 1.), (1., 1.), (0.3, 0.), (np.ones(shape[1:]), 1.)]:
            alpha = crys.effective_alpha(ref, Xc, A, shape=shape[1:])
            assert alpha is not ref and alpha['a1'] is ref['a1']
            assert np.array_equal(step(P, alpha), baseline)
            assert energy(P, alpha) == energy(P, ref)
            assert np.count_nonzero(crys.energy_density(P, Xc, A)) == 0
            count += 1
        crys.effective_alpha(ref, 0.5, 1., shape=shape[1:])
        assert np.array_equal(ref['a1'], before)
        assert all(crys.effective_alpha(ref, 0.5, 1.)[k] is v for k, v in ref.items() if k != 'a1')
    # Pure algebra supports broadcast ensembles without confusing components.
    P = rng.normal(size=(3, 2, 3, 5, 5))
    Xc = rng.uniform(size=(3, 5, 5))
    alpha = crys.effective_alpha(_alpha(), Xc, 0.8, shape=P.shape[1:])
    assert np.allclose(solver.bulk_force(P, alpha)-solver.bulk_force(P, _alpha()), crys.force(P, Xc, 0.8))
    for Xc, A, shape in [(1.1, 1., None), (-.1, 1., None), (np.nan, 1., None),
                          (.5, -1., None), (.5, np.inf, None), (.5, [1., 2.], None),
                          (np.ones((7, 8)), 1., (3, 5, 5))]:
        try:
            crys.effective_alpha(_alpha(), Xc, A, shape=shape)
        except ValueError:
            count += 1
        else:
            raise AssertionError('invalid input accepted')
    return dict(ok=True, numbers={'exact_limits_and_rejections': count})


def stationary():
    # f=-P²+P⁴ plus penalty; analytical branch p²=1/2-A(1-Xc)/4.
    ref = dict(a1=-1., a11=1., a12=0., a111=0., a112=0., a123=0.)
    A, Xc = 0.8, 0.25
    alpha = crys.effective_alpha(ref, Xc, A)
    root = np.sqrt(0.5 - A*(1-Xc)/4)
    P = np.array([[root], [0.], [0.]])
    residual = abs(solver.bulk_force(P, alpha)[0, 0])
    eps = 1e-5
    d = np.array([[eps], [0.], [0.]])
    h = (solver.bulk_force(P+d, alpha)-solver.bulk_force(P-d, alpha))/(2*eps)
    h0 = (solver.bulk_force(P+d, ref)-solver.bulk_force(P-d, ref))/(2*eps)
    error = abs(h[0, 0]-h0[0, 0]-A*(1-Xc))
    assert residual < 1e-12 and error < 1e-9 and h[0, 0] > 0
    return dict(ok=True, numbers={'root': root, 'stationary_residual': residual, 'curvature_shift_error': error})
