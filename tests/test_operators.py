"""The shared operator layer (solvers/operators.py).

  * cross-solver equivalence: bulk3d at Nz=1 IS the 2D engine (symbols, ops,
    Green operator, forces bit-for-bit); the film's in-plane symbols ARE the
    2D ones on the same in-plane grid;
  * the Nyquist rule, checked against an independent re-statement of it, for
    every odd symbol of every engine on even, odd and mixed grids (and even
    symbols are never masked);
  * the E_ext / static-eigenstrain normalisers accept and refuse the
    documented shapes;
  * a guard that the engines DELEGATE to the shared layer: the old private
    names are the shared functions themselves, and every engine builder
    returns the very objects the shared functions produced (spies, not source
    greps) -- a solver that re-implements a symbol fails here;
  * film q/theta gradient energy and force consistency across grid parities.
"""
import itertools
import os
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(name, '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from solvers import (bulk3d, elastic, film3d, grad4, kittel_afe, octa_tilt,
                     operators as op, solver)

for module in (solver, elastic, bulk3d, film3d, kittel_afe, octa_tilt):
    module.FFT_WORKERS = 1

G = dict(G11=1.3, G12=-0.4, G44=0.6, G44p=0.25)        # anisotropic: A12 != 0
GV = tuple(G.values())
C = elastic.build_C_tensor(1.7, .8, .6)
CT, QT = (1.7, .8, .6), (.09, -.03, .0125)
EL2D = (1.7, .8, .6, .09, -.03, .05)

N_KEY = {'x': 'Nx', 'y': 'Ny', 'z': 'Nz'}
ARRAY_AXIS = {'x': -1, 'y': -2, 'z': -3}


def grid2(nx, ny):
    return solver.make_grid(nx, ny, nx * .8, ny * .7)


def grid3(nx, ny, nz):
    return bulk3d.make_grid_3d(nx, ny, nz, nx * .8, ny * .7, nz * .9)


def gridf(nx, ny, nz=5):
    return film3d.make_grid_3d(nx, ny, nx * .8, ny * .7, nz, 2.4)


# ===========================================================================
# 1. Cross-solver equivalence
# ===========================================================================
@pytest.mark.parametrize('nx,ny', [(8, 8), (7, 9), (8, 7), (6, 5), (4, 4)])
def test_bulk3d_nz1_is_the_2d_engine(nx, ny):
    g2, g3 = grid2(nx, ny), bulk3d.make_grid_3d(nx, ny, 1, nx * .8, ny * .7, 1.0)
    for key in ('KXr', 'KYr', 'KX', 'KY'):
        np.testing.assert_array_equal(g3[key][0], g2[key])
    o2 = solver.build_operators(g2, *GV, 0.1)
    o3 = bulk3d.build_operators_3d(g3, *GV, 0.1)
    for key in ('A11', 'A22', 'A33', 'A12', 'D11', 'D22', 'D12'):
        np.testing.assert_array_equal(o3[key][0], o2[key], err_msg=key)
    for key in ('A13', 'A23', 'D13', 'D23', 'D33'):        # k3 == 0
        assert not np.any(o3[key]), key
    h2 = grad4.build_operators_hog(g2, *GV, 0.1, 0.3)
    h3 = grad4.build_operators_hog_3d(g3, *GV, 0.1, 0.3)
    for key in ('A11', 'A22', 'A33', 'A12'):
        np.testing.assert_array_equal(h3[key][0], h2[key], err_msg='hog ' + key)
    m2, m3 = op.masked_wavevectors_2d(g2), op.masked_wavevectors_3d(g3)
    np.testing.assert_array_equal(m3[0][0], m2[0])
    np.testing.assert_array_equal(m3[1][0], m2[1])
    assert not np.any(m3[2])
    O2, K2 = elastic.build_Omega(g2, C)
    O3, K3, z3 = bulk3d.build_Omega_3d(g3, C)
    np.testing.assert_array_equal(O3[0], O2)
    np.testing.assert_array_equal(K3[:, 0], K2)
    np.testing.assert_array_equal(z3[0], elastic.setup_elastic(g2, *EL2D)['zero_k'])
    np.testing.assert_array_equal(kittel_afe.build_ops_q_3d(g3, .7)['Aq'][0],
                                  kittel_afe.build_ops_q(g2, .7)['Aq'])
    np.testing.assert_array_equal(octa_tilt.build_ops_theta_3d(g3, .6)['Atheta'][0],
                                  octa_tilt.build_ops_theta(g2, .6)['Atheta'])
    # ... and therefore the operators' action
    P = np.random.default_rng(nx * ny).normal(size=(3, ny, nx))
    np.testing.assert_array_equal(bulk3d.grad_force_3d(P[:, None], o3)[:, 0],
                                  solver.grad_force(P, o2))
    np.testing.assert_array_equal(bulk3d.elec_force_3d(P[:, None], o3, (.01, 0., -.02))[:, 0],
                                  solver.elec_force(P, o2, (.01, 0., -.02)))
    # the energy DENSITIES group their terms differently (3D has the z pairs),
    # so they agree to round-off, not bit-for-bit -- the Nyquist part is shared
    np.testing.assert_allclose(bulk3d.grad_energy_density_3d(P[:, None], g3, o3)[0],
                               solver.grad_energy_density(P, g2, o2), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize('nx,ny', [(8, 8), (7, 9), (8, 7), (9, 6)])
def test_film_inplane_symbols_are_the_2d_ones(nx, ny):
    g2, gf = grid2(nx, ny), gridf(nx, ny)
    for key in ('KXr', 'KYr', 'KX', 'KY', 'Nxr'):
        np.testing.assert_array_equal(gf[key], g2[key])
    Gd = film3d._grad_dict(G)
    for f, s in zip(film3d._grad_A2d(gf, Gd), solver._grad_symbols(g2, *GV)):
        np.testing.assert_array_equal(f, s)
    hog = grad4.build_operators_hog(g2, *GV, 0.1, 0.02)
    for f, key in zip(film3d._grad_A2d(gf, Gd, 0.02), ('A11', 'A22', 'A12', 'A33')):
        np.testing.assert_array_equal(f, hog[key], err_msg=key)
    for f, s in zip(film3d._odd_k(gf), solver._modified_wavevectors_2d(g2)):
        np.testing.assert_array_equal(f, 1j * s)
    # the isotropic film symbol (scalar-G path) is the 2D isotropic A33
    np.testing.assert_array_equal(op.isotropic_symbol(gf, .9, 2),
                                  solver.build_operators(g2, .9, 0., .45, .45, .1)['A33'])


# ===========================================================================
# 2. The Nyquist rule in every engine (independent re-statement of the rule)
# ===========================================================================
def _on_nyquist(shape, grid, axes):
    """Entries on the Nyquist line/plane of an EVEN axis among `axes`."""
    m = np.zeros(shape, dtype=bool)
    for a in axes:
        N = grid[N_KEY[a]]
        if N % 2 == 0:
            idx = [slice(None)] * len(shape)
            idx[ARRAY_AXIS[a]] = N // 2
            m[tuple(idx)] = True
    return m


def _check_odd(sym, raw, grid, axes, what):
    on = _on_nyquist(sym.shape, grid, axes)
    assert not np.any(sym[on]), f"{what}: nonzero on an even-axis Nyquist plane"
    np.testing.assert_array_equal(sym[~on], raw[~on], err_msg=f"{what}: changed off-Nyquist")
    return bool(np.any(raw[on]))          # did the mask actually remove something?


def _check_even(sym, raw, what):
    np.testing.assert_array_equal(sym, raw, err_msg=f"{what}: an even symbol was masked")


def _raw_2d(g):
    KX, KY = g['KXr'], g['KYr']
    k2 = KX**2 + KY**2
    safe = np.where(k2 == 0, 1., k2)
    gg, off = G['G44'] + G['G44p'], G['G12'] + G['G44'] - G['G44p']
    return dict(A11=G['G11'] * KX**2 + gg * KY**2, A22=G['G11'] * KY**2 + gg * KX**2,
                A33=gg * k2, A12=off * KX * KY, D11=np.where(k2 == 0, 0., KX * KX / safe),
                D22=np.where(k2 == 0, 0., KY * KY / safe),
                D12=np.where(k2 == 0, 0., KX * KY / safe), KX=KX, KY=KY)


GRIDS_2D = list(itertools.product((4, 5, 6, 7), (4, 5, 6, 7)))
GRIDS_3D = list(itertools.product((4, 5), (4, 5), (1, 2, 3, 4)))


def test_nyquist_rule_2d_engine_and_film():
    removed = 0
    for nx, ny in GRIDS_2D:
        g, gf = grid2(nx, ny), gridf(nx, ny, 3)
        raw = _raw_2d(g)
        ops = solver.build_operators(g, *GV, .1)
        for key in ('A12', 'D12'):
            removed += _check_odd(ops[key], raw[key], g, 'xy', f'solver {key} {nx}x{ny}')
        for key in ('A11', 'A22', 'A33', 'D11', 'D22'):
            _check_even(ops[key], raw[key], f'solver {key} {nx}x{ny}')
        hog = grad4.build_operators_hog(g, *GV, .1, .3)
        k4 = raw['KX']**4 + raw['KY']**4
        for key in ('A11', 'A22', 'A33'):
            _check_even(hog[key], raw[key] + .3 * k4, f'grad4 {key} {nx}x{ny}')
        # first-derivative wavevectors: solver energy path, elastic Green K
        Ks = (solver._modified_wavevectors_2d(g), elastic._modified_wavevectors_2d(g),
              elastic.setup_elastic(g, *EL2D)['K'][:2], elastic.build_Omega(g, C)[1][:2])
        for K in Ks:
            _check_odd(K[0], raw['KX'], g, 'x', f'K_x {nx}x{ny}')
            _check_odd(K[1], raw['KY'], g, 'y', f'K_y {nx}x{ny}')
        # film: in-plane A12 and the odd i*k symbols
        A11, A22, A12, A33 = film3d._grad_A2d(gf, film3d._grad_dict(G), .02)
        _check_odd(A12, raw['A12'], gf, 'xy', f'film A12 {nx}x{ny}')
        _check_even(A11, raw['A11'] + .02 * k4, f'film A11 {nx}x{ny}')
        aK, bK = film3d._odd_k(gf)
        _check_odd(aK, 1j * raw['KX'], gf, 'x', f'film i*kx {nx}x{ny}')
        _check_odd(bK, 1j * raw['KY'], gf, 'y', f'film i*ky {nx}x{ny}')
        # sector symbols are even: never masked
        _check_even(kittel_afe.build_ops_q(g, .7)['Aq'], .7 * (raw['KX']**2 + raw['KY']**2), 'Aq')
        _check_even(octa_tilt.build_ops_theta(g, .6)['Atheta'],
                    .6 * (raw['KX']**2 + raw['KY']**2), 'Atheta')
    assert removed > 0          # the grid family does exercise the rule


def test_nyquist_rule_bulk3d():
    removed = 0
    off = G['G12'] + G['G44'] - G['G44p']
    pairs = {'A12': (0, 1), 'A13': (0, 2), 'A23': (1, 2)}
    for nx, ny, nz in GRIDS_3D:
        g = grid3(nx, ny, nz)
        K = (g['KXr'], g['KYr'], g['KZr'])
        k2 = K[0]**2 + K[1]**2 + K[2]**2
        safe = np.where(k2 == 0, 1., k2)
        ops = bulk3d.build_operators_3d(g, *GV, .1)
        for key, (i, j) in pairs.items():
            axes = 'xyz'[i] + 'xyz'[j]
            removed += _check_odd(ops[key], off * K[i] * K[j], g, axes, f'bulk3d {key}')
            _check_odd(ops['D' + key[1:]], np.where(k2 == 0, 0., K[i] * K[j] / safe), g, axes,
                       f'bulk3d D{key[1:]}')
        for i in range(3):
            others = [K[m]**2 for m in range(3) if m != i]
            _check_even(ops[f'A{i + 1}{i + 1}'],
                        G['G11'] * K[i]**2 + (G['G44'] + G['G44p']) * (others[0] + others[1]),
                        f'bulk3d A{i + 1}{i + 1}')
            _check_even(ops[f'D{i + 1}{i + 1}'], np.where(k2 == 0, 0., K[i] * K[i] / safe),
                        f'bulk3d D{i + 1}{i + 1}')
        for Km in (bulk3d._modified_wavevectors_3d(g), bulk3d.build_Omega_3d(g, C)[1],
                   bulk3d.setup_elastic_3d(g, CT, QT)['K']):
            for j in range(3):
                _check_odd(Km[j], K[j], g, 'xyz'[j], f'bulk3d K_{"xyz"[j]}')
        _check_even(kittel_afe.build_ops_q_3d(g, .7)['Aq'], .7 * k2, 'Aq 3d')
    assert removed > 0


def test_symbol_builders_do_not_mutate_the_grid():
    for g in (grid2(8, 6), grid3(4, 6, 2), gridf(8, 6)):
        before = {k: v.copy() for k, v in g.items() if isinstance(v, np.ndarray)}
        dim = 3 if 'KZr' in g else 2
        op.masked_wavevectors(g, dim)
        op.derivative_symbols(g, dim)
        op.grad_symbols(g, *GV, dim)
        op.depolarization_symbols(g, dim)
        op.green_operator(g, C, dim)
        for k, v in before.items():
            np.testing.assert_array_equal(g[k], v, err_msg=k)


def test_nyquist_correction_is_identity_on_odd_axes():
    P = np.random.default_rng(3).normal(size=(3, 7, 5))
    fg = np.random.default_rng(4).normal(size=(7, 5))
    assert op.add_nyquist_correction(fg, P, grid2(5, 7), 1.3, 2) is fg
    # one even axis: exactly that axis' alternating mode is added back
    g, P = grid2(6, 7), np.random.default_rng(5).normal(size=(3, 7, 6))
    out = op.add_nyquist_correction(np.zeros((7, 6)), P, g, 2.0, 2)
    proj = [op.nyquist_project_axis(P[i], -1, 6) for i in range(3)]
    np.testing.assert_allclose(out, sum(0.5 * 2.0 * (np.pi / g['hx'])**2 * p**2 for p in proj),
                               rtol=1e-14)


# ===========================================================================
# 3. Normalisers: accepted and refused shapes
# ===========================================================================
def test_vec3_field_shapes():
    for ndim in (3, 4, 5):
        for ens in (False, True):
            out = op.vec3_field((1., 2., 3.), ndim, ensemble=ens)
            assert out.shape == (3,) + (1,) * (ndim - 1)
            np.testing.assert_array_equal(out.ravel(), [1., 2., 3.])
    f = np.random.default_rng(0).normal(size=(3, 5, 6))
    np.testing.assert_array_equal(op.vec3_field(f, 3), f)                   # unbatched: as given
    out = op.vec3_field(f, 4, ensemble=True)                                # S inserted
    assert out.shape == (3, 1, 5, 6)
    np.testing.assert_array_equal(out[:, 0], f)
    fe = np.random.default_rng(1).normal(size=(3, 2, 5, 6))
    np.testing.assert_array_equal(op.vec3_field(fe, 4, ensemble=True), fe)  # member-resolved kept
    f3 = np.random.default_rng(2).normal(size=(3, 4, 5, 6))
    assert op.vec3_field(f3, 5, ensemble=True).shape == (3, 1, 4, 5, 6)    # bulk3d batch
    for bad, ndim in (((2, 5, 6), 4), ((4,), 3), ((3, 2, 2, 5, 6), 4), ((3, 6), 4)):
        with pytest.raises(ValueError, match='E_ext of shape'):
            op.vec3_field(np.zeros(bad), ndim, ensemble=True)
    assert solver._reshape_vec3 is op.vec3_field and bulk3d._reshape_vec3_3d is op.vec3_field


def test_ensemble_field_static_eigenstrain_shapes():
    s = np.random.default_rng(0).normal(size=(3, 3, 5, 6))
    out = op.ensemble_field(s, 2, 5, 'eps_star_static')        # (3,3,S,Ny,Nx) target
    assert out.shape == (3, 3, 1, 5, 6)
    np.testing.assert_array_equal(out[:, :, 0], s)
    se = np.random.default_rng(1).normal(size=(3, 3, 2, 5, 6))
    assert op.ensemble_field(se, 2, 5, 'eps_star_static') is se
    for bad, ndim in (((3, 5, 6), 5), ((3, 3, 5), 5), ((3, 3, 2, 2, 5, 6), 5), ((2, 3, 5, 6), 5)):
        with pytest.raises(ValueError, match='eps_star_static of shape'):
            op.ensemble_field(np.zeros(bad), 2, ndim, 'eps_star_static')
    assert elastic._ensemble_field is op.ensemble_field


def test_vec3_field_checked_film_shapes():
    shape = (3, 5, 6, 7)
    out = op.vec3_field_checked((1., 2., 3.), shape)
    assert out.shape == (3, 1, 1, 1)
    for ok in ((3, 5, 6, 7), (3, 1, 1, 7), (3, 5, 1, 1), (3, 1, 1, 1)):
        v = np.ones(ok)
        np.testing.assert_array_equal(op.vec3_field_checked(v, shape), v)
    for bad in ((3, 6, 7), (2,), (3, 2, 6, 7), (2, 5, 6, 7), (3, 5, 6, 7, 1)):
        with pytest.raises(ValueError, match='component-first field broadcastable'):
            op.vec3_field_checked(np.ones(bad), shape)
    with pytest.raises(ValueError, match='finite'):
        op.vec3_field_checked((np.nan, 0., 0.), shape)
    # the film engine uses exactly this rule
    g = gridf(7, 6)
    P = np.random.default_rng(0).normal(size=(3, 5, 6, 7))
    pf = film3d.poisson_factor(g, .885)
    with pytest.raises(ValueError, match='component-first field broadcastable'):
        film3d.elec_force_3d(P, g, pf, np.zeros((3, 6, 7)))


# ===========================================================================
# 4. Guard: the engines delegate to the shared layer
# ===========================================================================
ALIASES = [
    (solver, '_grad_symbols', 'grad_symbols_2d'),
    (solver, '_modified_wavevectors_2d', 'masked_wavevectors_2d'),
    (solver, '_nyquist_project_axis', 'nyquist_project_axis'),
    (solver, '_reshape_vec3', 'vec3_field'),
    (bulk3d, '_grad_symbols_3d', 'grad_symbols_3d'),
    (bulk3d, '_modified_wavevectors_3d', 'masked_wavevectors_3d'),
    (bulk3d, '_reshape_vec3_3d', 'vec3_field'),
    (elastic, '_modified_wavevectors_2d', 'masked_wavevectors_2d'),
    (elastic, '_ensemble_field', 'ensemble_field'),
    (film3d, '_odd_k', 'derivative_symbols_2d'),
]


@pytest.mark.parametrize('module,old,new', ALIASES,
                         ids=[f'{m.__name__.split(".")[-1]}.{o}' for m, o, _ in ALIASES])
def test_old_private_names_are_the_shared_functions(module, old, new):
    assert getattr(module, old) is getattr(op, new)


@pytest.fixture
def spy(monkeypatch):
    """spy('name') wraps operators.name; returns the list of its outputs."""
    def install(name):
        real, calls = getattr(op, name), []

        def wrapper(*a, **k):
            out = real(*a, **k)
            calls.append(out)
            return out
        monkeypatch.setattr(op, name, wrapper)
        return calls
    return install


def test_engine_ops_are_shared_symbol_objects(spy):
    A2, D2 = spy('grad_symbols_2d'), spy('depolarization_symbols_2d')
    ops = solver.build_operators(grid2(8, 7), *GV, .1)
    assert ops['A11'] is A2[-1][0] and ops['A12'] is A2[-1][2] and ops['A33'] is A2[-1][3]
    assert ops['D11'] is D2[-1][0] and ops['D12'] is D2[-1][2]

    A3, D3 = spy('grad_symbols_3d'), spy('depolarization_symbols_3d')
    ops3 = bulk3d.build_operators_3d(grid3(4, 5, 2), *GV, .1)
    for n, key in enumerate(('A11', 'A22', 'A33', 'A12', 'A13', 'A23')):
        assert ops3[key] is A3[-1][n], key
    for n, key in enumerate(('D11', 'D22', 'D33', 'D12', 'D13', 'D23')):
        assert ops3[key] is D3[-1][n], key

    gf = gridf(8, 6)
    A11, A22, A12, A33 = film3d._grad_A2d(gf, film3d._grad_dict(G))
    assert A12 is A2[-1][2] and A11 is A2[-1][0]
    hog = spy('add_hog')
    A11h, _, A12h, _ = film3d._grad_A2d(gf, film3d._grad_dict(G), .02)
    assert A11h is hog[-1][0] and A12h is A2[-1][2]
    h2 = grad4.build_operators_hog(grid2(6, 5), *GV, .1, .3)
    assert h2['A11'] is hog[-1][0] and h2['A33'] is hog[-1][2]
    h3 = grad4.build_operators_hog_3d(grid3(4, 5, 2), *GV, .1, .3)
    assert h3['A22'] is hog[-1][1]

    iso = spy('isotropic_symbol')
    assert kittel_afe.build_ops_q(grid2(6, 5), .7)['Aq'] is iso[-1]
    assert kittel_afe.build_ops_q_3d(grid3(4, 5, 2), .7)['Aq'] is iso[-1]
    assert octa_tilt.build_ops_theta(grid2(6, 5), .6)['Atheta'] is iso[-1]
    assert octa_tilt.build_ops_theta_3d(grid3(4, 5, 2), .6)['Atheta'] is iso[-1]


def test_engine_green_operators_are_shared(spy):
    gr = spy('green_operator')
    Omega, K = elastic.build_Omega(grid2(8, 7), C)
    assert Omega is gr[-1][0] and K is gr[-1][1]
    el = elastic.setup_elastic(grid2(8, 7), *EL2D)
    assert el['Omega'] is gr[-1][0] and el['zero_k'] is gr[-1][2]
    out = bulk3d.build_Omega_3d(grid3(4, 5, 2), C)
    assert all(a is b for a, b in zip(out, gr[-1]))
    el3 = bulk3d.setup_elastic_3d(grid3(4, 5, 2), CT, QT)
    assert el3['Omega'] is gr[-1][0] and el3['K'] is gr[-1][1]


def test_engine_energy_nyquist_correction_is_shared(spy):
    corr = spy('add_nyquist_correction')
    g2, g3 = grid2(8, 6), grid3(4, 6, 2)
    P2 = np.random.default_rng(0).normal(size=(3, 6, 8))
    P3 = np.random.default_rng(1).normal(size=(3, 2, 6, 4))
    assert solver.grad_energy_density(P2, g2, solver.build_operators(g2, *GV, .1)) is corr[-1]
    assert bulk3d.grad_energy_density_3d(P3, g3, bulk3d.build_operators_3d(g3, *GV, .1)) \
        is corr[-1]
    assert kittel_afe.grad_energy_density_q(P2, g2, .7) is corr[-1]
    assert kittel_afe.grad_energy_density_q_3d(P3, g3, .7) is corr[-1]
    assert octa_tilt.grad_energy_density_theta(P2, g2, .6) is corr[-1]
    assert octa_tilt.grad_energy_density_theta_3d(P3, g3, .6) is corr[-1]


def test_engine_normalisers_and_film_symbols_are_shared(spy):
    g = gridf(8, 6)
    P = np.random.default_rng(0).normal(size=(3, 5, 6, 8))
    chk, odd = spy('vec3_field_checked'), spy('derivative_symbols_2d')
    pf = film3d.poisson_factor(g, .885)
    _E, Eext = film3d.solve_poisson_factored(P, g, pf, (.01, 0., -.01))
    assert Eext is chk[-1] and len(odd) == 1
    k2, k4 = spy('k_squared'), spy('hog_symbol')
    film3d.grad_force_3d(P, g, .1)
    film3d.build_grad_factors(g, .1, .05, 1., H11=.02)
    film3d.hog_force_film(P, g, .02)
    assert len(k2) == 2 and len(k4) == 2
    ens = spy('ensemble_field')
    g2 = grid2(6, 5)
    static = np.random.default_rng(1).normal(scale=1e-3, size=(3, 3, 5, 6))
    elastic.elastic_solve(np.zeros((3, 2, 5, 6)), elastic.setup_elastic(g2, *EL2D,
                                                                        eps_star_static=static))
    g3 = grid3(4, 5, 2)
    static3 = np.random.default_rng(2).normal(scale=1e-3, size=(3, 3, 2, 5, 4))
    bulk3d.elastic_solve_3d(np.zeros((3, 2, 2, 5, 4)),
                            bulk3d.setup_elastic_3d(g3, CT, QT, eps_star_static=static3))
    assert len(ens) == 2 and ens[0].shape == (3, 3, 1, 5, 6) and ens[1].shape == (3, 3, 1, 2, 5, 4)


# ===========================================================================
# 5. Film q/theta gradient energy and force
# ===========================================================================
@pytest.mark.parametrize('nx,ny', [(8, 8), (8, 7), (7, 8), (7, 7)])
@pytest.mark.parametrize('energy,force', [
    (kittel_afe.grad_energy_q_film, kittel_afe.grad_force_q_film),
    (octa_tilt.grad_energy_theta_film, octa_tilt.grad_force_theta_film)])
def test_film_q_theta_energy_matches_force(energy, force, nx, ny):
    g = gridf(nx, ny)
    q = np.random.default_rng(0).normal(size=(3, 5, ny, nx))
    d = np.random.default_rng(1).normal(size=q.shape)
    eps = 1e-6
    num = (energy(q + eps * d, g, .07) - energy(q - eps * d, g, .07)) / (2 * eps)
    ana = g['hx'] * g['hy'] * float(np.sum(g['qw'][None, :, None, None] * force(q, g, .07) * d))
    assert abs(num - ana) <= 1e-6 * abs(ana)
