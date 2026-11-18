"""Fused kernels (execution.configure(..., fused=True)) on the two bulk3d options
SPEC §11 depol_N and SPEC §12 z_mode="plane_stress", against the NumPy reference:
agreement to round-off per step, coverage fallbacks that never change the physics,
and the restart contract. No timing. Deck: a two-well 8th-order Landau, spatial a1,
field-shaped E_ext and noise cutoff, as the NBT6BT runs use them."""
import os
import sys

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import scipy.fft as spfft

pytest.importorskip('numba')

from solvers import bulk3d, execution, units

RTOL = 1e-12
C = (204.344, 110.6672, 140.056)
Q = (0.05056514, -0.02082094, 0.02201071)
LANDAU = dict(a11=.044132, a12=1.181248, a111=1.469101, a112=-10.013069, a123=-39.957931,
              a1111=5.105353, a1112=0., a1122=74.787170, a1123=165.447908)
N_ANISO = np.array([[.40, .05, -.03], [.05, .30, .02], [-.03, .02, .30]])
DT, L = .2, .5857
GRIDS_2D = [(1, 16, 12), (1, 9, 11)]


@pytest.fixture(autouse=True)
def _reference_afterwards():
    yield
    execution.configure(1, 'numpy')


def _problem(shape, *, z_mode='plane_stress', k0='stress_free', depol_N=None, seed=3):
    Nz, Ny, Nx = shape
    rng = np.random.default_rng(seed)
    grid = bulk3d.make_grid_3d(Nx, Ny, Nz, 2. * Nx, 2. * Ny, 6. if Nz == 1 else 2. * Nz)
    ops = bulk3d.build_operators_3d(grid, .17, 0., .085, .085, 1.04, depol_N=depol_N)
    el = bulk3d.setup_elastic_3d(grid, C, Q, k0=k0, z_mode=z_mode)
    alpha = dict(LANDAU, a1=-.012 + .004 * rng.normal(size=shape))
    E = .01 * rng.normal(size=(3,) + shape)
    P = .3 * np.tanh(rng.normal(size=(3,) + shape))
    return dict(grid=grid, ops=ops, el=el, alpha=alpha, E=E, P=P, Pm=P + 1e-3 * rng.normal(size=P.shape))


def _trajectory(pr, kind, backend, steps=3, E=None, el=None):
    execution.configure(1, 'numba' if backend != 'numpy' else 'numpy', fused=backend == 'fused')
    grid, ops, alpha = pr['grid'], pr['ops'], pr['alpha']
    el = pr['el'] if el is None else el
    E = pr['E'] if E is None else E
    P, Pm, out = pr['P'], pr['Pm'], []
    rng = np.random.default_rng(11)
    if kind == 'sbdf1':
        M = bulk3d.build_Minv_3d(grid, ops, DT, L)
    elif kind == 'sbdf2':
        M = bulk3d.build_Minv2_3d(grid, ops, DT, L)
    else:
        kT = units.kT_langevin(300., grid['hx'], grid['hy'], grid['hz'])
        M = bulk3d.build_ou_operators_3d(ops, DT, L, kT, noise_cutoff_nm=8., grid=grid)
    for _ in range(steps):
        if kind == 'sbdf1':
            P = bulk3d.step_3d(P, alpha, ops, M, DT, L, E, elastic_params=el)
        elif kind == 'sbdf2':
            P, Pm = bulk3d.step_sbdf2_3d(P, Pm, alpha, ops, M, DT, L, E, elastic_params=el), P
        else:
            P = bulk3d.step_langevin_3d(P, alpha, ops, M, DT, L, E, rng, elastic_params=el)
        out.append(P.copy())
    return out, rng.bit_generator.state


def _assert_close(got, ref, rtol=RTOL):
    for a, b in zip(got, ref):
        assert a.dtype == np.float64 and a.shape == b.shape
        assert np.max(np.abs(a - b)) <= rtol * np.max(np.abs(b)), np.max(np.abs(a - b))


@pytest.mark.parametrize('shape', GRIDS_2D)
@pytest.mark.parametrize('kind', ['sbdf1', 'sbdf2', 'langevin'])
@pytest.mark.parametrize('k0', ['stress_free', 'clamped'])
@pytest.mark.parametrize('depol_N', [None, N_ANISO])
def test_plane_stress_matches_reference(shape, kind, k0, depol_N):
    pr = _problem(shape, k0=k0, depol_N=depol_N)
    ref, ref_rng = _trajectory(pr, kind, 'numpy')
    got, got_rng = _trajectory(pr, kind, 'fused')
    _assert_close(got, ref)
    assert got_rng == ref_rng


@pytest.mark.parametrize('shape', [(1, 16, 12), (6, 8, 10), (5, 7, 6)])
@pytest.mark.parametrize('kind', ['sbdf1', 'sbdf2', 'langevin'])
@pytest.mark.parametrize('depol_N', [.3, N_ANISO])
def test_depol_generalized_plane_strain_matches_reference(shape, kind, depol_N):
    pr = _problem(shape, z_mode='generalized_plane_strain', depol_N=depol_N)
    ref, ref_rng = _trajectory(pr, kind, 'numpy')
    got, got_rng = _trajectory(pr, kind, 'fused')
    _assert_close(got, ref)
    assert got_rng == ref_rng


@pytest.mark.parametrize('kind', ['sbdf1', 'langevin'])
def test_plane_stress_uniform_field(kind):
    pr = _problem((1, 16, 12))
    vec = (.02, -.01, .03)
    ref, _ = _trajectory(pr, kind, 'numpy', E=vec)
    got, _ = _trajectory(pr, kind, 'fused', E=vec)
    _assert_close(got, ref)


def test_plane_stress_is_not_the_default_mode():
    """The two modes must stay far apart on the fused path, and each must match its
    own reference. An el whose z_mode was ignored would keep the reduced Omega with
    the full C, which is neither (measured 2e-2 of max|P| per step before the fix)."""
    pr = _problem((1, 16, 12))
    gps = bulk3d.setup_elastic_3d(pr['grid'], C, Q)
    ps_f, _ = _trajectory(pr, 'sbdf1', 'fused', steps=1)
    ps_r, _ = _trajectory(pr, 'sbdf1', 'numpy', steps=1)
    g_f, _ = _trajectory(pr, 'sbdf1', 'fused', steps=1, el=gps)
    g_r, _ = _trajectory(pr, 'sbdf1', 'numpy', steps=1, el=gps)
    _assert_close(ps_f, ps_r)
    _assert_close(g_f, g_r)
    assert np.max(np.abs(ps_f[0] - g_f[0])) > 1e-6 * np.max(np.abs(ps_f[0]))


def test_plane_stress_ensemble_axis():
    pr = _problem((1, 16, 12))
    a1 = pr['alpha']['a1']
    P = np.stack([pr['P'], .7 * pr['P'] + .01], axis=1)
    batch = dict(pr, P=P, Pm=P + 1e-3, alpha=dict(pr['alpha'], a1=np.stack([a1, 1.1 * a1])),
                 E=(.01, 0., -.02))
    for kind in ('sbdf1', 'sbdf2', 'langevin'):
        ref, ref_rng = _trajectory(batch, kind, 'numpy')
        got, got_rng = _trajectory(batch, kind, 'fused')
        _assert_close(got, ref)
        assert got_rng == ref_rng


def test_uncovered_el_takes_the_ordinary_path_or_refuses():
    pr = _problem((1, 16, 12))
    # A shallow copy that adds a static eigenstrain is refused by the solve, fused too.
    bad = dict(pr['el'], eps_star_static=np.zeros((3, 3, 1, 16, 12)))
    with pytest.raises(ValueError, match='plane_stress'):
        _trajectory(pr, 'sbdf1', 'fused', steps=1, el=bad)
    # An unknown z_mode, or plane stress on a hand-edited Nz > 1 el, stays off the kernels:
    # the fused configuration then gives exactly what plain Numba gives.
    odd = dict(pr['el'], z_mode='other')
    a, _ = _trajectory(pr, 'sbdf1', 'numba', steps=1, el=odd)
    b, _ = _trajectory(pr, 'sbdf1', 'fused', steps=1, el=odd)
    np.testing.assert_array_equal(a[0], b[0])
    thick = _problem((4, 8, 6), z_mode='generalized_plane_strain')
    forced = dict(thick['el'], z_mode='plane_stress', C11_ps=C[0], C12_ps=C[1])
    a, _ = _trajectory(thick, 'sbdf1', 'numba', steps=1, el=forced)
    b, _ = _trajectory(thick, 'sbdf1', 'fused', steps=1, el=forced)
    np.testing.assert_array_equal(a[0], b[0])


def test_restart_inside_fused_is_exact():
    pr = _problem((1, 16, 12), depol_N=.2)
    whole, whole_rng = _trajectory(pr, 'langevin', 'fused', steps=6)
    execution.configure(1, 'numba', fused=True)
    grid, ops = pr['grid'], pr['ops']
    kT = units.kT_langevin(300., grid['hx'], grid['hy'], grid['hz'])
    ou = bulk3d.build_ou_operators_3d(ops, DT, L, kT, noise_cutoff_nm=8., grid=grid)
    rng = np.random.default_rng(11)
    P = pr['P']
    for _ in range(3):
        P = bulk3d.step_langevin_3d(P, pr['alpha'], ops, ou, DT, L, pr['E'], rng, elastic_params=pr['el'])
    saved, state = P.copy(), rng.bit_generator.state
    execution.configure(1, 'numba', fused=True)   # drops every scratch array
    rng = np.random.default_rng(0)
    rng.bit_generator.state = state
    P = saved
    for _ in range(3):
        P = bulk3d.step_langevin_3d(P, pr['alpha'], ops, ou, DT, L, pr['E'], rng, elastic_params=pr['el'])
    np.testing.assert_array_equal(P, whole[-1])
    assert rng.bit_generator.state == whole_rng


def test_nz1_two_axis_transform_is_the_three_axis_one():
    """periodic_step skips the z transform at Nz == 1; that must change no bit."""
    x = np.random.default_rng(5).normal(size=(3, 1, 16, 12))
    a = spfft.rfftn(x, axes=(-3, -2, -1))
    np.testing.assert_array_equal(a, spfft.rfftn(x, axes=(-2, -1)))
    np.testing.assert_array_equal(spfft.irfftn(a, s=(1, 16, 12), axes=(-3, -2, -1)),
                                  spfft.irfftn(a, s=(16, 12), axes=(-2, -1)))
