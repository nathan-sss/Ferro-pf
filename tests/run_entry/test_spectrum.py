"""RUN-ENTRY E3c (SPEC § 16.5): stages ``ac`` and ``fdt``, ``measure.spectrum``, replicas.

a  bit-for-bit against cases/wp13_debye_case.py at reduced length (numpy.array_equal, no tolerance);
b  absolute SI anchor, independent of that code: a purely quadratic Landau energy is an exact Debye relaxator,
   chi_deck(w) = chi0 / (1 - i w tau), chi0 = 1 / (2 a1), tau = 1 / (2 a1 L);
c  the same lock-in check on pf3d;  d  replicas, pooling, resume;  e  refusals;  f  SPEC example 2 through the CLI.
The rename pf2d_dynamic_susceptibility -> dielectric_spectrum is checked at the top.
"""
import copy
import csv
import importlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('MPLBACKEND', 'Agg')

from test_drive import EX, PYTHON, ROOT, _numpy_backend, refused, refuses, run_raw   # noqa: F401
from fftpf import dielectric_spectrum as ds, pf2d_engine, run_input, run_output, run_protocol, units
from fftpf.run_input import ConfigError

CASES = os.path.join(ROOT, 'cases')
EPS0_DECK = 8.8541878128e-12 * 1e9            # the SI vacuum permittivity in deck units, typed by hand

# --------------------------------------------------------------------------- the model of b, c, d: quadratic Landau only

A1, L_MOB = 0.6, 1.0
CHI0, TAU = 1.0 / (2 * A1), 1.0 / (2 * A1 * L_MOB)


def quad_raw(out, N=(8, 8), engine='pf2d', dt=0.02, T=298.0, stages=None, replicas=1, seed=1, a1=A1, eps_b=10.0,
             spectrum=None, checkpoint_every=0, extra_measure=None):
    """Inline material with alpha1 = a1 and every higher coefficient 0, no elasticity: the k = 0 mode is an exact OU process."""
    g = {'engine': engine, 'N': list(N), 'cell_nm': 1.0}
    if engine == 'pf2d':
        g['slab_thickness_nm'] = 1.0
    measure = {'spectrum': spectrum or {'quantities': ['eps_real', 'eps_imag', 'tan_delta'], 'routes': ['lockin']}}
    measure.update(extra_measure or {})
    return {'run': {'name': 'quad', 'seed': seed, 'replicas': replicas}, 'output': {'dir': str(out), 'checkpoint_every': checkpoint_every},
            'material': {'temperature_K': T,
                         'landau': dict(alpha1=a1, alpha11=0.0, alpha12=0.0, alpha111=0.0, alpha112=0.0, alpha123=0.0),
                         'gradient': dict(G110=1.0, G11_G110=0.51, G12_G110=-0.02, G44_G110=0.02, G44P_G110=0.0),
                         'elastic': dict(C11=275.0, C12=179.0, C44=54.3),
                         'electrostriction': dict(Q11_phys=0.11, Q12_phys=-0.045, Q44_phys=0.029)},
            'geometry': g, 'physics': {'electrostatics': {'eps_b': eps_b}, 'elastic': {'k0': 'off'}, 'noise': {'cutoff_nm': 0}},
            'numerics': {'dt': dt}, 'stage': stages, 'measure': measure}


def ac_stage(omega, periods=14, discard=8, E0=0.1, langevin=False, **kw):
    return dict(kind='ac', axis=[1, 0, 0], E0_kV_cm=E0, omega=list(omega), periods=periods, discard_periods=discard,
                langevin=langevin, **kw)


def read_csv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def column(rows, name, route):
    return np.array([float(r[name]) for r in rows if r['route'] == route])


# --------------------------------------------------------------------------- rename: three names, one module

def test_rename_three_names_one_object():
    old, new = importlib.import_module('fftpf.pf2d_dynamic_susceptibility'), importlib.import_module('fftpf.dielectric_spectrum')
    from solvers import dynsus, susceptibility
    assert old is new is dynsus is susceptibility is ds
    for name in ('welch_chi_pp', 'sample_correlation', 'fl_chi', 'kk_chi_prime', 'lockin_chi', 'pool_spectrum', 'debye_chi'):
        assert callable(getattr(ds, name))
    src = open(os.path.join(ROOT, 'fftpf', 'pf2d_dynamic_susceptibility.py')).read()
    assert len(src.splitlines()) <= 12 and 'dielectric_spectrum' in src          # a pure alias file


def test_wp13_case_still_imports(monkeypatch):
    monkeypatch.syspath_prepend(CASES)
    w = importlib.import_module('wp13_debye_case')
    assert w.dynsus is ds


# --------------------------------------------------------------------------- a. bit-for-bit against wp13_debye_case

@pytest.fixture
def wp13(monkeypatch):
    monkeypatch.syspath_prepend(CASES)
    w = importlib.import_module('wp13_debye_case')
    for name, value in dict(N_BURN_BLOCKS=3, BLOCK_LEN=100, N_STEPS_PER_SEED=3000, NPERSEG=512, MAX_LAG=100).items():
        monkeypatch.setattr(w, name, value)
    return w


def wp13_raw(w, out, stages):
    """wp13's model as an entry input: alpha, gradient, kappa, grid, dt, L, T, Dz = h and the zero start of the case."""
    from deck_parser import load_params_toml                     # cases/deck_parser.py, as wp13 reads the deck
    m = load_params_toml(os.path.join(ROOT, 'params', 'bto_298K.toml'))['materials']
    eps_b = w.KAPPA / units.relative_to_deck_permittivity(1.0)
    assert units.relative_to_deck_permittivity(eps_b) == w.KAPPA          # the very same float kappa
    routes = ['lockin'] if stages[-1]['kind'] == 'ac' else ['fdt']
    raw = quad_raw(out, N=(w.N, w.N), dt=w.DT, a1=w.A1, eps_b=eps_b, stages=stages,
                   spectrum={'quantities': ['eps_real'], 'routes': routes})
    raw['geometry'].update(cell_nm=w.H, slab_thickness_nm=w.H)
    raw['material']['gradient'] = dict(G110=m['G110'], G11_G110=m['G11_G110'], G12_G110=m['G12_G110'],
                                       G44_G110=m['G44_G110'], G44P_G110=m['G44P_G110'])
    raw['material']['temperature_K'] = w.T_KELVIN
    raw['numerics']['L'] = w.L_MOB
    return raw


def wp13_stages(w, tail):
    return [{'kind': 'anneal', 'start': 'zero', 'stop': {'n_steps': w.N_BURN_BLOCKS * w.BLOCK_LEN}}, tail]


def test_fdt_bit_for_bit_wp13(wp13, tmp_path, monkeypatch):
    w, seed = wp13, 5001
    tail = {'kind': 'fdt', 'n_steps': w.N_STEPS_PER_SEED, 'nperseg': w.NPERSEG, 'max_lag': w.MAX_LAG, 'axis': [1, 0, 0]}
    raw = wp13_raw(w, tmp_path / 'fdt', wp13_stages(w, tail))
    raw['run']['seed'] = seed
    res = run_raw(raw)
    b = res['build']
    assert b['kT'] == w.KT and b['dt'] == w.DT
    last = []
    with monkeypatch.context() as m:                                       # the reference exposes its last P by this hook only
        step = w.solver.step_langevin
        m.setattr(w.solver, 'step_langevin', lambda *a, **k: last.append(step(*a, **k)) or last[-1])
        _, omega_ref, chi_pp_ref, C_ref = w._run_one_fdt_seed(seed)
    x = res['stages'][1]['series']['fdt']['series']
    omega, chi_pp = ds.welch_chi_pp(x, w.DT, w.KT, nperseg=w.NPERSEG, window='hann')
    assert np.array_equal(omega, omega_ref) and np.array_equal(chi_pp, chi_pp_ref)
    assert np.array_equal(ds.sample_correlation(x, w.MAX_LAG), C_ref)
    assert np.array_equal(res['P'], last[-1])                                # final P of the trajectory
    # the entry's own numbers are the same estimator with kT x N_cells (per volume)
    z = np.load(os.path.join(res['run_dir'], 'spectrum_1_fdt.npz'))
    N_cells = w.N * w.N
    assert float(z['kT_eff']) == N_cells * w.KT and np.array_equal(z['omega'], omega_ref)
    assert np.array_equal(z['chi_pp'], ds.welch_chi_pp(x, w.DT, N_cells * w.KT, nperseg=w.NPERSEG)[1])
    assert np.allclose(z['chi_pp'] * N_cells, chi_pp_ref, rtol=1e-12, atol=0)
    assert np.array_equal(z['C'], C_ref)
    assert float(z['chi_static']) == C_ref[0] / (N_cells * w.KT)


def test_ac_bit_for_bit_wp13(wp13, tmp_path):
    w, seed, omega, n_periods, E0_kV = wp13, 5002, 0.3, 2, 0.5
    tail = ac_stage([omega], periods=n_periods, discard=0, E0=E0_kV, langevin=True)
    raw = wp13_raw(w, tmp_path / 'ac', wp13_stages(w, tail))
    raw['run']['seed'] = seed
    res = run_raw(raw)
    E0 = run_protocol.deck_field(E0_kV)                                    # the deck amplitude the entry applies
    chi_p, chi_pp = w.run_ac_seed((omega, E0, n_periods, seed))
    got = res['stages'][1]['spectrum']['lockin']
    assert got['chi_p'][0] == chi_p and got['chi_pp'][0] == chi_pp         # exactly equal, no tolerance
    x = np.load(os.path.join(res['run_dir'], 'series_1_ac_w0.3.npz'))
    assert x['series'].size == int(n_periods * (2 * np.pi / omega) / w.DT) and float(x['N_cells']) == w.N * w.N
    assert tuple(ds.lockin_chi(x['series'], w.DT, omega, w.N * w.N * E0)) == (chi_p, chi_pp)


# --------------------------------------------------------------------------- b. absolute anchor: the exact Debye relaxator

def test_engine_convention_factor_two():
    """The bulk force of alpha1 alone is 2 a1 P and the energy density is a1 P^2 (so chi0 = 1 / (2 a1)); wp13's Debye form agrees."""
    P = np.full((3, 4, 4), 0.2)
    P[1:] = 0.0
    alpha = dict(a1=A1, a11=0.0, a12=0.0, a111=0.0, a112=0.0, a123=0.0)
    assert np.allclose(pf2d_engine.bulk_force(P, alpha)[0], 2 * A1 * 0.2, rtol=1e-14)
    assert np.allclose(pf2d_engine.bulk_force(P, alpha)[1:], 0.0)
    f = lambda p: A1 * p * p                                                # f_bulk(P) = a1 |P|^2, hence dF/dP = 2 a1 P
    h = 1e-4
    assert abs((f(0.2 + h) - f(0.2 - h)) / (2 * h) - 2 * A1 * 0.2) < 1e-9
    w = 0.7
    assert np.allclose(ds.debye_chi(w, CHI0, TAU), CHI0 / (1 - 1j * w * TAU), rtol=1e-15)


def lockin_errors(tmp_path, dt, engine='pf2d', N=(8, 8), omega=(0.3, 1.2, 5.0), periods=14, discard=8, name='det'):
    raw = quad_raw(tmp_path / f'{name}_{dt:g}', N=N, engine=engine, dt=dt,
                   stages=[dict(ac_stage(omega, periods, discard), start='zero')])
    res = run_raw(raw)
    s = res['stages'][0]['spectrum']['lockin']
    chi = s['chi_p'] + 1j * s['chi_pp']
    gamma = 2 * A1 * L_MOB * dt
    exact = np.array([CHI0 / (1 - 1j * w * TAU) for w in omega])
    # the scheme itself: explicit Euler drift + exact OU on the k = 0 mode (Lin(0) = 0), E sampled at t_i, record after the step
    discrete = dt * L_MOB / (1 - np.exp(1j * np.array(omega) * dt) * (1 - gamma))
    return res, np.abs(chi / exact - 1), np.abs(chi / discrete - 1)


def test_deterministic_lockin_matches_the_debye_form_and_the_error_shrinks_with_dt(tmp_path):
    omega, kept = (0.3, 1.2, 5.0), 14 - 8
    err = {}
    for dt in (0.02, 0.01):
        _, err[dt], to_scheme = lockin_errors(tmp_path, dt, omega=omega)
        w = np.array(omega)
        # bound 1: |chi_measured / Debye - 1| <= 1.5 w dt (first-order scheme: explicit Euler drift, E sampled at the step start)
        assert np.all(err[dt] <= 1.5 * w * dt), (dt, err[dt], 1.5 * w * dt)
        # bound 2: the closed form of the scheme itself, up to the lock-in window (a non-integer number of periods, trapezoid)
        assert np.all(to_scheme <= 3 * w * dt / (2 * np.pi * kept)), (dt, to_scheme)
    assert np.all(err[0.01] <= err[0.02] / 1.8), (err[0.02], err[0.01])      # halving dt halves the error (first order)


def test_deterministic_lockin_eps_columns_against_a_hand_computation(tmp_path):
    res, err, _ = lockin_errors(tmp_path, 0.01, omega=(0.3, 1.2), name='eps')
    rows = read_csv(os.path.join(res['run_dir'], 'spectrum.csv'))
    w = column(rows, 'omega', 'lockin')
    cr, ci = column(rows, 'chi_real', 'lockin'), column(rows, 'chi_imag', 'lockin')
    er, ei, td = (column(rows, k, 'lockin') for k in ('eps_real', 'eps_imag', 'tan_delta'))
    assert np.allclose(er, 10.0 + cr / EPS0_DECK, rtol=1e-12) and np.allclose(ei, ci / EPS0_DECK, rtol=1e-12)
    assert np.allclose(td, ei / er, rtol=1e-12)
    want = 10.0 + (CHI0 / (1 - 1j * w * TAU)) / EPS0_DECK                      # eps_b + Debye chi in SI
    mag = np.abs(want - 10.0)                                                   # |chi| in SI: the scale of both parts
    assert np.all(np.abs(er - want.real) <= 1.5 * w * 0.01 * mag) and np.all(np.abs(ei - want.imag) <= 1.5 * w * 0.01 * mag)
    assert 'f_Hz' not in rows[0]                                                # time is not calibrated: no Hz column
    m = json.load(open(os.path.join(res['run_dir'], 'measures.json')))
    assert m['spectrum']['routes']['lockin']['eps_b'] == 10.0


def test_hz_column_only_with_seconds_per_deck_time(tmp_path):
    raw = quad_raw(tmp_path / 'hz', stages=[dict(ac_stage((1.2,), 6, 2), start='zero')], extra_measure={'seconds_per_deck_time': 1e-12})
    rows = read_csv(os.path.join(run_raw(raw)['run_dir'], 'spectrum.csv'))
    assert list(rows[0])[:3] == ['route', 'omega', 'f_Hz'] and abs(float(rows[0]['f_Hz']) - 1.2 / (2 * np.pi * 1e-12)) < 1e-3


def test_discarded_periods_are_dropped_with_the_clock_turned_back(tmp_path):
    """Seven of ten periods discarded: the kept window starts 7 periods + a fraction late, and the result still equals the
    closed form of the scheme (a wrong clock would be off by omega * n_skip * dt radians)."""
    w, dt, kept = 1.2, 0.02, 3
    b = run_raw(quad_raw(tmp_path / 'd7', dt=dt, stages=[dict(ac_stage((w,), 10, 7), start='zero')]))
    s = b['stages'][0]['spectrum']['lockin']
    gamma = 2 * A1 * L_MOB * dt
    scheme = dt * L_MOB / (1 - np.exp(1j * w * dt) * (1 - gamma))
    assert abs((s['chi_p'][0] + 1j * s['chi_pp'][0]) / scheme - 1) < 3 * w * dt / (2 * np.pi * kept)
    sa = np.load(os.path.join(b['run_dir'], 'series_0_ac_w1.2.npz'))
    n_skip = int(7 * (2 * np.pi / w) / dt)
    assert int(sa['n_skip']) == n_skip and sa['series'].size == n_skip + int(3 * (2 * np.pi / w) / dt)
    assert (w * n_skip * dt) % (2 * np.pi) > 0.05                    # the fractional-period offset the rotation has to undo
    a = run_raw(quad_raw(tmp_path / 'd0', dt=dt, stages=[dict(ac_stage((w,), 10, 0), start='zero')]))['stages'][0]['spectrum']['lockin']
    assert abs((a['chi_p'][0] + 1j * a['chi_pp'][0]) / scheme - 1) < 0.05            # no discard: the start transient is in


@pytest.fixture(scope='module')
def fdt_runs(tmp_path_factory):
    """FDT with 6 replicas on 8 x 8 and 16 x 16: static limit and spectrum of the per-volume chi_deck."""
    from solvers import execution
    execution.configure(1, 'numpy')
    out = {}
    for n in (8, 16):
        d = tmp_path_factory.mktemp(f'fdt{n}')
        stages = [{'kind': 'anneal', 'start': 'zero', 'stop': {'n_steps': 1500}},
                  {'kind': 'fdt', 'n_steps': 20000, 'nperseg': 2048, 'axis': [1, 0, 0]}]
        raw = quad_raw(d / 'run', N=(n, n), dt=0.025, stages=stages, replicas=6, seed=100,
                       spectrum={'quantities': ['eps_real', 'eps_imag'], 'routes': ['fdt']})
        out[n] = run_raw(raw)['run_dir']
    yield out
    execution.configure(workers=1, backend='numpy')


def t_stat(x):
    """Mean over replicas and its t = mean / standard error (ddof = 1)."""
    x = np.asarray(x)
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def test_fdt_replicas_match_the_debye_relaxator(fdt_runs):
    """chi_deck(0) = N Var(mean p) / kT = chi0 within 3 standard errors (gamma = 2 a1 L dt = 0.03: the explicit scheme adds
    gamma / 2 = 1.5 % to the variance, inside the error), and the spectrum follows chi0 / (1 - i w tau): the band
    averages of chi' (lowest 8 Welch bins) and chi'' (0.25 < w < 1.8; the lowest bins are cut by the per-segment mean) over
    the replicas differ from the Debye values by less than 3 standard errors of the replica spread."""
    for n, d in fdt_runs.items():
        m = json.load(open(os.path.join(d, 'measures.json')))['spectrum']['routes']['fdt']
        st = m['chi_static_deck']
        assert st['n'] == 6 and abs(st['mean'] - CHI0) < 3 * st['se'], (n, st)
        reps = [np.load(os.path.join(d, f'rep_{r}', 'spectrum_1_fdt.npz')) for r in range(6)]
        w = reps[0]['omega']
        ex = CHI0 / (1 - 1j * w * TAU)
        band = (w > 0.25) & (w < 1.8)
        low = np.arange(len(w)) < 8
        t_p = t_stat([(r['chi_p'] - ex.real)[low].mean() for r in reps])[1]
        t_pp = t_stat([(r['chi_pp'] - ex.imag)[band].mean() for r in reps])[1]
        assert abs(t_p) < 3 and abs(t_pp) < 3, (n, t_p, t_pp)
        rows = read_csv(os.path.join(d, 'spectrum.csv'))
        pw = column(rows, 'omega', 'fdt')
        assert np.array_equal(pw, w[:len(pw)]) and len(pw) >= 8 and w[0] > 0 and m['resolved_from'] == 'replicas'
        assert np.allclose(column(rows, 'chi_real', 'fdt'), np.mean([r['chi_p'] for r in reps], axis=0)[:len(pw)], rtol=1e-12, atol=0)


def test_per_volume_chi_is_independent_of_the_grid(fdt_runs):
    """The N_cells factor is gone: 8 x 8 (64 cells) and 16 x 16 (256 cells) give the same chi_deck within their errors."""
    s = [json.load(open(os.path.join(d, 'measures.json')))['spectrum']['routes']['fdt']['chi_static_deck'] for d in fdt_runs.values()]
    assert abs(s[0]['mean'] - s[1]['mean']) < 3 * np.hypot(s[0]['se'], s[1]['se'])
    # a missing N_cells (or a squared one) would show here: 64 x or 1/4 x
    assert 0.5 < s[0]['mean'] / s[1]['mean'] < 2.0
    z = [np.load(os.path.join(d, 'rep_0', 'series_1_fdt.npz')) for d in fdt_runs.values()]
    assert [int(a['N_cells']) for a in z] == [64, 256]
    # Var(s) = N kT / stiffness: the stored raw series carries N, the estimator divides it out
    for a in z:
        assert abs(np.var(a['series']) / (int(a['N_cells']) * float(a['kT'])) / CHI0 - 1) < 0.25


# --------------------------------------------------------------------------- c. pf3d

def test_pf3d_deterministic_lockin_is_debye(tmp_path):
    omega, dt = (1.2, 5.0), 0.02
    raw = quad_raw(tmp_path / 'p3', N=(6, 6, 4), engine='pf3d', dt=dt, stages=[dict(ac_stage(omega, 14, 8), start='zero')])
    res = run_raw(raw)
    s = res['stages'][0]['spectrum']['lockin']
    exact = np.array([CHI0 / (1 - 1j * w * TAU) for w in omega])
    err = np.abs((s['chi_p'] + 1j * s['chi_pp']) / exact - 1)
    assert np.all(err <= 1.5 * np.array(omega) * dt), err
    assert int(np.load(os.path.join(res['run_dir'], 'series_0_ac_w1.2.npz'))['N_cells']) == 144


# --------------------------------------------------------------------------- d. replicas

def replica_raw(out, checkpoint_every=0):
    stages = [{'kind': 'anneal', 'start': 'zero', 'stop': {'n_steps': 300}},
              ac_stage((1.0, 3.0), periods=4, discard=1, E0=200.0, langevin=True),
              {'kind': 'fdt', 'n_steps': 4000, 'nperseg': 256, 'axis': [1, 0, 0]}]
    return quad_raw(out, stages=stages, replicas=3, seed=7, dt=0.02, checkpoint_every=checkpoint_every,
                    spectrum={'quantities': ['eps_real', 'eps_imag', 'tan_delta'], 'routes': ['lockin', 'fdt']})


def test_replicas_write_rep_dirs_and_the_pooled_mean(tmp_path):
    res = run_raw(replica_raw(tmp_path / 'r3'))
    top = res['run_dir']
    assert sorted(f for f in os.listdir(top) if f.startswith('rep_')) == ['rep_0', 'rep_1', 'rep_2']
    for f in ('spectrum.csv', 'measures.json', 'resolved.toml', 'replicas.json', os.path.join('figures', 'spectrum.png')):
        assert os.path.isfile(os.path.join(top, f)), f
    meta = json.load(open(os.path.join(top, 'replicas.json')))
    assert meta['seeds'] == [7, 8, 9]
    assert [json.load(open(os.path.join(top, f'rep_{r}', 'provenance.json')))['seed'] for r in range(3)] == [7, 8, 9]
    reps = [read_csv(os.path.join(top, f'rep_{r}', 'spectrum.csv')) for r in range(3)]
    pooled = read_csv(os.path.join(top, 'spectrum.csv'))
    assert 'eps_real_se' in pooled[0] and 'eps_real_se' not in reps[0][0]
    for name in ('omega', 'chi_real', 'chi_imag', 'eps_real', 'eps_imag', 'tan_delta'):
        p = column(pooled, name, 'lockin')
        assert np.allclose(p, np.mean([column(r, name, 'lockin') for r in reps], axis=0), rtol=1e-12, atol=0), name
        if name != 'omega':
            sd = np.std([column(r, name, 'lockin') for r in reps], axis=0, ddof=1) / np.sqrt(3)
            assert np.allclose(column(pooled, name + '_se', 'lockin'), sd, rtol=1e-10, atol=0), name
    # the fdt rows of the pooled table are the mean of the replicas' chi' / chi'' on the shared Welch grid
    pw = column(pooled, 'omega', 'fdt')
    assert len(pw) > 0
    for name in ('chi_real', 'chi_imag'):
        full = [np.load(os.path.join(top, f'rep_{r}', 'spectrum_2_fdt.npz'))[{'chi_real': 'chi_p', 'chi_imag': 'chi_pp'}[name]]
                for r in range(3)]
        assert np.allclose(column(pooled, name, 'fdt'), np.mean(full, axis=0)[:len(pw)], rtol=1e-12, atol=0)
    assert not os.path.isdir(os.path.join(top, 'rep_3'))


def test_interrupted_replicated_run_resumes_to_the_same_spectrum(tmp_path):
    whole = run_raw(replica_raw(tmp_path / 'whole', checkpoint_every=1))['run_dir']
    raw = replica_raw(tmp_path / 'cut', checkpoint_every=1)
    with pytest.raises(run_output.RunInterrupted):
        run_raw(raw, stop_after_checkpoints=5)              # replica 0 writes 3 checkpoints; replica 1 stops after its second
    cut = str(tmp_path / 'cut')
    assert os.path.isfile(os.path.join(cut, 'rep_0', 'measures.json')) and not os.path.isfile(os.path.join(cut, 'rep_1', 'measures.json'))
    assert not os.path.isdir(os.path.join(cut, 'rep_2')) and not os.path.isfile(os.path.join(cut, 'spectrum.csv'))
    rep0 = os.path.getmtime(os.path.join(cut, 'rep_0', 'measures.json'))
    res = run_raw(raw)
    assert [r['skipped'] for r in res['replicas']] == [True, False, False]
    assert os.path.getmtime(os.path.join(cut, 'rep_0', 'measures.json')) == rep0           # the finished replica was not touched
    for f in ('spectrum.csv', os.path.join('rep_1', 'spectrum.csv'), os.path.join('rep_1', 'trace.csv'),
              os.path.join('rep_2', 'spectrum.csv')):
        assert open(os.path.join(cut, f), 'rb').read() == open(os.path.join(whole, f), 'rb').read(), f
    other = replica_raw(tmp_path / 'cut')
    other['run']['seed'] = 99
    with pytest.raises(ConfigError, match='not a replicated run of this configuration'):
        run_raw(other)


# --------------------------------------------------------------------------- e. refusals

def spec_raw(tmp_path, stages, **kw):
    return quad_raw(tmp_path / 'ref', stages=stages, **kw)


def test_refusals(tmp_path):
    fdt = {'kind': 'fdt', 'n_steps': 400, 'nperseg': 64, 'axis': [1, 0, 0]}
    ac = dict(ac_stage((1.0,), 4, 1), start='zero')
    both = {'quantities': ['eps_real'], 'routes': ['lockin', 'fdt']}
    msg = refuses(spec_raw(tmp_path, [ac, dict(fdt, langevin=False)], spectrum=both), 'stage[1].langevin')
    assert 'Langevin' in msg
    refuses(_no_temperature(tmp_path, [ac, fdt], both), 'stage[1].langevin')                          # no temperature, no Langevin
    refuses(spec_raw(tmp_path, [ac], spectrum=both), 'measure.spectrum.routes')                       # fdt route, no fdt stage
    refuses(spec_raw(tmp_path, [fdt], spectrum={'quantities': ['eps_real'], 'routes': ['lockin']}), 'measure.spectrum.routes')
    refuses(spec_raw(tmp_path, [dict(ac, omega=[1.0, 0.0])]), 'stage[0].omega[1]')
    refuses(spec_raw(tmp_path, [dict(ac, omega=[-1.0])]), 'stage[0].omega[0]')
    refuses(spec_raw(tmp_path, [dict(ac, periods=4, discard_periods=4)]), 'stage[0].discard_periods')
    refuses(spec_raw(tmp_path, [dict(ac, periods=4, discard_periods=9)]), 'stage[0].discard_periods')
    refuses(spec_raw(tmp_path, [dict(ac, omega=[1.0, 1.0000001])]), 'stage[0].omega')
    refuses(spec_raw(tmp_path, [dict(fdt, nperseg=500)], spectrum={'quantities': ['eps_real'], 'routes': ['fdt']}), 'stage[0].nperseg')
    refuses(spec_raw(tmp_path, [dict(fdt, max_lag=300)], spectrum={'quantities': ['eps_real'], 'routes': ['fdt']}), 'stage[0].max_lag')
    refuses(spec_raw(tmp_path, [ac, ac, fdt], spectrum=both), 'measure.spectrum.routes')               # one stage per route
    plain = spec_raw(tmp_path, [dict(kind='anneal', stop={'n_steps': 5})], replicas=2)
    plain['measure'] = {}
    refuses(plain, 'run.replicas')                                                                    # nothing to pool
    no_axis = {k: v for k, v in fdt.items() if k != 'axis'}
    refuses(spec_raw(tmp_path, [no_axis], spectrum={'quantities': ['eps_real'], 'routes': ['fdt']}), 'stage[0].axis')
    assert run_input.resolve(spec_raw(tmp_path, [ac, no_axis], spectrum=both))['stage'][1]['axis'] == [1, 0, 0]   # follows the ac axis


def _no_temperature(tmp_path, stages, spectrum):
    raw = spec_raw(tmp_path, stages, spectrum=spectrum)
    del raw['material']['temperature_K']
    return raw


def test_ac_and_fdt_on_film_are_refused_without_an_anchor(tmp_path):
    film = {'run': {'name': 'f'}, 'material': {'deck': 'bto_lcc2005_paper', 'temperature_K': 300.0},
            'geometry': {'engine': 'film', 'N': [8, 8, 4], 'film_thickness_nm': 20.0},
            'physics': {'elastic': {'misfit_strain_pct': -0.5}},
            'stage': [{'kind': 'ac', 'axis': [1, 0, 0], 'E0_kV_cm': 1.0, 'omega': [1.0], 'periods': 4}]}
    assert 'no anchor case' in refuses(film, 'stage[0].kind')
    film['stage'] = [{'kind': 'fdt', 'n_steps': 400, 'nperseg': 64, 'axis': [1, 0, 0]}]
    refuses(film, 'stage[0].kind')


# --------------------------------------------------------------------------- f. SPEC example 2 and the figure

def test_spec_example_2_through_the_cli():
    path = os.path.join(EX, 'dielectric_spectrum.toml')
    env = dict(os.environ, OMP_NUM_THREADS='1', MPLBACKEND='Agg')
    for cmd in ('check', 'explain'):
        out = subprocess.run([PYTHON, '-m', 'fftpf', cmd, path], capture_output=True, text=True, cwd=ROOT, env=env)
        assert out.returncode == 0, out.stderr
        small = subprocess.run([PYTHON, '-m', 'fftpf', cmd, os.path.join(EX, 'debye_quadratic.toml')],
                               capture_output=True, text=True, cwd=ROOT, env=env)
        assert small.returncode == 0, small.stderr
    assert 'replicas: 8' in out.stdout and 'stage[1] ac' in out.stdout and 'stage[2] fdt' in out.stdout
    cfg = run_input.load(path)
    assert cfg['stage'][2]['max_lag'] == 8192 and cfg['stage'][2]['burn_steps'] == 0 and cfg['stage'][2]['axis'] == [1.0, 0.0, 0.0]
    assert run_input.resolve(__import__('tomllib').loads(run_input.dumps(cfg)), base_dir=EX) == cfg      # resolved.toml round trip


def test_spectrum_figure_is_a_png_with_both_routes(tmp_path):
    res = run_raw(replica_raw(tmp_path / 'fig'))
    for png in (os.path.join(res['run_dir'], 'figures', 'spectrum.png'), os.path.join(res['run_dir'], 'rep_0', 'figures', 'spectrum.png')):
        data = open(png, 'rb').read()
        assert data[:8] == b'\x89PNG\r\n\x1a\n' and len(data) > 5000
        import matplotlib.image as mpimg
        img = mpimg.imread(png)
        assert img.ndim == 3 and img.shape[0] > 300 and img.shape[1] > 300


def test_figure_axis_label_is_reduced_unless_calibrated(tmp_path, monkeypatch):
    from fftpf import run_figures
    seen = []
    import matplotlib.axes
    orig = matplotlib.axes.Axes.set_xlabel
    monkeypatch.setattr(matplotlib.axes.Axes, 'set_xlabel', lambda self, label, *a, **k: seen.append(label) or orig(self, label, *a, **k))
    t = {'lockin': {'omega': np.array([1.0, 2.0]), 'f_Hz': np.array([1.0, 2.0]), 'eps_real': np.array([1.0, 2.0])}}
    run_figures.spectrum(str(tmp_path / 'a.png'), t, ['eps_real'], 't', calibrated=False)
    run_figures.spectrum(str(tmp_path / 'b.png'), t, ['eps_real'], 't', calibrated=True)
    assert seen == ['reduced frequency (1 / deck time)', 'frequency (Hz)']
