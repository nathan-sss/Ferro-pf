"""RUN-ENTRY E2 / E3a (SPEC § 16.5): builder equality and bit-for-bit anchors.

"Bit-for-bit" = the existing path and the entry path run in this process on the same inputs and
numpy.array_equal holds on the final P and on every recorded energy number.
"""
import copy
import os
import subprocess
import sys
import tomllib
from argparse import Namespace

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from cases import bto_film_biaxial as bfb
from fftpf import (casekit, disorder_fields, film_engine, mat_registry, pf2d_engine, pf3d_engine,
                   random_field_mechanisms, run_builder, run_input, run_protocol, state_io, timestep_verify, units)
from fftpf.run_input import ConfigError
from solvers import execution

EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples')
PYTHON = sys.executable
DEMO_T_TOTAL = 400.0


@pytest.fixture(autouse=True)
def _numpy_backend():
    execution.configure(1, 'numpy')
    yield
    execution.configure(workers=1, backend='numpy')


def tree_equal(a, b, path='root'):
    """Every ndarray / number / string of two nested dicts equal (arrays by array_equal)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f'{path}: keys differ'
        for k in a:
            tree_equal(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), f'{path}: arrays differ'
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            tree_equal(x, y, f'{path}[{i}]')
    elif isinstance(a, (int, float, str, bool, type(None), np.generic)):
        assert a == b or (a != a and b != b), f'{path}: {a!r} != {b!r}'


def stage_raw(tmp, name='run', **over):
    raw = {'run': {'name': name, 'seed': 0}, 'output': {'dir': str(tmp / name)}}
    raw.update(copy.deepcopy(over))
    return raw


def run_raw(raw):
    return run_protocol.run(run_input.resolve(raw))


# --------------------------------------------------------------------------- a. builder equality

def test_builder_equals_casekit_2d_and_3d_and_film(tmp_path):
    deck = mat_registry.get('bto_lcc2005_8th', T=298.0)
    mat = {'deck': 'bto_lcc2005_8th', 'temperature_K': 298.0}
    st = [{'kind': 'relax'}]
    # pf2d, elastic off: casekit.build_2d
    b = run_builder.build(run_input.resolve(stage_raw(
        tmp_path, material=mat, geometry={'engine': 'pf2d', 'N': [12, 12], 'cell_nm': 0.4},
        physics={'elastic': {'k0': 'off'}}, numerics={'dt': 0.02}, stage=st)))
    grid, ops = casekit.build_2d(deck, 12, 12, 0.4)
    tree_equal(grid, b['grid'])
    tree_equal(ops, b['ops'])
    assert b['el'] is None and b['kT'] is None and b['E_static'] == (0.0, 0.0, 0.0)
    tree_equal(deck['landau'], b['alpha'])
    # pf3d, elastic on, plane-strain default: casekit.build_3d (the Q44_phys / 4 trap)
    b = run_builder.build(run_input.resolve(stage_raw(
        tmp_path, material=mat, geometry={'engine': 'pf3d', 'N': [8, 8, 8], 'cell_nm': 1.0},
        numerics={'dt': 0.02}, stage=st)))
    grid, ops, el = casekit.build_3d(deck, 8, 1.0, elastic=True, k0='stress_free')
    tree_equal(grid, b['grid'])
    tree_equal(ops, b['ops'])
    tree_equal(el, b['el'])
    # film: cases/bto_film_biaxial.setup
    a = Namespace(nx=8, ny=8, nz=5, lx_nm=3.2, ly_nm=3.2, thickness_nm=1.6, temperature_c=25.0, strain_pct=-0.5,
                  eps_b=10.0, substrate_layers=3, substrate_material='ss316_rt')
    deck_f, grid_f, el_f, kappa = bfb.setup(a)
    raw = stage_raw(tmp_path, material={'deck': 'bto_lcc2005_paper', 'temperature_K': 25.0 + 273.15},
                    geometry={'engine': 'film', 'N': [8, 8, 5], 'cell_nm': 0.4, 'film_thickness_nm': 1.6},
                    physics={'electrostatics': {'eps_b': 10.0},
                             'elastic': {'misfit_strain_pct': -0.5, 'substrate_layers': 3, 'substrate': 'ss316_rt'}},
                    numerics={'dt': 0.01}, stage=st)
    b = run_builder.build(run_input.resolve(raw))
    tree_equal(grid_f, b['grid'])
    tree_equal(el_f, b['el'])
    assert b['kappa'] == kappa and b['film']['G'] == deck_f['gradient']['G11']
    tree_equal(deck_f['landau'], b['alpha'])


def test_gaussian_fields_equal_disorder_fields(tmp_path):
    base = dict(material={'deck': 'bto_lcc2005_8th', 'temperature_K': 298.0}, numerics={'dt': 0.02},
                geometry={'engine': 'pf3d', 'N': [6, 6, 4], 'cell_nm': 1.0}, stage=[{'kind': 'relax'}])
    fld = [{'target': 'E_ext', 'kind': 'gaussian', 'sigma_kV_cm': 30.0, 'correlation_nm': 3.0, 'seed': 11},
           {'target': 'landau.alpha1', 'kind': 'gaussian', 'sigma': 0.01, 'correlation_nm': 2.0, 'seed': 5}]
    b = run_builder.build(run_input.resolve(stage_raw(tmp_path, field=fld, **base)))
    g = b['grid']
    rms = units.si_to_deck_field(30.0 * 1e5)
    want = np.stack([disorder_fields.random_field_gaussian_3d(g, rms, 3.0, 11 + c) for c in range(3)])
    assert np.array_equal(b['E_static'], want)
    a1 = mat_registry.get('bto_lcc2005_8th', T=298.0)['landau']['a1']
    assert np.array_equal(b['alpha']['a1'], a1 + disorder_fields.random_field_gaussian_3d(g, 0.01, 2.0, 5))
    # 2D uses the 2D generator
    base['geometry'] = {'engine': 'pf2d', 'N': [8, 6], 'cell_nm': 1.0}
    b = run_builder.build(run_input.resolve(stage_raw(tmp_path, field=fld[:1], physics={'elastic': {'k0': 'off'}}, **base)))
    want = np.stack([disorder_fields.random_field_gaussian(b['grid'], rms, 3.0, 11 + c) for c in range(3)])
    assert np.array_equal(b['E_static'], want)


def test_no_cutoff_and_inline_material(tmp_path):
    deck = mat_registry.get('bto_lcc2005_8th', T=298.0)
    inv = {v: k for k, v in mat_registry._ALPHA_TO_ENGINE.items()}
    mat = {'landau': {inv[k]: v for k, v in deck['landau'].items()},
           'gradient': {'G110': 1.0, 'G11_G110': 0.51, 'G12_G110': -0.02, 'G44_G110': 0.02, 'G44P_G110': 0.0},
           'elastic': dict(deck['elastic']), 'electrostriction': dict(deck['electrostriction']),
           'temperature_K': 298.0}
    b = run_builder.build(run_input.resolve(stage_raw(
        tmp_path, material=mat, geometry={'engine': 'pf2d', 'N': [8, 8], 'cell_nm': 0.4, 'slab_thickness_nm': 1.0},
        physics={'elastic': {'k0': 'off'}, 'noise': {'cutoff_nm': 0}}, numerics={'dt': 0.02},
        stage=[{'kind': 'anneal', 'stop': {'n_steps': 3}}])))
    tree_equal(deck['gradient'], b['deck']['gradient'])
    tree_equal(deck['landau'], b['alpha'])
    assert b['noise_cutoff_nm'] is None and b['kT'] == units.kT_langevin(298.0, 0.4, 0.4, 1.0)
    assert b['kappa'] == units.relative_to_deck_permittivity(10.0)      # inline deck, eps_b absent


# --------------------------------------------------------------------------- b. demo_casekit_2d anchor

@pytest.fixture(scope='module')
def demo2d(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('demo2d')
    deck = mat_registry.get('bto_lcc2005_8th', T=298.0)
    grid, ops = casekit.build_2d(deck, Nx=64, Ny=64, h=0.4)
    E_d = random_field_mechanisms.charge_disorder_field(
        grid, conc=0.5, delta_q=3.0, lambda_s_nm=2.0, eps_r=1000.0, a_lattice_nm=0.4049, seed=1)
    np.savez(tmp / 'E_d.npz', E_d=E_d)
    P0 = pf2d_engine.random_smooth_field(grid, 3, amplitude=0.005, seed=0)
    dt, _ = timestep_verify.verified_dt(pf2d_engine, P0, deck['landau'], ops, grid, L=1.0, E_ext=E_d, T_total=DEMO_T_TOTAL)
    return dict(tmp=tmp, deck=deck, grid=grid, ops=ops, E_d=E_d, P0=P0, dt=dt)


def demo2d_raw(d, dt, name, n_max=300):
    return stage_raw(
        d['tmp'], name, material={'deck': 'bto_lcc2005_8th', 'temperature_K': 298.0},
        geometry={'engine': 'pf2d', 'N': [64, 64], 'cell_nm': 0.4}, physics={'elastic': {'k0': 'off'}},
        numerics={'dt': dt, 'dt_verify_time': DEMO_T_TOTAL},
        field=[{'target': 'E_ext', 'kind': 'from_file', 'file': str(d['tmp'] / 'E_d.npz'), 'key': 'E_d'}],
        stage=[{'kind': 'anneal', 'langevin': False, 'sample_every': 50,
                'start': {'kind': 'random', 'amplitude': 0.005, 'kmax_frac': 0.3, 'seed': 0},
                'stop': {'block_stationary': 0.01, 'n_max': n_max}}])


@pytest.mark.parametrize('variant', ['fixed', 'verified'])
def test_demo_casekit_2d_anchor(demo2d, variant):
    d = demo2d
    dt = 0.1 if variant == 'fixed' else 'verified'
    res = run_raw(demo2d_raw(d, dt, variant))
    if variant == 'verified':
        assert res['build']['dt'] == d['dt']                    # the identical float
        dt = d['dt']
    P, trace, why, state = casekit.anneal(pf2d_engine, d['P0'], d['deck']['landau'], d['ops'], d['grid'], dt=dt,
                                          E_ext=d['E_d'], n_max=300, sample_every=50, return_state=True)
    st = res['stages'][0]
    assert np.array_equal(st['P'], P)
    assert st['trace'] == trace and len(trace) == 6
    assert st['stop_reason'] == why and st['step'] == state['step'] and st['t'] == state['t']
    assert np.array_equal(res['P'], P)


# --------------------------------------------------------------------------- c. demo_casekit_3d anchor

@pytest.fixture(scope='module')
def demo3d():
    deck = mat_registry.get('bto_lcc2005_8th', T=298.0)
    grid, ops, el = casekit.build_3d(deck, N=16, d=1.0, elastic=True, k0='stress_free')
    P0 = pf3d_engine.random_smooth_field_3d(grid, 3, amplitude=0.005, kmax_frac=0.5, seed=0)
    dt, _ = timestep_verify.verified_dt(pf3d_engine, P0, deck['landau'], ops, grid, L=1.0, T_total=DEMO_T_TOTAL,
                                        elastic_params=el)
    kT = units.kT_langevin(298.0, grid['hx'], grid['hy'], grid['hz'])
    return dict(deck=deck, grid=grid, ops=ops, el=el, P0=P0, dt=dt, kT=kT)


def demo3d_raw(tmp, dt, name, cutoff=None):
    raw = stage_raw(
        tmp, name, material={'deck': 'bto_lcc2005_8th', 'temperature_K': 298.0},
        geometry={'engine': 'pf3d', 'N': [16, 16, 16], 'cell_nm': 1.0},
        numerics={'dt': dt, 'dt_verify_time': DEMO_T_TOTAL},
        stage=[{'kind': 'anneal', 'seed': 0, 'sample_every': 25,
                'start': {'kind': 'random', 'amplitude': 0.005, 'kmax_frac': 0.5, 'seed': 0},
                'stop': {'block_stationary': 0.01, 'n_max': 150}}])
    if cutoff is not None:
        raw['physics'] = {'noise': {'cutoff_nm': cutoff}}
    return raw


def test_demo_casekit_3d_anchor(demo3d, tmp_path):
    d = demo3d
    # the demo passes no noise_cutoff_nm: cutoff_nm = 0 disables the cutoff
    res = run_raw(demo3d_raw(tmp_path, 'verified', 'nocut', cutoff=0))
    assert res['build']['dt'] == d['dt'] and res['build']['kT'] == d['kT'] and res['build']['noise_cutoff_nm'] is None
    P, trace, why, state = casekit.anneal(pf3d_engine, d['P0'], d['deck']['landau'], d['ops'], d['grid'], dt=d['dt'],
                                          kT=d['kT'], elastic_params=d['el'], seed=0, n_max=150, sample_every=25,
                                          return_state=True)
    st = res['stages'][0]
    assert np.array_equal(st['P'], P) and st['trace'] == trace and st['stop_reason'] == why
    assert st['rng_state'] == state['rng_state']
    # the default cutoff (2.0 nm) equals casekit.anneal(noise_cutoff_nm=2.0)
    res = run_raw(demo3d_raw(tmp_path, d['dt'], 'cut2'))
    assert res['build']['noise_cutoff_nm'] == 2.0
    P2, trace2, why2 = casekit.anneal(pf3d_engine, d['P0'], d['deck']['landau'], d['ops'], d['grid'], dt=d['dt'],
                                      kT=d['kT'], noise_cutoff_nm=2.0, elastic_params=d['el'], seed=0, n_max=150,
                                      sample_every=25)
    st = res['stages'][0]
    assert np.array_equal(st['P'], P2) and st['trace'] == trace2 and st['stop_reason'] == why2
    assert not np.array_equal(P, P2)                          # the cutoff changes the trajectory


# --------------------------------------------------------------------------- d. film anchor

def test_film_relax_anchor(tmp_path):
    n_steps, dt, seed = 20, 0.01, 2
    a = Namespace(nx=8, ny=8, nz=5, lx_nm=3.2, ly_nm=3.2, thickness_nm=1.6, temperature_c=25.0, strain_pct=-0.5,
                  eps_b=10.0, substrate_layers=3, substrate_material=None)
    deck, grid, el, kappa = bfb.setup(a)
    alpha, G = deck['landau'], deck['gradient']['G11']
    P = bfb.initial_state(grid, seed)
    pf = film_engine.poisson_factor(grid, kappa)
    gf = film_engine.build_grad_factors(grid, G, dt, 1.0)
    gf2 = film_engine.build_grad_factors_sbdf2(grid, G, dt, 1.0)
    prev = None
    for _ in range(n_steps):
        new = (film_engine.step_sbdf2(P, prev, alpha, grid, gf2, pf, dt, 1.0, el=el) if prev is not None else
               film_engine.step(P, alpha, grid, gf, pf, dt, 1.0, el=el))
        prev, P = P, new
    diag = bfb.diagnostics(P, alpha, grid, G, kappa, pf, el)
    dens, tot = film_engine.total_energy_3d(P, alpha, grid, kappa, G, poisson_factors=pf, el=el)
    raw = stage_raw(tmp_path, material={'deck': 'bto_lcc2005_paper', 'temperature_K': 25.0 + 273.15},
                    geometry={'engine': 'film', 'N': [8, 8, 5], 'cell_nm': 0.4, 'film_thickness_nm': 1.6},
                    physics={'electrostatics': {'eps_b': 10.0},
                             'elastic': {'misfit_strain_pct': -0.5, 'substrate_layers': 3}},
                    numerics={'dt': dt, 'scheme': 'sbdf2'},
                    stage=[{'kind': 'relax', 'start': {'kind': 'random', 'amplitude': 0.005, 'kmax_frac': 0.5, 'seed': seed},
                            'stop': {'tol': 1e-12, 'n_max': n_steps}}])
    res = run_raw(raw)
    st = res['stages'][0]
    assert st['step'] == n_steps and st['stop_reason'] == 'n_max'
    assert np.array_equal(st['P'], P)
    last = st['samples'][-1]
    assert last['step'] == n_steps and last['f_total'] == diag['f_total_deck'] and last['E_total'] == float(tot['total'])
    assert np.allclose(last['mean_P'], bfb.volume_mean(P, grid), rtol=1e-12, atol=1e-15)    # same weighted mean, summed differently


# --------------------------------------------------------------------------- e. CLI end to end

def cli(*args, cwd=ROOT):
    env = dict(os.environ, MPLBACKEND='Agg')
    return subprocess.run([PYTHON, '-m', 'fftpf', *args], capture_output=True, text=True, cwd=cwd, env=env)


def example(tmp_path, name, out=None):
    text = open(os.path.join(EX, name)).read().replace('dir = "OUT"', f'dir = "{out or tmp_path / "out"}"')
    path = tmp_path / name
    path.write_text(text)
    return path


def test_cli_run_check_explain(tmp_path):
    case = example(tmp_path, 'tiny_pf2d.toml')
    r = cli('check', str(case))
    assert r.returncode == 0, r.stderr
    r = cli('explain', str(case))
    assert r.returncode == 0 and 'pf2d_engine' in r.stdout and 'stage[0] anneal' in r.stdout
    r = cli('run', str(case))
    assert r.returncode == 0, r.stderr
    run_dir = r.stdout.strip().splitlines()[-1]
    assert run_dir == str(tmp_path / 'out')
    for f in ('resolved.toml', 'provenance.json', 'trace.csv', 'state_0_anneal.npz', 'measures.json', 'quicklook.png'):
        assert os.path.isfile(os.path.join(run_dir, f)), f
    # the in-process run of the same file
    res = run_protocol.run(str(example(tmp_path, 'tiny_pf2d.toml', out=tmp_path / 'inproc')))
    final = res['stages'][0]['samples'][-1]
    rows = open(os.path.join(run_dir, 'trace.csv')).read().splitlines()
    cols = dict(zip(rows[0].split(','), rows[-1].split(',')))
    assert float(cols['E_total']) == final['E_total'] and int(cols['step']) == 60
    # resolved.toml re-runs to an identical final state
    cfg = run_input.load(os.path.join(run_dir, 'resolved.toml'))
    cfg['output']['dir'] = str(tmp_path / 'rerun')
    again = run_protocol.run(cfg)
    saved = state_io.load_state_npz(os.path.join(run_dir, 'state_0_anneal.npz'))
    assert np.array_equal(again['P'], res['P']) and np.array_equal(saved['P'], res['P'])
    # a run never overwrites
    r = cli('run', str(case))
    assert r.returncode == 2 and 'output.dir' in r.stderr


def test_cli_check_misspelt_key(tmp_path):
    path = example(tmp_path, 'tiny_pf2d.toml')
    path.write_text(path.read_text().replace('dt = 0.02', 'dtt = 0.02'))
    r = cli('check', str(path))
    assert r.returncode == 2 and 'numerics.dtt' in r.stderr
    assert not os.path.exists(tmp_path / 'out')
    r = cli('check', str(example(tmp_path, 'tiny_pf2d.toml')).replace('tiny_pf2d', 'missing'))
    assert r.returncode == 2


@pytest.mark.parametrize('name', ['tiny_pf3d.toml', 'tiny_film.toml'])
def test_example_files_check_explain_and_run(tmp_path, name):
    case = example(tmp_path, name)
    assert cli('check', str(case)).returncode == 0
    assert cli('explain', str(case)).returncode == 0
    res = run_protocol.run(str(case))
    assert np.all(np.isfinite(res['P']))
    assert os.path.isfile(os.path.join(res['run_dir'], 'quicklook.png'))


def test_auto_dir_follows_casekit_naming(tmp_path, monkeypatch):
    monkeypatch.setenv('FFT_RUN_ROOT', str(tmp_path))
    monkeypatch.delenv('FFT_RUN_ID', raising=False)
    raw = tomllib.load(open(os.path.join(EX, 'tiny_pf2d.toml'), 'rb'))
    raw['output']['dir'] = 'auto'
    d1 = run_raw(raw)['run_dir']
    d2 = run_raw(raw)['run_dir']
    assert os.path.dirname(d1) == str(tmp_path) and d1.endswith('_tiny_pf2d') and d2 == d1 + '_2'
    monkeypatch.setenv('FFT_RUN_ID', 'job42')
    assert run_raw(raw)['run_dir'] == str(tmp_path / 'job42')


# --------------------------------------------------------------------------- f. builder-level refusals

MAT = {'deck': 'bto_lcc2005_8th', 'temperature_K': 298.0}


def base(tmp, engine='pf2d', **over):
    geo = {'pf2d': {'engine': 'pf2d', 'N': [6, 6], 'cell_nm': 1.0, 'slab_thickness_nm': 1.0},
           'pf3d': {'engine': 'pf3d', 'N': [4, 4, 4], 'cell_nm': 1.0},
           'film': {'engine': 'film', 'N': [4, 4, 4], 'cell_nm': 0.4, 'film_thickness_nm': 1.2}}[engine]
    raw = stage_raw(tmp, material=copy.deepcopy(MAT), geometry=geo, numerics={'dt': 0.02}, stage=[{'kind': 'relax'}])
    raw['physics'] = {'elastic': {'k0': 'off'}} if engine == 'pf2d' else {}
    if engine == 'film':
        raw['physics'] = {'elastic': {'misfit_strain_pct': -0.5, 'substrate_layers': 2}}
    for k, v in over.items():
        raw[k] = v
    return raw


def refused(raw, prefix):
    with pytest.raises(ConfigError) as ei:
        run_builder.build(run_input.resolve(raw))
    assert str(ei.value).startswith(prefix + ':'), str(ei.value)
    return str(ei.value)


def test_refusals(tmp_path):
    t = tmp_path
    np.savez(t / 'f.npz', bad=np.zeros((3, 2, 2)), small=np.zeros((6, 6)), E=np.zeros((3, 6, 6)), P=np.zeros((3, 5, 5)))
    refused(base(t, field=[{'target': 'eps_star', 'kind': 'from_file', 'file': str(t / 'f.npz'), 'key': 'small'}]),
            'field[0].target')
    msg = refused(base(t, field=[{'target': 'E_ext', 'kind': 'from_file', 'file': str(t / 'f.npz'), 'key': 'bad'}]), 'field[0].file')
    assert 'deck units' in msg and '(3, 6, 6)' in msg
    refused(base(t, field=[{'target': 'E_ext', 'kind': 'from_file', 'file': str(t / 'f.npz'), 'key': 'nope'}]), 'field[0].key')
    refused(base(t, field=[{'target': 'landau.alpha1', 'kind': 'from_file', 'file': str(t / 'f.npz'), 'key': 'E'}]), 'field[0].file')
    two = [{'target': 'landau.alpha1', 'kind': 'from_file', 'file': str(t / 'f.npz'), 'key': 'small'},
           {'target': 'landau.alpha1', 'kind': 'gaussian', 'sigma': 0.01, 'correlation_nm': 2.0, 'seed': 1}]
    refused(base(t, field=two), 'field[1].target')
    refused(base(t, 'film', field=[{'target': 'E_ext', 'kind': 'gaussian', 'sigma_kV_cm': 1.0, 'correlation_nm': 1.0, 'seed': 1}]),
            'field[0].kind')
    refused(base(t, stage=[{'kind': 'relax', 'start': {'kind': 'from_file', 'file': str(t / 'f.npz'), 'key': 'P'}}]),
            'stage[0].start.file')
    refused(base(t, physics={'elastic': {'k0': 'clamped'}}), 'physics.elastic.k0')                 # pf2d
    film = base(t, 'film')
    film['physics']['elastic']['k0'] = 'off'
    refused(film, 'physics.elastic.k0')
    film = base(t, 'film', stage=[{'kind': 'anneal', 'stop': {'n_steps': 3}}])
    film['material'] = {'deck': 'bto_lcc2005_8th', 'temperature_K': 298.15}                           # anisotropic G
    refused(film, 'material.gradient')
    film = base(t, 'film')
    film['physics']['elastic']['substrate'] = 'ss316_rt'
    film['material']['temperature_K'] = 400.0                                                          # room-temperature-only reference
    refused(film, 'physics.elastic.substrate')
    mat = {'landau': {'alpha1': -1.0, 'alpha11': 1.0, 'alpha12': 1.0, 'alpha111': 0.0, 'alpha112': 0.0, 'alpha123': 0.0}, 'gradient': {'G110': 0.1},
           'elastic': {'C11': 1.0, 'C12': 0.5, 'C44': 0.5},
           'electrostriction': {'Q11_phys': 0.1, 'Q12_phys': -0.03, 'Q44_phys': 0.05}}
    refused(base(t, material=mat), 'material.gradient.G11_G110')
    raw = base(t, 'pf3d', geometry={'engine': 'pf3d', 'N': [4, 4, 1], 'cell_nm': 1.0})
    raw['physics'] = {'elastic': {'k0': 'off', 'z_mode': 'plane_stress'}}
    refused(raw, 'physics.elastic.z_mode')


def test_output_dir_must_be_new_or_empty(tmp_path):
    (tmp_path / 'busy').mkdir()
    (tmp_path / 'busy' / 'x').write_text('x')
    raw = base(tmp_path, output={'dir': str(tmp_path / 'busy')})
    with pytest.raises(ConfigError, match='^output.dir:'):
        run_raw(raw)
    assert os.listdir(tmp_path / 'busy') == ['x']
