"""RUN-ENTRY E3b (SPEC § 16.5): stages cool / hold / pe_loop, checkpoint / resume, loop measures.

"Bit-for-bit" = both paths run in this process and numpy.array_equal holds on P (final and every saved state), and every
number both paths record is equal (driver B writes trace numbers with 10 significant digits: those are compared as
text of the entry's float formatted the same way).
"""
import copy
import csv
import importlib.util
import json
import os
import subprocess
import sys

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from fftpf import (mat_registry, pf2d_energy_storage_loop as esl, pf2d_engine, run_builder, run_input, run_measure,
                   run_output, run_protocol, state_io, units)
from fftpf.run_input import ConfigError
from solvers import execution

EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples')
PYTHON = sys.executable
NUMBA = 'numba' if importlib.util.find_spec('numba') else 'numpy'      # the backend driver B's load_engine picks


@pytest.fixture(autouse=True)
def _numpy_backend():
    execution.configure(1, 'numpy')
    yield
    execution.configure(workers=1, backend='numpy')


def run_raw(raw, **kw):
    return run_protocol.run(run_input.resolve(raw), **kw)


def refused(raw, prefix):
    with pytest.raises(ConfigError) as ei:
        run_builder.build(run_input.resolve(raw))
    assert str(ei.value).startswith(prefix + ':'), str(ei.value)
    return str(ei.value)


def refuses(raw, prefix):
    with pytest.raises(ConfigError) as ei:
        run_input.resolve(raw)
    assert str(ei.value).startswith(prefix + ':'), str(ei.value)
    return str(ei.value)


def read_csv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- a. unipolar anchor (reference A)

def test_unipolar_anchor_run_unipolar(tmp_path):
    N, dt, n1, n_dwell, Emax_kV, seed = 16, 0.05, 4, 20, 300.0, 3
    deck_path = os.path.join(ROOT, 'params', 'pbzo_triplewell_yang2025.toml')
    case = esl.build_case(N=N, delta_nm=1.0, eps_b=50.0)
    grid, ops, alpha = case['grid'], case['ops'], case['alpha']
    Minv = pf2d_engine.build_Minv(grid, ops, dt, 1.0)
    P0 = pf2d_engine.random_smooth_field(grid, 3, amplitude=0.01, kmax_frac=0.3, seed=seed)
    # the entry converts the kV/cm path point by point, the way it converts every applied field
    e_path = np.array([units.si_to_deck_field(e * 1e5) for e in esl.e_path_unipolar(n1, Emax_kV)])
    ref = esl.run_unipolar(alpha, ops, Minv, dt, n_dwell, P0, e_path)
    raw = {'run': {'name': 'uni', 'seed': seed}, 'output': {'dir': str(tmp_path / 'uni')},
           'material': {'deck': deck_path}, 'geometry': {'engine': 'pf2d', 'N': [N, N], 'cell_nm': 1.0},
           'physics': {'electrostatics': {'eps_b': 50.0}, 'elastic': {'k0': 'off'}}, 'numerics': {'dt': dt},
           'stage': [{'kind': 'pe_loop', 'waveform': 'unipolar', 'axis': [1, 0, 0], 'E_max_kV_cm': Emax_kV,
                      'n_points': n1, 'n_dwell': n_dwell,
                      'start': {'kind': 'random', 'amplitude': 0.01, 'kmax_frac': 0.3, 'seed': seed}}],
           'measure': {'loop': ['Pmax', 'Pr', 'W_store', 'W_rec', 'W_loss', 'eta']}}
    res = run_raw(raw)
    st = res['stages'][0]
    assert [s['Pn'] for s in st['samples']] == list(ref['P1_full'])           # mean P path, exact
    assert np.array_equal(np.array([units.si_to_deck_field(s['E_kV_cm'] * 1e5) for s in st['samples']]), ref['E_full'])
    assert np.array_equal(res['P'], ref['P_final'])
    want = esl.energy_integrals(ref['E_full'], ref['P1_full'], n1)
    loop = json.load(open(os.path.join(res['run_dir'], 'measures.json')))['stages'][0]['loop']
    for k in ('W_store', 'W_rec', 'W_loss', 'eta'):
        assert loop[f'{k}_deck' if k != 'eta' else k] == want[k]
        if k != 'eta':
            assert loop[k] == want[k] * 1e3
    assert loop['Pmax'] == ref['P1_full'][n1] and loop['Pr'] == ref['P1_full'][-1]
    assert os.path.isfile(os.path.join(res['run_dir'], 'state_0_pe_loop_Emax.npz'))
    rows = read_csv(os.path.join(res['run_dir'], 'trace.csv'))
    assert len(rows) == 2 * n1 + 1 and rows[0]['T_K'] == 'nan'                 # no temperature stated: NaN


# --------------------------------------------------------------------------- b. NBT6BT anchor (driver B)

def runs_root():
    env = os.environ.get('FFT_RUNS_ROOT')
    if env:
        return env
    common = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                            capture_output=True, text=True).stdout.strip()
    return os.path.join(os.path.dirname(common), 'Runs')


def driver_dir():
    d = os.path.join(runs_root(), 'NBT6BT', 'two_well_20261104')
    if not os.path.isfile(os.path.join(d, 'run_nbt6bt.py')):
        pytest.skip(f'frozen driver not found at {d} (set FFT_RUNS_ROOT to the Runs folder)')
    return d


LANDAU = ('a11', 'a12', 'a111', 'a112', 'a123', 'a1111', 'a1112', 'a1122', 'a1123')
PROTOCOL_B = [{'stage': 'cool', 'rungs': [[400, 30], [350, 30], [300, 30]], 'record_every': 10, 'label': 'cool_0'},
              {'stage': 'hold', 'T_K': 300, 'n_steps': 30, 'record_every': 10, 'label': 'hold_1'},
              {'stage': 'pe', 'T_K': 300, 'n': [1, 1, 1], 'Emax_kVcm': 60, 'steps_per_sweep': 20, 'n_cycles': 1.25,
               'record_every': 10, 'label': 'pe_2'},
              {'stage': 'remanent', 'T_K': 300, 'ramp_steps': 20, 'hold_steps': 20, 'record_every': 10, 'label': 'remanent_3'}]


# config_l3pe.json (NBT6BT two_well_20261104), task 0, on a 32 x 32 grid
STUB = {'N': [32, 32, 1],
 'h_nm': 2.0,
 'Dz_nm': 6.0,
 'dt': 0.2,
 'L': 0.5857142857142856,
 'P_R_ref': 0.42475999999999997,
 'alpha_SI': 102439.02439024393,
 'T0bar_K': 446.04392468766434,
 'beta_SI': 1.707317073170732e-10,
 'eps_b': 117.14285714285714,
 'landau_SI': {'a11': 44132111.39997179,
               'a12': 1181247649.6047523,
               'a111': 1469100596.3540802,
               'a112': -10013068594.927107,
               'a123': -39957930751.03719,
               'a1111': 5105352555.499684,
               'a1112': 0.0,
               'a1122': 74787169845.312,
               'a1123': 165447907602.76056},
 'sigma_long_K': 40,
 'xi_long_nm': 60,
 'seed_long': 11,
 'sigma_nano_K': 20,
 'xi_nano_nm': 6,
 'seed_nano': 12,
 'E_rf_kVcm': 16.799999999999997,
 'xi_rf_nm': 6,
 'seeds_rf': [13, 14, 15],
 'C_GPa': [204.344, 110.6672, 140.05599999999998],
 'Q_tensor': [0.05056513979773945, -0.020820939916716245, 0.022010707911957174],
 'k0': 'stress_free',
 'z_mode': 'plane_stress',
 'thermal_seed': 8,
 'noise_cutoff_nm': 8,
 'record_every': 10,
 'checkpoint_every': 50}


def physical_task_stub():
    """The same numbers as config_l3pe.json, typed here, for the tests that need no access to Runs/."""
    return STUB


def physical_task(d):
    """The physical parameters of the L3 P-E task (config_l3pe.json), at a reduced grid."""
    task = json.load(open(os.path.join(d, 'config_l3pe.json')))['tasks'][0]
    keep = {k: v for k, v in task.items() if k not in ('protocol', 'start', 'name')}
    return dict(keep, N=[32, 32, 1], record_every=10, checkpoint_every=50)


def entry_nbt_raw(task, out, checkpoint_every=0, white_std=0.02, start_seed=5):
    """The entry-file twin of the reduced driver-B task (same physics, same stages)."""
    c = task
    return {
        'run': {'name': 'nbt_anchor', 'seed': c['thermal_seed']}, 'output': {'dir': str(out), 'checkpoint_every': checkpoint_every},
        'material': {'landau_si': dict({'alpha' + k[1:]: c['landau_SI'][k] for k in LANDAU},
                                       alpha1={'slope': c['alpha_SI'], 'T0_K': c['T0bar_K']}),
                     'gradient_si': {'isotropic': c['beta_SI']},
                     'elastic': dict(zip(('C11', 'C12', 'C44'), c['C_GPa'])),
                     'electrostriction': {'Q11_phys': c['Q_tensor'][0], 'Q12_phys': c['Q_tensor'][1], 'Q44_phys': 4 * c['Q_tensor'][2]}},
        'geometry': {'engine': 'pf3d', 'N': c['N'], 'cell_nm': c['h_nm'], 'slab_thickness_nm': c['Dz_nm']},
        'physics': {'electrostatics': {'eps_b': c['eps_b']}, 'elastic': {'k0': c['k0'], 'z_mode': c['z_mode']},
                    'noise': {'cutoff_nm': c['noise_cutoff_nm']}},
        'numerics': {'dt': c['dt'], 'L': c['L'], 'backend': NUMBA},
        'field': [{'target': 'T0', 'kind': 'gaussian', 'sigma_K': c['sigma_long_K'], 'correlation_nm': c['xi_long_nm'], 'seed': c['seed_long']},
                  {'target': 'T0', 'kind': 'gaussian', 'sigma_K': c['sigma_nano_K'], 'correlation_nm': c['xi_nano_nm'], 'seed': c['seed_nano']},
                  {'target': 'E_ext', 'kind': 'gaussian', 'sigma_kV_cm': c['E_rf_kVcm'], 'correlation_nm': c['xi_rf_nm'],
                   'seeds': c['seeds_rf']}],
        'stage': [{'kind': 'cool', 'rungs': [[400, 30], [350, 30], [300, 30]], 'langevin': True, 'record_every': 10,
                   'start': {'kind': 'white', 'std': white_std, 'seed': start_seed}},
                  {'kind': 'hold', 'T_K': 300, 'n_steps': 30, 'langevin': True, 'record_every': 10},
                  {'kind': 'pe_loop', 'T_K': 300, 'axis': [1, 1, 1], 'E_max_kV_cm': 60, 'waveform': 'triangle', 'cycles': 1.25,
                   'steps_per_cycle': 80, 'record_every': 10, 'langevin': True},
                  {'kind': 'hold', 'T_K': 300, 'n_steps': 40, 'ramp_steps': 20, 'langevin': True, 'record_every': 10}],
        'measure': {'loop': ['Pmax', 'Pr', 'Ec']},
    }


@pytest.fixture(scope='module')
def driver():
    d = driver_dir()
    sys.dont_write_bytecode = True                    # nothing is written into the Runs folder
    sys.path.insert(0, d)
    import run_nbt6bt as drv
    drv.load_engine(ROOT, 1)
    yield drv, d
    sys.path.remove(d)


def run_driver_b(drv, d, tmp, task):
    raw = dict(task, tasks=[dict(name='NBT6BT_anchor', start={'random_seed': 5, 'random_std': 0.02}, protocol=copy.deepcopy(PROTOCOL_B))])
    cfg = drv.task_config(raw, 0)
    out = tmp / 'driverB'
    P = drv.run_task(cfg, out, base=out)
    return P, out


B_STATES = {'state_cool_0_T400': 'state_0_cool_T400', 'state_cool_0_T350': 'state_0_cool_T350', 'state_cool_0_T300': 'state_0_cool_T300',
            'state_hold_1_end': 'state_1_hold_end', 'state_pe_2_c1_Emax': 'state_2_pe_loop_c1_Emax',
            'state_pe_2_c1_E0down': 'state_2_pe_loop_c1_E0down', 'state_pe_2_c1_E0up': 'state_2_pe_loop_c1_E0up',
            'state_pe_2_c2_Emax': 'state_2_pe_loop_c2_Emax', 'state_remanent_3_ramped': 'state_3_hold_ramped',
            'state_remanent_3_end': 'state_3_hold_end'}


def test_nbt6bt_anchor_driver_b(driver, tmp_path):
    drv, d = driver
    task = physical_task(d)
    assert task == STUB, 'config_l3pe.json changed: refresh STUB'
    P_b, out_b = run_driver_b(drv, d, tmp_path, task)
    res = run_raw(entry_nbt_raw(task, tmp_path / 'entry'))
    assert np.array_equal(res['P'], P_b)                                        # final float64 state
    with np.load(out_b / 'checkpoint.npz') as z:
        assert np.array_equal(res['P'], z['P'])
    run_dir = res['run_dir']
    for b_name, e_name in B_STATES.items():                                     # every saved state (driver B stores float32)
        with np.load(out_b / f'{b_name}.npz') as z:
            p_b = z['P']
        p_e = state_io.load_state_npz(os.path.join(run_dir, e_name + '.npz'))['P']
        assert p_b.dtype == np.float32 and np.array_equal(p_e.astype(np.float32), p_b), b_name
    rows_b, rows_e = read_csv(out_b / 'trace.csv'), read_csv(os.path.join(run_dir, 'trace.csv'))
    assert len(rows_b) == len(rows_e) == 30
    for rb, re_ in zip(rows_b, rows_e):
        assert int(rb['step']) == int(re_['step_total']) and int(rb['stage_step']) == int(re_['step'])
        for cb, ce in (('T_K', 'T_K'), ('E_kVcm', 'E_kV_cm'), ('Px', 'Px'), ('Py', 'Py'), ('Pz', 'Pz'), ('Pn', 'Pn')):
            assert rb[cb] == f'{float(re_[ce]):.10g}', (cb, rb, re_)
        assert rb['energy_MJ_m3'] == f"{float(re_['f_total']) * 1e3:.10g}"
        assert rb['time'] == f"{float(re_['t']):.10g}"
    m = json.load(open(os.path.join(run_dir, 'measures.json')))
    assert m['stages'][2]['loop']['cycle'] == 1


def test_nbt6bt_anchor_numpy_backend(driver, tmp_path, monkeypatch):
    """Same anchor on the NumPy reference kernels (driver B's backend switch forced off)."""
    drv, d = driver
    orig = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec', lambda n, *a, **k: None if n == 'numba' else orig(n, *a, **k))
    drv.load_engine(ROOT, 1)
    try:
        task = physical_task(d)
        P_b, _ = run_driver_b(drv, d, tmp_path, task)
        raw = entry_nbt_raw(task, tmp_path / 'entry')
        raw['numerics']['backend'] = 'numpy'
        assert np.array_equal(run_raw(raw)['P'], P_b)
    finally:
        monkeypatch.undo()
        drv.load_engine(ROOT, 1)


# --------------------------------------------------------------------------- shared tiny cases

def tiny2d(tmp, name, stages, seed=4, **over):
    raw = {'run': {'name': name, 'seed': seed}, 'output': {'dir': str(tmp / name)},
           'material': {'deck': 'bto_lcc2005_8th', 'temperature_K': 298.0},
           'geometry': {'engine': 'pf2d', 'N': [12, 12], 'cell_nm': 1.0, 'slab_thickness_nm': 1.0},
           'physics': {'electrostatics': {'eps_b': 10.0}, 'elastic': {'k0': 'off'}},
           'numerics': {'dt': 0.02}, 'stage': stages}
    for k, v in over.items():
        raw[k] = v
    return raw


def tiny3d(tmp, name, stages, **over):
    raw = tiny2d(tmp, name, stages)
    raw['geometry'] = {'engine': 'pf3d', 'N': [6, 6, 6], 'cell_nm': 1.0}
    raw['physics'] = {'electrostatics': {'eps_b': 10.0}}
    raw.update(over)
    return raw


# --------------------------------------------------------------------------- c. checkpoint / resume

def resume_case_2d_langevin(tmp, name):
    return tiny2d(tmp, name, [
        {'kind': 'cool', 'rungs': [[330.0, 25], [298.0, 25]], 'langevin': True, 'record_every': 5},
        {'kind': 'pe_loop', 'axis': [1, 0, 0], 'E_max_kV_cm': 200.0, 'cycles': 1.0, 'steps_per_cycle': 40,
         'record_every': 5, 'langevin': True}],
        output={'dir': str(tmp / name), 'checkpoint_every': 15}, measure={'energy_terms': True, 'loop': ['Pr', 'Pmax', 'Ec']})


def resume_case_3d_sbdf2(tmp, name):
    return tiny3d(tmp, name, [
        {'kind': 'hold', 'n_steps': 30, 'langevin': False, 'record_every': 5},
        {'kind': 'pe_loop', 'axis': [1, 1, 0], 'E_max_kV_cm': 150.0, 'cycles': 1.0, 'steps_per_cycle': 32,
         'record_every': 4, 'langevin': False},
        {'kind': 'hold', 'n_steps': 20, 'E_kV_cm': 0.0, 'ramp_steps': 8, 'langevin': False, 'record_every': 5}],
        numerics={'dt': 0.02, 'scheme': 'sbdf2'}, output={'dir': str(tmp / name), 'checkpoint_every': 12},
        measure={'loop': ['Pr', 'Pmax']})


def resume_case_nbt(tmp, name):
    task = physical_task(driver_dir())
    return entry_nbt_raw(task, tmp / name, checkpoint_every=25)


def same_run(dir_a, dir_b, res_a, res_b):
    assert np.array_equal(res_a['P'], res_b['P'])
    states = sorted(f for f in os.listdir(dir_a) if f.startswith('state_'))
    assert states and states == sorted(f for f in os.listdir(dir_b) if f.startswith('state_'))
    for f in states:
        assert np.array_equal(state_io.load_state_npz(os.path.join(dir_a, f))['P'],
                              state_io.load_state_npz(os.path.join(dir_b, f))['P']), f
    for f in ('trace.csv', 'measures.json'):
        assert open(os.path.join(dir_a, f), 'rb').read() == open(os.path.join(dir_b, f), 'rb').read(), f


@pytest.mark.parametrize('case,stop', [('2d_langevin', 2), ('2d_langevin', 5), ('3d_sbdf2', 3), ('3d_sbdf2', 6), ('nbt', 2), ('nbt', 4)])
def test_resume_equals_uninterrupted(tmp_path, case, stop):
    make = dict(zip(('2d_langevin', '3d_sbdf2', 'nbt'), (resume_case_2d_langevin, resume_case_3d_sbdf2, resume_case_nbt)))[case]
    if case == 'nbt':
        driver_dir()
    full = run_raw(make(tmp_path, 'full'))
    raw = make(tmp_path, 'cut')
    with pytest.raises(run_output.RunInterrupted):
        run_raw(raw, stop_after_checkpoints=stop)
    cut = tmp_path / 'cut'
    assert os.path.isfile(cut / 'checkpoint.npz') and not os.path.isfile(cut / 'measures.json')
    with open(cut / 'trace.csv', 'ab') as fh:                 # rows written after the checkpoint must be cut away
        fh.write(b'garbage,written,after,the,checkpoint\n')
    res = run_raw(raw)
    same_run(full['run_dir'], res['run_dir'], full, res)
    assert sorted(os.listdir(cut)) == sorted(os.listdir(tmp_path / 'full'))


def test_resume_is_mid_stage_and_rng_carried(tmp_path):
    raw = resume_case_2d_langevin(tmp_path, 'cut')
    with pytest.raises(run_output.RunInterrupted):
        run_raw(raw, stop_after_checkpoints=2)
    with np.load(tmp_path / 'cut' / 'checkpoint.npz') as z:
        meta, p_prev = json.loads(str(z['meta'])), z['P_prev']
    assert (meta['stage_index'], meta['stage_step']) == (0, 30) and meta['summaries'] == []
    assert p_prev.size == 0 and meta['rng']['bit_generator'] == 'PCG64' and meta['trace_bytes'] > 0
    assert set(meta) >= {'identity', 'E', 'n', 'E_entry', 'step0', 't', 'rng', 'trace_bytes', 'summaries'}


def test_checkpoint_of_another_config_and_busy_dir(tmp_path):
    raw = resume_case_2d_langevin(tmp_path, 'cut')
    with pytest.raises(run_output.RunInterrupted):
        run_raw(raw, stop_after_checkpoints=1)
    other = copy.deepcopy(raw)
    other['numerics']['dt'] = 0.01
    with pytest.raises(ConfigError, match='^output.dir:.*different configuration'):
        run_raw(other)
    other = copy.deepcopy(raw)                      # only `output` differs: still the same run
    other['output']['checkpoint_every'] = 7
    with pytest.raises(run_output.RunInterrupted):
        run_raw(other, stop_after_checkpoints=1)
    (tmp_path / 'busy').mkdir()
    (tmp_path / 'busy' / 'x').write_text('x')
    raw['output']['dir'] = str(tmp_path / 'busy')
    with pytest.raises(ConfigError, match='^output.dir:.*not empty'):
        run_raw(raw)


def test_quenched_fields_are_reloaded_not_regenerated(tmp_path, monkeypatch):
    raw = tiny3d(tmp_path, 'q', [{'kind': 'hold', 'n_steps': 20, 'langevin': True, 'record_every': 5}],
                 field=[{'target': 'E_ext', 'kind': 'gaussian', 'sigma_kV_cm': 20.0, 'correlation_nm': 2.0, 'seed': 3}],
                 output={'dir': str(tmp_path / 'q'), 'checkpoint_every': 8})
    with pytest.raises(run_output.RunInterrupted):
        run_raw(raw, stop_after_checkpoints=1)
    saved = np.load(tmp_path / 'q' / 'quenched.npz')['field_0']
    from fftpf import disorder_fields

    def boom(*a, **k):
        raise AssertionError('a resume regenerated a quenched field')
    monkeypatch.setattr(disorder_fields, 'random_field_gaussian_3d', boom)
    res = run_raw(raw)
    assert np.array_equal(res['build']['E_static'], saved)


def test_cli_stop_hook_and_resume(tmp_path):
    case = tmp_path / 'case.toml'
    case.write_text(f'''[run]
name = "cli"
seed = 2
[material]
deck = "bto_lcc2005_8th"
temperature_K = 298.0
[geometry]
engine = "pf2d"
N = [8, 8]
slab_thickness_nm = 1.0
[physics]
electrostatics = {{ eps_b = 10.0 }}
elastic = {{ k0 = "off" }}
[numerics]
dt = 0.02
[[stage]]
kind = "hold"
n_steps = 30
record_every = 5
[output]
dir = "{tmp_path / 'out'}"
checkpoint_every = 10
''')
    env = dict(os.environ, MPLBACKEND='Agg')
    r = subprocess.run([PYTHON, '-m', 'fftpf', 'run', str(case), '--stop-after-checkpoints', '1'], capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 3 and 'interrupted' in r.stderr and not (tmp_path / 'out' / 'measures.json').exists()
    r = subprocess.run([PYTHON, '-m', 'fftpf', 'run', str(case)], capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / 'out' / 'measures.json').exists()
    res = run_protocol.run(str(case).replace('case.toml', 'case.toml'))          # a finished run resumes from its last checkpoint: no-op
    assert res['stages'] == []


# --------------------------------------------------------------------------- d. temperature change, independent of driver B

def test_two_rung_cool_equals_manual_segments(tmp_path):
    seed, dt, n1, n2, T1, T2 = 4, 0.02, 20, 20, 330.0, 298.0
    raw = tiny2d(tmp_path, 'cool', [{'kind': 'cool', 'rungs': [[T1, n1], [T2, n2]], 'langevin': True, 'record_every': 10}])
    res = run_raw(raw)
    # the same, by direct engine calls: one carried generator, alpha / kT / OU operators rebuilt at each temperature
    grid = pf2d_engine.make_grid(Nx=12, Ny=12, Lx=12.0, Ly=12.0)
    kappa = units.relative_to_deck_permittivity(10.0)
    P = pf2d_engine.random_smooth_field(grid, 3, amplitude=0.005, kmax_frac=0.3, seed=seed)
    rng = np.random.default_rng(seed)
    mid = None
    for T, n in ((T1, n1), (T2, n2)):
        deck = mat_registry.get('bto_lcc2005_8th', T=T)
        ops = pf2d_engine.build_operators(grid, **deck['gradient'], kappa=kappa)
        ou = pf2d_engine.build_ou_operators(ops, dt, 1.0, units.kT_langevin(T, grid['hx'], grid['hy'], 1.0),
                                            noise_cutoff_nm=2.0, grid=grid)
        for _ in range(n):
            P = pf2d_engine.step_langevin(P, deck['landau'], ops, ou, dt, 1.0, (0.0, 0.0, 0.0), rng)
        mid = P if mid is None else mid
        if T == T1:
            mid = P.copy()
    assert np.array_equal(res['P'], P)
    assert np.array_equal(state_io.load_state_npz(os.path.join(res['run_dir'], 'state_0_cool_T330.npz'))['P'], mid)
    rows = read_csv(os.path.join(res['run_dir'], 'trace.csv'))
    assert [float(r['T_K']) for r in rows] == [T1] * 3 + [T2] * 2 and rows[0]['step'] == '0'
    assert res['build']['T'] == T2 and res['build']['kT'] == units.kT_langevin(T2, 1.0, 1.0, 1.0)


def test_rng_is_one_generator_for_the_run(tmp_path):
    hold = lambda **kw: dict({'kind': 'hold', 'n_steps': 20, 'langevin': True, 'record_every': 10}, **kw)
    two = run_raw(tiny2d(tmp_path, 'two', [hold(), hold()]))
    one = run_raw(tiny2d(tmp_path, 'one', [hold(n_steps=40)]))
    assert np.array_equal(two['P'], one['P'])                      # the second stage continues the first stage's noise
    own = run_raw(tiny2d(tmp_path, 'own', [hold(), hold(seed=4)]))     # run.seed = 4: a new generator replays the draws
    assert not np.array_equal(own['P'], one['P'])
    again = run_raw(tiny2d(tmp_path, 'again', [hold(), hold(seed=5)]))
    assert not np.array_equal(again['P'], own['P'])


def test_hold_ramp_is_driver_b_remanent_formula(tmp_path):
    E0, ramp = 120.0, 8
    res = run_raw(tiny2d(tmp_path, 'rem', [
        {'kind': 'pe_loop', 'axis': [1, 0, 0], 'E_max_kV_cm': E0, 'cycles': 0.25, 'steps_per_cycle': 40, 'langevin': False, 'record_every': 5},
        {'kind': 'hold', 'n_steps': 12, 'ramp_steps': ramp, 'langevin': False, 'record_every': 1}]))
    rows = [r for r in read_csv(os.path.join(res['run_dir'], 'trace.csv')) if r['stage'] == '1']
    assert [float(r['E_kV_cm']) for r in rows] == [E0 * (1. - s / ramp) if s < ramp else 0.0 for s in range(13)]
    assert os.path.isfile(os.path.join(res['run_dir'], 'state_1_hold_ramped.npz'))


def test_hold_field_ramps_between_two_values(tmp_path):
    res = run_raw(tiny2d(tmp_path, 'ramp', [
        {'kind': 'hold', 'n_steps': 6, 'axis': [0, 1, 0], 'E_kV_cm': 40.0, 'ramp_steps': 4, 'langevin': False, 'record_every': 1}]))
    es = [float(r['E_kV_cm']) for r in read_csv(os.path.join(res['run_dir'], 'trace.csv'))]
    assert es == [40.0 + (0.0 - 40.0) * (1. - s / 4) if s < 4 else 40.0 for s in range(7)]


def test_white_start_slab_and_unit_axis(tmp_path):
    std, seed = 0.03, 9
    raw = tiny3d(tmp_path, 'w', [{'kind': 'hold', 'n_steps': 1, 'langevin': False,
                                  'start': {'kind': 'white', 'std': std, 'seed': seed}}],
                 geometry={'engine': 'pf3d', 'N': [6, 6, 1], 'cell_nm': 1.0, 'slab_thickness_nm': 4.0})
    b = run_builder.build(run_input.resolve(raw))
    assert b['grid']['Lz'] == 4.0 and b['grid']['hz'] == 4.0
    P0 = run_builder.make_start(b['adapter'], run_input.resolve(raw)['stage'][0]['start'], 'stage[0]', b['space'])
    assert np.array_equal(P0, np.random.default_rng(seed).normal(0., std, (3, 1, 6, 6)))
    assert run_protocol.unit_axis([1, 2, 3]) == [float(x) for x in np.asarray([1., 2., 3.]) / np.linalg.norm([1, 2, 3])]
    refuses(tiny3d(tmp_path, 'w2', [{'kind': 'hold', 'n_steps': 1}], geometry={'engine': 'pf3d', 'N': [6, 6, 2], 'cell_nm': 1.0,
                                                                              'slab_thickness_nm': 4.0}), 'geometry.slab_thickness_nm')


def test_field_orders_seed_and_seeds(tmp_path):
    """`seeds` follows driver B (sigma * unit, x 1e5, x 1e-9 per component); `seed` keeps rms = conv(sigma * 1e5)."""
    from fftpf import disorder_fields
    g = {'target': 'E_ext', 'kind': 'gaussian', 'sigma_kV_cm': 16.8, 'correlation_nm': 2.0}
    base = tiny3d(tmp_path, 'f', [{'kind': 'hold', 'n_steps': 1, 'langevin': False}])
    grid = run_builder.build(run_input.resolve(base))['grid']
    b1 = run_builder.build(run_input.resolve(dict(base, field=[dict(g, seed=11)])))
    b2 = run_builder.build(run_input.resolve(dict(base, field=[dict(g, seeds=[11, 12, 13])])))
    unit = [disorder_fields.random_field_gaussian_3d(grid, 1.0, 2.0, s) for s in (11, 12, 13)]
    assert np.array_equal(b2['E_static'], np.stack([units.si_to_deck_field(16.8 * u * 1e5) for u in unit]))
    rms = units.si_to_deck_field(16.8 * 1e5)
    assert np.array_equal(b1['E_static'], np.stack([disorder_fields.random_field_gaussian_3d(grid, rms, 2.0, 11 + c) for c in range(3)]))
    assert np.allclose(b1['E_static'], b2['E_static'], rtol=1e-12, atol=0)         # the same field, last bits aside


# --------------------------------------------------------------------------- e. loop measures

def synthetic_cycle(m, Ps=0.3, Ec=20.0, Emax=60.0, slope=0.01):
    """One triangle cycle of 4m + 1 points on piecewise-linear branches: known Pmax, Pr = slope*Ec, Ec."""
    q = np.linspace(0.0, Emax, m + 1)
    E = np.concatenate([q, q[::-1][1:], -q[1:], -q[::-1][1:]])
    up = lambda e: np.clip(slope * (e - Ec), -Ps, Ps)
    dn = lambda e: np.clip(slope * (e + Ec), -Ps, Ps)
    P = np.concatenate([up(q), dn(q[::-1][1:]), dn(-q[1:]), up(-q[::-1][1:])])
    return E, P


def as_samples(E, P, every, offset=0):
    return [dict(step=offset + i * every, E_kV_cm=float(e), Pn=float(p)) for i, (e, p) in enumerate(zip(E, P))]


def test_loop_measures_on_a_known_series():
    m, every = 12, 7
    E, P = synthetic_cycle(m)
    assert len(E) == 4 * m + 1 and E[m] == 60.0 and E[2 * m] == 0.0 and E[3 * m] == -60.0
    first = as_samples(E, P * 0.4, every)[:-1]                         # an unlike first cycle: only the LAST one counts
    last = as_samples(E, P, every, offset=4 * m * every)               # (cycle 2 starts at the point cycle 1 ends on)
    st = dict(waveform='triangle', steps_per_cycle=4 * m * every, cycles=2.0)
    out = run_measure.loop(first + last, st, ['Pmax', 'Pr', 'Ec', 'W_rec', 'eta'], every)
    assert out['cycle'] == 2 and out['waveform'] == 'triangle'
    assert out['Pmax'] == pytest.approx(0.3, abs=1e-14) and out['Pr'] == pytest.approx(0.2, abs=1e-14)
    assert out['Ec'] == pytest.approx(20.0, abs=1e-12)
    # the energy measures are energy_integrals on the positive half of the cycle, E converted to deck units
    Ed = np.array([units.si_to_deck_field(e * 1e5) for e in E[:2 * m + 1]])
    w = esl.energy_integrals(Ed, P[:2 * m + 1], m)
    assert out['W_rec_deck'] == w['W_rec'] and out['W_rec'] == w['W_rec'] * 1e3 and out['eta'] == w['eta']
    # one cycle, no crossing on a branch -> Ec None with a note, never a made-up number
    flat = run_measure.loop(as_samples(E, np.abs(P) + 1, every), dict(st, cycles=1.0), ['Ec'], every)
    assert flat['Ec'] is None and 'Ec_note' in flat


def test_unipolar_series_measures():
    n1 = 6
    E = np.concatenate([[0.0], np.linspace(0, 80, n1 + 1)[1:], np.linspace(80, 0, n1 + 1)[1:]])
    P = np.concatenate([[0.0], 0.001 * E[1:n1 + 1], 0.001 * E[n1 + 1:] + 0.01])
    out = run_measure.loop(as_samples(E, P, 5), dict(waveform='unipolar', n_points=n1), ['Pmax', 'Pr', 'W_store', 'W_loss'], 5)
    assert out['Pmax'] == P[n1] and out['Pr'] == P[-1] and 'n_points' in out
    w = esl.energy_integrals(np.array([units.si_to_deck_field(e * 1e5) for e in E]), P, n1)
    assert out['W_store_deck'] == w['W_store'] and out['W_loss'] == w['W_loss'] * 1e3


def test_real_triangle_loop_measures_json_equals_trace(tmp_path):
    m = 30                                                             # records per quarter: 600 steps / record_every 20
    raw = tiny2d(tmp_path, 'loop', [
        {'kind': 'pe_loop', 'axis': [1, 0, 0], 'E_max_kV_cm': 600.0, 'cycles': 2.0, 'steps_per_cycle': 2400, 'record_every': 20,
         'langevin': False}],
        measure={'loop': ['Pr', 'Ec', 'Pmax', 'W_rec', 'W_loss', 'eta']}, numerics={'dt': 0.05})
    res = run_raw(raw)
    rows = [r for r in read_csv(os.path.join(res['run_dir'], 'trace.csv')) if r['stage'] == '0']
    assert len(rows) == 8 * m + 1 and rows[m]['E_kV_cm'] == '600.0' and rows[4 * m]['E_kV_cm'] == '0.0'
    assert [r['E_kV_cm'] for r in rows[:2]] == ['0.0', '20.0']          # the first quarter is the virgin sweep 0 -> +Emax
    last = rows[4 * m:]                                                # cycle 2: steps 2400 .. 4800
    E, Pn = np.array([float(r['E_kV_cm']) for r in last]), np.array([float(r['Pn']) for r in last])
    loop = json.load(open(os.path.join(res['run_dir'], 'measures.json')))['stages'][0]['loop']
    assert loop['cycle'] == 2 and loop['Pmax'] == Pn[m] and loop['Pr'] == (Pn[2 * m] - Pn[4 * m]) / 2
    assert Pn[m] > 0 > Pn[3 * m]                                       # the loop switches
    cross = lambda lo, hi, up: next(E[j] + (0 - Pn[j]) * (E[j + 1] - E[j]) / (Pn[j + 1] - Pn[j])
                                    for j in range(lo, hi) if ((Pn[j] < 0 <= Pn[j + 1]) if up else (Pn[j] > 0 >= Pn[j + 1])))
    assert loop['Ec'] == (cross(0, m, True) - cross(m, 3 * m, False)) / 2 and 100 < loop['Ec'] < 600
    w = esl.energy_integrals(np.array([units.si_to_deck_field(e * 1e5) for e in E[:2 * m + 1]]), Pn[:2 * m + 1], m)
    assert loop['W_rec_deck'] == w['W_rec'] and loop['eta'] == w['eta'] and loop['W_loss'] == w['W_loss'] * 1e3


# --------------------------------------------------------------------------- engines: film and pf3d smoke runs

def film_raw(tmp, name, stages, **over):
    raw = {'run': {'name': name, 'seed': 1}, 'output': {'dir': str(tmp / name)},
           'material': {'deck': 'bto_lcc2005_paper', 'temperature_K': 298.0},
           'geometry': {'engine': 'film', 'N': [4, 4, 4], 'cell_nm': 0.4, 'film_thickness_nm': 1.2},
           'physics': {'electrostatics': {'eps_b': 10.0}, 'elastic': {'misfit_strain_pct': -0.5, 'substrate_layers': 2}},
           'numerics': {'dt': 0.01}, 'stage': stages}
    raw.update(over)
    return raw


def test_film_cool_hold_pe_loop_run(tmp_path):
    res = run_raw(film_raw(tmp_path, 'film', [
        {'kind': 'cool', 'rungs': [[320.0, 10], [298.0, 10]], 'langevin': True, 'record_every': 10},
        {'kind': 'hold', 'n_steps': 8, 'langevin': True, 'record_every': 4},
        {'kind': 'pe_loop', 'axis': [0, 0, 1], 'E_max_kV_cm': 100.0, 'cycles': 1.0, 'steps_per_cycle': 16, 'record_every': 2, 'langevin': True}],
        measure={'loop': ['Pmax', 'Pr']}))
    assert np.all(np.isfinite(res['P'])) and res['build']['T'] == 298.0
    rows = read_csv(os.path.join(res['run_dir'], 'trace.csv'))
    assert [r['E_kV_cm'] for r in rows if r['stage'] == '2'][:3] == ['0.0', '50.0', '100.0']
    assert float(rows[-1]['Pn']) == float(rows[-1]['Pz'])               # axis z: Pn = film-weighted mean P_z


def test_pf3d_langevin_stages_use_the_slab_as_dz(tmp_path):
    raw = tiny3d(tmp_path, 'slab', [{'kind': 'hold', 'n_steps': 4, 'langevin': True}],
                 geometry={'engine': 'pf3d', 'N': [6, 6, 1], 'cell_nm': 1.0, 'slab_thickness_nm': 4.0})
    b = run_builder.build(run_input.resolve(raw))
    assert b['kT'] == units.kT_langevin(298.0, 1.0, 1.0, 4.0)


# --------------------------------------------------------------------------- f. refusals

def hold_raw(tmp, **stage):
    return tiny2d(tmp, 'r', [dict({'kind': 'hold', 'n_steps': 4, 'langevin': False}, **stage)])


def test_input_refusals(tmp_path):
    t = tmp_path
    pe = lambda **kw: tiny2d(t, 'r', [dict({'kind': 'pe_loop', 'axis': [1, 0, 0], 'E_max_kV_cm': 10.0, 'steps_per_cycle': 8,
                                               'langevin': False}, **kw)])
    for wf in ('sine', 'staircase'):
        assert 'not built in this version' in refuses(pe(waveform=wf), 'stage[0].waveform')
    refuses(tiny2d(t, 'r', [{'kind': 'pe_loop', 'axis': [1, 0, 0], 'E_max_kV_cm': 10.0, 'langevin': False}]), 'stage[0].steps_per_cycle')
    refuses(pe(steps_per_cycle=10), 'stage[0].steps_per_cycle')
    refuses(pe(cycles=1.3), 'stage[0].cycles')
    refuses(pe(record_every=3), 'stage[0].record_every')
    refuses(pe(n_dwell=3), 'stage[0].n_dwell')
    refuses(pe(axis=[0, 0, 0]), 'stage[0].axis')

    def unipolar(**kw):
        raw = pe(waveform='unipolar', n_points=3, n_dwell=4)
        del raw['stage'][0]['steps_per_cycle']
        raw['stage'][0].update(kw)
        return raw
    run_input.resolve(unipolar())
    refuses(pe(waveform='unipolar', n_points=3, n_dwell=4), 'stage[0].steps_per_cycle')
    no_dwell = unipolar()
    del no_dwell['stage'][0]['n_dwell']
    refuses(no_dwell, 'stage[0].n_dwell')
    refuses(unipolar(langevin=True), 'stage[0].langevin')
    raw = unipolar()
    raw['numerics']['scheme'] = 'sbdf2'
    refuses(raw, 'numerics.scheme')
    refuses(unipolar(record_every=2), 'stage[0].record_every')
    refuses(dict(unipolar(), measure={'loop': ['Ec']}), 'measure.loop')                 # a unipolar waveform has no Ec
    refuses(dict(pe(), measure={'loop': ['Pr']}, stage=[dict(pe()['stage'][0], cycles=0.5)]), 'measure.loop')
    refuses(hold_raw(t, E_kV_cm=5.0), 'stage[0].axis')
    refuses(hold_raw(t, ramp_steps=9), 'stage[0].ramp_steps')
    refuses(tiny2d(t, 'r', [{'kind': 'cool', 'rungs': [[300.0]], 'langevin': False}]), 'stage[0].rungs[0]')
    refuses(tiny2d(t, 'r', [{'kind': 'cool', 'rungs': [], 'langevin': False}]), 'stage[0].rungs')
    refuses(tiny2d(t, 'r', [{'kind': 'cool', 'rungs': [[300.0, 5]], 'langevin': False}], measure={'loop': ['Pr']}), 'measure.loop')


def test_temperature_law_refusals(tmp_path):
    t = tmp_path
    at = lambda T: [{'kind': 'cool', 'rungs': [[298.0, 3], [T, 3]], 'langevin': False}]
    run_input.resolve(tiny2d(t, 'r', at(330.0)))                                       # registry deck, generator takes T
    msg = refuses(tiny2d(t, 'r', at(330.0), material={'deck': os.path.join(ROOT, 'params', 'bto_298K.toml'), 'temperature_K': 298.0}),
                  'stage[0].rungs[1]')
    assert 'fixed-temperature deck' in msg
    inline = {'landau': {'alpha1': -1.0, 'alpha11': 1.0, 'alpha12': 1.0, 'alpha111': 0.0, 'alpha112': 0.0, 'alpha123': 0.0}, 'gradient': {'G110': 0.1, 'G11_G110': 1.0, 'G12_G110': 0.0,
              'G44_G110': 0.5, 'G44P_G110': 0.5}, 'elastic': {'C11': 1.0, 'C12': 0.5, 'C44': 0.5},
              'electrostriction': {'Q11_phys': 0.1, 'Q12_phys': -0.03, 'Q44_phys': 0.05}, 'temperature_K': 298.0}
    assert 'fixed-temperature deck' in refuses(tiny2d(t, 'r', at(330.0), material=inline), 'stage[0].rungs[1]')
    refuses(tiny2d(t, 'r', [{'kind': 'hold', 'n_steps': 2, 'T_K': 350.0, 'langevin': False}], material=inline), 'stage[0].T_K')
    law = dict(inline, landau_si={'alpha1': {'slope': 1e5, 'T0_K': 400.0}, 'alpha11': 1.0, 'alpha12': 1.0, 'alpha111': 0.0, 'alpha112': 0.0, 'alpha123': 0.0})
    del law['landau'], law['temperature_K']
    run_input.resolve(tiny2d(t, 'r', at(330.0), material=law))
    refuses(tiny2d(t, 'r', [{'kind': 'hold', 'n_steps': 2, 'langevin': False}], material=law), 'material.temperature_K')
    refuses(tiny2d(t, 'r', at(330.0), material=dict(law, landau_si=dict(law['landau_si'], alpha1={'slope': 1e5}))), 'material.landau_si.alpha1.T0_K')
    refuses(tiny2d(t, 'r', at(330.0), material=dict(inline, landau={'alpha1': {'slope': 1.0, 'T0_K': 1.0}, 'alpha11': 1.0, 'alpha12': 1.0, 'alpha111': 0.0, 'alpha112': 0.0, 'alpha123': 0.0})),
            'material.landau.alpha1')
    both = dict(law, gradient_si={'isotropic': 1e-10})
    refuses(tiny2d(t, 'r', at(330.0), material=both), 'material.gradient_si')
    t0 = {'target': 'T0', 'kind': 'gaussian', 'sigma_K': 10.0, 'correlation_nm': 2.0, 'seed': 1}
    refuses(tiny2d(t, 'r', at(330.0), field=[t0]), 'field[0].target')                  # registry deck: no T0 law
    run_input.resolve(tiny2d(t, 'r', at(330.0), material=law, field=[t0]))
    refuses(tiny2d(t, 'r', at(330.0), material=law, field=[{'target': 'landau.alpha1', 'kind': 'gaussian', 'sigma': 0.1,
                                                              'correlation_nm': 2.0, 'seed': 1}]), 'field[0].target')


def test_field_key_refusals(tmp_path):
    g = lambda **kw: tiny2d(tmp_path, 'r', [{'kind': 'hold', 'n_steps': 2, 'langevin': False}],
                            field=[dict({'target': 'E_ext', 'kind': 'gaussian', 'sigma_kV_cm': 5.0, 'correlation_nm': 2.0}, **kw)])
    refuses(g(), 'field[0].seed')
    refuses(g(seed=1, seeds=[1, 2, 3]), 'field[0].seeds')
    refuses(g(seeds=[1, 1, 2]), 'field[0].seeds')
    refuses(g(seeds=[1, 2]), 'field[0].seeds')
    refuses(tiny2d(tmp_path, 'r', [{'kind': 'hold', 'n_steps': 2, 'langevin': False}],
                   field=[{'target': 'T0', 'kind': 'gaussian', 'sigma': 5.0, 'correlation_nm': 2.0, 'seeds': [1, 2, 3]}]), 'field[0].sigma_K')
    refuses(tiny2d(tmp_path, 'r', [{'kind': 'hold', 'n_steps': 2, 'langevin': False}],
                   field=[{'target': 'T0', 'kind': 'gaussian', 'sigma_K': 5.0, 'correlation_nm': 2.0, 'seeds': [1, 2, 3]}]), 'field[0].seeds')
    refuses(tiny2d(tmp_path, 'r', [{'kind': 'hold', 'n_steps': 2, 'langevin': False,
                                    'start': {'kind': 'white', 'seed': 1}}]), 'stage[0].start.std')


def test_builder_refuses_a_deck_whose_operators_depend_on_t(tmp_path, monkeypatch):
    raw = tiny2d(tmp_path, 'r', [{'kind': 'cool', 'rungs': [[298.0, 2], [330.0, 2]], 'langevin': False}])
    real = mat_registry.get

    def drifting(name, **kw):
        d = real(name, **kw)
        d['gradient'] = dict(d['gradient'], G11=d['gradient']['G11'] * (1 + kw.get('T', 298.0) * 1e-6))
        return d
    monkeypatch.setattr(mat_registry, 'get', drifting)
    refused(raw, 'material.deck')


# --------------------------------------------------------------------------- the SPEC § 16.3 example 1 and explain

def test_pe_loop_example_check_and_explain(tmp_path):
    """Example 1 passes `check` verbatim (E4 built its measure.domain_map): resolve + the builder's precheck,
    and `explain` describes its stages."""
    case = os.path.join(EX, 'pe_loop.toml')
    cfg = run_input.load(case)
    run_builder.precheck(cfg)
    assert cfg['stage'][0]['sample_every'] == 100 and cfg['stage'][1]['record_every'] == 100
    env = dict(os.environ, MPLBACKEND='Agg')
    r = subprocess.run([PYTHON, '-m', 'fftpf', 'check', str(case)], capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([PYTHON, '-m', 'fftpf', 'explain', str(case)], capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0 and 'stage[1] pe_loop: start previous, triangle to 200.0 kV/cm' in r.stdout, r.stdout + r.stderr


def test_explain_new_stages(tmp_path):
    cfg = run_input.resolve(entry_nbt_raw(physical_task_stub(), tmp_path / 'e'))
    text = run_builder.explain(cfg)
    assert 'stage[0] cool: start white' in text and 'a1 = slope * (T - T0(r))' in text and 'kT = units.kT_langevin(400' in text
