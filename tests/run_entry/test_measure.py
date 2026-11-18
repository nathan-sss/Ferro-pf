"""RUN-ENTRY E4, first part (SPEC § 16.5): phase fractions and domain maps.

The fractions are compared with the frozen readout Runs/NBT6BT/two_well_20261104/readout_nbt6bt.py by exact float
equality (no tolerance); that part is skipped without the local Runs/ folder.  The portable tests (c, d, e, f) need
nothing outside the repository.
"""
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

from test_drive import (EX, PYTHON, ROOT, _numpy_backend, driver, driver_dir, entry_nbt_raw, film_raw, physical_task,   # noqa: F401
                        read_csv, run_driver_b, run_raw, same_run, tiny2d, tiny3d)
from fftpf import run_figures, run_input, run_measure, run_output, state_io
from fftpf.run_input import ConfigError

PRM = dict(P_ref=0.4, angle_deg=15.0, P_min=0.05, R_factor=0.7)
KEYS = ('R', 'T', 'T15', 'other', 'low')


@pytest.fixture(scope='module')
def readout():
    d = driver_dir()
    sys.dont_write_bytecode = True                    # nothing is written into the Runs folder
    sys.path.insert(0, d)
    try:
        yield importlib.import_module('readout_nbt6bt')
    finally:
        sys.path.remove(d)


def prm_of(P_ref, **kw):
    return dict(PRM, P_ref=P_ref, **kw)


# --------------------------------------------------------------------------- a. bit-equal to the frozen readout

def boundary_field(P_ref):
    """Vectors on the class boundaries: 15 deg from [100] and from [111], |P| at P_min and at 0.7 P_ref (and one ulp each
    side), zero vectors, exact axes and diagonals."""
    c, s = np.cos(np.radians(15.)), np.sin(np.radians(15.))
    u111, w = np.ones(3) / np.sqrt(3.), np.array([2., -1., -1.]) / np.sqrt(6.)
    dirs = [np.array([c, s, 0.]), np.array([c, 0., s]), np.array([c, s / np.sqrt(2.), s / np.sqrt(2.)]),
            c * u111 + s * w, c * u111 - s * w, u111, np.array([1., 0., 0.]), np.array([1., 1., 0.]) / np.sqrt(2.),
            np.array([1., 1., 1.]), np.array([-1., 1., -1.]), np.array([0.3, 0.3, 0.3]), np.array([1., 0.2, 0.2])]
    mags = [0.05, np.nextafter(0.05, 0.), np.nextafter(0.05, 1.), 0.7 * P_ref, np.nextafter(0.7 * P_ref, 0.),
            np.nextafter(0.7 * P_ref, 1.), P_ref, 0.3]
    cells = [m * d for d in dirs for m in mags] + [np.zeros(3)] * 3
    return np.array(cells).T.reshape(3, 1, len(cells))


def random_fields(seed=0):
    rng = np.random.default_rng(seed)
    out = [rng.normal(0., 0.3, (3, 5, 17, 19)), rng.normal(0., 0.3, (3, 33, 31)), rng.normal(0., 0.03, (3, 3, 8, 8))]
    z = rng.normal(0., 0.3, (3, 4, 9, 9))
    z[:, :, :3] = 0.                                                           # zero vectors
    z[:, 0, 4] = np.array([0.4, 0., 0.])[:, None]                              # exact <100> cells
    return out + [z]


def test_fractions_equal_frozen_readout_random_and_boundary(readout):
    P_ref = 0.42475999999999997
    for P in random_fields() + [boundary_field(P_ref)]:
        want = readout.fractions(P, P_ref)
        got = run_measure.fractions(P, prm_of(P_ref))
        assert got == want, (got, want)                                        # exact float equality, every class
        assert set(got) == set(KEYS)
    assert readout.COS15 == float(np.cos(np.radians(prm_of(1.)['angle_deg']))) and (readout.P_MIN, readout.R_FACTOR) == (0.05, 0.7)


def test_fractions_equal_frozen_readout_on_a_real_saved_state(readout, tmp_path):
    P_ref = 0.2
    raw = tiny2d(tmp_path, 'real', [{'kind': 'anneal', 'langevin': True, 'stop': {'n_steps': 60}}],
                 measure={'phase_fractions': {'P_ref': P_ref}})
    res = run_raw(raw)
    P = state_io.load_state_npz(os.path.join(res['run_dir'], 'state_0_anneal.npz'))['P']
    assert np.array_equal(P, res['P'])
    want = readout.fractions(P, P_ref)
    assert run_measure.fractions(P, prm_of(P_ref)) == want
    m = json.load(open(os.path.join(res['run_dir'], 'measures.json')))
    assert m['stages'][0]['fractions_end'] == want                             # JSON round-trips floats exactly
    assert len(set(want.values())) > 2 and want['T'] > 0                        # the state is not a degenerate one


# --------------------------------------------------------------------------- b. NBT6BT anchor: trace columns

def test_nbt6bt_anchor_trace_fraction_columns(driver, tmp_path):
    drv, d = driver
    task = physical_task(d)
    P_b, out_b = run_driver_b(drv, d, tmp_path, task)
    raw = entry_nbt_raw(task, tmp_path / 'entry')
    raw['measure']['phase_fractions'] = {'P_ref': task['P_R_ref']}
    res = run_raw(raw)
    assert np.array_equal(res['P'], P_b)
    rows_b, rows_e = read_csv(out_b / 'trace.csv'), read_csv(os.path.join(res['run_dir'], 'trace.csv'))
    assert len(rows_b) == len(rows_e) == 30
    for rb, re_ in zip(rows_b, rows_e):
        for k in KEYS:                                                         # driver B prints 10 significant digits
            assert rb[f'f_{k}'] == f"{float(re_[f'f_{k}']):.10g}", (k, rb, re_)
    m = json.load(open(os.path.join(res['run_dir'], 'measures.json')))
    last = rows_e[-1]
    assert m['stages'][3]['fractions_end'] == {k: float(last[f'f_{k}']) for k in KEYS}
    assert set(m['stages'][2]['fractions_at']) == {'c1_Emax', 'c1_E0down', 'c1_E0up', 'c2_Emax'}


# --------------------------------------------------------------------------- c. portable: known classes

def uniform(vec, shape=(1, 4, 4)):
    return np.array(vec, float).reshape(3, 1, 1, 1) * np.ones((3,) + shape)


def test_hand_built_fields_have_the_known_classes():
    a = 0.4 / np.sqrt(3.)
    cases = [(uniform([a, a, a]), dict(R=1., T=0., T15=0., other=0., low=0.)),           # <111>, full magnitude
             (uniform([-a, a, -a]), dict(R=1., T=0., T15=0., other=0., low=0.)),
             (uniform([0.4, 0., 0.]), dict(R=0., T=1., T15=1., other=0., low=0.)),        # <100>
             (uniform([0., 0., -0.4]), dict(R=0., T=1., T15=1., other=0., low=0.)),
             (uniform([0.3, 0.3, 0.]), dict(R=0., T=0., T15=0., other=1., low=0.)),       # <110>
             (uniform([0.004, 0., 0.]), dict(R=0., T=0., T15=0., other=0., low=1.)),      # tiny |P|
             (uniform([0., 0., 0.]), dict(R=0., T=0., T15=0., other=0., low=1.)),
             (uniform([0.1, 0.1, 0.1]), dict(R=0., T=0., T15=0., other=1., low=0.))]      # <111> below 0.7 P_ref: not R
    for P, want in cases:
        assert run_measure.fractions(P, PRM) == want
    half = np.concatenate([uniform([a, a, a], (1, 4, 2)), uniform([0.4, 0., 0.], (1, 4, 2))], axis=3)   # left R, right T
    assert run_measure.fractions(half, PRM) == dict(R=.5, T=.5, T15=.5, other=0., low=0.)
    wide = np.concatenate([uniform([0.4, 0.14, 0.], (1, 4, 2)), uniform([0.4, 0., 0.], (1, 4, 2))], axis=3)  # 19 deg: T, not T15
    f = run_measure.fractions(wide, PRM)
    assert (f['T'], f['T15']) == (1., .5)


def test_partition_identity_and_t15_inside_t_on_random_fields():
    for P in random_fields(3):
        for prm in (PRM, prm_of(0.26), prm_of(0.5, angle_deg=22.0)):
            f = run_measure.fractions(P, prm)
            assert f['R'] + f['T'] + f['other'] + f['low'] == pytest.approx(1., abs=1e-14)
            assert f['T15'] <= f['T']
            c = run_measure.classes(P, prm)
            assert not np.any(c['R'] & c['T']) and not np.any(c['T15'] & ~c['T'])
            assert np.array_equal(c['R'] | c['T'] | c['other'] | c['low'], np.ones(P.shape[1:], bool))


# --------------------------------------------------------------------------- d. film weighting

def test_film_fractions_are_weighted_by_qw_by_layers():
    a = 0.4 / np.sqrt(3.)
    Nz = 5
    qw = np.array([.5, 1., 1., 1., .5])
    P = uniform([0.4, 0., 0.], (Nz, 4, 4))                                     # inside: T
    P[:, 0] = np.array([a, a, a])[:, None, None]                              # surface layers: R
    P[:, -1] = np.array([a, a, a])[:, None, None]
    f = run_measure.fractions(P, PRM, qw)
    assert f['R'] == pytest.approx((qw[0] + qw[-1]) / qw.sum(), rel=1e-13)    # independent value: 1 / 4
    assert f['T'] == f['T15'] and f['T'] == pytest.approx(qw[1:-1].sum() / qw.sum(), rel=1e-13)
    assert run_measure.fractions(P, PRM)['R'] == pytest.approx(2. / 5., rel=1e-13)     # unweighted would give 2 / 5
    # a mixed layer: half R, half T cells in the middle layer
    P[:, 2, :, :2] = np.array([a, a, a])[:, None, None]
    f = run_measure.fractions(P, PRM, qw)
    want_R = sum(w * m for w, m in zip(qw, (1., 0., .5, 0., 1.))) / qw.sum()
    assert f['R'] == pytest.approx(want_R, rel=1e-13)


def test_film_run_uses_grid_qw(tmp_path):
    res = run_raw(film_raw(tmp_path, 'film', [{'kind': 'anneal', 'langevin': False, 'stop': {'n_steps': 6}}],
                           measure={'phase_fractions': {'P_ref': 0.2}}))
    ad, P = res['build']['adapter'], res['P']
    qw = ad.grid['qw']
    layers = [run_measure.fractions(P[:, z:z + 1], dict(PRM, P_ref=0.2)) for z in range(P.shape[1])]
    want = {k: float(np.average([l[k] for l in layers], weights=qw)) for k in KEYS}
    m = json.load(open(os.path.join(res['run_dir'], 'measures.json')))
    got = m['stages'][0]['fractions_end']
    assert got == pytest.approx(want, rel=1e-12, abs=1e-15)                   # layer means weighted independently
    assert got == run_measure.phase_fractions(ad, P, dict(PRM, P_ref=0.2))


# --------------------------------------------------------------------------- e. domain maps

def maps_case(tmp, name='maps', at=('anneal:end', 'pe_loop:E0+', 'pe_loop:E0-'), kind=('phase', 'direction'), **over):
    raw = tiny2d(tmp, name, [
        {'kind': 'anneal', 'langevin': True, 'stop': {'n_steps': 10}},
        {'kind': 'pe_loop', 'axis': [1, 0, 0], 'E_max_kV_cm': 200.0, 'cycles': 2.0, 'steps_per_cycle': 40, 'record_every': 5,
         'langevin': True}],
        measure={'phase_fractions': {'P_ref': 0.2}, 'domain_map': {'at': list(at), 'kind': list(kind)}}, **over)
    return raw


def png_shape(path):
    import matplotlib.image as mpimg
    with open(path, 'rb') as fh:
        assert fh.read(8) == b'\x89PNG\r\n\x1a\n'
    return mpimg.imread(path).shape


def test_pf2d_domain_maps_end_to_end(tmp_path):
    res = run_raw(maps_case(tmp_path))
    fig = os.path.join(res['run_dir'], 'figures')
    want = {f'state_{stem}_{k}.png' for stem in ('0_anneal', '1_pe_loop_c2_E0down', '1_pe_loop_c2_E0up') for k in ('phase', 'direction')}
    assert set(os.listdir(fig)) == want                                        # the LAST cycle's E0+ / E0- states
    for f in want:
        assert os.path.getsize(os.path.join(fig, f)) > 5000
        h, w, c = png_shape(os.path.join(fig, f))
        assert (h, w) == (1240, 1080) and c in (3, 4)                          # 6.2 x 5.4 in at 200 dpi
    # the figure comes after its state file; every state file named by a selector exists
    for stem in ('0_anneal', '1_pe_loop_c2_E0down', '1_pe_loop_c2_E0up'):
        assert os.path.isfile(os.path.join(res['run_dir'], f'state_{stem}.npz'))
    m = json.load(open(os.path.join(res['run_dir'], 'measures.json')))
    assert set(m['stages'][1]['fractions_at']) == {f'c{k}_{e}' for k in (1, 2) for e in ('Emax', 'E0down', 'E0up')}
    assert m['stages'][0]['fractions_at'] == {} and set(m['stages'][0]['fractions_end']) == set(KEYS)
    rows = read_csv(os.path.join(res['run_dir'], 'trace.csv'))
    assert all(f'f_{k}' in rows[0] for k in KEYS)
    assert m['stages'][1]['fractions_end'] == {k: float(rows[-1][f'f_{k}']) for k in KEYS}


def test_direction_default_kind_and_other_selectors(tmp_path):
    raw = maps_case(tmp_path, at=['end', 'pe_loop:Emax', 'pe_loop:c1_E0down', '1:end', '0:end'], kind=['direction'])
    del raw['measure']['phase_fractions']
    raw['measure']['domain_map'] = {'at': ['pe_loop:Emax', 'pe_loop:c1_E0down', 'anneal:end', 'pe_loop:end']}
    cfg = run_input.resolve(raw)
    assert cfg['measure']['domain_map']['kind'] == ['direction']
    stems = [t['stem'] for t in run_measure.domain_map_targets(cfg)]
    assert stems == ['state_1_pe_loop_c2_Emax', 'state_1_pe_loop_c1_E0down', 'state_0_anneal', 'state_1_pe_loop_c2_E0up']
    res = run_raw(raw)
    assert sorted(os.listdir(os.path.join(res['run_dir'], 'figures'))) == sorted(s + '_direction.png' for s in stems)
    assert 'phase_fractions' not in json.load(open(os.path.join(res['run_dir'], 'measures.json')))['stages'][0]


def test_pf3d_domain_map_is_the_mid_plane(tmp_path, monkeypatch):
    import matplotlib.axes
    titles = []
    real = matplotlib.axes.Axes.set_title
    monkeypatch.setattr(matplotlib.axes.Axes, 'set_title', lambda self, label, *a, **k: (titles.append(label), real(self, label, *a, **k))[1])
    raw = tiny3d(tmp_path, 'm3', [{'kind': 'anneal', 'langevin': False, 'stop': {'n_steps': 5}}],
                 measure={'phase_fractions': {'P_ref': 0.2}, 'domain_map': {'kind': ['phase', 'direction']}})
    res = run_raw(raw)
    assert sorted(os.listdir(os.path.join(res['run_dir'], 'figures'))) == ['state_0_anneal_direction.png', 'state_0_anneal_phase.png']
    assert len([t for t in titles if 'mid-plane z layer 3 of 6' in t]) == 2
    titles.clear()
    raw = tiny3d(tmp_path, 'm1', [{'kind': 'anneal', 'langevin': False, 'stop': {'n_steps': 5}}],
                 geometry={'engine': 'pf3d', 'N': [6, 6, 1], 'cell_nm': 1.0}, measure={'domain_map': {}})
    run_raw(raw)
    assert len([t for t in titles if 'only z layer' in t]) == 1


def test_domain_map_refusals(tmp_path):
    raw = maps_case(tmp_path)
    del raw['measure']['phase_fractions']
    with pytest.raises(ConfigError, match=r'^measure\.domain_map\.kind: .*phase_fractions'):
        run_input.resolve(raw)
    for bad in ('pe_loop:Emax2', 'anneal:E0+', 'cool:end', 'pe_loop:T300', '5:end', 'relax:end'):
        raw = maps_case(tmp_path, at=[bad])
        with pytest.raises(ConfigError, match=r'^measure\.domain_map\.at\[0\]: .*matches no state'):
            run_input.resolve(raw)
    for bad in ('nocolon', ':end', 'pe_loop:'):
        with pytest.raises(ConfigError, match=r'^measure\.domain_map\.at: '):
            run_input.resolve(maps_case(tmp_path, at=[bad]))
    with pytest.raises(ConfigError, match=r'^measure\.domain_map\.kind\[0\]: must be one of'):
        run_input.resolve(maps_case(tmp_path, kind=['magma']))
    raw = maps_case(tmp_path)
    raw['measure']['phase_fractions'] = {'P_ref': 0.05}                        # 0.7 * 0.05 < P_min = 0.05
    with pytest.raises(ConfigError, match=r'^measure\.phase_fractions\.R_factor: .*overlap'):
        run_input.resolve(raw)
    raw['measure']['phase_fractions'] = {'angle_deg': 15.}
    with pytest.raises(ConfigError, match=r'^measure\.phase_fractions\.P_ref: required'):
        run_input.resolve(raw)
    raw['measure']['phase_fractions'] = {'P_ref': 0.3, 'angle_deg': 30.}
    with pytest.raises(ConfigError, match=r'^measure\.phase_fractions\.angle_deg'):
        run_input.resolve(raw)
    raw['measure']['phase_fractions'] = {'P_ref': 0.3, 'colour': 'red'}
    with pytest.raises(ConfigError, match=r'^measure\.phase_fractions\.colour: unknown key'):
        run_input.resolve(raw)


def test_phase_fraction_defaults_and_resolved_roundtrip(tmp_path):
    import tomllib
    cfg = run_input.resolve(maps_case(tmp_path))
    assert cfg['measure']['phase_fractions'] == dict(P_ref=0.2, angle_deg=15.0, P_min=0.05, R_factor=0.7)
    assert run_input.resolve(tomllib.loads(run_input.dumps(cfg))) == cfg
    cfg = run_input.resolve(tiny2d(tmp_path, 'd', [{'kind': 'anneal', 'stop': {'n_steps': 3}}], measure={'domain_map': {}}))
    assert cfg['measure']['domain_map'] == dict(at='all', kind=['direction']) and 'phase_fractions' not in cfg['measure']


def test_saved_states_are_exactly_the_state_files_of_a_run(tmp_path):
    raw = tiny2d(tmp_path, 'all', [
        {'kind': 'anneal', 'langevin': False, 'stop': {'n_steps': 4}},
        {'kind': 'cool', 'rungs': [[330.0, 6], [298.0, 6]], 'langevin': False, 'record_every': 3},
        {'kind': 'hold', 'n_steps': 8, 'E_kV_cm': 50.0, 'axis': [1, 0, 0], 'ramp_steps': 4, 'langevin': False, 'record_every': 4},
        {'kind': 'hold', 'n_steps': 4, 'langevin': False},
        {'kind': 'pe_loop', 'axis': [1, 0, 0], 'E_max_kV_cm': 100.0, 'cycles': 1.5, 'steps_per_cycle': 8, 'record_every': 2, 'langevin': False},
        {'kind': 'pe_loop', 'axis': [0, 1, 0], 'E_max_kV_cm': 100.0, 'cycles': 1.75, 'steps_per_cycle': 8, 'record_every': 2, 'langevin': False},
        {'kind': 'pe_loop', 'axis': [1, 0, 0], 'E_max_kV_cm': 100.0, 'waveform': 'unipolar', 'n_points': 2, 'n_dwell': 2}],
        measure={'domain_map': {}})                                            # at = "all"
    res = run_raw(raw)
    files = sorted(f[:-4] for f in os.listdir(res['run_dir']) if f.startswith('state_'))
    cfg = run_input.resolve(raw)
    assert sorted(s['stem'] for s in run_measure.saved_states(cfg)) == files
    assert sorted(f[:-4] for f in os.listdir(os.path.join(res['run_dir'], 'figures'))) == sorted(f + '_direction' for f in files)
    ends = {t['stem'] for t in run_measure.domain_map_targets(dict(cfg, measure=dict(cfg['measure'], domain_map=dict(at=['hold:end', 'pe_loop:end'], kind=['direction']))))}
    assert ends == {'state_2_hold_end', 'state_3_hold_end', 'state_4_pe_loop_c2_E0down', 'state_6_pe_loop_end'}   # stage 5 stops at -Emax: no end state


def test_spec_example_1_passes_check_verbatim():
    path = os.path.join(EX, 'pe_loop.toml')
    cfg = run_input.load(path)
    assert cfg['measure']['domain_map'] == dict(at=['anneal:end', 'pe_loop:E0+', 'pe_loop:E0-'], kind=['direction'])
    stems = [t['stem'] for t in run_measure.domain_map_targets(cfg)]
    assert stems == ['state_0_anneal', 'state_1_pe_loop_c2_E0down', 'state_1_pe_loop_c2_E0up']
    env = dict(os.environ, MPLBACKEND='Agg')
    for command in ('check', 'explain'):
        r = subprocess.run([PYTHON, '-m', 'fftpf', command, path], capture_output=True, text=True, cwd=ROOT, env=env)
        assert r.returncode == 0, r.stderr


def test_missing_matplotlib_skips_figures_with_one_warning(tmp_path, monkeypatch, capsys):
    def no_plot(*a, **k):
        raise ImportError('no matplotlib')
    monkeypatch.setattr(run_figures, 'domain_map', no_plot)
    res = run_raw(maps_case(tmp_path))
    out = capsys.readouterr().out
    assert out.count('matplotlib is not installed') == 1
    assert not os.path.isdir(os.path.join(res['run_dir'], 'figures')) or not os.listdir(os.path.join(res['run_dir'], 'figures'))
    assert os.path.isfile(os.path.join(res['run_dir'], 'measures.json'))


# --------------------------------------------------------------------------- f. resume

def resume_maps_case(tmp, name):
    raw = tiny2d(tmp, name, [
        {'kind': 'anneal', 'langevin': True, 'stop': {'n_steps': 20}, 'sample_every': 5},
        {'kind': 'cool', 'rungs': [[330.0, 20], [298.0, 20]], 'langevin': True, 'record_every': 5},
        {'kind': 'pe_loop', 'axis': [1, 0, 0], 'E_max_kV_cm': 200.0, 'cycles': 1.0, 'steps_per_cycle': 40, 'record_every': 5,
         'langevin': True}],
        output={'dir': str(tmp / name), 'checkpoint_every': 15}, measure={'loop': ['Pr', 'Pmax'], 'phase_fractions': {'P_ref': 0.2},
                                                                         'domain_map': {'kind': ['phase', 'direction']}})
    return raw


@pytest.mark.parametrize('stop', [2, 4, 7])
def test_resume_gives_the_same_trace_measures_and_figure_names(tmp_path, stop):
    full = run_raw(resume_maps_case(tmp_path, 'full'))
    raw = resume_maps_case(tmp_path, 'cut')
    with pytest.raises(run_output.RunInterrupted):
        run_raw(raw, stop_after_checkpoints=stop)
    cut = tmp_path / 'cut'
    assert not os.path.isfile(cut / 'measures.json')
    res = run_raw(raw)
    same_run(full['run_dir'], res['run_dir'], full, res)                       # P, state files, trace.csv and measures.json bytes
    figs = sorted(os.listdir(os.path.join(full['run_dir'], 'figures')))
    assert figs == sorted(os.listdir(os.path.join(res['run_dir'], 'figures'))) and len(figs) == 2 * (1 + 2 + 3)
    assert sorted(os.listdir(cut)) == sorted(os.listdir(tmp_path / 'full'))
    m = json.load(open(cut / 'measures.json'))
    assert set(m['stages'][1]['fractions_at']) == {'T330', 'T298'} and len(m['stages'][2]['fractions_at']) == 3
