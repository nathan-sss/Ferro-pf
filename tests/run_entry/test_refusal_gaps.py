"""RUN-ENTRY (SPEC § 16.4): the refusal gaps found by the documentation pass.

1  replicas > 1 without measure.spectrum is refused at load; a single ac / fdt run without it works and writes the series
2  an inline material needs the six sixth-order Landau coefficients (the registry zero-fills only the eighth-order ones)
3  a substrate-only (or otherwise incomplete) registry deck as material.deck is refused at load; a substrate must have an elastic sector
4  `check` raises every refusal that `run` raises before its first step (one code path, ``build(cfg, dry=True)``)
5  `explain` refuses what `check` refuses, with the same message and exit code
6  an output.dir / FFT_RUN_ROOT that cannot be written is a ConfigError on output.dir
7  in-process use with NumPy already imported: a ConfigError that names the environment variables
8  the manual says how to resume an `auto` directory
"""
import copy
import glob
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pytest

from test_drive import EX, ROOT, _numpy_backend, refuses, run_raw     # noqa: F401
from test_run import base, stage_raw     # noqa: F401
from test_spectrum import ac_stage, quad_raw
from fftpf import mat_registry, run_builder, run_input, run_output, run_protocol
from fftpf.run_input import ConfigError


def cli(command, path, cwd=None, env=None):
    e = dict(os.environ, PYTHONPATH=ROOT, MPLBACKEND='Agg')
    e.update(env or {})
    return subprocess.run([sys.executable, '-m', 'fftpf', command, str(path)], capture_output=True, text=True, env=e,
                          cwd=cwd or ROOT)


def check_refuses(raw, prefix):
    """The file loads, `check` (the dry build) refuses it with ``prefix:``; returns the message."""
    cfg = run_input.resolve(raw)
    with pytest.raises(ConfigError) as ei:
        run_builder.check(cfg)
    assert str(ei.value).startswith(prefix + ':'), str(ei.value)
    return str(ei.value)


# --------------------------------------------------------------------------- 1. replicas without measure.spectrum

def test_replicas_without_spectrum_refused_at_load(tmp_path):
    raw = quad_raw(tmp_path / 'o', stages=[ac_stage([0.05], periods=3, discard=1)], replicas=2)
    del raw['measure']['spectrum']
    msg = refuses(raw, 'run.replicas')
    assert 'measure.spectrum' in msg


def test_single_ac_run_without_spectrum_writes_the_series(tmp_path):
    raw = quad_raw(tmp_path / 'o', stages=[ac_stage([0.05], periods=3, discard=1)])
    del raw['measure']['spectrum']
    res = run_raw(raw)
    d = res['run_dir']
    assert glob.glob(os.path.join(d, 'series_0_ac_w*.npz')) and os.path.isfile(os.path.join(d, 'spectrum_0_ac.npz'))
    assert not os.path.exists(os.path.join(d, 'spectrum.csv'))
    assert 'spectrum' not in open(os.path.join(d, 'measures.json')).read().split('"stages"')[0]


# --------------------------------------------------------------------------- 2. inline material: sixth order required

@pytest.mark.parametrize('table, coef', [('landau', 'alpha111'), ('landau', 'alpha112'), ('landau', 'alpha123'),
                                         ('landau_si', 'alpha111'), ('landau_si', 'alpha123')])
def test_inline_material_needs_the_sixth_order_set(tmp_path, table, coef):
    raw = quad_raw(tmp_path / 'o', stages=[{'kind': 'relax'}])
    raw['measure'] = {}
    landau = raw['material'].pop('landau')
    if table == 'landau_si':
        landau = {k: v * 1e9 for k, v in landau.items()}
    del landau[coef]
    raw['material'][table] = landau
    refuses(raw, f'material.{table}.{coef}')


def test_inline_material_without_eighth_order_runs(tmp_path):
    """The registry (input_validation.LANDAU_OPTIONAL) and the engine (alpha.get) default the eighth-order coefficients to 0."""
    raw = quad_raw(tmp_path / 'o', stages=[{'kind': 'anneal', 'langevin': False, 'stop': {'n_steps': 2}}])
    raw['measure'] = {}
    assert not {'alpha1111', 'alpha1112', 'alpha1122', 'alpha1123'} & set(raw['material']['landau'])
    res = run_raw(raw)
    assert np.all(np.isfinite(res['P']))


# --------------------------------------------------------------------------- 3. decks without the P sectors

@pytest.mark.parametrize('deck, missing', [('ss316_rt', 'landau'), ('mgo_cubic_rt', 'gradient'), ('pto_haun1987', 'gradient')])
def test_deck_without_the_p_sectors_is_refused_at_load(tmp_path, deck, missing):
    raw = base(tmp_path)
    raw['material'] = {'deck': deck, 'temperature_K': 298.0}
    msg = refuses(raw, 'material.deck')
    assert missing in msg and 'sector' in msg


def test_substrate_deck_message_says_substrate(tmp_path):
    raw = base(tmp_path)
    raw['material'] = {'deck': 'ss316_rt', 'temperature_K': 298.0}
    assert 'substrate' in refuses(raw, 'material.deck')


def test_toml_deck_without_sectors_or_sixth_order_is_refused_by_check(tmp_path):
    """A params/*.toml path is read by the builder, so `check` (the dry build) refuses it, not the load."""
    src = open(os.path.join(ROOT, 'params', 'bto_298K.toml')).read()
    only_elastic = ''.join(src[src.index('[elastic]'):].split('\n\n')[0]) + '\n'
    (tmp_path / 'sub.toml').write_text('[meta]\nname = "x"\n' + only_elastic)
    raw = base(tmp_path)
    raw['material'] = {'deck': str(tmp_path / 'sub.toml'), 'temperature_K': 298.0}
    msg = check_refuses(raw, 'material.deck')
    assert 'landau' in msg and 'sector' in msg
    lines = [ln for ln in src.split('\n') if not ln.strip().startswith('alpha111')]
    (tmp_path / 'no111.toml').write_text('\n'.join(lines))
    raw['material'] = {'deck': str(tmp_path / 'no111.toml'), 'temperature_K': 298.0}
    assert 'alpha111' in check_refuses(raw, 'material.deck') or 'a111' in check_refuses(raw, 'material.deck')


def test_substrate_must_have_the_elastic_sector(tmp_path, monkeypatch):
    raw = base(tmp_path, engine='film')
    raw['physics']['elastic']['substrate'] = 'ss316_rt'
    run_input.resolve(raw)                                          # fine as is
    real = mat_registry.get
    monkeypatch.setattr(mat_registry, 'get',
                        lambda name, **kw: {k: v for k, v in real(name, **kw).items() if k != 'elastic'} if name == 'ss316_rt'
                        else real(name, **kw))
    refuses(raw, 'physics.elastic.substrate')


# --------------------------------------------------------------------------- 4. check = the same build path, dry

def _npz(tmp, **arrays):
    p = tmp / 'a.npz'
    np.savez(p, **arrays)
    return str(p)


def case_field_shape(t):
    return base(t, field=[{'target': 'E_ext', 'kind': 'from_file', 'file': _npz(t, bad=np.zeros((3, 2, 2))), 'key': 'bad'}]), 'field[0].file'


def case_field_key(t):
    return base(t, field=[{'target': 'landau.alpha11', 'kind': 'from_file', 'file': _npz(t, a=np.zeros((6, 6))), 'key': 'zz'}]), 'field[0].key'


def case_start_shape(t):
    raw = base(t, stage=[{'kind': 'relax', 'start': {'kind': 'from_file', 'file': _npz(t, P=np.zeros((3, 5, 5))), 'key': 'P'}}])
    return raw, 'stage[0].start.file'


def case_pf2d_clamped(t):
    raw = base(t)
    raw['physics'] = {'elastic': {'k0': 'clamped'}}
    return raw, 'physics.elastic.k0'


def case_film_clamped(t):
    raw = base(t, engine='film')
    raw['physics']['elastic']['k0'] = 'clamped'
    return raw, 'physics.elastic.k0'


def case_film_gaussian(t):
    raw = base(t, engine='film')
    raw['field'] = [{'target': 'E_ext', 'kind': 'gaussian', 'sigma_kV_cm': 1.0, 'correlation_nm': 1.0, 'seed': 1}]
    return raw, 'field[0].kind'


def case_duplicate_landau(t):
    f = {'target': 'landau.alpha11', 'kind': 'gaussian', 'sigma': 1e-3, 'correlation_nm': 1.0}
    return base(t, field=[dict(f, seed=1), dict(f, seed=2)]), 'field[1].target'


def case_film_langevin_gradient(t):
    raw = base(t, engine='film')
    raw['stage'] = [{'kind': 'anneal', 'langevin': True, 'stop': {'n_steps': 2}}]
    return raw, 'material.gradient'


def case_inline_subkey(t):
    raw = quad_raw(t / 'o', stages=[{'kind': 'relax'}])
    raw['measure'] = {}
    del raw['material']['elastic']['C44']
    return raw, 'material.elastic.C44'


def case_eps_star(t):
    return base(t, field=[{'target': 'eps_star', 'kind': 'from_file', 'file': _npz(t, a=np.zeros((6, 6))), 'key': 'a'}]), 'field[0].target'


CASES = [case_field_shape, case_field_key, case_start_shape, case_pf2d_clamped, case_film_clamped, case_film_gaussian,
         case_duplicate_landau, case_film_langevin_gradient, case_inline_subkey, case_eps_star]


@pytest.mark.parametrize('case', CASES, ids=[c.__name__[5:] for c in CASES])
def test_check_raises_what_run_raises(tmp_path, case):
    raw, prefix = case(tmp_path)
    check_refuses(raw, prefix)
    with pytest.raises(ConfigError) as ei:                           # and `run` gives the same message
        run_protocol.run(run_input.resolve(raw))
    assert str(ei.value).startswith(prefix + ':')
    assert not os.path.exists(tmp_path / 'run')                      # nothing left behind


def test_check_sees_a_deck_sector_that_changes_with_T(tmp_path, monkeypatch):
    raw = base(tmp_path)
    raw['material'] = {'deck': 'bto_lcc2005_8th', 'temperature_K': 300.0}
    raw['stage'] = [{'kind': 'cool', 'rungs': [[300.0, 5], [310.0, 5]], 'langevin': False}]
    real = mat_registry.get

    def get(name, **kw):
        d = real(name, **kw)
        if kw.get('T') == 310.0:
            d = dict(d, elastic=dict(d['elastic'], C11=d['elastic']['C11'] + 1.0))
        return d
    monkeypatch.setattr(mat_registry, 'get', get)
    msg = check_refuses(raw, 'material.deck')
    assert 'changes with T' in msg


def test_check_is_cli_visible_and_leaves_no_directory(tmp_path):
    raw, _ = case_field_shape(tmp_path)
    raw['output'] = {'dir': str(tmp_path / 'made' / 'run')}
    cfg = run_input.resolve(raw)
    path = tmp_path / 'case.toml'
    path.write_text(run_input.dumps(cfg))
    r = cli('check', path)
    assert r.returncode == 2 and 'field[0].file' in r.stderr, r.stderr
    assert not os.path.exists(tmp_path / 'made')


def test_check_stays_fast_on_the_examples():
    for path in sorted(glob.glob(os.path.join(ROOT, 'docs', 'run_examples', '*.toml'))):
        cfg = run_input.load(path)
        t = time.perf_counter()
        run_builder.check(cfg)
        assert time.perf_counter() - t < 1.0, path


def test_check_accepts_a_good_from_file_field(tmp_path):
    raw = base(tmp_path, field=[{'target': 'E_ext', 'kind': 'from_file', 'file': _npz(tmp_path, E=np.zeros((3, 6, 6))), 'key': 'E'}])
    run_builder.check(run_input.resolve(raw))


# --------------------------------------------------------------------------- 5. explain = check first

@pytest.mark.parametrize('case', [case_field_shape, case_pf2d_clamped, case_inline_subkey], ids=lambda c: c.__name__[5:])
def test_explain_refuses_what_check_refuses(tmp_path, case):
    raw, prefix = case(tmp_path)
    path = tmp_path / 'case.toml'
    path.write_text(run_input.dumps(run_input.resolve(raw)))
    a, b = cli('check', path), cli('explain', path)
    assert a.returncode == b.returncode == 2
    assert a.stderr == b.stderr and prefix in b.stderr and not b.stdout


# --------------------------------------------------------------------------- 6. output.dir that cannot be written

def test_output_dir_below_a_file_is_a_config_error(tmp_path):
    (tmp_path / 'afile').write_text('x')
    raw = base(tmp_path)
    raw['output'] = {'dir': str(tmp_path / 'afile' / 'sub')}
    with pytest.raises(ConfigError) as ei:
        run_protocol.run(run_input.resolve(raw))
    assert str(ei.value).startswith('output.dir:')
    with pytest.raises(ConfigError) as ei:
        run_protocol.check(run_input.resolve(raw))
    assert str(ei.value).startswith('output.dir:')


@pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() == 0, reason='root can write anywhere')
def test_read_only_parent_is_a_config_error(tmp_path):
    ro = tmp_path / 'ro'
    ro.mkdir()
    ro.chmod(0o500)
    try:
        raw = base(tmp_path)
        raw['output'] = {'dir': str(ro / 'run')}
        with pytest.raises(ConfigError) as ei:
            run_protocol.run(run_input.resolve(raw))
        assert str(ei.value).startswith('output.dir:')
        with pytest.raises(ConfigError):
            run_output.make_dir(run_input.resolve(raw))
    finally:
        ro.chmod(0o700)


def test_unwritable_FFT_RUN_ROOT_is_a_config_error(tmp_path, monkeypatch):
    (tmp_path / 'afile').write_text('x')
    monkeypatch.setenv('FFT_RUN_ROOT', str(tmp_path / 'afile' / 'x'))
    monkeypatch.delenv('FFT_RUN_ID', raising=False)
    raw = base(tmp_path)
    raw['output'] = {'dir': 'auto'}
    cfg = run_input.resolve(raw)
    with pytest.raises(ConfigError) as ei:
        run_protocol.run(cfg)
    assert str(ei.value).startswith('output.dir:') and 'FFT_RUN_ROOT' in str(ei.value)
    with pytest.raises(ConfigError):
        run_protocol.check(cfg)


def test_check_does_not_create_a_missing_output_dir(tmp_path):
    raw = base(tmp_path)
    raw['output'] = {'dir': str(tmp_path / 'deep' / 'er' / 'run')}
    run_protocol.check(run_input.resolve(raw))
    assert not os.path.exists(tmp_path / 'deep')


# --------------------------------------------------------------------------- 7. NumPy already imported

def test_in_process_use_with_numpy_imported_names_the_variables(tmp_path):
    case = tmp_path / 'case.toml'
    text = open(os.path.join(EX, 'tiny_pf2d.toml')).read().replace('dir = "OUT"', f'dir = "{tmp_path / "out"}"')
    case.write_text(text)
    code = ("import numpy\n"
            "from fftpf import run_input, run_protocol\n"
            f"try:\n    run_protocol.run({str(case)!r})\n"
            "except run_input.ConfigError as e:\n    print('CONFIGERROR', e)\n")
    env = {k: v for k, v in os.environ.items() if k not in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
    env['PYTHONPATH'] = ROOT
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert 'CONFIGERROR numerics.threads:' in out
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'python -m fftpf'):
        assert name in out, out
    assert not os.path.exists(tmp_path / 'out')


# --------------------------------------------------------------------------- 8. the manual

def _manual():
    return open(os.path.join(ROOT, 'docs', 'RUN_INPUT.md'), encoding='utf-8').read()


def test_manual_says_how_to_resume_an_auto_directory():
    text = _manual()
    sec = text[text.index('## 5 Resume and replicas'):text.index('## 6 ')]
    assert re.search(r'auto.*never resumes unless `FFT_RUN_ID`', sec, re.S) and 'explicit' in sec and 'output.dir' in sec


def test_manual_section_9_has_no_start_only_refusals_left():
    text = _manual()
    sec = text[text.index('## 9 Refusals'):text.index('## 10 ')]
    assert 'Start:' not in sec and '**start**' not in sec
    for gone in ("KeyError: 'electrostriction'", "KeyError: 'spectrum'", "KeyError: 'a111'"):
        assert gone not in sec
