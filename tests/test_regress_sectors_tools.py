"""Regressions: q/theta eigenstrain keeps the base static eigenstrain,
piecewise_regions_1d wraps, the 2D random-field generator refuses a bulk3d
grid, tools/job.py follows the process group / tails -n 0 / writes
atomically, and casekit.anneal refuses a restart with a changed L."""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(name, '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from solvers import (bulk3d, casekit, disorder, elastic, film3d, film_elastic as fe,
                     kittel_afe, octa_tilt, quenched_fields, solver)
from tools import job

for module in (solver, elastic, bulk3d, film3d, kittel_afe, octa_tilt):
    module.FFT_WORKERS = 1

LAM = dict(Lam11=.05, Lam12=-.02, Lam44=.03)
SECTORS = [(kittel_afe.with_q_eigenstrain, kittel_afe.with_q_eigenstrain_3d,
            kittel_afe.with_q_eigenstrain_film, kittel_afe.lam_tensor_3d),
           (octa_tilt.with_theta_eigenstrain, octa_tilt.with_theta_eigenstrain_3d,
            octa_tilt.with_theta_eigenstrain_film, octa_tilt.lam_tensor_theta_3d)]
_sleep = time.sleep


# --------------------------------------------------------------------------
# D1: the q/theta eigenstrain is ADDED to the base's static eigenstrain
# --------------------------------------------------------------------------
@pytest.mark.parametrize('with_2d, with_3d, with_film, lam_3d', SECTORS)
def test_order_eigenstrain_keeps_static_2d(with_2d, with_3d, with_film, lam_3d):
    grid = solver.make_grid(12, 10, 12., 10.)
    static = disorder.assemble_eps_star_static(
        grid, disorder.make_random_strain(grid, .005, 2., seed=1))
    el = elastic.setup_elastic(grid, 1.7, .8, .6, .09, -.03, .05, eps_star_static=static)
    rng = np.random.default_rng(0)
    P, q = rng.normal(0, .3, (3, 10, 12)), rng.normal(0, .3, (3, 10, 12))
    lam = (LAM['Lam11'], LAM['Lam12'], LAM['Lam44'])
    # q == 0 must leave the base untouched: same static, same force on P
    zero = with_2d(el, np.zeros_like(q), *lam)
    np.testing.assert_array_equal(elastic.elastic_force(P, zero)[0],
                                  elastic.elastic_force(P, el)[0])
    # q != 0: static + q's eigenstrain, and re-feeding the output is idempotent
    out = with_2d(el, q, *lam)
    np.testing.assert_array_equal(out['eps_star_static'], static + elastic.eigenstrain(q, *lam))
    np.testing.assert_array_equal(with_2d(with_2d(el, 2 * q, *lam), q, *lam)['eps_star_static'],
                                  out['eps_star_static'])
    assert el['eps_star_static'] is static            # the base dict is not mutated


def test_total_energy_pq_keeps_static():
    grid = solver.make_grid(12, 10, 12., 10.)
    static = disorder.assemble_eps_star_static(
        grid, disorder.make_random_strain(grid, .005, 2., seed=1))
    el = elastic.setup_elastic(grid, 1.7, .8, .6, .09, -.03, .05, eps_star_static=static)
    ops = solver.build_operators(grid, 1., 0., .5, .5, .1)
    P = np.random.default_rng(3).normal(0, .3, (3, 10, 12))
    zero = dict(a1=0., a11=0., a12=0., a111=0., a112=0., a123=0.)
    dens, _ = kittel_afe.total_energy_pq(P, np.zeros_like(P), zero, zero,
                                         dict(t1=0., t2=0., u1=0., u2=0.), ops, grid, 1.,
                                         elastic_params=el, Lam=LAM)
    expected = float(elastic.elastic_energy_density(elastic.elastic_solve(P, el)).mean())
    assert dens['elastic'] == expected


@pytest.mark.parametrize('with_2d, with_3d, with_film, lam_3d', SECTORS)
def test_order_eigenstrain_keeps_static_3d(with_2d, with_3d, with_film, lam_3d):
    grid = bulk3d.make_grid_3d(6, 5, 4, 6., 5., 4.)
    rng = np.random.default_rng(5)
    s = rng.normal(0, .003, (3, 3, 4, 5, 6))
    static = .5 * (s + s.swapaxes(0, 1))
    el = bulk3d.setup_elastic_3d(grid, (1.7, .8, .6), (.09, -.03, .05 / 4), eps_star_static=static)
    q = rng.normal(0, .3, (3, 4, 5, 6))
    out = with_3d(el, q, LAM)
    np.testing.assert_array_equal(out['eps_star_static'],
                                  static + bulk3d.eigenstrain_3d(q, *lam_3d(LAM)))
    np.testing.assert_array_equal(with_3d(out, q, LAM)['eps_star_static'], out['eps_star_static'])


@pytest.mark.parametrize('with_2d, with_3d, with_film, lam_3d', SECTORS)
def test_order_eigenstrain_keeps_static_film(with_2d, with_3d, with_film, lam_3d):
    grid = film3d.make_grid_3d(7, 5, 7 * .7, 5 * .8, 5, 2.4)
    stack = fe.make_stack_grid(grid, hs_dx=2)
    probe = fe.setup_film_elastic(stack, 275., 179., 54.3, .1, -.034, .029, e0=-.003)
    shape, lo = (3, 3, probe['Nz'], probe['Ny'], probe['Nx']), probe['film_lo']
    rng = np.random.default_rng(9)
    static = np.zeros(shape)
    s = rng.normal(0, .002, (3, 3) + shape[2:])
    static[:, :, lo:] = (.5 * (s + s.swapaxes(0, 1)))[:, :, lo:]
    el = fe.setup_film_elastic(stack, 275., 179., 54.3, .1, -.034, .029, e0=-.003,
                               eps_star_static=static)
    P = rng.normal(scale=.12, size=(3, 5, 5, 7))
    q = rng.normal(scale=.2, size=P.shape)
    np.testing.assert_array_equal(fe.elastic_force_film(P, with_film(el, np.zeros_like(q), LAM))[0],
                                  fe.elastic_force_film(P, el)[0])
    out = with_film(el, q, LAM)
    expected = static.copy()
    expected[:, :, lo:] += elastic.eigenstrain(q, LAM['Lam11'], LAM['Lam12'], LAM['Lam44'])
    np.testing.assert_array_equal(out['eps_star_static'], expected)
    assert not np.any(out['eps_star_static'][:, :, :lo])


# --------------------------------------------------------------------------
# D2: piecewise_regions_1d wraps its last segment through 0
# --------------------------------------------------------------------------
def test_piecewise_regions_wrap():
    grid = solver.make_grid(10, 4, 10., 4.)
    f = disorder.piecewise_regions_1d(grid, 0, [2., 6.], [1., 5.])
    np.testing.assert_array_equal(f, np.tile([5., 5., 1, 1, 1, 1, 5, 5, 5, 5], (4, 1)))
    np.testing.assert_array_equal(disorder.piecewise_regions_1d(grid, 0, [3.], [7.]),
                                  np.full((4, 10), 7.))
    grid = solver.make_grid(4, 10, 4., 10.)
    f = disorder.piecewise_regions_1d(grid, 1, [3., 7.], [-1., 2.])
    np.testing.assert_array_equal(f[:, 0], [2., 2, 2, -1, -1, -1, -1, 2, 2, 2])


# --------------------------------------------------------------------------
# D3: the 2D random-field generator refuses a bulk3d grid with Nz > 1
# --------------------------------------------------------------------------
def test_random_field_gaussian_refuses_bulk3d_grid():
    grid = bulk3d.make_grid_3d(8, 6, 3, 8., 6., 3.)
    with pytest.raises(ValueError, match='random_field_gaussian_3d'):
        disorder.random_field_gaussian(grid, .3, 1.5, seed=0)
    with pytest.raises(ValueError, match='random_field_gaussian_3d'):
        quenched_fields.charge_disorder_field(grid, conc=.5, delta_q=3., lambda_s_nm=2.,
                                              eps_r=1000., a_lattice_nm=.4049, seed=1)
    # 2D grids and film grids (in-plane spectrum) are unaffected
    for grid in (solver.make_grid(8, 6, 8., 6.), film3d.make_grid_3d(8, 6, 8., 6., 3, 2.)):
        field = disorder.random_field_gaussian(grid, .3, 1.5, seed=0)
        assert field.shape == (6, 8) and np.isclose(field.std(), .3)


# --------------------------------------------------------------------------
# D4: tools/job.py
# --------------------------------------------------------------------------
_CASE = ('import os, signal, sys, time\n'
         'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
         'with open(sys.argv[1], "w") as fh:\n'
         '    fh.write(str(os.getpid()))\n'
         'time.sleep(60)\n')                                  # bounded even if teardown fails
_WRAPPER = ('import subprocess, sys\n'
            'subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]]).wait()\n')


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(job, 'JOBS_DIR', str(tmp_path / 'jobs'))
    monkeypatch.setattr(job, 'RUNS_DIR', str(tmp_path / 'runs'))
    return tmp_path


def _record(tmp_path, job_id, pid=None, pgid=None):
    log = tmp_path / 'run.log'
    log.touch()
    rec = dict(id=job_id, script='x.py', argv=['x.py'], pid=pid, pgid=pgid,
               started=job._now(), ended=None, exit=None, state='running',
               log=os.path.relpath(log, job._ROOT),
               run_dir=os.path.relpath(tmp_path, job._ROOT) + '/',
               commit='unknown', dirty=False, nice=10, env={})
    job._write(rec)
    return rec


def _dead(pid):
    """Gone, or a zombie waiting for init to reap it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        with open(f'/proc/{pid}/stat') as fh:
            return fh.read().rsplit(')', 1)[1].split()[0] == 'Z'
    except OSError:
        return False


def _wait(predicate, timeout=10.):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        _sleep(.05)
    return predicate()


@pytest.fixture
def case_group(tmp_path):
    """A job-shaped process group: a wrapper (own session, dies at SIGTERM)
    whose child ignores SIGTERM. The wrapper is reaped at once, as init reaps
    a real job.py wrapper; the whole group is SIGKILLed on teardown."""
    marker = tmp_path / 'case.pid'
    wrapper = subprocess.Popen([sys.executable, '-c', _WRAPPER, _CASE, str(marker)],
                               start_new_session=True)
    reaper = threading.Thread(target=wrapper.wait, daemon=True)
    reaper.start()
    try:
        assert _wait(lambda: marker.exists() and marker.read_text())
        yield wrapper.pid, int(marker.read_text()), reaper
    finally:
        try:
            os.killpg(wrapper.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        reaper.join(5.)


def test_job_kill_reaches_case_that_ignores_sigterm(jobs, case_group, monkeypatch, capsys):
    wrapper_pid, case_pid, _ = case_group
    _record(jobs, 'k1', pid=wrapper_pid, pgid=wrapper_pid)
    monkeypatch.setattr(job.time, 'sleep', lambda s: _sleep(s / 10))
    job.kill(argparse.Namespace(job='k1'))
    assert _wait(lambda: _dead(case_pid)), 'case survived job.py kill'
    assert job._read('k1')['state'] == 'killed'


def test_job_status_follows_process_group(jobs, case_group, capsys):
    wrapper_pid, case_pid, reaper = case_group
    _record(jobs, 's1', pid=wrapper_pid, pgid=wrapper_pid)
    os.kill(wrapper_pid, signal.SIGTERM)                 # the wrapper only
    reaper.join(10.)
    assert not reaper.is_alive() and not _dead(case_pid)
    assert job._refresh('s1')['state'] == 'running'      # the case still computes
    job.status(argparse.Namespace(job='all'))
    assert 'running' in capsys.readouterr().out
    # a group that is really gone is still closed out as killed
    done = subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
    done.wait()
    _record(jobs, 's2', pid=done.pid, pgid=done.pid)
    assert job._refresh('s2')['state'] == 'killed'


def test_job_tail_n0_prints_nothing(jobs, capsys):
    rec = _record(jobs, 't1', pid=os.getpid(), pgid=os.getpgid(0))
    with open(os.path.join(job._ROOT, rec['log']), 'w') as fh:
        fh.write('one\ntwo\nthree\n')
    job.tail(argparse.Namespace(job='t1', n=0, follow=False))
    assert capsys.readouterr().out == ''
    job.tail(argparse.Namespace(job='t1', n=2, follow=False))
    assert capsys.readouterr().out == 'two\nthree\n'


def test_job_write_is_atomic(jobs, monkeypatch, capsys):
    rec = _record(jobs, 'w1', pid=os.getpid(), pgid=os.getpgid(0))
    path = job._job_path('w1')
    dump = json.dump

    def concurrent_reader(obj, fh, **kw):               # a status/kill reading mid-write
        with open(path) as other:
            assert json.load(other)['state'] == 'running'
        return dump(obj, fh, **kw)

    monkeypatch.setattr(job.json, 'dump', concurrent_reader)
    job._write(dict(rec, state='done'))
    monkeypatch.setattr(job.json, 'dump', dump)
    assert job._read('w1')['state'] == 'done'
    assert os.listdir(job.JOBS_DIR) == ['w1.json']
    # a writer's temp file caught mid-flight is not listed as a job
    open(path + '.12345.tmp', 'w').close()
    job.status(argparse.Namespace(job='all'))
    assert len(capsys.readouterr().out.strip().splitlines()) == 2


# --------------------------------------------------------------------------
# D6: casekit.anneal(restart=...) refuses a changed mobility L
# --------------------------------------------------------------------------
def test_anneal_restart_refuses_changed_mobility():
    grid = solver.make_grid(8, 8, 8., 8.)
    ops = solver.build_operators(grid, 1., 0., .5, .5, .1)
    alpha = dict(a1=-.3, a11=.2, a12=.5, a111=.4, a112=.3, a123=.6)
    P0 = np.random.default_rng(4).normal(0, .05, (3, 8, 8))
    kw = dict(dt=.01, n_max=2, sample_every=1, tol=-1., return_state=True)
    P, _, _, state = casekit.anneal(solver, P0, alpha, ops, grid, L=1., **kw)
    with pytest.raises(ValueError, match='restart'):
        casekit.anneal(solver, P, alpha, ops, grid, L=2., restart=state, **kw)
    assert casekit.anneal(solver, P, alpha, ops, grid, L=1., restart=state, **kw)[3]['step'] == 4
