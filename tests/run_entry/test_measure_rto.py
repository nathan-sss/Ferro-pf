"""RUN-ENTRY E4, scheme "rto" of measure.phase_fractions: tetragonal / orthorhombic / rhombohedral cones (SPEC § 16.12).

Portable: nothing outside the repository.  The look-test at the end is a check of the classifier on a physical example
(uniform 0D states of a registry deck), not a physics claim about the deck.
"""
import itertools
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('MPLBACKEND', 'Agg')

from test_drive import read_csv, run_raw, same_run, tiny2d   # noqa: E402
from fftpf import landau_0d_diagnostics, mat_registry, run_builder, run_input, run_measure   # noqa: E402
from fftpf.run_input import ConfigError   # noqa: E402

PRM = dict(scheme='rto', angle_deg=15.0, P_min=0.05)
KEYS = ('R', 'O', 'T', 'other', 'low')
SUM_TOL = 1e-12        # five cell means of k/N: the sum is 1 up to a few ulp (~1e-16); 1e-12 leaves room for the film weighting


def uniform(vec, shape=(1, 4, 4)):
    return np.array(vec, float).reshape(3, 1, 1, 1) * np.ones((3,) + shape)


def only(f, key):
    return f == {k: float(k == key) for k in KEYS}


def family_vectors():
    """The 6 <100>, 12 <110>, 8 <111> unit vectors, by family."""
    out = {'T': [], 'O': [], 'R': []}
    for v in itertools.product((-1., 0., 1.), repeat=3):
        n = int(np.count_nonzero(v))
        if n:
            out['TOR'[n - 1]].append(np.array(v) / np.sqrt(n))
    return {k: np.array(v) for k, v in out.items()}


# --------------------------------------------------------------------------- a. hand-built fields

def test_pure_directions_in_every_sign_and_permutation_variant():
    fam = family_vectors()
    assert [len(fam[k]) for k in 'TOR'] == [6, 12, 8]
    for key, vecs in fam.items():
        for v in vecs:
            for mag in (0.06, 0.3, 0.8):                       # |P| from just above P_min to large: only the direction counts
                assert only(run_measure.fractions(uniform(mag * v), PRM), key), (key, v, mag)


def tilted(n0, w, deg):
    return np.cos(np.radians(deg)) * n0 + np.sin(np.radians(deg)) * w


def test_vectors_just_inside_and_just_outside_the_cone():
    fam = family_vectors()
    perp = {'T': (np.array([1., 0., 0.]), np.array([0., 1., 0.])),
            'O': (np.array([1., 1., 0.]) / np.sqrt(2.), np.array([0., 0., 1.])),
            'R': (np.array([1., 1., 1.]) / np.sqrt(3.), np.array([1., -1., 0.]) / np.sqrt(2.))}
    for key, (n0, w) in perp.items():
        assert np.isclose(n0 @ w, 0.)
        assert run_measure.fractions(uniform(0.3 * tilted(n0, w, 14.99)), PRM)[key] == 1.
        f = run_measure.fractions(uniform(0.3 * tilted(n0, w, 15.01)), PRM)
        assert only(f, 'other'), (key, f)
    for ang in (5., 17.4):                                     # the angle is a parameter: a 16 deg vector is T at 17.4 only
        f = run_measure.fractions(uniform(tilted(*perp['T'], 16.) * 0.3), dict(PRM, angle_deg=ang))
        assert only(f, 'T' if ang > 16 else 'other')


def test_low_and_zero_vector():
    assert only(run_measure.fractions(uniform([0.04, 0., 0.]), PRM), 'low')
    assert only(run_measure.fractions(uniform([0.02, 0.02, 0.02]), PRM), 'low')       # |P| = 0.035
    assert only(run_measure.fractions(uniform([0., 0., 0.]), PRM), 'low')
    assert only(run_measure.fractions(uniform([0.05, 0., 0.]), PRM), 'T')               # |P| = P_min is not low (as in "rt")
    assert only(run_measure.fractions(uniform([0.04, 0., 0.]), dict(PRM, P_min=0.03)), 'T')


def test_three_way_stripes_give_a_third_each():
    s = np.concatenate([uniform([0.3, 0., 0.], (1, 4, 2)), uniform([0.2, -0.2, 0.], (1, 4, 2)),
                        uniform([0.17, 0.17, -0.17], (1, 4, 2))], axis=3)
    f = run_measure.fractions(s, PRM)
    assert f['T'] == f['O'] == f['R'] == 1. / 3. and f['other'] == f['low'] == 0.


def test_film_weighting_as_in_rt():
    rng = np.random.default_rng(3)
    P = rng.normal(0., 0.25, (3, 3, 4, 5))
    qw = np.array([0.5, 1., 0.5])
    got = run_measure.fractions(P, PRM, qw)
    layers = [run_measure.fractions(P[:, z:z + 1], PRM) for z in range(3)]
    for k in KEYS:
        assert abs(got[k] - sum(w * l[k] for w, l in zip(qw, layers)) / qw.sum()) < SUM_TOL
    assert abs(sum(got.values()) - 1.) < SUM_TOL


# --------------------------------------------------------------------------- b, c. partition and brute force

def mixed_field(shape, seed):
    """Cells: a random one of the 26 directions, tilted by a random angle up to 40 deg, random |P| (some below P_min)."""
    rng = np.random.default_rng(seed)
    dirs = np.concatenate(list(family_vectors().values()))
    n = int(np.prod(shape))
    d = dirs[rng.integers(0, len(dirs), n)]
    noise = rng.normal(size=(n, 3))
    noise -= (noise * d).sum(1, keepdims=True) * d
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    ang = np.radians(rng.uniform(0., 40., n))
    mag = np.where(rng.random(n) < 0.1, rng.uniform(0., 0.06, n), rng.uniform(0.1, 0.5, n))
    P = mag[:, None] * (np.cos(ang)[:, None] * d + np.sin(ang)[:, None] * noise)
    return P.T.reshape((3,) + shape)


@pytest.mark.parametrize('shape', [(24, 30), (1, 24, 30), (5, 9, 11)])
def test_partition_identity(shape):
    P = mixed_field(shape, seed=len(shape) + shape[-1])
    m = run_measure.classes(P, PRM)
    assert set(m) == set(KEYS)
    count = sum(m[k].astype(int) for k in KEYS)
    assert (count == 1).all()                                   # every cell in exactly one class: exact, on the masks
    f = run_measure.fractions(P, PRM)
    assert abs(sum(f.values()) - 1.) < SUM_TOL                  # float summation of five means only
    assert all(f[k] > 0. for k in KEYS), f                      # the generator does populate every class


def brute_force(P, angle_deg, P_min):
    """Class per cell from the explicit angles (arccos of the dot product) to all 6 + 12 + 8 direction vectors."""
    fam = family_vectors()
    flat = P.reshape(3, -1).T
    mag = np.linalg.norm(flat, axis=1)
    unit = flat / np.where(mag > 0., mag, 1.)[:, None]
    ang = {k: np.degrees(np.arccos(np.clip(unit @ v.T, -1., 1.))).min(1) for k, v in fam.items()}
    m = {k: (ang[k] <= angle_deg) & (mag >= P_min) for k in 'TOR'}
    m['low'] = mag < P_min
    m['other'] = ~(m['T'] | m['O'] | m['R'] | m['low'])
    return {k: v.reshape(P.shape[1:]) for k, v in m.items()}


@pytest.mark.parametrize('shape,angle', [((40, 50), 15.), ((6, 12, 14), 15.), ((40, 50), 5.), ((40, 50), 17.5)])
def test_masks_equal_brute_force_angles(shape, angle):
    P = mixed_field(shape, seed=11)
    ref = brute_force(P, angle, 0.05)
    got = run_measure.classes(P, dict(PRM, angle_deg=angle))
    for k in KEYS:
        assert (got[k] == ref[k]).all(), k
    # phase_index agrees with the masks (0 R, 1 O, 2 T, 3 other, 4 low)
    idx = run_measure.phase_index(P, dict(PRM, angle_deg=angle))
    for i, k in enumerate(KEYS):
        assert ((idx == i) == got[k]).all()


# --------------------------------------------------------------------------- d, e. schema

def case(tmp, pf, **kw):
    return tiny2d(tmp, 'c', [{'kind': 'anneal', 'stop': {'n_steps': 3}}], measure={'phase_fractions': pf}, **kw)


def test_rt_is_the_default_and_unchanged(tmp_path):
    a = run_input.resolve(case(tmp_path, {'P_ref': 0.2}))['measure']['phase_fractions']
    b = run_input.resolve(case(tmp_path, {'P_ref': 0.2, 'scheme': 'rt'}))['measure']['phase_fractions']
    assert a == b == dict(P_ref=0.2, angle_deg=15.0, P_min=0.05, R_factor=0.7)
    assert run_measure.scheme_of(a) == 'rt' and run_measure.fraction_keys(a) == run_measure.FRACTION_KEYS


def test_rto_defaults_and_roundtrip(tmp_path):
    import tomllib
    cfg = run_input.resolve(case(tmp_path, {'scheme': 'rto'}))
    assert cfg['measure']['phase_fractions'] == dict(scheme='rto', angle_deg=15.0, P_min=0.05)
    assert run_input.resolve(tomllib.loads(run_input.dumps(cfg))) == cfg
    assert run_input.resolve(case(tmp_path, {'scheme': 'rto', 'angle_deg': 17.5}))['measure']['phase_fractions']['angle_deg'] == 17.5


def test_refusals(tmp_path):
    for pf, msg in [({'scheme': 'rto', 'angle_deg': 17.6}, r'angle_deg: must be <= 17.5 in scheme .rto.'),
                    ({'scheme': 'rto', 'angle_deg': 20.0}, r'angle_deg: must be <= 17.5'),
                    ({'scheme': 'rto', 'P_ref': 0.3}, r'P_ref: is not used by scheme .rto.'),
                    ({'scheme': 'rto', 'R_factor': 0.7}, r'R_factor: is not used by scheme .rto.'),
                    ({'scheme': 'rhombo'}, r'scheme: must be one of'),
                    ({'scheme': 'rt'}, r'P_ref: required'),
                    ({'angle_deg': 15.}, r'P_ref: required')]:
        with pytest.raises(ConfigError, match=r'^measure\.phase_fractions\.' + msg):
            run_input.resolve(case(tmp_path, pf))


def test_explain_names_the_scheme_and_thresholds(tmp_path):
    txt = run_builder.explain(run_input.resolve(case(tmp_path, {'scheme': 'rto', 'angle_deg': 12.0, 'P_min': 0.07})))
    line = [l for l in txt.splitlines() if l.startswith('phase fractions')]
    assert len(line) == 1 and 'scheme rto' in line[0] and '12 deg' in line[0] and '0.07' in line[0]
    txt = run_builder.explain(run_input.resolve(case(tmp_path, {'P_ref': 0.2})))
    assert 'scheme rt ' in txt and 'P_ref = 0.14' in txt


# --------------------------------------------------------------------------- f. end to end and resume

def rto_run_case(tmp, name):
    return tiny2d(tmp, name, [
        {'kind': 'anneal', 'langevin': True, 'stop': {'n_steps': 20}, 'sample_every': 5},
        {'kind': 'cool', 'rungs': [[330.0, 20], [298.0, 20]], 'langevin': True, 'record_every': 5}],
        output={'dir': str(tmp / name), 'checkpoint_every': 15},
        measure={'phase_fractions': {'scheme': 'rto'}, 'domain_map': {'kind': ['phase', 'direction']}})


def test_rto_run_writes_columns_and_phase_map(tmp_path):
    res = run_raw(rto_run_case(tmp_path, 'r'))
    rows = read_csv(os.path.join(res['run_dir'], 'trace.csv'))
    head = open(os.path.join(res['run_dir'], 'trace.csv')).readline().strip().split(',')
    assert [c for c in head if c.startswith('f_') and c != 'f_total'] == ['f_' + k for k in KEYS]
    for r in rows:
        assert abs(sum(float(r['f_' + k]) for k in KEYS) - 1.) < SUM_TOL
    m = json.load(open(os.path.join(res['run_dir'], 'measures.json')))
    assert set(m['stages'][1]['fractions_end']) == set(KEYS)
    assert m['stages'][1]['fractions_end'] == {k: float(rows[-1]['f_' + k]) for k in KEYS}
    figs = os.listdir(os.path.join(res['run_dir'], 'figures'))
    png = [f for f in figs if f.endswith('_phase.png')]
    assert png and all(open(os.path.join(res['run_dir'], 'figures', f), 'rb').read(8) == b'\x89PNG\r\n\x1a\n' for f in png)


@pytest.mark.parametrize('stop', [1, 3])
def test_rto_resume_equals_uninterrupted(tmp_path, stop):
    from fftpf import run_output
    full = run_raw(rto_run_case(tmp_path, 'full'))
    raw = rto_run_case(tmp_path, 'cut')
    with pytest.raises(run_output.RunInterrupted):
        run_raw(raw, stop_after_checkpoints=stop)
    res = run_raw(raw)
    same_run(full['run_dir'], res['run_dir'], full, res)                       # trace.csv and measures.json bytes, states
    assert open(os.path.join(full['run_dir'], 'trace.csv'), 'rb').read() == open(os.path.join(res['run_dir'], 'trace.csv'), 'rb').read()


# --------------------------------------------------------------------------- g. look-test

@pytest.mark.parametrize('T_K,want', [(340., 'T'), (245., 'O'), (100., 'R')])
def test_look_classifier_on_the_uniform_stable_state_of_bto_lcc2005_8th(T_K, want):
    """LOOK-TEST, not a physics claim: at three temperatures well inside the T, O and R ranges of the deck (0D transitions of
    the lowest-energy stable C/T/O/R branch, landau_0d_diagnostics.symmetry_stationary_points: R->O between 202 and 203 K,
    O->T between 281 and 282 K, T->C between 397 and 398 K; the independent tools/landau_minimize.minimize_bulk agrees at
    these three temperatures), the uniform stable state is filled into a small field and must be classified T, O, R."""
    alpha = mat_registry.get('bto_lcc2005_8th', T=T_K)['landau']
    stable = [r for r in landau_0d_diagnostics.symmetry_stationary_points(alpha) if r['stable']]
    assert stable[0]['branch'] == want, (T_K, [r['branch'] for r in stable])       # the deck gives the phase expected here
    f = run_measure.fractions(uniform(stable[0]['P'], (1, 6, 6)), PRM)
    assert only(f, want), f
