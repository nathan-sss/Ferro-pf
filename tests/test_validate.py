"""Setup-time input validation (solvers/validate.py): every check rejects a bad
input with a clear message, and every deck, registry entry and fixture the
repository uses still passes and still gives the same numbers."""
import glob
import os
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '1')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from cases.multicore_fixture import Fixture
from solvers import (bulk3d, casekit, composite, crystallinity, deck_parser, elastic,
                     film3d, film_elastic, materials, materials_archive, polycrystal,
                     solver, validate)

DECK = materials.get('bto_lcc2005_8th', T=298.0)
ALPHA = DECK['landau']
C = DECK['elastic']
Q = DECK['electrostriction']
GRADIENT = DECK['gradient']
KAPPA = DECK['kappa_b']


def grid2(n=8, h=0.4):
    return solver.make_grid(Nx=n, Ny=n, Lx=n * h, Ly=n * h)


def grid3(n=6, h=0.5):
    return bulk3d.make_grid_3d(Nx=n, Ny=n, Nz=n, Lx=n * h, Ly=n * h, Lz=n * h)


def raises(match):
    return pytest.raises(validate.ValidationError, match=match)


# --------------------------------------------------------------------------
# The key lists are the engine's own
# --------------------------------------------------------------------------
def test_landau_keys_match_the_engine():
    assert validate.LANDAU_REQUIRED == solver.ALPHA_KEYS
    assert validate.LANDAU_OPTIONAL == solver.ALPHA_KEYS_8TH
    assert set(materials._ALPHA_TO_ENGINE) == set(deck_parser.TOML_SECTIONS['landau'])
    assert set(materials._ALPHA_TO_ENGINE.values()) == set(validate.LANDAU_KEYS)
    assert issubclass(validate.ValidationError, ValueError)


# --------------------------------------------------------------------------
# Everything the repository ships passes
# --------------------------------------------------------------------------
@pytest.mark.parametrize('name', [name for name, _ in materials.list()])
def test_every_registry_entry_passes(name):
    deck = materials.get(name)
    assert validate.deck(deck) is deck


TOMLS = sorted(glob.glob(os.path.join(ROOT, 'params', '*.toml'))
               + glob.glob(os.path.join(ROOT, 'params', 'substrates', '*.toml')))


@pytest.mark.parametrize('path', TOMLS, ids=os.path.basename)
def test_every_toml_deck_passes(path):
    validate.toml_deck(path)
    deck = materials.from_toml(path)
    assert validate.deck(deck) is deck


def test_array_valued_deck_passes():
    x = np.linspace(0.45, 0.85, 12).reshape(3, 4)
    deck = materials.get('pzt_haun_x', x=x)
    assert np.shape(deck['landau']['a1']) == (3, 4)
    assert np.shape(deck['elastic']['C11']) == (3, 4)


@pytest.mark.parametrize('engine', ['2d', 'bulk3d', 'film3d'])
def test_multicore_fixtures_pass(engine):
    fixture = Fixture(engine, 8)
    assert fixture.P.shape[0] == 3
    if engine != 'film3d':
        module = solver if engine == '2d' else bulk3d
        casekit.anneal(module, fixture.P, fixture.alpha, fixture.ops, fixture.grid,
                       dt=fixture.dt, elastic_params=fixture.el, n_max=2, sample_every=1)


def test_extra_alpha_keys_found_in_the_repository_pass():
    grid = grid2()
    ops = solver.build_operators(grid, **GRADIENT, kappa=KAPPA)
    P = solver.random_smooth_field(grid, 3, 0.01, seed=1)
    shell = dict(a1=0.5, a11=0.0, a12=0.0, a111=0.0, a112=0.0, a123=0.0)
    L = np.zeros((8, 8))
    L[:, :2] = 1.0
    theta = np.where(np.arange(8)[None, :] < 4, 0.0, 30.0) * np.ones((8, 1))
    cos, sin = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    variants = {
        'polycrystal _cos/_sin': (polycrystal, polycrystal.make_alpha(ALPHA, shell, L, cos, sin)),
        'composite fields': (solver, composite.alpha_field(L, ALPHA, dict(ALPHA, a1=0.3))),
        'crystallinity a1 field': (solver, crystallinity.effective_alpha(
            ALPHA, Xc=np.full((8, 8), 0.5), A=0.2, shape=(8, 8))),
        '6th-order only': (solver, {k: ALPHA[k] for k in solver.ALPHA_KEYS}),
    }
    for label, (engine, alpha) in variants.items():
        validate.landau_coefficients(alpha, field_shape=(8, 8))
        out = casekit.anneal(engine, P, alpha, ops, grid, dt=0.01, n_max=2, sample_every=1)
        assert np.all(np.isfinite(out[0])), label


def test_ensemble_restart_and_field_e_ext_pass():
    grid = grid2()
    ops = solver.build_operators(grid, **GRADIENT, kappa=KAPPA)
    P = np.stack([solver.random_smooth_field(grid, 3, 0.01, seed=s) for s in range(2)], axis=1)
    E_ext = np.zeros((3, 1, 8, 8))
    P1, _, _, state = casekit.anneal(solver, P, ALPHA, ops, grid, dt=0.01, E_ext=E_ext,
                                     n_max=2, sample_every=1, return_state=True)
    casekit.anneal(solver, P1, ALPHA, ops, grid, dt=0.01, E_ext=E_ext, n_max=2,
                   sample_every=1, restart=state)


def test_full_spectrum_grid_variant_passes():
    """cases/wp16_d17_analysis.py (and the K5 gate it transcribes) swap the
    half-spectrum k-arrays for the full ones; that must stay legal."""
    grid = dict(grid2())
    grid['KXr'], grid['KYr'] = grid['KX'], grid['KY']
    solver.build_operators(grid, 0.51, -0.02, 0.02, 0.0, 1e12)


def test_negative_gradient_coefficients_pass():
    """Modulated-phase decks (Runs/Wang2026_NNBM5_*) use G < 0 with H11 > 0."""
    solver.build_operators(grid2(), -1.3, 0.0, -0.65, -0.65, KAPPA)


def test_validation_changes_no_result(monkeypatch):
    grid = grid2()
    P = solver.random_smooth_field(grid, 3, 0.01, seed=3)

    def run():
        ops = solver.build_operators(grid, **GRADIENT, kappa=KAPPA)
        el = elastic.setup_elastic(grid, C['C11'], C['C12'], C['C44'],
                                   Q['Q11_phys'], Q['Q12_phys'], Q['Q44_phys'])
        return casekit.anneal(solver, P, ALPHA, ops, grid, dt=0.01, elastic_params=el,
                              n_max=3, sample_every=1)

    checked = run()
    for name in ('build_operators', 'setup_elastic', 'anneal'):
        monkeypatch.setattr(validate, name, lambda *a, **k: None)
    unchecked = run()
    assert checked[0].tobytes() == unchecked[0].tobytes()
    assert checked[1] == unchecked[1]


def test_inputs_are_not_modified():
    alpha = dict(ALPHA, a1=np.full((8, 8), ALPHA['a1']))
    before = {k: np.array(v, copy=True) for k, v in alpha.items()}
    validate.landau_coefficients(alpha, field_shape=(8, 8))
    assert alpha.keys() == before.keys()
    assert all(np.array_equal(alpha[k], before[k]) for k in alpha)


# --------------------------------------------------------------------------
# Landau dicts
# --------------------------------------------------------------------------
@pytest.mark.parametrize('key, match', [
    ('alpha1', "deck-style name; the engine key is 'a1'"),
    ('alpha1123', "deck-style name; the engine key is 'a1123'"),
    ('A11', "'A11' \\(misspelled; the engine key is 'a11'"),
    ('a1121', "'a1121' \\(not a term of the engine"),
    ('a_1', "misspelled; the engine key is 'a1'"),
    ('Alpha_12', "deck-style name; the engine key is 'a12'"),
    ('aq1', "'aq1'"),
])
def test_unrecognized_landau_key_is_rejected(key, match):
    with raises(match):
        validate.landau_coefficients(dict(ALPHA, **{key: 0.1}))


def test_unrecognized_landau_key_is_rejected_at_anneal():
    grid = grid2()
    ops = solver.build_operators(grid, **GRADIENT, kappa=KAPPA)
    P = solver.random_smooth_field(grid, 3, 0.01, seed=1)
    with raises('casekit.anneal alpha: unrecognized Landau key'):
        casekit.anneal(solver, P, dict(ALPHA, a1121=0.2), ops, grid, dt=0.01, n_max=1,
                       sample_every=1)


def test_landau_values():
    with raises(r"missing required Landau key\(s\) \['a12'\]"):
        validate.landau_coefficients({k: v for k, v in ALPHA.items() if k != 'a12'})
    validate.landau_coefficients({'a1': 0.1}, require=False)
    with raises(r"alpha\['a11'\] is not finite"):
        validate.landau_coefficients(dict(ALPHA, a11=np.nan))
    with raises(r"alpha\['a1'\] must be a real number or array, got '0.1'"):
        validate.landau_coefficients(dict(ALPHA, a1='0.1'))
    with raises(r"alpha\['a1'\] must be a real number"):
        validate.landau_coefficients(dict(ALPHA, a1=None))
    with raises('does not broadcast onto the grid'):
        validate.landau_coefficients(dict(ALPHA, a1=np.zeros((8, 7))), field_shape=(8, 8))
    with raises('does not broadcast onto the grid'):
        validate.landau_coefficients(dict(ALPHA, a1=np.zeros((2, 8, 8))), field_shape=(8, 8))
    with raises('incompatible shapes'):
        validate.landau_coefficients(dict(ALPHA, a1=np.zeros((4, 4)), a11=np.zeros((3, 3))))
    with raises('expected a dict'):
        validate.landau_coefficients([0.1] * 6)


# --------------------------------------------------------------------------
# P and E_ext at anneal start
# --------------------------------------------------------------------------
def test_anneal_start_checks():
    grid = grid2()
    ops = solver.build_operators(grid, **GRADIENT, kappa=KAPPA)
    P = solver.random_smooth_field(grid, 3, 0.01, seed=1)

    def go(P=P, E_ext=(0., 0., 0.), engine=solver, alpha=ALPHA):
        casekit.anneal(engine, P, alpha, ops, grid, dt=0.01, E_ext=E_ext, n_max=1,
                       sample_every=1)

    with raises(r'P has shape \(3, 8, 7\)'):
        go(P=P[:, :, :7])
    with raises(r'P has shape \(2, 8, 8\)'):
        go(P=P[:2])
    with raises(r"grid lacks \['Nz'\] for a 3D engine"):
        go(engine=bulk3d)
    bad = P.copy()
    bad[0, 0, 0] = np.nan
    with raises('P is not finite'):
        go(P=bad)
    with raises(r'E_ext has shape \(4,\)'):
        go(E_ext=(0., 0., 0., 1.))
    with raises(r'E_ext has shape \(\)'):
        go(E_ext=0.0)
    with raises('E_ext is not finite'):
        go(E_ext=(0., np.inf, 0.))
    ensemble = np.stack([P, P], axis=1)
    with raises(r'E_ext has shape \(3, 8, 8\)'):
        go(P=ensemble, E_ext=np.zeros((3, 8, 8)))
    with raises(r'missing required Landau'):
        go(alpha={'a1': 0.1})
    with pytest.raises(ValueError, match='is not solver or bulk3d'):
        go(engine=film3d)      # left to anneal's own check


# --------------------------------------------------------------------------
# Grids and operators
# --------------------------------------------------------------------------
def test_grid_checks():
    g = grid2()
    with raises(r'inconsistent with Lx/Nx.*L = N\*h'):
        solver.build_operators(dict(g, hx=0.5), **GRADIENT, kappa=KAPPA)
    with raises(r"grid\['Lx'\] must be > 0"):
        solver.build_operators(dict(g, Lx=-3.2), **GRADIENT, kappa=KAPPA)
    with raises(r"grid\['Nx'\] must be an integer"):
        solver.build_operators(dict(g, Nx=8.0), **GRADIENT, kappa=KAPPA)
    with raises(r"grid\['Ny'\] must be >= 1"):
        solver.build_operators(dict(g, Ny=0), **GRADIENT, kappa=KAPPA)
    with raises(r"grid lacks \['hy'\]"):
        solver.build_operators({k: v for k, v in g.items() if k != 'hy'}, **GRADIENT,
                               kappa=KAPPA)
    with raises('a 2D engine needs 2-D arrays'):
        solver.build_operators(grid3(), **GRADIENT, kappa=KAPPA)
    with raises(r"grid\['hz'\]=1.0 is inconsistent with Lz/Nz"):
        bulk3d.build_operators_3d(dict(grid3(), hz=1.0), **GRADIENT, kappa=KAPPA)


def test_operator_coefficient_checks():
    g = grid2()
    with raises('solver.build_operators: kappa must be > 0'):
        solver.build_operators(g, **GRADIENT, kappa=0.0)
    with raises('kappa is not finite'):
        solver.build_operators(g, **GRADIENT, kappa=np.nan)
    with raises(r'kappa array shape \(4, 4\) != grid \(8, 8\)'):
        solver.build_operators(g, **GRADIENT, kappa=np.ones((4, 4)))
    with raises('G11 is not finite'):
        solver.build_operators(g, **dict(GRADIENT, G11=np.inf), kappa=KAPPA)
    with raises('G44 must be a scalar'):
        solver.build_operators(g, **dict(GRADIENT, G44=np.ones((8, 8))), kappa=KAPPA)
    with raises('kappa0_override must be > 0'):
        bulk3d.build_operators_3d(grid3(), **GRADIENT, kappa=np.full((6, 6, 6), KAPPA),
                                  kappa0_override=-1.0)


# --------------------------------------------------------------------------
# Elastic setups
# --------------------------------------------------------------------------
@pytest.mark.parametrize('c11, c12, c44, rule', [
    (100.0, 120.0, 50.0, r'C11 > \|C12\|'),
    (100.0, -60.0, 50.0, r'C11 \+ 2\*C12 > 0'),
    (200.0, 70.0, 0.0, r'C44 > 0'),
])
def test_stiffness_must_be_positive_definite(c11, c12, c44, rule):
    q = (Q['Q11_phys'], Q['Q12_phys'], Q['Q44_phys'])
    with raises('elastic.setup_elastic: cubic stiffness .* not positive-definite .*' + rule):
        elastic.setup_elastic(grid2(), c11, c12, c44, *q)
    with raises('bulk3d.setup_elastic_3d: cubic stiffness c1111=.*' + rule):
        bulk3d.setup_elastic_3d(grid3(), C=(c11, c12, c44), Q=(0.1, -0.03, 0.01))
    stack = film_elastic.make_stack_grid(film3d.make_grid_3d(4, 4, 1.6, 1.6, 5, 1.6), hs_dx=2)
    with raises('film_elastic.setup_film_elastic: cubic stiffness .*' + rule):
        film_elastic.setup_film_elastic(stack, c11, c12, c44, *q, e0=-0.01)


def test_elastic_2d_checks():
    g = grid2()
    c = (C['C11'], C['C12'], C['C44'])
    q = (Q['Q11_phys'], Q['Q12_phys'], Q['Q44_phys'])
    elastic.setup_elastic(g, *c, np.full((8, 8), q[0]), *q[1:])          # Q field: fine
    elastic.setup_elastic(g, *c, *q, eps_star_static=np.zeros((3, 3, 2, 8, 8)))  # ensemble
    with raises('C11 is not finite'):
        elastic.setup_elastic(g, np.nan, *c[1:], *q)
    with raises('C11 must be a scalar'):
        elastic.setup_elastic(g, np.full((8, 8), c[0]), *c[1:], *q)
    with raises('Q12 is not finite'):
        elastic.setup_elastic(g, *c, q[0], np.nan, q[2])
    with raises(r'Q11 has shape \(8, 7\)'):
        elastic.setup_elastic(g, *c, np.zeros((8, 7)), *q[1:])
    with raises(r'eps_star_static must have shape \(3, 3, 8, 8\)'):
        elastic.setup_elastic(g, *c, *q, eps_star_static=np.zeros((8, 8, 3, 3)))
    with raises('eps_m must be a scalar'):
        elastic.setup_elastic(g, *c, *q, eps_m=np.zeros(2))


def test_elastic_3d_checks():
    g = grid3()
    with raises('C=\\(c1111, c1122, c1212\\) must have 3 entries'):
        bulk3d.setup_elastic_3d(g, C=(275.0, 179.0), Q=(0.1, -0.03, 0.01))
    with raises(r"Q\[1\] Q1212 is not finite"):
        bulk3d.setup_elastic_3d(g, C=(275.0, 179.0, 54.3), Q={0: (0.1, -0.03, 0.01),
                                                              1: (0.1, -0.03, np.nan)},
                                labels=np.zeros((6, 6, 6), dtype=int))
    with raises(r'Q1111 has shape \(5, 6, 6\)'):
        bulk3d.setup_elastic_3d(g, C=(275.0, 179.0, 54.3), Q=(np.zeros((5, 6, 6)), 0.0, 0.0))


def test_film_checks():
    grid = film3d.make_grid_3d(4, 4, 1.6, 1.6, 5, 1.6)
    q = (Q['Q11_phys'], Q['Q12_phys'], Q['Q44_phys'])
    c = (C['C11'], C['C12'], C['C44'])
    with raises('needs the film\\+substrate stack grid'):
        film_elastic.setup_film_elastic(grid, *c, *q, e0=-0.01)
    with raises('C_substrate needs substrate layers'):
        film_elastic.setup_film_elastic(film_elastic.make_stack_grid(grid, hs_dx=0), *c, *q,
                                        e0=-0.01, C_substrate=(200.0, 70.0, 60.0))
    stack = film_elastic.make_stack_grid(grid, hs_dx=2)
    with raises('e0 must be a scalar'):
        film_elastic.setup_film_elastic(stack, *c, *q, e0=np.zeros(2))
    with raises(r'Q11 has shape \(7, 4, 4\)'):
        film_elastic.setup_film_elastic(stack, *c, np.zeros((7, 4, 4)), *q[1:], e0=-0.01)
    with raises('film3d.make_grid_3d: Nz must be an integer'):
        film3d.make_grid_3d(4, 4, 1.6, 1.6, 5.5, 1.6)
    film3d.make_grid_3d(4, 4, 1.6, 1.6, 5.0, 1.6)                       # int(Nz) is applied
    with raises('Nz must be >= 2'):
        film3d.make_grid_3d(4, 4, 1.6, 1.6, 1, 1.6)
    with raises('hf must be > 0'):
        film3d.make_grid_3d(4, 4, 1.6, 1.6, 5, 0.0)
    with raises('kappa_b must be > 0'):
        film3d.poisson_factor(grid, -0.1)
    with raises(r'kappa_b must be a scalar or a \(Nz,\)=\(5,\) node profile'):
        film3d.poisson_factor(grid, np.ones(4))
    film3d.poisson_factor(grid, np.linspace(0.1, 0.2, 5))


# --------------------------------------------------------------------------
# Decks
# --------------------------------------------------------------------------
def test_deck_checks():
    with raises("lacks its required 'meta'"):
        validate.deck({k: v for k, v in DECK.items() if k != 'meta'})
    with raises(r"unknown deck sector\(s\) \['landau_q'\]"):
        validate.deck(dict(DECK, landau_q={}))
    with raises("sector 'gradient' must have exactly"):
        validate.deck(dict(DECK, gradient=dict(GRADIENT, G13=0.0)))
    with raises("sector 'elastic' must have exactly"):
        validate.deck(dict(DECK, elastic={'C11': 1.0, 'C12': 0.5}))
    with raises('not positive-definite'):
        validate.deck(dict(DECK, elastic=dict(C, C44=-1.0)))
    with raises("deck-style name; the engine key is 'a1'"):
        validate.deck(dict(DECK, landau=dict(ALPHA, alpha1=0.0)))
    with raises("'landau' is present but empty"):
        validate.deck(dict(DECK, landau={}))
    with raises('kappa_b must be > 0'):
        validate.deck(dict(DECK, kappa_b=0.0))


def test_casekit_builders_name_missing_sectors():
    with raises(r"casekit.build_2d: deck lacks sector\(s\) \['gradient', 'kappa_b'\]"):
        casekit.build_2d(materials_archive.get('pmn_clean_x0'), 8, 8, 0.4)
    with raises(r"casekit.build_3d: deck lacks sector\(s\) \['elastic', 'electrostriction'\]"):
        casekit.build_3d(materials_archive.get('bfo_pan2019'), 4, 0.5, elastic=True)
    casekit.build_3d(materials_archive.get('bfo_pan2019'), 4, 0.5)            # elastic off: fine
    with raises('cell size must be > 0'):
        casekit.build_2d(DECK, 8, 8, -0.4)
    with raises(r'size\[0\] must be an integer'):
        casekit.build_2d(DECK, 8.0, 8, 0.4)


GOOD_TOML = """[meta]
source = "test"
[landau]
alpha1 = -0.1
alpha11 = 0.2
[gradient]
G110 = 1.0
G11_G110 = 1.0
G12_G110 = 0.0
G44_G110 = 0.5
G44P_G110 = 0.5
[elastic]
C11 = 200.0
C12 = 70.0
C44 = 60.0
"""


@pytest.mark.parametrize('edit, match', [
    (('alpha11 = 0.2', 'alpha11 = 0.2\nalpha1121 = 0.3'), r"\['alpha1121'\] are not read"),
    (('alpha11 = 0.2', 'a11 = 0.2'), r"\['a11'\] are not read"),
    (('[landau]', '[Landau]'), r'section \[Landau\] is never read'),
    (('C44 = 60.0\n', ''), r"\[elastic\] has only \['C11', 'C12'\]"),
    (('G44P_G110 = 0.5\n', ''), 'G110 but not all of'),
    (('alpha11 = 0.2', 'alpha11 = inf'), r'\[landau\] alpha11 is not finite'),
    (('C11 = 200.0', 'C11 = "200"'), r'\[elastic\] C11 must be a real number'),
    (('C44 = 60.0', 'C44 = -60.0'), 'not positive-definite'),
])
def test_toml_deck_rejections(tmp_path, edit, match):
    good = tmp_path / 'good.toml'
    good.write_text(GOOD_TOML)
    assert set(materials.from_toml(str(good))) == {'landau', 'gradient', 'elastic', 'meta'}
    bad = tmp_path / 'bad.toml'
    bad.write_text(GOOD_TOML.replace(*edit))
    with raises(match):
        materials.from_toml(str(bad))
