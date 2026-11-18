"""Regression tests for the 2026-11 materials/deck fixes (items A1-A6).

Each test failed on the code before its fix:
  A1 materials.pzt_haun_x_Q: Q12 had two sign errors (+0.0192 at x=0.5).
  A2 materials._bto_bc1984_6th_flat (archived 2026-11 to
     materials_archive._bto_bc1984_6th_flat): alpha111 used the Celsius
     reference 120 with T in kelvin (-7.07 at 298 K, potential unbounded below).
  A3 deck_parser.materials_to_toml / literature_params.write_toml demanded or
     placeholdered every key of the optional q/theta sectors.
  A4 composite.alpha_field/_multi dropped the 8th-order Landau keys.
  A5 polycrystal.make_alpha called float() on array-valued coefficients.
  A6 loopfit.check_global_min_constraint missed an FE well beyond P2.
"""
import os
import sys
import tomllib

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from solvers import (composite, crystallinity, deck_parser, landau0d,  # noqa: E402
                     literature_params, loopfit, materials, materials_archive, polycrystal,
                     solver, su_landis)

PARAMS = os.path.join(_ROOT, "params")
ARCHIVE = os.path.join(PARAMS, "archive")   # decks no computation loads (2026-11)


def _toml(name):
    with open(os.path.join(PARAMS, name), "rb") as fh:
        return tomllib.load(fh)


# --------------------------------------------------------------------------
# A1 -- Haun PZT Q12(x) sign, and every committed artifact derived from it
# --------------------------------------------------------------------------
def test_pzt_haun_q12_sign_matches_pto_end_member_and_mpb_value():
    pto = _toml("archive/pto_Haun1987.toml")["electrostriction"]
    Q11, Q12, _Q44 = materials.pzt_haun_x_Q(1.0)
    assert Q11 == pytest.approx(pto["Q11_phys"], abs=1e-4)   # 0.089, already right
    assert Q12 == pytest.approx(pto["Q12_phys"], abs=1e-4)   # -0.026 (was -0.0008)
    assert materials.pzt_haun_x_Q(0.5)[1] == pytest.approx(-0.0460, abs=1e-4)  # was +0.0192
    assert np.all(materials.pzt_haun_x_Q(np.linspace(0.0, 1.0, 101))[1] < 0)


def test_q12_derived_decks_regenerated():
    # archive/pzt_Haun_x0.6.toml is literature_params.pzt_haun_x(x=0.6)'s own output
    pzt = _toml("archive/pzt_Haun_x0.6.toml")["electrostriction"]
    gen = literature_params.pzt_haun_x(x=0.6, T=298.0)
    assert pzt["Q12_phys"] < 0
    assert pzt["Q12_phys"] == gen["Q12_phys"]
    # pbzo_kittel_lum2022.toml borrows PZT x=0.4's Q and its Q12/Q11 ratio for Lam12
    pbzo = _toml("pbzo_kittel_lum2022.toml")
    Q11, Q12, Q44 = materials.pzt_haun_x_Q(0.4)
    assert pbzo["electrostriction"] == {"Q11_phys": Q11, "Q12_phys": Q12, "Q44_phys": Q44}
    lam = pbzo["eigenstrain_q"]
    assert lam["Lam12"] < 0
    assert lam["Lam12"] == pytest.approx(lam["Lam11"] * Q12 / Q11, rel=1e-12)


# --------------------------------------------------------------------------
# A2 -- Bell-Cross 6th-order BaTiO3 alpha111 in kelvin
# --------------------------------------------------------------------------
def test_bc1984_alpha111_uses_kelvin_reference():
    for T, ref in ((298.0, "bto_298K.toml"), (118.0, "bto_118K.toml")):
        a111 = materials_archive.get("bto_bc1984_6th", T=T)["landau"]["a111"]
        assert a111 == pytest.approx(_toml(ref)["landau"]["alpha111"], rel=1e-12)  # 8.004 / 17.94
    deck = materials.from_toml(os.path.join(ARCHIVE, "bto_BC1984_6th.toml"))
    assert deck["landau"] == materials_archive.get("bto_bc1984_6th", T=298.0)["landau"]


def test_bc1984_deck_has_bounded_tetragonal_minimum_at_298K():
    alpha = materials.from_toml(os.path.join(ARCHIVE, "bto_BC1984_6th.toml"))["landau"]
    # bounded below: the sextic form is positive in every direction (no 8th order here)
    rng = np.random.default_rng(0)
    for n in list(landau0d.DIRECTIONS.values()) + list(rng.normal(size=(200, 3))):
        b = landau0d.radial_coefficients(alpha, n)
        assert b[4] == 0.0 and b[3] > 0
    rows = landau0d.symmetry_stationary_points(alpha)
    ground = rows[0]
    assert ground["branch"] == "T" and ground["stable"]
    assert ground["energy_deck"] < 0
    assert 0.1 < ground["magnitude"] < 0.5


# --------------------------------------------------------------------------
# A3 -- the TOML generators skip the optional q/theta sectors
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["bto_298K", "bto_118K"])
def test_deck_parser_regenerates_bto_298K_118K(tmp_path, name):
    committed = open(os.path.join(PARAMS, name + ".toml")).read()
    data = deck_parser.load_params_toml(os.path.join(PARAMS, name + ".toml"))
    m, meta = data["materials"], data["meta"]
    # a minimal MOOSE .i deck carrying exactly the committed key set
    deck = tmp_path / meta["source_deck"]
    deck.parent.mkdir(parents=True)
    deck.write_text("[Materials]\n  [./constants]\n    type = GenericConstantMaterial\n"
                    f"    prop_names = '{' '.join(m)}'\n"
                    f"    prop_values = '{' '.join(repr(v) for v in m.values())}'\n"
                    "  [../]\n[]\n")
    out = tmp_path / (name + ".toml")
    text = deck_parser.write_params_toml(str(deck), str(out), str(tmp_path), meta["extracted"])
    assert "landau_q" not in text and "theta" not in text
    # byte-equal from [landau] on; [meta] lacks only the lines added by hand
    # after generation (registry_name/reader/eps_b_true)
    assert text[text.index("[landau]"):] == committed[committed.index("[landau]"):]
    regen = deck_parser.load_params_toml(str(out))
    assert regen["materials"] == m
    assert {k: regen["meta"][k] for k in ("source_deck", "extracted")} == \
           {k: meta[k] for k in ("source_deck", "extracted")}


def test_literature_write_toml_matches_committed_without_placeholders(tmp_path):
    specs = [("archive/bto_LCC2005_8th.toml", literature_params.bto_lcc2005_8th(T=298.0)),
             ("archive/bto_BC1984_6th.toml", materials_archive.bto_bc1984_6th_legacy(T=298.0)),
             ("archive/pto_Haun1987.toml", literature_params.pto_haun1987(T=298.0)),
             ("archive/pzt_Haun_x0.6.toml", literature_params.pzt_haun_x(x=0.6, T=298.0)),
             ("bto_LCC2005_paper.toml", literature_params.bto_lcc2005_paper(T=298.0))]
    for name, params in specs:
        text = literature_params.write_toml(params, str(tmp_path / os.path.basename(name)))
        committed = open(os.path.join(PARAMS, name)).read()
        assert "aq1 = REQUIRED" not in text and "[eigenstrain_theta]" not in text, name
        assert text[text.index("[landau]"):] == committed[committed.index("[landau]"):], name
    # a genuinely unsourced P-sector field still gets its REQUIRED line
    assert "# G110 = REQUIRED" in open(tmp_path / "pto_Haun1987.toml").read()


# --------------------------------------------------------------------------
# A4 -- composite alpha fields carry the 8th-order keys
# --------------------------------------------------------------------------
def test_composite_alpha_fields_keep_8th_order_terms():
    host = su_landis.deck()["landau"]                 # 8th order: a1112 = 5.74
    filler = composite.si_to_deck_alpha(composite.pe_filler_landau_si(50.0))  # 6 keys only
    assert host["a1112"] > 5.0 and "a1112" not in filler
    L = np.zeros((4, 6))
    L[:, 3:] = 1.0
    labels = L.astype(int)
    P = np.random.default_rng(1).normal(0.0, 0.3, (3, 4, 6))
    for alpha in (composite.alpha_field(L, host, filler),
                  composite.alpha_field_multi(labels, [host, filler])):
        assert set(alpha) == set(solver.ALPHA_KEYS + solver.ALPHA_KEYS_8TH)
        assert np.all(alpha["a1112"][:, :3] == host["a1112"]) and np.all(alpha["a1112"][:, 3:] == 0.0)
        F = solver.bulk_force(P, alpha)
        assert np.allclose(F[:, :, :3], solver.bulk_force(P, host)[:, :, :3], rtol=1e-13, atol=0.0)
        assert np.allclose(F[:, :, 3:], solver.bulk_force(P, filler)[:, :, 3:], rtol=1e-13, atol=0.0)
    # 6th-order-only inputs: same keys and values as before (no zero 8th-order fields)
    six = {k: host[k] for k in solver.ALPHA_KEYS}
    a = composite.alpha_field(L, six, filler)
    assert tuple(a) == solver.ALPHA_KEYS
    assert all(np.array_equal(a[k], six[k] * (1.0 - L) + filler[k] * L) for k in solver.ALPHA_KEYS)
    assert tuple(composite.alpha_field_multi(labels, [six, filler])) == solver.ALPHA_KEYS


# --------------------------------------------------------------------------
# A5 -- polycrystal.make_alpha accepts array-valued coefficients
# --------------------------------------------------------------------------
def test_polycrystal_make_alpha_accepts_array_coefficients():
    core = su_landis.deck()["landau"]
    shell = su_landis.shell_alpha_deck(20.0)
    shape = (8, 8)
    L = np.zeros(shape)
    L[:2] = 1.0
    cth, sth = np.ones(shape), np.zeros(shape)
    Xc = np.linspace(0.5, 1.0, 64).reshape(shape)
    core_x = crystallinity.effective_alpha(core, Xc, 0.8, shape=shape)
    assert np.ndim(core_x["a1"]) == 2
    out = polycrystal.make_alpha(core_x, shell, L, cth, sth)
    assert np.array_equal(out["a1"], core_x["a1"] * (1.0 - L) + shell["a1"] * L)
    ref = polycrystal.make_alpha(core, shell, L, cth, sth)       # all-scalar path unchanged
    for k in polycrystal.ALPHA_KEYS:
        if k != "a1":
            assert np.array_equal(out[k], ref[k]) and type(out[k]) is type(ref[k])
    P = np.random.default_rng(2).normal(0.0, 0.2, (3,) + shape)
    assert np.all(np.isfinite(polycrystal.bulk_force(P, out)))
    with pytest.raises(ValueError, match="broadcast"):
        polycrystal.make_alpha(dict(core, a1=np.ones(5)), shell, L, cth, sth)


# --------------------------------------------------------------------------
# A6 -- loopfit global-minimum constraint looks beyond P2
# --------------------------------------------------------------------------
def test_global_min_constraint_rejects_fe_well_beyond_P2(capsys):
    a1, a11, a111 = 0.06, -0.4932, 1.0
    P2 = loopfit.spinodal_points(a1, a11, a111)[1]
    assert loopfit.f_energy(P2, a1, a11, a111) > 0           # the old f(P2) check passes...
    assert loopfit.thresholds(a1, a11, a111)[1] < 0           # ...because E_A < 0
    P = np.linspace(0.0, 1.0, 20001)
    assert loopfit.f_energy(P, a1, a11, a111).min() < -1.9e-4  # FE well at P=0.498 is lower
    with pytest.raises(loopfit.TripleWellConstraintError, match="global minimum"):
        loopfit.check_global_min_constraint(a1, a11, a111)
    capsys.readouterr()
    # the shipped Yang triple-well deck still passes, same return and same printed line
    y = _toml("pbzo_triplewell_yang2025.toml")["landau"]
    P2y, f2y = loopfit.check_global_min_constraint(y["alpha1"], y["alpha11"], y["alpha111"],
                                                   label="seed0")
    assert capsys.readouterr().out == (
        "[loopfit] global-min constraint (seed0): P2=sqrt(u2)=0.359307737  "
        "f(0)=0.000000e+00  f(P2)=1.704717e-02  PASS\n")
    assert (P2y, f2y) == pytest.approx((0.359307737, 1.7047172e-2), rel=1e-7)


def test_composite_alpha_field_multi_rejects_labels_without_deck():
    """A label >= K (no deck) used to get alpha = 0 at every key -- a silent
    zero-Landau cell. Now refused; in-range labels are unchanged."""
    from solvers import composite
    decks = [{k: float(i + 1) for k in composite.ALPHA_KEYS} for i in range(2)]
    ok = composite.alpha_field_multi(np.array([[0, 1], [1, 0]]), decks)
    np.testing.assert_array_equal(ok[composite.ALPHA_KEYS[0]], [[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="no deck"):
        composite.alpha_field_multi(np.array([[0, 1], [2, 0]]), decks)
