"""SPEC S36 (WP25 Phase 2, stage 2A) -- the unified material-coefficient
registry. Answers ONE question: "what Landau/gradient/elastic/electrostriction
coefficients does THIS material have?" -- nothing about disorder mechanisms
(quenched_fields.py, stage 2B) or spatial composition (composition.py, stage
2B) lives here; this module is pure "materials.get(name, T=...) -> deck".

ABSORBS (logic migrated, not copy-pasted -- the old modules become thin
parameter-binding shims that call the functions below, SPEC S36.2 R4):
  - solvers/literature_params.py: bto_lcc2005_8th/_paper, bto_bc1984_6th,
    pto_haun1987, pzt_haun_x (+ its _pzt_T0/_pzt_C0/pzt_haun_x_landau_si/
    pzt_haun_x_Q/pzt_haun_x_elastic_si internals).
  - solvers/pan2019_field.py: ALL of it (bfo/bto/sto end members -- "a
    paper-specific module" stops being a concept; BFO/BTO/STO are three
    ordinary registry entries here, SPEC S36.3).
  - solvers/pmn_field.py: ONLY pmn_clean_alpha1 and pmn_pt_landau_placeholder
    (the two MATERIAL potentials). Everything else in that module -- the
    disorder-MECHANISM layer (derive_ed_stats_*/defect_field_*/
    bond_field_cor_alpha1 and its T0_PMN_CW_K/C_CW_PMN_K constants) -- is
    UNTOUCHED, staying put for stage 2B (quenched_fields.py).
  - solvers/pzt_field.py: ONLY landau_field/Q_field/elastic_C_field/
    elastic_C_at (the x-dependent Haun formulas). Everything else --
    composition_field/composition_std*/vegard_eigenstrain_field/
    defect_field/formula_units_per_cell -- is UNTOUCHED, staying put for
    stage 2B (composition.py).

NEW SURFACE (SPEC S36.2/S36.3):
  materials.get(name, **kw) -> deck        engine-native a1..a123 keys
  materials.list() -> [(name, provenance)] one line of provenance per entry
  materials.from_toml(path) -> deck        same schema, read from params/*.toml

ARCHIVED (2026-11, user decision): the entries no computation called --
bto_bc1984_6th, the Pan 2019 deck wrappers bfo_pan2019/bto_pan2019/
sto_pan2019, pmn_clean_x0 and pmn_pt_placeholder -- moved with their builders
to solvers/materials_archive.py (ARCHIVED_REGISTRY, same get/list surface);
their TOML snapshots are in params/archive/. materials.get(<archived name>)
raises a KeyError that points there (_ARCHIVED below). This module never
imports the archive. The ABSORBS list above is the 2A migration record and
still names the archived functions.

DECK SCHEMA (SPEC S36.2 R1, binding): a plain dict, NOT a class --
  {"landau": {a1..a123 present},           engine-native (solver.ALPHA_KEYS);
              8th-order a1111..a1123 keys present (possibly 0.0) iff the
              source potential's own 6th-vs-8th-order status set them, same
              convention the legacy alpha1111.. zero-fills already used --
              genuinely UNSOURCED coefficients (e.g. PMN's a11..) are simply
              ABSENT, never zero-filled,
   "gradient": {"G11","G12","G44","G44p"},  ABSOLUTE values (ratio*G110
              already multiplied through), so **deck["gradient"] feeds
              solver.build_operators directly,
   "elastic": {"C11","C12","C44"},
   "electrostriction": {"Q11_phys","Q12_phys","Q44_phys"},
   "kappa_b": float,
   "meta": {...provenance...}}
A sector a material LACKS is ABSENT from the dict entirely (KeyError beats a
silent wrong zero) -- never present with None/0 filler. Reference quantities
that do not survive into the new schema (G110 itself, the legacy G11_G110-
style ratios) are kept in "meta" for provenance/debugging (SPEC S36 coordinator
note: "G110 等参照量放 meta 备查").

R2 (engine-native keys everywhere): "landau" uses a1..a123 (= solver.ALPHA_KEYS
+ the optional 8th-order a1111..a1123), never the legacy alpha1..alpha123
naming -- killing the repo's most common wiring bug (ENGINE_GUIDE FAQ #1).
Legacy modules keep emitting alpha1..-style keys through their shims.

R5 (dependency direction): this module imports ONLY solvers.units,
solvers.deck_parser and solvers.validate (setup-time deck checks, itself
importing nothing engine-side), all via plain package-internal imports
(`from solvers import X`) -- never anything from cases/. R5 (solvers/ never
imports cases/, SPEC S35.4/S36.2) holds tree-wide since WP25 Phase 2C
cleared the last four violations; it is enforced by
gates/g29f_dependency_direction.py and gates/g28a_import_hygiene.py, not by
convention.

Every entry below is table-registered in _REGISTRY at the bottom of this
file; gates/g29a_materials_equivalence.py and gates/g29b_deck_schema_dualpath.py
iterate materials.list() (not a hand-copied name list) so a future registry
addition cannot silently escape equivalence/schema coverage.
"""
import os
from fftpf._locations import PARAMS_DIR as _DECK_DIR
from functools import partial
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from fftpf import units
from fftpf import deck_reader as deck_parser
from fftpf import input_validation as validate

PARAMS_DIR = str(_DECK_DIR)

# ==========================================================================
# Shared plumbing: legacy-flat-dict <-> new-schema-deck conversion.
#
# "flat" here means a dict using the OLD literature_params.py convention:
# alpha1..alpha1123 (Landau), G110 + G11_G110/G12_G110/G44_G110/G44P_G110
# (gradient ratios), permittivity (kappa_b), C11/C12/C44 (elastic),
# Q11_phys/Q12_phys/Q44_phys (electrostriction) -- any key ABSENT or set to
# None means "not sourced". Every material builder below produces (flat,
# meta) as its OWN computation; materials.get() runs that through
# _flat_to_deck for the new schema, and each legacy shim (literature_params.
# py/pan2019_field.py/pmn_field.py/pzt_field.py) consumes `flat`/`meta`
# DIRECTLY (bypassing _flat_to_deck) to reconstruct the exact old dict shape
# -- so both the new deck and the legacy dict trace back to the SAME
# underlying numbers, not two independent computations that happen to agree.
# ==========================================================================
_ALPHA_TO_ENGINE = {
    "alpha1": "a1", "alpha11": "a11", "alpha12": "a12",
    "alpha111": "a111", "alpha112": "a112", "alpha123": "a123",
    "alpha1111": "a1111", "alpha1112": "a1112", "alpha1122": "a1122", "alpha1123": "a1123",
}
_GRADIENT_RATIO_KEYS = ("G11_G110", "G12_G110", "G44_G110", "G44P_G110")
_GRADIENT_RATIO_TO_ABS = {"G11_G110": "G11", "G12_G110": "G12", "G44_G110": "G44", "G44P_G110": "G44p"}


def _flat_to_deck(flat, meta):
    """flat: legacy-shaped dict (see module docstring); values may be None
    (not sourced -> absent) or scalar/ndarray (sourced -> kept, array-aware
    for x(r)-dependent materials like pzt_haun_x). meta: provenance dict,
    copied (not mutated) and augmented with G110/the four gradient ratios
    when present (SPEC S36 coordinator note: reference quantities that don't
    survive into the absolute-value gradient schema go to meta)."""
    deck = {}

    landau = {ek: flat[ak] for ak, ek in _ALPHA_TO_ENGINE.items() if flat.get(ak) is not None}
    if landau:
        deck["landau"] = landau

    meta = dict(meta)
    G110 = flat.get("G110")
    if G110 is not None:
        meta["G110"] = G110
    ratios = {}
    for rk in _GRADIENT_RATIO_KEYS:
        v = flat.get(rk)
        if v is not None:
            meta[rk] = v
            ratios[rk] = v
    if G110 is not None and len(ratios) == len(_GRADIENT_RATIO_KEYS):
        deck["gradient"] = {_GRADIENT_RATIO_TO_ABS[rk]: ratios[rk] * G110 for rk in _GRADIENT_RATIO_KEYS}

    C11, C12, C44 = flat.get("C11"), flat.get("C12"), flat.get("C44")
    if C11 is not None and C12 is not None and C44 is not None:
        deck["elastic"] = {"C11": C11, "C12": C12, "C44": C44}

    Q11, Q12, Q44 = flat.get("Q11_phys"), flat.get("Q12_phys"), flat.get("Q44_phys")
    if Q11 is not None and Q12 is not None and Q44 is not None:
        deck["electrostriction"] = {"Q11_phys": Q11, "Q12_phys": Q12, "Q44_phys": Q44}

    kb = flat.get("permittivity")
    if kb is not None:
        deck["kappa_b"] = kb

    deck["meta"] = meta
    return deck


# ==========================================================================
# (a) bto_lcc2005_8th -- literature_params.py's PRODUCTION STANDARD.
# Verbatim port of the SI formulas (Rabe/Ahn/Triscone 2007 App A (Chen),
# Table 1 "Eight-order" column); mechanics/gradient/permittivity kept from
# bto_298K.toml (HM2006), per SPEC WP4 9.1(a).
# ==========================================================================
_BTO_298_TOML = os.path.join(PARAMS_DIR, "bto_298K.toml")


def _bto_298_mechanics():
    m = deck_parser.load_params_toml(_BTO_298_TOML)["materials"]
    return dict(C11=m["C11"], C12=m["C12"], C44=m["C44"],
                G110=m["G110"], G11_G110=m["G11_G110"], G12_G110=m["G12_G110"],
                G44_G110=m["G44_G110"], G44P_G110=m["G44P_G110"],
                permittivity=m["permittivity"],
                Q11_phys=m["Q11_phys"], Q12_phys=m["Q12_phys"], Q44_phys=m["Q44_phys"])


def _bto_lcc2005_8th_flat(T=298.0):
    si = dict(
        alpha1=4.124e5 * (T - 388),
        alpha11=-209.7e6,
        alpha12=7.974e8,
        alpha111=129.4e7,
        alpha112=-1.950e9,
        alpha123=-2.5009e9,
        alpha1111=3.863e10,
        alpha1112=2.529e10,
        alpha1122=1.637e10,
        alpha1123=1.367e10,
    )
    flat = {k: units.si_to_deck_landau(v) for k, v in si.items()}
    flat.update(_bto_298_mechanics())
    meta = dict(source="Rabe/Ahn/Triscone 2007 App A (Chen), Table 1 'Eight-order' column, "
                        "PDF p.376/book p.365", T=T,
                note="mechanics/gradient/permittivity kept from bto_298K.toml (HM2006), "
                     "per SPEC WP4 9.1(a); wall energies under this set are NOT literature-"
                     "anchored -- only monodomain gates (G6b) apply")
    return flat, meta


def bto_lcc2005_8th(T=298.0):
    """BaTiO3 8th-order (Rabe/Ahn/Triscone 2007 App A, Table 1 'Eight-order'; mechanics from bto_298K.toml) -> R1 deck."""
    flat, meta = _bto_lcc2005_8th_flat(T)
    return _flat_to_deck(flat, meta)


# --------------------------------------------------------------------------
# (a') bto_lcc2005_paper -- SAME LCC2005 8th-order alphas, mechanics replaced
# by Li & Chen APL 88, 072905 (2006)'s own Ref. 18 footnote; gradient backed
# out via dx_nm through the paper's own dx=sqrt(G110/alpha0) formula. See
# literature_params.py's (now-shim) docstring for the full provenance essay
# -- unchanged, just relocated.
# --------------------------------------------------------------------------
def _bto_lcc2005_paper_flat(T=298.0, dx_nm=0.4):
    base_flat, _base_meta = _bto_lcc2005_8th_flat(T=T)
    landau = {k: base_flat[k] for k in
              ("alpha1", "alpha11", "alpha12", "alpha111", "alpha112", "alpha123",
               "alpha1111", "alpha1112", "alpha1122", "alpha1123")}
    alpha0 = abs(_bto_lcc2005_8th_flat(T=298.0)[0]["alpha1"])  # deck units, T=298K == paper's 25C
    G110 = dx_nm**2 * alpha0

    flat = dict(landau)
    flat.update(dict(
        C11=units.si_to_deck_elastic(1.78e11), C12=units.si_to_deck_elastic(0.964e11),
        C44=units.si_to_deck_elastic(1.22e11),
        Q11_phys=0.10, Q12_phys=-0.034, Q44_phys=0.029,
        G110=G110, G11_G110=1.0, G12_G110=0.0, G44_G110=0.5, G44P_G110=0.5,
        permittivity=_bto_298_mechanics()["permittivity"],
    ))
    meta = dict(
        source="Li & Chen, Appl. Phys. Lett. 88, 072905 (2006), Ref. 18 footnote (mechanics) "
               "+ this paper's own Eq 3 (landau, == bto_lcc2005_8th) + Eq 7 (isotropic gradient)",
        T=T, dx_nm=dx_nm,
        note="C11/C12/C44/Q11/Q12/Q44 are the paper's OWN footnote-18 numbers (not Rabe App A -- "
             "no 'which Q column' ambiguity, see WP6_REPORT.md). G110 is NOT given by the paper "
             "(only G11/G110=1.0 and dx=sqrt(G110/alpha0)); chosen here via dx=0.4nm (documented "
             "free parameter, NOT tuned to any result) and backed out through the paper's own "
             "formula. G-ratios (0,0.5,0.5) are the unique isotropic case of SPEC's general "
             "gradient form, not a free choice. permittivity has no basis in the paper at all; "
             "inherited from bto_298K.toml (HM2006) as a documented placeholder.")
    return flat, meta


def bto_lcc2005_paper(T=298.0, dx_nm=0.4):
    """BaTiO3 with Li & Chen APL 88, 072905 (2006) Ref.18 mechanics and that paper's own gradient -> R1 deck."""
    flat, meta = _bto_lcc2005_paper_flat(T, dx_nm)
    return _flat_to_deck(flat, meta)


# --------------------------------------------------------------------------
# (c) pto_haun1987 -- PbTiO3, Table 5. G110/permittivity unsourced (App A
# gives none for PTO); G-ratios are BTO's own values as a placeholder start.
# --------------------------------------------------------------------------
def _pto_haun1987_flat(T=298.0):
    si = dict(
        alpha1=3.8e5 * (T - 752),
        alpha11=-0.73e8,
        alpha12=7.5e8,
        alpha111=2.6e8,
        alpha112=6.1e8,
        alpha123=-37e8,
    )
    flat = {k: units.si_to_deck_landau(v) for k, v in si.items()}
    for k in ("alpha1111", "alpha1112", "alpha1122", "alpha1123"):
        flat[k] = 0.0

    s11, s12, s44 = 8.0e-12, -2.5e-12, 9.0e-12  # m^2/N, PDF Table 5
    c11_si, c12_si, c44_si = units.cubic_compliance_to_stiffness(s11, s12, s44)

    flat.update(dict(
        C11=units.si_to_deck_elastic(c11_si), C12=units.si_to_deck_elastic(c12_si),
        C44=units.si_to_deck_elastic(c44_si),
        Q11_phys=0.089, Q12_phys=-0.026, Q44_phys=0.0675,
        G110=None, G11_G110=0.51, G12_G110=-0.02, G44_G110=0.02, G44P_G110=0.0,
        permittivity=None,
    ))
    meta = dict(source="Rabe/Ahn/Triscone 2007 App A (Chen), Table 5, PDF p.378/book p.368", T=T,
                note="G110/permittivity are None (PLACEHOLDER REQUIRED, not in App A); "
                     "G-ratios given are BTO's own values as a starting-point assumption, "
                     "not PTO-validated. C11/C12/C44 from cubic compliance inversion "
                     "(units.cubic_compliance_to_stiffness) of Table 5's s11/s12/s44.")
    return flat, meta


def pto_haun1987(T=298.0):
    """PbTiO3 (Rabe App A, Table 5) -> R1 deck; gradient and permittivity unsourced."""
    flat, meta = _pto_haun1987_flat(T)
    return _flat_to_deck(flat, meta)


# --------------------------------------------------------------------------
# (d) pzt_haun_x -- x-dependent Haun formula set, PDF p.367 + Table 4.
# Array-aware throughout (x may be a scalar or an (Ny,Nx) array) -- this is
# load-bearing: pzt_field.py's landau_field/Q_field/elastic_C_field shim
# straight through to these same functions, and stage 2B's
# composition.deck_field depends on that array-awareness surviving.
# --------------------------------------------------------------------------
_PZT_TABLE4_X = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
_PZT_TABLE4_S11 = np.array([8.8, 10.5, 8.6, 8.4, 8.2, 8.1]) * 1e-12   # m^2/N
_PZT_TABLE4_S12 = np.array([-2.9, -3.7, -2.8, -2.7, -2.6, -2.5]) * 1e-12
_PZT_TABLE4_S44 = np.array([24.6, 28.7, 21.2, 17.5, 14.4, 12.0]) * 1e-12


def _pzt_T0(x):
    return 462.63 + 843.4 * x - 2105.5 * x**2 + 4041.8 * x**3 - 3828.3 * x**4 + 1337.8 * x**5


def _pzt_C0(x):
    """Vectorized over arrays (np.where, not scalar if/elif) so
    pzt_haun_x_landau_si can evaluate a whole composition field x(r)
    pointwise. Scalar callers get a plain float back; array callers get an
    ndarray, same shape as x. Boundary x=0.5 on the low branch (<=)."""
    x = np.asarray(x, dtype=float)
    if np.any(x < 0.0) or np.any(x > 1.0):
        raise ValueError(f"pzt_haun_x: x outside App A's fitted range [0,1] "
                          f"(min={x.min()!r}, max={x.max()!r})")
    lo = (2.1716 / (1 + 500.05 * (x - 0.5)**2) + 0.131 * x + 2.01) * 1e5
    hi = (2.8339 / (1 + 126.56 * (x - 0.5)**2) + 1.4132) * 1e5
    out = np.where(x <= 0.5, lo, hi)
    return out if out.ndim > 0 else float(out)


def pzt_haun_x_landau_si(x, T):
    """The p.367 formula set, in SI units. Returns
    (a1,a11,a12,a111,a112,a123)."""
    eps0 = 8.85e-12
    C0 = _pzt_C0(x)
    T0 = _pzt_T0(x)
    a1 = (T - T0) / (2 * eps0 * C0)
    a11 = (10.612 - 22.655 * x + 10.955 * x**2) * 1e13 / C0
    a111 = (12.026 - 17.296 * x + 9.179 * x**2) * 1e13 / C0
    a112 = (4.2904 - 3.3754 * x + 58.804 * np.exp(-29.397 * x)) * 1e14 / C0
    eta1 = (2.6213 + 0.42743 * x - (9.6 + 0.012501 * x) * np.exp(-12.6 * x)) * 1e14 / C0
    eta2 = (0.887 - 0.76973 * x + (16.225 - 0.088651 * x) * np.exp(-21.255 * x)) * 1e15 / C0
    a12 = eta1 / 3 - a11
    a123 = eta2 - 3 * a111 - 6 * a112
    return a1, a11, a12, a111, a112, a123


def pzt_haun_x_Q(x):
    """Q11,Q12,Q44(x), unscaled SI==deck. Q44 carries the source's own
    explicit 1/2 prefactor -- NOT symmetrized with Q11/Q12.

    Q12 is NEGATIVE across the whole range: its rational and linear terms
    both carry a minus sign. An earlier transcription had them as plus,
    which made Q12 positive near the morphotropic boundary (+0.0192 at
    x=0.5, +0.0027 at x=0.6). Two checks fix the sign. At x=1 the set must
    reproduce the PbTiO3 end member of the same source (pto_haun1987 /
    Table 5: Q11 0.089, Q12 -0.026). Q11 does, and Q12 gives -0.0260 only
    with these signs (the old ones gave -0.0008). At x=0.5 it gives
    -0.0460, the standard PZT 50/50 value."""
    Q11 = 0.029578 / (1 + 200 * (x - 0.5)**2) + 0.042796 * x + 0.045624
    Q12 = -0.026568 / (1 + 200 * (x - 0.5)**2) - 0.012093 * x - 0.013386
    Q44 = 0.5 * (0.025325 / (1 + 200 * (x - 0.5)**2) + 0.020857 * x + 0.046147)
    return Q11, Q12, Q44


def pzt_haun_x_elastic_si(x, mechanics=True):
    """C11,C12,C44 (SI Pa) via linear interpolation of Table 4's compliances
    over x in [0.4,0.9], then cubic compliance inversion. mechanics=False/
    None skips the range check and returns None. x may be scalar or an
    (Ny,Nx) array."""
    if mechanics is False or mechanics is None:
        return None
    xarr = np.asarray(x, dtype=float)
    if np.any(xarr < _PZT_TABLE4_X[0]) or np.any(xarr > _PZT_TABLE4_X[-1]):
        raise ValueError(f"pzt_haun_x_elastic_si: x outside Table 4's fitted range "
                          f"[{_PZT_TABLE4_X[0]},{_PZT_TABLE4_X[-1]}] "
                          f"(min={xarr.min()!r}, max={xarr.max()!r}; pass mechanics=None to skip)")
    s11 = np.interp(xarr, _PZT_TABLE4_X, _PZT_TABLE4_S11)
    s12 = np.interp(xarr, _PZT_TABLE4_X, _PZT_TABLE4_S12)
    s44 = np.interp(xarr, _PZT_TABLE4_X, _PZT_TABLE4_S44)
    c11, c12, c44 = units.cubic_compliance_to_stiffness(s11, s12, s44)
    if xarr.ndim == 0:
        return float(c11), float(c12), float(c44)
    return c11, c12, c44


def _pzt_haun_x_flat(x=0.6, T=298.0, mechanics=True):
    a1, a11, a12, a111, a112, a123 = pzt_haun_x_landau_si(x, T)
    landau_si = dict(alpha1=a1, alpha11=a11, alpha12=a12, alpha111=a111, alpha112=a112, alpha123=a123)
    flat = {k: units.si_to_deck_landau(v) for k, v in landau_si.items()}
    for k in ("alpha1111", "alpha1112", "alpha1122", "alpha1123"):
        flat[k] = 0.0

    Q11, Q12, Q44 = pzt_haun_x_Q(x)
    flat.update(dict(Q11_phys=Q11, Q12_phys=Q12, Q44_phys=Q44,
                      G110=None, G11_G110=0.51, G12_G110=-0.02, G44_G110=0.02, G44P_G110=0.0,
                      permittivity=None))
    c = pzt_haun_x_elastic_si(x, mechanics=mechanics)
    if c is not None:
        c11_si, c12_si, c44_si = c
        flat.update(dict(C11=units.si_to_deck_elastic(c11_si), C12=units.si_to_deck_elastic(c12_si),
                          C44=units.si_to_deck_elastic(c44_si)))
    else:
        flat.update(dict(C11=None, C12=None, C44=None))
    meta = dict(source="Rabe/Ahn/Triscone 2007 App A (Chen), p.367/PDF p.378 formula set + "
                        "Table 4 compliances, book p.368", x=x, T=T,
                note="Q44 carries the source's own explicit 1/2 prefactor (kept, not "
                     "symmetrized with Q11/Q12); G110/permittivity None (PLACEHOLDER "
                     "REQUIRED, not given in App A; G-ratios are BTO's own as a starting "
                     "point, not PZT-validated)")
    return flat, meta


def pzt_haun_x(x=0.6, T=298.0, mechanics=True):
    """Pb(Zr1-xTix)O3 Haun x-dependent set (Rabe App A p.367 + Table 4) -> R1 deck; x scalar or array."""
    flat, meta = _pzt_haun_x_flat(x, T, mechanics)
    return _flat_to_deck(flat, meta)


# ==========================================================================
# Pan et al., Science 365, 578 (2019) BFO/BTO/STO end members. Verbatim port
# of pan2019_field.py's formulas -- see that module's (now-shim) docstring
# for the full B2-D forensics essay (BTO alpha111 Pan-vs-Sluka constant,
# STO alpha1's 36.5-vs-35.5 discrepancy, BFO's untraceable citation) --
# unchanged, just relocated. ARCHIVED 2026-11: the R1-deck wrappers
# bfo_pan2019/bto_pan2019/sto_pan2019 (and _pan2019_common_flat, the
# _*_pan2019_flat builders, the _PAN2019_SOURCE citation) moved to
# solvers/materials_archive.py -- no computation called them. What stays
# here is what pan2019_field.py's live shim calls.
# ==========================================================================
PAN2019_T_REF_K = 298.0  # declared room-T reference (W0/WP16/WP20 convention)

PAN2019_G110_SI = 7.04e-11       # C^-2 m^4 N, as printed (SOURCED, dossier iii)
PAN2019_G11_G110 = 1.2
PAN2019_G12_G110 = 0.0
PAN2019_G44_G110 = 0.6
PAN2019_G44P_G110 = 0.6           # dossier: "G44=G44'" -- their "isotropic" means only this equality
PAN2019_G110_DECK = units.si_to_deck_gradient(PAN2019_G110_SI)

# Background permittivity: UNSOURCED for this paper; repo-standing 10*eps0
# fallback (Wang, Ma, Li, Britson & Chen, Acta Mater 61 (2013) 7591 lineage).
# SPEC V2-02.1 (Lens 2 U3, ruling D7) -- LABEL CORRECTED, VALUES FROZEN: the base
# 0.08854187 is itself eps_b = 10 under the derived mapping kappa_deck =
# 0.0088541878128*eps_b (solvers/units.py), so PAN2019_KAPPA_B_DECK = 0.8854187
# is eps_b = 100, not the Wang-2013 eps_b = 10 this fallback names. The B2A/B2C/
# B2E and W-lane numbers produced with it stand, labeled eps_b = 100.
PAN2019_EPS0_DECK_BASE = 0.08854187
PAN2019_KAPPA_B_MULT = 10.0
PAN2019_KAPPA_B_DECK = PAN2019_KAPPA_B_MULT * PAN2019_EPS0_DECK_BASE

def _pan2019_bfo_alpha1_si(T):
    return 4.9e5 * (T - 1103.0)


def _pan2019_bfo_landau_si(T):
    return dict(alpha1=_pan2019_bfo_alpha1_si(T), alpha11=6.5e8, alpha12=1.0e8,
                alpha111=0.0, alpha112=0.0, alpha123=0.0)


def _pan2019_bto_alpha1_si(T):
    return 3.34e5 * (T - 381.0)


def _pan2019_bto_alpha11_si(T):
    return 4.69e6 * (T - 393.0) - 2.02e8


def _pan2019_bto_alpha111_si(T):
    return -5.52e7 * (T - 393.0) + 2.76e8


def _pan2019_bto_landau_si(T):
    return dict(alpha1=_pan2019_bto_alpha1_si(T), alpha11=_pan2019_bto_alpha11_si(T), alpha12=3.23e8,
                alpha111=_pan2019_bto_alpha111_si(T), alpha112=4.47e9, alpha123=4.91e9)


def _pan2019_bto_alpha111_si_sluka_literal(T):
    """ALTERNATE, not used by the production _pan2019_bto_landau_si path
    (coordinator ruling, reports/LIT_pan2019_forensics.md): Sluka et al.,
    Nat Commun 3, 748 (2012) Table 1's own printed constant is 2.76e9, not
    Pan's printed 2.76e8. Exists solely for gates/g26c_endmember_0d.py's
    zero-compute sensitivity comparison."""
    return -5.52e7 * (T - 393.0) + 2.76e9


def _pan2019_sto_alpha1_si(T):
    return 7.06e5 * (T - 36.5)


def _pan2019_sto_landau_si(T):
    return dict(alpha1=_pan2019_sto_alpha1_si(T), alpha11=1.7e9, alpha12=1.37e9,
                alpha111=0.0, alpha112=0.0, alpha123=0.0)


def _si_to_deck_landau_dict(d_si):
    """{alpha_name: SI value} -> {alpha_name: deck value}, uniform x1e-9."""
    return {k: units.si_to_deck_landau(v) for k, v in d_si.items()}


# ==========================================================================
# PMN material potentials (SPEC S21.3). ONLY pmn_clean_alpha1 -- the
# pmn_clean_x0 deck wrapper and pmn_pt_placeholder were ARCHIVED 2026-11 to
# solvers/materials_archive.py (no computation called them). pmn_field.py's
# disorder-mechanism layer (bond_field_cor_alpha1, derive_ed_stats_*,
# defect_field_*, and their T0_PMN_CW_K/C_CW_PMN_K/EPS0_SI constants) is
# UNTOUCHED, stage 2B's job. _EPS0_SI_CODATA below is a DELIBERATE, isolated
# duplicate of pmn_field.EPS0_SI's numeric literal (8.8541878128e-12,
# CODATA) -- kept local rather than imported so that materials.py stays a
# strict dependency LEAF under pmn_field.py (pmn_field's shim imports
# materials, so the reverse import would be circular); both copies must
# stay numerically identical (a CODATA constant, not expected to ever
# change) if this module or pmn_field.py is ever edited again.
# ==========================================================================
_EPS0_SI_CODATA = 8.8541878128e-12
_PMN_T0_CW_K_DEFAULT = 398.0   # K, Viehland, Jang, Cross & Wuttig, PRB 46, 8003 (1992)
_PMN_C_CW_K_DEFAULT = 1.2e5    # K, same source


def pmn_clean_alpha1(T, T0_K=_PMN_T0_CW_K_DEFAULT, C_cw_K=_PMN_C_CW_K_DEFAULT):
    """Pure-PMN clean-limit alpha1(T), DECK units -- SPEC S21.3 ruling 2.
    alpha1 = (T-T0)/(2*eps0*C_CW) (factor-of-2 note: this project's own
    already-validated Curie-Weiss-to-alpha1 convention, literature_params.
    pzt_haun_x_landau_si's a1=(T-T0)/(2*eps0*C0), is used here too)."""
    return units.si_to_deck_landau((T - T0_K) / (2.0 * _EPS0_SI_CODATA * C_cw_K))


def _substrate_reference(filename, T=298.15):
    """Elastic-only room-temperature reference; no extrapolation to hot runs.

    This engineering usage window is an adoption policy, not a measured
    validity range. Crystal axes and data limitations are retained in meta.
    """
    if not 293.15 <= T <= 303.15:
        raise ValueError("Substrate reference is room-temperature only (293.15-303.15 K)")
    return from_toml(os.path.join(PARAMS_DIR, "substrates", filename))


# ==========================================================================
# Registry + public surface (SPEC S36.2/S36.3).
# ==========================================================================
_REGISTRY = {
    "ss316_rt": (partial(_substrate_reference, "ss316_rt.toml"),
        "AISI 316 stainless steel: room-temperature elastic-only reference"),
    "ss304_rt": (partial(_substrate_reference, "ss304_rt.toml"),
        "AISI 304 stainless steel: room-temperature elastic-only reference"),
    "silicon_cubic_rt": (partial(_substrate_reference, "silicon_cubic_rt.toml"),
        "Single-crystal silicon: room-temperature elastic-only reference"),
    "fused_silica_hpfs_rt": (partial(_substrate_reference, "fused_silica_hpfs_rt.toml"),
        "Corning HPFS amorphous fused silica (SiO2): room-temperature elastic-only reference"),
    "mgo_cubic_rt": (partial(_substrate_reference, "mgo_cubic_rt.toml"),
        "Single-crystal magnesium oxide (MgO): room-temperature elastic-only reference"),
    "bto_lcc2005_8th": (bto_lcc2005_8th,
                         "BaTiO3 8th-order (Rabe/Ahn/Triscone 2007 App A, Table 1 Eight-order); "
                         "mechanics from bto_298K.toml (HM2006) -- production standard"),
    "bto_lcc2005_paper": (bto_lcc2005_paper,
                           "BaTiO3 (Li & Chen APL 88, 072905 (2006) Ref.18 footnote mechanics + "
                           "this paper's own Eq.3/Eq.7 gradient) -- Tier1-3 film reproduction"),
    "pto_haun1987": (pto_haun1987,
                      "PbTiO3 (Rabe/Ahn/Triscone 2007 App A, Table 5); gradient/permittivity "
                      "unsourced (PZT x->1 end member)"),
    "pzt_haun_x": (pzt_haun_x,
                   "Pb(Zr1-xTix)O3, x-dependent Haun formula set (Rabe/Ahn/Triscone 2007 App A "
                   "p.367 + Table 4); gradient/permittivity unsourced; x scalar or array"),
}

# Entries archived 2026-11 (no computation called them): their generators
# and registry rows live in solvers/materials_archive.py (ARCHIVED_REGISTRY).
# Kept here as plain names only -- this module never imports the archive --
# so get() can say where an archived name went.
_ARCHIVED = ("bfo_pan2019", "bto_bc1984_6th", "bto_pan2019", "pmn_clean_x0",
             "pmn_pt_placeholder", "sto_pan2019")


def get(name, **kw):
    """materials.get(name, T=..., **kw) -> deck (SPEC S36.2 R1 schema,
    engine-native a1..a123 keys). Raises KeyError with the available name
    list if `name` is not registered, or a pointer to
    solvers.materials_archive if `name` is an archived entry."""
    if name not in _REGISTRY:
        if name in _ARCHIVED:
            raise KeyError(f"materials.get: {name!r} is archived (no computation uses it); "
                            f"use solvers.materials_archive.get({name!r})")
        raise KeyError(f"materials.get: unknown material {name!r}; "
                        f"available: {sorted(_REGISTRY)}")
    fn, _provenance = _REGISTRY[name]
    return validate.deck(fn(**kw), where=f"materials.get({name!r})")


def list():
    """[(name, one-line provenance), ...], sorted by name."""
    return [(name, provenance) for name, (_fn, provenance) in sorted(_REGISTRY.items())]


def from_toml(path):
    """Read a params/*.toml deck (deck_parser's flat TOML_SECTIONS schema)
    and return it in the SAME R1 deck schema materials.get() produces --
    the "second reading path" ENGINE_GUIDE S2.3 flags, unified here (SPEC
    S36.3's explicit target: "TOML-backed decks and formula-generated decks
    currently have two different reading paths -- get() hides that")."""
    validate.toml_deck(path)
    data = deck_parser.load_params_toml(path)
    flat = dict(data["materials"])
    meta = dict(data["meta"])
    meta.setdefault("source_toml", os.path.relpath(path, _ROOT))
    return validate.deck(_flat_to_deck(flat, meta), where=f"materials.from_toml({path!r})")
