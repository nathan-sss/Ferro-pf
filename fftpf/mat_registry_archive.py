"""Archived material-registry entries: generators that no computation calls.

User decision, 2026-11: the engine's built-in material parameters are of
little value, so the registry entries that no case, run, solver code path or
test called were ARCHIVED here, not deleted. NOTHING IN THE ENGINE IMPORTS
THIS MODULE -- solvers/materials.py does not, no other solvers/ module does,
and no cases/ or Runs/ script does. It is not in solvers.__all__; reach it
explicitly:

    from fftpf import mat_registry_archive as materials_archive
    deck = materials_archive.get("bto_bc1984_6th", T=298.0)
    materials_archive.list()

materials.get(<archived name>) raises a KeyError that points here.

WHAT MOVED (code verbatim, numbers unchanged; each function returns exactly
what it returned from its old home):
  - from solvers/materials.py: bto_bc1984_6th + _bto_bc1984_6th_flat;
    the Pan 2019 R1-deck wrappers bfo_pan2019/bto_pan2019/sto_pan2019 +
    their _*_pan2019_flat builders, _pan2019_common_flat and the
    _PAN2019_SOURCE citation they share; pmn_clean_x0 + _pmn_clean_x0_flat;
    pmn_pt_placeholder.
  - from solvers/literature_params.py: its bto_bc1984_6th shim (the legacy
    alpha1..-keyed dict), here as bto_bc1984_6th_legacy.
  - from solvers/pmn_field.py: pmn_pt_landau_placeholder.
WHAT STAYED in solvers/materials.py and is imported from there: the deck
plumbing (_flat_to_deck, _bto_298_mechanics), the Pan 2019 low-level
formulas and constants (_pan2019_*_si, _si_to_deck_landau_dict, PAN2019_*;
solvers/pan2019_field.py calls them) and pmn_clean_alpha1 with its defaults
(solvers/pmn_field.py calls it).

The matching TOML snapshots are in params/archive/ (see its README.md);
generate_archive_toml() below rewrites the one whose generator lives here
(bto_BC1984_6th.toml) through literature_params.write_toml.

RESTORE an entry: move its functions back into solvers/materials.py, put its
ARCHIVED_REGISTRY row back into materials._REGISTRY and drop its name from
materials._ARCHIVED.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fftpf import units
from fftpf.mat_registry import (PAN2019_G110_DECK, PAN2019_G11_G110, PAN2019_G12_G110,
                               PAN2019_G44_G110, PAN2019_G44P_G110, PAN2019_KAPPA_B_DECK,
                               PAN2019_T_REF_K, PARAMS_DIR, _PMN_C_CW_K_DEFAULT,
                               _PMN_T0_CW_K_DEFAULT, _bto_298_mechanics, _flat_to_deck,
                               _pan2019_bfo_landau_si, _pan2019_bto_landau_si,
                               _pan2019_sto_landau_si, _si_to_deck_landau_dict,
                               pmn_clean_alpha1)

ARCHIVE_PARAMS_DIR = os.path.join(PARAMS_DIR, "archive")


# --------------------------------------------------------------------------
# (b) bto_bc1984_6th -- reference only, Table 1 "Sixth order" column.
#
# alpha111 uses T-393, with T in kelvin like the other two T-dependent
# coefficients. The table prints (T-120) for alpha111. That 120 is the
# Bell & Cross reference temperature in degrees Celsius (393 K = 120 C),
# left unconverted while alpha1 (T-381) and alpha11 (T-393) were converted
# to kelvin. Taken literally with T in kelvin, it gave alpha111(298 K) =
# -7.07 deck, which left this 6th-order potential (no 8th-order terms)
# unbounded below. With T-393 the value is 8.004 at 298 K and 17.94 at
# 118 K, the same numbers as params/bto_298K.toml and bto_118K.toml (same
# Bell-Cross lineage). Sluka et al., Nat. Commun. 3, 748 (2012) Table 1
# prints the same (393-T) form (reports/LIT_pan2019_forensics.md).
# --------------------------------------------------------------------------
def _bto_bc1984_6th_flat(T=298.0):
    si = dict(
        alpha1=3.34e5 * (T - 381),
        alpha11=4.69e6 * (T - 393) - 202e6,
        alpha12=3.23e8,
        alpha111=-5.52e7 * (T - 393) + 2.76e9,
        alpha112=4.47e9,
        alpha123=4.919e9,
    )
    flat = {k: units.si_to_deck_landau(v) for k, v in si.items()}
    for k in ("alpha1111", "alpha1112", "alpha1122", "alpha1123"):
        flat[k] = 0.0
    flat.update(_bto_298_mechanics())
    meta = dict(source="Rabe/Ahn/Triscone 2007 App A (Chen), Table 1 'Sixth order' column, "
                        "PDF p.376/book p.365", T=T,
                note="REFERENCE ONLY (SPEC WP4 9.1b) -- not fit to reproduce k_t180.i; "
                     "mechanics/gradient/permittivity borrowed from bto_298K.toml (same material)")
    return flat, meta


def bto_bc1984_6th(T=298.0):
    """BaTiO3 6th-order (Rabe App A, Table 1 'Sixth order') -> R1 deck; reference only, not fit to k_t180.i."""
    flat, meta = _bto_bc1984_6th_flat(T)
    return _flat_to_deck(flat, meta)


def bto_bc1984_6th_legacy(T=298.0):
    """Was literature_params.bto_bc1984_6th: _bto_bc1984_6th_flat reshaped to the frozen alpha1..-keyed, "_meta"-suffixed dict that literature_params.write_toml serializes."""
    flat, meta = _bto_bc1984_6th_flat(T)
    out = dict(flat)
    out["_meta"] = meta
    return out


# ==========================================================================
# Pan et al., Science 365, 578 (2019) BFO/BTO/STO end members -- the R1-deck
# wrappers. The formulas and constants they read stay in solvers/materials.py
# (solvers/pan2019_field.py, a live shim, calls those); see pan2019_field.py's
# docstring for the B2-D forensics essay.
# ==========================================================================
_PAN2019_SOURCE = ("Pan, Yang, Zhang, Han, Chen, Yang, et al., Science 365, 578 (2019), "
                    "Supplementary Materials p.3-4 (SM eq S1-S7), via "
                    "reports/LIT_pan2019_science.md (iii), page-verified pSM.3-4 pdftoppm render")


def _pan2019_common_flat(landau_si_dict):
    """landau (6 keys only, no 8th-order zero-fill -- the pan2019 dossier
    never states an 8th-order term, so it stays genuinely ABSENT, not 0.0)
    + the shared, sourced gradient sector + the shared, convention-inherited
    kappa_b."""
    flat = _si_to_deck_landau_dict(landau_si_dict)
    flat.update(dict(G110=PAN2019_G110_DECK, G11_G110=PAN2019_G11_G110, G12_G110=PAN2019_G12_G110,
                      G44_G110=PAN2019_G44_G110, G44P_G110=PAN2019_G44P_G110,
                      permittivity=PAN2019_KAPPA_B_DECK))
    return flat


def _bfo_pan2019_flat(T=PAN2019_T_REF_K):
    flat = _pan2019_common_flat(_pan2019_bfo_landau_si(T))
    meta = dict(source=_PAN2019_SOURCE, T=T,
                note="BFO end-member of the ternary BiFeO3-BaTiO3-SrTiO3 region-mixture model "
                     "(SPEC S34.2). provenance_flag: UNRESOLVED (Pan's SM cites these numbers to "
                     "refs 44/45, but neither ref actually contains BFO coefficients -- used "
                     "as-printed, reports/LIT_pan2019_forensics.md).")
    return flat, meta


def bfo_pan2019(T=PAN2019_T_REF_K):
    """BiFeO3 end member of Pan 2019's ternary region mixture -> R1 deck; numeric source untraceable, flagged."""
    flat, meta = _bfo_pan2019_flat(T)
    return _flat_to_deck(flat, meta)


def _bto_pan2019_flat(T=PAN2019_T_REF_K):
    flat = _pan2019_common_flat(_pan2019_bto_landau_si(T))
    meta = dict(source=_PAN2019_SOURCE, T=T,
                note="BTO end-member of the ternary BiFeO3-BaTiO3-SrTiO3 region-mixture model "
                     "(SPEC S34.2). alpha111 uses Pan's LITERAL 2.76e8 constant (reproduction-"
                     "fidelity priority); the Sluka-2012-literal alternative (2.76e9, matching "
                     "this repo's own bto_298K.toml) is available via "
                     "_pan2019_bto_alpha111_si_sluka_literal for sensitivity checks "
                     "(reports/LIT_pan2019_forensics.md).")
    return flat, meta


def bto_pan2019(T=PAN2019_T_REF_K):
    """BaTiO3 end member of Pan 2019's ternary region mixture -> R1 deck; alpha111 Pan-literal 2.76e8."""
    flat, meta = _bto_pan2019_flat(T)
    return _flat_to_deck(flat, meta)


def _sto_pan2019_flat(T=PAN2019_T_REF_K):
    flat = _pan2019_common_flat(_pan2019_sto_landau_si(T))
    meta = dict(source=_PAN2019_SOURCE, T=T,
                note="STO end-member of the ternary BiFeO3-BaTiO3-SrTiO3 region-mixture model "
                     "(SPEC S34.2). alpha1 uses Pan's LITERAL 36.5K constant (reproduction-"
                     "fidelity priority) -- Rabe App A's own Table 3 prints 35.5K for the same "
                     "lineage number, a genuine +-1K discrepancy in Pan's own transcription "
                     "(reports/LIT_pan2019_forensics.md); not significant at the ~0.4% level.")
    return flat, meta


def sto_pan2019(T=PAN2019_T_REF_K):
    """SrTiO3 end member of Pan 2019's ternary region mixture -> R1 deck; alpha1 Pan-literal 36.5 K."""
    flat, meta = _sto_pan2019_flat(T)
    return _flat_to_deck(flat, meta)


# ==========================================================================
# PMN material potentials (SPEC S21.3). pmn_clean_alpha1 itself (and its
# _PMN_T0_CW_K_DEFAULT/_PMN_C_CW_K_DEFAULT defaults) stays in
# solvers/materials.py -- solvers/pmn_field.py calls it.
# ==========================================================================
def _pmn_clean_x0_flat(T=298.0, T0_K=_PMN_T0_CW_K_DEFAULT, C_cw_K=_PMN_C_CW_K_DEFAULT):
    """Only alpha1 is sourced -- alpha11..alpha123 are genuinely UNSOURCED
    (not zero; see pmn_field.ALPHA11_PMN_BRACKET/_NOTE for the bracket that
    was considered and rejected as too speculative to include even as a
    dial), so they stay ABSENT from `flat` (never a key at all)."""
    flat = {"alpha1": pmn_clean_alpha1(T, T0_K, C_cw_K)}
    meta = dict(
        source="Viehland, Jang, Cross & Wuttig, Phys. Rev. B 46, 8003 (1992) (primary; alpha1 "
               "only) -- cross-checked Al-Barakaty/Prosandeev/Wang/Dkhil/Bellaiche, PRB 91, "
               "214117 (2015) and a third search-corroborated source",
        T=T, T0_K=T0_K, C_cw_K=C_cw_K,
        note="SPEC S21.3 ruling 2 (hybrid a'): alpha1 MEASURED (PMN's own high-T Curie-Weiss "
             "fit); alpha11.. left absent (order-of-magnitude bracket only, NOT PMN-derived, "
             "see pmn_field.ALPHA11_PMN_BRACKET/_NOTE); alpha111/112/123 REQUIRED, too "
             "speculative to bracket with zero PMN-PT composition-trend data points in hand.")
    return flat, meta


def pmn_clean_x0(T=298.0, T0_K=_PMN_T0_CW_K_DEFAULT, C_cw_K=_PMN_C_CW_K_DEFAULT):
    """Pure PMN (x_PT=0) clean-limit Curie-Weiss alpha1(T) -> R1 deck; alpha11.. unsourced and absent."""
    flat, meta = _pmn_clean_x0_flat(T, T0_K, C_cw_K)
    return _flat_to_deck(flat, meta)


def pmn_pt_placeholder(x_pt=0.30, T=298.0):
    """0.70PMN-0.30PT: citation confirmed (Hangbo Zhang et al., PRB 96,
    054109 (2017)), numeric coefficients NOT transcribed (paywalled, HTTP
    403 on both journals.aps.org and a ResearchGate mirror -- see
    pmn_field.py's module docstring for the full research trail). Every
    sector is genuinely absent -- this registry entry exists so the
    placeholder identity is discoverable via materials.list(), matching
    params/pmn_pt_landau_x0.30.toml (also entirely REQUIRED/commented out)."""
    meta = dict(
        source="Hangbo Zhang et al., Phys. Rev. B 96, 054109 (2017), 'Phase coexistence and "
               "Landau expansion parameters for a 0.70Pb(Mg1/3Nb2/3)O3-0.30PbTiO3 single "
               "crystal' -- citation confirmed, numeric coefficients NOT transcribed "
               "(paywalled, HTTP 403 on both journals.aps.org and a ResearchGate mirror)",
        x_pt=x_pt, T=T,
        note="PMN-PT has no Haun/Rabe-App-A-style continuous function of x_PT in the "
             "accessible literature (see WP15_REPORT.md); Peng Lv's 2018 UCLA dissertation, "
             "read directly, independently confirms 'large uncertainties' in exactly this "
             "kind of composition-resolved PMN-PT Landau fit. All fields REQUIRED -- a future "
             "WP with institutional journal access should transcribe Zhang et al. 2017's Table "
             "directly, not guess.")
    return _flat_to_deck({}, meta)


def pmn_pt_landau_placeholder(x_pt=0.30, T=298.0):
    """Structure-only placeholder (see pmn_field.py's PMN-PT comment block):
    all alpha_i None (REQUIRED, not transcribed here). Caller (e.g. a TOML
    writer) is expected to render None entries as "# key = REQUIRED"
    comments, same as literature_params.write_toml already does for PZT's
    own placeholders.

    Moved here from solvers/pmn_field.py, where it was the legacy-shaped
    shim over pmn_pt_placeholder (registry name "pmn_pt_placeholder",
    matching params/archive/pmn_pt_landau_x0.30.toml); nothing called it.
    pmn_field.py's "PMN-PT Landau backbone" comment block keeps the research
    trail (Zhang et al. 2017 paywalled, Peng Lv 2018). Every sector of that
    deck is absent (SPEC §36.2 R1 "sectors a material lacks are absent"), so
    reconstructing this function's all-None legacy shape is a pure
    key-enumeration exercise, not a computation."""
    d = pmn_pt_placeholder(x_pt=x_pt, T=T)
    landau = {k: None for k in
              ("alpha1", "alpha11", "alpha12", "alpha111", "alpha112", "alpha123",
               "alpha1111", "alpha1112", "alpha1122", "alpha1123")}
    out = dict(landau)
    out.update(dict(G110=None, G11_G110=None, G12_G110=None, G44_G110=None, G44P_G110=None,
                     permittivity=None, C11=None, C12=None, C44=None,
                     Q11_phys=None, Q12_phys=None, Q44_phys=None))
    out["_meta"] = dict(d["meta"])
    return out


# ==========================================================================
# Archived registry + surface mirroring materials.get/materials.list.
# Rows are the ones removed from materials._REGISTRY, descriptions verbatim.
# ==========================================================================
ARCHIVED_REGISTRY = {
    "bto_bc1984_6th": (bto_bc1984_6th,
                        "BaTiO3 6th-order (Rabe/Ahn/Triscone 2007 App A, Table 1 Sixth-order); "
                        "reference only, not fit to k_t180.i"),
    "bfo_pan2019": (bfo_pan2019,
                     "BiFeO3 end member (Pan et al., Science 365, 578 (2019) SM p.3-4); BFO's own "
                     "numeric source is untraceable within Pan's own citations, flagged"),
    "bto_pan2019": (bto_pan2019,
                     "BaTiO3 end member (Pan et al. 2019 SM); alpha111 Pan-literal 2.76e8 "
                     "(Sluka-2012-literal 2.76e9 alternative available for sensitivity)"),
    "sto_pan2019": (sto_pan2019,
                     "SrTiO3 end member (Pan et al. 2019 SM); alpha1 Pan-literal 36.5K "
                     "(Rabe App A's own 35.5K for the same lineage number, flagged)"),
    "pmn_clean_x0": (pmn_clean_x0,
                      "Pure PMN (x_PT=0) clean-limit alpha1(T) Curie-Weiss (Viehland et al., PRB "
                      "46, 8003 (1992)); alpha11.. unsourced, absent"),
    "pmn_pt_placeholder": (pmn_pt_placeholder,
                            "0.70PMN-0.30PT (Zhang et al., PRB 96, 054109 (2017)); citation "
                            "confirmed, coefficients NOT transcribed (paywalled) -- all sectors absent"),
}


def get(name, **kw):
    """materials_archive.get(name, T=..., **kw) -> deck, the same R1 schema materials.get() returns; KeyError for a name that is not archived."""
    if name not in ARCHIVED_REGISTRY:
        raise KeyError(f"materials_archive.get: {name!r} is not an archived entry; "
                        f"archived: {sorted(ARCHIVED_REGISTRY)} (live entries: solvers.materials.get)")
    fn, _provenance = ARCHIVED_REGISTRY[name]
    return fn(**kw)


def list():
    """[(name, one-line provenance), ...] over the archived entries, sorted by name."""
    return [(name, provenance) for name, (_fn, provenance) in sorted(ARCHIVED_REGISTRY.items())]


def generate_archive_toml(out_dir=ARCHIVE_PARAMS_DIR):
    """Rewrite the archived snapshot whose generator lives here (bto_BC1984_6th.toml, T=298 K) via literature_params.write_toml; literature_params.generate_all_toml covers the others."""
    from fftpf import mat_chen2007_decks as literature_params
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bto_BC1984_6th.toml")
    literature_params.write_toml(bto_bc1984_6th_legacy(T=298.0), out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    generate_archive_toml()
