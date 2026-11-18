"""SPEC S34.2 (B2-A): Pan et al., Science 365, 578 (2019) BFO/BTO/STO Landau
coefficient generator functions, transcribed VERBATIM from
reports/LIT_pan2019_science.md (iii) (page-verified pSM.3-4, pdftoppm render,
NOT pdftotext OCR -- that dossier's own caveat). T-dependent generator-function
pattern, repo precedent: solvers/pmn_field.py's pmn_clean_alpha1(T) +
params/archive/pmn_clean_x0.toml (TOML stores a numeric snapshot at one
reference T; this module supplies the formula for any other T). Discrete
region-mixture model (dossier (iii): "mixtures of BTO, STO and BFO regions... each cell
wholesale to one fixed end-member identity, not a per-cell blend") -- these
functions feed solvers/composite.py's make_label_field_multi/alpha_field_multi,
NOT a continuous Vegard interpolation (pzt_field.py's x(r) machinery, dossier
(iv): explicitly the WRONG analog here).

Units: dossier (iii) transcribes SI values ("units C^-2m^2N/C^-4m^6N/C^-6m^10N
as printed", i.e. standard Landau-coefficient SI, same convention every other
_si-suffixed function in this repo already uses, e.g. pmn_field's
defect_field_substitutional). si_to_deck_landau (units.py, WP4 S9.2) applies
uniformly regardless of the P-power a coefficient multiplies -- SAME factor
for alpha1/alpha11/alpha111/alpha112/alpha123, verified in units.py's own
docstring, reused here unmodified via si_to_deck_landau_dict below (mirrors
composite.py's own si_to_deck_alpha wrapper, same one-line pattern).

T reference: dossier (vi) -- "no numeric simulation T anywhere (room
temperature is the obvious but unstated inference)" -- T_REF_K=298.0 is this
project's own standing room-T convention (W0/WP16/WP20 all use 298K), a
DECLARED choice, not a paper-stated number.

Gradient sector: dossier (iii), SHARED (not region-dependent) --
G11/G110=1.2, G12/G110=0, G44/G110=G44'/G110=0.6, G110=7.04e-11 SI [C^-2m^4N]
-- "their own values -- a SOURCED gradient sector" (SPEC S34.2), transcribed
as-is via the NEW units.si_to_deck_gradient (first use in this repo, see that
function's docstring for the derivation).

Background permittivity kappa_b: dossier (ii) -- Pan's own f_elec (eq S7) has
NO background-permittivity term at all (only -Pi*Ei^ex - (1/2)Ei^in*Pi); this
source gives no kappa_b number. Repo-standing fallback for an unsourced
kappa_b: 10*eps0 (Wang, Ma, Li, Britson & Chen, Acta Mater 61 (2013) 7591
lineage), the SAME convention already used by WP9/pzt_elastic_spm_g14e.py,
WP11/wp11_matrix.py, WP15-16/wp16_pmn_pilot.py, W0/w0_relaxor_demo.py, and
params/nnbm_wang2026.toml's own "REUSED fallback KAPPA_B=10*eps0" note --
CONVENTION-INHERITED, UNSOURCED for THIS paper specifically, flagged here
exactly as those precedents flag it for theirs.

B2-D FORENSICS RESOLUTION (reports/LIT_pan2019_forensics.md, coordinator
ruling 2026-08-28, mid-B2-A): bto_alpha111_si(298.0)=5.52e9 (Pan's own
printed alpha111=-5.52e7(T-393)+2.76e8) diverges from this repo's
INDEPENDENT bto_298K.toml (Ferret-.i-file-extracted, LCC2005 lineage,
gate-certified) alpha111=8.004e9. Traced: Pan's BTO block is a direct copy
of ref 45 (Sluka et al., Nat Commun 3, 748 (2012)) Table 1, which prints the
SAME (T-393) reference and leading coefficient but a constant of 2.76e9, not
Pan's 2.76e8 -- Pan's SM drops one order of magnitude off that one constant.
The Sluka-literal alternative (bto_alpha111_si_sluka_literal below) gives
8.004e9 at 298K, EXACTLY bto_298K.toml's own value -- strong indirect
corroboration of "what Pan meant", but an inference, not a re-derivation.
RULING: main deck (bto_landau_deck, params/bto_pan2019.toml) uses Pan's
LITERAL 2.76e8 (reproduction-fidelity priority); gates/g26c_endmember_0d.py
carries a zero-compute 0D sensitivity line comparing both candidates' T=298K
ground state -- if a later domain-structure classification comes out wrong,
this pick is suspect #2 (suspect #1 = elastic OFF).

BFO PROVENANCE (B2-D forensics): Pan's own SM cites BFO's alpha1/alpha11/
alpha12 to refs (44,45) alongside BTO/STO, but NEITHER ref actually contains
BFO coefficients -- Rabe App A (ref 44) has no BiFeO3 section at all
(exhaustive page scan); Sluka 2012 (ref 45) is entirely about BaTiO3/KNbO3,
zero BFO occurrences anywhere in its full text. BFO's true numeric source is
uncited/untraceable within Pan's own paper (a documentation gap in Pan's own
citations, not a paywall/access problem) -- used AS-PRINTED (pM.2) with this
provenance-unresolved flag on record, no independent cross-check exists.

SPEC §36 (WP25 Phase 2, stage 2A, 2026-08-30): every formula/constant below
moved to solvers/materials.py, which also registered BFO/BTO/STO as the
entries "bfo_pan2019"/"bto_pan2019"/"sto_pan2019" ("a paper-specific
module" stops being a concept, per SPEC §36.3). Those three R1-deck entries had no
caller and were ARCHIVED 2026-11 to solvers/materials_archive.py
(materials_archive.get("bto_pan2019", T=...)); the formulas and constants
this shim calls stay in solvers/materials.py, and the elastic/
electrostriction decks params/{bfo,bto,sto}_pan2019.toml stay in params/
(cases/b2e_pan2019_3d_case.py reads them).
This module is now a thin PARAMETER-BINDING shim (SPEC §36.2 R4): every
function/constant here calls or equals the matching solvers.materials name,
bit-for-bit identical to the pre-migration values (gates/
g29a_materials_equivalence.py). This module's name and every function/
constant name are PERMANENT (SPEC §35 ruling (3), extended by §36.5) -- new
code should call solvers.materials directly."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Re-exported ON PURPOSE, not a dead import: gates/g26c_endmember_0d.py:105
# reaches it as `pf.units.si_to_deck_landau(...)`, so this binding is part of
# this shim's frozen public surface (SPEC §36.2 R4, §35 ruling (3)) and must
# not be removed as "unused" -- an in-module AST scan alone will call it dead.
from fftpf import units
from fftpf import mat_registry as materials

T_REF_K = materials.PAN2019_T_REF_K  # declared room-T reference, see module docstring


# PARAMETER-BINDING SHIM (SPEC §36.2 R4): every function and constant below
# has its signature, defaults and return shape FROZEN.
# --------------------------------------------------------------------------
# BFO (dossier iii): alpha1=4.9e5*(T-1103), alpha11=6.5e8, alpha12=1.0e8,
# alpha111=alpha112=alpha123=0.
# --------------------------------------------------------------------------
def bfo_alpha1_si(T):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bfo_alpha1_si. BFO alpha1(T) in SI."""
    return materials._pan2019_bfo_alpha1_si(T)


def bfo_landau_si(T):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bfo_landau_si. BFO alpha1..alpha123 dict in SI."""
    return materials._pan2019_bfo_landau_si(T)


# --------------------------------------------------------------------------
# BTO (dossier iii): alpha1=3.34e5*(T-381), alpha11=4.69e6*(T-393)-2.02e8,
# alpha12=3.23e8 (T-independent), alpha111=-5.52e7*(T-393)+2.76e8,
# alpha112=4.47e9, alpha123=4.91e9 (both T-independent).
# --------------------------------------------------------------------------
def bto_alpha1_si(T):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bto_alpha1_si. BTO alpha1(T) in SI."""
    return materials._pan2019_bto_alpha1_si(T)


def bto_alpha11_si(T):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bto_alpha11_si. BTO alpha11(T) in SI."""
    return materials._pan2019_bto_alpha11_si(T)


def bto_alpha111_si(T):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bto_alpha111_si. BTO alpha111(T) in SI, Pan-literal constant."""
    return materials._pan2019_bto_alpha111_si(T)


def bto_landau_si(T):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bto_landau_si. BTO alpha1..alpha123 dict in SI."""
    return materials._pan2019_bto_landau_si(T)


def bto_alpha111_si_sluka_literal(T):
    """ALTERNATE, NOT used by bto_landau_si/bto_landau_deck (production path
    stays Pan-literal, coordinator ruling above): ref 45 (Sluka et al., Nat
    Commun 3, 748 (2012)) Table 1's OWN printed alpha111=(393-T)*5.52e7+2.76e9
    -- algebraically (393-T)*5.52e7 == -5.52e7*(T-393), so this differs from
    bto_alpha111_si ONLY in the constant (2.76e9 here vs Pan's printed
    2.76e8). Exists solely for gates/g26c_endmember_0d.py's zero-compute
    sensitivity comparison."""
    return materials._pan2019_bto_alpha111_si_sluka_literal(T)


# --------------------------------------------------------------------------
# STO (dossier iii): alpha1=7.06e5*(T-36.5) -- FLAGGED per SPEC S34.1: the
# dossier reads Pan's own SI PDF as 36.5, vs Rabe App A's OWN printed
# 7.06(T-35.5) for what may be the same lineage number; B2-D forensics
# verifies which is right (reports/LIT_pan2019_forensics.md, not yet on disk
# at B2-A build time) -- transcribed here AS PRINTED IN THE DOSSIER (36.5),
# per SPEC S34.2's "exact transcription" instruction, not pre-empting B2-D's
# verdict. alpha11=1.7e9, alpha12=1.37e9, alpha111=alpha112=alpha123=0.
# --------------------------------------------------------------------------
def sto_alpha1_si(T):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_sto_alpha1_si. STO alpha1(T) in SI, Pan-literal 36.5 K."""
    return materials._pan2019_sto_alpha1_si(T)


def sto_landau_si(T):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_sto_landau_si. STO alpha1..alpha123 dict in SI."""
    return materials._pan2019_sto_landau_si(T)


def si_to_deck_landau_dict(d_si):
    """{alpha_name: SI value} -> {alpha_name: deck value}, uniform x1e-9
    (units.si_to_deck_landau), mirrors composite.py's own si_to_deck_alpha."""
    return materials._si_to_deck_landau_dict(d_si)


def bfo_landau_deck(T=T_REF_K):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bfo_landau_si, then si_to_deck_landau_dict -- BFO alphas in deck units."""
    return si_to_deck_landau_dict(bfo_landau_si(T))


def bto_landau_deck(T=T_REF_K):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bto_landau_si, then si_to_deck_landau_dict -- BTO alphas in deck units."""
    return si_to_deck_landau_dict(bto_landau_si(T))


def sto_landau_deck(T=T_REF_K):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_sto_landau_si, then si_to_deck_landau_dict -- STO alphas in deck units."""
    return si_to_deck_landau_dict(sto_landau_si(T))


# --------------------------------------------------------------------------
# Shared gradient sector (dossier iii, SOURCED -- see module docstring).
# --------------------------------------------------------------------------
G110_SI = materials.PAN2019_G110_SI       # C^-2 m^4 N, as printed
G11_G110 = materials.PAN2019_G11_G110
G12_G110 = materials.PAN2019_G12_G110
G44_G110 = materials.PAN2019_G44_G110
G44P_G110 = materials.PAN2019_G44P_G110   # dossier: "G44=G44'" -- their "isotropic" means only this equality

G110_DECK = materials.PAN2019_G110_DECK

# --------------------------------------------------------------------------
# Shared background permittivity: UNSOURCED for this paper (module docstring);
# repo-standing 10*eps0 fallback. SPEC V2-02.1 (Lens 2 U3, D7): the base constant
# is 10*eps0 in deck units, so KAPPA_B_DECK = 10 * 0.08854187 = 0.8854187 is
# eps_b = 100, not 10 -- label corrected here, values frozen (see materials.py).
# --------------------------------------------------------------------------
EPS0_DECK_BASE = materials.PAN2019_EPS0_DECK_BASE  # WP15/WP16 "kappa_b * this = deck permittivity" base constant
KAPPA_B_MULT = materials.PAN2019_KAPPA_B_MULT
KAPPA_B_DECK = materials.PAN2019_KAPPA_B_DECK


if __name__ == "__main__":
    for name, fn in (("BFO", bfo_landau_deck), ("BTO", bto_landau_deck), ("STO", sto_landau_deck)):
        d = fn(T_REF_K)
        print(f"{name} deck @ T={T_REF_K}K:", {k: f"{v:.10g}" for k, v in d.items()})
    print(f"G110_deck={G110_DECK:.10g}  KAPPA_B_DECK={KAPPA_B_DECK:.10g}")
