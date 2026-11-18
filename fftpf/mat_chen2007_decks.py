"""WP4 §9.1: new parameter sets from Rabe/Ahn/Triscone 2007 Appendix A
(Long-Qing Chen), PDF pages 376-383 (book pp. 363-371), transcribed and
cross-checked directly against the rendered PDF pages (not re-typed from
SPEC.md's summary) -- see WP4_REPORT.md for the coefficient-by-coefficient
verification (24/24 exact matches for sets (a)-(c); set (d) is transcribed
fresh from the PDF's printed formula, SPEC.md only sketches its structure).

Every function returns SI-valued Landau coefficients THEN converts them to
deck units via units.si_to_deck_landau (the WP4 §9.2 converter, gated by
G6a) -- so callers never see raw SI numbers by accident. Electrostriction Q
and elastic C are also returned in deck units (Q needs no conversion; C via
units.si_to_deck_elastic after any compliance inversion).

SPEC §36 (WP25 Phase 2, stage 2A, 2026-08-30): the FORMULAS below moved to
solvers/materials.py (the new unified registry -- materials.get("bto_"
"lcc2005_8th", T=...) etc., engine-native a1..a123 keys, SPEC §36.2 R1
schema). Every function in this module is now a thin PARAMETER-BINDING shim
(SPEC §36.2 R4): it calls the matching materials.py function and reshapes
the result back into this module's ORIGINAL alpha1..alpha123-keyed,
"_meta"-suffixed dict shape -- bit-for-bit identical to the pre-migration
output, verified by gates/g29a_materials_equivalence.py. This module's own
name, every function name/signature, and every returned dict's shape are
PERMANENT (SPEC §35 ruling (3), extended to Phase 2 by §36.5) -- new code
should call solvers.materials directly instead of adding new callers here.
One exception, by user decision 2026-11: bto_bc1984_6th had no caller and
was archived with its generator to solvers/materials_archive.py (as
bto_bc1984_6th_legacy, same dict), so it is no longer importable from here.
write_toml/generate_all_toml stay in this module: they serialize the legacy
alpha1..-keyed/"_meta" dict shape this module's functions still produce, so
they keep working through the shim (generate_all_toml now writes the
archived snapshots to params/archive/).

_bto_298_mechanics (the bto_298K.toml reader SPEC §9.1(a) originally kept
here) MOVED to solvers.materials._bto_298_mechanics -- it had no external
caller (grep-confirmed) beyond this module's own three BTO functions below,
all of which now call materials.py directly, so keeping a second copy here
would be exactly the "copy-paste, two live copies" this migration is
explicitly meant to avoid (SPEC §36.3 2A). Removed, not shimmed.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fftpf import mat_registry as materials
from fftpf.deck_reader import TOML_SECTIONS, _section_is_absent


# --------------------------------------------------------------------------
# (a) bto_LCC2005_8th -- PRODUCTION STANDARD. PDF Table 1, "Eight-order"
# column (verified exact against the rendered page, all 10 coefficients).
#
# SHIM (SPEC §36 stage 2A): formula now lives in
# solvers.materials._bto_lcc2005_8th_flat -- this wrapper just reshapes the
# (flat, meta) pair it returns back into this module's original
# alpha1..-keyed, "_meta"-suffixed dict, bit-for-bit (gates/
# g29a_materials_equivalence.py).
# PARAMETER-BINDING SHIM (SPEC §36.2 R4): signature and return shape FROZEN.
# --------------------------------------------------------------------------
def bto_lcc2005_8th(T=298.0):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._bto_lcc2005_8th_flat, reshaped to this module's frozen alpha1..-keyed dict."""
    flat, meta = materials._bto_lcc2005_8th_flat(T)
    out = dict(flat)
    out["_meta"] = meta
    return out


# --------------------------------------------------------------------------
# (a') bto_LCC2005_paper -- SPEC WP6 §12: the SAME LCC2005 8th-order alphas
# as (a) above (Li & Chen APL 88, 072905 (2006)'s own Ref. 18 footnote
# reproduces this potential to the precision it prints -- cross-checked
# coefficient-by-coefficient against bto_lcc2005_8th()'s SI values: 9/10
# exact, alpha123 differs -2.5009e9 (Rabe App A) vs -2.500e9 (paper's own
# 4-sig-fig footnote) = 0.036% rounding, not a discrepancy), but with
# mechanics (C11/C12/C44/Q11/Q12/Q44) replaced by the paper's OWN Ref. 18
# footnote values (its literal parameter list, so no Rabe App A "which Q
# column" ambiguity applies -- see WP6_REPORT.md). SPEC: "do NOT touch the
# HM2006-based sets" -- this is a new, separate set, bto_lcc2005_8th()
# itself is untouched (called here only to reuse its landau part).
#
# Gradient/permittivity: the paper gives NEITHER a numeric G110 nor a
# permittivity anywhere in the (3-page) letter -- only the RATIO
# G11/G110=1.0 (isotropic, their Eq 7) and the formula dx=sqrt(G110/alpha0).
# Per SPEC's explicit "choose G110 accordingly -- document the value": we
# choose dx=0.4nm (comparable to this project's own HM2006-based
# domain-wall-resolving grids, e.g. bto_298K h=0.25nm) and back out
# G110=dx^2*alpha0 from the paper's own formula -- alpha0:=|alpha1(25C)|
# read from bto_lcc2005_8th(T=298)'s own deck-unit alpha1 (298K is this
# project's 25C, see bto_lcc2005_8th's T-388 Kelvin-shifted formula).
# G12_G110=0, G44_G110=G44P_G110=0.5 is the ONLY isotropic degenerate case
# of SPEC §1.2's general 4-constant Ferret-convention gradient energy that
# reduces to the paper's Eq 7 (algebra in WP6_REPORT.md) -- NOT a free
# choice, a transcription of the paper's own stated model. Permittivity:
# no basis in the paper at all; inherited from bto_298K.toml (HM2006,
# same material) as a documented placeholder, same spirit as (c)/(d)'s
# "BTO's own values as a starting point" precedent below.
#
# SHIM (SPEC §36 stage 2A): formula now lives in
# solvers.materials._bto_lcc2005_paper_flat.
# PARAMETER-BINDING SHIM (SPEC §36.2 R4): signature and return shape FROZEN.
# --------------------------------------------------------------------------
def bto_lcc2005_paper(T=298.0, dx_nm=0.4):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._bto_lcc2005_paper_flat, reshaped to this module's frozen alpha1..-keyed dict."""
    flat, meta = materials._bto_lcc2005_paper_flat(T, dx_nm)
    out = dict(flat)
    out["_meta"] = meta
    return out


# --------------------------------------------------------------------------
# (b) bto_BC1984_6th -- ARCHIVED 2026-11 (no computation called it). The
# formula (Rabe App A Table 1 "Sixth order", verified exact against the
# rendered page, all 6 coefficients; a111 with T-393 in kelvin, not the
# page's Celsius 120 -- see the comment on
# solvers.materials_archive._bto_bc1984_6th_flat) and this
# module's former bto_bc1984_6th shim (now materials_archive.
# bto_bc1984_6th_legacy, same legacy dict) live in
# solvers/materials_archive.py; its snapshot is
# params/archive/bto_BC1984_6th.toml.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# (c) pto_Haun1987 -- PbTiO3, PDF Table 5 (verified exact against the
# rendered page, all 6 Landau + 3 Q + 3 compliance values).
#
# SHIM (SPEC §36 stage 2A): formula now lives in
# solvers.materials._pto_haun1987_flat.
# PARAMETER-BINDING SHIM (SPEC §36.2 R4): signature and return shape FROZEN.
# --------------------------------------------------------------------------
def pto_haun1987(T=298.0):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pto_haun1987_flat, reshaped to this module's frozen alpha1..-keyed dict."""
    flat, meta = materials._pto_haun1987_flat(T)
    out = dict(flat)
    out["_meta"] = meta
    return out


# --------------------------------------------------------------------------
# (d) pzt_Haun_x generator -- function of composition x (Ti content),
# PDF p.367/book p.378 formula set, transcribed fresh from the rendered
# page (SPEC.md only sketches the structure, doesn't give the numbers).
# Q44 carries an explicit 1/2 prefactor in the source that Q11/Q12 do NOT
# -- kept exactly as printed, per SPEC's explicit instruction.
#
# SHIM (SPEC §36 stage 2A): every formula/table below now lives in
# solvers.materials (_pzt_T0/_pzt_C0/pzt_haun_x_landau_si/pzt_haun_x_Q/
# pzt_haun_x_elastic_si/_pzt_haun_x_flat) -- these six names are thin
# pass-throughs, kept because gates/g6d_pzt.py calls lp._pzt_T0/lp._pzt_C0
# directly (underscore-prefixed, so `import *` would never have forwarded
# them -- same lesson SPEC §35's octa_tilt.py shim already learned the hard
# way, applied here from the start) and several gates/cases call
# lp.pzt_haun_x_landau_si/_Q/_elastic_si directly.
# PARAMETER-BINDING SHIM (SPEC §36.2 R4): signature and return shape FROZEN.
# --------------------------------------------------------------------------
def _pzt_T0(x):
    return materials._pzt_T0(x)


def _pzt_C0(x):
    return materials._pzt_C0(x)


def pzt_haun_x_landau_si(x, T):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials.pzt_haun_x_landau_si, pass-through (SI values)."""
    return materials.pzt_haun_x_landau_si(x, T)


def pzt_haun_x_Q(x):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials.pzt_haun_x_Q, pass-through (electrostriction Q, no unit conversion)."""
    return materials.pzt_haun_x_Q(x)


def pzt_haun_x_elastic_si(x, mechanics=True):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials.pzt_haun_x_elastic_si, pass-through (SI values)."""
    return materials.pzt_haun_x_elastic_si(x, mechanics=mechanics)


def pzt_haun_x(x=0.6, T=298.0, mechanics=True):
    """PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pzt_haun_x_flat, reshaped to this module's frozen alpha1..-keyed dict."""
    flat, meta = materials._pzt_haun_x_flat(x, T, mechanics)
    out = dict(flat)
    out["_meta"] = meta
    return out


# --------------------------------------------------------------------------
# TOML writer for params/ (SPEC WP4 §9.1: "New parameter sets under
# params/"). Same [landau]/[gradient]/[electrostatics]/[elastic]/
# [electrostriction] schema as deck_parser.materials_to_toml (SPEC §6.1),
# but with a literature-citation [meta] block instead of a deck path, and
# None-valued fields (SPEC's "leave a REQUIRED field the case must set")
# OMITTED with an explanatory comment rather than written as a fake number.
# The optional q/theta sections (deck_parser._OPTIONAL_TOML_SECTIONS) are
# skipped entirely when the set has none of their keys: none of the sets
# here has an AFE q or a tilt theta sector, so those keys are not missing
# data and get no REQUIRED placeholder lines.
# --------------------------------------------------------------------------
def write_toml(params, out_path):
    """Write a legacy alpha1..-keyed params dict to a SPEC §6.1 TOML file; None-valued REQUIRED fields are omitted with a comment, never faked."""
    meta = params["_meta"]
    lines = ["[meta]"]
    for k, v in meta.items():
        lines.append(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {float(v)!r}")
    lines.append("")
    for section, keys in TOML_SECTIONS.items():
        if _section_is_absent(section, params):
            continue
        lines.append(f"[{section}]")
        for k in keys:
            v = params.get(k)
            if v is None:
                lines.append(f"# {k} = REQUIRED (not given in the source; case must set explicitly)")
            else:
                # cast off numpy scalar types (e.g. np.float64, whose repr()
                # under numpy>=2.0 is "np.float64(...)", not valid TOML) --
                # some formulas above (np.exp etc.) produce numpy scalars.
                lines.append(f"{k} = {float(v)!r}")
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


def generate_all_toml(out_dir):
    """Write the concrete (default-parameter) TOML for each WP4 §9.1 set
    whose generator lives here. pzt_Haun_x is a generator (function of x);
    this writes only its default validation instance (x=0.6) -- see
    pzt_haun_x() for arbitrary x.

    `out_dir` is the params/ directory: names are relative to it. The
    snapshots no computation loads were archived 2026-11 and are written to
    out_dir/archive/ (params/archive/README.md). The bto_BC1984_6th set's
    generator is archived too; solvers.materials_archive.
    generate_archive_toml writes that one."""
    specs = [
        ("archive/bto_LCC2005_8th.toml", bto_lcc2005_8th(T=298.0)),
        ("archive/pto_Haun1987.toml", pto_haun1987(T=298.0)),
        ("archive/pzt_Haun_x0.6.toml", pzt_haun_x(x=0.6, T=298.0)),
        ("bto_LCC2005_paper.toml", bto_lcc2005_paper(T=298.0)),  # SPEC WP6 §12
    ]
    for name, params in specs:
        out_path = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_toml(params, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    generate_all_toml(os.path.join(_ROOT, "params"))
