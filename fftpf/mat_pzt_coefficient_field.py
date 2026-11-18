"""SPEC WP9 §15.1: PZT material-field layer.

Given a composition field x(r) (a scalar or an (Ny,Nx) array, Ti fraction in
Pb(Zr_{1-x}Ti_x)O3) and temperature T, evaluate ALL Haun coefficient fields
pointwise by calling solvers/materials.py's generator functions directly
-- "one implementation, not a copy" (SPEC): the formulas themselves live
only in solvers/materials.py (array-safe there; see the SPEC §36 stage-2A
note below); this module just wraps them for field (array) use and
adds the disorder-channel plumbing SPEC §15.1 asks for.

Three disorder channels (independently switchable, "Ferret-era wiring-
verification method" -- SPEC §15.1 bullet 2):
  ch1 random-bond: x(r) -> local Landau/T0 -- just call landau_field()/
      Q_field() with a NON-uniform x; no separate helper needed.
  ch2 Vegard eigenstrain: vegard_eigenstrain_field() below, feeds the
      existing eps_star_static path (SPEC §6.2b).
  ch3 charged-defect field: defect_field() below, feeds the (now
      field-capable, see solver.py's WP9 _reshape_vec3 edit) E_ext argument.

Elastic C(x) (Table-4 compliance interpolation, materials.
pzt_haun_x_elastic_si) is COMPUTED and exposed via elastic_C_at, but Phase A
never builds a spatial C(r) field -- the solver always runs FROZEN,
homogeneous C11/C12/C44 evaluated at a single scalar x (normally x_bar); see
WP9_REPORT.md's freeze-cost estimate for the quantified cost of that choice.

SPEC §36 (WP25 Phase 2, stage 2A, 2026-08-30): landau_field/Q_field/
elastic_C_field/elastic_C_at (the x-dependent Haun formula evaluators) are
now thin shims calling solvers.materials.pzt_haun_x_landau_si/_Q/
_elastic_si directly (the SAME array-aware functions literature_params.
pzt_haun_x's own shim calls) -- "one implementation, not a copy" now means
solvers/materials.py, not literature_params.py; this module's own
`from solvers import literature_params as lp` import is accordingly gone
(unused once these four functions stopped calling it). Bit-for-bit
identical to the pre-migration output (gates/g29a_materials_equivalence.py,
and the existing gates/g13c_composition_degenerate.py cross-check, which
this shim now satisfies BY CONSTRUCTION -- literature_params.pzt_haun_x and
this module's landau_field/Q_field/elastic_C_at trace back to the exact
same solvers.materials computation).

SPEC §36 (WP25 Phase 2, stage 2B, 2026-08-30): composition_field/
composition_std/composition_std_correlated/vegard_eigenstrain_field/
formula_units_per_cell are now thin PARAMETER-BINDING shims onto
solvers/composition.py's generalized geometry-layer functions of the same
name -- DLNA_DX_PZT/A_LATTICE_PZT_NM (below, UNCHANGED -- several gates/
cases read pf.DLNA_DX_PZT/pf.A_LATTICE_PZT_NM directly, e.g.
gates/g13b_field_fd_consistency.py, cases/wp11c_matrix.py) are passed
through explicitly as the "a_lattice"/"dlna_dx" arguments those general
functions now require (PZT's own numbers ALSO recorded independently as
solvers.composition.PRESET_PZT, SPEC S36.2 R3's "named preset, not a
hard-coded default"). defect_field is likewise now a shim, but onto
solvers/quenched_fields.py instead -- ch3's declared-dial stack-of-3
Gaussian field never depended on any PZT-specific number, so it belongs in
the disorder-MECHANISM layer, not the composition-GEOMETRY layer (SPEC
S36.3; see quenched_fields.gaussian_random_field). Bit-for-bit equivalence:
gates/g29d_composition_equivalence.py. `from solvers import disorder` is
accordingly gone (unused once these five functions stopped calling it
directly).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from fftpf import mat_registry as materials  # SPEC S36 stage 2A -- landau_field/Q_field/elastic_C_field/elastic_C_at shim
from fftpf import units
from fftpf import composition_field as composition  # SPEC S36 stage 2B -- composition_field/_std/_std_correlated/vegard/formula_units_per_cell shim
from fftpf import random_field_mechanisms as quenched_fields  # SPEC S36 stage 2B -- defect_field shim

# --------------------------------------------------------------------------
# SPEC §15.3 provenance: d ln(a)/dx (Vegard slope), PZT pseudocubic lattice
# parameter vs composition. Source: Shirane, Hoshino & Suzuki (1952) X-ray
# lattice-parameter data for the PbZrO3-PbTiO3 solid solution, as reproduced
# in Jaffe/Cook/Jaffe (1971)-lineage secondary literature (graphical read,
# NOT a digitized table -- see WP9_REPORT.md for the exact source and the
# uncertainty band). Using the PSEUDOCUBIC cell parameter a_pc=[a^2*c]^(1/3)
# (smooth across the tetragonal/rhombohedral MPB even though a,c individually
# kink there), the slope near x_bar=0.48 reads off as
#   da_pc/dx ~= -0.16 +- 0.05 Angstrom per unit x,  a_pc(x~0.48) ~= 4.06 A
# giving dlna_dx = (1/a)(da/dx). Sign: Ti4+ (0.605 A, 6-coord) is smaller
# than Zr4+ (0.72 A), so more Ti (larger x) shrinks the cell -- correct sign.
DLNA_DX_PZT = -0.040  # per unit x; documented uncertainty +-0.015 (see above)

# Perovskite pseudocubic lattice constant near the MPB (same source), used
# ONLY for the x(r) generator's formula-units-per-cell count below.
A_LATTICE_PZT_NM = 0.406  # nm (4.06 Angstrom)


def landau_field(x, T):
    """SPEC §15.1: alpha1(x,T) via T0(x),C0(x); alpha11/111/112 via the App A
    formulas incl. eta1/eta2 -> alpha12,alpha123. x: scalar or (Ny,Nx) array
    (materials.pzt_haun_x_landau_si is array-safe after WP9's
    _pzt_C0 vectorization). Returns a dict of DECK-unit alpha_i (a1..a123),
    same shape as x, plus the 8th-order keys pinned to 0.0 (the Haun
    potential is 6th-order only) -- ready for solver.bulk_force/
    bulk_energy_density (sympy-lambdified numpy already broadcasts over
    array coefficients).

    PARAMETER-BINDING SHIM (SPEC §36.2 R4): signature and return shape FROZEN."""
    xarr = np.asarray(x, dtype=float)
    a1, a11, a12, a111, a112, a123 = materials.pzt_haun_x_landau_si(xarr, T)
    alpha = dict(
        a1=units.si_to_deck_landau(a1), a11=units.si_to_deck_landau(a11),
        a12=units.si_to_deck_landau(a12), a111=units.si_to_deck_landau(a111),
        a112=units.si_to_deck_landau(a112), a123=units.si_to_deck_landau(a123),
    )
    for k8 in ("a1111", "a1112", "a1122", "a1123"):
        alpha[k8] = 0.0
    return alpha


def Q_field(x):
    """Q11,Q12,Q44(x) -- SPEC §15.1. x scalar or array; materials.
    pzt_haun_x_Q has no branching, already array-safe. Returns (Q11,Q12,Q44).

    PARAMETER-BINDING SHIM (SPEC §36.2 R4): signature and return shape FROZEN."""
    xarr = np.asarray(x, dtype=float)
    return materials.pzt_haun_x_Q(xarr)


def elastic_C_field(x, mechanics=True):
    """WP10 SPEC §16: C11,C12,C44 (deck units) as a SPATIAL field -- x may be
    an (Ny,Nx) array (or scalar), via materials.pzt_haun_x_elastic_si's
    WP10 array vectorization ("one implementation, not a copy": elastic_C_at
    below stays scalar-only/unchanged for Phase-A frozen-C callers; this is
    the array-capable sibling Phase B's inhomogeneous-C(r) solver needs).
    Caller must keep x within Table 4's [0.4,0.9] fitted range (np.clip
    before calling -- e.g. WP10's disorder fields, whose x_bar=0.48 is close
    to the 0.4 edge).

    PARAMETER-BINDING SHIM (SPEC §36.2 R4): signature and return shape FROZEN."""
    c = materials.pzt_haun_x_elastic_si(x, mechanics=mechanics)
    if c is None:
        return None
    c11_si, c12_si, c44_si = c
    return (units.si_to_deck_elastic(c11_si), units.si_to_deck_elastic(c12_si),
            units.si_to_deck_elastic(c44_si))


def elastic_C_at(x, mechanics=True):
    """Elastic C11,C12,C44 (deck units) at a SINGLE scalar composition x, via
    materials.pzt_haun_x_elastic_si (Table-4 compliance
    interpolation + cubic inversion). Phase A only ever calls this at one
    scalar x at a time (x_bar for the frozen solver C, or an x-window
    endpoint for the freeze-cost estimate) -- never as a spatial field -- so
    pzt_haun_x_elastic_si's existing scalar-only [0.4,0.9] range check
    (SPEC §9.1d) is reused UNCHANGED, not vectorized.

    PARAMETER-BINDING SHIM (SPEC §36.2 R4): signature and return shape FROZEN."""
    c = materials.pzt_haun_x_elastic_si(float(x), mechanics=mechanics)
    if c is None:
        return None
    c11_si, c12_si, c44_si = c
    return (units.si_to_deck_elastic(c11_si), units.si_to_deck_elastic(c12_si),
            units.si_to_deck_elastic(c44_si))


# --------------------------------------------------------------------------
# ch2: Vegard eigenstrain (SPEC §15.1) -- isotropic chemical strain from a
# LOCAL composition deviation, added into the existing static-eigenstrain
# path (SPEC §6.2b's eps_star_static, disorder.assemble_eps_star_static's
# {xx,yy,zz,...} component-dict convention).
#
# SHIM (SPEC S36 stage 2B, module docstring): calls
# solvers.composition.vegard_eigenstrain_field with DLNA_DX_PZT passed
# through explicitly.
# --------------------------------------------------------------------------
def vegard_eigenstrain_field(x, x_bar, dlna_dx=DLNA_DX_PZT):
    """eps_chem = dlna_dx*(x-x_bar)*I3. x may be an (Ny,Nx) array or a
    scalar (matching shape returned) -- a scalar x=x_bar gives an exact
    all-zero field (degenerate-limit sanity check)."""
    return composition.vegard_eigenstrain_field(x, x_bar, dlna_dx)


# --------------------------------------------------------------------------
# x(r) generator (SPEC §15.1 bullet 3): reuse the master-grid Gaussian-
# filter machinery (disorder.random_field_gaussian, already master-grid/
# resample-capable via disorder.make_random_strain_master/
# resample_strain_dict -- no new machinery needed for a future multi-
# resolution use, just call those on this field the same way they're
# already called on eps_star_static components).
#
# SHIM (SPEC S36 stage 2B, module docstring): the four functions below call
# solvers.composition's generalized geometry-layer functions of the same
# name, with A_LATTICE_PZT_NM passed through explicitly.
# --------------------------------------------------------------------------
def formula_units_per_cell(hx, hy, a_lattice=A_LATTICE_PZT_NM, hz=None):
    """N_f = formula units per coarse-grain cell = cell volume / a^3 (SPEC:
    "compute from cell volume / a^3; document"). The 2D engine has no
    independent out-of-plane grid spacing to draw "cell volume" from, so --
    absent any other basis -- the cell is treated as a CUBE of edge h (hz
    defaults to hx): a documented WP9 choice, not a literature number.

    FLOORED per edge at a_lattice itself: at this project's production grid
    spacings (h~0.1-0.25nm) the FFT grid is FINER than PZT's ~0.4nm lattice
    constant, so a literal h^3/a^3 gives N_f<1 -- an unphysical "less than
    one formula unit," since composition is only statistically meaningful
    once averaged over >=1 actual unit cell. Each edge is clipped to
    max(h, a_lattice) before the volume ratio -- see WP9_REPORT.md and
    composition_field's matching xi floor below."""
    return composition.formula_units_per_cell(hx, hy, a_lattice, hz=hz)


def composition_std(x_bar, N_f):
    """Random-substitutional-alloy mixing limit (SPEC §15.1): variance of
    the mean Ti-fraction over N_f independent B-site draws = x(1-x)/N_f.

    SUPERSEDED for x(r) field generation by composition_std_correlated
    below (SPEC §15.5 coordinator ruling, 2026-08-21) -- kept unchanged
    here (WP9's G13a/G-calibration/kappa_b-bracket callers still use it
    as documented working values, see WP9_REPORT.md); WP10's G14e case
    and all Phase C work use the correlation-volume formula instead."""
    return composition.composition_std(x_bar, N_f)


def composition_std_correlated(x_bar, xi_x, a_lattice=A_LATTICE_PZT_NM, t_eff=None):
    """SPEC §15.5 coordinator ruling (2026-08-21), BINDING for WP10's G14e
    and all Phase C work -- supersedes WP9's grid-cell N_f flooring (which
    the WP9_REPORT.md itself found unusably large, ~0.50, comparable to the
    entire [0,1] range): the random-mixing statement is about the mean
    over a CORRELATION VOLUME, not a grid cell --
        std(x_bar over xi_x) = sqrt( x(1-x) * a^3 / (xi_x^2 * t_eff) )
    with t_eff=xi_x by default (isotropic correlation volume) and a the
    pseudocubic lattice constant. xi_x is the physical knob (processing/
    clustering scale) -- SPEC: "scan it, never the std independently".
    Worked example in the ruling: x=0.48, a=0.405nm, xi_x=4nm -> std~0.016
    (this module's A_LATTICE_PZT_NM=0.406nm gives 0.0162, matching to the
    ruling's own quoted precision)."""
    return composition.composition_std_correlated(x_bar, xi_x, a_lattice, t_eff=t_eff)


def composition_field(grid, x_bar, std, xi_x=None, seed=0, a_lattice=A_LATTICE_PZT_NM):
    """x(r) = x_bar + a zero-mean fluctuation of the given std (SPEC §15.1
    bullet 3: amplitude computed via composition_std, NOT free), via
    disorder.random_field_gaussian's spectral Gaussian filter (the reused
    "master-grid Gaussian-filter machinery", already master-grid/resample-
    capable for future multi-resolution use).

    xi_x: OPTIONAL clustering length ON TOP of a physical floor at
    a_lattice (one unit cell) -- the field is never smoothed at less than
    one lattice constant, matching formula_units_per_cell's floor (the
    random-mixing std formula already assumes averaging over >=1 formula
    unit, so the field itself must not vary faster than that scale).
    xi_x<=a_lattice (including the None/0 default) uses the floor alone,
    i.e. the "no EXTRA clustering beyond one unit cell" limit."""
    return composition.composition_field(grid, x_bar, std, a_lattice, xi_x=xi_x, seed=seed)


# --------------------------------------------------------------------------
# ch3: charged-defect random field (SPEC §15.1) -- generalizes the constant
# E_ext 3-vector to a field E_d(r), reusing disorder.random_field_gaussian
# per component (same spectral Gaussian filter machinery as
# make_random_strain), independently seeded.
#
# SHIM (SPEC S36 stage 2B, module docstring): this function never depended
# on any PZT-specific number, so it shims onto quenched_fields.py's
# declared-dial primitive instead of composition.py.
# --------------------------------------------------------------------------
def defect_field(grid, rms, xi, seed):
    """Returns (3,Ny,Nx): a field-valued E_ext, consumed by solver.py's
    (WP9-generalized) _reshape_vec3 wherever a constant E_ext 3-vector used
    to be the only option."""
    return np.stack([quenched_fields.gaussian_random_field(grid, rms, xi, seed + i) for i in range(3)])
