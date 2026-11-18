"""SPEC S36 (WP25 Phase 2, stage 2B) -- composition.py: the GEOMETRY layer
for how composition varies in space, and what that means geometrically.
Answers "how does composition x(r) distribute in space, and what
geometric/mixing consequence does that have" -- x(r) statistics, Vegard
eigenstrain, and (new) generic multi-material deck mixing.

Material FORMULAS do NOT live here -- "what are material X's Landau/Q/C
coefficients at composition x" is solvers/materials.py's job (stage 2A,
SPEC S36.3: "材料公式不进这里"). This module only ever consumes a material
function as an opaque, already-array-aware callable (deck_field below); it
never contains a Haun-style polynomial-in-x formula of its own.

ABSORBS (SPEC S36.3, logic migrated + GENERALIZED -- lattice constant/Vegard
slope that were PZT module constants become plain arguments -- not
copy-pasted; solvers/pzt_field.py's own functions become thin
parameter-binding shims, SPEC S36.2 R4): composition_field/composition_std/
composition_std_correlated/vegard_eigenstrain_field/formula_units_per_cell.
Every one of these was ALREADY array-aware and took its numeric knobs as
arguments -- the only thing that was PZT-specific about them was the
DEFAULT value of a_lattice/dlna_dx (pzt_field.A_LATTICE_PZT_NM/DLNA_DX_PZT).
Those move to PRESET_PZT below, a NAMED preset rather than a hard-coded
default -- the SAME discipline SPEC S36.2 R3 applies to quenched_fields.py's
PRESET_PMN.

NEW (SPEC S36.3, the generalization the user asked for by name):
  mix_decks(fractions, decks, rule="linear") -- n-ary LINEAR deck mixing,
    the general form of what cases/b2a_pan2019_case.py hand-rolled for the
    BFO/BTO/STO ternary via solvers/composite.py's alpha_field (K=2,
    fractions=(1-L,L)) and alpha_field_multi (K-ary, ONE-HOT fractions
    derived from a discrete label field -- see this function's own
    docstring for why "linear mixing fed one-hot weights" is EXACTLY
    composite.py's own "whole-cell assignment", not an approximation of
    it). Verified bit-for-bit against cases/b2a_pan2019_case.py's own
    actual ternary configuration, gates/g29d_composition_equivalence.py.
  deck_field(x_field, material_fn, *args, **kwargs) -- apply an
    already-array-aware material function pointwise over a composition
    field x(r). This depends on stage 2A's own preserved array-awareness of
    solvers.materials.pzt_haun_x_landau_si/_Q/_elastic_si (the same
    functions solvers/pzt_field.py's landau_field/Q_field/elastic_C_field
    already call) -- a documented name for a pattern that module already
    relied on implicitly, made explicit and material-agnostic here.

EQUIVALENCE: gates/g29d_composition_equivalence.py.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from fftpf import disorder_fields as disorder

# ==========================================================================
# PRESET_PZT -- SPEC S36.2 R3 (the same discipline quenched_fields.py's
# PRESET_PMN documents): PZT's own composition-geometry numbers, as a named
# preset, not a hard-coded function default. Independent literals, matching
# pzt_field.A_LATTICE_PZT_NM/DLNA_DX_PZT (which stay defined there,
# UNCHANGED -- many existing gates/cases read pf.DLNA_DX_PZT/pf.
# A_LATTICE_PZT_NM directly, e.g. gates/g13b_field_fd_consistency.py,
# cases/wp11c_matrix.py -- both must keep working, SPEC S35 ruling (3)).
# ==========================================================================
PRESET_PZT = dict(
    # nm, pseudocubic lattice parameter near the MPB (Shirane, Hoshino &
    # Suzuki 1952 lineage, see pzt_field.A_LATTICE_PZT_NM's own provenance note).
    a_lattice=0.406,
    # per unit x, Vegard slope da_pc/dx / a_pc near x_bar~0.48 (same source,
    # see pzt_field.DLNA_DX_PZT's own provenance note).
    dlna_dx=-0.040,
)


def formula_units_per_cell(hx, hy, a_lattice, hz=None):
    """N_f = formula units per coarse-grain cell = cell volume / a_lattice^3
    (generalizes pzt_field.py's own function of the same name, SPEC S15.1:
    "compute from cell volume / a^3"). `a_lattice` is now a REQUIRED
    argument (was pzt_field.A_LATTICE_PZT_NM by default -- see PRESET_PZT).
    The 2D engine has no independent out-of-plane grid spacing to draw "cell
    volume" from, so -- absent any other basis -- the cell is treated as a
    CUBE of edge h (hz defaults to hx), a documented modeling choice, not a
    literature number. FLOORED per edge at a_lattice itself: composition is
    only statistically meaningful once averaged over >=1 actual unit cell
    (a literal h^3/a_lattice^3 with h<a_lattice would give an unphysical
    N_f<1); each edge is clipped to max(h, a_lattice) before the volume
    ratio."""
    if hz is None:
        hz = hx
    ex, ey, ez = max(hx, a_lattice), max(hy, a_lattice), max(hz, a_lattice)
    return (ex * ey * ez) / a_lattice**3


def composition_std(x_bar, N_f):
    """Random-substitutional-alloy mixing limit: variance of the mean
    composition over N_f independent B-site draws = x(1-x)/N_f. Already
    fully material-agnostic in pzt_field.py (no lattice constant or other
    material number ever entered this formula) -- relocated verbatim, no
    generalization needed beyond the move itself."""
    return float(np.sqrt(x_bar * (1.0 - x_bar) / N_f))


def composition_std_correlated(x_bar, xi_x, a_lattice, t_eff=None):
    """The mean over a CORRELATION VOLUME, not a grid cell: std(x_bar over
    xi_x) = sqrt(x(1-x)*a_lattice^3/(xi_x^2*t_eff)), t_eff=xi_x by default
    (isotropic correlation volume). `a_lattice` is now a REQUIRED argument
    (was pzt_field.A_LATTICE_PZT_NM by default -- see PRESET_PZT). xi_x is
    the physical knob (processing/clustering scale) -- scan IT, never the
    std independently."""
    if t_eff is None:
        t_eff = xi_x
    return float(np.sqrt(x_bar * (1.0 - x_bar) * a_lattice**3 / (xi_x**2 * t_eff)))


def composition_field(grid, x_bar, std, a_lattice, xi_x=None, seed=0):
    """x(r) = x_bar + a zero-mean fluctuation of the given std, via
    disorder.random_field_gaussian's spectral Gaussian filter (the NUMERICAL
    layer -- this function's own job is only to add the deterministic mean
    x_bar back on and floor the clustering length at one lattice constant,
    both physics/geometry decisions, R3's own layer boundary). `a_lattice`
    is now a REQUIRED argument (was pzt_field.A_LATTICE_PZT_NM by default --
    see PRESET_PZT). xi_x: OPTIONAL clustering length ON TOP of a physical
    floor at a_lattice (one unit cell) -- the field is never smoothed at
    less than one lattice constant, matching formula_units_per_cell's own
    floor (the random-mixing std formula already assumes averaging over
    >=1 formula unit). xi_x<=a_lattice (including the None/0 default) uses
    the floor alone."""
    xi_eff = max(xi_x or 0.0, a_lattice)
    field = disorder.random_field_gaussian(grid, rms=std, xi=xi_eff, seed=seed)
    return x_bar + field


def composition_field_3d(grid, x_bar, std, a_lattice, xi_x=None, seed=0):
    """SPEC V2-03 A6. 3D twin of composition_field: the SAME one-lattice-
    constant clustering floor and the SAME deterministic mean, now realized
    through disorder.random_field_gaussian_3d on a bulk3d.make_grid_3d grid --
    shape (Nz,Ny,Nx). Nothing in composition_std/composition_std_correlated/
    formula_units_per_cell changes for 3D: N_f is already a CELL VOLUME over
    a_lattice^3 and formula_units_per_cell takes the real hz as its fourth
    argument (`hz=None` only means "the 2D engine has no third spacing, treat
    the cell as a cube" -- see its own docstring), so a 3D caller passes
    formula_units_per_cell(hx, hy, a_lattice, hz) and gets the genuine cell
    volume, no cube assumption anywhere."""
    xi_eff = max(xi_x or 0.0, a_lattice)
    field = disorder.random_field_gaussian_3d(grid, rms=std, xi=xi_eff, seed=seed)
    return x_bar + field


def vegard_eigenstrain_field(x, x_bar, dlna_dx):
    """eps_chem = dlna_dx*(x-x_bar)*I3 -- isotropic chemical (Vegard) strain
    from a local composition deviation, ready for the existing
    eps_star_static path (disorder.assemble_eps_star_static's {xx,yy,zz,...}
    component-dict convention). `dlna_dx` is now a REQUIRED argument (was
    pzt_field.DLNA_DX_PZT by default -- see PRESET_PZT). x may be an
    (Ny,Nx) array or a scalar (matching shape returned) -- a scalar x=x_bar
    gives an exact all-zero field (degenerate-limit sanity check). SPEC
    V2-03 A6: a 3D (Nz,Ny,Nx) x needs NO change here and got none -- the body
    is a single shape-preserving elementwise expression, so it already
    returns a 3D component dict, which assemble_eps_star_static (itself
    shape-generic, WP4 S10e) then lifts to (3,3,Nz,Ny,Nx). Checked, not
    assumed: K2.generators_nz1_vs_2d compares the 3D-array call against the
    2D one."""
    dev = dlna_dx * (np.asarray(x, dtype=float) - x_bar)
    return dict(xx=dev, yy=dev, zz=dev)


# ==========================================================================
# NEW: n-ary linear deck mixing (SPEC S36.3).
# ==========================================================================
def mix_decks(fractions, decks, rule="linear"):
    """N-ary LINEAR mixing of K flat coefficient dicts: out[key] =
    sum_k fractions[k]*decks[k][key], over keys common to EVERY deck (a key
    missing from any one deck is silently DROPPED from the output, never
    zero-filled -- the same "absent beats a wrong zero" spirit SPEC S36.2
    R1 states for materials.py's own deck schema).

    `fractions`: K per-point weight fields (arrays, broadcastable against
    each other) or scalars, aligned with `decks` -- fractions[k] applies to
    EVERY key of decks[k] UNIFORMLY (one weight per whole deck, not a
    per-key weight -- matching how solvers/composite.py's own alpha_field/
    alpha_field_multi always move an entire material's coefficient set
    together, never one Landau term at a time). `decks`: K flat dicts
    (key -> scalar/array) -- typically ONE SECTOR of a materials.get()-style
    R1-schema deck (e.g. [d["landau"] for d in decks], since sectors can be
    independently absent per R1's own convention and "meta" is provenance
    TEXT a linear numeric rule has no business combining -- call this once
    per sector to mix full nested decks, that is a deliberate choice left
    to the caller, not automated here), or a bare alpha-coefficient dict
    like cases/b2a_pan2019_case.py's own material_decks() output.
    rule="linear" is the only rule implemented (ValueError otherwise).

    WHY THIS IS THE GENERAL FORM OF composite.py's DISCRETE mixing, not
    merely an alternative to it: feed fractions[k] = (labels==k).astype(
    float) (a ONE-HOT indicator field derived from a discrete K-ary label
    field, solvers.composite.make_label_field_multi's own output) and this
    function reproduces composite.alpha_field_multi's hard "whole-cell"
    selection EXACTLY, bit-for-bit, not merely to some tolerance: at every
    cell exactly one fractions[k] is 1.0 and every other is 0.0 (labels
    partition the domain), and 0.0*x/1.0*x/summing several exact zeros with
    one exact value are all EXACT IEEE-754 operations -- there is no
    rounding step where the two constructions could diverge. Similarly,
    fractions=(1-L, L) reproduces composite.alpha_field's own K=2 affine
    blend exactly. Both equivalences are verified numerically (not merely
    asserted) in gates/g29d_composition_equivalence.py, against
    cases/b2a_pan2019_case.py's actual ternary configuration."""
    if rule != "linear":
        raise ValueError(f"mix_decks: unknown rule {rule!r} (only 'linear' implemented)")
    K = len(decks)
    if len(fractions) != K:
        raise ValueError(f"mix_decks: {len(fractions)} fractions given for {K} decks")
    keys = set(decks[0])
    for d in decks[1:]:
        keys &= set(d)
    out = {}
    for key in keys:
        field = fractions[0] * decks[0][key]
        for k in range(1, K):
            field = field + fractions[k] * decks[k][key]
        out[key] = field
    return out


def deck_field(x_field, material_fn, *args, **kwargs):
    """Apply an already ARRAY-AWARE material function pointwise over a
    composition field x(r) -- literally `material_fn(x_field, *args,
    **kwargs)`. This is the generic form of what solvers/pzt_field.py's own
    landau_field(x,T)/Q_field(x)/elastic_C_field(x,...) already do for the
    Haun formula set specifically: stage 2A's own array-vectorization of
    solvers.materials.pzt_haun_x_landau_si/_Q/_elastic_si (x may be a
    scalar or an (Ny,Nx) array, same-shape output) is what makes evaluating
    a material function "pointwise over a field" require NO special-casing
    on the material function's side -- so this wrapper needs none either.
    Named and documented here, rather than left as an implicit convention
    each new x(r)-dependent material must rediscover, so the "material_fn
    must be array-aware for this to work" contract has ONE place it is
    written down. `*args`/`**kwargs` pass through unchanged (e.g. a
    temperature T=298.0 that is not itself a function of x(r))."""
    return material_fn(x_field, *args, **kwargs)
