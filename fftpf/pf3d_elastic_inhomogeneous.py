"""SPEC V2-03 C1: FULL-TENSOR inhomogeneous-C(r) elasticity for bulk3d --
the 3D twin of elastic.py's WP10 elastic SPM (Wang et al., Acta Mater 61
(2013) 7591, Appendix A), generalized from that module's CUBIC-field C(r)
to an arbitrary C_ijkl(r).

WHAT IS REUSED vs REBUILT (bulk3d.py's own section discipline):
  - bulk3d.build_Omega_3d / bulk3d._rfftn/_irfftn / bulk3d.eigenstrain_3d /
    bulk3d._resolve_Q_field_3d / bulk3d.elastic_force_explicit_3d: REUSED
    UNCHANGED. The Green's operator is built from C0 ALONE (Appendix A Eq
    A9/A10 -- the reference tensor, never the local c(r)), so it is exactly
    the homogeneous module's own operator, and Q resolution/eigenstrain/
    force are C-agnostic.
  - elastic._green_delta_eps: REUSED, with bulk3d's 3-axis transform pair,
    exactly as bulk3d.elastic_solve_3d passes it. Same function, a
    DIFFERENT (C(r)-dependent) source per iterate -- elastic.py's own
    elastic_solve_spm makes the identical reuse in 2D.
  - elastic.contract_C (full-tensor einsum, constant C0) for the zero-order
    iterate, elastic.contract_C_cubic / elastic.contract_S_cubic for the
    CUBIC-input path: REUSED so that this module's cubic path does exactly
    the arithmetic elastic.py does (gates/core K2 `elastic_spm_nz1_vs_2d`
    is a roundoff-level anchor and would drift if it were rewritten here).

THE FIXED POINT (derived here, not copied). Mechanical equilibrium is
div sigma = 0 with sigma(r) = C(r):(ebar + de(r) - eps*(r)) and
C(r) = C0 + dC(r). Splitting the stiffness and using that ebar is uniform
(div(C0:ebar) = 0):

    div[C0:de] = div[ C0:eps* - dC:(ebar + de - eps*) ]

and elastic._green_delta_eps(tau) is by construction the operator returning
the periodic, zero-mean de that solves div(C0:de) = div(tau) (read off its
homogeneous caller: tau = C:eps* there gives exactly the WP2 solution). So
the inner fixed point at FIXED ebar is

    tau^(n) = C0:eps* - dC:( ebar + de^(n-1) - eps* )                 (*)
    de^(n)  = green( tau^(n) ),        de^(0) = green( C0:(eps* - ebar) )

which this module evaluates in elastic.py's own algebraically-identical
grouping tau^(n) = C(r):(eps* - ebar) - dC(r):de^(n-1): the two differ by
the SPATIALLY UNIFORM tensor C0:ebar, and a uniform source has zero
divergence (its only nonzero spectral bin is k=0, which the operator zeroes
anyway), so it cannot change de. NOTE THE SIGN: the dC term enters (*) with
a MINUS, because green() inverts div(C0:de) = div(tau) and the dC stress
moves to the right-hand side. dC = 0 makes tau^(n) the homogeneous WP2
source and the iteration converges at n=1 -- SPEC §16's "zero order =
existing homogeneous solver", here bulk3d.elastic_solve_3d.

THE OUTER LOOP (ebar). k0="stress_free": <sigma> = 0 gives
<C>:ebar = <C:eps*> - <C:de>, i.e.

    ebar = <C(r)>^-1 : ( <C(r):eps*(r)> - <C(r):de(r)> )

with <C(r)> the TRUE volume mean of the local stiffness -- NEVER the
Green's-operator reference C0, even when C0_override moves C0 away from
that mean. That is elastic.py's load-bearing WP10 lesson (setup_elastic_spm's
docstring: using S0 = C0^-1 there leaves the outer residual stuck at ~1e-2
forever), and it is why this module stores C_MEAN separately from C0.
<C:eps*> is P-fixed, so it is computed once per solve, not per outer
iteration. k0="clamped": ebar = 0 identically (bulk3d's own Dirichlet
convention), so there is NO outer update at all -- the mean stress is
generally nonzero by construction and is REPORTED (resid_outer), never
driven to zero.

C(r) INPUT, three forms (SPEC V2-03 C1), all resolved to ONE stored field:
  (i)   {"C11": ., "C12": ., "C44": .} -- cubic fields, each scalar or
        (Nz,Ny,Nx). elastic.py's 2D contract.
  (ii)  {label_value: (3,3,3,3), ...} -- a per-label dict of FULL tensors
        resolved against `labels` (the polycrystal contract of V2-05:
        rotated grains), the same np.where-chained idiom
        bulk3d._resolve_Q_field_3d uses for Q.
  (iii) a full (3,3,3,3,Nz,Ny,Nx) field.

CONTRACTION PATH, and why it is not a free choice: form (i) is cubic in the
global frame BY CONSTRUCTION, so C:e is elastic.contract_C_cubic (15
operations, and the same arithmetic elastic.py does). Forms (ii)/(iii) are
contracted with the FULL einsum "ijkl...,kl...->ij..." -- the cubic closed
form is used ONLY for form (i), never inferred from the numbers, because a
rotated cubic grain is NOT cubic in the global frame and the closed form
would silently drop its off-cubic components. Setup records which path is
in force (el["cubic"]) and, for form (i), CHECKS the closed form against the
full einsum on this field (_verify_cubic_path_3d -- bulk3d's own
_verify_voigt_mapping_3d discipline: a factor-of-2 shear claim is never
asserted). gates/core K2 `elastic_spm_fulltensor_equals_cubic` pins the two
paths against each other on the same physics.

MEMORY. The resolved C(r) is stored as FULL tensor components, (3,3,3,3,
Nz,Ny,Nx) float64 = 648 bytes per grid point: 21 MB at 32^3, 72 MB at 48^3,
170 MB at 64^3. The 81 slots hold only 21 independent numbers (36 before
major symmetry); they are kept in full for exactly the reason elastic.py's
own module docstring gives for storing (3,3,...) strains -- the Voigt
shear-factor bookkeeping falls out of the einsum instead of being
hand-coded. dC(r) is a SECOND such field, built only on the general path
(forms (ii)/(iii)); the cubic path keeps three (Nz,Ny,Nx) dC fields
instead, so its total is 648 + 24 bytes/point rather than 1296. Callers
that need 64^3 with a general C(r) should expect ~0.4 GB of resident
elastic state.

DISPATCH. bulk3d.elastic_solve_3d dispatches here when el.get("spm") is
set (its one hook, a lazy import); bulk3d.elastic_force_3d,
total_energy_3d, total_force_3d, _explicit_N_3d and the steppers are
therefore transparent already -- they only ever go through those two.
An el built by bulk3d.setup_elastic_3d has no "spm" key and is
byte-identical to before (gates/core K2 `elastic_spm_off_byte_3d`).

NOT batch-safe (no ensemble axis) -- the same documented scope limit
elastic.elastic_solve_spm carries.

OPT-IN CG (setup_elastic_spm_3d(..., method="cg")): the same equations
solved by preconditioned conjugate gradients (elastic._spm_cg_elastic, with
this module's contraction and 3-axis Green apply) -- no C0 condition,
O(sqrt(contrast)) iterations, ebar solved inside the one CG system under
"stress_free", ensemble axis supported on both C(r) paths.
docs/CG_SPM.md. The default method="basic" is untouched.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from fftpf import pf3d_engine as bulk3d, elastic_base as elastic, spectral_operators as operators, units

# The 6 independent components of a symmetric 3x3 tensor. Same SET as
# elastic._SYM6, deliberately in the (23,13,12) shear order the 6x6 assembly
# below reads most naturally; nothing here is index-compatible with that
# module's own storage order, so it is a local constant, not an import.
_SYM6 = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

_CUBIC_KEYS = ("C11", "C12", "C44")


# --------------------------------------------------------------------------
# C(r) resolution -- the three SPEC V2-03 C1 input forms -> one stored field
# --------------------------------------------------------------------------
def cubic_tensor_field(C11, C12, C44, shape):
    """The cubic C_ijkl of elastic.build_C_tensor, with (Nz,Ny,Nx)-shaped
    (or scalar) coefficients: shape (3,3,3,3)+shape.

    elastic.build_C_tensor itself cannot do this -- it multiplies a (3,3,3,3)
    einsum basis by the scalar coefficients, which does not broadcast against
    a coefficient FIELD -- so the SAME formula
    (C12*d_ij*d_kl + C44*(d_ik*d_jl + d_il*d_jk), then C11-C12-2*C44 added on
    the iiii slots) is written out slot by slot instead, filling one
    preallocated array. Written this way and not as three broadcast products
    because each of those would allocate a full (3,3,3,3)+shape temporary
    (3 x 72 MB at 48^3). Agreement with elastic.build_C_tensor at constant
    coefficients is checked by _verify_cubic_path_3d at every setup."""
    C11f, C12f, C44f = (np.broadcast_to(np.asarray(v, dtype=float), shape)
                        for v in (C11, C12, C44))
    out = np.zeros((3, 3, 3, 3) + tuple(shape))
    for i in range(3):
        for k in range(3):
            out[i, i, k, k] += C12f                 # C12 * d_ij d_kl
    for i in range(3):
        for j in range(3):
            out[i, j, i, j] += C44f                 # C44 * d_ik d_jl
            out[i, j, j, i] += C44f                 # C44 * d_il d_jk
    for i in range(3):
        out[i, i, i, i] += (C11f - C12f - 2 * C44f)
    return out


def _resolve_C_field_3d(C, labels, shape):
    """C -> (C_field (3,3,3,3)+shape, cubic_triple or None, form).

    form (i)  dict with EXACTLY the keys C11/C12/C44 -> cubic fields, and the
              triple is returned so the closed-form contraction can be used;
    form (ii) any other dict -> {label_value: (3,3,3,3)} resolved against
              `labels` by the np.where chain bulk3d._resolve_Q_field_3d uses
              ("WHOLE-CELL assignment... exact at EVERY cell by
              construction"; every label value in `labels` needs an
              entry, ValueError otherwise), cubic_triple None;
    form (iii) an array of shape (3,3,3,3)+shape, cubic_triple None."""
    if isinstance(C, dict) and set(C) == set(_CUBIC_KEYS):
        C11, C12, C44 = (C[k] for k in _CUBIC_KEYS)
        for name, v in zip(_CUBIC_KEYS, (C11, C12, C44)):
            s = np.shape(v)
            if s not in ((), tuple(shape)):
                raise ValueError(f"setup_elastic_spm_3d: cubic field {name} must be a scalar "
                                 f"or have shape {tuple(shape)}, got {s}")
        return cubic_tensor_field(C11, C12, C44, shape), (C11, C12, C44), "cubic_fields"

    if isinstance(C, dict):
        if any(isinstance(k, str) for k in C):
            raise ValueError("setup_elastic_spm_3d: a dict C with string keys is read as the "
                             f"cubic-field form and must have exactly {list(_CUBIC_KEYS)}; got "
                             f"{sorted(map(str, C))}. A per-label tensor dict is keyed by the "
                             "label VALUES in `labels`.")
        if labels is None:
            raise ValueError("setup_elastic_spm_3d: C given as a per-label dict but no "
                             "`labels` array supplied")
        if np.shape(labels) != tuple(shape):
            raise ValueError(f"setup_elastic_spm_3d: labels.shape {np.shape(labels)} "
                             f"!= grid shape {tuple(shape)}")
        field = np.zeros((3, 3, 3, 3) + tuple(shape))
        covered = np.zeros(tuple(shape), dtype=bool)
        for label_value, Cl in C.items():
            Cl = np.asarray(Cl, dtype=float)
            if Cl.shape != (3, 3, 3, 3):
                raise ValueError(f"setup_elastic_spm_3d: C[{label_value!r}] must be a "
                                 f"(3,3,3,3) tensor, got {Cl.shape}")
            mask = (np.asarray(labels) == label_value)
            covered |= mask
            field = np.where(mask, Cl[..., None, None, None], field)
        if not covered.all():
            # An unlisted label would keep the zeros the chain starts from: a
            # silent C=0 void, not an error the solve could ever report.
            raise ValueError(f"setup_elastic_spm_3d: labels "
                             f"{np.unique(np.asarray(labels)[~covered]).tolist()} occur in "
                             f"`labels` but have no entry in the per-label C dict (keys "
                             f"{list(C)}); every region needs its own (3,3,3,3) tensor")
        return field, None, "label_tensors"

    field = np.asarray(C, dtype=float)
    if field.shape != (3, 3, 3, 3) + tuple(shape):
        raise ValueError(f"setup_elastic_spm_3d: C must be a cubic-field dict, a per-label "
                         f"tensor dict, or an array of shape {(3, 3, 3, 3) + tuple(shape)}; "
                         f"got shape {field.shape}")
    return field, None, "tensor_field"


def _verify_tensor_symmetries_3d(C_field, tol=1e-10):
    """C_ijkl = C_jikl = C_ijlk (minor) and C_ijkl = C_klij (major), checked
    on the resolved field. NOT decoration: elastic._green_delta_eps's stated
    PRECONDITION is a SYMMETRIC source (it transforms 6 components, not 9),
    which holds iff C has the minor symmetries; and 0.5*sigma:e is the
    elastic energy whose P-derivative is -sigma:d(eps*)/dP only if C has the
    major symmetry -- i.e. a C without these is not merely inaccurate here,
    it silently breaks the force-energy consistency K1 checks. Relative to
    max|C|; raises, never degrades silently. Skipped for cubic-field input,
    which has all three by construction."""
    scale = max(float(np.max(np.abs(C_field))), 1e-300)
    for name, other in (("minor_ij", np.swapaxes(C_field, 0, 1)),
                        ("minor_kl", np.swapaxes(C_field, 2, 3)),
                        ("major", np.transpose(C_field, (2, 3, 0, 1)
                                               + tuple(range(4, C_field.ndim))))):
        d = float(np.max(np.abs(C_field - other))) / scale
        if d > tol:
            raise AssertionError(f"setup_elastic_spm_3d: C(r) violates its {name} symmetry "
                                 f"(relative max|C - C^T| = {d:.3e} > {tol:.1e})")


def _verify_cubic_path_3d(C_field, cubic, tol=1e-12):
    """For form-(i) input: the CLOSED-FORM contraction this module will use
    must reproduce the full-tensor einsum on the SAME stored field, on a
    random symmetric strain. bulk3d._verify_voigt_mapping_3d's own
    discipline, applied to the field rather than to a deck triple: the
    factor-of-2 shear convention of contract_C_cubic is the trap SPEC §5
    names, so it is never assumed -- and this simultaneously checks that
    cubic_tensor_field really built elastic.build_C_tensor's tensor."""
    rng = np.random.default_rng(20260916)
    e = rng.standard_normal((3, 3) + C_field.shape[4:])
    e = 0.5 * (e + np.swapaxes(e, 0, 1))
    s_cubic = elastic.contract_C_cubic(cubic[0], cubic[1], cubic[2], e)
    s_tensor = contract_C_field(C_field, e)
    scale = max(float(np.max(np.abs(s_tensor))), 1e-300)
    d = float(np.max(np.abs(s_cubic - s_tensor))) / scale
    if d > tol:
        raise AssertionError("setup_elastic_spm_3d: the cubic closed form disagrees with the "
                             f"full-tensor contraction of the stored C(r) (relerr {d:.3e})")
    return d


# --------------------------------------------------------------------------
# Contractions and the mean-stiffness inverse
# --------------------------------------------------------------------------
def contract_C_field(C_field, e):
    """sigma_ij(r) = C_ijkl(r) e_kl(r), the full rank-4 einsum. e/sigma
    (3,3,...) against C_field (3,3,3,3,...) -- elastic.contract_C's own
    pattern with the field axes carried on BOTH operands."""
    return np.einsum("ijkl...,kl...->ij...", C_field, e)


def _sym6_matrix(A):
    """The 6x6 matrix M with b_vec = M @ x_vec for b_ij = A_ijkl x_kl and
    SYMMETRIC x, where both vectors hold the six TENSOR components in _SYM6
    order (x23, x13, x12 are tensor strains, NOT engineering shears). The
    off-diagonal columns carry A_ijkl + A_ijlk because x_kl and x_lk are the
    same unknown -- writing that sum out is exactly what keeps this free of
    the Voigt factor-of-2 convention the shear slots would otherwise need."""
    M = np.empty((6, 6))
    for r, (i, j) in enumerate(_SYM6):
        for c, (k, l) in enumerate(_SYM6):
            M[r, c] = A[i, j, k, l] if k == l else A[i, j, k, l] + A[i, j, l, k]
    return M


def solve_sym_tensor(A, b):
    """The symmetric x solving A_ijkl x_kl = b_ij, for a (3,3,3,3) A and a
    symmetric (3,3) b -- i.e. the compliance of A applied to b, without ever
    forming a compliance tensor or touching Voigt notation. Used for the
    general-C(r) ebar update; the cubic path uses elastic.contract_S_cubic
    instead (same answer, elastic.py's own arithmetic), and setup checks the
    two against each other."""
    v = np.linalg.solve(_sym6_matrix(A), np.array([b[i, j] for i, j in _SYM6]))
    x = np.zeros((3, 3))
    for c, (k, l) in enumerate(_SYM6):
        x[k, l] = x[l, k] = v[c]
    return x


def _verify_mean_inverse_3d(C_mean, cubic_mean, tol=1e-9):
    """solve_sym_tensor must actually INVERT the mean stiffness (round trip
    C_mean : x == b on a random symmetric b), and -- when the input is cubic
    -- must agree with elastic.contract_S_cubic's closed-form cubic
    compliance. The second leg is the one that catches the engineering-shear
    factor of 2 (elastic.contract_S_cubic's own docstring names it); the
    first holds for ANY admissible C. Returns (round_trip_relerr,
    cubic_relerr or nan)."""
    rng = np.random.default_rng(20260917)
    b = rng.standard_normal((3, 3))
    b = 0.5 * (b + b.T)
    x = solve_sym_tensor(C_mean, b)
    back = np.einsum("ijkl,kl->ij", C_mean, x)
    rt = float(np.max(np.abs(back - b))) / max(float(np.max(np.abs(b))), 1e-300)
    if rt > tol:
        raise AssertionError(f"setup_elastic_spm_3d: <C(r)> is not invertible on symmetric "
                             f"strains (round-trip relerr {rt:.3e})")
    cub = float("nan")
    if cubic_mean is not None:
        s11, s12, s44v = units.cubic_stiffness_to_compliance(*cubic_mean)
        x_cub = elastic.contract_S_cubic(s11, s12, s44v, b)
        cub = float(np.max(np.abs(x_cub - x))) / max(float(np.max(np.abs(x))), 1e-300)
        if cub > tol:
            raise AssertionError("setup_elastic_spm_3d: the cubic closed-form compliance and "
                                 f"the generic symmetric solve disagree (relerr {cub:.3e})")
    return rt, cub


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
def setup_elastic_spm_3d(grid, C, Q, k0="stress_free", labels=None, eps_star_static=None,
                         C0_override=None, spm_tol=1e-10, spm_max_iter=None,
                         outer_tol=1e-10, outer_max_iter=50, method="basic",
                         cg_precond=True):
    """Build the elastic_params dict for an INHOMOGENEOUS C(r) (this
    module's docstring has the input forms, the fixed point and the memory
    figures).

    method: "basic" (default -- the fixed point below, bit-for-bit the
    pre-existing path and el dict) or "cg" (opt-in: the Galerkin/
    preconditioned conjugate-gradient solve of the same discrete equations,
    elastic._spm_cg_elastic -- no C0 condition, O(sqrt(contrast))
    iterations, one solve covers ebar under "stress_free"; docs/CG_SPM.md).
    cg_precond (cg only): True preconditions with the Green's operator of C0
    (the mean, or C0_override), False runs plain Galerkin CG with the
    identity-reference compatibility projection. spm_max_iter=None means
    100 for "basic" and elastic.CG_DEFAULT_MAX_ITER CG iterations for "cg".

    Everything except C is bulk3d.setup_elastic_3d's own contract,
    verbatim: Q is a tensor triple or a per-label dict resolved through
    `labels` (bulk3d._resolve_Q_field_3d -- TENSOR Q1111/Q1122/Q1212, never
    Voigt, SPEC §37.5), k0 is "stress_free" (default) or "clamped", and
    eps_star_static is an optional (3,3,Nz,Ny,Nx) static eigenstrain added to
    eps*(P) before the mechanical solve (SPEC V2-03 A2; its P-derivative is
    exactly zero, so no force formula changes).

    C0 (the Green's-operator REFERENCE) is the volume mean of C(r) unless
    C0_override is given -- a (3,3,3,3) tensor, or a (C11,C12,C44) triple
    for a cubic reference (elastic.setup_elastic_spm's own override form,
    used by its G14b C0-independence demonstration). The converged FIXED
    POINT is C0-independent, but whether the iteration reaches it is not:
    the basic scheme contracts only while C0 bounds every local stiffness,
    0 < C(r) < 2*C0 in each stiffness eigenmode, i.e. C0 >~ max C(r)/2. The
    default mean breaks that once a stiff minority phase exceeds ~2x the
    mean (an 8^3 box with one 2^3 inclusion of contrast 2.25 already
    diverges), and a C0_override near the midpoint 0.5*(min C + max C) of
    the moduli is then required. elastic_solve_spm_3d raises on a diverged
    solve and warns on one stopped short of its tolerances.
    <C(r)> (key "C_mean") is ALWAYS the true field mean and is what the ebar
    update inverts, override or no override -- the WP10 lesson, see the
    module docstring.

    Returned keys: "spm" (True -- what bulk3d.elastic_solve_3d dispatches
    on), "C_field"/"C_mean"/"C0"/"cubic"/"C_form", the cubic triples
    "C11"/"C12"/"C44" + "dC11"/"dC12"/"dC44" (form (i) only) or "dC_field"
    (forms (ii)/(iii)), "S_MEAN" (form (i) only), the Green machinery
    "Omega"/"K"/"zero_k"/"green_pre", the resolved "Q1111"/"Q1122"/"Q1212",
    the raw "C"/"Q"/"k0"/"labels"/"eps_star_static", the four
    tolerance/cap knobs and "Nx"/"Ny"/"Nz".

    NOT frozen after setup: elastic_solve_spm_3d(..., warm_start=True)
    writes "_warm_ebar"/"_warm_delta_eps" back into THIS dict, exactly as
    elastic.elastic_solve_spm does -- copy the dict if a caller needs a
    cold-start el."""
    if k0 not in ("stress_free", "clamped"):
        raise ValueError(f"setup_elastic_spm_3d: k0 must be 'stress_free' or 'clamped', "
                         f"got {k0!r}")
    spm_max_iter = elastic._resolve_spm_method(method, spm_max_iter, "setup_elastic_spm_3d")
    shape = (grid["Nz"], grid["Ny"], grid["Nx"])
    C_field, cubic, form = _resolve_C_field_3d(C, labels, shape)

    out = dict(spm=True, C=C, Q=Q, k0=k0, labels=labels, C_field=C_field,
               C_form=form, cubic=cubic is not None)

    # ---- C0 (reference) and <C(r)> (the ebar update's operator) ----------
    if cubic is not None:
        # Mirror elastic.setup_elastic_spm's own arithmetic exactly: the mean
        # of each cubic coefficient, then elastic.build_C_tensor of the three.
        # (Taking C_field.mean(...) instead would differ in the last bits and
        # would cost K2 `elastic_spm_nz1_vs_2d` its roundoff-level anchor.)
        C11_mean, C12_mean, C44_mean = (float(np.mean(v)) for v in cubic)
        C_mean = elastic.build_C_tensor(C11_mean, C12_mean, C44_mean)
        cubic_mean = (C11_mean, C12_mean, C44_mean)
        out["S_MEAN"] = units.cubic_stiffness_to_compliance(*cubic_mean)
    else:
        _verify_tensor_symmetries_3d(C_field)
        C_mean = C_field.mean(axis=(-3, -2, -1))
        cubic_mean = None

    if C0_override is None:
        C0 = C_mean
        cubic0 = cubic_mean
    elif np.shape(C0_override) == (3, 3, 3, 3):
        C0 = np.asarray(C0_override, dtype=float)
        cubic0 = None
    elif np.shape(C0_override) == (3,):
        cubic0 = tuple(float(v) for v in C0_override)
        C0 = elastic.build_C_tensor(*cubic0)
    else:
        raise ValueError("setup_elastic_spm_3d: C0_override must be a (3,3,3,3) tensor or a "
                         f"(C11,C12,C44) triple, got shape {np.shape(C0_override)}")

    # ---- dC(r) = C(r) - C0 ----------------------------------------------
    if cubic is not None:
        if cubic0 is None:
            raise ValueError("setup_elastic_spm_3d: a cubic-field C(r) needs a cubic "
                             "(C11,C12,C44) C0_override (a general tensor reference would "
                             "leave dC(r) non-cubic, which the closed-form path cannot carry)")
        out["C11"], out["C12"], out["C44"] = cubic
        out["C11_0"], out["C12_0"], out["C44_0"] = cubic0
        out["dC11"] = np.asarray(cubic[0], dtype=float) - cubic0[0]
        out["dC12"] = np.asarray(cubic[1], dtype=float) - cubic0[1]
        out["dC44"] = np.asarray(cubic[2], dtype=float) - cubic0[2]
        out["cubic_path_relerr"] = _verify_cubic_path_3d(C_field, cubic)
    else:
        out["dC_field"] = C_field - C0[..., None, None, None]

    out["C0"], out["C_mean"] = C0, C_mean
    out["mean_inverse_relerr"], out["mean_cubic_relerr"] = _verify_mean_inverse_3d(
        C_mean, cubic_mean)

    # ---- Green machinery from C0 ALONE (Appendix A Eq A9/A10) -----------
    Omega, K, zero_k = bulk3d.build_Omega_3d(grid, C0)
    out.update(Omega=Omega, K=K, zero_k=zero_k,
               green_pre=elastic.green_precompute(K, Omega))

    Q1111, Q1122, Q1212 = bulk3d._resolve_Q_field_3d(Q, labels, shape)
    if eps_star_static is not None and np.shape(eps_star_static) != (3, 3) + shape:
        raise ValueError(f"setup_elastic_spm_3d: eps_star_static must have shape "
                         f"{(3, 3) + shape}, got {np.shape(eps_star_static)}")
    out.update(Q1111=Q1111, Q1122=Q1122, Q1212=Q1212,
               eps_star_static=eps_star_static,
               spm_tol=spm_tol, spm_max_iter=spm_max_iter,
               outer_tol=outer_tol, outer_max_iter=outer_max_iter,
               Nx=grid["Nx"], Ny=grid["Ny"], Nz=grid["Nz"])
    if method == "cg":
        # Only a CG el gains keys: a "basic" el is the pre-existing dict, key for key.
        out.update(spm_method="cg", cg_precond=bool(cg_precond))
        if cg_precond:
            out["cg_ref_inv6"] = np.linalg.inv(_sym6_matrix(C0))   # C_ref = C0
        else:
            I_s = elastic.build_C_tensor(1.0, 0.0, 0.5)          # identity on sym. tensors
            Om_I, K_I, zk_I = bulk3d.build_Omega_3d(grid, I_s)
            out.update(cg_ref_inv6=None, cg_Omega=Om_I, cg_K=K_I, cg_zero_k=zk_I,
                       cg_green_pre=elastic.green_precompute(K_I, Om_I))
    return out


# --------------------------------------------------------------------------
# solve
# --------------------------------------------------------------------------
def _C0_hint_3d(el):
    """The C0_override suggestion elastic._spm_convergence_check prints for
    a diverged solve: the min/max midpoint of the moduli -- per cubic
    coefficient for form (i) (elastic.py's own hint), per tensor component
    for forms (ii)/(iii)."""
    if el["cubic"]:
        return elastic._cubic_C0_hint(el["C11"], el["C12"], el["C44"])
    return ("a (3,3,3,3) C0_override near the component-wise min/max midpoint of C(r), "
            "0.5*(C_field.min(axis=(-3,-2,-1)) + C_field.max(axis=(-3,-2,-1)))")


def elastic_solve_spm_3d(P, el, warm_start=True):
    """The SPM solve: inner fixed point on delta_eps(r) at fixed ebar (module
    docstring's (*), tol el["spm_tol"], cap el["spm_max_iter"]), wrapped in
    the outer ebar loop (k0="stress_free": <sigma> = 0 through the TRUE
    <C(r)>, tol el["outer_tol"], cap el["outer_max_iter"]; k0="clamped":
    ebar = 0, no outer update).

    Returns bulk3d.elastic_solve_3d's OWN keys -- eps_star, ebar, delta_eps,
    e, sigma -- plus the diagnostics elastic.elastic_solve_spm returns
    (n_inner, n_outer, resid_inner, resid_outer) and one this module adds
    (n_inner_total, every inner iteration of the call), so a gate or report
    can read iteration counts off the same dict. resid_outer is a CONVERGENCE
    number under "stress_free" and a REPORTED mean-stress level under
    "clamped" (where a nonzero <sigma> is the physics, not an error); either
    way it is that of the RETURNED state, also when outer_max_iter runs out.

    NON-CONVERGENCE is reported here, not left to the caller: the stepping
    path never reads these diagnostics (bulk3d._explicit_N_3d discards
    elastic_force_3d's `sol`), and the outer test alone is driven to ~1e-11
    even when delta_eps has diverged. So elastic._spm_convergence_check
    runs on the final state, before the warm cache is written: non-finite
    sigma, or resid_inner > elastic.SPM_DIVERGED_RESID, raises RuntimeError
    (with a C0_override suggestion); resid_inner > spm_tol, or resid_outer >
    outer_tol under "stress_free", warns (RuntimeWarning) and returns the
    result unchanged.

    warm_start=True (the default) is two pure starting-GUESS optimizations,
    neither of which can move the converged fixed point (the stopping
    criterion is unchanged): each outer iteration's inner solve starts from
    the previous outer iterate's delta_eps, and the converged (ebar,
    delta_eps) is cached in `el` itself for the NEXT call with the same
    el object -- exploiting that P moves little between time steps, exactly
    as elastic.elastic_solve_spm does (and with the same consequence: the
    FIRST call on a freshly built el is always a cold start). Pass
    warm_start=False for a history-free solve; a finite-difference check
    that wants strict determinism should do that, or drop the two cache
    keys between calls.

    An el built with setup_elastic_spm_3d(..., method="cg") is solved by
    _elastic_solve_spm_3d_cg instead (same returned keys; docs/CG_SPM.md)."""
    if el.get("spm_method") == "cg":
        return _elastic_solve_spm_3d_cg(P, el, warm_start=warm_start)
    eps_star = bulk3d.eigenstrain_3d(P, el["Q1111"], el["Q1122"], el["Q1212"])
    static = el.get("eps_star_static")
    if static is not None:
        if P.ndim == 5:                         # ensemble P: bulk3d.elastic_solve_3d's own alignment
            static = operators.ensemble_field(static, 2, eps_star.ndim, "eps_star_static")
        eps_star = eps_star + static            # V2-03 A2, elastic.py's WP3 contract
    shape = (el["Nz"], el["Ny"], el["Nx"])
    K, Omega, zero_k, green_pre = el["K"], el["Omega"], el["zero_k"], el["green_pre"]
    C0, cubic = el["C0"], el["cubic"]

    def contract_C(e):
        """sigma = C(r):e -- the closed form for form-(i) input (checked
        against the einsum at setup), the full einsum otherwise."""
        if cubic:
            return elastic.contract_C_cubic(el["C11"], el["C12"], el["C44"], e)
        return contract_C_field(el["C_field"], e)

    def contract_dC(e):
        if cubic:
            return elastic.contract_C_cubic(el["dC11"], el["dC12"], el["dC44"], e)
        return contract_C_field(el["dC_field"], e)

    def green(tau):
        return elastic._green_delta_eps(tau, K, Omega, zero_k, shape,
                                        fwd=bulk3d._rfftn, inv=bulk3d._irfftn, pre=green_pre)

    def inner_solve(ebar, guess=None):
        driver = eps_star - ebar[..., None, None, None]
        if guess is None or guess.shape != eps_star.shape:
            d_eps = green(elastic.contract_C(C0, driver))   # zero order: the C0-only solve
        else:
            d_eps = guess                                   # warm start
        n_inner, resid = 0, 0.0
        for n_inner in range(1, el["spm_max_iter"] + 1):
            d_eps_new = green(contract_C(driver) - contract_dC(d_eps))
            denom = float(np.max(np.abs(d_eps_new)))
            num = float(np.max(np.abs(d_eps_new - d_eps)))
            resid = num / denom if denom > 0 else 0.0
            d_eps = d_eps_new
            if resid <= el["spm_tol"]:
                break
        return d_eps, n_inner, resid

    def mean_compliance(b):
        """<C(r)>^-1 : b. The cubic closed form where it applies (elastic.py's
        own arithmetic), the generic symmetric solve otherwise; setup checked
        that the two agree."""
        if cubic:
            return elastic.contract_S_cubic(*el["S_MEAN"], b)
        return solve_sym_tensor(el["C_mean"], b)

    clamped = (el["k0"] == "clamped")
    ebar_warm = el.get("_warm_ebar") if warm_start else None
    d_eps_warm = el.get("_warm_delta_eps") if warm_start else None
    if clamped:
        ebar = np.zeros((3, 3))
    elif ebar_warm is not None and ebar_warm.shape == (3, 3):
        ebar = ebar_warm.copy()
    else:
        ebar = eps_star.mean(axis=(-3, -2, -1))

    d_eps, n_inner, resid_inner = inner_solve(ebar, guess=d_eps_warm)
    # n_inner is elastic.py's own diagnostic -- the LAST inner solve's count,
    # which after the first outer iteration is warm-started and so is small by
    # construction. n_inner_total sums every inner iteration of this call and
    # is the one that measures cost/conditioning (gates/core K4
    # `elastic_spm_residual_3d` reports it against contrast and Zener ratio).
    n_inner_total = n_inner

    # <C(r):eps*(r)> is P-fixed: computed once, not inside the outer loop.
    sigma0_mean = contract_C(eps_star).mean(axis=(-3, -2, -1))
    n_outer, outer_resid = 1, float("inf")
    exhausted = False
    for n_outer in range(1, (1 if clamped else el["outer_max_iter"]) + 1):
        e = ebar[..., None, None, None] + d_eps - eps_star
        sigma = contract_C(e)
        sigma_mean = sigma.mean(axis=(-3, -2, -1))
        outer_resid = (float(np.max(np.abs(sigma_mean)))
                       / max(float(np.max(np.abs(sigma))), 1e-300))
        if clamped or outer_resid <= el["outer_tol"]:
            break                                  # clamped: <sigma> is REPORTED, not driven
        ebar = mean_compliance(sigma0_mean - contract_C(d_eps).mean(axis=(-3, -2, -1)))
        d_eps, n_inner, resid_inner = inner_solve(ebar, guess=d_eps)
        n_inner_total += n_inner
    else:
        # outer_max_iter exhausted without the break: the outer_resid above
        # was measured BEFORE the last ebar/d_eps update, i.e. it belongs to
        # a state this function does not return.
        exhausted = True

    e = ebar[..., None, None, None] + d_eps - eps_star
    sigma = contract_C(e)
    if exhausted:
        # Report the residual of the state actually returned. A converged (or
        # clamped) solve keeps the value it broke on -- same ebar/d_eps, so
        # the same sigma and the same number -- and never takes this branch.
        outer_resid = (float(np.max(np.abs(sigma.mean(axis=(-3, -2, -1)))))
                       / max(float(np.max(np.abs(sigma))), 1e-300))
    elastic._spm_convergence_check("bulk3d_spm.elastic_solve_spm_3d", sigma, n_inner,
                                   resid_inner, n_outer, outer_resid, el,
                                   check_outer=not clamped, c0_hint=lambda: _C0_hint_3d(el))
    if warm_start:
        el["_warm_ebar"] = ebar
        el["_warm_delta_eps"] = d_eps
    return dict(eps_star=eps_star, ebar=ebar, delta_eps=d_eps, e=e, sigma=sigma,
                n_inner=n_inner, n_inner_total=n_inner_total, n_outer=n_outer,
                resid_inner=resid_inner, resid_outer=outer_resid)


# --------------------------------------------------------------------------
# method="cg": the Galerkin / preconditioned CG solve (elastic.py's section
# comment above _spm_cg_elastic has the formulation; docs/CG_SPM.md the
# benchmark). This module supplies only the 3D pieces: its own C(r)
# contraction, the 3-axis Green apply, and C_ref^-1 through the
# Voigt-factor-free 6x6 symmetric solve.
# --------------------------------------------------------------------------
def _sym6_apply_inverse(inv6, b):
    """x with A_ijkl x_kl = b_ij for symmetric b of shape (3,3[,S]), given
    inv6 = inv(_sym6_matrix(A)) -- solve_sym_tensor's arithmetic, batched
    over a trailing ensemble axis."""
    x6 = np.tensordot(inv6, np.stack([b[i, j] for i, j in _SYM6]), axes=(1, 0))
    x = np.empty(np.shape(b))
    for c, (k, l) in enumerate(_SYM6):
        x[k, l] = x6[c]
        x[l, k] = x6[c]
    return x


def _elastic_solve_spm_3d_cg(P, el, warm_start=True):
    """elastic_solve_spm_3d for an el built with method="cg". Same
    eigenstrain / static / ensemble handling and the SAME returned keys:
    n_inner = n_inner_total = the CG iteration count, n_outer = 1 (no outer
    loop: under "stress_free" ebar is an unknown of the CG system and
    <sigma> = 0 one of its equations; under "clamped" ebar = 0 is fixed),
    resid_inner the CG relative residual, resid_outer the RETURNED state's
    max|<sigma>|/max|sigma| exactly as the basic solve reports it (a
    convergence number under stress_free, the reported mean-stress level
    under clamped). elastic._spm_convergence_check runs on the final state;
    the warm start uses the basic path's own "_warm_ebar"/"_warm_delta_eps"."""
    eps_star = bulk3d.eigenstrain_3d(P, el["Q1111"], el["Q1122"], el["Q1212"])
    static = el.get("eps_star_static")
    if static is not None:
        if P.ndim == 5:                         # ensemble P: bulk3d.elastic_solve_3d's own alignment
            static = operators.ensemble_field(static, 2, eps_star.ndim, "eps_star_static")
        eps_star = eps_star + static
    shape = (el["Nz"], el["Ny"], el["Nx"])
    cubic = el["cubic"]

    def contract_C(e):
        if cubic:
            return elastic.contract_C_cubic(el["C11"], el["C12"], el["C44"], e)
        return contract_C_field(el["C_field"], e)

    if el.get("cg_precond", True):
        K, Omega, zero_k, pre = el["K"], el["Omega"], el["zero_k"], el["green_pre"]
        inv6 = el.get("cg_ref_inv6")
        if inv6 is None:                       # a basic el switched by hand: C_ref = its C0
            inv6 = np.linalg.inv(_sym6_matrix(el["C0"]))
    else:
        K, Omega, zero_k, pre = el["cg_K"], el["cg_Omega"], el["cg_zero_k"], el["cg_green_pre"]
        inv6 = None                            # C_ref = identity

    def gamma(tau):
        return elastic._green_delta_eps(tau, K, Omega, zero_k, shape,
                                        fwd=bulk3d._rfftn, inv=bulk3d._irfftn, pre=pre)

    def mean_inv(m):
        return m if inv6 is None else _sym6_apply_inverse(inv6, m)

    clamped = (el["k0"] == "clamped")
    mshape = eps_star.shape[:-3]
    ebar_w = el.get("_warm_ebar") if warm_start else None
    d_w = el.get("_warm_delta_eps") if warm_start else None
    if d_w is not None and np.shape(d_w) != eps_star.shape:
        d_w = None
    if clamped:
        ebar0, ebar_fixed = None, np.zeros(mshape)
    else:
        ok = ebar_w is not None and np.shape(ebar_w) == mshape and d_w is not None
        ebar0, ebar_fixed = (ebar_w if ok else None), None
        d_w = d_w if ok else None
    ebar, d_eps, n_cg, resid, load = elastic._spm_cg_elastic(
        eps_star, contract_C, gamma, mean_inv, 3, el["spm_tol"], el["outer_tol"],
        el["spm_max_iter"], ebar_fixed=ebar_fixed, ebar0=ebar0, d_eps0=d_w)

    e = ebar[..., None, None, None] + d_eps - eps_star
    sigma = contract_C(e)
    sigma_mean_abs = np.abs(sigma.mean(axis=(-3, -2, -1)))
    outer_resid = float(np.max(sigma_mean_abs)) / max(float(np.max(np.abs(sigma))), 1e-300)
    floor = np.all(elastic._cg_outer_at_floor(sigma_mean_abs.max(axis=(0, 1)), load))
    elastic._spm_convergence_check("bulk3d_spm.elastic_solve_spm_3d[cg]", sigma, n_cg, resid,
                                   1, outer_resid, el, check_outer=not (clamped or floor),
                                   c0_hint=lambda: _C0_hint_3d(el), method="cg")
    if warm_start:
        el["_warm_ebar"] = ebar
        el["_warm_delta_eps"] = d_eps
    return dict(eps_star=eps_star, ebar=ebar, delta_eps=d_eps, e=e, sigma=sigma,
                n_inner=n_cg, n_inner_total=n_cg, n_outer=1,
                resid_inner=resid, resid_outer=outer_resid)
