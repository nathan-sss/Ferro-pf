"""SPEC S32 (WP23/W2): higher-order-gradient (H11) module -- the S3 lane
(SPEC S26.1's parked candidate, now built). New, standalone: solver.py is
NOT modified (S32 architecture ruling, S32.4 R2), so the S26.4(c) lean
zero-regression regime applies (spot-check g0 + G21a + G24b, no full-suite
rerun required for this module-only WP).

GATES: g25_verified_dt, g25a-d and g29e. All six import this module under
its pre-WP24 shim name, `from solvers import hog`, and this file's own
H11-specific functions still carry the *_hog suffix (build_operators_hog,
hog_energy_density, grad_energy_density_hog, total_energy_hog,
total_force_hog) -- so grepping "grad4" in gates/ finds nothing.
Every `*_hog` name is THIS module's own, under its pre-WP24 file name
(`hog.py`); `solvers/hog.py` is the permanent forwarding shim (SPEC S35) and
there is no second H11 implementation anywhere in the tree.

============================================================================
PHYSICS (SPEC S32.4 R1, page-verified against NC 2025 eq.3, pNC.7):
============================================================================
f_grad = sum_i { 0.5*G11*[(dPi/dx)^2+(dPi/dy)^2]  [existing solver.py G-menu,
         isotropic reduction, see build_operators_hog docstring]
       + 0.5*H11*[(d^2Pi/dx^2)^2 + (d^2Pi/dy^2)^2] }   i=1,2,3   <- NEW HERE

Component-wise (NOT the isotropic Laplacian-squared |grad^2 P|^2 SPEC
S26.1's line originally anticipated -- S32.4 R1 supersedes that wording,
page-verified against NC 2025's own eq.3, which sums (d^2Pi/dxj^2)^2 over
j=x,y,z, i.e. a SUM OF SQUARES of pure second partials, not a squared
Laplacian): in k-space this is H11*(kx^4+ky^4) PER COMPONENT, not H11*|k|^4
-- the two agree exactly ON-AXIS (ky=0 or kx=0) but differ by a factor of 2
at 45 degrees (kx=ky): (kx^4+ky^4)=k^4/2 there, vs (kx^2+ky^2)^2=k^4. This is
why G25b (gates/) requires a k-point ON THE GRID DIAGONAL -- it is the only
way to distinguish the two candidate forms numerically.

0.5 PREFACTOR (flagged, a deliberate, small deviation from NC2025's own
eq.3, which prints the coefficient WITHOUT a compensating 1/2): chosen so
that the k-space FORCE multiplier added to A11/A22/A33 is EXACTLY
H11*(kx^4+ky^4), matching S32.4 R1's literal wording and G25b's literal
"closed form ... + H11*(kx^4+ky^4)" check, with no stray factor of 2 --
mirrors this engine's OWN pre-existing G11 convention exactly (solver.py's
grad_energy_density already carries the SAME 0.5 prefactor on its G11 term,
and its own A11=G11*(kx^2+ky^2) diagonal multiplier already comes out with
no extra factor for the identical reason -- verified by direct calculus-of-
variations here, exactly parallel to solver.py's own G-term derivation).
Immaterial to any physics claim: H11 is UNSOURCED in both papers (S32.1
gap list) and DIALED here (params/nnbm_wang2026.toml [meta]) -- the
prefactor choice is absorbed into what numeric value gets dialed for H11,
not a deviation from any fitted/sourced number. Self-consistency (not
literal-eq.3-prefactor-matching) is what G25a's FD check verifies.

============================================================================
ARCHITECTURE (SPEC S32.4 R2): the k^4 term sits ENTIRELY in the IMPLICIT
linear operator, exactly like the existing G-gradient term (SPEC's own
split: bulk[+elastic] explicit, grad+depolarization implicit -- H11 is
linear in P, same class as G11, so it joins the SAME implicit family, not
a new explicit term). Consequence: solver.py's OWN build_Minv,
build_ou_operators, step, and step_langevin are ALL already generic over
the `ops` dict's A11/A22/A33/A12 entries (verified by reading: _lin_matrix
reads ONLY ops['A11'/'A22'/'A33'/'A12'/'D11'/'D22'/'D12'/'kappa'] plus the
3D-only slots it takes from ops.get(key, 0.0); grad_force reads ONLY
ops['A11'/'A22'/'A33'/'A12'] -- neither hard-codes how those arrays were
built) -- so build_operators_hog below, which returns an ops dict with
H11*(kx^4+ky^4) already added into A11/A22/A33, can be handed DIRECTLY to
solver.build_Minv/build_ou_operators/grad_force UNCHANGED and they compute
the exact combined (G+H11) implicit operator/force for free.
The build_Minv()/step() wrappers below exist for API self-containment and
discoverability (SPEC's own "bring its own step function", afe.py
precedent) -- NOT because the stepping math differs; they are thin,
one-line delegates to solver.build_Minv/solver.step. The matching
build_ou_operators()/step_langevin() wrappers were DELETED in WP28 P4 (SPEC
§38.1 B4 = REVIEW_arch_minimality.md F7: 0 references under either the
grad4 or the hog name, SPEC §32's science-exit protocol being deterministic
-- xi OFF, R4); a Langevin caller uses solver.build_ou_operators/
solver.step_langevin on these same ops directly, which is what the deleted
wrappers did.
SPEC §10 (OU-NEG): with G11 < 0 < H11 these ops make Lin(k) INDEFINITE --
genuinely negative near k*, which is the AFE stripe instability itself.
solver.build_ou_operators (and bulk3d.build_ou_operators_3d on the 3D ops
below) now keep those negative eigenvalues and the k* modes grow. Before
§10 the OU builders clipped them to 0, so every Langevin run on G < 0 ops
evolved WITHOUT the k* instability (the deterministic build_Minv/step path
never clipped and is unaffected). The film OU carries no H11 and refuses
G11 < 0 (film3d_langevin.build_ou_operators_film).

The ONE place that genuinely needs NEW math is ENERGY: solver.py's
grad_energy_density computes f_grad directly in REAL SPACE from the scalar
G11/G12/G44/G44p values (it never reads the A-arrays at all), so it has no
way to "pick up" the H11 addition automatically -- hog_energy_density below
supplies that missing piece, and grad_energy_density_hog/total_energy_hog
combine it with solver.py's own (unmodified) G-term energy.

Deck: params/nnbm_wang2026.toml (Wang et al., Nano Energy 156, 112129
(2026) SI Landau/C/Q table; NC 2025 Methods for grid/dx/BC; G11/H11 DIALED,
both papers publish zero gradient numerics -- see the deck's own [meta]).
"""
import os
from fftpf._locations import PARAMS_DIR as _DECK_DIR
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from fftpf import pf2d_engine as solver, elastic_base as elastic, pf3d_engine as bulk3d
from fftpf import spectral_operators as operators


# ==========================================================================
# Operator construction: solver.build_operators() + H11*(kx^4+ky^4) added
# to the diagonal multipliers A11/A22/A33 (SPEC S32.4 R2, verbatim recipe).
# ==========================================================================
def build_operators_hog(grid, G11, G12, G44, G44p, kappa, H11):
    """Isotropic first-gradient note (params/nnbm_wang2026.toml convention,
    SPEC S32.4 R3 "expressed through the existing G-menu isotropically"):
    NC 2025's own alpha1 term is a SINGLE isotropic coefficient (no
    G11-vs-G44 split, eq.3) -- the caller reaches this via G12=0,
    G44=G44p=0.5*G11 (WP6's own already-validated isotropic reduction,
    bto_LCC2005_paper.toml: "the UNIQUE isotropic case... A11=A22=A33=
    G110*k^2, A12=0"), reused verbatim here rather than re-derived.

    H11 is added identically to A11, A22, AND A33 (NOT just the in-plane
    P1/P2 slots) because NC2025 eq.3 sums i=1,2,3 with the SAME (d^2Pi/
    dxj^2)^2 structure for every component, including P3 -- the
    out-of-plane component gets the same higher-order-gradient treatment
    as P1/P2, unlike D(k) (depolarization), which is genuinely 2D/in-plane
    only (D33=0 identically, solver.build_operators's own convention,
    untouched here)."""
    ops = solver.build_operators(grid, G11, G12, G44, G44p, kappa)
    ops = dict(ops)  # shallow copy -- do not mutate solver.build_operators's return
    k4 = operators.hog_symbol(grid, 2)          # kx^4 + ky^4, the shared H11 symbol
    ops["A11"], ops["A22"], ops["A33"] = operators.add_hog(
        (ops["A11"], ops["A22"], ops["A33"]), H11, k4)
    ops["H11"] = H11  # stashed for provenance/diagnostics only; not read by solver.py
    return ops


# ==========================================================================
# H11 energy (NEW math -- see module docstring for the 0.5-prefactor note
# and its calculus-of-variations justification).
# ==========================================================================
def _second_partials(field2d, grid):
    """(d^2f/dx^2, d^2f/dy^2) via -kx^2*, -ky^2* in k-space then ifft --
    mirrors solver._spectral_grad_xy's pattern for FIRST partials (that
    function is private to solver.py and not reused here to avoid a
    private-API dependency; the rfft2/irfft2 idiom is duplicated exactly,
    same discipline as solver._lin_matrix's own "deliberately duplicates
    build_Minv's entries rather than sharing a helper" precedent)."""
    s = (grid["Ny"], grid["Nx"])
    fk = solver._rfft2(field2d)
    fxx = solver._irfft2(-(grid["KXr"] ** 2) * fk, s)
    fyy = solver._irfft2(-(grid["KYr"] ** 2) * fk, s)
    return fxx, fyy


def hog_energy_density(P, grid, H11):
    """f_hog = 0.5*H11*sum_i[(d^2Pi/dx^2)^2+(d^2Pi/dy^2)^2], i=1,2,3
    (component-wise, SPEC S32.4 R1) -- the H11-only piece; solver.py's
    grad_energy_density supplies the G-term separately (see
    grad_energy_density_hog)."""
    P1xx, P1yy = _second_partials(P[0], grid)
    P2xx, P2yy = _second_partials(P[1], grid)
    P3xx, P3yy = _second_partials(P[2], grid)
    return 0.5 * H11 * (P1xx**2 + P1yy**2 + P2xx**2 + P2yy**2 + P3xx**2 + P3yy**2)


def grad_energy_density_hog(P, grid, ops):
    """Total gradient energy = solver.py's own (unmodified) G-term [reads
    ops's G11/G12/G44/G44p SCALARS, unaffected by the A-array H11
    augmentation] + this module's H11 piece [reads ops['H11'], stashed by
    build_operators_hog]."""
    return solver.grad_energy_density(P, grid, ops) + hog_energy_density(P, grid, ops["H11"])


def total_energy_hog(P, alpha, ops, E_ext, grid, elastic_params=None, spm_tol=1e-10):
    """Mirrors solver.total_energy's (dens, tot) contract exactly, with
    grad_energy_density replaced by grad_energy_density_hog."""
    fb = solver.bulk_energy_density(P, alpha)
    fg = grad_energy_density_hog(P, grid, ops)
    fe = solver.elec_energy_density(P, ops, E_ext, spm_tol=spm_tol)
    dens = dict(bulk=float(fb.mean()), grad=float(fg.mean()), elec=float(fe.mean()))
    if elastic_params is not None:
        sol = elastic.elastic_solve(P, elastic_params)
        dens["elastic"] = float(elastic.elastic_energy_density(sol).mean())
    dens["total"] = sum(dens.values())
    tot = {k: v * grid["Volume"] for k, v in dens.items()}
    return dens, tot


def total_force_hog(P, alpha, ops, E_ext, elastic_params=None, spm_tol=1e-10):
    """Fb+Fg+Fe[+Fel]. Fg is JUST solver.grad_force(P, ops) -- ops's
    A11/A22/A33 already carry the H11 addition (build_operators_hog), so
    the FORCE side needs no new k-space math at all, only ENERGY did
    (grad_energy_density doesn't read the A-arrays)."""
    Fb = solver.bulk_force(P, alpha)
    Fg = solver.grad_force(P, ops)
    Fe = solver.elec_force(P, ops, E_ext, spm_tol=spm_tol)
    if elastic_params is None:
        return Fb, Fg, Fe, Fb + Fg + Fe
    Fel, _ = elastic.elastic_force(P, elastic_params)
    return Fb, Fg, Fe, Fel, Fb + Fg + Fe + Fel


# ==========================================================================
# Step functions -- thin delegates to solver.py (see ARCHITECTURE note
# above for why no new stepping math is needed). Kept as hog.py's own
# named entry points per SPEC's "bring its own step function" instruction
# (afe.py precedent) and for caller self-containment. The DETERMINISTIC pair
# only: the Langevin pair had 0 callers and was deleted in WP28 P4.
# ==========================================================================
def build_Minv(grid, ops, dt, L):
    """Delegates to solver.build_Minv -- the H11 sector needs no stepping math of its own (ARCHITECTURE note above)."""
    return solver.build_Minv(grid, ops, dt, L)


def step(P, alpha, ops, Minv, dt, L, E_ext, elastic_params=None):
    """Deterministic semi-implicit step, H11-augmented ops -- delegates to
    solver.step unchanged (SPEC S32's disorder axis is E_rand/xi=OFF, R4:
    'deterministic TDGL + declared small random IC', so this is the
    primary stepper for the WP23 science-exit runs)."""
    return solver.step(P, alpha, ops, Minv, dt, L, E_ext, elastic_params)


# ==========================================================================
# SPEC V2-03 A5 -- the SAME H11 sector on the bulk3d engine. TWO INDEPENDENT
# SOLVERS (SPEC Part II ruling (3)): nothing below is shared with the 2D
# functions above beyond the elementwise Landau menu bulk3d itself already
# reuses and the k-space symbol helpers every engine takes from
# solvers/operators.py (hog_symbol/add_hog, here at dim=3 on bulk3d's own
# (Nz,Ny,Nxr) rfftn grid); the second partials are taken with bulk3d's own
# transform pair.
#
# NC2025 eq.3 in 3D is the SAME sum of squares of PURE second partials, now
# over j = x,y,z:
#   f_hog = 0.5*H11*sum_i [ (d2Pi/dx2)^2 + (d2Pi/dy2)^2 + (d2Pi/dz2)^2 ]
# i.e. the k-space force multiplier is H11*(kx^4+ky^4+kz^4) PER COMPONENT --
# COMPONENT-WISE, never the isotropic (k^2)^2 = |k|^4. The two agree on any
# axis and differ by a factor 3 on the body diagonal kx=ky=kz
# (3*(k^2/3)^2 = k^4/3 vs k^4), which is why the K3 anchor
# `hog_dispersion_3d` puts a k-point THERE (the 2D item's own diagonal
# argument, one axis stronger).
#
# ARCHITECTURE, unchanged from 2D: the k^4 term is LINEAR in P, so it sits
# entirely in the implicit operator -- bulk3d.build_Minv_3d/build_Minv2_3d/
# build_ou_operators_3d/grad_force_3d read ops["A11"/"A22"/"A33"/...] and
# never hard-code how those arrays were built, so an ops dict from
# build_operators_hog_3d steps through bulk3d.step_3d/step_sbdf2_3d
# UNCHANGED and the k^4 term is implicit for free (no hook needed in
# bulk3d.py, and gates/core K2 `hog_h11_zero_degenerate_3d` pins H11=0 to
# the plain bulk3d path bit-for-bit). ENERGY is again the one piece that
# needs new math (bulk3d.grad_energy_density_3d computes f_grad in real
# space from the G scalars and never reads the A-arrays).
# ==========================================================================
def build_operators_hog_3d(grid, G11, G12, G44, G44p, kappa, H11, kappa0_override=None,
                           spm_warm=False):
    """bulk3d.build_operators_3d(...) + H11*(kx^4+ky^4+kz^4) added to the
    diagonal multipliers A11/A22/A33 -- the 3D mirror of
    build_operators_hog, on bulk3d's own half-spectrum grid. H11 goes into
    all three components for NC2025 eq.3's own reason (the sum runs over
    i=1,2,3 with the same structure), exactly as in 2D; the off-diagonals
    A12/A13/A23 and the whole D(k) block are untouched. H11 is stashed under
    "H11" for the energy helpers, as in 2D.

    kappa0_override / spm_warm are bulk3d.build_operators_3d's own two
    kappa(r)-SPM knobs, passed straight through (defaults identical, so a
    call without them is byte-identical to before). They used to be
    dropped, which left bulk3d's documented remedy for a DIVERGING
    electrostatic SPM -- kappa0_override near (kappa_min+kappa_max)/2, see
    bulk3d's module docstring difference (1) -- unreachable on this lane."""
    ops = bulk3d.build_operators_3d(grid, G11, G12, G44, G44p, kappa,
                                    kappa0_override=kappa0_override, spm_warm=spm_warm)
    ops = dict(ops)  # shallow copy -- do not mutate build_operators_3d's return
    k4 = operators.hog_symbol(grid, 3)          # kx^4 + ky^4 + kz^4, shared H11 symbol
    ops["A11"], ops["A22"], ops["A33"] = operators.add_hog(
        (ops["A11"], ops["A22"], ops["A33"]), H11, k4)
    ops["H11"] = H11  # stashed for provenance/diagnostics only; not read by bulk3d.py
    return ops


def _second_partials_3d(field3d, grid):
    """(d^2f/dx^2, d^2f/dy^2, d^2f/dz^2) of one real (Nz,Ny,Nx) field via
    -k^2* in k-space then ifft -- _second_partials's 3D mirror, using
    bulk3d's own transform pair (bulk3d.FFT_WORKERS, deliberately 1) exactly
    as bulk3d._spectral_grad_xyz does for FIRST partials."""
    s = (grid["Nz"], grid["Ny"], grid["Nx"])
    fk = bulk3d._rfftn(field3d)
    fxx = bulk3d._irfftn(-(grid["KXr"] ** 2) * fk, s)
    fyy = bulk3d._irfftn(-(grid["KYr"] ** 2) * fk, s)
    fzz = bulk3d._irfftn(-(grid["KZr"] ** 2) * fk, s)
    return fxx, fyy, fzz


def hog_energy_density_3d(P, grid, H11):
    """f_hog = 0.5*H11*sum_i[(d^2Pi/dx^2)^2+(d^2Pi/dy^2)^2+(d^2Pi/dz^2)^2],
    i=1,2,3 (component-wise, section comment above) -- the H11-only piece;
    bulk3d.grad_energy_density_3d supplies the G-term separately. The 0.5
    prefactor is the 2D module's own documented convention (module
    docstring), which is what makes the force multiplier exactly
    H11*(kx^4+ky^4+kz^4)."""
    P1xx, P1yy, P1zz = _second_partials_3d(P[0], grid)
    P2xx, P2yy, P2zz = _second_partials_3d(P[1], grid)
    P3xx, P3yy, P3zz = _second_partials_3d(P[2], grid)
    return 0.5 * H11 * (P1xx**2 + P1yy**2 + P1zz**2 + P2xx**2 + P2yy**2 + P2zz**2
                        + P3xx**2 + P3yy**2 + P3zz**2)


def grad_energy_density_hog_3d(P, grid, ops):
    """bulk3d's own (unmodified) G-term energy + this module's H11 piece
    (reads ops["H11"], stashed by build_operators_hog_3d)."""
    return bulk3d.grad_energy_density_3d(P, grid, ops) + hog_energy_density_3d(P, grid, ops["H11"])


def total_energy_hog_3d(P, alpha, ops, E_ext, grid, elastic_params=None):
    """Mirrors bulk3d.total_energy_3d's (dens, tot) contract exactly, with
    grad_energy_density_3d replaced by grad_energy_density_hog_3d."""
    fb = bulk3d.bulk_energy_density(P, alpha)
    fg = grad_energy_density_hog_3d(P, grid, ops)
    fe = bulk3d.elec_energy_density_3d(P, ops, E_ext)
    dens = dict(bulk=float(fb.mean()), grad=float(fg.mean()), elec=float(fe.mean()))
    if elastic_params is not None:
        sol = bulk3d.elastic_solve_3d(P, elastic_params)
        dens["elastic"] = float(bulk3d.elastic_energy_density_3d(sol).mean())
    dens["total"] = sum(dens.values())
    tot = {k: v * grid["Volume"] for k, v in dens.items()}
    return dens, tot


def total_force_hog_3d(P, alpha, ops, E_ext, elastic_params=None):
    """Fb+Fg+Fe[+Fel], bulk3d.total_force_3d's own return-arity convention.
    Fg is JUST bulk3d.grad_force_3d(P, ops) -- ops's A11/A22/A33 already
    carry the H11 addition, so the FORCE side needs no new k-space math at
    all (only ENERGY did), exactly as in 2D."""
    Fb = bulk3d.bulk_force(P, alpha)
    Fg = bulk3d.grad_force_3d(P, ops)
    Fe = bulk3d.elec_force_3d(P, ops, E_ext)
    if elastic_params is None:
        return Fb, Fg, Fe, Fb + Fg + Fe
    Fel, _ = bulk3d.elastic_force_3d(P, elastic_params)
    return Fb, Fg, Fe, Fel, Fb + Fg + Fe + Fel


# ==========================================================================
# Dispersion relation (SPEC S32 architecture ruling: "AFE stripe mechanism:
# G11<0, H11>0 => dispersion minimum at k*=sqrt(-G11/(2*H11))" -- the
# minimum along a coordinate axis; k_star's docstring has the diagonal
# global minimum of this component-wise form). Linearized
# about P=0: for THIS engine's f_bulk=alpha1*P^2+... convention (no 0.5
# prefactor -- bulk_force's own d/dP gives 2*alpha1*P at P=0, verified by
# this engine's OWN established P0=sqrt(-alpha1/(2*alpha11)) double-well
# identity used throughout, e.g. reports/W0_DEMO_NOTE.md), the single-mode
# relaxation rate is Lambda(k) = 2*alpha1 + G11*(kx^2+ky^2) + H11*(kx^4+ky^4)
# for the ISOTROPIC gradient reduction (A11=A22=A33=G11*(kx^2+ky^2), see
# build_operators_hog) -- SAME for every P-component under that reduction.
# G25b (gates/) verifies this exactly via a single-step closed-form test.
# ==========================================================================
def dispersion_rate(alpha1, G11, H11, kx, ky):
    """Lambda(k) = 2*alpha1 + G11*(kx^2+ky^2) + H11*(kx^4+ky^4) -- the
    linearized-about-P=0 single-Fourier-mode rate, isotropic-G reduction."""
    return 2.0 * alpha1 + G11 * (kx**2 + ky**2) + H11 * (kx**4 + ky**4)


def k_star(G11, H11):
    """k* = sqrt(-G11/(2*H11)), the minimum of dispersion_rate ALONG A
    COORDINATE AXIS (ky=0 or kx=0) -- the ON-AXIS minimum, not the global
    one. Because the H11 term is component-wise (kx^4+ky^4, not |k|^4),
    Lambda(k) separates into one G11*kj^2 + H11*kj^4 term per axis, each
    minimized at |kj| = k*: the GLOBAL minimum sits on the grid diagonal at
    (+-k*, +-k*), |k| = sqrt(2)*k*, where Lambda - 2*alpha1 = -G11^2/(2*H11)
    is twice as deep as on-axis. In 3D (build_operators_hog_3d,
    kx^4+ky^4+kz^4) the global minimum is on the body diagonal
    (+-k*, +-k*, +-k*), |k| = sqrt(3)*k*, three times as deep. So k* is
    each Cartesian component of the most unstable wavevector, and the
    on-axis stripe's wavenumber. Only meaningful (real, positive) for
    G11<0<H11, the AFE-stripe regime this module exists for."""
    if not (G11 < 0.0 and H11 > 0.0):
        raise ValueError(f"k_star requires G11<0<H11 (AFE-stripe regime); got G11={G11}, H11={H11}")
    return np.sqrt(-G11 / (2.0 * H11))


def lambda_star(G11, H11):
    """lambda* = 2*pi/k* (nm, same length unit as the grid): the period of
    the on-axis (coordinate-aligned) stripe, equivalently the period along
    each axis of the globally most unstable mode -- NOT 2*pi/|k| of that
    diagonal mode, which is lambda*/sqrt(2) in 2D, lambda*/sqrt(3) in 3D
    (see k_star)."""
    return 2.0 * np.pi / k_star(G11, H11)


# ==========================================================================
# Deck loader -- deliberately NOT routed through cases/deck_parser.py's
# load_params_toml/TOML_SECTIONS (that would require an additive edit to a
# shared file; afe.py's own load_pbzo_deck precedent DID extend
# TOML_SECTIONS, but this module keeps strictly to "new files only" scope
# instead -- a self-contained tomllib read, same manual-remapping STYLE as
# afe.py, just with no shared-file dependency at all).
# ==========================================================================
NNBM_TOML = os.path.join(_DECK_DIR, "nnbm_wang2026.toml")


def load_nnbm_deck(path=None):
    """Returns dict(alpha={a1..a123}, gradient={G110,G11_G110,...}, H11=,
    kappa_b=, elastic={C11,C12,C44}, Q={Q11,Q12,Q44}, sigma_max=,
    xi_erand_nm=, meta={...}) -- params/nnbm_wang2026.toml's own schema."""
    import tomllib
    with open(path or NNBM_TOML, "rb") as f:
        data = tomllib.load(f)
    landau = data["landau"]
    alpha = dict(a1=landau["alpha1"], a11=landau["alpha11"], a12=landau["alpha12"],
                 a111=landau["alpha111"], a112=landau["alpha112"], a123=landau["alpha123"])
    grad = data["gradient"]
    G110 = grad["G110"]
    gradient = dict(G11=G110 * grad["G11_G110"], G12=G110 * grad["G12_G110"],
                     G44=G110 * grad["G44_G110"], G44p=G110 * grad["G44P_G110"])
    elastic_c = data["elastic"]
    Qd = data["electrostriction"]
    Q = dict(Q11=Qd["Q11_phys"], Q12=Qd["Q12_phys"], Q44=Qd["Q44_phys"])
    meta = data.get("meta", {})
    return dict(alpha=alpha, gradient=gradient, H11=data["higher_order_gradient"]["H11"],
                kappa_b=data["electrostatics"]["permittivity"],
                elastic=dict(C11=elastic_c["C11"], C12=elastic_c["C12"], C44=elastic_c["C44"]),
                Q=Q, sigma_max=meta["disorder_sigma_max_deck"],
                xi_erand_nm=meta["disorder_xi_erand_nm"], meta=meta)
