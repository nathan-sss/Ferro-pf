"""film3d.py -- WP7a (SPEC.md §13.0 binding architecture + §13.1 scope/gates).

Tier C scaffolding: film electrostatics (short-circuit Poisson) + TDGL
bulk+grad for the TRUE thin-film geometry -- periodic in-plane (x1,x2),
NON-periodic thickness (x3). Elasticity is OPTIONAL and IS wired here
(WP7c, SPEC §13.3): pass `el=` (a film_elastic.setup_film_elastic output) to
explicit_N/step/step_sbdf2/probe_stable_dt/total_energy_3d/relax; `el=None`
is this file's original WP7a bulk+grad+elec path. GATES: G12a/G12b for the
el= path (`from solvers import film3d as f3`).

DISCRETIZATION (SPEC §13.0, binding): FFT in (x1,x2) [scipy.fft rfft2,
same half-spectrum convention as solver.py's WP4 §10b -- axis order (Ny,Nxr)]
TIMES second-order finite difference in x3. Every linear solve (the Poisson
equation, the implicit gradient operator) is PER-(k1,k2)-MODE BANDED
(tridiagonal), solved with a vectorized batched Thomas algorithm
(solve_tridiag_batched/_factored below) that loops only over the small Nz
axis and treats the ~Ny*Nxr modes as ordinary numpy broadcast batch axes --
no global 3D dense operator is ever built, matching SPEC's explicit ban.
Factorization (the O(Nz) elimination sweep) is precomputed once per
(dt, grid, kappa_b) via factor_tridiag_batched -- SPEC: "precomputed/
factorized where profitable" -- mirroring solver.py's build_Minv "invert
once" pattern.

FIELDS (SPEC §13.0): P (3 comps) lives in the film only; phi lives in the
film "between electrodes" (short circuit: phi=0 at both film surfaces).
Both P and phi are discretized on the SAME z NODE grid: Nz nodes spanning
z in [0, hf] inclusive, hz = hf/(Nz-1) (z=0 is the film/substrate
interface, matching Li 2002 Acta Mater's own coordinate origin, Fig. 2).
STACK LAYOUT NOTE: SPEC's grid description also lists a substrate stack
(hs=12dx) needed once u (WP7b) is added; that stack is NOT built here --
WP7a's own field list scopes P and phi to the film ALONE, and u isn't
solved yet, so there is nothing for a substrate grid to do in this WP.
Extending z downward past 0 (for WP7b) is additive, not a restructuring:
z=0 stays the interface either way.

GRID SPACING: dx = sqrt(G110/alpha0) (paper's own formula, WP6 Tier 2's
dx=0.4nm choice reused verbatim, per SPEC §13.0: "as in WP6 Tier 2"). The
paper's text writes "128dx x 128dx x 36dx" with ONE symbol dx for all three
axes; hf=20dx, hs=12dx (=32dx) leaves 4dx unreconciled against the stated
36dx total (WP6_REPORT.md already flagged this exact gap and could not
close it from the 3-page Letter's text alone). WP7a re-examined this
against Li 2002 Acta Mater (the cited film-mechanics methods paper) and did
NOT find a resolution either: that paper's OWN numerical section uses a
DIFFERENT grid convention for its own PbTiO3 example (Dx3/l0=0.5 while
Dx1/l0=Dx2/l0=1.0 -- an ANISOTROPIC z-spacing, with a DIFFERENT gradient-
coefficient ratio set than APL2006's isotropic one), so it cannot be
carried over to APL2006's BaTiO3 grid without more evidence than the
Letter provides. Since WP7a needs only the FILM extent (no substrate/
buffer yet), this project reads APL2006's "hf=20dx" at face value with an
ISOTROPIC dx (matching the single-symbol sentence and WP6 Tier 2's existing
choice): PAPER_NZ = 21 nodes (20 intervals of dx spanning hf exactly). The
4dx gap is deferred to WP7b, where the substrate stack is actually built
and a convention must be picked (or re-verified against Li 2002 more
carefully / the AIP SI if one exists) -- flagged, not guessed past.

BOUNDARY CONDITIONS (SPEC §13.0: "verify: expected natural dP/dx3=0"):
F = int [f_bulk + f_elas + f_elec + f_grad] dV with NO explicit surface or
interface energy term anywhere in this model (Li 2002 Acta Mater, Sec. 2:
"we ignore any possible surface and interface contributions to the free
energy as discussed in Refs. 24 and 25" [Binder 1981; Tilley & Zeks 1984 --
the classic extrapolation-length/surface-anisotropy papers]; APL2006 adds
none either). Varying F w.r.t. P_i and integrating f_grad's z-part by parts
over the FINITE range [0,hf] produces a genuine boundary term
G11*(dP_i/dz)*deltaP_i at z=0 and z=hf; since nothing else in F constrains
P at the surface (P is free there -- only the interior PDE governs it),
stationarity for ARBITRARY deltaP_i forces this term to vanish identically,
i.e. dP_i/dz=0 at both surfaces -- the classical "natural" boundary
condition of the calculus of variations. CONFIRMED (not assumed).

GRADIENT OPERATOR DERIVATION (isotropic G, paper Eq. 7: f_grad =
(G11/2)*sum_i |grad P_i|^2, i.e. A(k)=G11*k^2*I3 exactly -- WP6_REPORT.md
already proved this algebraically for bto_LCC2005_paper.toml's G-ratios,
reused verbatim). The z-part is discretized via a FACE-based finite-volume
functional (not a naive nodal Laplacian), so the implicit tridiagonal solve
is EXACTLY the gradient of a well-defined discrete energy (needed for
G10e's energy-monotonicity gate to test something that is actually true):
    F_grad_z = (G11/(2*hz)) * sum_{faces j=0..Nz-2} (P_{j+1} - P_j)^2
Differentiating w.r.t. each node P_j and dividing by the trapezoidal
quadrature weight at that node (qw: hz at interior nodes, hz/2 at the two
boundary nodes -- the standard finite-volume node weight, since a boundary
node's control volume is a half-cell) gives the FORCE operator applied
below (_grad_z_bands): a tridiagonal matrix, uniform diagonal 2*G11/hz^2 at
EVERY row (boundary included), uniform off-diagonal -G11/hz^2 EXCEPT the
outermost connections (row 0<->1 and row Nz-2<->Nz-1), which are DOUBLED to
-2*G11/hz^2 (a boundary node has only one neighboring face, so it carries
that face's full flux rather than splitting it as an interior node does).
This is the discrete-energy-consistent form of the "ghost-node reflection"
Neumann stencil; it happens to coincide with that classical construction
exactly, confirmed here by direct derivation rather than assumed.

ELECTROSTATIC FORCE SIGN (mirrors solver.py's elec_force convention,
"the 1/2 disappears"): F_elec = -(1/2) int P.E_d dV with E_d = L[P] for a
LINEAR map P->E_d (the Poisson solve is linear in P). L is self-adjoint
w.r.t. int P.E_d dV -- a standard electrostatic reciprocity argument, and
this docstring gives the film-geometry-specific proof since the BCs differ
from solver.py's periodic case: for two source fields P^(1),P^(2) with
resulting phi^(1),phi^(2) (same linear BVP, same BCs), int P^(1).E^(2) dV
= int phi^(2)*div(P^(1)) dV [integrate by parts once; the boundary term
vanishes because phi^(2)=0 exactly on both z-faces (Dirichlet) and the
in-plane periodic faces cancel pairwise] = int phi^(2)*kappa_b*lap(phi^(1))
dV [governing PDE] = -int kappa_b*grad(phi^(1)).grad(phi^(2)) dV
[integrate by parts again; same boundary-vanishing argument], which is
manifestly SYMMETRIC under (1)<->(2). Hence int P^(1).E^(2) = int P^(2).E^(1)
(self-adjoint), so delta F_elec/delta P_i = -(1/2)[L[P]]_i-(1/2)[L^T[P]]_i
= -[L[P]]_i = -E_d,i (no extra 1/2) -- verified analytically, and this is
also a NON-NEGATIVE quadratic cost (F_elec = +(1/2) int kappa_b|grad phi|^2
dV >= 0, same identity with (1)=(2)), consistent with G10b's zero-field
result being the electrostatic energy FLOOR, not just a stationary point
(see gates/g10c_bulk_limit.py's docstring for why this matters there).

DISCRETE NOTE: the source divergence B and returned depolarisation field
are quadrature adjoints. With trapezoidal nodal weights W and interior
Poisson equation A phi = -B P, E = W^{-1} B^T hz phi. Consequently
F_elec = -<P,E>_W/2 is non-negative and its discrete derivative is -E.
Odd spectral derivatives vanish on their own Nyquist axis. At the two
surface nodes this conjugate E3 is a one-cell potential difference, not a
second-order pointwise field. Use physical_electric_field_3d for pointwise
field diagnostics; do not substitute it into the energy or TDGL force.

Route A only (SPEC §13.0): electrostatic force is EXPLICIT (part of N(P),
alongside the bulk Landau force); the gradient operator is the ONLY
implicit piece. Route B (a monolithic per-mode (P,phi) banded solve) is
not built here -- it is authorized only after route A's measured stable dt
is reported (see reports/WP7A_REPORT.md's G10e entry).

SPEC V2-03 B4/B5 (anisotropic G11/G12/G44/G44' gradient, and the H11
higher-order gradient with its RULED implicit pentadiagonal z solve) are
ADDITIVE: `G` may be the scalar G11 this file has always taken, or a dict
{G11,G12,G44,G44p}; `H11` defaults to None. Omitting both is byte-identical
to every pre-V2-03 call. The full derivation -- discrete energies, the
natural boundary conditions they produce, the isotropic-reduction dispatch,
and the Nyquist convention for the odd-order mixed terms -- is in the
section comment above _grad_scalar_G, not repeated here.

Reuses solver.py's f_bulk (bulk_energy_density, sympy-lambdified;
bulk_force, solver.py's hand-written pow-free Horner form since WP28 T1,
symbolically verified against the same _F_BULK at import)
UNCHANGED -- both are already elementwise/broadcast-safe over ANY trailing
shape, so they work on this module's (3,Nz,Ny,Nx) P arrays with zero
modification, per SPEC's "reuse solver.py pieces where clean (sympy
f_bulk, params)".
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import scipy.fft as spfft

from fftpf import pf2d_engine as solver
from fftpf import mat_chen2007_decks as lp
from fftpf import spectral_operators as operators
from fftpf import input_validation as validate

FFT_WORKERS = -1


def _rfft2(x):
    return spfft.rfft2(x, axes=(-2, -1), workers=FFT_WORKERS)


def _irfft2(xk, s):
    return spfft.irfft2(xk, s=s, axes=(-2, -1), workers=FFT_WORKERS)


def _dz_2nd_order(fk, hz):
    """d/dz of an array along axis 0 (the z axis by this module's
    convention), 2nd-order accurate throughout: central differences in the
    interior, one-sided 3-point (2nd order) at both ends. Used for both
    dP3/dz (Poisson's source) and dphi/dz (recovering E3). Two nodes (Nz=2)
    carry only a linear profile: both get its one-cell difference, exact
    for it (the Poisson path then has no interior row and overwrites E3's
    two ends, so these values are not consumed there)."""
    d = np.empty_like(fk)
    if fk.shape[0] == 2:
        d[:] = (fk[1] - fk[0]) / hz
        return d
    d[1:-1] = (fk[2:] - fk[:-2]) / (2 * hz)
    d[0] = (-3 * fk[0] + 4 * fk[1] - fk[2]) / (2 * hz)
    d[-1] = (3 * fk[-1] - 4 * fk[-2] + fk[-3]) / (2 * hz)
    return d


# ==========================================================================
# Grid: in-plane via solver.make_grid (WP4 10b rfft2 half-spectrum, reused
# verbatim) + a z NODE grid spanning the film thickness only (see module
# docstring, STACK LAYOUT NOTE).
# ==========================================================================
PAPER_NXY = 128
PAPER_DX_NM = 0.4          # WP6 Tier 2's choice, reused verbatim (SPEC §13.0)
PAPER_HF_DX = 20           # paper text: hf = 20*dx
PAPER_NZ = PAPER_HF_DX + 1  # 21 nodes (20 intervals) spanning hf, see docstring


def make_grid_3d(Nx, Ny, Lx, Ly, Nz, hf):
    """NAME COLLISION: bulk3d has its own make_grid_3d(Nx, Ny, Nz, Lx, Ly,
    Lz) -- positional #3 is Lx here and Nz there, so a positional call to the
    wrong module builds a silently wrong box (ENGINE_GUIDE §2.1 has the full
    collision table). In-plane grid fields (Nx,Ny,Lx,Ly,hx,hy,KX,KY,Nxr,KXr,KYr,...) exactly
    as solver.make_grid, plus Nz z-nodes spanning [0,hf] inclusive (hz =
    hf/(Nz-1)), the trapezoidal z-quadrature weight array qw (hz at interior
    nodes, hz/2 at the two boundary nodes -- SPEC's natural-BC finite-volume
    weighting, see module docstring), and Volume3D = Lx*Ly*hf (grid['Volume']
    from solver.make_grid is the 2D in-plane AREA only -- kept as-is,
    Volume3D is this module's own addition to avoid ambiguity)."""
    validate.film_make_grid_3d(Nx, Ny, Lx, Ly, Nz, hf)
    grid = solver.make_grid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
    Nz = int(Nz)
    hz = hf / (Nz - 1)
    z = np.arange(Nz) * hz
    qw = np.full(Nz, hz)
    qw[0] = qw[-1] = hz / 2.0
    grid.update(Nz=Nz, hf=hf, hz=hz, z=z, qw=qw, Volume3D=Lx * Ly * hf)
    return grid


def paper_grid(Nx=PAPER_NXY, Ny=PAPER_NXY, Nz=PAPER_NZ, dx_nm=PAPER_DX_NM):
    """Convenience: the paper's own in-plane size/spacing (SPEC §13.0),
    film-only z extent (see module docstring)."""
    Lx = Nx * dx_nm
    Ly = Ny * dx_nm
    hf = (Nz - 1) * dx_nm
    return make_grid_3d(Nx, Ny, Lx, Ly, Nz, hf)


# ==========================================================================
# Params (SPEC §13.0: paper mechanics set, isotropic G; reuse
# literature_params.py's existing loader unchanged, per "reuse ... params").
# ==========================================================================
def paper_mechanics():
    """SPEC §13.0's paper parameter set: literature_params.bto_lcc2005_paper(T=298 K, dx_nm=PAPER_DX_NM)."""
    return lp.bto_lcc2005_paper(T=298.0, dx_nm=PAPER_DX_NM)


def paper_alpha(T_C):
    """SPEC WP6-style alpha dict at temperature T_C (deg C) -- same pattern
    as cases/film_tier2.py's alpha_at(), duplicated here (film3d.py has no
    reason to import a cases/ file, and this is 6 lines)."""
    T_K = T_C + 273.0
    full = lp.bto_lcc2005_8th(T=T_K)
    return dict(a1=full["alpha1"], a11=full["alpha11"], a12=full["alpha12"],
                a111=full["alpha111"], a112=full["alpha112"], a123=full["alpha123"],
                a1111=full["alpha1111"], a1112=full["alpha1112"],
                a1122=full["alpha1122"], a1123=full["alpha1123"])


# KAPPA_B: the OPEN PHYSICS INPUT (SPEC §13.0). APL2006's own text never
# gives a numeric background permittivity anywhere (confirmed: full read of
# the 3-page Letter). Its ref. 31 (Li, Hu, Chen, J. Appl. Phys. 97, 034112
# (2005)) could not be located as a free PDF (WebSearch attempts logged in
# reports/WP7A_REPORT.md). Its ref. 25 (Li, Hu, Liu, Chen, Appl. Phys. Lett.
# 81, 427 (2002) -- the DIRECT methodological precursor establishing the
# SAME short-circuit electrostatic formulation, cited in APL2006's own text
# alongside ref. 26 for "how E_i depends on P_i and the electric BCs", and
# itself built on "Ref. 19" = Li 2002 Acta Mater, i.e. THIS SPEC's other
# named reference) DOES give a number: kappa11=kappa22=kappa33=66.0
# (relative permittivity), citing Haun, Furman, Jang, Mckinstry & Cross,
# J. Appl. Phys. 62, 3331 (1987) -- the SAME Haun 1987 source already used
# elsewhere in this project's parameter library (params/archive/pto_Haun1987.toml).
# Per SPEC's explicit fallback ("if genuinely unstated, run the WP7a gates
# with kappa_b = our deck value AND document a sensitivity note"): since
# ref. 31 ITSELF (not just its lineage) could not be read, KAPPA_B_DECK is
# the DEFAULT used by every gate below; KAPPA_B_PAPER_LINEAGE is reported
# as the sourced sensitivity comparison (not silently substituted).
# SPEC V2-02.1 (Lens 2 U1, ruling D7) -- LABELS CORRECTED, VALUES FROZEN. The
# derived mapping is kappa_deck = 0.0088541878128*eps_b (solvers/units.py,
# gates/core/k5_materials.py `permittivity_convention`), so the inherited deck
# number 0.08854187 is eps_b = 10 (the Ferret original: "here we use e_b = 10",
# PRB 74, 104014 (2006)), NOT eps_b = 1; and 66x that is eps_b = 660, NOT the
# ref-25 / Haun-1987 eps_b = 66 this lineage intended. Every film paper-lineage
# run therefore carried 10x WEAKER depolarization than the paper meant. Whether
# to re-run at the true eps_b = 66 (deck 0.5843763) is escalated as user
# decision D15; until then the stored numbers stand, labeled 10 / 660.
KAPPA_B_DECK = paper_mechanics()["permittivity"]         # HM2006-inherited placeholder (eps_b=10)
KAPPA_B_PAPER_LINEAGE = 66.0 * KAPPA_B_DECK               # ref 25 / Haun 1987 lineage 66x (eps_b=660)


# ==========================================================================
# Batched (per-mode) tridiagonal solver -- SPEC §13.0's "banded solvers,
# precomputed/factorized where profitable". Both call sites in this module
# (the Neumann gradient operator, the Dirichlet Poisson operator) have
# sub/sup that do NOT depend on the (k1,k2) mode (only the diagonal does,
# via +G11*k^2 or +kappa_b*k^2) -- so sub/sup are always plain 1D (n-1,)
# arrays; diag carries the batch shape. Standard Thomas algorithm, no
# pivoting (both operators are diagonally dominant, see each _bands
# function's docstring) -- the O(n) elimination loop is the only
# sequential part; every array op inside it is vectorized over the full
# mode batch (~Ny*Nxr ~ thousands of independent systems per call).
# ==========================================================================
def factor_tridiag_batched(sub, diag, sup):
    """Precompute the forward-elimination coefficients ONCE for a fixed
    (sub,diag,sup) -- reused every time step (only the RHS changes).
    sub,sup: 1D (n-1,). diag: (n,) or (n,)+batch_shape. n=0 (no unknowns:
    a 2-node film's interior-only Poisson system) gives empty factors."""
    n = diag.shape[0]
    batch_shape = diag.shape[1:]
    cp = np.empty((max(n - 1, 0),) + batch_shape)
    piv = np.empty((n,) + batch_shape)
    if n:
        piv[0] = diag[0]
    for i in range(1, n):
        cp[i - 1] = sup[i - 1] / piv[i - 1]
        piv[i] = diag[i] - sub[i - 1] * cp[i - 1]
    return dict(sub=sub, cp=cp, piv=piv, n=n)


def solve_tridiag_factored(factors, rhs):
    """rhs: (n,)+batch_shape[+extra trailing axes broadcastable against
    batch_shape, e.g. a component axis -- numpy broadcasting handles it
    with zero special-casing]. Returns the same shape as rhs."""
    sub, cp, piv, n = factors["sub"], factors["cp"], factors["piv"], factors["n"]
    if n == 0:
        return np.empty_like(rhs)   # empty system: rhs is (0,...) too
    dp = np.empty_like(rhs)
    dp[0] = rhs[0] / piv[0]
    for i in range(1, n):
        dp[i] = (rhs[i] - sub[i - 1] * dp[i - 1]) / piv[i]
    x = np.empty_like(dp)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def solve_tridiag_batched(sub, diag, sup, rhs):
    """One-shot (factor+solve) entry point -- used where the operator isn't
    reused across many calls (kept for gate self-checks / one-off use;
    every hot-path call site below uses the factor-once pattern instead)."""
    return solve_tridiag_factored(factor_tridiag_batched(sub, diag, sup), rhs)


# ==========================================================================
# Poisson (short-circuit): kappa_b * laplacian_3d(phi) = div(P), phi=0 at
# both film surfaces (Dirichlet -- directly removes the constant-mode null
# space the ORIGINAL L1 periodic-in-all-directions model needed a special
# k=0 case for; no such special-casing is needed here).
# ==========================================================================
def _kappa_faces(kappa_nodes):
    """Face-centred permittivity between consecutive z NODES: the HARMONIC
    mean, kappa_{j+1/2} = 2/(1/kappa_j + 1/kappa_{j+1}), returned as a
    (Nz-1,) array (entry j is the face between nodes j and j+1).

    WHY HARMONIC AND NOT ARITHMETIC (SPEC V2-03 A7 asks for the choice AND
    its justification). The face coefficient's only job in the finite-volume
    row below is to carry the flux kappa*dphi/dz between two node centres,
    i.e. it is a CONDUCTANCE in series across the half-cell on each side. For
    a kappa step aligned with a node face, the exact series conductance of
    (hz/2 of kappa_j) + (hz/2 of kappa_{j+1}) is hz/((hz/2)/kappa_j +
    (hz/2)/kappa_{j+1}) = the harmonic mean -- so with this rule the discrete
    sum over faces sum_j hz/kappa_{j+1/2} equals the continuum integral
    int dz/kappa(z) EXACTLY, which is precisely the quantity the
    series-capacitor (dead-layer) closed form is built from. The arithmetic
    mean does not: its error is -hz*(kappa_a-kappa_b)^2 /
    (2*kappa_a*kappa_b*(kappa_a+kappa_b)) per stepped face, first order in hz.
    Gated by K3.film_dead_layer_closed_form (exact, <=1e-8).

    The == guard is what makes the CONSTANT profile byte-identical to the
    scalar path (K2.film_kappa_z_const_byte): 2*k*k/(k+k) is k**2/k, which is
    NOT bit-for-bit k for a general float, so equal neighbours return the node
    value itself rather than a recomputed one."""
    ka, kb = kappa_nodes[:-1], kappa_nodes[1:]
    return np.where(ka == kb, ka, 2.0 * ka * kb / (ka + kb))


def _poisson_bands(grid, kappa_b):
    """Interior-only (nodes 1..Nz-2) Dirichlet tridiagonal: SYMMETRIC,
    diagonally dominant (SPD) standard 3-point Laplacian -- no ghost-point
    trick needed (Dirichlet supplies the boundary VALUE directly, unlike
    Neumann which must eliminate an unknown ghost derivative).

    SPEC V2-03 A7 -- kappa_b is EITHER a plain scalar (the homogeneous
    background this file has always had; that branch is the ORIGINAL code,
    untouched and therefore byte-identical) OR a (Nz,) NODE profile
    kappa(z), for dead layers / graded interfacial permittivity in the film
    lane. The profile branch discretizes the genuine inhomogeneous equation
    div(kappa(z) grad phi) = div P per in-plane mode as the standard
    finite-volume row

        [kappa_{j+1/2}(phi_{j+1}-phi_j) - kappa_{j-1/2}(phi_j-phi_{j-1})]/hz^2
            - kappa_j*k_perp^2*phi_j  =  source_j

    (sign-flipped below to match the existing rhs = -source convention), with
    kappa_{j+1/2} the HARMONIC face average (_kappa_faces, which carries the
    derivation) and kappa_j the node value in the in-plane k^2 term. A
    CONSTANT profile collapses to exactly the scalar expressions above -- the
    face average returns the node value, kappa_{j-1/2}+kappa_{j+1/2} is
    exactly 2*kappa, and the operand order is the same -- so the two paths are
    bit-for-bit equal (K2.film_kappa_z_const_byte).

    NOTHING ELSE IN THE POISSON PATH NEEDS A kappa BRANCH, checked not
    assumed: solve_poisson_factored/elec_force_3d/elec_energy_density_3d take
    the FACTORS, never kappa, and the module docstring's ELECTROSTATIC FORCE
    SIGN reciprocity proof only ever uses kappa as the coefficient inside
    div(kappa grad phi) -- both integrations by parts go through verbatim for
    kappa(z), giving the same symmetric -int kappa(z) grad phi1 . grad phi2
    and hence the same dF_elec/dP_i = -E_d,i (no extra 1/2) and the same
    non-negative F_elec = +(1/2) int kappa(z)|grad phi|^2 >= 0.
    The symmetric positive Poisson matrix and matching source/field adjoints give exact discrete energy-force pairing also for kappa(z)."""
    Nz, hz = grid["Nz"], grid["hz"]
    n_int = Nz - 2
    k2 = operators.k_squared(grid, 2)
    kap = np.asarray(kappa_b, dtype=float)
    if kap.ndim == 0:
        sub = np.full(max(n_int - 1, 0), -kappa_b / hz**2)
        sup = sub.copy()
        diag_row = 2 * kappa_b / hz**2 + kappa_b * k2
        diag = np.broadcast_to(diag_row[None, :, :], (n_int,) + diag_row.shape)
        return sub, diag, sup
    if kap.shape != (Nz,):
        raise ValueError(f"_poisson_bands: kappa_b must be a scalar or a (Nz,)={Nz} "
                         f"node profile, got shape {kap.shape}")
    if not np.all(kap > 0.0):
        raise ValueError("_poisson_bands: kappa(z) must be strictly positive everywhere")
    kf = _kappa_faces(kap)                      # (Nz-1,), face j+1/2 between nodes j and j+1
    sub = -kf[1:n_int] / hz**2                  # row i (node i+1) <- node i, i.e. face i+1/2
    sup = sub.copy()                            # symmetric: row i -> node i+2 is face i+3/2
    diag = ((kf[:n_int] + kf[1:n_int + 1])[:, None, None] / hz**2
            + kap[1:Nz - 1][:, None, None] * k2)
    return sub, diag, sup


def poisson_factor(grid, kappa_b):
    """Factorize the film's z-tridiagonal Poisson operator once per (grid,
    kappa_b), for solve_poisson_factored. SPEC V2-03 A7: kappa_b is a scalar
    OR a (Nz,) kappa(z) node profile -- see _poisson_bands."""
    validate.poisson_factor(grid, kappa_b)
    return factor_tridiag_batched(*_poisson_bands(grid, kappa_b))


def solve_poisson_factored(P, grid, factors, E_ext=(0.0, 0.0, 0.0)):
    """Solve for phi given P, return conjugate (E, E_ext_broadcast) as real-
    space (3,Nz,Ny,Nx) arrays (mirrors solver.py's convention of keeping
    E_ext un-redistributed, added only by the caller -- see elec_force_3d).

    SPEC V2-03 C2 carry-over (d.1): when `factors` is a film3d_spm factors
    object (duck-typed on the factors["spm"] flag B3 sets, which a
    poisson_factor() output never has), the kappa(x,y,z) SPM runs instead.
    THIS is the single dispatch point rather than explicit_N, deliberately:
    total_energy_3d does NOT go through explicit_N -- it calls
    elec_energy_density_3d directly -- so a hook in explicit_N would pair an
    SPM force with a kappa0(z)-only energy, and K1 `fd_film_full_coupling_spm`
    would (correctly) fail. One hook here routes elec_force_3d,
    elec_energy_density_3d and therefore explicit_N / step / step_sbdf2 /
    total_energy_3d through the same converged field, and cannot get out of
    sync (relax / probe_stable_dt take kappa_b and build banded
    poisson_factor() factors themselves). No recursion: film3d_spm itself calls
    this function with factors["kappa0_factors"], which has no "spm" key.
    Two optional slots a caller may set on the factors dict: "spm_tol" (the
    fixed point's tolerance, default film3d_spm's own 1e-10) and "_warm_E",
    the warm start this branch writes back after every solve (elastic.py's
    el["_warm_*"] contract -- a pure starting-GUESS cache that cannot move
    the fixed point)."""
    if factors.get("spm"):
        from fftpf import film_electrostatic_inhomogeneous as fsp    # local: film3d_spm imports this module
        E, Eext, _n, _r = fsp.solve_poisson_spm(P, grid, factors, E_ext,
                                                tol=factors.get("spm_tol", 1e-10),
                                                warm=factors.get("_warm_E"))
        factors["_warm_E"] = E
        return E, Eext
    Ny, Nx = grid["Ny"], grid["Nx"]
    hz = grid["hz"]
    # Odd spectral derivatives must vanish on their own Nyquist axis so
    # source and recovery both map real fields to real fields.
    aK, bK = operators.derivative_symbols_2d(grid)
    Pk = _rfft2(P)  # (3,Nz,Ny,Nxr)
    divxy_k = aK[None, :, :] * Pk[0] + bK[None, :, :] * Pk[1]
    dP3dz_k = _dz_2nd_order(Pk[2], hz)
    source_k = divxy_k + dP3dz_k  # (Nz,Ny,Nxr), z is axis 0 already
    rhs_int = -source_k[1:-1]
    phi_int = solve_tridiag_factored(factors, rhs_int)
    phi_k = np.zeros_like(Pk[0])
    phi_k[1:-1] = phi_int
    E1k = -aK[None, :, :] * phi_k
    E2k = -bK[None, :, :] * phi_k
    E3k = -_dz_2nd_order(phi_k, hz)
    # Conjugate field for the interior divergence B, under trapezoidal W:
    # E = W^-1 B^T (hz*phi). Interior rows give -central_grad(phi);
    # the half-weight boundary nodes give these one-cell differences.
    # Thus -<P,E>/2 = -<B P,phi>/2 >= 0 and dF/dP = -E exactly.
    E3k[0] = -phi_k[1] / hz
    E3k[-1] = phi_k[-2] / hz
    Ek = np.stack([E1k, E2k, E3k])
    E = _irfft2(Ek, (Ny, Nx))
    # (3,) -> (3,1,1,1) (the repaired uniform-field baseline); a field must be
    # component-first and broadcastable to P; finite. The film's (stricter,
    # ensemble-free) rule lives with the periodic engines' in operators.py.
    Eext = operators.vec3_field_checked(E_ext, P.shape)
    return E, Eext


def physical_electric_field_3d(P, grid, factors, E_ext=(0.0, 0.0, 0.0)):
    """Pointwise field diagnostic with second-order one-sided surface E3.

    Interior values equal solve_poisson_factored's energy-conjugate field.
    This surface reconstruction is NOT the discrete energy derivative and
    must not be used in elec_force_3d or elec_energy_density_3d.
    """
    E, Eext = solve_poisson_factored(P, grid, factors, E_ext)
    E = E.copy()
    E[2, 0] = 2 * E[2, 0] - E[2, 1]
    E[2, -1] = 2 * E[2, -1] - E[2, -2]
    return E, Eext


def elec_force_3d(P, grid, poisson_factors, E_ext=(0.0, 0.0, 0.0)):
    """delta F_elec/delta P_i = -E_d,i - E_ext,i (self-adjoint envelope
    argument, film-geometry reciprocity proof in the module docstring)."""
    E, Eext = solve_poisson_factored(P, grid, poisson_factors, E_ext)
    return -E - Eext


def elec_energy_density_3d(P, grid, poisson_factors, E_ext=(0.0, 0.0, 0.0)):
    """f_elec = -0.5*P.E_d - P.E_ext with the film-geometry Poisson solve, shape (Nz,Ny,Nx)."""
    E, Eext = solve_poisson_factored(P, grid, poisson_factors, E_ext)
    return (-0.5 * (P[0] * E[0] + P[1] * E[1] + P[2] * E[2])
            - (P[0] * Eext[0] + P[1] * Eext[1] + P[2] * Eext[2]))


# ==========================================================================
# Gradient operator (Neumann, isotropic G11) -- see module docstring
# "GRADIENT OPERATOR DERIVATION" for the full derivation of these bands.
# ==========================================================================
def _grad_z_bands(Nz, hz, G11):
    sub = np.full(Nz - 1, -G11 / hz**2)
    sup = np.full(Nz - 1, -G11 / hz**2)
    sub[-1] *= 2.0   # row Nz-1 (boundary): single face, carries the full flux
    sup[0] *= 2.0    # row 0 (boundary): single face, carries the full flux
    diag = np.full(Nz, 2.0 * G11 / hz**2)
    return sub, diag, sup


# ==========================================================================
# SPEC V2-03 B4 (anisotropic gradient) + B5 (H11, implicit pentadiagonal).
#
# WHAT IS NEW AND WHAT IS UNTOUCHED. Every function below that already
# existed keeps its ORIGINAL body verbatim on the path it always had --
# `G` a plain scalar G11 and `H11` omitted. The dict / H11 branches are
# additive; K2 `film_grad_isotropic_byte` and `film_hog_h11_zero_byte` pin
# the two degeneracies bit-for-bit.
#
# B4 -- THE DISCRETE ANISOTROPIC GRADIENT ENERGY (derived, not assumed).
# SPEC 1.2's f_grad, split by which derivatives are in-plane and which are
# d/dz (P_i,j = dP_i/dx_j):
#
#   f_grad = [ 1/2 G11 (P1,1^2+P2,2^2) + G12 P1,1 P2,2
#              + 1/2 G44 (P1,2+P2,1)^2 + 1/2 G44' (P1,2-P2,1)^2
#              + 1/2 (G44+G44') (P3,1^2+P3,2^2) ]                  (A) in-plane
#          + [ 1/2 (G44+G44') (P1,3^2+P2,3^2) + 1/2 G11 P3,3^2 ]   (B) pure z
#          + [ G12 P3,3 (P1,1+P2,2)
#              + (G44-G44') (P1,3 P3,1 + P2,3 P3,2) ]              (C) mixed
#
# (B) and (C) follow from expanding SPEC 1.2's (P_i,3 +/- P_3,i)^2 pairs:
# 1/2 G44 (P1,3+P3,1)^2 + 1/2 G44' (P1,3-P3,1)^2 = 1/2 (G44+G44')(P1,3^2 +
# P3,1^2) + (G44-G44') P1,3 P3,1, and likewise for the (2,3) pair.
# (A) is EXACTLY solver.build_operators' A(k) -- checked term by term:
# A11 = G11 k1^2 + (G44+G44') k2^2, A22 = G11 k2^2 + (G44+G44') k1^2,
# A12 = (G12+G44-G44') k1 k2, A33 = (G44+G44')(k1^2+k2^2) -- so the in-plane
# half of the film's operator IS the 2D engine's, reused verbatim (K3
# `film_aniso_dispersion` pins that against bulk3d's own A(k) at k3=0).
#
# DISCRETIZATION. (A) is nodal: spectral in-plane derivatives at each z
# node, trapezoidal qw weight -- the same quadrature the isotropic path
# already used. (B)+(C) live on the FACES between consecutive z nodes, with
#     D_i = (P_i,j+1 - P_i,j)/hz               (face-centred d/dz)
#     S_i = (P_i,j + P_i,j+1)/2                (face-centred nodal average)
# and the in-plane derivative of S taken spectrally, weight hz per face:
#     q_face = 1/2 (G44+G44')(D1^2+D2^2) + 1/2 G11 D3^2
#              + G12 D3 (dS1/dx + dS2/dy)
#              + (G44-G44')(D1 dS3/dx + D2 dS3/dy)
# The FORCE is the EXACT gradient of area*[sum_j qw_j <(A)_j> + hz sum_face
# <q_face>] (K1 `fd_film_grad_aniso`, 1e-9 class): the local 6x6 face
# element K(a,b) = X(-k)^T C X(k) is derived SYMBOLICALLY
# (_derive_grad_face_element, film_elastic._derive_local_element's idiom,
# including its Hermitian-pairing root-cause fix -- the naive real-symbol
# Hessian is symmetric but not Hermitian off k=0), then scattered face by
# face into a 3x3 BLOCK-TRIDIAGONAL z operator and row-scaled by hz/qw(z).
#
# NATURAL BC, NOT IMPOSED. Nothing is placed beyond the first/last node, so
# node 0 sees only face 1/2 and node Nz-1 only face Nz-3/2: the boundary
# ROWS come out of the same sum, and their content is exactly the discrete
# statement of the continuum natural BC df/d(P_i,3) = 0, i.e.
#     t1 = (G44+G44') P1,3 + (G44-G44') P3,1
#     t2 = (G44+G44') P2,3 + (G44-G44') P3,2
#     t3 = G11 P3,3 + G12 (P1,1+P2,2)
# vanishing at z=0 and z=hf. CONSEQUENCE WORTH STATING (it is a real
# difference from the isotropic path, and K3 `film_aniso_dispersion` gates
# it): with G12 != 0 or G44 != G44' a z-UNIFORM field is no longer a
# stationary configuration -- its surface rows carry force_i = [A(k)P]_i -/+
# (2/hz) t_i with t1 = (G44-G44') dP3/dx, t2 = (G44-G44') dP3/dy,
# t3 = G12 (dP1/dx+dP2/dy), which is zero only in the isotropic reduction.
# The SPEC line "the z-terms vanish under natural BC" is therefore exact in
# the film INTERIOR for every G, and exact everywhere only when those two
# combinations vanish; the K3 item checks the interior against A(k) and the
# two surface rows against the closed form above.
#
# ISOTROPIC REDUCTION, and why the dict path DISPATCHES to the scalar one.
# G12=0, G44=G44'=G11/2 gives G44+G44' = G11 EXACTLY (x/2+x/2 is exact in
# IEEE754) and G44-G44' = 0 exactly, so (C) disappears, (B) becomes
# 1/2 G11 sum_i D_i^2 -- the isotropic path's own face functional -- and (A)
# collapses to A(k) = G11 k^2 I. The 3x3 block operator is then exactly
# three copies of the scalar tridiagonal one. `_grad_scalar_G` detects that
# with three float equalities and returns the scalar, so the dict call takes
# the ORIGINAL code path and is byte-identical rather than round-off-close
# (the same discipline as _kappa_faces' `==` guard above). This is not an
# approximation: the operators are equal, only the arithmetic order would
# differ. K2 `film_grad_isotropic_byte` checks BOTH -- the public API at
# 0.0, and the forced block assembly (`_grad_force_block`,
# `_build_grad_factors_block`) at round-off, so the block path itself is
# still measured against the scalar one.
#
# NYQUIST CONVENTION (checked numerically, not assumed; the one rule for
# every engine is stated in solvers/operators.py). The mixed terms
# (C) are the only ones carrying an ODD power of an in-plane wavenumber.
# For those the in-plane symbol is taken as i*k with the Nyquist bin zeroed
# (_odd_k) -- the standard spectral rule for odd derivatives. Reason: the
# unmasked i*k_nyq symbol is not the diagonalization of ANY real operator
# there (it breaks the rfft2 half-spectrum's conjugate constraint), so the
# per-mode implicit SOLVE would be inverting something the explicit force
# does not apply. Masking makes the per-mode block operator a faithful
# diagonalization, and energy and force use the same masked symbol so the
# pair stays exactly adjoint. Of the in-plane symbols (A), the EVEN ones
# (A11, A22, A33) and H11's kx^4+ky^4 stay unmasked, as in
# solver._grad_symbols; the mixed A12 ~ kx*ky is odd x odd and, like
# solver._grad_symbols' A12, is zeroed on each even-axis Nyquist line
# (_grad_A2d) for the same reason.
#
# B5 -- H11, IMPLICIT PENTADIAGONAL (SPEC V2-03 B5 RULING). Component-wise,
# NC2025 eq.3 as in grad4.py:
#     f_hog = 1/2 H11 sum_i [ (d2Pi/dx2)^2 + (d2Pi/dy2)^2 + (d2Pi/dz2)^2 ]
# In-plane that is H11*(kx^4+ky^4) added to the diagonal multipliers (the 2D
# module's own recipe). In z the discrete functional is built on the
# INTERIOR second differences alone,
#     F_hog_z = area*hz * sum_{j=1..Nz-2} < 1/2 H11 sum_i Q_ij^2 >,
#     Q_ij = (P_i,j+1 - 2 P_i,j + P_i,j-1)/hz^2,
# with nothing at j=0 or j=Nz-1. Its exact gradient (H11*hz/qw)*D2^T D2 is
# PENTADIAGONAL in z; rows 0 and 1 (and their mirrors) are the natural BCs
# falling OUT of the functional, not imposed: stationarity of row 0 reads
# Q_1 = 0 (d2P/dz2 = 0 at the surface) and row 1 reads -2 Q_1 + Q_2 = 0,
# which with the former gives Q_2 = 0 as well, i.e. d3P/dz3 = 0. The
# interior rows are the standard (1,-4,6,-4,1)/hz^4 biharmonic stencil, so
# the interior symbol is 16 sin^4(k3 hz/2)/hz^4 -> k3^4 at order 2 (K3
# `film_hog_dispersion`).
# Explicit treatment would need dt <~ hz^4/(8 L H11) ~ 1.5e-4 at hz=0.4 nm,
# H11=22 -- hence the RULING; `factor_pentadiag_batched`/
# `solve_pentadiag_factored` below mirror the tridiagonal pair, and
# `factor_block_pentadiag_batched`/`solve_block_pentadiag_factored` mirror
# film_elastic's block pair so H11 works with the anisotropic G too (it did
# fall out of the same code path). Both are written in the SAME L*U
# normalisation as the tridiagonal ones (U unit-diagonal, cp = superdiag /
# pivot), which is what makes the outer bands' exact zeros collapse the
# recurrences onto the tridiagonal ones BIT-FOR-BIT when H11 = 0.
# ==========================================================================
_GRAD_KEYS = ("G11", "G12", "G44", "G44p")


def _grad_scalar_G(G):
    """The scalar G11 this call is EQUIVALENT to, or None if it genuinely
    needs the 3x3 block path. A non-dict G is returned UNCHANGED (never
    re-wrapped through float()), so the original path's arithmetic is
    untouched object-for-object. A dict is reduced only on the three EXACT
    equalities of the isotropic case (see the section comment)."""
    if isinstance(G, dict):
        G11, G12, G44, G44p = (G[k] for k in _GRAD_KEYS)
        if G12 == 0.0 and G44 == G44p and (G44 + G44p) == G11:
            return G11
        return None
    if np.ndim(G) != 0:
        raise ValueError("film3d: G must be a scalar G11 or a dict "
                         f"{_GRAD_KEYS}, got shape {np.shape(G)}")
    return G


def _grad_dict(G):
    """Validate + unpack the anisotropic G dict (floats)."""
    if not isinstance(G, dict):
        raise TypeError(f"film3d: expected a dict {_GRAD_KEYS}, got {type(G).__name__}")
    missing = [k for k in _GRAD_KEYS if k not in G]
    if missing:
        raise ValueError(f"film3d: G dict is missing {missing} (needs {_GRAD_KEYS})")
    return {k: float(G[k]) for k in _GRAD_KEYS}


# (i*kx, i*ky) for the ODD-order in-plane symbols of the mixed z-terms,
# Nyquist bin zeroed on each one's own axis -- see the NYQUIST CONVENTION note
# above. Shapes (Ny,Nxr). The shared operators.derivative_symbols_2d (the
# same masked wavevectors solver/elastic use); this name is kept as its alias
# (film_elastic imports it).
_odd_k = operators.derivative_symbols_2d


_GRAD_FACE_ELEMENT = None


def _derive_grad_face_element():
    """sympy derivation of the local 6x6 face element K(a,b) = X(-k)^T C X(k)
    of the (B)+(C) face functional -- film_elastic._derive_local_element's
    idiom, with its Hermitian-pairing fix: the DOFs here are per-mode complex
    Fourier amplitudes, so the stiffness is the CONJUGATED pairing, not a
    naive real-symbol Hessian.

    Generalized strain e = [D1, D2, D3, a*S1, b*S2, a*S3, b*S3] (affine in
    the in-plane symbols a=d/dx, b=d/dy, since every entry carries at most
    ONE in-plane derivative), q_face = 1/2 e^H C e with C symmetric and
    carrying G44+G44' / G11 on the D-diagonal, G12 pairing D3 with a*S1 and
    b*S2, and G44-G44' pairing D1/D2 with a*S3 / b*S3. Because C has NO
    entry pairing two a/b-carrying rows, K is AFFINE in (a,b) -- asserted
    here, not assumed -- so K = K0 + a*Ka + b*Kb with K0 symmetric (the pure
    z part) and Ka/Kb antisymmetric (the mixed part), which is exactly the
    Hermiticity of K(k) once a = i*k1 is substituted. Both properties are
    asserted below. Imported LAZILY (sympy is not needed by the scalar path,
    which every legacy caller takes) and cached.

    Returns three lambdified callables K0/Ka/Kb of (hz, G11, G12, G44, G44p)."""
    import sympy as sp

    hz, a, b = sp.symbols("hz a b")
    G11, G12, G44, G44p = sp.symbols("G11 G12 G44 G44p", real=True)
    dof = sp.symbols("P1m P2m P3m P1p P2p P3p", real=True)
    P1m, P2m, P3m, P1p, P2p, P3p = dof

    D = [(P1p - P1m) / hz, (P2p - P2m) / hz, (P3p - P3m) / hz]
    S = [(P1m + P1p) / 2, (P2m + P2p) / 2, (P3m + P3p) / 2]
    e = sp.Matrix([D[0], D[1], D[2], a * S[0], b * S[1], a * S[2], b * S[2]])
    X = e.jacobian(dof)                       # 7x6, affine in (a,b)

    gg, c44m = G44 + G44p, G44 - G44p
    C = sp.zeros(7, 7)
    C[0, 0] = gg
    C[1, 1] = gg
    C[2, 2] = G11
    C[2, 3] = C[3, 2] = G12
    C[2, 4] = C[4, 2] = G12
    C[0, 5] = C[5, 0] = c44m
    C[1, 6] = C[6, 1] = c44m

    K = sp.expand(X.subs({a: -a, b: -b}).T * C * X)     # 6x6
    K0 = sp.expand(K.subs({a: 0, b: 0}))
    Ka = sp.expand(sp.diff(K, a))
    Kb = sp.expand(sp.diff(K, b))
    if sp.simplify(K - (K0 + a * Ka + b * Kb)) != sp.zeros(6, 6):
        raise AssertionError("film3d: grad face element is not affine in (a,b)")
    if sp.simplify(K0 - K0.T) != sp.zeros(6, 6):
        raise AssertionError("film3d: grad face element K0 is not symmetric")
    if sp.simplify(Ka + Ka.T) != sp.zeros(6, 6) or sp.simplify(Kb + Kb.T) != sp.zeros(6, 6):
        raise AssertionError("film3d: grad face element Ka/Kb are not antisymmetric")

    args = (hz, G11, G12, G44, G44p)
    return (sp.lambdify(args, K0, "numpy"), sp.lambdify(args, Ka, "numpy"),
            sp.lambdify(args, Kb, "numpy"))


def _grad_face_element():
    global _GRAD_FACE_ELEMENT
    if _GRAD_FACE_ELEMENT is None:
        _GRAD_FACE_ELEMENT = _derive_grad_face_element()
    return _GRAD_FACE_ELEMENT


def _grad_block_bands(grid, Gd):
    """The 3x3 block-tridiagonal z-part of the ANISOTROPIC gradient force
    operator, per in-plane mode: (sub, diag, sup) of shapes (Nz-1,Ny,Nxr,3,3),
    (Nz,Ny,Nxr,3,3), (Nz-1,Ny,Nxr,3,3), complex. Index convention copied from
    film_elastic's batched block Thomas (and from this module's scalar bands):
    sub[i-1] is the block at (row i, col i-1), sup[i-1] at (row i-1, col i).

    Assembly = scatter the local face element (node-, node+) into the global
    nodes and row-scale by hz/qw(z) -- the SAME finite-volume weighting the
    scalar `_grad_z_bands` carries (1 at interior nodes, 2 at the two
    surfaces, because a surface node owns half a cell but a whole face)."""
    fn_K0, fn_Ka, fn_Kb = _grad_face_element()
    Nz, hz, qw = grid["Nz"], grid["hz"], grid["qw"]
    if Nz < 2:
        raise ValueError("film3d: the anisotropic gradient needs Nz >= 2")
    args = (hz, Gd["G11"], Gd["G12"], Gd["G44"], Gd["G44p"])
    K0 = np.asarray(fn_K0(*args), dtype=float)
    Ka = np.asarray(fn_Ka(*args), dtype=float)
    Kb = np.asarray(fn_Kb(*args), dtype=float)
    aK, bK = operators.derivative_symbols_2d(grid)         # (Ny,Nxr) complex
    K = (K0[None, None] + Ka[None, None] * aK[..., None, None]
         + Kb[None, None] * bK[..., None, None])           # (Ny,Nxr,6,6)

    Kmm, Kmp = K[..., 0:3, 0:3], K[..., 0:3, 3:6]
    Kpm, Kpp = K[..., 3:6, 0:3], K[..., 3:6, 3:6]
    w = (hz / qw)[:, None, None, None, None]               # (Nz,1,1,1,1)
    diag = np.empty((Nz, grid["Ny"], grid["Nxr"], 3, 3), dtype=complex)
    diag[1:-1] = (Kmm + Kpp)[None]
    diag[0] = Kmm                                          # no face below node 0
    diag[-1] = Kpp                                         # no face above node Nz-1
    diag *= w
    sup = Kmp[None] * w[:-1]
    sub = Kpm[None] * w[1:]
    return sub, diag, sup


def _grad_A2d(grid, Gd, H11=None):
    """(A11, A22, A12, A33) -- SPEC 1.2's in-plane A(k) on the film's rfft2
    half-spectrum grid: operators.grad_symbols_2d, the SAME function
    solver.build_operators calls (so the film cannot drift from the 2D
    engine); plus H11*(kx^4+ky^4) on the three diagonals when H11 is given
    (operators.add_hog / hog_symbol, grad4's recipe).

    The odd x odd mixed symbol A12 is zeroed on each even-axis Nyquist line
    (the shared rule): on the kx-Nyquist column irfft2 discards it from the
    explicit force/energy anyway, so the per-mode implicit block (_A_block)
    must not invert it either, and masking the ky-Nyquist row too keeps the
    energy x<->y transposition-symmetric. Odd axes have no Nyquist line, so
    odd-grid numbers are unchanged."""
    A11, A22, A12, A33 = operators.grad_symbols_2d(grid, *(Gd[k] for k in _GRAD_KEYS))
    if H11 is not None:
        A11, A22, A33 = operators.add_hog((A11, A22, A33), H11, operators.hog_symbol(grid, 2))
    return A11, A22, A12, A33


def _A_block(grid, Gd, H11=None):
    """_grad_A2d packed as (Ny,Nxr,3,3) blocks for the block operator."""
    A11, A22, A12, A33 = _grad_A2d(grid, Gd, H11)
    blk = np.zeros((grid["Ny"], grid["Nxr"], 3, 3), dtype=complex)
    blk[..., 0, 0] = A11
    blk[..., 0, 1] = A12
    blk[..., 1, 0] = A12
    blk[..., 1, 1] = A22
    blk[..., 2, 2] = A33
    return blk


def _apply_block_bands(sub, diag, sup, X):
    """Matrix action of a block-tridiagonal z operator. X: (Nz,...,3)."""
    def mm(M, v):
        return np.einsum("...ij,...j->...i", M, v)
    T = np.empty_like(X)
    T[0] = mm(diag[0], X[0]) + mm(sup[0], X[1])
    if X.shape[0] > 2:
        T[1:-1] = (mm(sub[:-1], X[:-2]) + mm(diag[1:-1], X[1:-1]) + mm(sup[1:], X[2:]))
    T[-1] = mm(sub[-1], X[-2]) + mm(diag[-1], X[-1])
    return T


# ==========================================================================
# H11 z-bands (SPEC V2-03 B5): (H11*hz/qw) * D2^T D2, D2 the INTERIOR second
# difference. Built by accumulating the rank-1 outer product of each
# interior row's (1,-2,1)/hz^2 stencil into five bands -- O(Nz), no dense
# matrix, and the boundary rows come out of the SAME accumulation (nothing
# is added for j=0 / j=Nz-1, which IS the natural BC).
# ==========================================================================
def _hog_z_bands(Nz, hz, H11, qw):
    """(sub2, sub1, diag, sup1, sup2), lengths (Nz-2, Nz-1, Nz, Nz-1, Nz-2).
    Same index convention as the tridiagonal bands: sub_p[i-p] is the entry
    at (row i, col i-p), sup_p[i] at (row i, col i+p)."""
    d0 = np.zeros(Nz)
    d1 = np.zeros(max(Nz - 1, 0))
    d2 = np.zeros(max(Nz - 2, 0))
    c = 1.0 / hz**2
    w = (c, -2.0 * c, c)
    for j in range(1, Nz - 1):                 # interior second differences only
        base = j - 1
        for p in range(3):
            d0[base + p] += w[p] * w[p]
            if p + 1 < 3:
                d1[base + p] += w[p] * w[p + 1]
            if p + 2 < 3:
                d2[base + p] += w[p] * w[p + 2]
    scale = H11 * hz / qw                      # (Nz,), the 1/qw force convention
    return (d2 * scale[2:], d1 * scale[1:], d0 * scale,
            d1 * scale[:-1], d2 * scale[:-2])


def _apply_pentadiag_z(bands, Pk):
    """Matrix action of a 1D (mode-independent) pentadiagonal z operator on a
    (3,Nz,Ny,Nxr) k-space field."""
    sub2, sub1, d0, sup1, sup2 = bands
    n = Pk.shape[1]
    out = d0[None, :, None, None] * Pk
    if n > 1:
        out[:, 1:] += sub1[None, :, None, None] * Pk[:, :-1]
        out[:, :-1] += sup1[None, :, None, None] * Pk[:, 1:]
    if n > 2:
        out[:, 2:] += sub2[None, :, None, None] * Pk[:, :-2]
        out[:, :-2] += sup2[None, :, None, None] * Pk[:, 2:]
    return out


# ==========================================================================
# Batched PENTADIAGONAL Thomas -- the B5 solver, mirroring
# factor_tridiag_batched/solve_tridiag_factored above (scalar) and
# film_elastic.factor_block_tridiag_batched/solve_block_tridiag_factored
# (3x3 blocks). Same L*U normalisation as those (U unit-diagonal), which is
# what makes H11 = 0 collapse them onto the tridiagonal pair BIT-FOR-BIT:
#   cc[i-1] = sub1[i-1] - sub2[i-2]*cp1[i-2]          (-> sub1[i-1])
#   piv[i]  = diag[i] - sub2[i-2]*cp2[i-2] - cc[i-1]*cp1[i-1]
#   cp1[i]  = (sup1[i] - cc[i-1]*cp2[i-1])/piv[i]     (-> sup1[i]/piv[i])
#   cp2[i]  = sup2[i]/piv[i]                          (-> 0)
# No pivoting, exactly as the tridiagonal pair: the implicit operator is
# I + dt*L*(PSD gradient + PSD H11) plus a diagonally dominant z part.
# ==========================================================================
def factor_pentadiag_batched(sub2, sub1, diag, sup1, sup2):
    """sub2,sup2: 1D (n-2,); sub1,sup1: 1D (n-1,); diag: (n,)+batch_shape."""
    n = diag.shape[0]
    batch = diag.shape[1:]
    cc = np.empty((max(n - 1, 0),) + batch)
    cp1 = np.empty((max(n - 1, 0),) + batch)
    cp2 = np.empty((max(n - 2, 0),) + batch)
    piv = np.empty((n,) + batch)
    piv[0] = diag[0]
    if n > 1:
        cp1[0] = sup1[0] / piv[0]
        if n > 2:
            cp2[0] = sup2[0] / piv[0]
        cc[0] = sub1[0]
        piv[1] = diag[1] - cc[0] * cp1[0]
        if n > 2:
            cp1[1] = (sup1[1] - cc[0] * cp2[0]) / piv[1]
        if n > 3:
            cp2[1] = sup2[1] / piv[1]
    for i in range(2, n):
        cc[i - 1] = sub1[i - 1] - sub2[i - 2] * cp1[i - 2]
        piv[i] = diag[i] - sub2[i - 2] * cp2[i - 2] - cc[i - 1] * cp1[i - 1]
        if i <= n - 2:
            cp1[i] = (sup1[i] - cc[i - 1] * cp2[i - 1]) / piv[i]
        if i <= n - 3:
            cp2[i] = sup2[i] / piv[i]
    return dict(sub2=sub2, cc=cc, cp1=cp1, cp2=cp2, piv=piv, n=n)


def solve_pentadiag_factored(factors, rhs):
    """rhs: (n,)+batch_shape[+trailing axes]; returns the same shape."""
    sub2, cc, cp1, cp2 = (factors["sub2"], factors["cc"], factors["cp1"], factors["cp2"])
    piv, n = factors["piv"], factors["n"]
    y = np.empty_like(rhs)
    y[0] = rhs[0] / piv[0]
    if n > 1:
        y[1] = (rhs[1] - cc[0] * y[0]) / piv[1]
    for i in range(2, n):
        y[i] = (rhs[i] - sub2[i - 2] * y[i - 2] - cc[i - 1] * y[i - 1]) / piv[i]
    x = np.empty_like(y)
    x[n - 1] = y[n - 1]
    if n > 1:
        x[n - 2] = y[n - 2] - cp1[n - 2] * x[n - 1]
    for i in range(n - 3, -1, -1):
        x[i] = y[i] - cp1[i] * x[i + 1] - cp2[i] * x[i + 2]
    return x


def factor_block_pentadiag_batched(sub2, sub1, diag, sup1, sup2):
    """3x3-block twin of factor_pentadiag_batched, in film_elastic's block
    conventions (piv_inv cached, einsum matmuls). Bands: (n-2|n-1|n, ...,3,3)."""
    n = diag.shape[0]
    batch = diag.shape[1:-2]

    def mm(A, B):
        return np.einsum("...ij,...jk->...ik", A, B)
    cc = np.empty((max(n - 1, 0),) + batch + (3, 3), dtype=complex)
    cp1 = np.empty((max(n - 1, 0),) + batch + (3, 3), dtype=complex)
    cp2 = np.empty((max(n - 2, 0),) + batch + (3, 3), dtype=complex)
    piv_inv = np.empty((n,) + batch + (3, 3), dtype=complex)
    piv_inv[0] = np.linalg.inv(diag[0])
    if n > 1:
        cp1[0] = mm(piv_inv[0], sup1[0])
        if n > 2:
            cp2[0] = mm(piv_inv[0], sup2[0])
        cc[0] = sub1[0]
        piv_inv[1] = np.linalg.inv(diag[1] - mm(cc[0], cp1[0]))
        if n > 2:
            cp1[1] = mm(piv_inv[1], sup1[1] - mm(cc[0], cp2[0]))
        if n > 3:
            cp2[1] = mm(piv_inv[1], sup2[1])
    for i in range(2, n):
        cc[i - 1] = sub1[i - 1] - mm(sub2[i - 2], cp1[i - 2])
        piv = diag[i] - mm(sub2[i - 2], cp2[i - 2]) - mm(cc[i - 1], cp1[i - 1])
        piv_inv[i] = np.linalg.inv(piv)
        if i <= n - 2:
            cp1[i] = mm(piv_inv[i], sup1[i] - mm(cc[i - 1], cp2[i - 1]))
        if i <= n - 3:
            cp2[i] = mm(piv_inv[i], sup2[i])
    return dict(sub2=sub2, cc=cc, cp1=cp1, cp2=cp2, piv_inv=piv_inv, n=n)


def solve_block_pentadiag_factored(factors, rhs):
    """rhs: (n,...,3); returns the same shape."""
    sub2, cc, cp1, cp2 = (factors["sub2"], factors["cc"], factors["cp1"], factors["cp2"])
    piv_inv, n = factors["piv_inv"], factors["n"]

    def mv(A, v):
        return np.einsum("...ij,...j->...i", A, v)
    y = np.empty_like(rhs)
    y[0] = mv(piv_inv[0], rhs[0])
    if n > 1:
        y[1] = mv(piv_inv[1], rhs[1] - mv(cc[0], y[0]))
    for i in range(2, n):
        y[i] = mv(piv_inv[i], rhs[i] - mv(sub2[i - 2], y[i - 2]) - mv(cc[i - 1], y[i - 1]))
    x = np.empty_like(y)
    x[n - 1] = y[n - 1]
    if n > 1:
        x[n - 2] = y[n - 2] - mv(cp1[n - 2], x[n - 1])
    for i in range(n - 3, -1, -1):
        x[i] = y[i] - mv(cp1[i], x[i + 1]) - mv(cp2[i], x[i + 2])
    return x


# ==========================================================================
# H11 energy / force helpers (SPEC V2-03 B5). Both are the H11 piece ALONE;
# the G piece is grad_energy_3d / grad_force_3d, and build_grad_factors
# combines the two into ONE (block-)pentadiagonal implicit operator.
# ==========================================================================
def hog_energy_film(P, grid, H11):
    """F_hog (a total, deck-energy units): area*sum_j qw_j <in-plane part> +
    area*hz*sum_{j=1..Nz-2} <z part>, the functional whose EXACT gradient is
    hog_force_film (K1 `fd_film_hog`)."""
    Ny, Nx = grid["Ny"], grid["Nx"]
    area = grid["Lx"] * grid["Ly"]
    qw, hz, Nz = grid["qw"], grid["hz"], grid["Nz"]
    Pk = _rfft2(P)
    Pxx = _irfft2(-(grid["KXr"]**2)[None, None, :, :] * Pk, (Ny, Nx))
    Pyy = _irfft2(-(grid["KYr"]**2)[None, None, :, :] * Pk, (Ny, Nx))
    f_in = 0.5 * H11 * (Pxx**2 + Pyy**2).sum(axis=0)
    F = area * float(np.sum(qw * f_in.mean(axis=(-2, -1))))
    if Nz >= 3:
        Q = (P[:, 2:] - 2.0 * P[:, 1:-1] + P[:, :-2]) / hz**2
        f_z = 0.5 * H11 * (Q**2).sum(axis=0)
        F += area * hz * float(np.sum(f_z.mean(axis=(-2, -1))))
    return F


def hog_energy_density_film(P, grid, H11):
    """hog_energy_film / Volume3D -- the module's own dens convention (see
    total_energy_3d). NOT a pointwise (Nz,Ny,Nx) field: F_hog is the sum of
    TWO different z quadratures (qw for the in-plane part, hz over interior
    nodes for the z part) and no single nodal array reproduces it exactly."""
    return hog_energy_film(P, grid, H11) / grid["Volume3D"]


def hog_force_film(P, grid, H11):
    """delta F_hog/delta P_i in this module's (1/qw) convention, (3,Nz,Ny,Nx)."""
    Ny, Nx = grid["Ny"], grid["Nx"]
    Pk = _rfft2(P)
    k4 = operators.hog_symbol(grid, 2)
    Fk = H11 * k4[None, None, :, :] * Pk
    Fk = Fk + _apply_pentadiag_z(_hog_z_bands(grid["Nz"], grid["hz"], H11, grid["qw"]), Pk)
    return _irfft2(Fk, (Ny, Nx))


# ==========================================================================
# Anisotropic gradient energy / force (SPEC V2-03 B4) and the shared
# public entry points. `G` is a scalar G11 (the original path) or a dict
# {G11,G12,G44,G44p}; `H11` is None (the original path) or a number.
# ==========================================================================
def _grad_force_block(P, grid, Gd, H11=None):
    """The ANISOTROPIC force, block path -- called directly by K2
    `film_grad_isotropic_byte` to measure the block assembly itself even
    where the public API dispatches to the scalar one."""
    Ny, Nx = grid["Ny"], grid["Nx"]
    sub, diag, sup = _grad_block_bands(grid, Gd)
    Pk = _rfft2(P)                                      # (3,Nz,Ny,Nxr)
    Tz = _apply_block_bands(sub, diag, sup, np.transpose(Pk, (1, 2, 3, 0)))
    Fk = np.transpose(Tz, (3, 0, 1, 2))
    A11, A22, A12, A33 = _grad_A2d(grid, Gd)
    Fk = Fk + np.stack([A11 * Pk[0] + A12 * Pk[1],
                        A12 * Pk[0] + A22 * Pk[1],
                        A33 * Pk[2]])
    F = _irfft2(Fk, (Ny, Nx))
    if H11 is not None:
        F = F + hog_force_film(P, grid, H11)
    return F


def _grad_energy_block(P, grid, Gd):
    """The ANISOTROPIC discrete gradient energy (a total), face functional
    included -- the exact functional whose gradient is _grad_force_block."""
    Ny, Nx = grid["Ny"], grid["Nx"]
    area = grid["Lx"] * grid["Ly"]
    qw, hz = grid["qw"], grid["hz"]
    G11, G12, G44, G44p = (Gd[k] for k in _GRAD_KEYS)
    gg, c44m = G44 + G44p, G44 - G44p

    Pk = _rfft2(P)
    A11, A22, A12, A33 = _grad_A2d(grid, Gd)
    inplane_force = _irfft2(np.stack([A11 * Pk[0] + A12 * Pk[1],
                                      A12 * Pk[0] + A22 * Pk[1], A33 * Pk[2]]), (Ny, Nx))
    f_in = 0.5 * np.sum(P * inplane_force, axis=0)
    F = area * float(np.sum(qw * f_in.mean(axis=(-2, -1))))

    aK, bK = operators.derivative_symbols_2d(grid)  # masked odd symbols, see the note
    Sx = _irfft2(aK[None, None, :, :] * Pk, (Ny, Nx))
    Sy = _irfft2(bK[None, None, :, :] * Pk, (Ny, Nx))
    D = (P[:, 1:] - P[:, :-1]) / hz
    Sxf = 0.5 * (Sx[:, 1:] + Sx[:, :-1])
    Syf = 0.5 * (Sy[:, 1:] + Sy[:, :-1])
    f_face = (0.5 * gg * (D[0]**2 + D[1]**2) + 0.5 * G11 * D[2]**2
              + G12 * D[2] * (Sxf[0] + Syf[1])
              + c44m * (D[0] * Sxf[2] + D[1] * Syf[2]))
    return F + area * hz * float(np.sum(f_face.mean(axis=(-2, -1))))


def grad_energy_3d(P, grid, G, H11=None):
    """F_grad, a TOTAL (not a density) -- the exact discrete functional whose
    gradient is grad_force_3d. The in-plane quadratic form uses the same even k^2 symbol as the force,
    including Nyquist modes that squared sampled first derivatives lose."""
    Ny, Nx = grid["Ny"], grid["Nx"]
    area = grid["Lx"] * grid["Ly"]
    qw, hz = grid["qw"], grid["hz"]
    Gs = _grad_scalar_G(G)
    if Gs is not None:
        G11 = Gs
        Pk = _rfft2(P)
        k2 = operators.k_squared(grid, 2)
        minus_laplacian = _irfft2(k2[None, None] * Pk, (Ny, Nx))
        fg_inplane = 0.5 * G11 * np.sum(P * minus_laplacian, axis=0)
        F_grad_inplane = area * float(np.sum(qw * fg_inplane.mean(axis=(-2, -1))))
        dPz = P[:, 1:] - P[:, :-1]
        F_grad_z = (G11 * area / (2 * hz)) * float(np.sum((dPz**2).mean(axis=(-2, -1))))
        F = F_grad_inplane + F_grad_z
    else:
        F = _grad_energy_block(P, grid, _grad_dict(G))
    if H11 is not None:
        F = F + hog_energy_film(P, grid, H11)
    return F


def grad_energy_density_3d(P, grid, G, H11=None):
    """grad_energy_3d / Volume3D -- the module's dens convention; see
    hog_energy_density_film for why this is a scalar and not a field."""
    return grad_energy_3d(P, grid, G, H11=H11) / grid["Volume3D"]


def _build_grad_factors_block(grid, G, dt, L, H11=None, sbdf2=False):
    """The 3x3 block implicit factors -- called directly by K2
    `film_grad_isotropic_byte` to measure the block path even where
    build_grad_factors dispatches to the scalar one."""
    from fftpf import film_elastic as fel     # circular at module scope
    Gd = _grad_dict(G)
    c0, c1 = (3.0, 2.0 * dt * L) if sbdf2 else (1.0, dt * L)
    sub_b, diag_b, sup_b = _grad_block_bands(grid, Gd)
    diag = c0 * np.eye(3) + c1 * (diag_b + _A_block(grid, Gd, H11)[None])
    sub, sup = c1 * sub_b, c1 * sup_b
    if H11 is None:
        f = fel.factor_block_tridiag_batched(sub, diag, sup)
        f["kind"] = "block"
        return f
    hs2, hs1, hd0, hu1, hu2 = _hog_z_bands(grid["Nz"], grid["hz"], H11, grid["qw"])
    eye = np.eye(3)
    f = factor_block_pentadiag_batched(
        c1 * hs2[:, None, None, None, None] * eye,
        sub + c1 * hs1[:, None, None, None, None] * eye,
        diag + c1 * hd0[:, None, None, None, None] * eye,
        sup + c1 * hu1[:, None, None, None, None] * eye,
        c1 * hu2[:, None, None, None, None] * eye)
    f["kind"] = "block_penta"
    return f


def _solve_grad_factored(factors, Xk):
    """Apply the inverse of whichever implicit gradient operator `factors`
    holds to a (3,Nz,Ny,Nxr) k-space field. The tri branch is step()'s own
    pre-V2-03 moveaxis/solve/moveaxis, unchanged."""
    kind = factors.get("kind", "tri")
    if kind in ("tri", "penta"):
        RHS = np.moveaxis(Xk, 1, 0)             # (Nz,3,Ny,Nxr): z first
        out = (solve_tridiag_factored(factors, RHS) if kind == "tri"
               else solve_pentadiag_factored(factors, RHS))
        return np.moveaxis(out, 0, 1)
    RHS = np.transpose(Xk, (1, 2, 3, 0))        # (Nz,Ny,Nxr,3): blocks last
    if kind == "block":
        from fftpf import film_elastic as fel
        out = fel.solve_block_tridiag_factored(factors, RHS)
    else:
        out = solve_block_pentadiag_factored(factors, RHS)
    return np.transpose(out, (3, 0, 1, 2))


def grad_force_3d(P, grid, G, H11=None):
    """delta F_grad/delta P_i, applied EXPLICITLY (matrix action, not a
    solve) -- used for diagnostics/energy-force self-checks and the SBDF2
    bootstrap's implicit reference; step()/step_sbdf2() apply the SAME
    operator IMPLICITLY via solve_tridiag_factored instead.

    SPEC V2-03 B4/B5: `G` is the scalar G11 (the ORIGINAL body below, kept
    verbatim) or a dict {G11,G12,G44,G44p} (the 3x3 block path); `H11` adds
    the higher-order-gradient term. Each branch is the EXACT gradient of
    grad_energy_3d's matching branch (K1 fd_film_grad_aniso / fd_film_hog)."""
    Gs = _grad_scalar_G(G)
    if Gs is None:
        return _grad_force_block(P, grid, _grad_dict(G), H11)
    G11 = Gs
    Nz, Ny, Nx = grid["Nz"], grid["Ny"], grid["Nx"]
    hz = grid["hz"]
    k2 = operators.k_squared(grid, 2)
    sub, diag, sup = _grad_z_bands(Nz, hz, G11)
    Pk = _rfft2(P)  # (3,Nz,Ny,Nxr)
    Tz = np.zeros_like(Pk)
    Tz[:, 0] = diag[0] * Pk[:, 0] + sup[0] * Pk[:, 1]
    Tz[:, 1:-1] = (sub[:-1, None, None] * Pk[:, :-2]
                   + diag[1:-1, None, None] * Pk[:, 1:-1]
                   + sup[1:, None, None] * Pk[:, 2:])
    Tz[:, -1] = sub[-1] * Pk[:, -2] + diag[-1] * Pk[:, -1]
    Fk = Tz + G11 * k2[None, None, :, :] * Pk
    F = _irfft2(Fk, (Ny, Nx))
    if H11 is not None:
        F = F + hog_force_film(P, grid, H11)
    return F


def build_grad_factors(grid, G, dt, L, H11=None, stab=0.0):
    """M(k) = I + dt*L*[grad force operator] -- SPEC's implicit gradient,
    factorized once per dt (mirrors solver.py's build_Minv).

    SPEC V2-03 B4/B5. `G` scalar (or a dict that reduces EXACTLY to the
    isotropic case, see _grad_scalar_G) and `H11=None` take the ORIGINAL
    three lines, unchanged -> a "tri" factorization. `H11` given adds the
    pentadiagonal z part + H11*(kx^4+ky^4); a genuinely anisotropic dict `G`
    makes the operator 3x3 block-(tri|penta)diagonal. The `kind` tag tells
    _solve_grad_factored which solver to use; it is the only new dict key,
    and solve_tridiag_factored ignores it (kittel_afe / octa_tilt call that
    directly with these factors).

    H11 = 0.0 is NOT short-circuited: the pentadiagonal recurrences collapse
    onto the tridiagonal ones bit-for-bit on exactly-zero outer bands, which
    is what K2 `film_hog_h11_zero_byte` measures.

    stab=S > 0 (opt-in linear stabilization, docs/ADAPTIVE_DT.md) adds
    dt*L*S to the diagonal and records S in the factors ("stab" key); step()
    must get the same S. Supported on the tri/penta (scalar G) operators;
    the anisotropic 3x3 block path raises NotImplementedError. stab=0 is the
    original code path, bit for bit."""
    Gs = _grad_scalar_G(G)
    if Gs is None:
        if stab:
            raise NotImplementedError("film3d.build_grad_factors: stab is supported for a "
                                      "scalar (isotropic) G only, not the 3x3 block operator")
        return _build_grad_factors_block(grid, G, dt, L, H11=H11)
    G11 = Gs
    Nz = grid["Nz"]
    hz = grid["hz"]
    k2 = operators.k_squared(grid, 2)
    sub_z, diag_z, sup_z = _grad_z_bands(Nz, hz, G11)
    if stab:
        from fftpf import timestep_adaptive as adaptive
        stab = adaptive.check_stab(stab)
    if H11 is None:
        sub = dt * L * sub_z
        sup = dt * L * sup_z
        diag = 1.0 + dt * L * diag_z[:, None, None] + dt * L * G11 * k2[None, :, :]
        if stab:
            diag = diag + dt * L * stab
        factors = factor_tridiag_batched(sub, diag, sup)
        factors["kind"] = "tri"
        return adaptive.tag_operator(factors, stab) if stab else factors
    hs2, hs1, hd0, hu1, hu2 = _hog_z_bands(Nz, hz, H11, grid["qw"])
    k4 = operators.hog_symbol(grid, 2)
    diag = (1.0 + dt * L * (diag_z + hd0)[:, None, None] + dt * L * G11 * k2[None, :, :]
            + dt * L * H11 * k4[None, :, :])
    if stab:
        diag = diag + dt * L * stab
    factors = factor_pentadiag_batched(dt * L * hs2, dt * L * (sub_z + hs1), diag,
                                       dt * L * (sup_z + hu1), dt * L * hu2)
    factors["kind"] = "penta"
    return adaptive.tag_operator(factors, stab) if stab else factors


def build_grad_factors_hog(grid, G, H11, dt, L):
    """SPEC V2-03 B5's own spelling of build_grad_factors(..., H11=H11)."""
    return build_grad_factors(grid, G, dt, L, H11=H11)


def build_grad_factors_sbdf2(grid, G, dt, L, H11=None, stab=0.0):
    """M2 = 3I + 2*dt*L*[grad force operator] (SBDF2's implicit operator,
    same pattern as solver.py's build_Minv2). Same V2-03 B4/B5 contract as
    build_grad_factors; the scalar/H11=None path is the original code.
    stab=S > 0 adds 2*dt*L*S to the diagonal (build_grad_factors' contract)."""
    Gs = _grad_scalar_G(G)
    if Gs is None:
        if stab:
            raise NotImplementedError("film3d.build_grad_factors_sbdf2: stab is supported for "
                                      "a scalar (isotropic) G only, not the 3x3 block operator")
        return _build_grad_factors_block(grid, G, dt, L, H11=H11, sbdf2=True)
    G11 = Gs
    Nz = grid["Nz"]
    hz = grid["hz"]
    k2 = operators.k_squared(grid, 2)
    sub_z, diag_z, sup_z = _grad_z_bands(Nz, hz, G11)
    if stab:
        from fftpf import timestep_adaptive as adaptive
        stab = adaptive.check_stab(stab)
    if H11 is None:
        sub = 2 * dt * L * sub_z
        sup = 2 * dt * L * sup_z
        diag = 3.0 + 2 * dt * L * diag_z[:, None, None] + 2 * dt * L * G11 * k2[None, :, :]
        if stab:
            diag = diag + 2 * dt * L * stab
        factors = factor_tridiag_batched(sub, diag, sup)
        factors["kind"] = "tri"
        return adaptive.tag_operator(factors, stab) if stab else factors
    hs2, hs1, hd0, hu1, hu2 = _hog_z_bands(Nz, hz, H11, grid["qw"])
    k4 = operators.hog_symbol(grid, 2)
    diag = (3.0 + 2 * dt * L * (diag_z + hd0)[:, None, None]
            + 2 * dt * L * G11 * k2[None, :, :] + 2 * dt * L * H11 * k4[None, :, :])
    if stab:
        diag = diag + 2 * dt * L * stab
    factors = factor_pentadiag_batched(2 * dt * L * hs2, 2 * dt * L * (sub_z + hs1), diag,
                                       2 * dt * L * (sup_z + hu1), 2 * dt * L * hu2)
    factors["kind"] = "penta"
    return adaptive.tag_operator(factors, stab) if stab else factors


# ==========================================================================
# TDGL stepping (SPEC §13.0: gradient implicit; bulk + electrostatic force
# EXPLICIT, route A). SPEC §13.3: elastic_force_film wired in as a THIRD
# explicit force, via an optional `el` param (film_elastic.setup_film_elastic
# output) -- default None reproduces WP7a's no-elasticity behavior exactly
# (every WP7a/WP7b call site omits `el`, so the regression suite is
# untouched). This mirrors solver.py's WP1->WP2 `elastic_params=None`
# precedent exactly (see solver.py's step/_explicit_N/relax). film_elastic
# is imported LAZILY (inside the `el is not None` branch, not at module
# top) because film_elastic.py itself imports film3d -- a top-level import
# here would be circular; a local import is the standard break for that,
# and costs nothing extra since Python caches the module after first import.
# ==========================================================================
def explicit_N(P, alpha, grid, poisson_factors, E_ext=(0.0, 0.0, 0.0), el=None):
    """N(P) is this module's mass-matrix-weighted TDGL RHS: qw(z)*dP(z)/dt =
    -L*dF/dP(z), i.e. N(z) := (1/qw(z))*dF/dP(z) for EVERY term (this is
    what makes bulk_force/grad_force_3d/elec_force_3d usable here directly --
    bulk_force is pointwise (no z-coupling, so trivially (1/qw)-consistent);
    grad_force_3d is EXPLICITLY built this way per its own docstring
    ("divide by qw"); elec_force_3d likewise, by construction).
    elastic_force_film is NOT built this way -- film_elastic.py's own
    (validated, G11f/G11e) convention integrates against a UNIFORM hz*hx*hy
    per P-node (interior AND both z-boundaries alike), i.e.
    dF_elastic/dP(z) = hz*elastic_force_film(z) uniformly, unlike bulk/grad/
    elec's qw(z)-varying (hz at interior, hz/2 at the 2 z-boundaries)
    density convention. So elastic_force_film's raw output needs an extra
    hz/qw(z) rescale here to become (1/qw)-consistent: 1x at interior
    nodes, 2x at the two z-boundary nodes (z=0 film/substrate interface,
    z=hf free surface). Caught by a direct check (not assumed): the
    unscaled sum left the WP6 Tier-1 minimizer P_min NOT a stationary point
    of step() at the two z-boundary nodes (residual N ~0.024, vs ~7e-10 at
    every interior node and ~7e-10 everywhere once this scaling is applied)
    -- see WP7C_REPORT.md. G12a's own FD gate is unaffected (it evaluates
    F via total_energy_3d and the analytic force by combining bulk/grad/
    elec [qw-weighted] and elastic [uniform-hz-weighted] as two SEPARATE
    integrals, exactly this function's post-fix behavior, never through the
    old unscaled explicit_N)."""
    Fb = solver.bulk_force(P, alpha)  # reused unchanged, see module docstring
    Fe = elec_force_3d(P, grid, poisson_factors, E_ext)
    N = Fb + Fe
    if el is not None:
        from fftpf import film_elastic as fel
        if "_force_solver" in el:
            Fel = el["_force_solver"](P, el=el)
        else:
            Fel, _ = fel.elastic_force_film(P, el)
        qw_ratio = (grid["hz"] / grid["qw"])[None, :, None, None]  # 1 interior, 2 at both z-boundaries
        N = N + Fel * qw_ratio
    return N


def step(P, alpha, grid, grad_factors, poisson_factors, dt, L, E_ext=(0.0, 0.0, 0.0), el=None,
         stab=0.0):
    """One semi-implicit step, gradient implicit in z+k: P^(n+1) = grad_factors^-1 . [P^n_hat - dt*L*FFT(N(P^n))].
    stab=S > 0: RHS = (1+dt*L*S)*P^n_hat - dt*L*FFT(N), grad_factors from
    build_grad_factors(..., stab=S) (refused otherwise). stab=0 is the
    original arithmetic, bit for bit."""
    if stab:
        from fftpf import timestep_adaptive as adaptive
        adaptive.stabilized_operator(grad_factors, stab, "film3d.step")
    elif isinstance(grad_factors, dict) and grad_factors.get("stab"):
        raise ValueError("film3d.step: grad_factors were built with stab > 0; pass the same stab")
    if solver._FUSED is not None:   # opt-in, execution.configure(..., fused=True)
        P_new = solver._FUSED.film_step("sbdf1", P, None, alpha, grid, grad_factors,
                                        poisson_factors, dt, L, E_ext, el, stab)
        if P_new is not None:
            return P_new
    N = explicit_N(P, alpha, grid, poisson_factors, E_ext, el=el)
    Pk = _rfft2(P)
    Nk = _rfft2(N)
    if stab:
        Pk_new = _solve_grad_factored(grad_factors, (1.0 + dt * L * stab) * Pk - dt * L * Nk)
    else:
        Pk_new = _solve_grad_factored(grad_factors, Pk - dt * L * Nk)
    return _irfft2(Pk_new, (grid["Ny"], grid["Nx"]))


def step_sbdf2(P_n, P_nm1, alpha, grid, grad_factors2, poisson_factors, dt, L, E_ext=(0.0, 0.0, 0.0), el=None,
               stab=0.0):
    """One SBDF2 step from (P_n, P_nm1), N evaluated at the extrapolant P* = 2*P_n - P_nm1; grad_factors2 from build_grad_factors_sbdf2.
    stab=S > 0: S*P implicit at n+1 and explicit at P* (RHS + 2*dt*L*S*FFT(P*)),
    grad_factors2 from build_grad_factors_sbdf2(..., stab=S). stab=0 is the
    original arithmetic, bit for bit."""
    if stab:
        from fftpf import timestep_adaptive as adaptive
        adaptive.stabilized_operator(grad_factors2, stab, "film3d.step_sbdf2")
    elif isinstance(grad_factors2, dict) and grad_factors2.get("stab"):
        raise ValueError("film3d.step_sbdf2: grad_factors2 were built with stab > 0; pass the same stab")
    if solver._FUSED is not None:   # opt-in, execution.configure(..., fused=True)
        P_new = solver._FUSED.film_step("sbdf2", P_n, P_nm1, alpha, grid, grad_factors2,
                                        poisson_factors, dt, L, E_ext, el, stab)
        if P_new is not None:
            return P_new
    P_star = 2 * P_n - P_nm1
    N = explicit_N(P_star, alpha, grid, poisson_factors, E_ext, el=el)
    Pk_n = _rfft2(P_n)
    Pk_nm1 = _rfft2(P_nm1)
    Nk = _rfft2(N)
    if stab:
        Pk_new = _solve_grad_factored(grad_factors2, 4 * Pk_n - Pk_nm1 - 2 * dt * L * Nk
                                      + (2 * dt * L * stab) * (2 * Pk_n - Pk_nm1))
    else:
        Pk_new = _solve_grad_factored(grad_factors2, 4 * Pk_n - Pk_nm1 - 2 * dt * L * Nk)
    return _irfft2(Pk_new, (grid["Ny"], grid["Nx"]))


def _probe_run(P0, alpha, grid, G, poisson_factors, L, E_ext, dt, n_probe, p_cap, scheme,
               el=None, H11=None):
    if scheme == "sbdf1":
        gf = build_grad_factors(grid, G, dt, L, H11=H11)
        P = P0.copy()
        for _ in range(n_probe):
            P = step(P, alpha, grid, gf, poisson_factors, dt, L, E_ext, el=el)
            if not np.all(np.isfinite(P)) or np.max(np.abs(P)) > p_cap:
                return False
        return True
    elif scheme == "sbdf2":
        gf1 = build_grad_factors(grid, G, dt, L, H11=H11)
        gf2 = build_grad_factors_sbdf2(grid, G, dt, L, H11=H11)
        P_nm1 = P0.copy()
        P_n = step(P_nm1, alpha, grid, gf1, poisson_factors, dt, L, E_ext, el=el)
        if not np.all(np.isfinite(P_n)) or np.max(np.abs(P_n)) > p_cap:
            return False
        for _ in range(n_probe - 1):
            P_new = step_sbdf2(P_n, P_nm1, alpha, grid, gf2, poisson_factors, dt, L, E_ext, el=el)
            if not np.all(np.isfinite(P_new)) or np.max(np.abs(P_new)) > p_cap:
                return False
            P_nm1, P_n = P_n, P_new
        return True
    raise ValueError(f"unknown scheme {scheme!r}")


def probe_stable_dt(P0, alpha, grid, G, kappa_b, L, E_ext=(0.0, 0.0, 0.0), dt0=0.02,
                     n_probe=200, growth=2.0, p_cap=None, max_doublings=30, scheme="sbdf1",
                     el=None, H11=None):
    """SPEC §2's protocol, adapted (same as solver.py's probe_stable_dt).
    el=None (default): route-A stability without elasticity, WP7a behavior
    UNCHANGED. el=<film_elastic setup>: SPEC §13.3's "re-probe stable dt
    with elasticity on" -- elastic_force_film is explicit (route A), so it
    only affects the STABILITY of the explicit part being probed here, not
    the implicit grad_factors machinery itself."""
    if p_cap is None:
        p_cap = max(10.0 * np.max(np.abs(P0)), 5.0)
    poisson_factors = poisson_factor(grid, kappa_b)
    dt = dt0
    last_stable = None
    for _ in range(max_doublings):
        if not _probe_run(P0, alpha, grid, G, poisson_factors, L, E_ext, dt, n_probe, p_cap,
                          scheme, el=el, H11=H11):
            break
        last_stable = dt
        dt *= growth
    if last_stable is None:
        raise RuntimeError(f"film3d.probe_stable_dt: dt0={dt0} already unstable ({n_probe} steps)")
    return last_stable / 2.0


# ==========================================================================
# Energies (SPEC §2: "compute ALL energies in real space"). z-integration
# uses trapezoidal quadrature (qw: hz/2 at the two boundary nodes) for the
# NODAL densities (bulk, elec, in-plane grad); the z-gradient term uses its
# own FACE-based total (see module docstring) since it is not a nodal
# density -- both are exactly consistent with grad_force_3d's derivation.
# ==========================================================================
def total_energy_3d(P, alpha, grid, kappa_b, G, poisson_factors=None, E_ext=(0.0, 0.0, 0.0),
                    el=None, H11=None):
    """Returns (densities dict, totals dict): bulk/grad/elec[/hog][/elastic]/total.
    SPEC V2-03 B4/B5: `G` is a scalar G11 or a {G11,G12,G44,G44p} dict, and
    `H11` (default None -> no "hog" key, totals bit-identical) adds the
    higher-order-gradient energy.
    Density = total / Volume3D (Lx*Ly*hf), matching solver.py's dens/tot
    pairing. el=None (default): WP7a behavior unchanged, no "elastic" key.
    el=<film_elastic setup>: integrates relaxed elastic energy over ALL
    stack faces, including substrate (area*hz*sum(mean)). Its derivative
    acts on film P only. Dividing by film volume reports energy per film
    volume; film-only elastic energy is not the relaxed system functional."""
    if poisson_factors is None:
        poisson_factors = poisson_factor(grid, kappa_b)
    area = grid["Lx"] * grid["Ly"]
    qw = grid["qw"]

    fb = solver.bulk_energy_density(P, alpha)  # (Nz,Ny,Nx)
    F_bulk = area * float(np.sum(qw * fb.mean(axis=(-2, -1))))

    # Use the same quadratic form as the gradient force, including Nyquist.
    F_grad = grad_energy_3d(P, grid, G)

    fe = elec_energy_density_3d(P, grid, poisson_factors, E_ext)
    F_elec = area * float(np.sum(qw * fe.mean(axis=(-2, -1))))

    tot = dict(bulk=F_bulk, grad=F_grad, elec=F_elec)
    if H11 is not None:
        tot["hog"] = hog_energy_film(P, grid, H11)
    if el is not None:
        from fftpf import film_elastic as fel
        sol = fel.elastic_solve_film(P, el)
        # The displacement is relaxed on the whole stack. Its substrate
        # strain energy also varies with P and belongs to the potential.
        dens_el = 0.5 * np.einsum("ije...,ije...->e...", sol["sigma_face"], sol["e_face"])
        tot["elastic"] = area * el["hz"] * float(np.sum(dens_el.mean(axis=(-2, -1))))
    tot["total"] = sum(tot.values())
    dens = {k: v / grid["Volume3D"] for k, v in tot.items()}
    return dens, tot


def relax(P0, alpha, grid, G, kappa_b, L, E_ext=(0.0, 0.0, 0.0), dt=None, tol_F=1e-8,
          n_max=20000, check_every=10, n_plateau=20, dt0=0.02, n_probe=200, scheme="sbdf1",
          el=None, H11=None):
    """Minimal relax loop (no dt_adapt/tol_P/post_step -- not needed by any
    WP7a gate; solver.py's fuller relax() is the model this deliberately
    trims, see film3d module docstring). el=None (default): WP7a behavior
    unchanged. el=<film_elastic setup> (SPEC §13.3): threaded through
    probe_stable_dt/step/step_sbdf2/total_energy_3d -- same "no change when
    omitted" contract as every other function in this module."""
    poisson_factors = poisson_factor(grid, kappa_b)
    if dt is None:
        dt = probe_stable_dt(P0, alpha, grid, G, kappa_b, L, E_ext=E_ext, dt0=dt0,
                              n_probe=n_probe, scheme=scheme, el=el, H11=H11)
    gf1 = build_grad_factors(grid, G, dt, L, H11=H11)
    hist = []
    F_prev = None
    plateau = 0

    def _F(P):
        _, tot = total_energy_3d(P, alpha, grid, kappa_b, G, poisson_factors=poisson_factors,
                                 E_ext=E_ext, el=el, H11=H11)
        return tot["total"]

    def _check(n, P, hist, F_prev, plateau):
        F = _F(P)
        hist.append((n, F))
        stop = False
        if F_prev is not None:
            if abs(F - F_prev) / (check_every * dt) < tol_F:
                plateau += 1
                stop = plateau >= n_plateau
            else:
                plateau = 0
        return stop, F, plateau

    if scheme == "sbdf1":
        P = P0.copy()
        n_run = 0
        for n in range(1, n_max + 1):
            P_new = step(P, alpha, grid, gf1, poisson_factors, dt, L, E_ext, el=el)
            if not np.all(np.isfinite(P_new)):
                return P, dt, n_run, hist
            P = P_new
            n_run = n
            if n % check_every == 0:
                stop, F_prev, plateau = _check(n, P, hist, F_prev, plateau)
                if stop:
                    break
        return P, dt, n_run, hist

    elif scheme == "sbdf2":
        gf2 = build_grad_factors_sbdf2(grid, G, dt, L, H11=H11)
        P_nm1 = P0.copy()
        P_n = step(P_nm1, alpha, grid, gf1, poisson_factors, dt, L, E_ext, el=el)
        n_run = 1
        for n in range(2, n_max + 1):
            P_new = step_sbdf2(P_n, P_nm1, alpha, grid, gf2, poisson_factors, dt, L, E_ext, el=el)
            if not np.all(np.isfinite(P_new)):
                return P_n, dt, n_run, hist
            P_nm1, P_n = P_n, P_new
            n_run = n
            if n % check_every == 0:
                stop, F_prev, plateau = _check(n, P_n, hist, F_prev, plateau)
                if stop:
                    break
        return P_n, dt, n_run, hist

    raise ValueError(f"unknown scheme {scheme!r}")


if __name__ == "__main__":
    g = paper_grid()
    print(f"paper grid: {g['Nx']}x{g['Ny']}x{g['Nz']}  Lx=Ly={g['Lx']}nm  hf={g['hf']}nm  hz={g['hz']}nm")
    print(f"KAPPA_B_DECK={KAPPA_B_DECK:.6f}  KAPPA_B_PAPER_LINEAGE={KAPPA_B_PAPER_LINEAGE:.6f}")
