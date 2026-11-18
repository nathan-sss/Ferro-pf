"""film_elastic.py -- WP7b (SPEC.md §13.2): per-mode block-tridiagonal FD
film elasticity for the true film+substrate geometry (Li & Chen APL 88,
072905 (2006); elastic solution method of Li, Hu, Liu, Chen, Acta Mater 50
(2002) 395, "Li 2002" below).

SCOPE (§13.2, superseding the WP7a-era placeholder): u(k1,k2,z), 3x3 blocks,
free-surface top (sigma_i3=0), Dirichlet substrate bottom (u=0), substrate
elastic and eigenstrain-free, in-plane misfit e0 imposed per Li 2002 §2.
Elastic driving force on P via -sigma:d(eps*)/dP as in elastic.py. This
module owns no stepping loop (mirrors elastic.py's own WP2-era isolation),
but IS wired into film3d.py: film3d.explicit_N calls elastic_force_film and
film3d.total_energy_3d calls elastic_solve_film, both behind their optional
`el=` argument (WP7c "full coupling", SPEC §13.3). GATES: G11a-f (this
module standalone), G12a/G12b (the coupled el= path, `from solvers import
film_elastic as fe`).

ARCHITECTURE (SPEC §13.0/13.2, binding): FFT in-plane (x1,x2) exactly as
film3d.py; second-order FD in x3 over the FULL stack (substrate h_s=12dx +
film h_f=20dx, isotropic dx, reusing film3d.PAPER_DX_NM/PAPER_HF_DX). Every
linear solve is per-(k1,k2)-mode BLOCK-tridiagonal (3x3 blocks, one block
per z-node), assembled and solved with a batched BLOCK Thomas algorithm
(factor_block_tridiag_batched/solve_block_tridiag_factored below) -- no
global 3D dense operator, matching SPEC's explicit ban.

DISCRETIZATION METHOD (energy-consistent FEM, chosen over a naive collocated
strong-form FD stencil): the equilibrium PDE is c_ijkl u_k,lj = c_ijkl
eps*_kl,j (Li 2002 eq 20) with free-surface/Dirichlet BCs (eq 21/22). Rather
than discretizing this PDE directly (risking a non-self-adjoint operator --
exactly film3d.py's own documented "DISCRETE NOTE" caveat for its Poisson
solve), this module discretizes the underlying ELASTIC ENERGY functional
with 2-node ("element") pieces spanning each z-face, giving an EXACT natural
(traction-free) BC at the open top end for free (no element beyond the last
node contributes there), mirroring exactly how film3d.py's own _grad_z_bands
was derived from a discrete finite-volume functional. The local element
strain-vs-dof map and the C-tensor contraction are built SYMBOLICALLY
(sympy) from the full eps*_ij/C_ijkl tensor contraction (elastic.py's "do
NOT hand-decouple" philosophy: generic contractions, not per-component
formulas) once at import time (_derive_local_element). The per-mode
equilibrium operator K(k) is the HERMITIAN (conjugated/sesquilinear)
pairing hz*X(-k)^T@Cmat@X(k) -- NOT the plain bilinear X(k)^T@Cmat@X(k) a
naive real-symbol Hessian dE/du=0 would give, which is exactly symmetric
but Hermitian only at k=0 (see _derive_local_element's docstring for the
root-cause account of this, fixed post-WP7B; self-adjointness of the
CORRECT sesquilinear form is what's guaranteed, not of the naive one).

WHERE ebar (Li 2002's macroscopic misfit strain e0) ENTERS -- worked out
carefully (a first-pass, more naive treatment gives a DIFFERENT, wrong,
answer -- see git history / WP7B_REPORT.md "ebar is film-only" note): Li
2002 states explicitly (§3, eq 22 discussion) that the substrate is
undisturbed "at a distance x3=-h_s ... beyond which the elastic deformation
in the substrate is ignored" -- i.e. ebar is the FILM's own shape
deformation relative to ITS OWN stress-free lattice, and must NOT be
applied to the substrate (a substrate permanently carrying the full
mismatch strain far from the interface would contradict "sufficiently
thick substrate ... totally controls" the film, and would fight the
u(-h_s)=0 boundary condition). Since ebar varies only with z (uniform in
x1,x2: e0 within the film, 0 in the substrate) it is exactly a k1=k2=0-only
physical effect -- so rather than adding it as a special post-hoc constant
(elastic.py's L1-periodic-model approach, forced by having no z-axis to
place it on), this module folds ebar directly into the SAME per-face
eigenstrain SOURCE field fed to the general per-mode solver:
    src_ij(z) = eps*_ij(z) - ebar_ij*[1 if z in film else 0]
so "strain(u) - src" is the correct total elastic strain (ebar_ij + eta_ij
- eps*_ij) at every z, and the k=0 mode of the SAME general machinery
automatically reproduces the paper's own u^A+u^B split -- no special-casing
needed anywhere else in the code (verified against an independent 1D
closed-form derivation, gate G11a).

Reuses elastic.py's eigenstrain()/build_C_tensor()/contract_C()/
elastic_force_explicit() UNCHANGED (broadcast-safe over any trailing
shape). Reuses film3d.py's rfft2/irfft2 helpers and PAPER_DX_NM/PAPER_HF_DX/
PAPER_NZ (one-way import, film3d does not depend on this module -- no
circularity, unlike elastic.py<->solver.py's documented reason for
duplicating that pair instead).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import sympy as sp

from fftpf import elastic_base as elastic
from fftpf import film_engine as f3
from fftpf import input_validation as validate

PAPER_HS_DX = 12  # substrate thickness in dx units, SPEC §13.0 "hs=12dx"


# ==========================================================================
# Stack grid: extend film3d's film-only z-grid downward to include the
# substrate. Node 0 = z=-hs (Dirichlet, dropped from the solve); nodes
# film_lo..Nz-1 (film_lo=Nz_s) are the SAME nodes film3d.py's own film-only
# grid uses (z=0..hf), so P (which lives only there) plugs in unchanged.
# ==========================================================================
def make_stack_grid(film_grid, hs_dx=PAPER_HS_DX):
    """Extend a film3d z-grid downward by hs_dx substrate nodes; adds Nz_film/Nz_s/hs/film_lo, film node indices unchanged."""
    hz = film_grid["hz"]
    Nz_film = film_grid["Nz"]
    hs = hs_dx * hz
    z_sub = -hz * np.arange(hs_dx, 0, -1)      # hs_dx points, z=-hs..-hz
    z = np.concatenate([z_sub, film_grid["z"]])  # length Nz_s+Nz_film
    grid = dict(film_grid)
    grid.update(Nz_film=Nz_film, Nz=hs_dx + Nz_film, Nz_s=hs_dx, hs=hs,
                z=z, film_lo=hs_dx)
    return grid


def paper_stack_grid(hs_dx=PAPER_HS_DX, **kwargs):
    """make_stack_grid over film3d.paper_grid(**kwargs) -- SPEC §13.0's film+substrate stack at the paper's own size."""
    return make_stack_grid(f3.paper_grid(**kwargs), hs_dx=hs_dx)


# ==========================================================================
# ebar (Li 2002 §2/§3): e0 in-plane misfit, ebar_33 from sigma_33bar=0 using
# ONLY C11/C12 (cubic, axes aligned with the crystal frame => c_i3kl=0 for
# i=1,2 and k,l in-plane, so ebar_13=ebar_23=0 trivially satisfies Li 2002's
# "choose ebar_i3 s.t. c_i3kl*ebar_kl=0" for i=1,2 without solving anything).
# NOTE this is A DIFFERENT, SIMPLER formula than elastic.py's
# homogeneous_strain_misfit/_film_full: those carry an <eps*> mean term
# because the L1/WP6 periodic models have NO literal substrate region for
# the eigenstrain's average to be absorbed into (the whole periodic cell
# IS "the film"); here a real substrate exists and Li 2002's ebar is
# defined independently of the eigenstrain entirely (module docstring).
# ==========================================================================
def homogeneous_strain_film_e0(e0, C11, C12):
    """Li 2002 substrate-clamped homogeneous strain for misfit e0: diag(e0, e0, -2*(C12/C11)*e0), shape (3,3)."""
    ebar = np.zeros((3, 3))
    ebar[0, 0] = e0
    ebar[1, 1] = e0
    ebar[2, 2] = -(C12 / C11) * 2.0 * e0
    return ebar


# ==========================================================================
# Local 2-node ("element") energy, symbolic derivation (module docstring).
# One element spans one z-face: dof = [u1n,u2n,u3n, u1p,u2p,u3p] (node n =
# "below"/z_j, node p = "above"/z_j+1). Strain-at-face uses average
# interpolation for in-plane-derivative components (e11,e22,e12) and
# central FD/hz for out-of-plane ones (e33,e13,e23) -- SAME formula used
# numerically in _face_strain below (copied, not re-derived, so the
# assembled K/M operator and the readback strain are guaranteed consistent
# -- this is exactly what gate G11f checks).
# ==========================================================================
def _face_strain_symbolic(iq1, iq2, hz, u1n, u2n, u3n, u1p, u2p, u3p):
    e11 = iq1 * (u1n + u1p) / 2
    e22 = iq2 * (u2n + u2p) / 2
    e33 = (u3p - u3n) / hz
    e12 = (iq2 * (u1n + u1p) / 2 + iq1 * (u2n + u2p) / 2) / 2
    e13 = ((u1p - u1n) / hz + iq1 * (u3n + u3p) / 2) / 2
    e23 = ((u2p - u2n) / hz + iq2 * (u3n + u3p) / 2) / 2
    return e11, e22, e33, e12, e13, e23


def _face_strain(u_n, u_p, iq1, iq2, hz):
    """Numeric (numpy) twin of _face_strain_symbolic, evaluated on actual
    field arrays. u_n,u_p: (3,...) [u1,u2,u3 first axis]. iq1,iq2: real k
    arrays (NOT pre-multiplied by 1j -- multiplied by 1j here, matching the
    symbolic convention exactly) broadcastable against u_n[0]. Returns the
    6 face-strain components (e11,e22,e33,e12,e13,e23), each shape
    matching np.broadcast(u_n[0], iq1)."""
    u1n, u2n, u3n = u_n[0], u_n[1], u_n[2]
    u1p, u2p, u3p = u_p[0], u_p[1], u_p[2]
    a, b = 1j * iq1, 1j * iq2
    e11 = a * (u1n + u1p) / 2
    e22 = b * (u2n + u2p) / 2
    e33 = (u3p - u3n) / hz
    e12 = (b * (u1n + u1p) / 2 + a * (u2n + u2p) / 2) / 2
    e13 = ((u1p - u1n) / hz + a * (u3n + u3p) / 2) / 2
    e23 = ((u2p - u2n) / hz + b * (u3n + u3p) / 2) / 2
    return e11, e22, e33, e12, e13, e23


def _derive_local_element():
    """Derives X0,X1,X2 (the strain-vs-dof map X(iq1,iq2)=X0+X1*iq1+X2*iq2 --
    6 strain components x 6 dof, AFFINE in iq1,iq2 since each strain
    component is built from a SINGLE spatial derivative of u -- see
    _face_strain_symbolic, no iq1*iq2 or iq^2 term ever appears there) and
    Cmat (the (6,6) real symmetric matrix s.t. energy_half=0.5*d^T@Cmat@d
    reproduces the SAME cubic-C contraction, factor-2-on-shear convention,
    SPEC §1.4).

    ROOT-CAUSE FIX (WP7B_REPORT.md "Hermiticity" finding): the per-mode
    stiffness/RHS must be assembled as the CONJUGATED (Hermitian/sesquilinear)
    pairing K(k) = hz*X(-k)^T @ Cmat @ X(k) [= hz*X(k)^H @ Cmat @ X(k), since
    X(-k)=conj(X(k)) for real k], NOT the plain (bilinear) X(k)^T @ Cmat @
    X(k). The previous implementation built a SINGLE formal Hessian
    K_local(iq1,iq2) = X(iq1,iq2)^T @ Cmat @ X(iq1,iq2) by differentiating a
    real-symbol energy expression (dof, iq1, iq2 all sympy `real=True`
    symbols) and only substituted the actual complex value iq1=i*k1 (etc.)
    AFTER differentiating. That is mathematically exact ONLY if "dof" is a
    real nodal displacement (ordinary real-space FEM) -- but here "dof" is
    reinterpreted as a COMPLEX per-mode Fourier amplitude, for which the
    stationarity condition of a REAL physical energy requires Wirtinger
    (conjugate) calculus, not a naive real-symbol Jacobian. Consequence:
    K_local(iq1,iq2), being a plain Hessian, is symmetric for ANY iq1,iq2
    (K_local=K_local^T identically, even before substitution) -- so
    substituting iq->ik gives an operator that is EXACTLY symmetric
    (K=K^T) but Hermitian ONLY at k=0 (where X is real, so ^T=^H trivially).
    For k!=0 this is a genuinely different (and, checked directly: unphysical
    -- non-real eigenvalues) operator, not a small correction: e.g. the
    iq1^2-coefficient term picks up the WRONG SIGN relative to the correct
    Hermitian pairing (the textbook "naive Hessian of a formal real
    expression gives -k^2 instead of the correct +k^2 for a gradient-energy
    operator" trap). elastic.py's L1 Omega(k)=[C_ijkl k_j k_l]^-1 sidesteps
    this entirely by building the per-mode operator from the STRONG-FORM PDE
    directly (each derivative applied ONCE, i*k_j times i*k_l = -k_j*k_l,
    real and correctly signed by construction) -- this module's energy-first
    derivation needed the explicit conjugation instead.
    """
    iq1, iq2, hz = sp.symbols("iq1 iq2 hz", real=True)
    C11, C12, C44 = sp.symbols("C11 C12 C44", real=True)
    u1n, u2n, u3n, u1p, u2p, u3p = sp.symbols("u1n u2n u3n u1p u2p u3p", real=True)
    dof = [u1n, u2n, u3n, u1p, u2p, u3p]

    strain_vec = sp.Matrix(_face_strain_symbolic(iq1, iq2, hz, *dof))  # 6x1
    X_of_iq = strain_vec.jacobian(dof)     # 6x6, AFFINE in (iq1,iq2)
    at0 = {iq1: 0, iq2: 0}
    X0_sym = X_of_iq.subs(at0)
    X1_sym = X_of_iq.diff(iq1)             # constant (X_of_iq is linear in iq1,iq2)
    X2_sym = X_of_iq.diff(iq2)

    d11, d22, d33, d12, d13, d23 = sp.symbols("d11 d22 d33 d12 d13 d23", real=True)
    # Cubic contraction, elastic.py's convention (tensor components, factor
    # 2 on shear -- SPEC §1.4 "sigma12 = 2*C44*eps12" etc.)
    s11 = C11 * d11 + C12 * (d22 + d33)
    s22 = C11 * d22 + C12 * (d11 + d33)
    s33 = C11 * d33 + C12 * (d11 + d22)
    s12, s13, s23 = 2 * C44 * d12, 2 * C44 * d13, 2 * C44 * d23
    energy_half = sp.Rational(1, 2) * (s11 * d11 + s22 * d22 + s33 * d33
                                        + 2 * s12 * d12 + 2 * s13 * d13 + 2 * s23 * d23)
    dvec = [d11, d22, d33, d12, d13, d23]
    Cmat_sym = sp.Matrix([sp.diff(energy_half, dd) for dd in dvec]).jacobian(dvec)  # 6x6

    fn_X0 = sp.lambdify((hz,), X0_sym, "numpy")
    fn_X1 = sp.lambdify((hz,), X1_sym, "numpy")   # hz-independent (constant); kept for a uniform call signature
    fn_X2 = sp.lambdify((hz,), X2_sym, "numpy")
    fn_Cmat = sp.lambdify((C11, C12, C44), Cmat_sym, "numpy")
    return fn_X0, fn_X1, fn_X2, fn_Cmat


_FN_X0, _FN_X1, _FN_X2, _FN_CMAT = _derive_local_element()


def _local_matrices(hz, C11, C12, C44):
    """Numeric (6,6) real matrices at fixed (hz,C11,C12,C44): X0,X1,X2 (the
    strain-vs-dof map X(iq1,iq2)=X0+X1*iq1+X2*iq2) and Cmat (energy_half =
    0.5*d^T@Cmat@d)."""
    X0 = np.asarray(_FN_X0(hz), dtype=float)
    X1 = np.asarray(_FN_X1(hz), dtype=float)
    X2 = np.asarray(_FN_X2(hz), dtype=float)
    Cmat = np.asarray(_FN_CMAT(C11, C12, C44), dtype=float)
    return X0, X1, X2, Cmat


# ==========================================================================
# Batched BLOCK-tridiagonal Thomas algorithm (3x3 blocks). Same factor-once/
# solve-many-RHS pattern as film3d.py's scalar solve_tridiag_batched, now
# with 3x3 matrix inverse/matmul (via einsum) at each elimination step
# instead of scalar divide/multiply. Convention: leading axis n = node
# index (or n-1 for sub/sup), TRAILING two axes = the (3,3) block / last
# axis = the 3-vector -- moveaxis'd to/from this module's public
# component-first (3,...) convention at the setup_film_elastic/
# elastic_solve_film boundary (mirrors film3d.step()'s own moveaxis usage).
# ==========================================================================
def factor_block_tridiag_batched(sub, diag, sup):
    """sub,sup: (n-1,...,3,3). diag: (n,...,3,3)."""
    n = diag.shape[0]
    batch_shape = diag.shape[1:-2]
    cp = np.empty((n - 1,) + batch_shape + (3, 3), dtype=complex)
    piv_inv = np.empty((n,) + batch_shape + (3, 3), dtype=complex)
    piv_inv[0] = np.linalg.inv(diag[0])
    for i in range(1, n):
        cp[i - 1] = np.einsum("...ij,...jk->...ik", piv_inv[i - 1], sup[i - 1])
        piv = diag[i] - np.einsum("...ij,...jk->...ik", sub[i - 1], cp[i - 1])
        piv_inv[i] = np.linalg.inv(piv)
    return dict(sub=sub, cp=cp, piv_inv=piv_inv, n=n)


def solve_block_tridiag_factored(factors, rhs):
    """rhs: (n,...,3[,extra broadcastable trailing axes]). Returns same shape."""
    sub, cp, piv_inv, n = factors["sub"], factors["cp"], factors["piv_inv"], factors["n"]
    dp = np.empty_like(rhs)
    dp[0] = np.einsum("...ij,...j->...i", piv_inv[0], rhs[0])
    for i in range(1, n):
        corr = rhs[i] - np.einsum("...ij,...j->...i", sub[i - 1], dp[i - 1])
        dp[i] = np.einsum("...ij,...j->...i", piv_inv[i], corr)
    x = np.empty_like(dp)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - np.einsum("...ij,...j->...i", cp[i], x[i + 1])
    return x


# ==========================================================================
# Setup: assemble the (uniform-C, uniform-hz => position-independent local
# blocks) global operator bands ONCE per (grid, C11, C12, C44, e0).
# ==========================================================================
def setup_film_elastic(stack_grid, C11, C12, C44, Q11, Q12, Q44, e0, eps_star_static=None,
                       C_substrate=None):
    """Assemble+factorize the stack's per-k FE stiffness once per (grid, C, Q, e0) -> the `el` dict elastic_solve_film consumes.

    C_substrate: optional C11/C12/C44 triplet for a homogeneous substrate.
    Film C, eigenstrain source and ebar remain film-specific; default None
    retains the historical homogeneous stack. Both layers share interface
    displacement and the assembled equilibrium enforces traction balance.
    A stack without substrate layers (hs_dx=0) rejects C_substrate.

    eps_star_static (SPEC V2-03 A2, elastic.py's own WP3 contract): optional
    STATIC (3,3,Nz,Ny,Nx) FULL-STACK eigenstrain field added to eps*(P)
    before the mechanical solve (elastic_solve_film), exactly as
    elastic.elastic_solve adds elastic.setup_elastic's own eps_star_static.
    Its P-derivative is exactly zero, so elastic_force_film and the
    node-averaged-stress chain rule are untouched. None (the default) is
    BYTE-IDENTICAL to every pre-V2-03 call -- the addition is not executed
    at all on that path (gates/core K2 `eps_star_static_none_film`).
    The SUBSTRATE block (z < film_lo) must be exactly zero and is CHECKED
    here rather than silently dropped: this lane's face source is built from
    the FILM block only (see elastic_solve_film's source-assembly note --
    substrate and interface-straddling faces carry no eigenstrain by
    construction, which is what keeps the force's chain rule exact), so a
    nonzero substrate eigenstrain would not drive the solve at all."""
    validate.setup_film_elastic(stack_grid, C11, C12, C44, Q11, Q12, Q44, e0, eps_star_static, C_substrate)
    Nz = stack_grid["Nz"]
    hz = stack_grid["hz"]
    film_lo = stack_grid["film_lo"]
    # Use the same real-field derivative in stiffness, RHS and strain
    # recovery. An odd derivative's own Nyquist mode is zero, not i*k.
    aK, bK = f3._odd_k(stack_grid)
    K1, K2 = aK.imag, bK.imag
    n_elem = Nz - 1

    X0, X1, X2, Cmat = _local_matrices(hz, C11, C12, C44)

    a, b = 1j * K1, 1j * K2  # (Ny,Nxr)
    X_k = X0[None, None] + X1[None, None] * a[..., None, None] + X2[None, None] * b[..., None, None]   # X(+k)
    X_mk = X0[None, None] - X1[None, None] * a[..., None, None] - X2[None, None] * b[..., None, None]  # X(-k)=conj(X(k))
    # Hermitian per-mode stiffness (root-cause fix -- see _derive_local_element
    # docstring and WP7B_REPORT.md): K(k) = hz * X(-k)^T @ Cmat @ X(k), NOT
    # hz * X(k)^T @ Cmat @ X(k) (exactly symmetric but not Hermitian for k!=0).
    K_el = hz * np.einsum("...ji,jk,...kl->...il", X_mk, Cmat, X_k)  # (Ny,Nxr,6,6)

    # P-INDEPENDENT, hoisted here so elastic_solve_film does not rebuild them
    # once per step (SPEC §39.1 A3' = REVIEW_wp27_efficiency_measured.md #5:
    # 5.8% of a solve at 24^2, 6.3% at 128^2). Same expressions, same inputs,
    # so byte-identical: X_mk above IS the X(-k) the solve used to recompute.
    M_el = hz * np.einsum("...ki,kj->...ij", X_mk, Cmat)   # (Ny,Nxr,6,6), RHS assembly
    C_tensor = elastic.build_C_tensor(C11, C12, C44)       # (3,3,3,3), face-stress contraction
    Knn = K_el[..., 0:3, 0:3]
    Knp = K_el[..., 0:3, 3:6]
    Kpn = K_el[..., 3:6, 0:3]
    Kpp = K_el[..., 3:6, 3:6]

    # Reduced system (node 0 = substrate bottom, u=0, DROPPED -- SPEC "u=0
    # at the substrate bottom"; module docstring's index derivation):
    #   diag[m] = Kpp+Knn for m=0..n_elem-2, Kpp alone for m=n_elem-1 (top,
    #     natural free-surface BC: no element beyond => no Knn contribution)
    #   sub[m]=Kpn, sup[m]=Knp for m=0..n_elem-2 (uniform, position-indep)
    diag = np.broadcast_to((Kpp + Knn)[None], (n_elem,) + Kpp.shape).copy()
    diag[-1] = Kpp
    sub = np.broadcast_to(Kpn[None], (n_elem - 1,) + Kpn.shape)
    sup = np.broadcast_to(Knp[None], (n_elem - 1,) + Knp.shape)

    # Two laterally homogeneous layers need only a direct banded solve.
    # Face j connects nodes j,j+1; faces below film_lo belong to substrate.
    substrate_tensor = None
    if C_substrate is not None:
        if film_lo < 1:
            raise ValueError("C_substrate needs substrate layers: this stack has none "
                             "(make_stack_grid hs_dx=0)")
        cs = _triplet(C_substrate, ("C11", "C12", "C44"), "C_substrate")
        if not all(np.isfinite(cs)) or not (cs[0]-cs[1] > 0 and cs[0]+2*cs[1] > 0 and cs[2] > 0):
            raise ValueError("C_substrate must be a positive-definite cubic stiffness")
        _, _, _, Cmat_s = _local_matrices(hz, *cs)
        Ks = hz * np.einsum("...ji,jk,...kl->...il", X_mk, Cmat_s, X_k)
        diag[:film_lo-1] = Ks[..., 3:6, 3:6] + Ks[..., :3, :3]
        diag[film_lo-1] = Ks[..., 3:6, 3:6] + Knn
        sub, sup = sub.copy(), sup.copy()
        sub[:film_lo-1] = Ks[..., 3:6, :3]
        sup[:film_lo-1] = Ks[..., :3, 3:6]
        substrate_tensor = elastic.build_C_tensor(*cs)

    ebar = homogeneous_strain_film_e0(e0, C11, C12)

    if eps_star_static is not None:
        want = (3, 3, Nz, stack_grid["Ny"], stack_grid["Nx"])
        if np.shape(eps_star_static) != want:
            raise ValueError(f"setup_film_elastic: eps_star_static must have shape {want}, "
                              f"got {np.shape(eps_star_static)}")
        if np.any(eps_star_static[:, :, :film_lo] != 0.0):
            raise ValueError("setup_film_elastic: eps_star_static must be exactly zero in the "
                              "substrate block [:, :, :film_lo] -- this lane sources the face "
                              "eigenstrain from the FILM block only (see the docstring)")

    # RETURNED-KEY CONTRACT (the dict elastic_solve_film and the G11/G12
    # gates consume; bulk3d.py:311's elastic_params CONTRACT block is the
    # pattern for this):
    #   Nz, Ny, Nx, Nz_film-derived n_elem, hz, film_lo -- stack geometry.
    #   K1, K2       -- (Ny,Nxr) in-plane real wavenumbers (rfft2 grid).
    #   diag/sub/sup -- the reduced block-tridiagonal bands; _factors is
    #                   their once-per-setup factorization.
    #   X0, X1, X2, Cmat -- the local element matrices, kept because
    #                   gates/g11b_machinery_exact.py rebuilds M_el from them
    #                   as its independent cross-check.
    #   M_el         -- (Ny,Nxr,6,6) = hz * X(-k)^T @ Cmat, the RHS assembly
    #                   operator. NEW in WP28 P5 (SPEC §39.1 A3'): it is
    #                   P-INDEPENDENT, so it is built once here instead of
    #                   once per elastic_solve_film call.
    #   C_tensor     -- (3,3,3,3) cubic stiffness for the face-stress
    #                   contraction. NEW in WP28 P5, same reason (it was
    #                   elastic.build_C_tensor(C11,C12,C44) inside the solve).
    #   ebar         -- (3,3) homogeneous misfit strain from e0.
    #   C11/C12/C44/Q11/Q12/Q44/e0 -- the raw spec values, inspectable.
    #   eps_star_static -- None (default) or the static eigenstrain field
    #                   elastic_solve_film adds to eps*(P) (V2-03 A2).
    result = dict(Nz=Nz, hz=hz, film_lo=film_lo, n_elem=n_elem, K1=K1, K2=K2,
                diag=diag, sub=sub, sup=sup, X0=X0, X1=X1, X2=X2, Cmat=Cmat, ebar=ebar,
                M_el=M_el, C_tensor=C_tensor, eps_star_static=eps_star_static,
                C11=C11, C12=C12, C44=C44, Q11=Q11, Q12=Q12, Q44=Q44, e0=e0,
                Ny=stack_grid["Ny"], Nx=stack_grid["Nx"],
                _factors=factor_block_tridiag_batched(sub, diag, sup))
    if substrate_tensor is not None:
        result.update(C_substrate=tuple(cs), C_tensor_substrate=substrate_tensor)
    return result


# ==========================================================================
# Solve: given P (film-only, (3,Nz_film,Ny,Nx)), build the eigenstrain
# source over the FULL stack (zero in substrate), fold in ebar (film-only,
# via the SAME source field -- module docstring), FFT in-plane, assemble
# the per-element RHS (M0/Ma/Mb, matching the K assembly above), solve the
# reduced block-tridiagonal system, IFFT back, then read off face
# strain/stress everywhere and node-averaged stress at the P-nodes (for the
# elastic driving force -- SPEC's "-sigma:d(eps*)/dP as in elastic.py").
# ==========================================================================
def elastic_solve_film(P, el):
    """Six stages, in order: (1) stack eigenstrain eps*(P), zero in the
    substrate, plus el["eps_star_static"] if the setup was given one (V2-03
    A2; None => that addition is not executed, byte-identical);
    (2) per-face source (eps* - ebar); (3) in-plane FFT +
    per-element RHS assembly; (4) reduced block-tridiagonal solve with
    el["_factors"]; (5) IFFT back, face strain and stress; (6) node-averaged
    stress at the P-nodes, for the elastic driving force.
    el: setup_film_elastic's output. P: (3,Nz_film,Ny,Nx),
    film only. Returns dict(u, e_face, sigma_face, sigma_node,
    eps_star_stack, src_face); elastic_force_film/elastic_energy_density_film
    consume sigma_node/e_face from it. Gates: G11a-f, G12a/G12b.

    SPEC V2-03 C2: dispatches to elastic_solve_film_spm when `el` was built
    by setup_film_elastic_spm (el["spm"]=True, inhomogeneous C(r) in the film
    block) -- exactly elastic.elastic_solve's own WP10 dispatch. Every OTHER
    el (setup_film_elastic, with or without eps_star_static) has no "spm"
    key at all, so el.get("spm") is None and this function is bit-identical
    to its pre-V2-03-C body (gates/core K2 `film_elastic_spm_off_byte`)."""
    if el.get("spm"):
        return elastic_solve_film_spm(P, el)
    Nz, film_lo = el["Nz"], el["film_lo"]
    Ny, Nx = el["Ny"], el["Nx"]
    hz = el["hz"]
    n_elem = el["n_elem"]

    eps_star_stack = np.zeros((3, 3, Nz, Ny, Nx))
    eps_star_stack[:, :, film_lo:] = elastic.eigenstrain(P, el["Q11"], el["Q12"], el["Q44"])
    static = el.get("eps_star_static")
    if static is not None:
        eps_star_stack = eps_star_stack + static   # V2-03 A2, elastic.py's WP3 contract
        # (substrate block is checked exactly zero at setup, so this changes
        #  the FILM eigenstrain only -- the face source below is unchanged
        #  in structure and its P-derivative is exactly what it was)

    # Face source (eps* - ebar, ebar film-only): built face-by-face, NOT by
    # node-averaging the full stack, because the face straddling the
    # interface (nodes film_lo-1 [substrate] <-> film_lo [first film node])
    # has its MIDPOINT physically in the substrate -- node-averaging it
    # against the film node's nonzero value would leak a spurious half-
    # strength source into a face that must be exactly zero (caught by a
    # pre-gate self-test against the G11a closed form: u(interface) came
    # out nonzero, which the closed form forbids). Substrate faces
    # (0..film_lo-1, INCLUDING that interface-straddling one) are all
    # zero; only the Nz_film-1 faces strictly inside the film get a
    # source, from node-averaging P's own eigenstrain WITHIN the film.
    eps_star_film = eps_star_stack[:, :, film_lo:]  # (3,3,Nz_film,Ny,Nx)
    eps_star_face_film = 0.5 * (eps_star_film[:, :, :-1] + eps_star_film[:, :, 1:])  # (3,3,Nz_film-1,Ny,Nx)
    src_face = np.zeros((3, 3, n_elem, Ny, Nx))
    src_face[:, :, film_lo:] = eps_star_face_film - el["ebar"][:, :, None, None, None]
    order = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    K1, K2 = el["K1"], el["K2"]
    # Faces 0..film_lo-1 are an EXACT zero block (the assignment above never
    # writes them), and rfft2 of exact zeros is exact zeros -- so transform
    # the film faces only and leave the substrate block at its initialized
    # zero (SPEC §39.1 A3'; byte-identical, LC C3). Shape unchanged.
    es_hat = np.zeros((6, n_elem, Ny, K1.shape[-1]), dtype=complex)  # (6,n_elem,Ny,Nxr)
    es_hat[:, film_lo:] = np.stack([f3._rfft2(src_face[i, j, film_lo:]) for i, j in order])

    # Hermitian pairing (see setup_film_elastic/_derive_local_element): M(k) =
    # hz*X(-k)^T @ Cmat, acting on es_hat(k) (the actual, non-conjugated
    # physical Fourier coefficient of the source) -- consistent with K_el's fix.
    # HOISTED: P-independent, built once by setup_film_elastic (SPEC §39.1 A3').
    M_el = el["M_el"]  # (Ny,Nxr,6,6)
    # f_local = M_el @ es (per element): (n_elem,Ny,Nxr,6)
    f_local = np.einsum("yxij,jeyx->eyxi", M_el, es_hat)
    f_n_all, f_p_all = f_local[..., 0:3], f_local[..., 3:6]  # (n_elem,Ny,Nxr,3)

    rhs = f_p_all.copy()
    rhs[:-1] += f_n_all[1:]  # RHS_reduced[m] = f_p(elem m) + f_n(elem m+1) [m<n_elem-1]; top: f_p alone

    u_reduced = solve_block_tridiag_factored(el["_factors"], rhs)  # (n_elem,Ny,Nxr,3)

    Nxr = K1.shape[-1]
    u_hat = np.zeros((Nz, Ny, Nxr, 3), dtype=complex)
    u_hat[1:] = u_reduced  # node 0 (=0) already zero
    u_hat = np.moveaxis(u_hat, -1, 0)  # (3,Nz,Ny,Nxr) -- this module's component-first convention
    u = f3._irfft2(u_hat, (Ny, Nx))  # (3,Nz,Ny,Nx) real

    # ---- face strain/stress everywhere (real space: ik-then-ifft for the
    # in-plane pieces per SPEC §2, real FD directly for the z pieces since
    # u's z-axis was never Fourier-transformed) ----
    u_hat_full = f3._rfft2(u)  # (3,Nz,Ny,Nxr)
    e11h, e22h, e33h, e12h, e13h, e23h = _face_strain(u_hat_full[:, :-1], u_hat_full[:, 1:], K1, K2, hz)
    e_face = np.zeros((3, 3, n_elem, Ny, Nx))
    comps = dict()
    for (i, j), hcomp in zip(order, (e11h, e22h, e33h, e12h, e13h, e23h)):
        comps[(i, j)] = f3._irfft2(hcomp, (Ny, Nx))
    for (i, j), val in comps.items():
        e_face[i, j] = val
        e_face[j, i] = val
    e_face_elastic = e_face - src_face  # ebar+eta-eps* everywhere (src already has ebar folded in)

    C = el["C_tensor"]  # HOISTED: P-independent (SPEC §39.1 A3')
    sigma_face = elastic.contract_C(C, e_face_elastic)  # (3,3,n_elem,Ny,Nx)
    if "C_tensor_substrate" in el:
        sigma_face[:, :, :film_lo] = elastic.contract_C(
            el["C_tensor_substrate"], e_face_elastic[:, :, :film_lo])

    # ---- node-averaged stress AT the P-nodes (film nodes only) for the
    # driving force -- this IS the exact d(discrete energy)/dP chain rule
    # (not an ad hoc interpolation): energy = sum_faces hz*0.5*sigma_face:
    # e_face_elastic, e_face_elastic = e_face - src_face, and src_face =
    # 0.5*(eps*(node_below)+eps*(node_above)) - ebar for a WITHIN-FILM face
    # (zero, node-independent, for substrate/interface-straddling faces).
    # So d(energy)/dP(node j) = -sum_{faces touching j whose src actually
    # DEPENDS on node j's P} 0.5*sigma_face : d(eps*(node j))/dP -- i.e.
    # EVERY P-dependent touching face contributes weight 0.5, regardless of
    # how many such faces touch that node (root-cause fix, WP7B_REPORT.md:
    # a previous version used weight 1.0 -- not 0.5 -- at the top node
    # (single touching face; "1/count-of-touching-faces" reasoning, wrong)
    # and included the interface-straddling face's sigma at the FIRST film
    # node with weight 0.5 even though that face's source is a hard,
    # P-INDEPENDENT zero (substrate faces never carry eigenstrain) and so
    # must be EXCLUDED, not averaged in -- both confirmed by a direct,
    # independent re-derivation of this chain rule matching FD to 6+ digits
    # once the interface/top handling was corrected; verified GENERAL, not
    # coincidental to those two nodes).
    Nz_film = Nz - film_lo
    sigma_node = np.zeros((3, 3, Nz_film, Ny, Nx))
    # film P-node j (local) = global node film_lo+j; touching faces
    # (global face index = global node index for the "below" face) are
    # (film_lo+j-1) and (film_lo+j); global face k connects nodes k,k+1.
    lo_face = film_lo - 1  # face BELOW the first film node (last substrate/first film interface)
    for j in range(Nz_film):
        below = lo_face + j        # exists for all j (>=0 since film_lo>=1)
        above = lo_face + j + 1    # exists unless j == Nz_film-1 (top, no face above)
        if below >= film_lo:       # False only for j==0 (interface-straddling, P-independent source)
            sigma_node[:, :, j] += 0.5 * sigma_face[:, :, below]
        if above <= n_elem - 1:    # False only for j==Nz_film-1 (top, no face beyond)
            sigma_node[:, :, j] += 0.5 * sigma_face[:, :, above]

    return dict(u=u, e_face=e_face_elastic, sigma_face=sigma_face, sigma_node=sigma_node,
                eps_star_stack=eps_star_stack, src_face=src_face)


def elastic_energy_density_film(sol, el):
    """Film-only elastic energy density (SPEC film3d.py convention: total /
    Volume3D=Lx*Ly*hf). Faces strictly within the film range (global index
    film_lo..n_elem-1, i.e. midpoint 0<z<hf -- NOT the interface-straddling
    face film_lo-1, whose midpoint is at z=-hz/2, physically IN the
    substrate, per the elastic_solve_film source-assembly note) each
    contribute hz*1/2*sigma:e -- an EXACT integral of the piecewise-face
    representation, no trapezoidal-weight subtlety (face quantities are
    already "cell-centered")."""
    film_lo, n_elem = el["film_lo"], el["n_elem"]
    sigma_f = sol["sigma_face"][:, :, film_lo:]
    e_f = sol["e_face"][:, :, film_lo:]
    dens = 0.5 * np.einsum("ije...,ije...->e...", sigma_f, e_f)  # (Nz_film-1,Ny,Nx)
    return dens  # per-face-cell energy density; caller integrates with hz*sum/Volume3D


def total_elastic_energy_film(sol, el, grid):
    """hz*sum(dens)/Volume3D, area-averaged first (matching film3d.py's own
    total_energy_3d pattern: mean over Ny,Nx, sum*hz over the film faces,
    normalize by Volume3D=Lx*Ly*hf)."""
    dens = elastic_energy_density_film(sol, el)  # (Nz_film-1,Ny,Nx)
    area = grid["Lx"] * grid["Ly"]
    total = area * el["hz"] * float(np.sum(dens.mean(axis=(-2, -1))))
    return total / grid["Volume3D"]


def elastic_force_film(P, el):
    """delta F_el/delta P (film-only, matches P's own shape) -- re-solves
    equilibrium at the given P (elastic.py's "1 elastic solve per step"
    convention). Returns (force, sol)."""
    sol = elastic_solve_film(P, el)
    force = elastic.elastic_force_explicit(P, sol["sigma_node"], el["Q11"], el["Q12"], el["Q44"])
    return force, sol


# ==========================================================================
# SPEC V2-03 C2: the FILM elastic SPM -- inhomogeneous cubic C(r) inside the
# film block, over the UNCHANGED per-mode block-tridiagonal C0 solve.
#
# WHAT IS SPLIT AND WHY. The whole film lane rests on the per-(k1,k2) block-
# tridiagonal structure of K(k) = hz*X(-k)^T@Cmat@X(k): that structure exists
# because Cmat is the SAME matrix at every in-plane point, so each mode
# decouples. A C that varies IN-PLANE destroys it. The SPM restores it the
# way solver.py's WP5 electrostatic SPM and elastic.py's WP10 elastic SPM do
# (Wang et al., Acta Mater 61 (2013) 7591, App. A): split
#
#     C(r) = C0 + dC(r)
#
# and move the dC piece to the RIGHT-HAND SIDE, where it is an ordinary
# source for the SAME C0 operator. Writing the discrete energy (module
# docstring; the real-space face form elastic_energy_density_film and
# gates/core K1 `fd_film_elastic` both use) as
#
#     E[u] = hx*hy*hz * sum_{faces,x,y} 1/2 * d^T Cmat(r) d,
#     d    = e_face(u) - src_face          (src = eps* - ebar, film-only)
#
# and calling A the linear map u -> e_face(u) (per mode: X(k)) and A* its
# adjoint (per mode: hz*X(-k)^T -- the SAME Hermitian pairing the K/M
# assembly already uses), stationarity dE/du = 0 is A*[C(r) d] = 0, i.e.
#
#     K0 u = M_el es   -   A*[ dC(r) : (A u - src) ].
#            \_________/       \____________________/
#          the EXISTING rhs        the explicit correction
#
# so the iteration is: solve the existing block-tridiagonal system, read the
# face strain back in real space, contract it with dC(r) pointwise, transform
# that correction stress back and subtract its A* assembly from the rhs,
# solve again. Zero order (dC == 0) is EXACTLY the existing homogeneous
# solve -- and BIT-FOR-BIT so, since the correction is then an exact array of
# zeros and `rhs0 - 0.0` is the identity on floats (K2 `film_elastic_spm_const`
# measures u at exactly 0.0 difference).
#
# THE FORCE STAYS THE EXACT GRADIENT OF THE DISCRETE ENERGY. The correction
# is assembled through the SAME element idiom as the eigenstrain source: dC
# lives on FACES as the average of its two nodal values (exactly how
# elastic_solve_film builds eps_star_face_film from nodal eps*), and the A*
# assembly is the SAME "f_p(elem m) + f_n(elem m+1)" reduction. So the
# converged u is the stationary point of the SAME functional whose face-wise
# density elastic_energy_density_film evaluates with the local C(r), and the
# envelope-theorem chain rule elastic_solve_film's own comment derives is
# untouched: sigma_node is still 0.5*sigma_face summed over the P-dependent
# touching faces, with sigma_face = C(r):e_face_elastic instead of C0:e.
# elastic_force_film therefore needs NO change at all (it consumes only
# sigma_node), and neither does elastic_energy_density_film /
# total_elastic_energy_film / film3d.total_energy_3d's elastic branch.
# Measured, not asserted: K1 `fd_film_elastic_spm`.
#
# WHERE THE WP10 "<C> vs C0" LESSON LANDS HERE -- AND WHERE IT DOES NOT
# (SPEC S16's elastic_solve_spm carries an OUTER ebar loop; this lane has
# none, and the reason is the substrate clamp):
#   * elastic.py's 2D periodic cell is stress-free in EVERY direction, so
#     ebar is a genuine unknown fixed by <sigma> = 0, and its update needs
#     the TRUE <c(r)>^-1 -- using the (possibly overridden) reference C0's
#     compliance there is the WP10 bug (setup_elastic_spm's docstring).
#   * The film has a substrate. Li 2002's misfit clamp FIXES ebar_11 =
#     ebar_22 = e0: no admissible in-plane-periodic u can change the k=0
#     in-plane strain at all (e11 = i*k1*u1 vanishes at k=0), so there is
#     nothing to iterate. The only k=0 unknown left is the z response, and
#     that is NOT a macroscopic constant either -- it is u3(z) itself,
#     carried node by node by the SAME block-tridiagonal solve, which
#     enforces the discrete traction-free condition <sigma_33>_xy = 0 on
#     EVERY face (the k=0 rows telescope from the free top down to the
#     Dirichlet bottom). That is strictly stronger than the periodic model's
#     single <sigma_33> = 0 and it is already exact at every iterate.
#     => n_outer is 0 by construction; outer_tol/outer_max_iter are accepted
#     for signature parity with elastic.setup_elastic_spm (and C1) and are
#     recorded in the el dict, never used. K4 `film_elastic_spm_residual`
#     reports the measured <sigma_33> residual instead of trusting this.
#   * ebar_33 = -2*(C12/C11)*e0 still needs A pair of moduli. The TRUE film
#     volume means are used (never C0_override) -- the WP10 lesson's letter.
#     Its value is a pure GAUGE on the discrete problem: shifting ebar_33 by
#     delta and u3 by delta*(z - z_interface) inside the film leaves d, and
#     therefore every strain, stress, energy and force, invariant (only u3
#     itself moves). Derived, then measured: K2 `film_elastic_spm_const`.
#
# Nyquist consistency: setup uses _odd_k for BOTH stiffness assembly and
# strain/source recovery. The derivative on its own Nyquist axis is zero,
# matching real-field Fourier differentiation; unmasked assembly formerly
# retained a term that irfft discarded during recovery. Broadband even/odd
# grid FD tests now check the same full-stack relaxed functional.
#
# THE SUBSTRATE IS ALWAYS HOMOGENEOUS AT C0 IN THIS WORK ORDER (SPEC V2-03
# C2, verbatim: "dC(r) != 0 only in the film block; the substrate keeps the
# homogeneous reference"). Consequence to state plainly rather than bury:
# C0_override is NOT the pure numerical knob it is in elastic.py. There, C0
# only picks the Green's operator and the converged answer is C0-independent
# (gate G14b). Here C0 is ALSO the substrate's material stiffness, so
# overriding it changes the physical problem. Use it to trade convergence
# rate against a substrate you meant to specify anyway; do not use it as a
# reference-invariance knob. (A substrate with its own C != C0 would need dC
# on the substrate faces too -- one line here, but outside this order.)
# ==========================================================================
_VOIGT_ORDER = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
_VOIGT_W = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)   # energy-conjugate weights, see _stress_vec6


def _triplet(spec, names, what):
    """(a,b,c) from a dict keyed by `names` or from any 3-sequence."""
    if isinstance(spec, dict):
        missing = [n for n in names if n not in spec]
        if missing:
            raise ValueError(f"{what}: dict is missing {missing} (needs {list(names)})")
        return tuple(spec[n] for n in names)
    v = tuple(spec)
    if len(v) != 3:
        raise ValueError(f"{what}: expected a dict keyed by {list(names)} or a 3-sequence, got {spec!r}")
    return v


def _film_mean(field):
    """Volume mean of one C component over the film nodes, with the EXACTNESS
    GUARD film3d._kappa_faces / film3d_spm.kappa0_inplane_mean use for the
    same reason: an all-equal field returns that value ITSELF, not a
    recomputed mean (a mean over n equal floats is not bit-for-bit that
    float in general). Without it dC would be 1e-16-ish instead of exactly
    zero and the degenerate case would be "small" rather than exact."""
    a = np.asarray(field, dtype=float)
    if a.ndim == 0:
        return float(a)
    lo, hi = float(a.min()), float(a.max())
    return lo if lo == hi else float(a.mean())


def _stress_vec6(sigma):
    """The 6-vector ENERGY-CONJUGATE to the face-strain vector
    (e11,e22,e33,e12,e13,e23) that _face_strain/_derive_local_element use:
    (s11, s22, s33, 2*s12, 2*s13, 2*s23).

    The factor 2 is not cosmetic and is exactly SPEC S5's usual suspect.
    Cmat is d^2(energy_half)/dd^2 with energy_half = 1/2*(s11 d11 + s22 d22 +
    s33 d33 + 2 s12 d12 + 2 s13 d13 + 2 s23 d23) and s12 = 2*C44*d12, so the
    shear entry is Cmat[3,3] = 4*C44 and (Cmat@d)_3 = 4*C44*d12 = 2*s12.
    M_el = hz*X(-k)^T@Cmat therefore acts on THIS vector, which is why the
    correction source must be built the same way. Checked numerically
    against M_el itself (K2 `film_elastic_spm_const`'s adjoint leg), not
    trusted to this derivation."""
    return [w * sigma[i, j] for w, (i, j) in zip(_VOIGT_W, _VOIGT_ORDER)]


def _face_strain_readback(u_reduced, el):
    """(u, e_face) from the reduced per-mode solution -- stage (5) of
    elastic_solve_film, copied expression for expression (including the
    irfft2 -> rfft2 round trip), so the SPM's iterate sees the SAME face
    strain the final energy/stress readback will see."""
    Nz, Ny, Nx, hz = el["Nz"], el["Ny"], el["Nx"], el["hz"]
    K1, K2 = el["K1"], el["K2"]
    u_hat = np.zeros((Nz, Ny, K1.shape[-1], 3), dtype=complex)
    u_hat[1:] = u_reduced                  # node 0 (=0) already zero
    u_hat = np.moveaxis(u_hat, -1, 0)      # (3,Nz,Ny,Nxr)
    u = f3._irfft2(u_hat, (Ny, Nx))
    u_hat_full = f3._rfft2(u)
    hcomps = _face_strain(u_hat_full[:, :-1], u_hat_full[:, 1:], K1, K2, hz)
    e_face = np.zeros((3, 3, el["n_elem"], Ny, Nx))
    for (i, j), hcomp in zip(_VOIGT_ORDER, hcomps):
        val = f3._irfft2(hcomp, (Ny, Nx))
        e_face[i, j] = val
        e_face[j, i] = val
    return u, e_face


def setup_film_elastic_spm(stack_grid, C_film, Q, e0, eps_star_static=None,
                            C0_override=None, spm_tol=1e-10, spm_max_iter=100,
                            outer_tol=1e-10, outer_max_iter=50):
    """SPEC V2-03 C2: the `el` dict elastic_solve_film dispatches to for an
    inhomogeneous cubic C(r) in the FILM block.

    C_film: dict(C11=..., C12=..., C44=...) (or a 3-sequence in that order),
      each entry a scalar or a (Nz_film,Ny,Nx) field ON THE FILM NODES -- the
      same nodes P lives on. Q: dict(Q11=,Q12=,Q44=) or a 3-sequence.
    C0_override=(C11_0,C12_0,C44_0): the reference the block-tridiagonal
      operator is built from AND, in this work order, the substrate's own
      stiffness (see the section comment above -- this is NOT elastic.py's
      pure numerical knob). Default: the film's TRUE volume mean per
      component, which makes the substrate exactly the film's mean material
      and reproduces setup_film_elastic exactly for a uniform C_film.
    spm_tol / spm_max_iter: the fixed point's stopping rule (SPEC's 1e-10 /
      100, elastic.setup_elastic_spm's own pair). outer_tol / outer_max_iter
      are accepted for signature parity and stored; the film geometry has no
      outer loop (section comment), so they are never used.

    The C0 bands, factorization, M_el, C_tensor and the eps_star_static
    contract are setup_film_elastic's, called here with C0 -- not a
    re-derivation -- so the inner operator IS the gated homogeneous one.
    Only `ebar` is replaced (true film means, never C0_override) and the SPM
    slots are added.

    NOT frozen after setup: elastic_solve_film_spm(..., warm_start=True)
    writes "_warm_u" back into THIS dict (elastic_solve_spm's contract).
    Copy the dict if a caller needs a cold-start el."""
    Nz_film, Ny, Nx = stack_grid["Nz_film"], stack_grid["Ny"], stack_grid["Nx"]
    C11f, C12f, C44f = _triplet(C_film, ("C11", "C12", "C44"), "setup_film_elastic_spm: C_film")
    Q11, Q12, Q44 = _triplet(Q, ("Q11", "Q12", "Q44"), "setup_film_elastic_spm: Q")

    want = (Nz_film, Ny, Nx)
    nodes = []
    for name, f in (("C11", C11f), ("C12", C12f), ("C44", C44f)):
        a = np.asarray(f, dtype=float)
        if a.ndim == 0:
            a = np.full(want, float(a))
        elif a.shape != want:
            raise ValueError(f"setup_film_elastic_spm: C_film[{name!r}] must be a scalar or have "
                             f"shape {want} (FILM nodes), got {a.shape}")
        nodes.append(a)
    means = tuple(_film_mean(a) for a in nodes)
    C0 = tuple(C0_override) if C0_override is not None else means

    el = dict(setup_film_elastic(stack_grid, C0[0], C0[1], C0[2], Q11, Q12, Q44, e0,
                                 eps_star_static=eps_star_static))
    film_lo, n_elem, hz = el["film_lo"], el["n_elem"], el["hz"]

    # Face fields: the SAME element idiom as elastic_solve_film's
    # eps_star_face_film (average of the two nodal values). Substrate faces
    # -- INCLUDING the interface-straddling one, whose midpoint is
    # physically in the substrate (elastic_solve_film's source note) -- stay
    # at C0 exactly, which is what "the substrate is homogeneous at C0" means
    # discretely.
    faces = [0.5 * (a[:-1] + a[1:]) for a in nodes]
    dC_face = tuple(f - c for f, c in zip(faces, C0))          # exactly 0.0 for a uniform C_film
    C_face = []
    for f, c in zip(faces, C0):
        full = np.full((n_elem, Ny, Nx), c)
        full[film_lo:] = f
        C_face.append(full)

    # A* = hz*X(-k)^T, the BARE adjoint (M_el = T_el @ Cmat, checked in K2):
    # the correction enters as an already-contracted stress, not through Cmat.
    a, b = 1j * el["K1"], 1j * el["K2"]
    X_mk = (el["X0"][None, None] - el["X1"][None, None] * a[..., None, None]
            - el["X2"][None, None] * b[..., None, None])
    T_el = hz * np.moveaxis(X_mk, -1, -2)                      # (Ny,Nxr,6,6)

    el.update(spm=True, C11_film=nodes[0], C12_film=nodes[1], C44_film=nodes[2],
              C_face=tuple(C_face), dC_face=dC_face, C_mean=means, C0_spm=C0,
              T_el=T_el, ebar=homogeneous_strain_film_e0(e0, means[0], means[1]),
              spm_tol=spm_tol, spm_max_iter=spm_max_iter,
              outer_tol=outer_tol, outer_max_iter=outer_max_iter)
    return el


def elastic_solve_film_spm(P, el, warm_start=True):
    """SPEC V2-03 C2's fixed point (section comment above for the derivation).
    Same six stages as elastic_solve_film, with stage (4) iterated:

      (1)-(3) eigenstrain, face source, rhs0 = M_el @ es  -- IDENTICAL lines;
      (4) u <- K0^-1 [rhs0 - A*(dC(r) : (e_face(u) - src))], repeated until
          max|du|/max|u| <= spm_tol (cap spm_max_iter). Iterate 0 is the
          plain homogeneous solve, i.e. dC == 0 stops at n_inner = 1 with
          resid exactly 0.0 on a cold start;
      (5)-(6) readback, sigma_face with the LOCAL C(r), node-averaged
          sigma_node -- IDENTICAL lines apart from which C contracts.

    Returns elastic_solve_film's keys plus (n_inner, n_outer, resid_inner,
    resid_outer); n_outer/resid_outer are 0 by construction -- the misfit
    clamp leaves no macroscopic unknown to iterate (section comment).

    warm_start=True (default) reuses el["_warm_u"] as the starting guess and
    writes the converged one back: a pure starting-GUESS optimization that
    cannot move the fixed point (same tol either way), for relax()'s stepping
    loop where P changes little between steps. A freshly built el has no
    cache, so the first call on any el is a cold start."""
    Nz, film_lo = el["Nz"], el["film_lo"]
    Ny, Nx = el["Ny"], el["Nx"]
    n_elem = el["n_elem"]

    # ---- stages (1)-(3): identical to elastic_solve_film ------------------
    eps_star_stack = np.zeros((3, 3, Nz, Ny, Nx))
    eps_star_stack[:, :, film_lo:] = elastic.eigenstrain(P, el["Q11"], el["Q12"], el["Q44"])
    static = el.get("eps_star_static")
    if static is not None:
        eps_star_stack = eps_star_stack + static

    eps_star_film = eps_star_stack[:, :, film_lo:]
    eps_star_face_film = 0.5 * (eps_star_film[:, :, :-1] + eps_star_film[:, :, 1:])
    src_face = np.zeros((3, 3, n_elem, Ny, Nx))
    src_face[:, :, film_lo:] = eps_star_face_film - el["ebar"][:, :, None, None, None]
    K1, K2 = el["K1"], el["K2"]
    Nxr = K1.shape[-1]
    es_hat = np.zeros((6, n_elem, Ny, Nxr), dtype=complex)
    es_hat[:, film_lo:] = np.stack([f3._rfft2(src_face[i, j, film_lo:]) for i, j in _VOIGT_ORDER])

    f_local = np.einsum("yxij,jeyx->eyxi", el["M_el"], es_hat)
    rhs0 = f_local[..., 3:6].copy()
    rhs0[:-1] += f_local[1:, ..., 0:3]

    # ---- stage (4): the fixed point --------------------------------------
    factors, T_el = el["_factors"], el["T_el"]
    dC11, dC12, dC44 = el["dC_face"]
    u_reduced = solve_block_tridiag_factored(factors, rhs0)   # zero order == the homogeneous solve
    warm = el.get("_warm_u") if warm_start else None
    if warm is not None and warm.shape == u_reduced.shape:
        u_reduced = warm
    n_inner, resid = 0, 0.0
    for n_inner in range(1, el["spm_max_iter"] + 1):
        _u, e_face = _face_strain_readback(u_reduced, el)
        d_face = e_face - src_face
        dsig = elastic.contract_C_cubic(dC11, dC12, dC44, d_face[:, :, film_lo:])
        tau_hat = np.zeros((6, n_elem, Ny, Nxr), dtype=complex)
        tau_hat[:, film_lo:] = np.stack([f3._rfft2(v) for v in _stress_vec6(dsig)])
        g_local = np.einsum("yxij,jeyx->eyxi", T_el, tau_hat)
        rhs = rhs0 - g_local[..., 3:6]
        rhs[:-1] -= g_local[1:, ..., 0:3]
        u_new = solve_block_tridiag_factored(factors, rhs)
        denom = float(np.max(np.abs(u_new)))
        num = float(np.max(np.abs(u_new - u_reduced)))
        resid = num / denom if denom > 0 else 0.0
        u_reduced = u_new
        if resid <= el["spm_tol"]:
            break
    if resid > el["spm_tol"]:
        print(f"[film_elastic.elastic_solve_film_spm] WARNING: SPM did NOT converge within "
              f"spm_max_iter={el['spm_max_iter']} (final resid={resid:.3e} > tol={el['spm_tol']:.0e}).")

    # ---- stages (5)-(6): identical, with the LOCAL C(r) -------------------
    u, e_face = _face_strain_readback(u_reduced, el)
    e_face_elastic = e_face - src_face
    sigma_face = elastic.contract_C_cubic(el["C_face"][0], el["C_face"][1], el["C_face"][2],
                                          e_face_elastic)

    Nz_film = Nz - film_lo
    sigma_node = np.zeros((3, 3, Nz_film, Ny, Nx))
    lo_face = film_lo - 1
    for j in range(Nz_film):
        below = lo_face + j
        above = lo_face + j + 1
        if below >= film_lo:
            sigma_node[:, :, j] += 0.5 * sigma_face[:, :, below]
        if above <= n_elem - 1:
            sigma_node[:, :, j] += 0.5 * sigma_face[:, :, above]

    if warm_start:
        el["_warm_u"] = u_reduced
    return dict(u=u, e_face=e_face_elastic, sigma_face=sigma_face, sigma_node=sigma_node,
                eps_star_stack=eps_star_stack, src_face=src_face,
                n_inner=n_inner, n_outer=0, resid_inner=resid, resid_outer=0.0)


if __name__ == "__main__":
    g = paper_stack_grid()
    print(f"paper stack grid: Nx=Ny={g['Nx']}, Nz={g['Nz']} (Nz_s={g['Nz_s']}+Nz_film={g['Nz_film']}), "
          f"hs={g['hs']}nm hf={g['hf']}nm hz={g['hz']}nm film_lo={g['film_lo']}")
