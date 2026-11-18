"""film3d_langevin.py -- SPEC V2-03 B1: the exact per-mode OU (Langevin)
stepper for the FILM geometry, the film-lane counterpart of
solver.step_langevin (2D periodic) and bulk3d.step_langevin_3d (3D periodic).

ADDITIVE MODULE. solvers/film3d.py is NOT touched by this file: the drift is
film3d.explicit_N verbatim (Landau + electrostatic force [+ elastic]), the
transforms are film3d._rfft2/_irfft2 (so film3d.FFT_WORKERS is the one FFT
knob, already set by every gate), and the implicit operator this module
propagates exactly is the SAME operator film3d.build_grad_factors inverts
implicitly. Nothing here is dimension-generic (SPEC Part II ruling (3)): it is
the film's own geometry -- periodic in-plane, natural-BC finite difference in
z, node-volume-weighted TDGL -- written out for the film alone.

======================================================================
1. WHY THE FILM NEEDS ITS OWN OU (and why one eigendecomposition suffices)
======================================================================
film3d's implicit piece is the gradient operator, per in-plane mode k:

    T_k = T_0 + G11*k^2*I          (Nz x Nz, the SAME for all 3 components)

with T_0 = film3d._grad_z_bands -- the natural-BC z-band whose two surface
rows carry a single face's full flux (film3d's module docstring derives it as
the exact gradient of the face-based discrete energy, divided by the node
quadrature weight qw). Because T_0 does not depend on k and the in-plane part
is a scalar multiple of the identity, ONE Nz x Nz eigendecomposition gives the
propagator for EVERY (j,k): the eigenvectors are k-independent and the
eigenvalues are lambda_j + G11*k^2. So the exact linear propagator
exp(-dt*L*T_k) costs two small matmuls per step, never a dense 3D operator
(SPEC's standing ban) and never a per-k dense expm.

T_0 is NOT symmetric as a plain matrix (row 0 has -2*G11/hz^2 where column 0
has -G11/hz^2). It is symmetric with respect to the FD node volumes: with

    W_hat = diag(w),  w_j = qw_j/hz = 1 at interior nodes, 1/2 at the two
                            surfaces        (half control volume),

W_hat*T_0 is symmetric, i.e. T_0 is self-adjoint in the W_hat inner product
and the right object is the generalized symmetric eigenproblem

    W_hat*T_0*v = lambda*W_hat*v   <=>   B*u = lambda*u,
    B = W_hat^(1/2) * T_0 * W_hat^(-1/2)  (symmetric),  v = W_hat^(-1/2)*u.

_z_eigen solves the symmetrized form with np.linalg.eigh (real symmetric ->
real eigenvalues, orthogonal U), so V = W_hat^(-1/2)*U is W_hat-orthonormal:
V^T W_hat V = I. Exactly one eigenvalue is 0 -- the z-uniform vector (every
row of T_0 sums to zero, boundary rows included) -- and the rest are positive
for G11 > 0; solver._eig_roundoff_floor floors roundoff-negatives to 0, the
same floor solver._ou_from_lin uses (SPEC §10). G11 < 0 makes T_0 genuinely
negative and is REFUSED by build_ou_operators_film: this OU carries only the
isotropic G11 (no H11 k^4 regulariser, no anisotropic G), so a G11 < 0 film
OU has no bounded propagator to give (SPEC §10.1(2)).

======================================================================
2. THE DISCRETE FDT FOR A NODE-VOLUME-WEIGHTED TDGL  (kT_deck_film)
======================================================================
film3d.explicit_N's own docstring states the film's equation of motion:

    qw_j * dP_j/dt = -L * dF/dP_j       per unit in-plane area,

i.e. with the node CONTROL VOLUME V_j = hx*hy*qw_j = (hx*hy*hz)*w_j,

    dP_j/dt = -(L/V_j) * dF/dP_j  + noise_j.

This is a TDGL whose mobility is node-volume-weighted (bulk3d's is not: every
3D node has the same volume). The Fokker-Planck stationary distribution of
dx/dt = -M dF/dx + eta is exp(-F/kBT) iff <eta eta> = 2*M*kB*T*delta, so the
film's FDT reads, per node and per step,

    Var[noise_j] = 2*(L/V_j)*kB*T*dt
                 = 2*L*dt * (kB*T/(hx*hy*hz)) / w_j.          (*)

THE NOISE IS WHITE PER NODE VOLUME: the two surface nodes, owning half a
cell, fluctuate exactly twice as hard as an interior node. That is the whole
discrete-FDT content of the film's half-cell weights, and it is what
kT_deck_film packages:

    kT_deck_film = kB*T_kelvin * 1e18 / (hx_nm*hy_nm*hz_nm)

-- kB*T as a deck ENERGY (the 1e18 of bulk3d.kT_deck_3d: deck energy density
is SI*1e-9 and deck length is SI*1e-9 per axis, so a true volume integral
picks up 1e-9/(1e-9)^3 = 1e18; the film's F IS a true volume integral, area *
sum(qw * density), unlike solver.py's 2D F which is an energy per unit
z-length and needs units.si_to_deck_kT's invented Dz) -- divided by the
INTERIOR node's control volume in nm^3. The surface half-cells are NOT in this
scalar: they are carried exactly by w_j, through the W_hat-orthonormal
eigenbasis, which is why one scalar kT suffices for every node. Same
discipline as units.si_to_deck_kT: explicit factors, a named choice, no opaque
constant.

EQUIVALENT MODE STATEMENT (what the gates measure). Write the quadratic part
of F in the eigenbasis. With P_hat = rfft2(P) (in-plane, axis order (Ny,Nxr)
exactly as film3d) and y = V^T W_hat P_hat / sqrt(Nx*Ny),

    F = (hx*hy*hz) * sum_over_full_k sum_j (1/2) * chi_j(k) * |y_j(k)|^2,
    chi_j(k) = lambda_j + G11*k^2 + (bulk curvature, if any),

so equipartition gives, exactly,

    <|y_j(k)|^2> = kT_deck_film / chi_j(k)        (K3.film_fdt_mode_variance)
    (1/2)*chi*<|y|^2> = kT/2 per mode             (K3.film_fdt_equipartition)

`mode_amplitudes` returns exactly this y. The 1/sqrt(Nx*Ny) is the in-plane
Parseval factor of the unnormalised rfft2 and nothing else.

CONVENTION NOTE -- RESOLVED, SPEC V2-04.0 (D16), 2026-09-16. This module's
kT IS the shared convention now: units.kT_langevin = k_B*T*1e18/V_cell_nm3,
which kT_deck_film delegates to. Because that kT is normalised PER NODE
VOLUME, the real-space noise draw below is rfft2'd WITHOUT a 1/sqrt(Ny*Nx) --
and solver.step_langevin / bulk3d.step_langevin_3d no longer apply one either,
so all three steppers now state the same per-mode law on the unnormalised
transform, <|rfft(P)|^2> = N_cells*kT/chi. The divergence this note used to
flag (reported in reports/wp_v2_03/section_P.md rather than quietly
reconciled) was measured in REVIEW_v2_03c_kt.md and ruled a numerical error in
the other two engines, not a convention split: their noise was weak by exactly
N_cells/V_cell. units.si_to_deck_kT and bulk3d.kT_deck_3d keep their
arithmetic but are unit conversions, not FDT coefficients, and no Langevin
call site uses them any more. K5 `kt_convention` is the authority.

======================================================================
3. HERMITIAN NOISE, self-conjugate columns
======================================================================
Identical discipline to solver.step_langevin, for the identical reason: ONE
full-array real-space draw per step from a single caller-supplied Generator,
then rfft2. rfft2 of a genuine i.i.d. real field is automatically
Hermitian-consistent -- correct conjugate-pair/self-conjugate variance split
at DC and at the Nyquist row/column -- so no per-mode conjugate bookkeeping is
hand-coded here either. The z transform (U^T, orthogonal, real) acts on a
DIFFERENT axis and commutes with the in-plane transform, and an orthogonal
real map of i.i.d. unit normals is again i.i.d. unit normals, so the z
eigenbasis preserves that property exactly.

Route A, unchanged: the Landau force, the electrostatic force and (optionally)
the elastic force are the EXPLICIT Euler-Maruyama drift -- film3d.explicit_N,
called verbatim -- and only the gradient operator is propagated exactly. The
O(dt) bias of that explicit drift is the same one G16a/K3.fdt_mode_variance
calibrates in 2D (the `bias_knob` number).

GATES: K3.film_fdt_mode_variance, K3.film_fdt_equipartition,
K2.film_langevin_kt0, K2.film_langevin_reproducibility, K4.film_ou_cost
(gates/core/_lib/{k2,k3,k4}_v2_03b_p.py).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from fftpf import film_engine as f3
from fftpf import spectral_operators as operators
from fftpf import pf2d_engine as solver
from fftpf import units


def node_weights(grid):
    """Dimensionless FD node weights w_j = qw_j/hz: exactly 1.0 at interior
    nodes and exactly 0.5 at the two surfaces (a boundary node owns half a
    control volume). The node's control volume is hx*hy*hz*w_j; the hx*hy*hz
    part is the scalar in kT_deck_film, the w_j part is what the
    W_hat-orthonormal eigenbasis carries."""
    return grid["qw"] / grid["hz"]


def _z_eigen(grid, G11):
    """The ONE eigendecomposition (module docstring §1): returns
    (lam, U, sqrt_w) with lam (Nz,) ascending, U (Nz,Nz) orthogonal, and
    V = U/sqrt(w) the W_hat-orthonormal eigenvectors of the natural-BC z-band
    T_0 = film3d._grad_z_bands(Nz, hz, G11). lam >= 0 for G11 >= 0, after the
    roundoff floor of SPEC §10 (solver._eig_roundoff_floor, the one
    solver._ou_from_lin uses); a G11 < 0 band would keep its genuinely negative
    eigenvalues, but its one caller, build_ou_operators_film, refuses G11 < 0."""
    Nz, hz = grid["Nz"], grid["hz"]
    sub, diag, sup = f3._grad_z_bands(Nz, hz, G11)
    T0 = np.diag(diag) + np.diag(sup, 1) + np.diag(sub, -1)
    sqrt_w = np.sqrt(node_weights(grid))
    B = T0 * sqrt_w[:, None] / sqrt_w[None, :]     # W_hat^(1/2) T_0 W_hat^(-1/2)
    B = 0.5 * (B + B.T)                            # symmetric by construction; make it so bit-wise
    lam, U = np.linalg.eigh(B)
    lam = solver._eig_roundoff_floor(lam)          # SPEC §10: floor roundoff-negatives (solver._ou_from_lin)
    return lam, U, sqrt_w


def build_ou_operators_film(grid, G11, dt, L, kT, noise_cutoff_nm=None):
    """Exact per-(j,k) OU propagator of the film's IMPLICIT operator
    T_k = T_0 + G11*k^2*I (SPEC V2-03 B1; SPEC §18 for the OU formulas):

        Lambda_jk = L*(lambda_j + G11*k^2)
        decay_jk  = exp(-Lambda_jk*dt)
        var_jk    = 2*kT*L*dt*phi1(2*Lambda_jk*dt)   [= kT*(1-exp(-2*Lambda*dt))/Lambda]

    written with solver._phi1 so the Lambda=0 mode -- the z-uniform eigenvector
    at k=0, the film's only zero mode -- needs no branch anywhere (solver._phi1's
    docstring carries that argument; it is reused, not re-derived).

    G11 < 0 IS REFUSED (ValueError, SPEC §10.1(2)). This OU carries only the
    isotropic G11 z-band + G11*k^2 -- no H11 k^4 term, no anisotropic G -- so
    with G11 < 0 every stiffness is <= 0 and its magnitude grows with k
    without bound: there is no regulariser and the film OU is ill-posed.
    (The 2D/3D OU builders keep negative eigenvalues instead: their ops can
    carry H11.)

    RETURNED KEYS: U/sqrt_w/lam (the z eigenbasis, §1), stiffness and Lambda
    (Nz,Ny,Nxr), decay and noise_std (Nz,Ny,Nxr, z-first -- the layout the step
    applies), plus the scalars kT/dt/L and n_inplane = Nx*Ny (the Parseval
    factor of `mode_amplitudes`).

    NOISE CUTOFF (optional, SPEC V2-04.0 D16 -- the same knob as
    solver.build_ou_operators). noise_cutoff_nm = l_c zeroes `noise_std` (never
    `decay`) on every mode with |k| >= 2*pi/l_c, so the excluded modes relax
    deterministically and step_langevin_film is untouched. None returns the
    pre-V2-04 dict bit for bit.
    THE FILM'S |k|^2, stated exactly. In-plane it is KXr^2+KYr^2, this
    builder's own `k2`. Out of plane the film has no Fourier index at all --
    it has the natural-BC z-band's eigenvalues lam_j (§1) -- so the effective
    z-wavenumber is read off the stiffness the eigenproblem already produced:

        kz_eff,j^2 = (lam_j - lam_0) / G11,   lam_0 = min(lam)

    lam_0 IS the z-uniform eigenvalue (every row of T_0 sums to zero, boundary
    rows included, so the constant vector is an exact null vector and
    _z_eigen's floor leaves it at 0.0); it is subtracted so the uniform z
    mode has kz_eff = 0 exactly, as a k=0 mode must. Dividing by G11 inverts
    lam = G11*kz^2, the same relation stiffness = lam + G11*k2 uses in plane,
    so one cutoff length means the same physical wavelength in all three
    directions. G11 = 0 leaves T_0 identically zero; kz_eff is then 0 for
    every j, which is what the formula's limit says. G11 < 0 (negative lam_j,
    for which lam_0 would no longer be the uniform mode) never reaches here:
    it is refused above (SPEC §10.1(2))."""
    if G11 < 0:
        raise ValueError(
            f"build_ou_operators_film: G11={G11!r} < 0 refused -- the film OU carries only "
            "the isotropic G11 z-band + G11*k^2, with no H11 (k^4) regulariser, so a G11 < 0 "
            "film OU is ill-posed (SPEC §10.1(2))")
    lam, U, sqrt_w = _z_eigen(grid, G11)
    k2 = operators.k_squared(grid, 2)                             # (Ny,Nxr)
    stiffness = lam[:, None, None] + G11 * k2[None, :, :]         # (Nz,Ny,Nxr)
    Lambda = L * stiffness
    var = 2.0 * kT * L * dt * solver._phi1(2.0 * Lambda * dt)
    out = dict(U=U, sqrt_w=sqrt_w, lam=lam, k2=k2,
               stiffness=stiffness, Lambda=Lambda,
               decay=np.exp(-Lambda * dt), noise_std=np.sqrt(var),
               kT=float(kT), dt=float(dt), L=float(L),
               n_inplane=int(grid["Nx"] * grid["Ny"]))
    if noise_cutoff_nm is None:
        return out
    kz2 = np.zeros_like(lam) if G11 == 0 else (lam - float(np.min(lam))) / G11
    keep = solver._noise_cutoff_mask(kz2[:, None, None] + k2[None, :, :], noise_cutoff_nm)
    out["noise_std"] = out["noise_std"] * keep
    out["kz_eff2"] = kz2
    out["noise_cutoff_nm"] = float(noise_cutoff_nm)
    out["noise_k_cut"] = 2.0 * np.pi / float(noise_cutoff_nm)
    out["noise_modes_kept"] = int(keep.sum())
    out["noise_modes_total"] = int(keep.size)
    return out


def step_langevin_film(P, alpha, grid, ou, poisson_factors, dt, L, E_ext, rng, el=None):
    """One first-order stochastic-splitting step of the film TDGL (SPEC §18,
    route A). Explicit Euler-Maruyama drift of the nonlinear force --
    film3d.explicit_N (Landau + electrostatic [+ elastic when el= is given]),
    called verbatim, so this stepper and film3d.step differ ONLY in how the
    gradient operator is advanced -- then the exact per-(j,k) OU update of the
    implicit gradient family: rfft2 in-plane, W_hat^(1/2) scaling and U^T in z,
    scalar decay+noise per mode, back.

    The noise: ONE full-array real-space standard-normal draw per step from the
    caller's Generator (bitwise (seed,step) reproducible -- K2.film_langevin_
    reproducibility), rfft2'd WITHOUT a 1/sqrt(Nx*Ny) rescale (module docstring
    §2's convention note: kT_deck_film is normalised per node volume, which
    puts that factor in kT rather than in the draw) and rotated by the same
    orthogonal U^T, which preserves i.i.d. unit variance exactly."""
    N = f3.explicit_N(P, alpha, grid, poisson_factors, E_ext, el=el)
    Pk_star = f3._rfft2(P) - dt * L * f3._rfft2(N)                 # (3,Nz,Ny,Nxr)
    sqrt_w, U = ou["sqrt_w"], ou["U"]

    Qk = Pk_star * sqrt_w[None, :, None, None]
    y = np.tensordot(U, Qk, axes=([0], [1]))                       # (Nz,3,Ny,Nxr), z-mode first

    eta = rng.standard_normal((3, grid["Nz"], grid["Ny"], grid["Nx"]))
    xi = np.tensordot(U, f3._rfft2(eta), axes=([0], [1]))          # (Nz,3,Ny,Nxr)

    y_new = ou["decay"][:, None] * y + ou["noise_std"][:, None] * xi
    Qk_new = np.tensordot(U, y_new, axes=([1], [0]))               # (Nz,3,Ny,Nxr), z-node first
    Pk_new = np.moveaxis(Qk_new, 0, 1) / sqrt_w[None, :, None, None]
    return f3._irfft2(Pk_new, (grid["Ny"], grid["Nx"]))


def mode_amplitudes(P, grid, ou):
    """The film's FDT observable: y_j(k) = V^T W_hat rfft2(P) / sqrt(Nx*Ny),
    shape (Nz,3,Ny,Nxr) (z-mode first, matching ou's decay/noise_std layout).
    Its equilibrium variance is kT/chi_j(k) exactly -- module docstring §2."""
    Qk = f3._rfft2(P) * ou["sqrt_w"][None, :, None, None]
    return np.tensordot(ou["U"], Qk, axes=([0], [1])) / np.sqrt(ou["n_inplane"])


def kT_deck_film(T_kelvin, hx_nm, hy_nm, hz_nm, k_B=units.K_BOLTZMANN_SI):
    """kT for step_langevin_film, in deck units, derived in the module
    docstring §2. SPEC V2-04.0 (D16) promoted this derivation to THE shared
    convention of all three engines, so the body is now one call to
    units.kT_langevin -- the same arithmetic, returned byte-identically, in
    the one place the convention is documented. The explicit factors
    (mirroring units.si_to_deck_kT's discipline) are kept here:

        kT_SI        = k_B*T_kelvin                         [J]
        kT_deck      = kT_SI * 1e18                         [deck energy -- the film's F
                                                             is a TRUE volume integral,
                                                             so bulk3d.kT_deck_3d's factor]
        V_cell_nm3   = hx_nm*hy_nm*hz_nm                    [interior node control volume]
        kT_deck_film = kT_deck / V_cell_nm3                 [deck energy density]

    The two SURFACE nodes' half control volumes are deliberately NOT in this
    scalar -- they are carried exactly by the weights w_j = 1/2 inside the
    W_hat-orthonormal eigenbasis, which is what makes the noise white per node
    volume with a single kT. Still NOT units.si_to_deck_kT (2D, energy per
    unit z-length) and NOT bulk3d.kT_deck_3d (a plain energy): those two are
    unit conversions of k_B*T into F_total units, not the FDT coefficient
    (SPEC V2-04.0, D16)."""
    return units.kT_langevin(T_kelvin, hx_nm, hy_nm, hz_nm, k_B=k_B)
