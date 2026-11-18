"""WP2: Khachaturyan homogeneous-C elasticity (SPEC §1.4).

Tensors (eigenstrain eps_star, homogeneous strain ebar, periodic strain
delta_eps, total elastic strain e, stress sigma) are stored as FULL symmetric
(3,3,Ny,Nx) arrays (both (i,j) and (j,i) populated), and C as the full
(3,3,3,3) elasticity tensor. This is the "do NOT hand-decouple" philosophy
SPEC §1.4 mandates for the Omega operator, applied consistently everywhere
in this module: the C:e contractions are plain einsum over real tensor
indices, so the Voigt shear-factor-2 bookkeeping (SPEC §5's "C:ε shear 2"
suspect) falls out of the einsum automatically instead of being hand-coded.
The ONE exception is _green_delta_eps, whose three einsums are written out
as explicit per-component sums (WP28 P6, SPEC §39.1 A2'(i)); each sum is
labelled with the einsum string it replaces, and the k-space formulas it
implements are SPEC §1.4's own, unchanged and byte-identical.

Cost note: the tensors are still STORED as full symmetric (3,3,...) arrays
(the WP2_REPORT.md simplicity trade), but since WP28 P6 only their 6
independent components are transformed -- 12 FFTs per Green apply, not 18,
which is SPEC §2's own count (SPEC §39.1 A2'(iii)). The tax that motivated
this was measured on the 3D twin (bulk3d.elastic_solve_3d): SPEC §38.1 A2 /
reports/REVIEW_hotpath_profile.md §1, transform share corrected in SPEC
§39.1 K2 -- 1.71 ms of the 5.75 ms elastic tax at 32^3, the rest
einsum-dominated. This 2D module has not itself been profiled.

Scope: homogeneous-C (this docstring's original WP2 scope). Two ebar modes
(SPEC §1.4): (a) stress-free (ebar_ij = <eps_star_ij>), the WP2 default;
(b) film misfit (SPEC §6.2c), selected by passing eps_m to setup_elastic.
WP3 §6.2b adds an optional static eigenstrain disorder field
(eps_star_static) added to eps*(P) before it is used to solve for
u/delta_eps/e/sigma -- the P-dependent driving force formula itself is
unchanged (eps_star_static's P-derivative is exactly zero).

WP10 SPEC §16 extends this module (bottom section) with INHOMOGENEOUS C(r)
support via an elastic SPM (setup_elastic_spm/elastic_solve_spm) -- see
that section's own docstring block. elastic_solve dispatches to it
automatically when el["spm"] is set; elastic_force needs no changes at all
(already dispatch-transparent, see its own docstring). Every function
above this point is UNCHANGED and still homogeneous-C only, stress-free
mode (a) or film-misfit mode (b), exactly as this original docstring
describes.

The very bottom section adds an OPT-IN conjugate-gradient solve of the same
SPM problem (setup_elastic_spm(..., method="cg"), _spm_cg_elastic, shared
with bulk3d_spm); the default method="basic" path is untouched.
docs/CG_SPM.md.
"""
import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import scipy.fft as spfft
import sympy as sp

from fftpf import spectral_operators as operators
from fftpf import units
from fftpf import input_validation as validate

# WP4 §10(b): scipy.fft rfft2/irfft2, workers=-1 -- duplicated (not imported)
# from solver.py's identical helpers to avoid a circular import (solver.py
# itself does `import elastic`); see solver.py's FFT_WORKERS block for the
# Hermitian-consistency argument (identical here: eps*, tau are real fields,
# Omega(k) is a real quadratic-in-k multiplier, so rfft2/irfft2 on the
# half-spectrum grid reproduce fft2/ifft2 exactly, up to roundoff).
FFT_WORKERS = -1
_CPU_KERNELS = None  # Optional fused arithmetic, selected by execution.py.


def _rfft2(x):
    return spfft.rfft2(x, axes=(-2, -1), workers=FFT_WORKERS)


def _irfft2(xk, s):
    return spfft.irfft2(xk, s=s, axes=(-2, -1), workers=FFT_WORKERS)


# --------------------------------------------------------------------------
# Cubic elasticity tensor C_ijkl (SPEC §1.4: deck C11, C12, C44)
# --------------------------------------------------------------------------
def build_C_tensor(C11, C12, C44):
    """Cubic stiffness C_ijkl built from deck C11/C12/C44 (SPEC §1.4), shape (3,3,3,3)."""
    d = np.eye(3)
    C = (C12 * np.einsum("ij,kl->ijkl", d, d)
         + C44 * (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d)))
    for i in range(3):
        C[i, i, i, i] += (C11 - C12 - 2 * C44)
    return C  # shape (3,3,3,3)


def contract_C(C, e):
    """sigma_ij = C_ijkl e_kl, e/sigma shape (3,3,Ny,Nx) or (3,3)."""
    return np.einsum("ijkl,kl...->ij...", C, e)


# --------------------------------------------------------------------------
# Eigenstrain eps*(P), SPEC §1.4 (TENSOR components, Q11_phys/Q12_phys/Q44_phys)
# --------------------------------------------------------------------------
def eigenstrain(P, Q11, Q12, Q44):
    """eps*_ij(P), SPEC §1.4, in this module's VOIGT-flavoured Q convention:
    the shear slot is e[0,1] = 0.5*Q44*P1*P2, so Q44 here is the printed
    Q44_phys of a Voigt-sourced deck (SPEC §1.4 forbids the legacy /4-hacked
    Q11/Q12/Q44 material block). bulk3d.eigenstrain_3d uses FULL-TENSOR
    Q_ijkl instead (e[0,1] = 2*Q1212*P1*P2), so the two agree iff
    Q44 = 4*Q1212 -- SPEC §37.5's named trap, derived at bulk3d.py's
    eigenstrain_3d docstring. P: (3,Ny,Nx); returns (3,3,Ny,Nx)."""
    if _CPU_KERNELS is not None:
        return _CPU_KERNELS.eigenstrain(P, Q11, Q12, Q44)
    P1, P2, P3 = P[0], P[1], P[2]
    e = np.zeros((3, 3) + P.shape[1:])
    e[0, 0] = Q11 * P1**2 + Q12 * (P2**2 + P3**2)
    e[1, 1] = Q11 * P2**2 + Q12 * (P1**2 + P3**2)
    e[2, 2] = Q11 * P3**2 + Q12 * (P1**2 + P2**2)
    e[0, 1] = e[1, 0] = 0.5 * Q44 * P1 * P2
    e[0, 2] = e[2, 0] = 0.5 * Q44 * P1 * P3
    e[1, 2] = e[2, 1] = 0.5 * Q44 * P2 * P3
    return e


def elastic_force_explicit(P, sigma, Q11, Q12, Q44):
    """delta F_el / delta P_m, SPEC §1.4 explicit formula (matches
    KhachaturyanCouplingPolarDerivative.C's dSpontaneousStrain.doubleContraction(sigma))."""
    if _CPU_KERNELS is not None:
        return _CPU_KERNELS.elastic_force(P, sigma, Q11, Q12, Q44)
    P1, P2, P3 = P[0], P[1], P[2]
    s11, s22, s33 = sigma[0, 0], sigma[1, 1], sigma[2, 2]
    s12, s13, s23 = sigma[0, 1], sigma[0, 2], sigma[1, 2]
    f1 = -(2 * P1 * (s11 * Q11 + (s22 + s33) * Q12) + Q44 * (s12 * P2 + s13 * P3))
    f2 = -(2 * P2 * (s22 * Q11 + (s11 + s33) * Q12) + Q44 * (s12 * P1 + s23 * P3))
    f3 = -(2 * P3 * (s33 * Q11 + (s11 + s22) * Q12) + Q44 * (s13 * P1 + s23 * P2))
    return np.stack([f1, f2, f3])


# --------------------------------------------------------------------------
# Symbolic cross-check (SPEC §1.4: "generate via sympy from eps*(P) and check
# both agree"). Run once at import time; raises if the explicit formula above
# doesn't match -sigma_ij d(eps*_ij)/dP_m for a generic symbolic sigma.
# --------------------------------------------------------------------------
def _verify_elastic_force_symbolic():
    P1, P2, P3, Q11, Q12, Q44 = sp.symbols("P1 P2 P3 Q11 Q12 Q44", real=True)
    s11, s22, s33, s12, s13, s23 = sp.symbols("s11 s22 s33 s12 s13 s23", real=True)
    Pv = [P1, P2, P3]

    es = {
        (0, 0): Q11 * P1**2 + Q12 * (P2**2 + P3**2),
        (1, 1): Q11 * P2**2 + Q12 * (P1**2 + P3**2),
        (2, 2): Q11 * P3**2 + Q12 * (P1**2 + P2**2),
        (0, 1): sp.Rational(1, 2) * Q44 * P1 * P2,
        (0, 2): sp.Rational(1, 2) * Q44 * P1 * P3,
        (1, 2): sp.Rational(1, 2) * Q44 * P2 * P3,
    }
    es[(1, 0)], es[(2, 0)], es[(2, 1)] = es[(0, 1)], es[(0, 2)], es[(1, 2)]
    sg = {(0, 0): s11, (1, 1): s22, (2, 2): s33,
          (0, 1): s12, (1, 0): s12, (0, 2): s13, (2, 0): s13, (1, 2): s23, (2, 1): s23}

    explicit = [
        -(2 * P1 * (s11 * Q11 + (s22 + s33) * Q12) + Q44 * (s12 * P2 + s13 * P3)),
        -(2 * P2 * (s22 * Q11 + (s11 + s33) * Q12) + Q44 * (s12 * P1 + s23 * P3)),
        -(2 * P3 * (s33 * Q11 + (s11 + s22) * Q12) + Q44 * (s13 * P1 + s23 * P2)),
    ]
    for m in range(3):
        generic = -sum(sp.diff(es[(i, j)], Pv[m]) * sg[(i, j)] for i in range(3) for j in range(3))
        if sp.simplify(generic - explicit[m]) != 0:
            raise AssertionError(f"elastic driving force mismatch at component {m}: "
                                  f"generic={generic} vs explicit={explicit[m]}")


_verify_elastic_force_symbolic()


# --------------------------------------------------------------------------
# Omega(k) = [C_ijkl k_j k_l]^-1, built once per (grid, C) -- SPEC §1.4:
# "build numerically from full C_ijkl and invert per k -- do NOT hand-decouple"
# --------------------------------------------------------------------------
# Real skew-adjoint first-derivative wavevectors (each component zeroed on its
# own even-axis Nyquist line, so the even quadratic-in-k elastic operator is a
# real quadratic form whose variation is the force): the shared
# operators.masked_wavevectors_2d -- the same object as
# solver._modified_wavevectors_2d; this name is kept as an alias of it.
_modified_wavevectors_2d = operators.masked_wavevectors_2d


def build_Omega(grid, C):
    """Omega(k) on the rfft2 half-spectrum grid (WP4 §10b), shape (Ny,Nxr,3,3),
    and K (3,Ny,Nxr) with k3 == 0: operators.green_operator(grid, C, 2), the
    one Green-operator builder bulk3d.build_Omega_3d shares (masked
    wavevectors; DC plus every even-axis Nyquist intersection regularised
    before the inverse and zeroed after)."""
    Omega, K, _zero_k = operators.green_operator(grid, C, 2)
    return Omega, K  # Omega shape (Ny,Nxr,3,3); K shape (3,Ny,Nxr)


def setup_elastic(grid, C11, C12, C44, Q11, Q12, Q44, eps_star_static=None, eps_m=None,
                   film_full=False):
    """eps_star_static: optional (3,3,Ny,Nx) static eigenstrain disorder field
    (SPEC §6.2b), added to eps*(P) before the mechanical solve. With a
    batched P (3,S,Ny,Nx) that one field is shared by every member, or a
    (3,3,S,Ny,Nx) one gives each member its own (_ensemble_field).
    eps_m: None => mode (a) stress-free (SPEC default). A float => mode (b)
    film misfit (SPEC §6.2c): ebar_11=ebar_22=eps_m, ebar_12=ebar_13=ebar_23=0,
    ebar_33 from sigma_33=0 -- UNLESS film_full=True (SPEC WP6 §12.1), which
    additionally solves ebar_13/ebar_23 from sigma_13=sigma_23=0 instead of
    assuming them zero (homogeneous_strain_film_full instead of
    homogeneous_strain_misfit; eps_m is still the e0 value in this mode).
    film_full=True therefore REQUIRES eps_m: without it the solve used to
    fall into mode (a) stress-free silently; it now raises ValueError.

    RETURNED KEYS: "C" (the full (3,3,3,3) tensor -- KEPT, gates/cases read it
    and build_Omega needs it), "C11"/"C12"/"C44" (this function's own three
    cubic arguments, stashed verbatim so elastic_solve can contract with
    contract_C_cubic -- WP28 T3, see elastic_solve), "Omega", "K", "zero_k",
    "green_pre" (WP28 P6), "Q11"/"Q12"/"Q44", "eps_star_static", "eps_m",
    "film_full", "Nx"/"Ny"."""
    validate.setup_elastic(grid, C11, C12, C44, Q11, Q12, Q44, eps_star_static, eps_m)
    if film_full and eps_m is None:
        raise ValueError("setup_elastic: film_full=True (SPEC WP6 §12.1 full film constraint) "
                         "needs eps_m, the in-plane misfit e0; without it the solve would run "
                         "stress-free mode (a)")
    C = build_C_tensor(C11, C12, C44)
    # build_Omega's own builder; zero_k is the set it regularised, i.e. the
    # masked-wavevector zero set (not the raw grid frequencies, whose even-axis
    # Nyquist signs would split the two).
    Omega, K, zero_k = operators.green_operator(grid, C, 2)
    # green_pre: the complex K/Omega copies the Green apply multiplies by,
    # built once here (WP28 P6, SPEC §39.1 A2'(i)) -- see green_precompute.
    return dict(C=C, C11=C11, C12=C12, C44=C44,
                Omega=Omega, K=K, Q11=Q11, Q12=Q12, Q44=Q44, zero_k=zero_k,
                green_pre=green_precompute(K, Omega),
                eps_star_static=eps_star_static, eps_m=eps_m, film_full=film_full,
                Nx=grid["Nx"], Ny=grid["Ny"])  # stashed for elastic_solve's irfft2


def homogeneous_strain_misfit(eps_star_mean, eps_m, C11, C12):
    """SPEC §1.4 mode (b) / §6.2c: ebar_11=ebar_22=eps_m, ebar_12=ebar_13=ebar_23=0,
    ebar_33 from C11(ebar_33-<eps*_33>) + C12(ebar_11-<eps*_11>) + C12(ebar_22-<eps*_22>) = 0."""
    e11, e22, e33 = eps_star_mean[0, 0], eps_star_mean[1, 1], eps_star_mean[2, 2]
    ebar33 = e33 - (C12 / C11) * ((eps_m - e11) + (eps_m - e22))
    ebar = np.zeros((3, 3))
    ebar[0, 0] = eps_m
    ebar[1, 1] = eps_m
    ebar[2, 2] = ebar33
    return ebar


def homogeneous_strain_film_full(eps_star_mean, e0, C11, C12):
    """SPEC WP6 §12.1: FULL film constraint -- ebar_11=ebar_22=e0, ebar_12=0;
    sigma_33=sigma_13=sigma_23=0 (mode (b)'s eps_m generalized: e0 plays the
    same role, PLUS the two shear components are now also solved instead of
    assumed zero).

    Cubic symmetry decouples shear from normal strains EXACTLY (build_C_tensor
    has C_1212=C_1313=C_2323=C44 and zero cross terms to any e_11/e_22/e_33),
    so sigma_i3 = 2*C44*(ebar_i3 - eps*_i3_mean) for i=1,2 -- sigma_i3=0 gives
    ebar_13=eps*_13_mean, ebar_23=eps*_23_mean directly, independent of C44's
    value (same "cubic C closed form" SPEC asks for; no 3x3 linear solve is
    actually needed). ebar_33 is the UNCHANGED homogeneous_strain_misfit
    formula (sigma_33=0 with the two prescribed in-plane strains -- e0 here
    plays eps_m's exact role)."""
    e11, e22, e33 = eps_star_mean[0, 0], eps_star_mean[1, 1], eps_star_mean[2, 2]
    e13, e23 = eps_star_mean[0, 2], eps_star_mean[1, 2]
    ebar33 = e33 - (C12 / C11) * ((e0 - e11) + (e0 - e22))
    ebar = np.zeros((3, 3))
    ebar[0, 0] = e0
    ebar[1, 1] = e0
    ebar[2, 2] = ebar33
    ebar[0, 2] = ebar[2, 0] = e13
    ebar[1, 2] = ebar[2, 1] = e23
    return ebar


# --------------------------------------------------------------------------
# Mechanical solve at fixed P (SPEC §1.4: tau_hat = [C:eps*_hat], u_hat, delta_eps)
# --------------------------------------------------------------------------
# The 6 independent components of a symmetric 3x3 tensor, in the order the
# 6-component storage below stacks them (SPEC §39.1 A2'(iii)).
_SYM6 = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def green_precompute(K, Omega):
    """The P-independent complex operands _green_delta_eps multiplies:
      Kc  (3,...)   -- K, as complex128
      Omc (3,3,...) -- Omega with its TRAILING (3,3) moved to the front,
                       contiguous, as complex128
    COMPLEX, not real, and in exactly this layout, because that is what
    np.einsum's own dtype promotion produced from (float64 K, complex128
    tau_hat) -- matching it is what makes the explicit sums byte-identical
    to the einsums they replace (SPEC §39.1 A2'(i)).
    Hoisted once at setup by setup_elastic / setup_elastic_spm /
    bulk3d.setup_elastic_3d (each documents the key in its own return);
    a caller that omits `pre` gets it rebuilt per call, same numbers.
    Costs one extra copy of Omega (2x its bytes: 2.5 MB at 32^3, 1.2 MB on
    a 128^2 grid)."""
    Kc = np.ascontiguousarray(K).astype(np.complex128)
    Omc = np.ascontiguousarray(np.moveaxis(Omega, (-2, -1), (0, 1))).astype(np.complex128)
    return dict(Kc=Kc, Omc=Omc)


def _green_delta_eps(tau, K, Omega, zero_k, shape, fwd=_rfft2, inv=_irfft2, pre=None):
    """delta_eps(r) from a real-space (3,3,[Nz,]Ny,Nx) source tau, via the shared
    Green's-operator machinery (SPEC §1.4's u_hat/delta_eps_hat formulas):
    u_hat = -i*Omega_im*k_j*tau_hat_mj, delta_eps_hat = sym(i*k_i*u_hat_j),
    delta_eps(k=0)=0. Extracted from elastic_solve (WP10 SPEC §16, a pure
    refactor -- zero behavior change, verified by the full regression suite)
    so it can be reused by elastic_solve_spm's SPM iteration below, which
    calls this SAME function once per iterate with a DIFFERENT (C(r)-
    dependent) source instead of the homogeneous tau=[C:eps*] SPEC gives here.

    WP4 §10(e): an optional LEADING ensemble axis broadcasts through -- K and
    Omega have no batch axis, so they broadcast in as if S=1 against a
    (3,3,S,Ny,Nx) tau, exactly as intended (bit-identical to the WP2 form for
    unbatched P, verified: gate G7e). That was the point of the "..." in the
    einsums this function used until WP28 P6; the explicit component sums
    below inherit it, because Kc[i] / Omc[i,j] are (Ny,Nxr) and numpy
    left-pads them against the (S,Ny,Nxr) spectra the same way.

    fwd/inv: the real forward/inverse transform pair, fwd(x) and inv(xk,
    shape). The defaults are this module's own 2-axis _rfft2/_irfft2 (which
    read elastic.FFT_WORKERS at call time, unchanged) -- every 2D caller is
    byte-identical to before.
    bulk3d passes its 3-axis _rfftn/_irfftn (at bulk3d.FFT_WORKERS=1, a
    DELIBERATELY different knob -- see solver.py's FFT_WORKERS block), which
    is the ONLY thing its own copy of this function ever differed in: that
    copy is now deleted (SPEC §38.1 B1 = REVIEW_arch_minimality.md F3,
    "line-for-line identical except _rfft2/_irfft2 -> _rfftn/_irfftn").
    The formulas are axis-count-agnostic already: K/Omega carry the axis
    count, and nothing below indexes a k-axis explicitly.

    pre: green_precompute(K, Omega)'s dict, hoisted at setup by every
    in-tree caller; None rebuilds it here (same numbers, one extra copy).

    PRECONDITION (WP28 P6, SPEC §39.1 A2'(iii)): tau must be SYMMETRIC in
    (i,j) -- only its 6 independent components are transformed, and only 6
    delta_eps components come back, so 12 transforms instead of 18. Every
    source in this repo is C:(a symmetric strain) with C cubic, which is
    symmetric slot-for-slot by construction (contract_C_cubic already reads
    only e[0,1]/e[0,2]/e[1,2] for the same reason)."""
    if pre is None:
        pre = green_precompute(K, Omega)
    Kc, Omc = pre["Kc"], pre["Omc"]
    if _CPU_KERNELS is not None:
        return _CPU_KERNELS.green_delta_eps(tau, Kc, Omc, zero_k, shape, fwd, inv)

    # tau_hat, 6 unique components in _SYM6 order (was fwd(tau), 9 transforms)
    t00, t11, t22, t01, t02, t12 = fwd(np.stack([tau[i, j] for i, j in _SYM6]))

    # replaces np.einsum("j...,mj...->m...", K, tau_hat) -- SPEC §39.1 A2'(i)
    # v_m(k) = sum_j k_j tau_hat_mj, j summed in the order 0,1,2
    v0 = Kc[0] * t00 + Kc[1] * t01 + Kc[2] * t02
    v1 = Kc[0] * t01 + Kc[1] * t11 + Kc[2] * t12
    v2 = Kc[0] * t02 + Kc[1] * t12 + Kc[2] * t22

    # replaces -1j * np.einsum("...im,m...->i...", Omega, v) -- SPEC §39.1 A2'(i)
    # u_hat_i(k) = -i * sum_m Omega_im v_m, m summed in the order 0,1,2
    u0 = -1j * (Omc[0, 0] * v0 + Omc[0, 1] * v1 + Omc[0, 2] * v2)
    u1 = -1j * (Omc[1, 0] * v0 + Omc[1, 1] * v1 + Omc[1, 2] * v2)
    u2 = -1j * (Omc[2, 0] * v0 + Omc[2, 1] * v1 + Omc[2, 2] * v2)
    u0 = u0 * (~zero_k)  # gauge choice, no net rigid translation
    u1 = u1 * (~zero_k)
    u2 = u2 * (~zero_k)

    # replaces KU = np.einsum("i...,j...->ij...", K, u_hat); 0.5j*(KU + KU^T)
    # delta_eps_hat_ij(k) = 0.5i (k_i u_hat_j + k_j u_hat_i), 6 unique
    # components in _SYM6 order -- SPEC §39.1 A2'(i)+(iii)
    de6 = np.empty((6,) + u0.shape, dtype=np.complex128)
    # The 3 diagonal lines say (k_i u_i + k_i u_i), NOT 2j*k_i*u_i, on purpose: that IS the 0.5j*(KU + KU^T) sum the einsum evaluated, and it is what keeps them byte-identical -- do not "simplify".
    de6[0] = 0.5j * (Kc[0] * u0 + Kc[0] * u0)
    de6[1] = 0.5j * (Kc[1] * u1 + Kc[1] * u1)
    de6[2] = 0.5j * (Kc[2] * u2 + Kc[2] * u2)
    de6[3] = 0.5j * (Kc[0] * u1 + Kc[1] * u0)
    de6[4] = 0.5j * (Kc[0] * u2 + Kc[2] * u0)
    de6[5] = 0.5j * (Kc[1] * u2 + Kc[2] * u1)
    de6 = de6 * (~zero_k)  # SPEC: delta_eps(k=0)=0

    d6 = inv(de6, shape)  # (6,...) real (was 9 inverse transforms)
    delta_eps = np.empty((3, 3) + d6.shape[1:], dtype=d6.dtype)
    for n, (i, j) in enumerate(_SYM6):
        delta_eps[i, j] = d6[n]
        delta_eps[j, i] = d6[n]
    return delta_eps


# Align a P-independent field (eps_star_static, a field-shaped E_ext) with a
# BATCHED quantity (ensemble axis S right after the component axes): the ONE
# implementation is operators.ensemble_field (docstring there), shared by
# elastic_solve, bulk3d.elastic_solve_3d, bulk3d_spm and the E_ext
# normaliser; this name is kept as an alias of it.
_ensemble_field = operators.ensemble_field


def elastic_solve(P, el):
    """Returns dict with eps_star (total, incl. static disorder if any),
    eps_star_P (P-dependent part only), ebar, delta_eps, e (total elastic
    strain), sigma.

    WP10 SPEC §16: dispatches to elastic_solve_spm when `el` was built by
    setup_elastic_spm (el["spm"]=True, inhomogeneous C(r)) -- every OTHER
    caller (el built by setup_elastic, homogeneous C) is completely
    unaffected, el.get("spm") is None/False for them, bit-identical to
    pre-WP10 (the full existing gate suite is the regression check).

    WP28 T3 (SPEC §39.1 A2'(ii) + the §40.5 coordinator ruling): the two
    HOMOGENEOUS-path contractions below were contract_C(el["C"], .), the
    81-component rank-4 einsum; they are now contract_C_cubic(C11,C12,C44, .),
    the 15-operation cubic form, which is what bulk3d.elastic_solve_3d already
    does since T2. THE POINT is not speed here but that both engines now do
    IDENTICAL ARITHMETIC, which restores gates/g30c's Nz=1 anchor: bulk3d's
    Nz=1 path reduces to this one BIT-FOR-BIT again (T2 alone had split them
    by 3-5e-16 relative). Scope, deliberately: elastic_solve_spm's own
    zero-order contract_C(C0, .) and film_elastic.py are UNTOUCHED, and
    el["C"] (the tensor) stays in the dict.
    EQUIVALENCE CLASS vs pre-T3: TOLERANCE, ~1e-11 relerr (no byte-class cubic
    contraction exists -- LC C2: np.einsum's 81-term accumulation is not the
    naive sequential order). GATES: G0e, G5c/d, G14a-e, G30a-e (G30c is the
    bit-for-bit one), G27a, G7e."""
    if el.get("spm"):
        return elastic_solve_spm(P, el)
    eps_star_P = eigenstrain(P, el["Q11"], el["Q12"], el["Q44"])  # (3,3,Ny,Nx)
    static = el.get("eps_star_static")
    if static is not None and P.ndim == 4:  # ensemble P (3,S,Ny,Nx): S axis on a (3,3,Ny,Nx) static
        static = operators.ensemble_field(static, 2, eps_star_P.ndim, "eps_star_static")
    eps_star = eps_star_P if static is None else eps_star_P + static

    s = (el["Ny"], el["Nx"])
    # WP28 T3 (§39.1 A2'(ii), §40.5 ruling): cubic form, same arithmetic as bulk3d
    tau = contract_C_cubic(el["C11"], el["C12"], el["C44"], eps_star)  # (3,3,Ny,Nx), real space
    delta_eps = _green_delta_eps(tau, el["K"], el["Omega"], el["zero_k"], s, pre=el["green_pre"])

    eps_star_mean = eps_star.mean(axis=(-2, -1))
    eps_m = el.get("eps_m")
    if eps_m is None:
        if el.get("film_full"):  # setup_elastic refuses this too; guards hand-edited el dicts
            raise ValueError("elastic_solve: el has film_full=True but no eps_m -- the full "
                             "film constraint needs the misfit e0 (setup_elastic's eps_m)")
        ebar = eps_star_mean  # mode (a) stress-free: ebar_ij = <eps*_ij>
    elif el.get("film_full"):
        # SPEC WP6 §12.1 mode (c): full film constraint (also NOT batch-safe,
        # same documented scope limit as mode (b) below).
        ebar = homogeneous_strain_film_full(eps_star_mean, eps_m, el["C"][0, 0, 0, 0], el["C"][0, 0, 1, 1])
    else:
        # mode (b) film misfit -- NOT batch-safe (SPEC scopes WP4 §10e to the
        # default stress-free mode; homogeneous_strain_misfit hard-codes a
        # (3,3) ebar, so an ensemble here would need a real fix, not just an
        # ellipsis -- no gate needs it, so left as the WP2 single-realization
        # form and documented as a scope limit rather than silently patched).
        ebar = homogeneous_strain_misfit(eps_star_mean, eps_m, el["C"][0, 0, 0, 0], el["C"][0, 0, 1, 1])

    e = ebar[..., None, None] + delta_eps - eps_star
    sigma = contract_C_cubic(el["C11"], el["C12"], el["C44"], e)  # WP28 T3, as above
    return dict(eps_star=eps_star, eps_star_P=eps_star_P, ebar=ebar, delta_eps=delta_eps, e=e, sigma=sigma)


def elastic_energy_density(sol):
    """f_el = 1/2 sigma:e (full double contraction -> shear factor 2 falls out of einsum)."""
    return 0.5 * np.einsum("ij...,ij...->...", sol["sigma"], sol["e"])


def elastic_force(P, el):
    """delta F_el / delta P (re-solves u at the given P -- the '1 elastic solve per
    step' of SPEC §2). Returns (force, sol) so callers can reuse sol for energy/diagnostics.

    WP10 SPEC §16: NO change needed here for the C(r) SPM path -- elastic_solve
    dispatches internally, and elastic_force_explicit only needs (P, sigma,
    Q11, Q12, Q44), agnostic to how sigma was computed (SPEC: "Driving force:
    unchanged envelope form -sigma:dEps*/dP with sigma from LOCAL C(r)")."""
    sol = elastic_solve(P, el)
    force = elastic_force_explicit(P, sol["sigma"], el["Q11"], el["Q12"], el["Q44"])
    return force, sol


# ==========================================================================
# WP10 SPEC §16: elastic SPM (Wang et al., Acta Mater 61 (2013) 7591,
# Appendix A) -- inhomogeneous C(r) = C0 + dC(r). Zero order (dC=0) is
# EXACTLY the homogeneous solver above: Appendix A Eqs (A7)-(A10) reduce to
# the SAME c0_ijkl/Green-tensor (their G_ik(q), our Omega) construction
# already built by build_C_tensor/build_Omega -- no new k-space operator is
# introduced, only (i) a new REAL-SPACE source term inside the SAME Green's-
# operator call (_green_delta_eps, shared with the homogeneous path above)
# and (ii) an outer homogeneous-strain (ebar) self-consistency loop (their
# Eq A16), absent from the homogeneous case because there <delta_sigma>=0
# identically (delta_eps has zero mean by construction; a spatially UNIFORM
# C0 commutes with that mean) -- with C(r) inhomogeneous, delta_eps(r) and
# C(r) are correlated, so <delta_sigma> generally is NOT zero and ebar must
# be iterated.
#
# Cubic-form shortcut: every C(r) this project ever builds (PZT's Table-4-
# interpolated C11(x),C12(x),C44(x), pzt_field.elastic_C_field) is CUBIC-
# SYMMETRIC pointwise, so C(r) and dC(r) are each fully described by 3
# scalar-or-(Ny,Nx)-array numbers, and c(r):e(r) is exactly SPEC §1.4's
# "Cubic contraction" formula evaluated with C11/C12/C44 as arrays instead
# of scalars (plain numpy broadcasting, pointwise/local -- no coupling
# between grid points, so this is EXACT, not an approximation, for any
# broadcastable C11/C12/C44 shape). This is a performance/memory choice
# (avoids materializing a (3,3,3,3,Ny,Nx) tensor), not a different formula:
# _verify_contract_C_cubic cross-checks it against the existing full-tensor
# contract_C()+build_C_tensor() path at import time.
# ==========================================================================
def contract_C_cubic(C11, C12, C44, e):
    """sigma_ij = C(r):e(r) for a CUBIC (possibly spatially-varying) C given
    as 3 scalar-or-array coefficients -- SPEC §1.4's cubic contraction
    formula, generalized to array coefficients. e/sigma shape (3,3,...)."""
    if _CPU_KERNELS is not None:
        return _CPU_KERNELS.contract_cubic(C11, C12, C44, e)
    s = np.zeros_like(e)
    e11, e22, e33 = e[0, 0], e[1, 1], e[2, 2]
    e12, e13, e23 = e[0, 1], e[0, 2], e[1, 2]
    s[0, 0] = C11 * e11 + C12 * (e22 + e33)
    s[1, 1] = C11 * e22 + C12 * (e11 + e33)
    s[2, 2] = C11 * e33 + C12 * (e11 + e22)
    s[0, 1] = s[1, 0] = 2 * C44 * e12
    s[0, 2] = s[2, 0] = 2 * C44 * e13
    s[1, 2] = s[2, 1] = 2 * C44 * e23
    return s


def contract_S_cubic(s11, s12, s44_voigt, sigma):
    """e_ij = S(r):sigma(r), the COMPLIANCE contraction for a cubic tensor
    given its VOIGT compliances (s11, s12, s44_voigt -- e.g.
    units.cubic_stiffness_to_compliance's output). NOT the same shear
    factor as contract_C_cubic: engineering/Voigt shear gamma12=2*e12
    relates to stress via gamma12=s44_voigt*sigma12, so the TENSOR strain is
    e12=(s44_voigt/2)*sigma12 -- HALF the Voigt value (contract_C_cubic's
    stiffness direction instead carries a FACTOR OF 2: sigma12=2*C44*e12).
    This is exactly the class of factor-of-2/4 shear bug SPEC §5 warns
    about; caught here by _verify_contract_C_cubic's numeric round-trip
    check below, not asserted from memory."""
    e = np.zeros_like(sigma)
    s11_, s22_, s33_ = sigma[0, 0], sigma[1, 1], sigma[2, 2]
    s12_, s13_, s23_ = sigma[0, 1], sigma[0, 2], sigma[1, 2]
    e[0, 0] = s11 * s11_ + s12 * (s22_ + s33_)
    e[1, 1] = s11 * s22_ + s12 * (s11_ + s33_)
    e[2, 2] = s11 * s33_ + s12 * (s11_ + s22_)
    e[0, 1] = e[1, 0] = 0.5 * s44_voigt * s12_
    e[0, 2] = e[2, 0] = 0.5 * s44_voigt * s13_
    e[1, 2] = e[2, 1] = 0.5 * s44_voigt * s23_
    return e


def _verify_contract_C_cubic():
    """Numeric self-check (run once at import, cheap): (1) contract_C_cubic
    (scalar C11/C12/C44) must agree with the existing full-tensor
    contract_C(build_C_tensor(...), e) path -- the two independent
    formulations WP2's "do NOT hand-decouple" philosophy and WP10's cubic-
    shortcut both need to agree on. (2) contract_S_cubic(cubic_stiffness_to_
    compliance(...)) must INVERT contract_C_cubic exactly -- catches the
    shear-factor class of bug SPEC §5 flags as the usual suspect, instead of
    trusting the derivation by inspection alone."""
    rng = np.random.default_rng(20260821)
    e = rng.standard_normal((3, 3, 4, 5))
    e = 0.5 * (e + np.swapaxes(e, 0, 1))  # symmetrize
    C11, C12, C44 = 210.3, 87.6, 54.1
    s_cubic = contract_C_cubic(C11, C12, C44, e)
    s_tensor = contract_C(build_C_tensor(C11, C12, C44), e)
    if not np.allclose(s_cubic, s_tensor, rtol=1e-13, atol=1e-13):
        raise AssertionError("contract_C_cubic disagrees with the full-tensor contract_C path")

    s11, s12, s44v = units.cubic_stiffness_to_compliance(C11, C12, C44)
    e_back = contract_S_cubic(s11, s12, s44v, s_cubic)
    if not np.allclose(e_back, e, rtol=1e-10, atol=1e-12):
        raise AssertionError("contract_S_cubic does not invert contract_C_cubic -- shear-factor bug")


_verify_contract_C_cubic()


# A final inner residual above this is a DIVERGED elastic SPM solve, not a
# slow one. The inner residual is the relative max-norm step of delta_eps
# (elastic_solve_spm's `resid`): a merely slow fixed point stops at the cap
# with a small one (G14b's sharp 2x inclusion needs ~400-600 iterations to
# reach 1e-10; a 98%-contrast field stops a few decades above spm_tol), while
# the C0-too-soft divergence (setup_elastic_spm's CONVERGENCE CONDITION)
# leaves it O(1) with the force off by orders of magnitude, or non-finite.
SPM_DIVERGED_RESID = 1e-3


def _cubic_C0_hint(C11, C12, C44):
    """The C0_override suggestion for a cubic-field C(r): the midpoint of
    each modulus' min and max over the grid -- the reference that satisfies
    the CONVERGENCE CONDITION for any positive-definite field."""
    mid = tuple(0.5 * (float(np.min(c)) + float(np.max(c))) for c in (C11, C12, C44))
    return "C0_override=(C11,C12,C44) near the min/max midpoint ({:.6g}, {:.6g}, {:.6g})".format(*mid)


def _spm_convergence_check(where, sigma, n_inner, resid_inner, n_outer, resid_outer, el,
                           check_outer, c0_hint, method="basic"):
    """The elastic SPM's own non-convergence report -- shared by
    elastic_solve_spm and bulk3d_spm.elastic_solve_spm_3d, called on the
    FINAL state before it is returned or cached. The inner loop stops at
    spm_max_iter whatever it holds, and the outer test (max|<sigma>|/
    max|sigma|) is driven to ~1e-11 by the ebar update even when
    delta_eps has diverged -- so without this check a diverged solve came
    back looking converged.
      - non-finite sigma, or resid_inner > SPM_DIVERGED_RESID: RuntimeError
        (nothing usable to return), naming resid_inner/n_inner and the
        C0_override remedy (c0_hint() builds the suggestion, only here);
      - resid_inner > spm_tol, or (check_outer: k0 "stress_free") resid_outer
        > outer_tol: RuntimeWarning, result returned unchanged -- the
        electrostatic SPM's own "[module.function] WARNING: SPM did NOT
        converge within ..." report (solver._spm_iterate), issued through
        `warnings` so a caller can filter or escalate it.
    A converged solve passes through with no side effect at all.
    method="cg" (the opt-in conjugate-gradient solve, _spm_cg_elastic) only
    changes the RuntimeError's wording: CG cannot diverge on a positive-
    definite C(r), so a residual that large means the iteration cap ran out
    (or C(r) is not positive definite), not that C0 is too soft."""
    if method == "cg" and not (np.all(np.isfinite(sigma)) and resid_inner <= SPM_DIVERGED_RESID):
        raise RuntimeError(
            f"[{where}] elastic CG solve FAILED: final relative residual resid_inner="
            f"{resid_inner:.3e} after n_inner={n_inner} CG iterations (spm_max_iter="
            f"{el['spm_max_iter']}, spm_tol={el['spm_tol']:.0e}), resid_outer={resid_outer:.3e}, "
            f"sigma {'finite' if np.all(np.isfinite(sigma)) else 'NON-FINITE'}. CG converges "
            "for any positive-definite C(r), in O(sqrt(stiffness contrast)) iterations: raise "
            "spm_max_iter, and check that every local C(r) is positive definite.")
    if not (np.all(np.isfinite(sigma)) and resid_inner <= SPM_DIVERGED_RESID):
        raise RuntimeError(
            f"[{where}] elastic SPM DIVERGED: final resid_inner={resid_inner:.3e} after "
            f"n_inner={n_inner} (spm_max_iter={el['spm_max_iter']}, spm_tol={el['spm_tol']:.0e}), "
            f"n_outer={n_outer}, resid_outer={resid_outer:.3e}, sigma "
            f"{'finite' if np.all(np.isfinite(sigma)) else 'NON-FINITE'}. The basic SPM "
            "iteration converges only while the Green's reference C0 bounds every local "
            "stiffness (0 < C(r) < 2*C0 in each stiffness eigenmode, i.e. C0 >~ max C / 2); "
            "this el's C0 (default: the volume mean <C(r)>) is too soft for its stiffest "
            f"cells. Rebuild the el with {c0_hint()} -- the converged answer does not depend "
            "on C0.")
    failed = []
    if resid_inner > el["spm_tol"]:
        failed.append(f"inner loop within spm_max_iter={el['spm_max_iter']} (final "
                      f"resid_inner={resid_inner:.3e} > spm_tol={el['spm_tol']:.0e}, "
                      f"n_inner={n_inner})")
    if check_outer and resid_outer > el["outer_tol"]:
        failed.append(f"outer ebar loop within outer_max_iter={el['outer_max_iter']} (final "
                      f"resid_outer={resid_outer:.3e} > outer_tol={el['outer_tol']:.0e}, "
                      f"n_outer={n_outer})")
    if failed:
        warnings.warn(f"[{where}] WARNING: elastic SPM did NOT converge: "
                      + "; ".join(failed) + ". The result is returned as is.",
                      RuntimeWarning, stacklevel=2)


def setup_elastic_spm(grid, C11_field, C12_field, C44_field, Q11, Q12, Q44,
                       eps_star_static=None, spm_tol=1e-10, spm_max_iter=None,
                       outer_tol=1e-10, outer_max_iter=50, C0_override=None,
                       method="basic", cg_precond=True):
    """WP10 SPEC §16: inhomogeneous-C(r) elasticity via the Wang et al. 2013
    Appendix A elastic SPM. C11_field/C12_field/C44_field: scalar or
    (Ny,Nx) array (cubic-form C(r), SPEC's "C(r)=C0+dC(r)").

    method: "basic" (default -- the fixed-point SPM below, bit-for-bit the
    pre-existing path and el dict) or "cg" (opt-in: the Galerkin/
    preconditioned conjugate-gradient solve of the SAME discrete equations,
    _spm_cg_elastic -- converges for any positive-definite C(r), no C0
    condition, O(sqrt(contrast)) iterations; docs/CG_SPM.md). cg_precond
    (cg only): True preconditions with the reference-medium Green's
    operator of C0 (the mean, or C0_override), False runs the plain
    Galerkin CG with the identity-reference compatibility projection.
    spm_max_iter=None means 100 for "basic" (its long-standing cap) and
    CG_DEFAULT_MAX_ITER CG iterations for "cg"; spm_tol is then the CG
    relative residual and outer_tol still bounds max|<sigma>|/max|sigma|.

    C0 = the ARITHMETIC MEAN of each field over the whole grid by default
    (SPEC: "C0 = arithmetic mean unless convergence dictates otherwise") --
    the SAME homogeneous reference used to build Omega/K (build_Omega,
    UNCHANGED from the homogeneous solver -- Appendix A's Green tensor G(q)
    always uses c0 only, never the local c(r), confirmed by Eq A9/A10, so
    reusing build_C_tensor/build_Omega here is not an approximation).
    C0_override=(C11_0,C12_0,C44_0): explicit override for the SPEC-required
    "show the converged answer is C0-independent" demonstration (gate G14b)
    -- not used by production callers (default: mean).

    CONVERGENCE CONDITION -- C0-independence is a property of the FIXED
    POINT, not of the iteration that has to reach it. The basic scheme in
    elastic_solve_spm contracts only while C0 bounds every local stiffness,
    0 < C(r) < 2*C0 in each stiffness eigenmode, i.e. C0 >~ max C(r)/2. The
    default mean breaks that once a stiff minority phase exceeds ~2x the
    mean (e.g. a 2^3 inclusion of contrast 2.25 in an 8^3 box), and the
    inner iteration then DIVERGES instead of merely slowing down; a
    C0_override near the midpoint 0.5*(min C + max C) of the moduli
    converges. elastic_solve_spm raises on a diverged solve and warns on
    one stopped short of its tolerances (_spm_convergence_check).

    IMPORTANT: C0 (the Green's-operator/zero-order REFERENCE tensor, freely
    choosable per SPEC's "unless convergence dictates otherwise") is a
    DIFFERENT quantity from the "<c_ijkl(r)>" appearing in Appendix A's Eq
    (A16) ebar update, which is ALWAYS the TRUE (area-weighted) mean of the
    actual local C(r) field, regardless of what reference C0 the Green's
    operator uses -- conflating the two (using S0=C0^-1 in the Eq A16 update
    even when C0_override != the true mean) is a bug: it makes the outer
    ebar loop converge to the WRONG fixed point (found empirically while
    building gate G14b's C0-independence check -- a C0_override 1.5x the
    matrix properties left the outer residual permanently stuck at ~1e-2,
    not decreasing even after 300 iterations, because the ebar update was
    solving <c(r)>:ebar = ... using the OVERRIDE's compliance instead of the
    true mean's). S_MEAN (below) is therefore ALWAYS computed from the
    field's true mean, independent of C0_override; S0 (the override-able
    reference tensor's own compliance) is kept only for potential diagnostic
    use, not used in the ebar update.

    Returns an el dict carrying el["spm"]=True, dispatched by elastic_solve
    (elastic_force needs no changes at all, see its docstring) with ZERO
    effect on any OTHER caller (el built by setup_elastic).

    NOT frozen after setup: elastic_solve_spm(..., warm_start=True) writes
    "_warm_ebar" and "_warm_delta_eps" back into THIS dict, so an el passed
    through solver.relax carries per-call history between steps (that is the
    warm start). Copy the dict if a caller needs a cold-start el."""
    spm_max_iter = _resolve_spm_method(method, spm_max_iter, "setup_elastic_spm")
    C11_mean = float(np.mean(C11_field))
    C12_mean = float(np.mean(C12_field))
    C44_mean = float(np.mean(C44_field))
    if C0_override is not None:
        C11_0, C12_0, C44_0 = C0_override
    else:
        C11_0, C12_0, C44_0 = C11_mean, C12_mean, C44_mean
    C0 = build_C_tensor(C11_0, C12_0, C44_0)
    # Same Green builder and zero_k as setup_elastic: the Green operator and
    # its gauge/regularization must agree exactly.
    Omega, K, zero_k = operators.green_operator(grid, C0, 2)

    dC11 = np.asarray(C11_field, dtype=float) - C11_0
    dC12 = np.asarray(C12_field, dtype=float) - C12_0
    dC44 = np.asarray(C44_field, dtype=float) - C44_0
    S0 = units.cubic_stiffness_to_compliance(C11_0, C12_0, C44_0)  # reference tensor's own compliance (diagnostic only)
    S_MEAN = units.cubic_stiffness_to_compliance(C11_mean, C12_mean, C44_mean)  # Eq (A16)'s <c(r)>^-1 -- ALWAYS the true mean

    el = dict(spm=True, C11=C11_field, C12=C12_field, C44=C44_field,
              C11_0=C11_0, C12_0=C12_0, C44_0=C44_0, C0=C0,
              dC11=dC11, dC12=dC12, dC44=dC44, S0=S0, S_MEAN=S_MEAN,
              Omega=Omega, K=K, zero_k=zero_k, green_pre=green_precompute(K, Omega),
              Q11=Q11, Q12=Q12, Q44=Q44, eps_star_static=eps_star_static,
              spm_tol=spm_tol, spm_max_iter=spm_max_iter,
              outer_tol=outer_tol, outer_max_iter=outer_max_iter,
              Nx=grid["Nx"], Ny=grid["Ny"])
    if method == "cg":
        # Only a CG el gains keys: a "basic" el is the pre-existing dict, key for key.
        el.update(spm_method="cg", cg_precond=bool(cg_precond))
        if not cg_precond:
            I_s = build_C_tensor(1.0, 0.0, 0.5)     # identity on symmetric tensors
            Om_I, K_I, zk_I = operators.green_operator(grid, I_s, 2)
            el.update(cg_Omega=Om_I, cg_K=K_I, cg_zero_k=zk_I,
                      cg_green_pre=green_precompute(K_I, Om_I))
    return el


def elastic_solve_spm(P, el, warm_start=True):
    """WP10 SPEC §16 elastic SPM solve (Wang et al. 2013 Appendix A): inner
    fixed-point on delta_eps(r) at FIXED ebar (tol spm_tol, cap spm_max_iter
    -- SPEC: "Fixed point on the strain field, tol 1e-10, cap 100"), wrapped
    in an outer loop that updates ebar from the <sigma>=0 condition (their
    Eq A16-style volume averages, S_MEAN=<c(r)>^-1 -- the TRUE field mean's
    compliance, NOT the possibly-overridden Green's-operator reference C0's
    compliance, see setup_elastic_spm's docstring) until the domain-mean
    stress is negligible vs max|sigma| (outer_tol, cap outer_max_iter -- not a SPEC
    number, a documented WP10 choice). Returns the SAME dict keys as the
    homogeneous elastic_solve (eps_star, eps_star_P, ebar, delta_eps, e,
    sigma) plus diagnostics (n_inner, n_outer, resid_inner, resid_outer) so
    gates/reports can inspect iteration counts (SPEC: "document iteration
    counts"). resid_outer is that of the RETURNED state, also when
    outer_max_iter runs out. The stepping path never reads these
    diagnostics (solver._explicit_N discards elastic_force's `sol`), so the
    solve checks itself before returning: a diverged solve (non-finite, or
    resid_inner > SPM_DIVERGED_RESID) raises RuntimeError, one stopped short
    of spm_tol/outer_tol warns (RuntimeWarning) -- _spm_convergence_check.

    Derivation notes (see WP10_REPORT.md for the full derivation):
    - inner source at outer-iterate m: tau^(n) = c(r):(eps*-ebar^(m)) -
      dc(r):delta_eps^(n-1) (Appendix A Eq A11/A14, n=0 term = the
      homogeneous c0-only solve -- SPEC's "zero order = existing homogeneous
      solver", n>=1 corrects with the FULL local c(r) on the eigenstrain-like
      term, which -- UNLIKE the n=0/homogeneous case -- does NOT let ebar
      drop out, since dc(r)*ebar is a genuinely position-dependent source
      once c(r) is inhomogeneous).
    - outer update (Eq A16, sigma_app=0 here -- SPEC's stress-free default,
      the only mode WP10 is in scope for): ebar_new = S_MEAN : (<c(r):eps*(r)>
      - <c(r):delta_eps(r)>), where <c(r):eps*(r)> is P-fixed (computed once,
      not per outer iteration) and <c(r):delta_eps(r)> uses the just-
      converged inner delta_eps(r) at the CURRENT ebar.

    NOT batch-safe (ensemble axis) -- same documented scope limit as
    elastic.py's misfit modes (b)/(c); not needed for WP10's G14a-e
    (single-realization solves/relaxations only).

    Performance (warm_start=True, default): measured 256^2, ~30-40%
    contrast cold-solve cost is ~250x a single homogeneous elastic_solve
    (n_inner~45, n_outer~6) -- too slow to call fresh every time step of a
    multi-thousand-step relax(). TWO warm-start levels, BOTH pure starting-
    GUESS optimizations that cannot change the converged fixed point (same
    tol-based stopping criterion either way, verified: gate G14a's
    degenerate-limit numbers are unaffected by warm_start, and a cold vs
    warm run at the SAME P converges to the SAME sigma/e to well under
    each gate's own tolerance): (1) each outer iteration's inner solve
    starts from the PREVIOUS outer iteration's delta_eps (instead of
    recomputing the zero-order c0-only guess every time -- ebar changes
    only a little between outer iterations); (2) the converged (ebar,
    delta_eps) is cached INSIDE the `el` dict itself (el["_warm_ebar"/
    "_warm_delta_eps"]) and reused as iteration 0's starting guess on the
    NEXT call with the SAME el object -- exploiting that P changes only a
    little between consecutive time steps in solver.relax()'s stepping
    loop (elastic_force/elastic_solve are called every step with the SAME
    el dict, by reference, so this needs zero changes to solver.py).
    A freshly-built el (setup_elastic_spm, no prior solve yet) has no
    cache, so the FIRST call on any el is always a cold start -- gates
    that build one el per test point (G14a-d) are therefore unaffected.

    An el built with setup_elastic_spm(..., method="cg") is solved by
    _elastic_solve_spm_cg instead (same returned keys; docs/CG_SPM.md)."""
    if el.get("spm_method") == "cg":
        return _elastic_solve_spm_cg(P, el, warm_start=warm_start)
    eps_star_P = eigenstrain(P, el["Q11"], el["Q12"], el["Q44"])
    static = el.get("eps_star_static")
    if static is not None and P.ndim == 4:  # ensemble P: as in elastic_solve (_ensemble_field)
        static = operators.ensemble_field(static, 2, eps_star_P.ndim, "eps_star_static")
    eps_star = eps_star_P if static is None else eps_star_P + static

    s = (el["Ny"], el["Nx"])
    K, Omega, zero_k = el["K"], el["Omega"], el["zero_k"]
    green_pre = el["green_pre"]
    C11, C12, C44 = el["C11"], el["C12"], el["C44"]
    dC11, dC12, dC44 = el["dC11"], el["dC12"], el["dC44"]
    C0 = el["C0"]
    s11_m, s12_m, s44v_m = el["S_MEAN"]  # Eq (A16)'s <c(r)>^-1 -- the TRUE field mean, NOT el["S0"]/C0 (see setup_elastic_spm's docstring)

    eps_star_mean = eps_star.mean(axis=(-2, -1))
    # <c(r):eps_star(r)>, FULL local c(r) (Appendix A's <sigma^0(r)>);
    # P is fixed across the outer loop, so this is the same every outer
    # iteration -- computed once, not inside the loop below.
    sigma0_mean = contract_C_cubic(C11, C12, C44, eps_star).mean(axis=(-2, -1))

    def _inner_solve(ebar, d_eps_guess=None):
        driver = eps_star - ebar[..., None, None]
        if d_eps_guess is None or d_eps_guess.shape != eps_star.shape:
            tau0 = contract_C(C0, driver)  # zero order: homogeneous C0 solve (Eq A9/A10)
            d_eps = _green_delta_eps(tau0, K, Omega, zero_k, s, pre=green_pre)
        else:
            d_eps = d_eps_guess  # warm start (see docstring)
        n_inner, resid = 0, 0.0
        for n_inner in range(1, el["spm_max_iter"] + 1):
            tau_n = (contract_C_cubic(C11, C12, C44, driver)
                      - contract_C_cubic(dC11, dC12, dC44, d_eps))
            d_eps_new = _green_delta_eps(tau_n, K, Omega, zero_k, s, pre=green_pre)
            denom = float(np.max(np.abs(d_eps_new)))
            num = float(np.max(np.abs(d_eps_new - d_eps)))
            resid = num / denom if denom > 0 else 0.0
            d_eps = d_eps_new
            if resid <= el["spm_tol"]:
                break
        return d_eps, n_inner, resid

    ebar_warm = el.get("_warm_ebar") if warm_start else None
    d_eps_warm = el.get("_warm_delta_eps") if warm_start else None
    if ebar_warm is not None and ebar_warm.shape == eps_star_mean.shape:
        ebar = ebar_warm.copy()
    else:
        ebar = eps_star_mean.copy()
    d_eps, n_inner, resid_inner = _inner_solve(ebar, d_eps_guess=d_eps_warm)
    n_outer = 0
    outer_resid = float("inf")
    exhausted = False
    for n_outer in range(1, el["outer_max_iter"] + 1):
        e = ebar[..., None, None] + d_eps - eps_star
        sigma = contract_C_cubic(C11, C12, C44, e)
        sigma_mean = sigma.mean(axis=(-2, -1))
        outer_resid = float(np.max(np.abs(sigma_mean))) / max(float(np.max(np.abs(sigma))), 1e-300)
        if outer_resid <= el["outer_tol"]:
            break
        delta_sigma_mean = contract_C_cubic(C11, C12, C44, d_eps).mean(axis=(-2, -1))
        ebar = contract_S_cubic(s11_m, s12_m, s44v_m, sigma0_mean - delta_sigma_mean)
        d_eps, n_inner, resid_inner = _inner_solve(ebar, d_eps_guess=d_eps)  # warm-start from prev outer iter
    else:
        # outer_max_iter exhausted without the break: the outer_resid above
        # was measured BEFORE the last ebar/d_eps update, i.e. it belongs to
        # a state this function does not return.
        exhausted = True

    e = ebar[..., None, None] + d_eps - eps_star
    sigma = contract_C_cubic(C11, C12, C44, e)
    if exhausted:
        # Report the residual of the state actually returned. A converged
        # solve keeps the value it broke on (same ebar/d_eps, so the same
        # sigma and the same number) and does not take this branch at all.
        outer_resid = (float(np.max(np.abs(sigma.mean(axis=(-2, -1)))))
                       / max(float(np.max(np.abs(sigma))), 1e-300))
    _spm_convergence_check("elastic.elastic_solve_spm", sigma, n_inner, resid_inner, n_outer,
                           outer_resid, el, check_outer=True,
                           c0_hint=lambda: _cubic_C0_hint(C11, C12, C44))
    if warm_start:
        el["_warm_ebar"] = ebar
        el["_warm_delta_eps"] = d_eps
    return dict(eps_star=eps_star, eps_star_P=eps_star_P, ebar=ebar, delta_eps=d_eps,
                e=e, sigma=sigma, n_inner=n_inner, n_outer=n_outer,
                resid_inner=resid_inner, resid_outer=outer_resid)


# ==========================================================================
# Opt-in CONJUGATE-GRADIENT solve of the same inhomogeneous-C(r) problem:
# setup_elastic_spm(..., method="cg") here, bulk3d_spm.setup_elastic_spm_3d
# (..., method="cg") in 3D (which reuses _spm_cg_elastic below). The default
# method="basic" never reaches this section. docs/CG_SPM.md has the
# derivation, the benchmark and the limitations; in short:
#
# GALERKIN FORM (Zeman et al., J. Comput. Phys. 229 (2010) 8065; Vondrejc,
# Zeman & Marek, Comput. Math. Appl. 68 (2014) 156). The unknown is the
# TOTAL strain eps = ebar + delta_eps, delta_eps compatible and zero-mean
# on this module's modified-wavevector grid -- exactly the space the basic
# SPM's _green_delta_eps produces, so both methods solve the SAME discrete
# equations and share their fixed point. Equilibrium div sigma = 0 with
# sigma = C(r):(eps - eps*) is G[sigma] = 0, G the L2-orthogonal projection
# onto that space. Under k0 "stress_free" the constant strains (the k=0
# block) belong to the space, so G[sigma] = 0 also says <sigma> = 0 and ONE
# solve yields ebar too -- no outer loop. Under "clamped" ebar is fixed and
# only delta_eps is unknown. eps -> G[C:eps] is self-adjoint positive-
# definite on the space in the inner product (a, b) = < a_ij b_ij >: the
# volume mean of the FULL 9-component contraction of the symmetric (3,3,...)
# storage, i.e. shear pairs counted twice -- the contraction 1/2 sigma:e
# already uses, so the CG energy norm IS the elastic energy and no Voigt
# factor is hand-coded (_sym_inner evaluates it on the 6 independent slots
# with the off-diagonal weight 2).
#
# PRECONDITIONER M = G C_ref G, C_ref uniform. Its inverse on a residual
# stress s is z = Gamma_ref[s] + C_ref^-1:<s> (the second term only while
# ebar is free), Gamma_ref = _green_delta_eps built on C_ref -- the basic
# SPM's own Green apply, so z's fluctuation part is exactly the step one
# basic iteration would take from the current iterate. cg_precond=True:
# C_ref = el's C0 (the mean, or C0_override); False: C_ref = the identity
# on symmetric tensors, i.e. plain Galerkin CG. kappa(M^-1 A) <= the ratio
# of the extreme eigenvalues of C_ref^-1/2 C(r) C_ref^-1/2 over the grid, so
# with C_ref = <C> the iteration count grows like sqrt(contrast), where the
# basic scheme's grows like the contrast and diverges once C0 < max C/2.
#
# The residual is kept UNPROJECTED, s = C:(eps* - eps) = -sigma, updated
# s -= alpha*C:p: every inner product pairs it with a vector of the
# compatible space, where G is the identity, so G is never applied -- ONE
# Green apply (12 FFTs in 3D) per iteration, like one basic iteration.
#
# STOPPING: relative residual in the preconditioner norm,
#     resid = sqrt( (r, M^-1 r) / (b, M^-1 b) ) <= spm_tol,
# b the residual of the zero start eps = ebar_fixed (or 0), AND, while
# ebar is free, max|<sigma>|/max|sigma| <= outer_tol (the basic outer
# test's own measure; also met once <sigma> is at the roundoff floor of the
# load stress, see _spm_cg_elastic). Both are re-checked on a freshly
# recomputed residual before returning; CG restarts from it if recursion
# drift made the updated one optimistic. The cold start is the basic one,
# eps = <eps*> (stress_free) or eps = ebar_fixed (clamped). An ensemble (S)
# axis runs S independent CGs in lockstep (per-member alpha/beta; a
# converged member is frozen).
# ==========================================================================
SPM_METHODS = ("basic", "cg")
CG_DEFAULT_MAX_ITER = 1000
# Weights of the _SYM6 slots in the full 9-component contraction a_ij b_ij.
_SYM6_W = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)


def _resolve_spm_method(method, spm_max_iter, where):
    """Validate `method` and resolve spm_max_iter=None to the method's
    default cap (100 for "basic" -- its long-standing value, so a basic el
    is unchanged -- and CG_DEFAULT_MAX_ITER for "cg")."""
    if method not in SPM_METHODS:
        raise ValueError(f"{where}: method must be one of {SPM_METHODS}, got {method!r}")
    if spm_max_iter is None:
        return 100 if method == "basic" else CG_DEFAULT_MAX_ITER
    return spm_max_iter


def _field_dot(a, b, nsp):
    """Sum over the nsp trailing (spatial) axes of a*b. A leading ensemble
    axis, if present, is kept: returns a float, or an (S,) array."""
    if a.ndim == nsp:
        return float(np.dot(a.ravel(), b.ravel()))
    n = a.shape[0]
    return np.einsum("sn,sn->s", a.reshape(n, -1), b.reshape(n, -1))


def _sym_inner(a, b, nsp):
    """sum_r sum_ij a_ij(r) b_ij(r) for SYMMETRIC (3,3,[S,]*space) a, b --
    the full 9-component contraction, evaluated on the 6 _SYM6 slots with
    the off-diagonal pairs weighted 2."""
    tot = 0.0
    for w, (i, j) in zip(_SYM6_W, _SYM6):
        tot = tot + w * _field_dot(a[i, j], b[i, j], nsp)
    return tot


def _safe_ratio(a, b):
    """a/b where b > 0, else 0 (elementwise; 0-d or (S,))."""
    b = np.asarray(b, dtype=float)
    return np.where(b > 0, np.asarray(a, dtype=float) / np.where(b > 0, b, 1.0), 0.0)


def _spm_cg_elastic(eps_star, contract, gamma, mean_inv, nsp, tol, outer_tol, max_iter,
                    ebar_fixed=None, ebar0=None, d_eps0=None):
    """Preconditioned CG on the Galerkin equations of the section comment.

    eps_star    (3,3,[S,]*space) total eigenstrain; nsp = number of space axes
    contract    e -> C(r):e
    gamma       residual stress s -> Gamma_ref[s] (zero-mean compatible strain)
    mean_inv    m -> C_ref^-1 : m for a (3,3[,S]) m (used only if ebar is free)
    ebar_fixed  None: ebar is UNKNOWN (stress_free -- <sigma> = 0 is part of
                the system); a (3,3[,S]) array: ebar fixed to it (clamped)
    ebar0, d_eps0  warm start; copied, never modified. Cold (None): eps =
                <eps*> while ebar is free (the basic solve's own start --
                exact, so sigma = 0 at once, for a uniform eps*), eps =
                ebar_fixed under clamped
    Returns (ebar, delta_eps, n_iter, resid, load): resid is the final,
    freshly recomputed relative residual (max over ensemble members), load
    the per-member max|C:(eps* - ebar_fixed or 0)| -- the stress scale of
    the zero start, whose roundoff floor _cg_outer_at_floor tests.

    The outer test max|<sigma>|/max|sigma| <= outer_tol also counts as met
    once max|<sigma>| <= 64*eps*load: a mean stress at the roundoff floor of
    the load cannot be reduced further, and for a (near-)uniform eps* -- whose
    exact sigma is 0 -- the ratio of two roundoff-level numbers is
    meaningless (the basic outer loop runs to outer_max_iter there)."""
    space = tuple(range(-nsp, 0))
    ex = (Ellipsis,) + (None,) * nsp
    npts = int(np.prod(eps_star.shape[-nsp:]))
    batched = eps_star.ndim == 3 + nsp
    free = ebar_fixed is None
    mshape = eps_star.shape[:-nsp]

    def bc(c):                                   # per-member scalar against a field
        return c[ex] if batched else c

    def precond(s, sm):
        return (mean_inv(sm) if free else None), gamma(s)

    def ip(s, sm, zb, zd):                       # (s, z) with z = zb + zd in the space
        v = _sym_inner(s, zd, nsp) / npts
        if free:
            v = v + np.einsum("ij...,ij...->...", sm, zb)
        return np.asarray(v, dtype=float)

    if free:
        ebar = eps_star.mean(axis=space) if ebar0 is None else np.array(ebar0, dtype=float)
    else:
        ebar = np.broadcast_to(np.asarray(ebar_fixed, dtype=float), mshape)
    d = np.zeros_like(eps_star) if d_eps0 is None else np.array(d_eps0, dtype=float)
    zero_start = not free and d_eps0 is None     # eps = ebar_fixed exactly: r0 IS b

    def residual():
        s_ = contract(eps_star - (d + ebar[ex]))
        return s_, (s_.mean(axis=space) if free else None)

    s, sm = residual()
    zb, zd = precond(s, sm)
    rz = ip(s, sm, zb, zd)
    if zero_start:
        nb2, sb = rz, s
    else:                                        # (b, M^-1 b): b = the zero start's residual
        sb = contract(eps_star) if free else contract(eps_star - ebar[ex])
        sbm = sb.mean(axis=space) if free else None
        nb2 = ip(sb, sbm, *precond(sb, sbm))
    load = np.abs(sb).max(axis=(0, 1) + space)
    den = np.where(nb2 > 0, nb2, rz)             # b == 0 (e.g. P = 0): relative to the start

    def outer_meas(s_, sm_):                     # per member: max|<sigma>| / max|sigma|
        if not free:
            return np.zeros(mshape[2:])
        num = np.abs(sm_).max(axis=(0, 1))
        den_ = np.abs(s_).max(axis=(0, 1) + space)
        return np.where(_cg_outer_at_floor(num, load), 0.0, num / np.maximum(den_, 1e-300))

    def rel(rz_):
        return np.sqrt(_safe_ratio(np.maximum(rz_, 0.0), den))

    pb, pd = zb, zd
    n, fresh = 0, True
    while True:
        resid, om = rel(rz), outer_meas(s, sm)
        conv = (resid <= tol) & (om <= outer_tol)
        if np.all(conv) or n >= max_iter:
            if not fresh:                        # verify on the TRUE residual
                s, sm = residual()
                zb, zd = precond(s, sm)
                rz = ip(s, sm, zb, zd)
                resid, om = rel(rz), outer_meas(s, sm)
                conv = (resid <= tol) & (om <= outer_tol)
                fresh = True
            if np.all(conv) or n >= max_iter:
                break
            pb, pd = zb, zd                      # restart from the true residual
        active = ~conv
        q = contract(pd + pb[ex]) if free else contract(pd)
        qm = q.mean(axis=space) if free else None
        pq = ip(q, qm, pb, pd)
        if np.any((pq <= 0) & (rz > 0) & active):
            raise RuntimeError("elastic CG: (p, C:p) <= 0 on a nonzero search direction -- "
                               "C(r) is not positive definite on symmetric strains")
        alpha = _safe_ratio(rz, pq) * active
        if free:
            ebar += alpha * pb
            sm = sm - alpha * qm
        d += bc(alpha) * pd
        s -= bc(alpha) * q
        zb, zd = precond(s, sm)
        rz_new = ip(s, sm, zb, zd)
        beta = _safe_ratio(rz_new, rz) * active
        pd = zd + bc(beta) * pd
        if free:
            pb = zb + beta * pb
        rz = rz_new
        n += 1
        fresh = False
    return np.array(ebar), d, n, float(np.max(resid)), load


def _cg_outer_at_floor(mean_abs_max, load):
    """True where a mean stress max|<sigma>| sits at the roundoff floor of
    the load stress scale (64 ulps of it) -- see _spm_cg_elastic."""
    return np.asarray(mean_abs_max) <= 64.0 * np.finfo(float).eps * np.asarray(load)


def _elastic_solve_spm_cg(P, el, warm_start=True):
    """elastic_solve_spm for an el built with method="cg" (stress-free: ebar
    is an unknown of the CG system, section comment above). Same eigenstrain
    / static-eigenstrain / ensemble handling and the SAME returned keys as
    the basic solve: n_inner is the CG iteration count, n_outer is 1 (there
    is no outer loop), resid_inner the CG relative residual, resid_outer the
    RETURNED state's max|<sigma>|/max|sigma| exactly as basic computes it;
    _spm_convergence_check runs on the final state. The warm start reads and
    writes the basic path's own "_warm_ebar"/"_warm_delta_eps" cache."""
    eps_star_P = eigenstrain(P, el["Q11"], el["Q12"], el["Q44"])
    static = el.get("eps_star_static")
    if static is not None and P.ndim == 4:  # ensemble P: as in elastic_solve (_ensemble_field)
        static = operators.ensemble_field(static, 2, eps_star_P.ndim, "eps_star_static")
    eps_star = eps_star_P if static is None else eps_star_P + static

    s = (el["Ny"], el["Nx"])
    C11, C12, C44 = el["C11"], el["C12"], el["C44"]
    if el.get("cg_precond", True):
        K, Omega, zero_k, pre = el["K"], el["Omega"], el["zero_k"], el["green_pre"]
        s11_0, s12_0, s44v_0 = el["S0"]          # C_ref = C0 (mean or override)

        def mean_inv(m):
            return contract_S_cubic(s11_0, s12_0, s44v_0, m)
    else:
        K, Omega, zero_k, pre = el["cg_K"], el["cg_Omega"], el["cg_zero_k"], el["cg_green_pre"]

        def mean_inv(m):                         # C_ref = identity on symmetric tensors
            return m

    def gamma(tau):
        return _green_delta_eps(tau, K, Omega, zero_k, s, pre=pre)

    def contract(e):
        return contract_C_cubic(C11, C12, C44, e)

    mshape = eps_star.shape[:-2]
    ebar_w = el.get("_warm_ebar") if warm_start else None
    d_w = el.get("_warm_delta_eps") if warm_start else None
    warm = (ebar_w is not None and d_w is not None and np.shape(ebar_w) == mshape
            and np.shape(d_w) == eps_star.shape)
    ebar, d_eps, n_cg, resid, load = _spm_cg_elastic(
        eps_star, contract, gamma, mean_inv, 2, el["spm_tol"], el["outer_tol"],
        el["spm_max_iter"], ebar0=ebar_w if warm else None, d_eps0=d_w if warm else None)

    e = ebar[..., None, None] + d_eps - eps_star
    sigma = contract_C_cubic(C11, C12, C44, e)
    sigma_mean_abs = np.abs(sigma.mean(axis=(-2, -1)))
    outer_resid = float(np.max(sigma_mean_abs)) / max(float(np.max(np.abs(sigma))), 1e-300)
    floor = np.all(_cg_outer_at_floor(sigma_mean_abs.max(axis=(0, 1)), load))
    _spm_convergence_check("elastic.elastic_solve_spm[cg]", sigma, n_cg, resid, 1, outer_resid,
                           el, check_outer=not floor,
                           c0_hint=lambda: _cubic_C0_hint(C11, C12, C44), method="cg")
    if warm_start:
        el["_warm_ebar"] = ebar
        el["_warm_delta_eps"] = d_eps
    return dict(eps_star=eps_star, eps_star_P=eps_star_P, ebar=ebar, delta_eps=d_eps,
                e=e, sigma=sigma, n_inner=n_cg, n_outer=1,
                resid_inner=resid, resid_outer=outer_resid)
