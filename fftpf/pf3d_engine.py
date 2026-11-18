"""SPEC S34.5 (B2-C): rfftn-based fully-3D-periodic engine, mirroring
solver.py's semi-implicit stepping contract (ops-dict pattern) for a THIRD
spatial axis (z) that solver.py has never had -- Pan et al., Science 365,
578 (2019) Fig 1B/C's own "32x32x32nm PBC cube" (dossier vi, (i)) needs a
genuine 3D box, not the 2D-in-plane+FD-z film3d.py geometry (dossier ix:
"film3d.py fit: NOT the right anchor... solver.py is the actual analog").

solver.py itself is NOT modified (SPEC's own instruction) -- everything here
is new. WHAT IS REUSED vs REBUILT, and why:
  - f_bulk energy/force (solver.bulk_energy_density/bulk_force): REUSED
    UNCHANGED, imported directly. Both are pure ELEMENTWISE functions of
    P[0],P[1],P[2] with no grid-shape assumption anywhere in their bodies
    (the energy is sympy-lambdified; the force is solver.py's hand-written
    pow-free Horner form since WP28 T1, symbolically verified against the
    same _F_BULK at import) -- they already work on a (3,Nz,Ny,Nx) field
    with zero code changes, verified in gates/g27b.
  - f_grad / f_elec (grad_force, elec_Ed, build_Minv, step, the Langevin/OU
    machinery): REBUILT here, not reused, because solver.py hardcodes a
    2-tuple (Ny,Nx) grid shape and a 2D rfft2/irfft2 call at every one of
    these sites (see e.g. solver.build_Minv's `Ny,Nx = grid["Ny"],
    grid["Nxr"]`, solver.step's `s=(ops["Ny"],ops["Nx"])`) -- there is no
    way to hand it a 3rd axis without editing solver.py itself, which is
    out of scope. This module re-implements the SAME semi-implicit split
    (SPEC §2: bulk[+elastic] explicit, grad+depolarization implicit) at the
    genuinely-3D k-space level (rfftn over axes=(-3,-2,-1), x is the
    halved/last axis, mirroring solver.py's own "z fastest... no, x
    fastest" axis-order convention: solver.py's (Ny,Nx) becomes this
    module's (Nz,Ny,Nx), so a Nz=1 slice reproduces solver.py's own (Ny,Nx)
    convention exactly -- see gates/g27a, the load-bearing cross-engine
    anchor SPEC S34.5 gate (a) asks for).
  - the k-space SYMBOLS themselves (A(k), D(k), the masked wavevectors,
    Omega(k), the Nyquist energy correction, the E_ext / static-eigenstrain
    normalisers): SHARED with solver.py and film3d.py through
    solvers/operators.py, which owns the one even-axis Nyquist rule; the
    private names below (_grad_symbols_3d, _modified_wavevectors_3d,
    _reshape_vec3_3d) are aliases of it.

GRADIENT OPERATOR A(k) -- SPEC S34.5 machinery bullet ("isotropic+
anisotropic gradient family incl. G44/G44p"), dossier (ii)'s eq S5:
  f_grad = (1/2)G11*sum_i(P_i,i)^2 + G12*sum_i!=j(P_i,i P_j,j, each pair
           once) + (1/2)G44*sum_{i<j}[(P_i,j+P_j,i)^2]
           + (1/2)G44'*sum_{i<j}[(P_i,j-P_j,i)^2]
(the FULL 3D form; solver.py's own f_grad is this exact expression with
every d/dz term dropped, i.e. the K3=0 slice below). The k-space stiffness
matrix A_ik(K) = sum_jl K_j K_l d^2f_grad/d(P_i,j)d(P_k,l) was derived
symbolically (sympy, calculus of variations on 9 independent gradient-
component symbols, scratch derivation, not transcribed from any external
source) and its K3=0 slice was checked to reproduce solver.build_operators's
own hand-coded A11/A22/A33/A12 formulas EXACTLY (not just numerically close
-- literally the same closed-form expressions), which is strong independent
confirmation the sign/factor convention here matches this engine's own
gate-tested one:
  A11 = G11*K1^2 + (G44+G44p)*(K2^2+K3^2)
  A22 = G11*K2^2 + (G44+G44p)*(K1^2+K3^2)
  A33 = G11*K3^2 + (G44+G44p)*(K1^2+K2^2)
  A12 = A21 = (G12+G44-G44p)*K1*K2
  A13 = A31 = (G12+G44-G44p)*K1*K3
  A23 = A32 = (G12+G44-G44p)*K2*K3
(K3=0 collapses this to solver.build_operators's A11/A22/A33/A12 bit for
bit, A13=A23=0 -- gate G27a/G27c both exercise this reduction numerically.)

ELECTROSTATICS -- SPEC S34.5 machinery bullet ("3D periodic electrostatics
(transverse projection + kappa_b, the 2D Poisson generalized)"): dossier
(ii)'s f_elec = -Pi*Ei^ex - (1/2)Ei^in*Pi is IDENTICAL in form to solver.py's
own convention; only the depolarization solve D_ij(k) changes, from
solver.py's in-plane-only D11=kx^2/k^2 etc (D33=D13=D23=0 identically,
because k3==0 is BUILT IN to a 2D engine) to the genuine 3D transverse
(longitudinal-projection) tensor D_ij(k) = K_i*K_j/|K|^2 for ALL i,j in
{1,2,3}, D(k=0)=0 -- the direct 3D generalization of the SAME physics
(depolarizing field of a periodic P(r), Poisson's equation
d_i[kappa d_i phi] = d_i P_i solved by projecting P onto the LONGITUDINAL
(along-k) direction).

k = 0 (SPEC § 11, DEPOL-K0). D(k=0)=0 is the TIN-FOIL convention: the
periodic box is short-circuited, so a uniform P pays no depolarization
energy and nothing in the electrostatics pays for a 180-degree wall. It stays
the default. build_operators_3d(..., depol_N=N) instead writes a symmetric
3x3 N (eigenvalues in [0,1]; a scalar n means n*I) into the k = 0 bin of the
six D symbols and touches no other bin:
    E_d(k=0) = -N.<P>/kappa0,   added energy density (1/2)<P>.N.<P>/kappa0.
Every reader of D -- elec_Ed_3d and the energy/force pair, build_Minv_3d,
build_Minv2_3d, build_ou_operators_3d (through solver._lin_matrix, so the
uniform mode gets Lin(0) = N/kappa0 and its exact OU decay and FDT variance)
and the ensemble axis -- picks the entry up with no change of its own.
Caller mapping (documented, not an API): the periodic replicas form a large
sphere embedded in a medium of relative permittivity eps_m (the Ewald
surface term), and with eps_b the background relative permittivity
(kappa0 = eps0*eps_b) the field inside a uniformly polarized sphere gives
    N = eps_b/(eps_b + 2*eps_m) * I;
eps_m -> infinity is the tin-foil default (N = 0), eps_m = 1 a sphere in
vacuum. Not defined with an array kappa (refused): the k = 0 condition of an
inhomogeneous-kappa Poisson problem is not this convention.

SPEC V2-03 B2 -- INHOMOGENEOUS kappa(x,y,z) BY THE SAME ELECTROSTATIC SPM
solver.py carries in 2D (WP5 SS11.1, Wang et al., Acta Mater 61 (2013) 7591
SS2). kappa may now ALSO be a real-space (Nz,Ny,Nx) array: the ops dict then
carries kappa0 (the homogeneous reference every Green's-function solve and
every implicit operator uses), dkappa = kappa(r)-kappa0 and spm=True, and
elec_Ed_3d dispatches to the fixed point _spm_iterate_3d below. A plain
scalar kappa leaves ops["dkappa"] None and EVERY code path bit-identical to
pre-V2-03 (K2 `kappa_const_3d` pins the stronger statement that a constant
ARRAY reproduces the scalar path exactly, too).

TWO DELIBERATE DIFFERENCES FROM solver.py's 2D SPM, both documented where
they bite rather than smoothed over:
  (1) kappa0 DEFAULT. solver.build_operators uses (kappa_min+kappa_max)/2
      (SPEC SS11.1's own 2D definition); SPEC V2-03 B2 words the 3D default
      as the MEAN, so that is what build_operators_3d does -- with
      kappa0_override for the cases where the mean is the wrong reference.
      It genuinely can be: the fixed point's contraction factor is bounded
      by max|dkappa|/kappa0 (the Green's operator D(k) is an orthogonal
      projector, norm 1), and that bound is MINIMIZED by (kappa_min+
      kappa_max)/2, not by the mean. For a thin high-kappa inclusion the
      mean sits close to the majority phase and the bound blows past 1 (the
      10x dead-layer stack of K3 `layered_dielectric_closed_form_3d`:
      5.4 at the mean vs 0.82 at the midpoint), i.e. the iteration DIVERGES
      at the default and converges at the override. Both numbers are
      reported by K4 `spm_residual_3d`; the mean is kept as the default
      because SPEC rules it, not because it is the better reference.
  (2) WARM START (SPEC V2-03 B2, which the 2D has no notion of). OFF by
      default, so every call is a pure function of (P, ops) and the Nz=1
      identity against solver.py's 2D SPM is exact call-for-call (K2
      `spm_nz1_vs_2d`, iteration counts included). build_operators_3d(...,
      spm_warm=True) turns on an ops-carried cache of the last converged
      E_d, which every later solve on that ops dict starts from -- fewer
      iterations inside a stepping loop, at the cost of making the result
      history-dependent at the tol level. _spm_iterate_3d's `warm=` argument
      is the same thing without the hidden state.

ELASTIC -- IMPLEMENTED, SPEC S37.3 (B2-E Stage E1): homogeneous-modulus 3D
Khachaturyan elastic solve (Green tensor from C, eigenstrain eps0_ij=
Q_ijkl(r) P_k P_l), periodic BC, the k-space Omega(k) machinery the S34.5
stub above deferred pending a sourced BFO C/Q set (SPEC S34.1) -- unblocked
by Zhang, J. Appl. Phys. 101, 114105 (2007) (SPEC S37.5, reports/
LIT_bfo_sources.md). Every public function below still accepts
elastic_params=None for API parity with solver.py's own signatures, and that
path is BYTE-IDENTICAL to the pre-E1 tree (gates/g30d_elastic_off_
regression.py); anything else must be the dict setup_elastic_3d(...) below
returns. Full architecture, the elastic_params contract, the tensor-vs-Voigt
Q convention (and its named factor-of-4 trap, SPEC S37.5), and the two k=0
conventions are documented where the code actually lives, just above
build_Omega_3d further down this file -- not duplicated here.

LANGEVIN/OU -- SPEC S34.5 machinery bullet ("Langevin/OU per R1's per-mode
law"): WP12/R1's construction (solver.build_ou_operators/step_langevin,
SPEC §18) is exact-discrete-FDT per LINEAR k-space mode of Lin(k)=A(k)+
D(k)/kappa, and is REUSED here whole (WP28 P3, SPEC §38.1 B2). The one thing
that used to block reuse was the off-diagonal slots: this module has
genuinely nonzero A13/A23/D13/D23/D33 (any kz!=0 mode) where a 2D engine
bakes them to zero. solver._lin_matrix now reads exactly those five with
ops.get(key, 0.0), so it fills the FULL symmetric 3x3 from a 3D ops dict and
the same zeros as before from a 2D one; build_ou_operators_3d below is then
two lines over solver._ou_from_lin (the eigh-based per-mode propagator math,
which never was grid-shape aware -- np.linalg.eigh batches over whatever
leading axes Lin carries -- and the same phi1 helper, already reused).
The kT this stepper takes is an energy DENSITY, units.kT_langevin(T,hx,hy,hz)
-- the one convention of all three engines since SPEC V2-04.0 (D16) -- and
step_langevin_3d does NOT divide its k-space noise by sqrt(N_cells), so the
per-mode law in the unnormalised rfftn variable is <|m|^2> = N_cells*kT/chi.
kT_deck_3d further down is a unit conversion of k_B*T into F_total units and
is NOT that coefficient. build_ou_operators_3d also takes the optional
noise_cutoff_nm of SPEC V2-04.0: the noise (never the decay) is zeroed on
every mode with |k| >= 2*pi/l_c.

FFT_WORKERS=1 (not solver.py's WP4 FFT_WORKERS=-1 multithreaded default):
this WP's own compute-discipline contract keeps every FFT single-threaded
(matches the session-wide OMP/OPENBLAS/MKL_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1 + nice -n 10
policy for this campaign) -- deliberately NOT reusing solver.py's
multithreaded default, a 3D rfftn is small enough here (<=32^3) that thread
setup overhead is not worth paying, and it keeps every gate/production run
reproducible under the exclusive-box/nohup discipline this repo already
uses for its heavier campaigns.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import scipy.fft as spfft

from fftpf import pf2d_engine as solver, units, elastic_base as elastic
from fftpf import spectral_operators as operators
from fftpf import input_validation as validate

FFT_WORKERS = 1


def _rfftn(x):
    return spfft.rfftn(x, axes=(-3, -2, -1), workers=FFT_WORKERS)


def _irfftn(xk, s):
    """s=(Nz,Ny,Nx) required (mirrors solver._irfft2's own docstring: the
    halved axis's even-vs-odd length can't be recovered from Nxr alone)."""
    return spfft.irfftn(xk, s=s, axes=(-3, -2, -1), workers=FFT_WORKERS)


# --------------------------------------------------------------------------
# Grid & k-space linear operators
# --------------------------------------------------------------------------
def make_grid_3d(Nx, Ny, Nz, Lx, Ly, Lz):
    """Mirrors solver.make_grid, one axis added. NAME COLLISION: film3d has
    its own make_grid_3d(Nx, Ny, Lx, Ly, Nz, hf) -- positional #3 is Nz here
    and Lx there, so a positional call to the wrong module builds a silently
    wrong box (ENGINE_GUIDE §2.1 has the full collision table). Axis order (Nz,Ny,Nx)
    throughout this module (component axis 0, then z,y,x) -- Nz=1 then
    slices to EXACTLY solver.py's own (Ny,Nx) convention (gate G27a).
    rfftn convention: x (the LAST axis) is halved, matching solver.py's own
    x-last rfft2 convention exactly (Nxr=Nx//2+1); z and y stay full-length
    (fftfreq, not rfftfreq)."""
    hx, hy, hz = Lx / Nx, Ly / Ny, Lz / Nz
    kx1d = operators.k_axis(Nx, hx)
    ky1d = operators.k_axis(Ny, hy)
    kz1d = operators.k_axis(Nz, hz)
    Nxr = Nx // 2 + 1
    kxr1d = operators.k_axis_half(Nx, hx)
    # Full grid (for real-space noise generators, solver.random_smooth_field's
    # own KX/KY precedent): shape (Nz,Ny,Nx).
    KX, KY, KZ = operators.k_mesh(kx1d, ky1d, kz1d)
    # Half-spectrum (rfftn) companion grid, shape (Nz,Ny,Nxr): every engine
    # FFT below is real (same Hermitian-consistency argument as solver.py's
    # own WP4 §10b comment -- every k-space multiplier built on THIS grid is
    # even under the joint sign flip, so rfftn/irfftn on the half-spectrum
    # reproduces fftn/ifftn bit-for-bit). Layout: solvers/operators.py.
    KXr, KYr, KZr = operators.k_mesh(kxr1d, ky1d, kz1d)
    return dict(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, hx=hx, hy=hy, hz=hz,
                kx1d=kx1d, ky1d=ky1d, kz1d=kz1d, KX=KX, KY=KY, KZ=KZ,
                Volume=Lx * Ly * Lz,
                Nxr=Nxr, kxr1d=kxr1d, KXr=KXr, KYr=KYr, KZr=KZr)


_DEPOL_N_TOL = 1e-12   # SPEC § 11.1(3): symmetry and eigenvalue-range tolerance on N


def _depol_tensor_3d(depol_N):
    """SPEC § 11.1(1)/(3): depol_N (a scalar n meaning n*I, or a 3x3 tensor)
    -> the validated symmetric (3,3) float64 N, or ValueError naming SPEC § 11.
    Refused: any other shape; a non-finite entry; |N - N^T| > _DEPOL_N_TOL
    (within it, 0.5*(N + N^T) is used -- bit-identical for an exactly
    symmetric N); an eigenvalue outside [0, 1] by more than _DEPOL_N_TOL
    (coordinator ruling: every ellipsoid and every Ewald sphere in eps_m >= 0
    has its depolarization eigenvalues in [0, 1], and D(k != 0) is an
    orthogonal projector, so no stiffness bound anywhere has to move).
    The trailing + 0.0 turns a -0.0 entry into 0.0."""
    N = np.asarray(depol_N, dtype=float)
    if N.ndim != 0 and N.shape != (3, 3):
        raise ValueError(f"build_operators_3d: depol_N must be a scalar or a 3x3 tensor, "
                         f"got shape {N.shape} (SPEC § 11.1(1))")
    if not np.all(np.isfinite(N)):
        raise ValueError("build_operators_3d: depol_N has a non-finite entry (SPEC § 11.1(3))")
    if N.ndim == 0:
        N = float(N) * np.eye(3)
    asym = float(np.max(np.abs(N - N.T)))
    if asym > _DEPOL_N_TOL:
        raise ValueError(f"build_operators_3d: depol_N is not symmetric (max|N - N^T| = "
                         f"{asym:.3e} > {_DEPOL_N_TOL:.0e}) (SPEC § 11.1(3))")
    N = 0.5 * (N + N.T) + 0.0
    lam = np.linalg.eigvalsh(N)
    if lam[0] < -_DEPOL_N_TOL or lam[-1] > 1.0 + _DEPOL_N_TOL:
        raise ValueError(f"build_operators_3d: depol_N eigenvalues {lam.tolist()} are not all "
                         f"in [0, 1] (SPEC § 11.1(3))")
    return N


def build_operators_3d(grid, G11, G12, G44, G44p, kappa, kappa0_override=None,
                       spm_warm=False, spm_method="basic", spm_cg_max_iter=None,
                       depol_N=None):
    """A(k) (grad, full symmetric 3x3) and D(k) (3D transverse/longitudinal-
    projection depolarization) -- module docstring derivations.

    kappa (SPEC V2-03 B2, module docstring's own SPM section): a plain
    SCALAR (homogeneous background -- ops["dkappa"] is None, every path below
    and every stepping path is bit-identical to pre-V2-03) or a real-space
    (Nz,Ny,Nx) ARRAY kappa(x,y,z). Array path: kappa0 = the MEAN (SPEC B2's
    own wording; `kappa0_override` replaces it -- see the docstring's
    difference (1) for when that is not optional), dkappa = kappa - kappa0,
    spm=True. kappa0 is stored BOTH as "kappa0" (the explicit name SPEC B2
    asks the ops dict to carry) and as "kappa" (the key build_Minv_3d /
    build_Minv2_3d / solver._lin_matrix already read) -- the SAME float under
    two names, which is what "the implicit operator keeps kappa0" means here:
    those three functions needed no edit at all.

    EXACTNESS GUARD: an array whose entries are all equal takes kappa0 =
    that value rather than its computed mean (a mean over n equal floats is
    not bit-for-bit that float in general). Same discipline, same reason as
    film3d._kappa_faces' `==` guard; it is what makes K2 `kappa_const_3d`
    exactly 0.0 instead of merely small.

    spm_warm=True adds the ops-carried warm-start cache (docstring
    difference (2)); the default False keeps every solve a pure function of
    (P, ops).

    spm_method: "basic" (default -- the fixed point, ops dict unchanged) or
    "cg" (opt-in, array kappa only: _spm_iterate_3d runs solver._spm_pcg,
    the preconditioned CG on the same equations -- converges for any
    kappa(r) > 0 and ANY kappa0, so the thin-inclusion divergence of
    difference (1) cannot occur; docs/CG_SPM.md). The ops dict then carries
    spm_method="cg" and spm_cg_max_iter (CG cap when no max_iter is passed;
    None -> solver.SPM_CG_DEFAULT_MAX_ITER). spm_warm works with either
    method. Ignored for a scalar kappa.

    depol_N (SPEC § 11, module docstring's k = 0 paragraph): None (default)
    is the tin-foil D_ij(k=0) = 0, and every returned array and key is
    bit-identical to the pre-§ 11 dict. Otherwise a scalar n (n*I) or a
    symmetric 3x3 N with eigenvalues in [0, 1]; D_ij(k=0) = N_ij in the k = 0
    bin only, so E_d(k=0) = -N.<P>/kappa0 and the energy density gains
    (1/2)<P>.N.<P>/kappa0. Ewald mapping: N = eps_b/(eps_b + 2*eps_m)*I for
    replicas in a medium eps_m (eps_m -> infinity is tin foil, eps_m = 1 a
    vacuum sphere). The validated tensor is stored as ops["depol_N"], a key
    present ONLY when depol_N was given (0 included). Refused, ValueError
    naming SPEC § 11: depol_N with an array kappa (the SPM path), and every
    N _depol_tensor_3d rejects (non-symmetric, eigenvalue outside [0, 1])."""
    validate.build_operators_3d(grid, G11, G12, G44, G44p, kappa, kappa0_override)
    if spm_method not in solver.SPM_METHODS:
        raise ValueError(f"build_operators_3d: spm_method must be one of "
                         f"{solver.SPM_METHODS}, got {spm_method!r}")
    # A(k) and D(k) from the shared operator layer (solvers/operators.py):
    # every mixed off-diagonal (A_ij, D_ij, i != j) is odd in its two axes and
    # zeroed on each of their even-axis Nyquist planes; diagonals unmasked.
    A11, A22, A33, A12, A13, A23 = operators.grad_symbols_3d(grid, G11, G12, G44, G44p)
    D11, D22, D33, D12, D13, D23 = operators.depolarization_symbols_3d(grid)

    kappa_arr = np.asarray(kappa, dtype=float)
    if kappa_arr.ndim == 0:
        if kappa0_override is not None:
            raise ValueError("build_operators_3d: kappa0_override is meaningless for a "
                             "scalar kappa (there is no dkappa to reference)")
        kappa0 = float(kappa_arr)
        dkappa = None
    else:
        shape = (grid["Nz"], grid["Ny"], grid["Nx"])
        if kappa_arr.shape != shape:
            raise ValueError(f"build_operators_3d: kappa array shape {kappa_arr.shape} "
                             f"!= grid {shape}")
        if not np.all(kappa_arr > 0.0):
            raise ValueError("build_operators_3d: kappa(x,y,z) must be strictly positive")
        kmin, kmax = float(kappa_arr.min()), float(kappa_arr.max())
        if kappa0_override is not None:
            kappa0 = float(kappa0_override)
        elif kmin == kmax:
            kappa0 = kmin                       # exactness guard, see docstring
        else:
            kappa0 = float(kappa_arr.mean())    # SPEC V2-03 B2's default
        dkappa = kappa_arr - kappa0

    if depol_N is not None:
        if dkappa is not None:
            raise ValueError("build_operators_3d: depol_N with an array kappa is refused -- the "
                             "k = 0 condition of an inhomogeneous-kappa Poisson problem is not "
                             "defined by this convention (SPEC § 11.1(3))")
        N = _depol_tensor_3d(depol_N)
        k0 = (operators.k_squared(grid, 3) == 0.0)  # the single k = 0 bin; no Nyquist plane holds it
        D11[k0], D22[k0], D33[k0] = N[0, 0], N[1, 1], N[2, 2]
        D12[k0], D13[k0], D23[k0] = N[0, 1], N[0, 2], N[1, 2]

    ops = dict(A11=A11, A22=A22, A33=A33, A12=A12, A13=A13, A23=A23,
               D11=D11, D22=D22, D33=D33, D12=D12, D13=D13, D23=D23,
               G11=G11, G12=G12, G44=G44, G44p=G44p, kappa=kappa0,
               kappa0=kappa0, dkappa=dkappa, spm=dkappa is not None,
               spm_warm=bool(spm_warm), _spm_warm=None,
               Nx=grid["Nx"], Ny=grid["Ny"], Nz=grid["Nz"], Nxr=grid["Nxr"])
    if spm_method == "cg" and dkappa is not None:   # a basic ops dict gains no key
        ops.update(spm_method="cg", spm_cg_max_iter=(solver.SPM_CG_DEFAULT_MAX_ITER
                                                     if spm_cg_max_iter is None
                                                     else int(spm_cg_max_iter)))
    if depol_N is not None:
        ops["depol_N"] = N
    return ops


# Base-G gradient operator A(k) on the rfftn half-spectrum: the shared
# operators.grad_symbols_3d (the 3D mirror of solver._grad_symbols, one
# Nyquist rule for both); the old private name stays as its alias.
_grad_symbols_3d = operators.grad_symbols_3d


def _spectral_grad_xyz(field3d, grid, KX=None, KY=None, KZ=None):
    """(df/dx, df/dy, df/dz) of a single real (Nz,Ny,Nx) field, mirrors
    solver._spectral_grad_xy's ik-then-ifft pattern, one axis added. Optional
    KX/KY/KZ override the grid wavevectors (each zeroed on its own even-axis
    Nyquist plane) for the local gradient-density path
    (_modified_wavevectors_3d)."""
    s = (grid["Nz"], grid["Ny"], grid["Nx"])
    fk = _rfftn(field3d)
    KX = grid["KXr"] if KX is None else KX
    KY = grid["KYr"] if KY is None else KY
    KZ = grid["KZr"] if KZ is None else KZ
    dfdx = _irfftn(1j * KX * fk, s)
    dfdy = _irfftn(1j * KY * fk, s)
    dfdz = _irfftn(1j * KZ * fk, s)
    return dfdx, dfdy, dfdz


def grad_energy_density_3d(P, grid, ops):
    """f_grad, full 3D form, as the collocation (pointwise) local-gradient
    density plus a positive even-axis Nyquist correction -- the 3D mirror of
    solver.grad_energy_density's representation note.

    The pointwise 3D |grad P|^2 form, evaluated with REAL skew-adjoint first
    derivatives (each derivative axis zeroed on its own even-axis Nyquist
    plane, via the same _modified_wavevectors_3d the elastic repair uses),
    drops the alternating (Nyquist) mode on every even axis while
    grad_force_3d retains the k^2 eigenvalues. Each missing diagonal Nyquist
    term is added back as a positive quadratic in the real Nyquist-projected
    component: 0.5 * a_ij * (pi/h_j)^2 * (NyquistProject_j P_i)^2, with
    a_ii = G11 and a_ij = G44+G44p for i != j. On odd grids the correction
    is exactly zero and the ORIGINAL pointwise density is restored
    pointwise. The result is a collocation local-gradient density + Nyquist
    correction -- NOT 0.5*P.(A P) -- whose grid mean is the energy density
    consistent with grad_force_3d and whose pointwise values keep the local
    |grad P|^2 meaning. G is rebuilt from the scalar coefficients (never
    ops['A*'], which grad4's 3D hog path augments with H11), preserving the
    H11 separation in grad_energy_density_hog_3d."""
    G11, G12, G44, G44p = ops["G11"], ops["G12"], ops["G44"], ops["G44p"]
    KX, KY, KZ = operators.masked_wavevectors_3d(grid)
    P1x, P1y, P1z = _spectral_grad_xyz(P[0], grid, KX, KY, KZ)
    P2x, P2y, P2z = _spectral_grad_xyz(P[1], grid, KX, KY, KZ)
    P3x, P3y, P3z = _spectral_grad_xyz(P[2], grid, KX, KY, KZ)
    diag = 0.5 * G11 * (P1x**2 + P2y**2 + P3z**2)
    cross = G12 * (P1x * P2y + P2y * P3z + P1x * P3z)
    sym = 0.5 * G44 * ((P1y + P2x)**2 + (P2z + P3y)**2 + (P1z + P3x)**2)
    antisym = 0.5 * G44p * ((P1y - P2x)**2 + (P2z - P3y)**2 + (P1z - P3x)**2)
    fg = diag + cross + sym + antisym
    a = operators.grad_nyquist_coefficients(G11, G44, G44p, 3)
    return operators.add_nyquist_correction(fg, P, grid, a, 3)


def grad_force_3d(P, ops):
    """delta F_grad / delta P = A(k) . P, full symmetric 3x3 A(k)."""
    s = (ops["Nz"], ops["Ny"], ops["Nx"])
    Pk = _rfftn(P)
    F1 = ops["A11"] * Pk[0] + ops["A12"] * Pk[1] + ops["A13"] * Pk[2]
    F2 = ops["A12"] * Pk[0] + ops["A22"] * Pk[1] + ops["A23"] * Pk[2]
    F3 = ops["A13"] * Pk[0] + ops["A23"] * Pk[1] + ops["A33"] * Pk[2]
    Fk = np.stack([F1, F2, F3])
    return _irfftn(Fk, s)


def _elec_Ed_core_3d(source, ops):
    """-E_d,i = (1/kappa0) D_ij(k) source_j(k), full 3D transverse projection.
    The ORIGINAL elec_Ed_3d body, parameterized on an arbitrary real-space
    vector `source` instead of hardcoding P (solver._elec_Ed_core's own
    move, for the same reason): it is elec_Ed_3d's homogeneous fast path
    with source=P AND the elliptic solve _spm_iterate_3d repeats each round
    with source=P+dkappa*E."""
    s = (ops["Nz"], ops["Ny"], ops["Nx"])
    Sk = _rfftn(source)
    kappa = ops["kappa"]
    Ed1k = -(ops["D11"] * Sk[0] + ops["D12"] * Sk[1] + ops["D13"] * Sk[2]) / kappa
    Ed2k = -(ops["D12"] * Sk[0] + ops["D22"] * Sk[1] + ops["D23"] * Sk[2]) / kappa
    Ed3k = -(ops["D13"] * Sk[0] + ops["D23"] * Sk[1] + ops["D33"] * Sk[2]) / kappa
    Edk = np.stack([Ed1k, Ed2k, Ed3k])
    return _irfftn(Edk, s)


def _spm_iterate_3d(P, ops, tol=1e-10, max_iter=None, warn_iter=None,
                    record_history=False, warm=None):
    """SPEC V2-03 B2 -- the 3D mirror of solver._spm_iterate (WP5 SS11.1,
    Wang et al. 2013 SS2), rfftn instead of rfft2 and the full 3D projector.
    Poisson d_i[kappa(r) d_i phi] = d_i P_i with kappa = kappa0 + dkappa(r)
    is rearranged so every iterate solves the SAME homogeneous-kappa0
    problem (_elec_Ed_core_3d) with the extra source dkappa(r)*E^(n-1):
    d_i[kappa0 d_i phi] = d_i[P_i + dkappa*E_i]. Zero order (n=0) has no
    extra source, i.e. it IS the old closed form.

    Returns (Ed_converged, Ed_uniform, n_iter, resid[, history]) -- the
    n=0 term comes back alongside because the hybrid stepping term
    _explicit_N_3d needs exactly their difference, and paying for a second
    solve to recover it would be silly.

    E_ext is deliberately NOT part of this iteration, for the reason
    solver._spm_iterate's docstring derives at length (keeping the P -> E_d
    map exactly linear and self-adjoint is what makes f_elec = -1/2 P.E_d
    and dF_elec/dP = -E_d a consistent pair; folding E_ext into the dkappa
    source makes E_d affine in P and breaks that by a clean ~5e-4, which is
    what gate G8c caught in 2D). Convergence is on
    max|E^n - E^(n-1)|/max|E^n| <= tol, cap max_iter, warn past warn_iter --
    SPEC's own 1e-10 / 200 / 80, the same three numbers as the 2D.

    warm: optional starting field for E^(0) (module docstring difference
    (2)). None -- the default, and what every call makes unless the ops dict
    was built with spm_warm=True -- starts from Ed_uniform, exactly as the
    2D does.

    max_iter / warn_iter: None means SPEC's 200 / 80 (the basic defaults).
    ops built with build_operators_3d(..., spm_method="cg") run
    solver._spm_pcg instead (same contract, warm included; n = CG
    iterations, resid = CG relative residual, max_iter=None ->
    ops["spm_cg_max_iter"]; docs/CG_SPM.md)."""
    if ops.get("spm_method") == "cg":
        return solver._spm_pcg(P, ops["dkappa"], lambda src: _elec_Ed_core_3d(src, ops), 3,
                               tol=tol, max_iter=(max_iter if max_iter is not None
                                                  else ops["spm_cg_max_iter"]),
                               warn_iter=warn_iter, record_history=record_history,
                               warm=warm, where="bulk3d._spm_iterate_3d")
    max_iter = 200 if max_iter is None else max_iter
    warn_iter = 80 if warn_iter is None else warn_iter
    dkappa = ops["dkappa"]
    Ed_uniform = _elec_Ed_core_3d(P, ops)
    Ed = Ed_uniform
    E_prev = Ed_uniform if warm is None else warm
    history = [] if record_history else None
    resid = float("inf")
    n = 0
    for n in range(1, max_iter + 1):
        S = dkappa * E_prev
        Ed = _elec_Ed_core_3d(P + S, ops)
        E_cur = Ed
        denom = float(np.max(np.abs(E_cur)))
        num = float(np.max(np.abs(E_cur - E_prev)))
        resid = num / denom if denom > 0 else 0.0
        if record_history:
            history.append(resid)
        E_prev = E_cur
        if resid <= tol:
            break
    if n > warn_iter:
        print(f"[bulk3d._spm_iterate_3d] WARNING: SPM ran {n} iterations (> warn_iter={warn_iter}), "
              f"final resid={resid:.3e} (tol={tol:.0e}).")
    if resid > tol:
        print(f"[bulk3d._spm_iterate_3d] WARNING: SPM did NOT converge within max_iter={max_iter} "
              f"(final resid={resid:.3e} > tol={tol:.0e}).")
    if record_history:
        return Ed, Ed_uniform, n, resid, history
    return Ed, Ed_uniform, n, resid


def _spm_solve_3d(P, ops, tol=1e-10):
    """_spm_iterate_3d + the ops-carried warm-start cache (ONE place reads
    and writes ops["_spm_warm"], so elec_Ed_3d and _explicit_N_3d cannot
    drift apart). With spm_warm False -- the default -- this is exactly
    _spm_iterate_3d(P, ops, tol=tol)."""
    warm = ops["_spm_warm"] if ops.get("spm_warm") else None
    Ed, Ed_uniform, n, resid = _spm_iterate_3d(P, ops, tol=tol, warm=warm)
    if ops.get("spm_warm"):
        ops["_spm_warm"] = Ed
    return Ed, Ed_uniform, n, resid


def elec_Ed_3d(P, ops, E_ext=(0., 0., 0.), spm_tol=1e-10):
    """Depolarization field E_d -- a function of P ALONE in both regimes
    (the homogeneous closed form when ops["dkappa"] is None, bit-identical
    to pre-V2-03; the SPM fixed point when it is an array). E_ext is
    accepted for call-site uniformity with elec_energy_density_3d/
    elec_force_3d (which DO use it, as the separate un-redistributed
    "-P.E_ext" term) and is NOT used here -- see _spm_iterate_3d. spm_tol
    is SPEC's production 1e-10; FD-style consistency checks tighten it (K1
    `fd_3d_kappa_x` uses 1e-13, as its 2D twin does)."""
    if ops.get("dkappa") is None:
        return _elec_Ed_core_3d(P, ops)
    Ed, _Ed_uniform, _n, _resid = _spm_solve_3d(P, ops, tol=spm_tol)
    return Ed


# (3,)-vector -> (3,1,1,1,...); a field-shaped E_ext (already (3,Nz,Ny,Nx) or
# broadcastable) passes through unchanged for an unbatched P; with a batched
# P (3,S,Nz,Ny,Nx) the callers pass ensemble=True, which inserts S into a
# (3,Nz,Ny,Nx) field (shared by every member), keeps a (3,S,Nz,Ny,Nx) one
# and refuses any other shape. REUSED, not reimplemented: the shared
# operators.vec3_field is ndim-generic, so the extra z axis needs no 3D copy
# (SPEC S38.1 B3 = REVIEW_arch_minimality.md F5 removed the old duplicate;
# gates/g27b and g27d). The same object as solver._reshape_vec3.
_reshape_vec3_3d = operators.vec3_field


def elec_energy_density_3d(P, ops, E_ext, spm_tol=1e-10):
    """f_elec = -0.5*P.E_d - P.E_ext, real space, shape (Nz,Ny,Nx); 3D mirror
    of solver.elec_energy_density. E_d goes through the SPM iteration when
    ops carries dkappa."""
    Ed = elec_Ed_3d(P, ops, E_ext, spm_tol=spm_tol)
    Eext = _reshape_vec3_3d(E_ext, P.ndim, ensemble=P.ndim == 5)
    return -0.5 * (P[0] * Ed[0] + P[1] * Ed[1] + P[2] * Ed[2]) - (P[0] * Eext[0] + P[1] * Eext[1] + P[2] * Eext[2])


def elec_force_3d(P, ops, E_ext, spm_tol=1e-10):
    """delta F_elec/delta P_i = -E_d,i - E_ext,i, shape (3,Nz,Ny,Nx); 3D
    mirror of solver.elec_force. The 1/2 disappears for the same envelope
    reason in both regimes: the SPM map P -> E_d is linear and self-adjoint
    at convergence (it is the exact inverse of the elliptic operator
    d_i[kappa d_i .], which is self-adjoint for ANY positive kappa(r)), so
    the homogeneous-kappa identity carries over unchanged -- measured by K1
    `fd_3d_kappa_x`, exactly as gate G8c measured it in 2D."""
    Ed = elec_Ed_3d(P, ops, E_ext, spm_tol=spm_tol)
    Eext = _reshape_vec3_3d(E_ext, P.ndim, ensemble=P.ndim == 5)
    return -Ed - Eext


# --------------------------------------------------------------------------
# ELASTIC -- SPEC S37.3 (B2-E Stage E1): 3D Khachaturyan homogeneous-C
# elastic module. WHAT IS REUSED vs REBUILT, and why (module docstring's own
# discipline, applied to this section):
#   - elastic.build_C_tensor / elastic.contract_C: REUSED UNCHANGED. Both are
#     pure functions of (C11,C12,C44)/(C,tensor) with NO grid-shape
#     assumption anywhere (build_C_tensor never touches a grid at all;
#     contract_C's einsum pattern "ijkl,kl...->ij..." broadcasts over
#     WHATEVER trailing shape the tensor carries, (Ny,Nx) or (Nz,Ny,Nx) or
#     scalar alike) -- exactly the "pure elementwise... already works with
#     zero code changes" class this module's own top docstring already
#     established for solver.bulk_energy_density/bulk_force. Reusing these
#     also means C_ijkl <-> (c1111,c1122,c1212) has exactly ONE
#     implementation in the whole repo, not two that could drift.
#   - elastic.build_Omega / build_Omega_3d: SHARED. Both are now
#     operators.green_operator(grid, C, dim) -- dim=2 stacks K=(KX,KY,0)
#     (k3==0 structurally), dim=3 the full masked (KX,KY,KZ) -- one formula
#     Omega(K)=[C:KK]^-1 and one zero-K regularisation, so the Nz=1
#     reduction to elastic.py's numbers (gates/g30c_degenerate_nz1_elastic.py,
#     this WP's own G27a-style cross-engine anchor) holds by construction.
#   - elastic._green_delta_eps: REUSED (WP28 P1 phase, SPEC §38.1 B1). It was
#     REBUILT here until the only 2D-specific thing in it -- the hardcoded
#     call to elastic.py's module-level 2-axis _rfft2/_irfft2 -- became the
#     fwd/inv arguments; elastic_solve_3d passes this module's _rfftn/_irfftn
#     (and so keeps bulk3d.FFT_WORKERS, which is deliberately 1 where
#     elastic.FFT_WORKERS is -1). The formulas u_hat=-i*Omega.(K.tau_hat),
#     delta_eps_hat=sym(i*K*u_hat), delta_eps(k=0)=0 were already written
#     with "..." over the k-grid, so nothing else had to change; the deleted
#     copy was byte-identical line for line (REVIEW_arch_minimality.md F3),
#     and the same g30c Nz=1 anchor still holds.
#   - elastic.elastic_energy_density: REUSED UNCHANGED (0.5*einsum(sigma,e),
#     no grid-shape assumption either).
#   - elastic.eigenstrain / elastic.elastic_force_explicit: NOT reused, by
#     design -- see eigenstrain_3d's own docstring for the exact reason
#     (THE NAMED TRAP, SPEC S37.5): elastic.py's own Q11/Q12/Q44 parameters
#     are a DIFFERENT (Voigt-flavored) convention from this module's
#     Q1111/Q1122/Q1212 (true rank-4-tensor, all four cubic-symmetry-
#     equivalent slots of the shear component populated by ONE scalar,
#     Zhang 2007 Eq.(2)-(3)'s own printed form) -- Q44(elastic.py) =
#     4*Q1212(this module), a genuine unit-convention difference, not a
#     bug in either file. Feeding one straight into the other's formula
#     is EXACTLY the factor-of-4 slip SPEC names; eigenstrain_3d/
#     elastic_force_explicit_3d below are fresh, tensor-native
#     implementations (each with their own from-scratch sympy self-check,
#     mirroring elastic.py's own _verify_elastic_force_symbolic
#     discipline), so this module NEVER passes a number through
#     elastic.py's Q-shaped argument slots.
#
# elastic_params CONTRACT (the dict every public function below accepts):
#   None (default): elastic term omitted entirely, BYTE-IDENTICAL to every
#     pre-E1 call (SPEC S37: "elastic_params=None path must remain
#     BYTE-IDENTICAL").
#   Otherwise: THE DICT RETURNED BY setup_elastic_3d(...) below (mirrors
#     solver.py's own convention EXACTLY: solver.py's elastic_params is
#     always elastic.setup_elastic(...)'s output, never a raw spec the
#     stepping functions resolve themselves -- resolving Omega/K fresh on
#     EVERY step/energy/force call would repeat an O(grid) batched 3x3
#     inversion every step, which SPEC's own "1 elastic solve per step" cost
#     model, inherited from elastic.py, explicitly budgets for ONCE per
#     step, not once per Green's-operator use). That dict carries, among its
#     resolved machinery (Omega/K/zero_k/C_tensor/...), the RAW spec keys
#     SPEC S37.3 names verbatim so `elastic_params["C"]`/`["Q"]`/`["k0"]`
#     are always inspectable on the resolved object too:
#       "C": (c1111, c1122, c1212) -- deck's GPa-scale convention (e.g.
#         C11=275.0 means 2.75e11 N/m^2, units.si_to_deck_elastic's own
#         *1e-9 factor -- READ from how params/archive/bto_BC1984_6th.toml's C11=275.0
#         already matches Sluka's 2.75e11 Pa exactly, not assumed). ONE
#         homogeneous tensor for the WHOLE domain (SPEC: "Region-wise C(r)
#         NOT in scope: frozen effective C per the G14e precedent").
#       "Q": either a single (Q1111, Q1122, Q1212) triple (homogeneous Q) OR
#         a {label_value: (Q1111,Q1122,Q1212)} dict, consumed against the
#         `labels` array (shape (Nz,Ny,Nx), composite.make_label_field_
#         multi_3d's own convention) passed to setup_elastic_3d -- SPEC:
#         "Eigenstrain ... allowed to vary per region through Q(r) (a
#         constant-Q field is the special case)". TENSOR form always
#         (Q1111=Q2222=Q3333, Q1122=every normal-normal cross slot,
#         Q1212=Q1221=Q2112=Q2121=Q1313=...=every shear-shear slot -- cubic
#         symmetry, SAME 3-independent-constant structure elastic.py's own
#         C11/C12/C44 already has), NEVER Voigt Q11/Q12/Q44 -- see
#         eigenstrain_3d's docstring for the exact conversion an existing
#         Voigt-sourced deck (Sluka/Rabe) needs before landing here.
#       "k0": "stress_free" (default) | "clamped" -- SPEC S37.3's own two
#         named conventions for the homogeneous strain ebar (see
#         setup_elastic_3d's docstring for the precise definition of each).
#       "eps_star_static": None (default) | a (3,3,Nz,Ny,Nx) STATIC
#         eigenstrain field added to eps*(P) before the solve (SPEC V2-03
#         A2, elastic.py's own WP3 contract; P-derivative exactly zero, so
#         no force formula changes). None => byte-identical to pre-V2-03.
#       "z_mode" (SPEC §12, PLANE-STRESS-2D): key present ONLY when
#         setup_elastic_3d(..., z_mode="plane_stress") was asked for (Nz == 1
#         only); absent => generalized plane strain, byte-identical to pre-§12.
#         With it, "C11_ps"/"C12_ps" (the reduced in-plane stiffness) are
#         stored too and "Omega"/"green_pre" are built from (C11_ps, C12_ps,
#         C44) -- see setup_elastic_3d's docstring.
#     and, resolved at setup and documented here because elastic_solve_3d
#     reads them every step (WP28 T2, SPEC S39.1 A2'(ii)):
#       "C11","C12","C44": the VOIGT triple of the same cubic C -- numerically
#         identical to "C" above (C11=c1111, C12=c1122, C44=c1212 for cubic
#         symmetry), carried under the names elastic.contract_C_cubic's
#         signature uses so the call site reads as the formula. Derived AND
#         checked against contract_C(C_tensor, .) at setup by
#         _verify_voigt_mapping_3d -- the factor-of-2 shear convention is
#         exactly the trap SPEC S5 flags, so it is never assumed.
# --------------------------------------------------------------------------
# Real skew-adjoint first-derivative wavevectors (each component zeroed on its
# own even-axis Nyquist plane): the shared operators.masked_wavevectors_3d,
# the 3D mirror of elastic._modified_wavevectors_2d; old name kept as alias.
_modified_wavevectors_3d = operators.masked_wavevectors_3d


def build_Omega_3d(grid, C):
    """Omega(k) = [C_ijkl k_j k_l]^-1 on the rfftn half-spectrum grid, full
    3-axis K = masked (KXr,KYr,KZr): returns (Omega (Nz,Ny,Nxr,3,3),
    K (3,Nz,Ny,Nxr), zero_k). The shared operators.green_operator(grid, C, 3)
    -- the function elastic.build_Omega calls with dim=2 (K=(KXr,KYr,0)), so
    the two engines' Green operators cannot drift and the Nz=1 reduction to
    elastic.py's numbers (gate g30c) holds by construction. Every zero-K mode
    (DC plus each intersection of even-axis Nyquist planes) is regularised
    before the inverse and zeroed after."""
    return operators.green_operator(grid, C, 3)


def eigenstrain_3d(P, Q1111, Q1122, Q1212):
    """eps0_ij(r) = Q_ijkl(r) P_k(r) P_l(r), FULL TENSOR FORM (SPEC S37.5's
    named-trap amendment, Zhang 2007 Eq.(2)-(3)): Q1212 populates ALL FOUR
    cubic-symmetry-equivalent shear slots (Q1212=Q1221=Q2112=Q2121), so the
    genuinely-full k,l=1..3 sum gives eps0_12 = Q_1212*P1*P2 + Q_1221*P2*P1 =
    2*Q1212*P1*P2 (the SAME "sum over both off-diagonal slots, factor of 2
    falls out of the einsum" mechanism elastic.py's own contract_C already
    relies on for STRESS, applied here to the EIGENSTRAIN side instead).

    Q1111/Q1122/Q1212 may be scalars OR (Nz,Ny,Nx)-broadcastable arrays (the
    region-wise-Q path, elastic.py's own contract_C_cubic precedent: "exact,
    not an approximation, for any broadcastable shape").

    THE NAMED TRAP (SPEC S37.5): this is Q1212 = Q_1212 as printed in
    Zhang's own tensor notation -- NEVER elastic.py's Q44 (its eigenstrain,
    e[0,1] = 0.5*Q44*P1*P2, is a DIFFERENT convention; equating the two
    algebraically: 2*Q1212*P1*P2 == 0.5*Q44*P1*P2 iff Q44 = 4*Q1212). A
    Voigt-sourced deck (Sluka 2012 BTO, Rabe App A STO -- both printed as
    Q11/Q12/Q44 exactly like elastic.py's own convention, per params/
    bto_pan2019.toml's/sto_pan2019.toml's own [electrostriction] comment)
    therefore needs Q1212 = Q44_voigt/4 (Q1111=Q11_voigt, Q1122=Q12_voigt
    unchanged -- the normal-normal slots carry NO such factor, only the
    shear-shear slot does; see this module's own g30b/report for the
    from-Zhang-numbers-directly worked example that catches a factor slip
    here analytically)."""
    if elastic._CPU_KERNELS is not None:
        return elastic._CPU_KERNELS.eigenstrain(P, Q1111, Q1122, 4 * Q1212)
    P1, P2, P3 = P[0], P[1], P[2]
    e = np.zeros((3, 3) + P.shape[1:])
    e[0, 0] = Q1111 * P1**2 + Q1122 * (P2**2 + P3**2)
    e[1, 1] = Q1111 * P2**2 + Q1122 * (P1**2 + P3**2)
    e[2, 2] = Q1111 * P3**2 + Q1122 * (P1**2 + P2**2)
    e[0, 1] = e[1, 0] = 2.0 * Q1212 * P1 * P2
    e[0, 2] = e[2, 0] = 2.0 * Q1212 * P1 * P3
    e[1, 2] = e[2, 1] = 2.0 * Q1212 * P2 * P3
    return e


def elastic_force_explicit_3d(P, sigma, Q1111, Q1122, Q1212):
    """delta F_el / delta P_m = -sigma_ij * d(eps0_ij)/dP_m, full tensor sum --
    derived fresh from eigenstrain_3d's own tensor-form eps0 (module
    docstring: NOT elastic.elastic_force_explicit, different Q convention).
    Verified against an independent from-scratch sympy derivation at import
    time (_verify_elastic_force_symbolic_3d below), mirroring elastic.py's
    own "generate via sympy and check both agree" discipline."""
    if elastic._CPU_KERNELS is not None:
        return elastic._CPU_KERNELS.elastic_force(P, sigma, Q1111, Q1122, 4 * Q1212)
    P1, P2, P3 = P[0], P[1], P[2]
    s11, s22, s33 = sigma[0, 0], sigma[1, 1], sigma[2, 2]
    s12, s13, s23 = sigma[0, 1], sigma[0, 2], sigma[1, 2]
    f1 = -(2 * P1 * (s11 * Q1111 + (s22 + s33) * Q1122) + 4 * Q1212 * (s12 * P2 + s13 * P3))
    f2 = -(2 * P2 * (s22 * Q1111 + (s11 + s33) * Q1122) + 4 * Q1212 * (s12 * P1 + s23 * P3))
    f3 = -(2 * P3 * (s33 * Q1111 + (s11 + s22) * Q1122) + 4 * Q1212 * (s13 * P1 + s23 * P2))
    return np.stack([f1, f2, f3])


def _verify_elastic_force_symbolic_3d():
    """Import-time self-check (elastic.py's own _verify_elastic_force_symbolic
    precedent): eigenstrain_3d's FULL k,l=1..3 tensor sum, differentiated
    symbolically, must match elastic_force_explicit_3d's closed form exactly."""
    import sympy as sp
    P1, P2, P3, Q1111, Q1122, Q1212 = sp.symbols("P1 P2 P3 Q1111 Q1122 Q1212", real=True)
    s11, s22, s33, s12, s13, s23 = sp.symbols("s11 s22 s33 s12 s13 s23", real=True)
    Pv = [P1, P2, P3]

    es = {
        (0, 0): Q1111 * P1**2 + Q1122 * (P2**2 + P3**2),
        (1, 1): Q1111 * P2**2 + Q1122 * (P1**2 + P3**2),
        (2, 2): Q1111 * P3**2 + Q1122 * (P1**2 + P2**2),
        (0, 1): 2 * Q1212 * P1 * P2,
        (0, 2): 2 * Q1212 * P1 * P3,
        (1, 2): 2 * Q1212 * P2 * P3,
    }
    es[(1, 0)], es[(2, 0)], es[(2, 1)] = es[(0, 1)], es[(0, 2)], es[(1, 2)]
    sg = {(0, 0): s11, (1, 1): s22, (2, 2): s33,
          (0, 1): s12, (1, 0): s12, (0, 2): s13, (2, 0): s13, (1, 2): s23, (2, 1): s23}

    explicit = [
        -(2 * P1 * (s11 * Q1111 + (s22 + s33) * Q1122) + 4 * Q1212 * (s12 * P2 + s13 * P3)),
        -(2 * P2 * (s22 * Q1111 + (s11 + s33) * Q1122) + 4 * Q1212 * (s12 * P1 + s23 * P3)),
        -(2 * P3 * (s33 * Q1111 + (s11 + s22) * Q1122) + 4 * Q1212 * (s13 * P1 + s23 * P2)),
    ]
    for m in range(3):
        generic = -sum(sp.diff(es[(i, j)], Pv[m]) * sg[(i, j)] for i in range(3) for j in range(3))
        if sp.simplify(generic - explicit[m]) != 0:
            raise AssertionError(f"bulk3d elastic driving force mismatch at component {m}: "
                                  f"generic={generic} vs explicit={explicit[m]}")


_verify_elastic_force_symbolic_3d()


def _resolve_Q_field_3d(Q, labels, shape):
    """Q: a single (Q1111,Q1122,Q1212) triple (returned as-is, scalars
    broadcast fine) OR a {label_value: (Q1111,Q1122,Q1212)} dict, resolved
    against `labels` (int array, shape==`shape`, composite.py's own
    label-field convention) into three (possibly-array) fields via
    np.where-chained lookup -- mirrors composite.alpha_field_multi's own
    "WHOLE-CELL assignment... exact at EVERY cell by construction" pattern.
    Every label value present in `labels` must have a dict entry: an
    unlisted one used to keep the zeros the chain starts from, i.e. a
    silently Q=0 (non-electrostrictive) region. Raises ValueError naming
    the missing labels instead."""
    if not isinstance(Q, dict):
        Q1111, Q1122, Q1212 = Q
        return Q1111, Q1122, Q1212
    if labels is None:
        raise ValueError("setup_elastic_3d: Q given as a per-region dict but no `labels` array supplied")
    if labels.shape != shape:
        raise ValueError(f"setup_elastic_3d: labels.shape {labels.shape} != grid shape {shape}")
    Q1111 = np.zeros(shape)
    Q1122 = np.zeros(shape)
    Q1212 = np.zeros(shape)
    covered = np.zeros(shape, dtype=bool)
    for label_value, (q1111, q1122, q1212) in Q.items():
        mask = (labels == label_value)
        covered |= mask
        Q1111 = np.where(mask, q1111, Q1111)
        Q1122 = np.where(mask, q1122, Q1122)
        Q1212 = np.where(mask, q1212, Q1212)
    if not covered.all():
        raise ValueError(f"setup_elastic_3d: labels {np.unique(labels[~covered]).tolist()} occur "
                         f"in `labels` but have no entry in the per-label Q dict (keys "
                         f"{list(Q)}); every region needs its own Q triple")
    return Q1111, Q1122, Q1212


def _verify_voigt_mapping_3d(C11, C12, C44, C_tensor, tol=1e-12):
    """Per-deck check (cheap, once per setup_elastic_3d call, WP28 T2): the
    Voigt triple handed to elastic.contract_C_cubic must reproduce the
    full-tensor elastic.contract_C(C_tensor, .) this module used before, on a
    random SYMMETRIC strain. Checks THIS deck's actual numbers, which
    elastic.py's own import-time _verify_contract_C_cubic (fixed made-up
    constants) cannot do -- so a deck whose C triple is not cubic-consistent
    is caught here rather than silently contracted with the wrong shear
    factor. Raises AssertionError; never silently degrades."""
    rng = np.random.default_rng(20260903)
    e = rng.standard_normal((3, 3, 4, 5))
    e = 0.5 * (e + np.swapaxes(e, 0, 1))
    s_cubic = elastic.contract_C_cubic(C11, C12, C44, e)
    s_tensor = elastic.contract_C(C_tensor, e)
    if not np.allclose(s_cubic, s_tensor, rtol=tol, atol=tol):
        raise AssertionError(
            "setup_elastic_3d: the cubic Voigt triple (C11,C12,C44)="
            f"({C11},{C12},{C44}) does not reproduce contract_C(C_tensor, e) "
            f"(max|d|={float(np.max(np.abs(s_cubic - s_tensor))):.3e})")


def setup_elastic_3d(grid, C, Q, k0="stress_free", labels=None, eps_star_static=None,
                     z_mode="generalized_plane_strain"):
    """Build the resolved elastic_params dict every public function in this
    module accepts (see the section docstring's own "elastic_params CONTRACT"
    for the full field list). C=(c1111,c1122,c1212); Q=triple or per-label
    dict (see _resolve_Q_field_3d); k0="stress_free" (SPEC S37.3 default,
    the mean strain that minimizes elastic energy at zero applied stress --
    ebar = <eps0>, elastic.py's own mode (a), reduces to it bit-for-bit at
    Nz=1, gates/g30c) | "clamped" (SPEC's "zero-mean-strain" variant --
    ebar = 0 identically, a Dirichlet constraint on the cell's overall
    homogeneous shape, generally leaving a nonzero mean internal stress;
    NOT an elastic.py mode -- new to this module).

    eps_star_static (SPEC V2-03 A2, elastic.py's own WP3 contract): optional
    STATIC (3,3,Nz,Ny,Nx) eigenstrain field -- defect/disorder strain that
    does not come from P -- added to eps*(P) BEFORE the mechanical solve
    (elastic_solve_3d), exactly as elastic.setup_elastic's own
    eps_star_static is added by elastic.elastic_solve. Its P-derivative is
    exactly zero, so EVERY force formula in this module is untouched
    (elastic_force_explicit_3d still contracts sigma with d eps*(P)/dP
    only). None (the default) is BYTE-IDENTICAL to every pre-V2-03 call: the
    addition is not executed at all on that path (gates/core K2
    `eps_star_static_none_3d`), and the Nz=1 static path reduces to
    elastic.py's 2D static path bit-for-bit (K2 `bulk3d_nz1_vs_2d_elastic`,
    static leg).

    z_mode (SPEC §12, PLANE-STRESS-2D): "generalized_plane_strain" (default)
    is the 3D Green's solve as it always was; at Nz == 1 (z-invariant
    columns) the KZ = 0 solve gives no local eps_33, so the total eps_33 is
    the box-uniform ebar_33 and any variation of eps*_33 costs energy.
    "plane_stress" (Nz == 1 only) is a thin plate free to change its
    thickness cell by cell: sigma_33 = sigma_13 = sigma_23 = 0 in EVERY cell.
    For cubic C the elastic strains then obey e_33 = -(C12/C11)(e_11 + e_22),
    e_13 = e_23 = 0 (so the total eps_i3 follows eps*_i3 locally), and the
    in-plane problem d_a sigma_ab = 0 (a, b in {1, 2}) is solved with the
    reduced stiffness C11_ps = C11 - C12^2/C11, C12_ps = C12 - C12^2/C11,
    C66 = C44 and only eps*_11, eps*_22, eps*_12 (all three P components
    enter them). k0 acts on the in-plane homogeneous strain only
    (stress_free: <eps*_ab>; clamped: 0). Electrostatics are untouched.
    Implementation: Omega is built from the cubic tensor (C11_ps, C12_ps,
    C44); at KZ = 0 its 3-row decouples, and with the i3 eigenstrain
    components zeroed u_3 = 0, so the existing Green apply solves exactly the
    in-plane problem; elastic_solve_3d then sets e_i3 as above and contracts
    sigma with the FULL C (sigma_33 = 0 to round-off, sigma_13 = sigma_23 =
    0 exactly). elastic_force_explicit_3d and elastic_energy_density_3d are
    unchanged and exact for this sigma (the i3 terms vanish). Refused:
    Nz != 1, eps_star_static (here or added later by a shallow copy)."""
    validate.setup_elastic_3d(grid, C, Q, labels, eps_star_static)
    if k0 not in ("stress_free", "clamped"):
        raise ValueError(f"setup_elastic_3d: k0 must be 'stress_free' or 'clamped', got {k0!r}")
    if z_mode not in ("generalized_plane_strain", "plane_stress"):
        raise ValueError("setup_elastic_3d: z_mode must be 'generalized_plane_strain' or "
                         f"'plane_stress' (SPEC §12), got {z_mode!r}")
    if z_mode == "plane_stress" and grid["Nz"] != 1:
        raise ValueError(f"setup_elastic_3d: z_mode='plane_stress' needs Nz == 1 (SPEC §12), "
                         f"got Nz = {grid['Nz']}")
    if z_mode == "plane_stress" and eps_star_static is not None:
        raise ValueError("setup_elastic_3d: z_mode='plane_stress' does not support "
                         "eps_star_static (SPEC §12)")
    c1111, c1122, c1212 = C
    C_tensor = elastic.build_C_tensor(c1111, c1122, c1212)  # REUSED, see section docstring
    # WP28 T2 (SPEC §40.3, §39.1 A2'(ii)): the VOIGT triple elastic.
    # contract_C_cubic wants. For CUBIC symmetry it is the SAME THREE NUMBERS
    # as this module's TENSOR triple, only relabelled: build_C_tensor puts
    # c1212 into C_1212, and sigma_12 = 2*C_1212*e12 = C44*gamma12 with the
    # engineering shear gamma12 = 2*e12, so C44 = c1212 exactly (likewise
    # C11 = c1111, C12 = c1122). Because that is a factor-of-2 shear claim --
    # the class of bug SPEC §5 names -- it is CHECKED, not asserted:
    C11, C12, C44 = c1111, c1122, c1212
    _verify_voigt_mapping_3d(C11, C12, C44, C_tensor)
    C_green = C_tensor
    if z_mode == "plane_stress":
        # SPEC §12: the in-plane Green's operator is that of the reduced stiffness.
        C11_ps, C12_ps = C11 - C12**2 / C11, C12 - C12**2 / C11
        C_green = elastic.build_C_tensor(C11_ps, C12_ps, C44)
    Omega, K, zero_k = build_Omega_3d(grid, C_green)
    shape = (grid["Nz"], grid["Ny"], grid["Nx"])
    Q1111, Q1122, Q1212 = _resolve_Q_field_3d(Q, labels, shape)
    if eps_star_static is not None and np.shape(eps_star_static) != (3, 3) + shape:
        raise ValueError(f"setup_elastic_3d: eps_star_static must have shape "
                          f"{(3, 3) + shape}, got {np.shape(eps_star_static)}")
    # green_pre: the complex K/Omega copies the Green apply multiplies by,
    # built once here (WP28 P6, SPEC §39.1 A2'(i)) -- elastic.green_precompute.
    el = dict(C=C, Q=Q, k0=k0, labels=labels,
              C_tensor=C_tensor, C11=C11, C12=C12, C44=C44,
              Omega=Omega, K=K, zero_k=zero_k,
              green_pre=elastic.green_precompute(K, Omega),
              Q1111=Q1111, Q1122=Q1122, Q1212=Q1212,
              eps_star_static=eps_star_static,
              Nx=grid["Nx"], Ny=grid["Ny"], Nz=grid["Nz"])
    if z_mode == "plane_stress":
        el.update(z_mode=z_mode, C11_ps=C11_ps, C12_ps=C12_ps)
    return el


def elastic_solve_3d(P, el):
    """Returns dict(eps_star, ebar, delta_eps, e, sigma) -- SAME keys/roles as
    elastic.elastic_solve's homogeneous-C path, minus that function's
    diagnostic eps_star_P key (this module returns the TOTAL eigenstrain
    only; the P-dependent part is eigenstrain_3d(P, ...) itself).

    SPEC V2-03 A2: el["eps_star_static"], when not None, is added to eps*(P)
    right here -- BEFORE tau, the Green's solve and ebar -- exactly as
    elastic.elastic_solve adds its own (`static = el.get("eps_star_static")`).
    A static eigenstrain has ZERO P-derivative, so elastic_force_3d /
    elastic_force_explicit_3d are untouched by it. el built without the
    kwarg (or with None) never executes the addition: byte-identical.

    WP28 T2 (SPEC S40.3, S39.1 A2'(ii)): the two contractions below were
    elastic.contract_C(el["C_tensor"], .), i.e. np.einsum("ijkl,kl...->ij...")
    over a full 81-component rank-4 tensor whose 3 independent numbers are all
    this cubic deck has. They are now elastic.contract_C_cubic(C11,C12,C44, .),
    the 15-operation cubic form (elastic.py itself is UNTOUCHED -- the function
    is reused, not rewritten). The Voigt triple is derived and CHECKED against
    the full-tensor path once per deck in setup_elastic_3d
    (_verify_voigt_mapping_3d); C_tensor stays in the dict and is still what
    build_Omega_3d uses.
    EQUIVALENCE CLASS: TOLERANCE, not byte, and no byte-class alternative
    exists -- LC C2 measured that even a hand-summed contraction in einsum's
    own k,l order differs from np.einsum at 1.0e-12, so einsum's 81-term
    accumulation is not the naive sequential order
    (REVIEW_wp27_efficiency_measured.md S3 C2). Measured on random P:
    1.3e-10 relerr on sigma, 1.6e-11 on the elastic driving force; 4.5e-11 /
    3e-12 on warm states.
    GATES: G30a-e (+G27a and G30c, the Nz=1 degeneracy that must still reduce
    to elastic.py's 2D path bit-for-bit).

    SPEC V2-03 C1: dispatches to solvers/bulk3d_spm.py when `el` was built by
    bulk3d_spm.setup_elastic_spm_3d (el["spm"]=True, inhomogeneous full-tensor
    C(r)) -- elastic.elastic_solve's own WP10 pattern. An el built by
    setup_elastic_3d has no "spm" key, so every OTHER caller is byte-identical
    to pre-V2-03 (gates/core K2 `elastic_spm_off_byte_3d`), and elastic_force_3d
    /total_energy_3d/total_force_3d/_explicit_N_3d need no change at all: they
    only ever reach the elastic solve through this function."""
    if el.get("spm"):
        from fftpf import pf3d_elastic_inhomogeneous as bulk3d_spm       # lazy: bulk3d_spm imports THIS module
        return bulk3d_spm.elastic_solve_spm_3d(P, el)
    eps_star = eigenstrain_3d(P, el["Q1111"], el["Q1122"], el["Q1212"])  # (3,3,Nz,Ny,Nx)
    static = el.get("eps_star_static")
    if el.get("z_mode") == "plane_stress":
        return _elastic_solve_plane_stress_3d(eps_star, static, el)
    if static is not None:
        if P.ndim == 5:                         # V2-03 C3 ensemble: S axis on the (3,3,Nz,Ny,Nx) static
            static = operators.ensemble_field(static, 2, eps_star.ndim, "eps_star_static")
        eps_star = eps_star + static            # V2-03 A2, elastic.py's WP3 contract
    shape = (el["Nz"], el["Ny"], el["Nx"])
    tau = elastic.contract_C_cubic(el["C11"], el["C12"], el["C44"], eps_star)  # REUSED
    # REUSED (SPEC §38.1 B1): elastic._green_delta_eps is axis-count-generic
    # once the transform pair is an argument -- pass THIS module's 3-axis
    # pair (at bulk3d.FFT_WORKERS, deliberately 1, not elastic's -1).
    delta_eps = elastic._green_delta_eps(tau, el["K"], el["Omega"], el["zero_k"], shape,
                                         fwd=_rfftn, inv=_irfftn, pre=el["green_pre"])

    if el["k0"] == "clamped":
        ebar = np.zeros((3, 3))
    else:
        ebar = eps_star.mean(axis=(-3, -2, -1))  # stress_free: ebar_ij = <eps0_ij>

    e = ebar[..., None, None, None] + delta_eps - eps_star
    sigma = elastic.contract_C_cubic(el["C11"], el["C12"], el["C44"], e)  # REUSED
    return dict(eps_star=eps_star, ebar=ebar, delta_eps=delta_eps, e=e, sigma=sigma)


def _elastic_solve_plane_stress_3d(eps_star, static, el):
    """SPEC §12 (z_mode="plane_stress", Nz == 1): per-cell sigma_i3 = 0.
    The in-plane problem is the existing Green apply on el["Omega"] (built
    from C11_ps, C12_ps, C44) with the i3 eigenstrain components zeroed, so
    u_3 = 0 and only eps*_ab (a, b in {1, 2}) is relaxed; then e_13 = e_23 =
    0, e_33 = -(C12/C11)(e_11 + e_22), and sigma = C:e with the full C.
    Same keys as elastic_solve_3d: eps_star is the full 3D eigenstrain,
    delta_eps the in-plane compatible fluctuation, and ebar the MEAN TOTAL
    strain <e + eps*> (as in the default mode): its in-plane block is the k0
    homogeneous strain, its i3 entries the box means of the locally free
    total eps_i3."""
    if static is not None:
        raise ValueError("elastic_solve_3d: z_mode='plane_stress' does not support "
                         "eps_star_static (SPEC §12)")
    eps_in = eps_star.copy()
    eps_in[2, :] = 0.0
    eps_in[:, 2] = 0.0
    shape = (el["Nz"], el["Ny"], el["Nx"])
    tau = elastic.contract_C_cubic(el["C11_ps"], el["C12_ps"], el["C44"], eps_in)
    delta_eps = elastic._green_delta_eps(tau, el["K"], el["Omega"], el["zero_k"], shape,
                                         fwd=_rfftn, inv=_irfftn, pre=el["green_pre"])
    if el["k0"] == "clamped":
        ebar = np.zeros((3, 3))
    else:
        ebar = eps_in.mean(axis=(-3, -2, -1))
    e = ebar[..., None, None, None] + delta_eps - eps_in
    e[2, :] = 0.0
    e[:, 2] = 0.0
    e[2, 2] = -(el["C12"] / el["C11"]) * (e[0, 0] + e[1, 1])
    sigma = elastic.contract_C_cubic(el["C11"], el["C12"], el["C44"], e)
    ebar = ebar.copy()
    ebar[2, :] = ebar[:, 2] = (e[2] + eps_star[2]).mean(axis=(-3, -2, -1))
    return dict(eps_star=eps_star, ebar=ebar, delta_eps=delta_eps, e=e, sigma=sigma)


elastic_energy_density_3d = elastic.elastic_energy_density  # REUSED UNCHANGED, see section docstring


def elastic_force_3d(P, el):
    """delta F_el / delta P (re-solves u at the given P, "1 elastic solve per
    step" -- SPEC S2/S37.3). Returns (force, sol)."""
    sol = elastic_solve_3d(P, el)
    force = elastic_force_explicit_3d(P, sol["sigma"], el["Q1111"], el["Q1122"], el["Q1212"])
    return force, sol


# --------------------------------------------------------------------------
# Landau (bulk) energy/force: REUSED, not reimplemented -- solver.
# bulk_energy_density/bulk_force are pure elementwise functions of
# P[0],P[1],P[2], no grid-shape assumption (verified in gates/g27b).
# --------------------------------------------------------------------------
bulk_energy_density = solver.bulk_energy_density
bulk_force = solver.bulk_force


def total_energy_3d(P, alpha, ops, E_ext, grid, elastic_params=None, spm_tol=1e-10):
    """Mirrors solver.total_energy's (dens, tot) contract. elastic_params=None
    (default) omits the f_el term entirely, BYTE-IDENTICAL to every pre-E1
    call (SPEC S37); otherwise elastic_params must be setup_elastic_3d(...)'s
    own output (see that section's docstring).

    SPEC V2-03 C3 (ensemble axis): an UNBATCHED P (3,Nz,Ny,Nx) takes the
    `float(f.mean())` lines below, unchanged and byte-identical. A BATCHED P
    (3,S,Nz,Ny,Nx) averages over the three SPACE axes only, so every value in
    both dicts is an (S,) array of PER-MEMBER numbers instead of one float --
    a collapsed float would be the ensemble mean of S different states, which
    is not an energy of anything. See the ensemble section below."""
    fb = bulk_energy_density(P, alpha)
    fg = grad_energy_density_3d(P, grid, ops)
    fe = elec_energy_density_3d(P, ops, E_ext, spm_tol=spm_tol)
    if P.ndim == 5:
        _ensemble_reject(elastic_params)
        sp = (-3, -2, -1)
        dens = dict(bulk=fb.mean(axis=sp), grad=fg.mean(axis=sp), elec=fe.mean(axis=sp))
        if elastic_params is not None:
            sol = elastic_solve_3d(P, elastic_params)
            dens["elastic"] = elastic_energy_density_3d(sol).mean(axis=sp)
        # Per member, the builtin sum over exact Python floats -- the SAME
        # arithmetic as the unbatched line below. CPython >= 3.12 sums exact
        # floats with Neumaier compensation but (S,) arrays with plain '+', so
        # summing the arrays differs in the last bit (K2.ensemble_s1_byte_3d).
        dens["total"] = np.array([sum(float(v[s]) for v in dens.values())
                                  for s in range(P.shape[1])])
    else:
        dens = dict(bulk=float(fb.mean()), grad=float(fg.mean()), elec=float(fe.mean()))
        if elastic_params is not None:
            sol = elastic_solve_3d(P, elastic_params)
            dens["elastic"] = float(elastic_energy_density_3d(sol).mean())
        dens["total"] = sum(dens.values())
    tot = {k: v * grid["Volume"] for k, v in dens.items()}
    return dens, tot


def total_force_3d(P, alpha, ops, E_ext, elastic_params=None, spm_tol=1e-10):
    """elastic_params=None (default): returns (Fb,Fg,Fe,sum), BYTE-IDENTICAL
    to every pre-E1 call. Otherwise (Fb,Fg,Fe,Fel,sum) -- mirrors solver.
    total_force's own return-arity convention exactly."""
    Fb = bulk_force(P, alpha)
    Fg = grad_force_3d(P, ops)
    Fe = elec_force_3d(P, ops, E_ext, spm_tol=spm_tol)
    if elastic_params is None:
        return Fb, Fg, Fe, Fb + Fg + Fe
    Fel, _ = elastic_force_3d(P, elastic_params)
    return Fb, Fg, Fe, Fel, Fb + Fg + Fe + Fel


# --------------------------------------------------------------------------
# Time stepping -- first-order semi-implicit (Chen & Shen), SPEC §2's split,
# rebuilt at the 3D k-space level (module docstring: WHY this can't just
# call solver.build_Minv/step directly).
# --------------------------------------------------------------------------
def build_Minv_3d(grid, ops, dt, L, stab=0.0):
    """M(k) = I3 + dt*L*[A(k)+D(k)/kappa], full 3x3 (off-diagonals A13/A23/
    D13/D23 now generally nonzero, unlike solver.build_Minv's 2D M which
    never populates the (0,2)/(1,2) slots at all). Shape (Nz,Ny,Nxr,3,3).
    stab=S > 0 adds dt*L*S*I3 (solver.build_Minv's stab contract; step_3d
    must get the same S). stab=0 is the original code path, bit for bit."""
    Nz, Ny, Nxr = grid["Nz"], grid["Ny"], grid["Nxr"]
    M = np.zeros((Nz, Ny, Nxr, 3, 3))
    M[..., 0, 0] = 1.0 + dt * L * (ops["A11"] + ops["D11"] / ops["kappa"])
    M[..., 1, 1] = 1.0 + dt * L * (ops["A22"] + ops["D22"] / ops["kappa"])
    M[..., 2, 2] = 1.0 + dt * L * (ops["A33"] + ops["D33"] / ops["kappa"])
    off12 = dt * L * (ops["A12"] + ops["D12"] / ops["kappa"])
    off13 = dt * L * (ops["A13"] + ops["D13"] / ops["kappa"])
    off23 = dt * L * (ops["A23"] + ops["D23"] / ops["kappa"])
    M[..., 0, 1] = M[..., 1, 0] = off12
    M[..., 0, 2] = M[..., 2, 0] = off13
    M[..., 1, 2] = M[..., 2, 1] = off23
    if stab:
        from fftpf import timestep_adaptive as adaptive
        stab = adaptive.check_stab(stab)
        for i in range(3):
            M[..., i, i] += dt * L * stab
        return adaptive.tag_operator(np.linalg.inv(M), stab)
    return np.linalg.inv(M)


def _explicit_N_3d(P, alpha, ops, E_ext, elastic_params):
    """elastic_params=None (default): BYTE-IDENTICAL to every pre-E1 call
    (SPEC S37). Otherwise adds the elastic driving force, explicit (SPEC S2:
    "the elastic driving force is explicit, not part of the M(k) operator"),
    mirrors solver._explicit_N's own elastic branch exactly.

    SPEC V2-03 B2 hybrid stepping, mirroring solver._explicit_N's own WP5
    branch: M(k) (build_Minv_3d/build_Minv2_3d) keeps the HOMOGENEOUS
    D(k)/kappa0 implicit operator unchanged, so the big-dt property survives
    a kappa(r) field; the difference between the true kappa(r) force and
    what M(k) implicitly assumes, dE = Ed_spm(P) - Ed_uniform(P), is added
    here EXPLICITLY. ops["dkappa"] is None (scalar kappa, every pre-V2-03
    caller) skips the branch, so N is bit-identical to before."""
    N = bulk_force(P, alpha) - _reshape_vec3_3d(E_ext, P.ndim, ensemble=P.ndim == 5)
    if ops.get("dkappa") is not None:
        Ed, Ed_uniform, _n, _resid = _spm_solve_3d(P, ops)
        N = N - (Ed - Ed_uniform)
    if elastic_params is not None:
        Fel, _ = elastic_force_3d(P, elastic_params)
        N = N + Fel
    return N


def step_3d(P, alpha, ops, Minv, dt, L, E_ext, elastic_params=None, stab=0.0):
    """One semi-implicit step, mirrors solver.step exactly (module docstring:
    same split, rebuilt for the 3D (Nz,Ny,Nxr) k-grid + full 3x3 Minv).

    SPEC V2-03 C3: accepts a (3,S,Nz,Ny,Nx) ensemble batch with no change to
    the arithmetic below -- see the ensemble section for the contract and for
    which elastic paths come along.

    stab=S > 0: solver.step's stabilized split, RHS = (1+dt*L*S)*P^n_hat -
    dt*L*FFT(N); Minv from build_Minv_3d(..., stab=S). stab=0 is the
    original arithmetic, bit for bit."""
    if stab:
        from fftpf import timestep_adaptive as adaptive
        Minv = adaptive.stabilized_operator(Minv, stab, "bulk3d.step_3d")
    elif getattr(Minv, "stab", 0.0):
        raise ValueError("bulk3d.step_3d: Minv was built with stab > 0; pass the same stab")
    if P.ndim == 5:
        _ensemble_reject(elastic_params)
    if solver._FUSED is not None:   # opt-in, execution.configure(..., fused=True)
        P_new = solver._FUSED.periodic_step("sbdf1", P, None, alpha, ops, Minv, dt, L, E_ext,
                                            elastic_params, stab, FFT_WORKERS, 3)
        if P_new is not None:
            return P_new
    N = _explicit_N_3d(P, alpha, ops, E_ext, elastic_params)
    s = (ops["Nz"], ops["Ny"], ops["Nx"])
    Pk = _rfftn(P)
    Nk = _rfftn(N)
    if stab:
        RHS = (1.0 + dt * L * stab) * Pk - dt * L * Nk
    else:
        RHS = Pk - dt * L * Nk                          # (3,Nz,Ny,Nxr)
    # replaces np.moveaxis(np.einsum("...ij,...j->...i", Minv,
    #   np.moveaxis(RHS,0,-1)), -1, 0) -- SPEC §39.1 A5, same explicit form
    # and the same strided views of the PUBLIC (Nz,Ny,Nxr,3,3) Minv as
    # solver.step (see its comment for the ellipsis/broadcast argument).
    Pk_new = np.empty_like(RHS)
    Pk_new[0] = Minv[..., 0, 0] * RHS[0] + Minv[..., 0, 1] * RHS[1] + Minv[..., 0, 2] * RHS[2]
    Pk_new[1] = Minv[..., 1, 0] * RHS[0] + Minv[..., 1, 1] * RHS[1] + Minv[..., 1, 2] * RHS[2]
    Pk_new[2] = Minv[..., 2, 0] * RHS[0] + Minv[..., 2, 1] * RHS[1] + Minv[..., 2, 2] * RHS[2]
    return _irfftn(Pk_new, s)


# --------------------------------------------------------------------------
# SPEC V2-03 A1: SBDF2 -- second-order semi-implicit, the 3D mirror of
# solver.build_Minv2/step_sbdf2 (WP4 §10c, Chen & Shen). SAME split (the
# A+D/kappa operator implicit, N explicit at the 2nd-order extrapolation
# P* = 2P^n - P^(n-1)), SAME bootstrap contract (two-level: the CALLER
# supplies P^(n-1); relax()/casekit.anneal/timestep.trajectory bootstrap
# step 1 with one step_3d call), rebuilt on this module's (Nz,Ny,Nxr)
# k-grid and full 3x3 M2 exactly as build_Minv_3d/step_3d rebuild the
# first-order pair:
#   [3I + 2dt*L*(A+D/kappa)] P^(n+1)_hat = 4P^n_hat - P^(n-1)_hat
#                                          - 2dt*L*FFT(N(2P^n - P^(n-1)))
# --------------------------------------------------------------------------
def build_Minv2_3d(grid, ops, dt, L, stab=0.0):
    """M2(k) = 3*I3 + 2*dt*L*[A(k)+D(k)/kappa], SBDF2's implicit operator --
    solver.build_Minv2's 3D mirror, with the full 3x3 population of
    build_Minv_3d (A13/A23/D13/D23/D33 have no 2D analogue). Shape
    (Nz,Ny,Nxr,3,3). stab=S > 0 adds 2*dt*L*S*I3 (same contract as
    build_Minv_3d)."""
    Nz, Ny, Nxr = grid["Nz"], grid["Ny"], grid["Nxr"]
    M = np.zeros((Nz, Ny, Nxr, 3, 3))
    M[..., 0, 0] = 3.0 + 2 * dt * L * (ops["A11"] + ops["D11"] / ops["kappa"])
    M[..., 1, 1] = 3.0 + 2 * dt * L * (ops["A22"] + ops["D22"] / ops["kappa"])
    M[..., 2, 2] = 3.0 + 2 * dt * L * (ops["A33"] + ops["D33"] / ops["kappa"])
    off12 = 2 * dt * L * (ops["A12"] + ops["D12"] / ops["kappa"])
    off13 = 2 * dt * L * (ops["A13"] + ops["D13"] / ops["kappa"])
    off23 = 2 * dt * L * (ops["A23"] + ops["D23"] / ops["kappa"])
    M[..., 0, 1] = M[..., 1, 0] = off12
    M[..., 0, 2] = M[..., 2, 0] = off13
    M[..., 1, 2] = M[..., 2, 1] = off23
    if stab:
        from fftpf import timestep_adaptive as adaptive
        stab = adaptive.check_stab(stab)
        for i in range(3):
            M[..., i, i] += 2 * dt * L * stab
        return adaptive.tag_operator(np.linalg.inv(M), stab)
    return np.linalg.inv(M)


def step_sbdf2_3d(P_n, P_nm1, alpha, ops, Minv2, dt, L, E_ext, elastic_params=None, stab=0.0):
    """One SBDF2 step (see the section comment above for the derivation) --
    mirrors solver.step_sbdf2 line for line, with this module's _rfftn/
    _irfftn and its own _explicit_N_3d. N is evaluated at the explicit
    2nd-order extrapolation P* = 2P^n - P^(n-1); the caller owns the
    bootstrap. At Nz=1 it reproduces solver.step_sbdf2 bit-for-bit
    (gates/core K2 `nz1_sbdf2_vs_2d`).

    SPEC V2-03 C3: accepts a (3,S,Nz,Ny,Nx) ensemble batch unchanged (the
    ensemble section's contract).

    stab=S > 0: solver.step_sbdf2's stabilized split (S*P implicit at n+1,
    explicit at P*); Minv2 from build_Minv2_3d(..., stab=S). stab=0 is the
    original arithmetic, bit for bit."""
    if stab:
        from fftpf import timestep_adaptive as adaptive
        Minv2 = adaptive.stabilized_operator(Minv2, stab, "bulk3d.step_sbdf2_3d")
    elif getattr(Minv2, "stab", 0.0):
        raise ValueError("bulk3d.step_sbdf2_3d: Minv2 was built with stab > 0; pass the same stab")
    if P_n.ndim == 5:
        _ensemble_reject(elastic_params)
    if solver._FUSED is not None:   # opt-in, execution.configure(..., fused=True)
        P_new = solver._FUSED.periodic_step("sbdf2", P_n, P_nm1, alpha, ops, Minv2, dt, L, E_ext,
                                            elastic_params, stab, FFT_WORKERS, 3)
        if P_new is not None:
            return P_new
    P_star = 2 * P_n - P_nm1
    N = _explicit_N_3d(P_star, alpha, ops, E_ext, elastic_params)
    s = (ops["Nz"], ops["Ny"], ops["Nx"])
    Pk_n = _rfftn(P_n)
    Pk_nm1 = _rfftn(P_nm1)
    Nk = _rfftn(N)
    RHS = 4 * Pk_n - Pk_nm1 - 2 * dt * L * Nk           # (3,Nz,Ny,Nxr)
    if stab:
        RHS = RHS + (2 * dt * L * stab) * (2 * Pk_n - Pk_nm1)
    # ONE idiom for the per-mode 3x3 apply everywhere in the tree -- see
    # step_3d/solver.step for the strided-view and broadcast argument.
    Pk_new = np.empty_like(RHS)
    Pk_new[0] = Minv2[..., 0, 0] * RHS[0] + Minv2[..., 0, 1] * RHS[1] + Minv2[..., 0, 2] * RHS[2]
    Pk_new[1] = Minv2[..., 1, 0] * RHS[0] + Minv2[..., 1, 1] * RHS[1] + Minv2[..., 1, 2] * RHS[2]
    Pk_new[2] = Minv2[..., 2, 0] * RHS[0] + Minv2[..., 2, 1] * RHS[1] + Minv2[..., 2, 2] * RHS[2]
    return _irfftn(Pk_new, s)


def _probe_run_3d(P0, alpha, ops, grid, L, E_ext, elastic_params, dt, n_probe, p_cap, scheme):
    """Run n_probe steps of `scheme` from P0; return True if it stayed finite
    and within p_cap throughout -- solver._probe_run's 3D mirror, including
    its sbdf2 bootstrap (one step_3d call, then n_probe-1 SBDF2 steps)."""
    Minv = build_Minv_3d(grid, ops, dt, L)
    if scheme == "sbdf1":
        P = P0.copy()
        for _ in range(n_probe):
            P = step_3d(P, alpha, ops, Minv, dt, L, E_ext, elastic_params=elastic_params)
            if not np.all(np.isfinite(P)) or np.max(np.abs(P)) > p_cap:
                return False
        return True
    elif scheme == "sbdf2":
        Minv2 = build_Minv2_3d(grid, ops, dt, L)
        P_nm1 = P0.copy()
        P_n = step_3d(P_nm1, alpha, ops, Minv, dt, L, E_ext, elastic_params=elastic_params)  # bootstrap
        if not np.all(np.isfinite(P_n)) or np.max(np.abs(P_n)) > p_cap:
            return False
        for _ in range(n_probe - 1):
            P_new = step_sbdf2_3d(P_n, P_nm1, alpha, ops, Minv2, dt, L, E_ext,
                                   elastic_params=elastic_params)
            if not np.all(np.isfinite(P_new)) or np.max(np.abs(P_new)) > p_cap:
                return False
            P_nm1, P_n = P_n, P_new
        return True
    raise ValueError(f"unknown scheme {scheme!r}, expected 'sbdf1' or 'sbdf2'")


def probe_stable_dt_3d(P0, alpha, ops, grid, L, E_ext=(0., 0., 0.), elastic_params=None,
                        dt0=0.02, n_probe=200, growth=2.0, p_cap=None, max_doublings=30,
                        scheme="sbdf1"):
    """Mirrors solver.probe_stable_dt exactly (same doubling protocol);
    rebuilt here because step_3d/build_Minv_3d are this module's own
    functions, not solver.py's. SPEC S32.5 dt-law addition applies: call
    this with the PRODUCTION forcing terms (quenched composite alpha field,
    E_ext) included, never an E_ext=0/homogeneous-alpha placeholder.
    scheme="sbdf2" (SPEC V2-03 A1) probes step_sbdf2_3d instead, bootstrapped
    the same way; scheme="sbdf1" (the default) is byte-identical to every
    pre-V2-03 call."""
    if p_cap is None:
        p_cap = max(10.0 * np.max(np.abs(P0)), 5.0)
    dt = dt0
    last_stable = None
    for _ in range(max_doublings):
        if not _probe_run_3d(P0, alpha, ops, grid, L, E_ext, elastic_params, dt, n_probe,
                             p_cap, scheme):
            break
        last_stable = dt
        dt *= growth
    if last_stable is None:
        raise RuntimeError(f"probe_stable_dt_3d: dt0={dt0} already unstable "
                            f"({n_probe} steps, scheme={scheme})")
    return last_stable / 2.0


# --------------------------------------------------------------------------
# R1's exact discrete-FDT Langevin/OU stepper (SPEC §18): REUSED from
# solver.py, not rebuilt (WP28 P3, SPEC §38.1 B2 = REVIEW_arch_minimality.md
# F4). solver._lin_matrix is shape-generic and reads its 3D-only slots with
# ops.get(key, 0.0), so a 3D ops dict -- which HAS A13/A23/D13/D23/D33 --
# populates all 9 entries through the same code that leaves them at 0.0 for
# solver.py's 2D ops; solver._ou_from_lin then does the eigh/decay/noise_amp
# construction, which never was grid-shape aware. _phi1 was already reused.
# --------------------------------------------------------------------------
def build_ou_operators_3d(ops, dt, L, kT, noise_cutoff_nm=None, grid=None):
    """Exact per-mode OU propagator for Lin(k)=A(k)+D(k)/kappa (SPEC §18) on
    this module's (Nz,Ny,Nxr) k-grid: IDENTICAL math to solver.
    build_ou_operators and now literally the same two lines (see that
    docstring for the formulas, the returned keys, the kT convention and the
    noise cutoff). Replaces a 10-line duplicate body plus a 13-line
    _lin_matrix_3d; byte-identical by construction, proven by gates/g27a-e
    and g30d.

    kT = units.kT_langevin(T, hx, hy, hz), an energy DENSITY, NOT kT_deck_3d
    (SPEC V2-04.0, D16). noise_cutoff_nm = l_c zeroes the stored noise
    amplitude where |k| >= 2*pi/l_c with |k|^2 = KXr^2+KYr^2+KZr^2 from
    `grid` (required with a cutoff, since ops carries no wavevectors);
    None returns the pre-V2-04 dict bit for bit.

    INDEFINITE Lin (SPEC §10): a genuinely negative eigenvalue (G < 0 ops
    from grad4.build_operators_hog_3d) is kept by solver._ou_from_lin and
    that mode grows by exp(-Lambda*dt) per step; only roundoff-negatives are
    floored. Before §10 it was clipped to 0, so G < 0 Langevin runs on this
    engine evolved without the k* instability."""
    Lin = solver._lin_matrix(ops)
    ou = solver._ou_from_lin(Lin, dt, L, kT)
    if noise_cutoff_nm is None:
        return ou
    if grid is None:
        raise ValueError("build_ou_operators_3d: noise_cutoff_nm needs grid= "
                         "(the KXr/KYr/KZr arrays the operators were built from)")
    k2 = operators.k_squared(grid, 3)
    return solver._apply_noise_cutoff(ou, k2, noise_cutoff_nm)


def step_langevin_3d(P, alpha, ops, ou, dt, L, E_ext, rng, elastic_params=None):
    """One first-order stochastic-splitting step, mirrors solver.
    step_langevin exactly (nonlinear bulk force explicit-Euler drift, exact
    per-mode OU update for the linear grad+depol family); single full-array
    standard-normal draw per step -> rfftn (SAME Hermitian-consistency-for-
    free argument as solver.step_langevin's own docstring, now over 3
    axes -- rfftn of a genuine i.i.d. real field still has total variance
    Nx*Ny*Nz at every stored k-bin by the identical Parseval argument).

    NOISE NORMALISATION (SPEC V2-04.0, ruling D16 -- a BEHAVIOR change, the
    same one solver.step_langevin took): the rfftn of the draw is used AS IS,
    the `/sqrt(Nz*Ny*Nx)` is gone, and `kT` is an energy DENSITY,
    units.kT_langevin(T, hx, hy, hz) -- NOT kT_deck_3d. The per-mode
    stationary law in this unnormalised mode variable is
    <|m_i(k)|^2> = N_cells*kT/chi_i(k); with the old pairing the physical
    temperature sampled was T*V_cell/N_cells. See solver.step_langevin's own
    note and REVIEW_v2_03c_kt.md for the forensics. The ensemble (S) path
    below goes through this same line: one draw of shape (3,S,Nz,Ny,Nx),
    rfftn over the trailing three axes, no normalisation, so S=1 stays
    bit-identical to the unbatched call (K2 `ensemble_s1_byte_3d`).

    SPEC V2-03 C3 (ensemble axis): the draw's shape is taken from P itself --
    (3,) + P.shape[1:] -- so an UNBATCHED P asks for exactly (3,Nz,Ny,Nx),
    the same count in the same C order from the same Generator as every
    pre-C3 call (bit-identical, K2 `ensemble_s1_byte_3d`), and a BATCHED P
    asks for (3,S,Nz,Ny,Nx): ONE full-array draw covering every member, which
    is what makes the S members' noise independent. Their independence
    therefore comes from the draw, not from S separate Generators -- and
    because numpy fills row-major, an S=1 batch consumes the SAME numbers in
    the SAME order as the unbatched call, so S=1 is bit-for-bit identical to
    no batching at all, Langevin included.
    CONSEQUENCE, stated rather than discovered later: a batched run does NOT
    reproduce S per-member unbatched runs member by member (the draws
    interleave differently); it reproduces their DISTRIBUTION. The
    deterministic steppers do reproduce per-member runs, to 1e-13 or better
    (K4 `ensemble_batching_3d`)."""
    if solver._FUSED is not None:   # opt-in, execution.configure(..., fused=True)
        if P.ndim == 5:
            _ensemble_reject(elastic_params)
        P_new = solver._FUSED.periodic_step("langevin", P, None, alpha, ops, ou, dt, L, E_ext,
                                            elastic_params, 0.0, FFT_WORKERS, 3, rng=rng,
                                            noise_shape=(3,) + P.shape[1:])
        if P_new is not None:
            return P_new
    N = _explicit_N_3d(P, alpha, ops, E_ext, elastic_params)
    if P.ndim == 5:
        _ensemble_reject(elastic_params)
    s = (ops["Nz"], ops["Ny"], ops["Nx"])
    Pk = _rfftn(P)
    Nk = _rfftn(N)
    Pk_star = Pk - dt * L * Nk

    eta = rng.standard_normal((3,) + P.shape[1:])
    eta_k = _rfftn(eta)                                 # NO 1/sqrt(N): kT is a density (V2-04.0)

    dc, nc = ou["decay_cf"], ou["noise_amp_cf"]         # (3,3,Nz,Ny,Nxr) complex
    # replaces np.einsum("...ij,...j->...i", ou["decay"], moveaxis(Pk_star,0,-1))
    # -- SPEC §39.1 A5, identical to solver.step_langevin's own explicit form
    decay_term = np.empty_like(Pk_star)
    decay_term[0] = dc[0, 0] * Pk_star[0] + dc[0, 1] * Pk_star[1] + dc[0, 2] * Pk_star[2]
    decay_term[1] = dc[1, 0] * Pk_star[0] + dc[1, 1] * Pk_star[1] + dc[1, 2] * Pk_star[2]
    decay_term[2] = dc[2, 0] * Pk_star[0] + dc[2, 1] * Pk_star[1] + dc[2, 2] * Pk_star[2]
    # replaces np.einsum("...ij,...j->...i", ou["noise_amp"], moveaxis(eta_k,0,-1))
    noise_term = np.empty_like(eta_k)
    noise_term[0] = nc[0, 0] * eta_k[0] + nc[0, 1] * eta_k[1] + nc[0, 2] * eta_k[2]
    noise_term[1] = nc[1, 0] * eta_k[0] + nc[1, 1] * eta_k[1] + nc[1, 2] * eta_k[2]
    noise_term[2] = nc[2, 0] * eta_k[0] + nc[2, 1] * eta_k[1] + nc[2, 2] * eta_k[2]
    Pk_new = decay_term + noise_term                    # same grouping as before
    return _irfftn(Pk_new, s)


block_stationary = solver.block_stationary  # dimension-agnostic (operates on a plain float list); reused unchanged


# --------------------------------------------------------------------------
# SPEC V2-03 C3: the ENSEMBLE (S) axis, WP4 SS10(e)'s contract on this engine.
#
# WHERE THE AXIS GOES, and why nothing above had to change. Exactly as in
# solver.py (see its own ensemble section for the full argument): the
# INTERNAL layout is (3,S,Nz,Ny,Nx) -- the realization axis is inserted
# AFTER the component axis, not before it -- so every P[0]/P[1]/P[2] index
# scattered through this module and elastic.py keeps meaning "component c,
# ALL realizations, ALL space", every rfftn/irfftn still names axes
# (-3,-2,-1), and every k-space operator, which is (Nz,Ny,Nxr[,3,3]),
# broadcasts against an (S,Nz,Ny,Nxr) spectrum by numpy's own left-padding
# rule. SPEC's prose lists the shape as (S,3,Nz,Ny,Nx); the two converters
# below make THAT the interchange format without forcing the internal order
# -- the same deliberate, documented deviation solver.py made.
#
# MEASURED, not assumed (K4 `ensemble_batching_3d`): with that layout,
# bulk_force, grad_force_3d, elec_force_3d (closed form AND the kappa(r) SPM
# fixed point), elastic_solve_3d/elastic_force_3d on a homogeneous C,
# _explicit_N_3d, step_3d and step_sbdf2_3d already reproduce S per-member
# runs BIT-FOR-BIT with no edit at all. Only three things needed one:
#   * step_langevin_3d's noise draw, which had the grid's shape hardcoded
#     (see its docstring);
#   * total_energy_3d, whose float(mean()) would have collapsed S different
#     states into one number;
#   * this section: the converters, and the loud refusal below.
# (That measurement used a constant E_ext and no eps_star_static; the S-axis
# alignment those two per-realization fields need is described just below.)
#
# WHICH ELASTIC PATHS COME ALONG (the work order asks this stated, so it is
# stated as measured, at this tree):
#   k0="stress_free", homogeneous C  -- BATCHED, bit-for-bit (ebar is
#       eps_star.mean over the three space axes, so it is (3,3,S) and its
#       [...,None,None,None] broadcast lands correctly). An eps_star_static
#       (setup forces it to (3,3,Nz,Ny,Nx)) and a field-shaped E_ext
#       (3,Nz,Ny,Nx) are per-realization fields: they need the S axis
#       inserted before they meet a batched strain/force, or numpy left-pads
#       them against the component axes (member s silently got static[j,s]
#       at S=3; S=1 grew an axis). elastic_solve_3d / the E_ext callers do
#       that insertion (elastic._ensemble_field), so these come along
#       bit-for-bit too.
#   k0="clamped"                     -- NOT SUPPORTED: that branch sets
#       ebar = np.zeros((3,3)) and then broadcasts ebar[...,None,None,None],
#       i.e. (3,3,1,1,1), against a (3,3,S,Nz,Ny,Nx) strain. It is one
#       missing axis, but it is in the ELASTIC section, which this package
#       does not own -- so it is REFUSED here, loudly, rather than patched
#       quietly somewhere it was not asked for.
#   el["spm"] (V2-03 C1's C(r) SPM) -- runs batched, but its fixed point
#       tests a POOLED residual over the whole batch, so the members come out
#       equal to per-member runs only to the solve's own tolerance (measured
#       1.5e-10 at S=3, outer_tol 1e-10), never bit-for-bit. Allowed; the
#       tolerance is the module's, not this axis's.
# --------------------------------------------------------------------------
def ensemble_to_internal_3d(P):
    """(S,3,Nz,Ny,Nx) -> (3,S,Nz,Ny,Nx). solver.ensemble_to_internal REUSED,
    not re-implemented: its body is one np.moveaxis(P, 0, 1), which is
    shape-agnostic -- the same alias idiom as bulk_force, _reshape_vec3_3d and
    block_stationary above."""
    return solver.ensemble_to_internal(P)


def ensemble_from_internal_3d(P):
    """(3,S,Nz,Ny,Nx) -> (S,3,Nz,Ny,Nx): the inverse, solver.
    ensemble_from_internal reused for the same reason."""
    return solver.ensemble_from_internal(P)


def _ensemble_reject(elastic_params):
    """Raise on the elastic setups the S axis cannot carry (see the section
    comment). Called by the three steppers and by total_energy_3d, and ONLY
    when P already carries the axis -- the unbatched path never reaches it."""
    if elastic_params is not None and elastic_params.get("k0") == "clamped":
        raise NotImplementedError(
            "ensemble axis with elastic_params is not supported in bulk3d for "
            "k0='clamped' (elastic_solve_3d's clamped branch builds a (3,3) ebar "
            "and broadcasts it as (3,3,1,1,1), which cannot meet a (3,3,S,Nz,Ny,Nx) "
            "strain). Use k0='stress_free', or run the members one at a time.")


# --------------------------------------------------------------------------
# IC helper -- mirrors solver.random_smooth_field, one axis added.
# --------------------------------------------------------------------------
def random_smooth_field_3d(grid, n_comp, amplitude, kmax_frac=0.3, seed=0, seeds=None):
    """n_comp band-limited (|k| < kmax_frac*max|k|) unit-variance random fields x amplitude, shape (n_comp,Nz,Ny,Nx).

    SPEC V2-03 C3: `seeds`, a sequence of S seeds, returns an ENSEMBLE BATCH
    in the internal layout (n_comp,S,Nz,Ny,Nx) whose member i is exactly this
    function at seed=seeds[i] -- the batch is stacked from the single-
    realization body, so "exactly" is bit-for-bit, and the ensemble IC needs
    no second generator to stay in step with the serial one. This is
    disorder.make_random_strain_batch's own contract (a `seeds` list, one
    realization per seed), which is the repo's established batch idiom.
    seeds=None (the default, every pre-C3 caller) ignores it entirely and
    `seed` alone is used: byte-identical."""
    if seeds is not None:
        return np.stack([random_smooth_field_3d(grid, n_comp, amplitude, kmax_frac=kmax_frac,
                                                seed=sd) for sd in seeds], axis=1)
    rng = np.random.default_rng(seed)
    KX, KY, KZ = grid["KX"], grid["KY"], grid["KZ"]
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    kmax = kmax_frac * kmag.max()
    out = np.zeros((n_comp, grid["Nz"], grid["Ny"], grid["Nx"]))
    for c in range(n_comp):
        raw = rng.standard_normal((grid["Nz"], grid["Ny"], grid["Nx"]))
        rawk = np.fft.fftn(raw)
        rawk[kmag > kmax] = 0.0
        field = np.fft.ifftn(rawk).real
        std = field.std()
        if std > 0:
            field = field / std
        out[c] = amplitude * field
    return out


# --------------------------------------------------------------------------
# kT unit conversion -- a NEW derivation, NOT units.si_to_deck_kT (that
# function is specific to solver.py's 2D engine, where grid['Volume']=Lx*Ly
# is an AREA and an arbitrary z-thickness Dz=hx has to be invented to give F
# units of a true energy -- see its own docstring). bulk3d's grid['Volume']
# =Lx*Ly*Lz IS a true SI volume once its nm numbers are cubed, so no
# adjustable length scale is needed at all.
#
# SPEC V2-04.0 (D16): this is a UNIT CONVERSION and not the FDT coefficient.
# The Langevin kT of all three engines is units.kT_langevin.
# --------------------------------------------------------------------------
def kT_deck_3d(T_kelvin, k_B=units.K_BOLTZMANN_SI):
    """Unit conversion of k_B*T into F_total units -- NOT the Langevin FDT
    coefficient (SPEC V2-04.0, D16). Use units.kT_langevin for that. Under
    the current noise convention (V2-04 / D16: no 1/sqrt(N_cells) on the
    k-space draw, kT an energy DENSITY) passing this number as
    step_langevin_3d's `kT` simulates T_kelvin*V_cell instead of T_kelvin,
    V_cell = hx*hy*hz in nm^3 taken as a pure number: kT_deck_3d(T) /
    units.kT_langevin(T, hx, hy, hz) = V_cell exactly (0.064 at 0.4 nm
    cells, i.e. ~16x too cold; cells above 1 nm^3 would run HOT). The
    N_cells/V_cell deficit REVIEW_v2_03c_kt.md (KT2) quotes was for the old
    pairing with the 1/sqrt(N_cells)-normalised noise (T*V_cell/N_cells).
    The derivation below is correct AS A UNIT CONVERSION and is unchanged.

    F_total_deck = F_total_SI * 1e18: deck energy density is SI*1e-9
    (units.SI_TO_DECK_ENERGY, the SAME building block si_to_deck_landau/
    _gradient use) and deck length is SI*1e-9 per axis (nm), so a true
    VOLUME integral picks up (1e-9)^3, not (1e-9)^1 -- F_deck = F_SI *
    SI_TO_DECK_ENERGY / (1e-9)^3 = F_SI * 1e-9 / 1e-27 = F_SI * 1e18.
    kT is an energy (same dimension as F_total, not a density), so the same
    factor applies (exp(-F/kT) is SI<->deck invariant iff numerator and
    denominator scale identically): kT_deck = k_B*T_kelvin * 1e18."""
    return k_B * T_kelvin * 1e18
