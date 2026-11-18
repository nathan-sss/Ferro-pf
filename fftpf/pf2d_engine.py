"""L1 semi-implicit Fourier-spectral TDGL solver, 2D periodic box.

Implements SPEC.md §1 (f_bulk, f_grad, f_elec) and §2 (Chen & Shen semi-implicit
stepper) for a 2D periodic box with 3-component P (k3 == 0, no z-dependence).
Grown past WP1's "no elasticity" scope since: the `elastic_params=` channel
into elastic.py on total_energy/total_force/step/relax (SPEC §2), the
inhomogeneous-kappa electrostatic SPM (WP5 §11.1, _spm_iterate), SBDF2 +
dt adaptation + ensemble/resample helpers (WP4 §10b-f), and Langevin OU
stepping (WP12 §18, build_ou_operators/step_langevin).

GATES: g0_consistency, g1a-c, g7b-g, g8a-d (SPM), g16a-e (Langevin), all
importing this module as `from solvers import solver`.

Units: whatever the deck uses (nm, deck energy-density units). Never converted.
Style: plain functions, arrays shaped (3,Ny,Nx) float64 / (Ny,Nx,...) spectra.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import scipy.fft as spfft
import sympy as sp

from fftpf import elastic_base as elastic
from fftpf import spectral_operators as operators
from fftpf import input_validation as validate

# --------------------------------------------------------------------------
# WP4 §10(b): FFT backend -- scipy.fft rfft2/irfft2 (multi-threaded via
# workers=-1) for every engine FFT, since every field transformed here (P,
# the explicit force N, and elastic.py's tau/eigenstrain fields) is REAL.
# All k-space multipliers built below (A, D, M, and elastic.py's Omega) are
# quadratic in (kx,ky) -- i.e. even under the joint sign flip (kx,ky) ->
# (-kx,-ky) -- which is exactly the condition for a real elementwise k-space
# multiplier to be Hermitian-consistent, so rfft2/irfft2 on the half-spectrum
# (kx in [0,Nx/2], full ky range) reproduce fft2/ifft2 on the full spectrum
# bit-for-bit up to roundoff (verified: gate G7b, <=1e-13 rel on a fixed
# state). "Complex path only where genuinely needed" (SPEC): nowhere in the
# per-step hot path needs one, so none remains there; the setup-time-only
# helpers (random_smooth_field here, cases/disorder.py's field generators)
# are untouched -- not the "engine" file layout targets and not measured by
# G7b's gate, so left as-is per Simplicity/Surgical-changes (see WP4_REPORT).
#
# SIX INDEPENDENT KNOBS, no linkage: FFT_WORKERS is module-level in solver.py
# (-1), elastic.py (-1), film3d.py (-1), kittel_afe.py (-1), octa_tilt.py
# (-1) and bulk3d.py (1, justified in its own header). Setting solver.
# FFT_WORKERS = 1 does NOT reach elastic.py, which solver.py itself calls on
# the elastic_params path -- a single-thread run (ENGINE_GUIDE §1.2) must set
# every module it touches.
# --------------------------------------------------------------------------
FFT_WORKERS = -1


def _rfft2(x):
    return spfft.rfft2(x, axes=(-2, -1), workers=FFT_WORKERS)


def _irfft2(xk, s):
    """s=(Ny,Nx) required: Nxr=Nx//2+1 alone can't recover odd-vs-even Nx."""
    return spfft.irfft2(xk, s=s, axes=(-2, -1), workers=FFT_WORKERS)

# --------------------------------------------------------------------------
# 1.1 f_bulk -- 6th/8th-order Landau energy, built symbolically once at import
# time (SPEC: "generate with sympy at init and lambdify to numpy (alpha's as
# arguments so arrays broadcast)"). _F_BULK below stays the SINGLE SOURCE OF
# TRUTH for both the energy and the force, but only the ENERGY is lambdified
# for production use: SYMPY VERIFIES, NUMPY EVALUATES -- the force is the
# hand-written pow-free Horner form in bulk_force(), and sympy's own
# sp.diff(_F_BULK, P_i) is what checks it at import (_verify_bulk_force_horner
# below, WP28 T1 / SPEC §40.3).
# --------------------------------------------------------------------------
_P1, _P2, _P3 = sp.symbols("P1 P2 P3", real=True)
_a1, _a11, _a12, _a111, _a112, _a123 = sp.symbols("a1 a11 a12 a111 a112 a123", real=True)
_a1111, _a1112, _a1122, _a1123 = sp.symbols("a1111 a1112 a1122 a1123", real=True)

_F_BULK = (
    _a1 * (_P1**2 + _P2**2 + _P3**2)
    + _a11 * (_P1**4 + _P2**4 + _P3**4)
    + _a12 * (_P1**2 * _P2**2 + _P2**2 * _P3**2 + _P1**2 * _P3**2)
    + _a111 * (_P1**6 + _P2**6 + _P3**6)
    + _a112 * (_P1**4 * (_P2**2 + _P3**2) + _P2**4 * (_P1**2 + _P3**2) + _P3**4 * (_P1**2 + _P2**2))
    + _a123 * _P1**2 * _P2**2 * _P3**2
    # WP4 §9.1(a)/§1.1: 8th-order terms, needed for bto_LCC2005_8th. The
    # complete, standard cubic-symmetry classification of degree-8
    # invariants in P1^2,P2^2,P3^2 (partitions of 4 into <=3 parts:
    # (4,0,0),(3,1,0),(2,2,0),(2,1,1)) -- matches the SAME pattern the 6th-
    # order terms above already follow, and its 4 terms match EXACTLY the 4
    # coefficient names Table 1 of the source PDF gives (alpha1111/1112/
    # 1122/1123), confirming this is the intended structure, not a guess.
    # Zero for every WP1-3 case (their TOML's alpha1111..alpha1123=0.0), so
    # this is an exact no-op there -- see ALPHA_KEYS_8TH's alpha.get(k,0.0)
    # fallback below for why no existing caller needs to change.
    + _a1111 * (_P1**8 + _P2**8 + _P3**8)
    + _a1112 * (_P1**6 * (_P2**2 + _P3**2) + _P2**6 * (_P1**2 + _P3**2) + _P3**6 * (_P1**2 + _P2**2))
    + _a1122 * (_P1**4 * _P2**4 + _P2**4 * _P3**4 + _P1**4 * _P3**4)
    + _a1123 * (_P1**4 * _P2**2 * _P3**2 + _P2**4 * _P1**2 * _P3**2 + _P3**4 * _P1**2 * _P2**2)
)

ALPHA_KEYS = ("a1", "a11", "a12", "a111", "a112", "a123")
ALPHA_KEYS_8TH = ("a1111", "a1112", "a1122", "a1123")  # optional; default 0.0, see bulk_force
_BULK_ARGS = (_P1, _P2, _P3, _a1, _a11, _a12, _a111, _a112, _a123, _a1111, _a1112, _a1122, _a1123)
_f_bulk_fn = sp.lambdify(_BULK_ARGS, _F_BULK, "numpy")
# Reference implementation of the force, kept ONLY for _verify_bulk_force_horner's
# numeric spot check (WP28 T1): production stepping calls bulk_force() below.
_df_bulk_fn_sympy = [sp.lambdify(_BULK_ARGS, sp.diff(_F_BULK, v), "numpy") for v in (_P1, _P2, _P3)]


# Selected by execution.py; keep this function stable for bulk3d's alias.
_BULK_FORCE_IMPL = None
# Opt-in fused step kernels (solvers.cpu_fused), set only by
# execution.configure(..., fused=True); bulk3d's steppers read it here too.
_FUSED = None


def bulk_energy_density(P, alpha):
    """f_bulk(P), the lambdified 6th/8th-order Landau density (SPEC §1.1); 8th-order alpha keys default to 0."""
    args = ([P[0], P[1], P[2]] + [alpha[k] for k in ALPHA_KEYS]
            + [alpha.get(k, 0.0) for k in ALPHA_KEYS_8TH])
    return _f_bulk_fn(*args)


def bulk_force(P, alpha):
    """delta F_bulk / delta P_i = d f_bulk / d P_i, real space, shape (3,...)
    float64 (the trailing shape of P: (3,Ny,Nx) here, (3,Nz,Ny,Nx) in bulk3d/
    film3d, an ensemble axis allowed in between). alpha values may be SCALARS
    or ARRAYS broadcasting against that trailing shape (composite.
    alpha_field_multi's alpha_i(r) fields); alpha's 8th-order keys
    (ALPHA_KEYS_8TH) are OPTIONAL, default 0.0 -- existing 6th-order-only
    callers (t180.py, r109.py, ...) don't need to supply them;
    bto_LCC2005_8th does.

    WP28 T1 (SPEC §40.3, §39.1 A1'): this REPLACED `np.stack([fn(*args) for fn
    in _df_bulk_fn])` over three sympy-lambdified derivatives. Those bodies
    spend 48 of their 279 ufunc passes per step in `P**n` with n != 2, i.e. a
    libm `pow` call per grid point (measured 46x the cost of `x*x`) -- the
    single largest cost in every engine's step (77-87% of it). The form below
    is pow-free: every power is reached by multiplication from q_i = P_i^2,
    and the six cubic-symmetric groups shared between the three components are
    computed once. Measured 14x (128^2, 8th-order) / 22x (32^3, 6th-order) on
    the force itself; 2.2-5.8x on the whole step.
    EQUIVALENCE CLASS: TOLERANCE, not byte -- the grouping reassociates the
    same sum. Max relerr 1.9e-12 on random P in +-1.2 P_s, <=1e-15 on warm
    states (REVIEW_wp27_efficiency_measured.md §2 #1 / §3 C1).
    PROOF that it is the same polynomial: _verify_bulk_force_horner() below,
    run at import, checks the grouping against sp.diff(_F_BULK, P_i)
    SYMBOLICALLY (sp.expand(...) == 0) and numerically against
    _df_bulk_fn_sympy.
    GATES: g0_consistency, g4_r109_recorded (2D recorded trajectories),
    G27a-e (bulk3d), G11f/G12a/G12b (film3d), G16a-e (Langevin), G20a-e,
    G22a-f, G25a-d -- and the full ALL_GATES=109 suite this WP owes.
    """
    if _BULK_FORCE_IMPL is not None:
        return _BULK_FORCE_IMPL(P, alpha)
    a1, a11, a12, a111, a112, a123 = (alpha[k] for k in ALPHA_KEYS)
    a1111, a1112, a1122, a1123 = (alpha.get(k, 0.0) for k in ALPHA_KEYS_8TH)
    # has8: skip the four 8th-order groups when every 8th-order alpha is
    # EXACTLY zero (the 6th-order decks: every pre-WP4 TOML). This changes
    # nothing numerically -- those groups would multiply exact 0.0 and add
    # exact 0.0 -- it only drops 45 of the 102 ufunc passes.
    has8 = any(np.any(np.asarray(a) != 0.0) for a in (a1111, a1112, a1122, a1123))

    P1, P2, P3 = P[0], P[1], P[2]
    q1, q2, q3 = P1 * P1, P2 * P2, P3 * P3   # q_i = P_i^2  (f_bulk is a polynomial in these)
    r1, r2, r3 = q1 * q1, q2 * q2, q3 * q3   # r_i = P_i^4
    if has8:
        t1, t2, t3 = r1 * q1, r2 * q2, r3 * q3   # t_i = P_i^6
    else:
        t1 = t2 = t3 = None                      # unused; the 8th-order groups are skipped

    # f_bulk is symmetric under permuting (P1,P2,P3), so component i is the
    # SAME expression with (i; j,k) cyclically relabelled: its own q_i, and
    # the other two entering only through symmetric combinations.
    cyclic = ((P1, q1, q2, q3, r2, r3, t2, t3),
              (P2, q2, q1, q3, r1, r3, t1, t3),
              (P3, q3, q1, q2, r1, r2, t1, t2))
    out = np.empty((3,) + P.shape[1:], dtype=np.result_type(P, np.float64))
    for i, (Pi, qi, qj, qk, rj, rk, tj, tk) in enumerate(cyclic):
        # The three symmetric functions of the OTHER two components that the
        # cubic-symmetric invariants of _F_BULK are built from:
        s = qj + qk        # q_j + q_k          (from a12, a112, a1112)
        rs = rj + rk       # q_j^2 + q_k^2      (from a112, a1122)
        p = qj * qk        # q_j q_k            (from a123, a1123)
        # d f_bulk / d P_i = 2 P_i * d f_bulk / d q_i, and d f/d q_i is a
        # CUBIC in q_i: c0 + c1 q_i + c2 q_i^2 + c3 q_i^3. One group per line,
        # 6th-order terms first, the 8th-order additions folded in below.
        c0 = a1 + a12 * s + a112 * rs + a123 * p              # d/dq_i of a1, a12, a112, a123
        c1 = 2 * a11 + 2 * a112 * s                           # d/dq_i of a11, a112
        c2 = 3 * a111                                         # d/dq_i of a111
        if has8:
            c0 = c0 + a1112 * (tj + tk) + a1123 * (p * s)     # + d/dq_i of a1112, a1123
            c1 = c1 + 2 * a1122 * rs + 2 * a1123 * p          # + d/dq_i of a1122, a1123
            c2 = c2 + 3 * a1112 * s                           # + d/dq_i of a1112
            # c3 = 4*a1111; Horner in q_i: ((c3 q + c2) q + c1) q + c0
            g = ((4 * a1111 * qi + c2) * qi + c1) * qi + c0
        else:
            g = (c2 * qi + c1) * qi + c0
        out[i] = 2 * Pi * g
    return out


def _verify_bulk_force_horner():
    """Import-time PERMANENT self-check for bulk_force's hand-written Horner
    grouping (WP28 T1, SPEC §40.3; same pattern as elastic.py's
    _verify_contract_C_cubic and bulk3d.py's _verify_elastic_force_symbolic_3d).
    (1) SYMBOLIC, exact: re-build the same c0/c1/c2/c3 grouping in sympy and
        require sp.expand(horner_i - sp.diff(_F_BULK, P_i)) == 0 -- so
        _F_BULK stays the single source of truth for the force too.
    (2) NUMERIC spot check (cheap: 8 points), against the lambdified
        derivatives _df_bulk_fn_sympy at random alphas -- catches a typo in
        the numpy transcription that the symbolic re-derivation above could
        not see, because it is written independently of it. Run TWICE, once
        per has8 branch: random 8th-order alphas exercise the 8th-order
        groups, and a second pass with those four set to exact 0.0 exercises
        the has8=False path every 6th-order deck (i.e. every 3D production
        deck) actually takes -- otherwise that branch would be covered
        symbolically but never executed here.
    Raises AssertionError on mismatch; never silently degrades."""
    a = dict(zip(ALPHA_KEYS + ALPHA_KEYS_8TH, _BULK_ARGS[3:]))
    q1, q2, q3 = _P1**2, _P2**2, _P3**2
    cyclic = ((_P1, q1, q2, q3), (_P2, q2, q1, q3), (_P3, q3, q1, q2))
    for i, (Pi, qi, qj, qk) in enumerate(cyclic):
        s, rs, p = qj + qk, qj**2 + qk**2, qj * qk
        ts, ps = qj**3 + qk**3, qj * qk * (qj + qk)
        c0 = a["a1"] + a["a12"] * s + a["a112"] * rs + a["a123"] * p + a["a1112"] * ts + a["a1123"] * ps
        c1 = 2 * a["a11"] + 2 * a["a112"] * s + 2 * a["a1122"] * rs + 2 * a["a1123"] * p
        c2 = 3 * a["a111"] + 3 * a["a1112"] * s
        c3 = 4 * a["a1111"]
        horner = 2 * Pi * (c0 + c1 * qi + c2 * qi**2 + c3 * qi**3)
        if sp.expand(horner - sp.diff(_F_BULK, (_P1, _P2, _P3)[i])) != 0:
            raise AssertionError(
                f"bulk_force Horner grouping != d f_bulk / d P{i + 1} (symbolic)")

    rng = np.random.default_rng(20260903)
    P = rng.uniform(-1.0, 1.0, (3, 8))
    alpha = {k: float(rng.uniform(-1.0, 1.0)) for k in ALPHA_KEYS + ALPHA_KEYS_8TH}
    alpha_6th = dict(alpha, **{k: 0.0 for k in ALPHA_KEYS_8TH})   # has8 False
    for branch, al in (("has8=True", alpha), ("has8=False", alpha_6th)):
        args = ([P[0], P[1], P[2]] + [al[k] for k in ALPHA_KEYS]
                + [al[k] for k in ALPHA_KEYS_8TH])
        ref = np.stack([fn(*args) for fn in _df_bulk_fn_sympy])
        if not np.allclose(bulk_force(P, al), ref, rtol=1e-11, atol=0.0):
            raise AssertionError(
                f"bulk_force Horner disagrees with the lambdified derivatives "
                f"(numeric, {branch})")


_verify_bulk_force_horner()


# --------------------------------------------------------------------------
# Grid & k-space linear operators (§1.2 grad, §1.3 elec)
# --------------------------------------------------------------------------
def make_grid(Nx, Ny, Lx, Ly):
    """2D periodic grid dict: hx/hy, full (KX,KY) shape (Ny,Nx), rfft2 half-spectrum (KXr,KYr) shape (Ny,Nxr), Volume=Lx*Ly."""
    hx, hy = Lx / Nx, Ly / Ny
    kx1d = operators.k_axis(Nx, hx)
    ky1d = operators.k_axis(Ny, hy)
    KX, KY = operators.k_mesh(kx1d, ky1d)  # shape (Ny,Nx); axis0=y, axis1=x
    # Half-spectrum (rfft2) companion grid for the engine's real-field FFTs
    # (WP4 §10b): rfft2 keeps the full range on axis0 (Ny) and only kx>=0 on
    # axis1 (Nxr=Nx//2+1) -- see the FFT_WORKERS block above for why this is
    # exact, not approximate, for every operator built on it below; the
    # layout is stated once in solvers/operators.py.
    Nxr = Nx // 2 + 1
    kxr1d = operators.k_axis_half(Nx, hx)
    KXr, KYr = operators.k_mesh(kxr1d, ky1d)  # shape (Ny,Nxr)
    return dict(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, hx=hx, hy=hy,
                kx1d=kx1d, ky1d=ky1d, KX=KX, KY=KY, Volume=Lx * Ly,
                Nxr=Nxr, kxr1d=kxr1d, KXr=KXr, KYr=KYr)


def build_operators(grid, G11, G12, G44, G44p, kappa, spm_method="basic",
                    spm_cg_max_iter=None):
    """A(k) (grad) and D(k) (depolarization) -- SPEC §1.2 / §1.3, built on the
    rfft2 half-spectrum grid (WP4 §10b).
    A13=A23=0 and D33=D13=D23=0 identically (k3==0); D(k=0)=0 (SPEC: E_d(k=0)=0).

    WP5 §11.1: `kappa` may be a plain scalar (homogeneous L1 background --
    every code path below is then BIT-IDENTICAL to pre-WP5, ops['dkappa']=None
    guards the new SPM branches off entirely, zero cost) OR a real-space
    (Ny,Nx) array (inhomogeneous kappa(x)). Array path: kappa0=(kappa_min+
    kappa_max)/2 (SPEC's own definition) becomes 'kappa' -- the single
    homogeneous reference every implicit operator (M/M2 here) and every SPM
    zero-order/iterate solve (elec_Ed/_spm_iterate below) uses; 'dkappa' =
    kappa(x)-kappa0 is the perturbation electrostatic_spm iterates against.

    spm_method: "basic" (default -- the fixed point, ops dict unchanged) or
    "cg" (opt-in, array kappa only: _spm_iterate then runs the
    preconditioned CG _spm_pcg on the same equations -- converges for any
    kappa(x) > 0 in O(sqrt(contrast)) iterations; docs/CG_SPM.md). A "cg"
    ops dict carries spm_method="cg" and spm_cg_max_iter (the CG cap used
    when a caller passes no max_iter; None -> SPM_CG_DEFAULT_MAX_ITER).
    With a scalar kappa there is no SPM and spm_method is ignored. An
    existing ops dict can be switched by setting those two keys."""
    validate.build_operators(grid, G11, G12, G44, G44p, kappa)
    if spm_method not in SPM_METHODS:
        raise ValueError(f"build_operators: spm_method must be one of {SPM_METHODS}, "
                         f"got {spm_method!r}")
    # A(k) and D(k) come from the shared operator layer (solvers/operators.py),
    # which owns the even-axis Nyquist rule: the odd x odd A12 and D12 are
    # zeroed on each even-axis Nyquist line, the diagonals are unmasked.
    A11, A22, A12, A33 = operators.grad_symbols_2d(grid, G11, G12, G44, G44p)
    D11, D22, D12 = operators.depolarization_symbols_2d(grid)

    kappa_arr = np.asarray(kappa, dtype=float)
    if kappa_arr.ndim == 0:
        kappa0 = float(kappa_arr)
        dkappa = None
    else:
        if kappa_arr.shape != (grid["Ny"], grid["Nx"]):
            raise ValueError(f"kappa array shape {kappa_arr.shape} != grid "
                              f"({grid['Ny']},{grid['Nx']})")
        kappa0 = float((kappa_arr.min() + kappa_arr.max()) / 2.0)  # SPEC §11.1
        dkappa = kappa_arr - kappa0

    ops = dict(A11=A11, A22=A22, A12=A12, A33=A33,
               D11=D11, D22=D22, D12=D12,
               G11=G11, G12=G12, G44=G44, G44p=G44p, kappa=kappa0, dkappa=dkappa,
               # stashed so grad_force/elec_Ed/elec_force/step can irfft2
               # back to the right real shape without a grid parameter:
               Nx=grid["Nx"], Ny=grid["Ny"])
    if spm_method == "cg" and dkappa is not None:   # a basic ops dict gains no key
        ops.update(spm_method="cg", spm_cg_max_iter=(SPM_CG_DEFAULT_MAX_ITER
                                                     if spm_cg_max_iter is None
                                                     else int(spm_cg_max_iter)))
    return ops


# The symbol helpers this module used to define live in solvers/operators.py
# (the ONE implementation of the even-axis Nyquist rule for every engine);
# the old private names stay importable as aliases of the shared functions.
# tests/test_spectral_consistency.py's fft2 reference applies the same
# declared Nyquist zeroing, so it checks convention consistency, not the raw
# KX*KY symbol.
_grad_symbols = operators.grad_symbols_2d
_modified_wavevectors_2d = operators.masked_wavevectors_2d
_nyquist_project_axis = operators.nyquist_project_axis


def _spectral_grad_xy(field2d, grid, KX=None, KY=None):
    """(df/dx, df/dy) of a single real (Ny,Nx) field via ik then ifft (SPEC §2:
    'compute ALL energies in real space; spectral derivatives via ik then ifft').
    rfft2/irfft2 (WP4 §10b): field2d is real, ik is purely imaginary times a
    real-even k, so the Hermitian symmetry needed for irfft2 holds exactly.
    Optional KX/KY override the grid wavevectors with real skew-adjoint
    wavevectors (each zeroed on its own even-axis Nyquist line) for the local
    gradient-density path (_modified_wavevectors_2d)."""
    s = (grid["Ny"], grid["Nx"])
    fk = _rfft2(field2d)
    KX = grid["KXr"] if KX is None else KX
    KY = grid["KYr"] if KY is None else KY
    dfdx = _irfft2(1j * KX * fk, s)
    dfdy = _irfft2(1j * KY * fk, s)
    return dfdx, dfdy


def grad_energy_density(P, grid, ops):
    """f_grad, SPEC §1.2, as the collocation (pointwise) local-gradient
    density plus a positive even-axis Nyquist correction.

    Representation note (repair of the even-grid Nyquist defect while
    preserving the pointwise local-gradient meaning): the pointwise
    |grad P|^2 form, evaluated with REAL skew-adjoint first derivatives
    (each derivative axis zeroed on its own even-axis Nyquist line), drops
    the alternating (Nyquist) mode on every even axis while grad_force keeps
    the k^2 eigenvalues, so energy and force would disagree there. We restore
    exactness by adding back each missing diagonal Nyquist term as a positive
    quadratic in the real Nyquist-projected component:
       0.5 * a_ij * (pi/h_j)^2 * (NyquistProject_j P_i)^2,
    a_ii = G11, a_ij = G44+G44p for i != j (the P3 out-of-plane diagonal in
    this 2D slice takes a_3j = G44+G44p, matching A33). On odd grids the
    correction is exactly zero and the ORIGINAL pointwise density is restored
    pointwise. The result is a collocation local-gradient density + Nyquist
    correction -- NOT 0.5*P.(A P): its grid mean equals the energy density
    consistent with grad_force, while its pointwise values keep the local
    |grad P|^2 meaning for wall_metrics and other local energy maps. G is
    rebuilt from the scalar coefficients (never ops['A*'], which the hog
    path augments with H11), so the H11 separation in
    grad4.grad_energy_density_hog is preserved."""
    G11, G12, G44, G44p = ops["G11"], ops["G12"], ops["G44"], ops["G44p"]
    KX, KY = operators.masked_wavevectors_2d(grid)
    P1x, P1y = _spectral_grad_xy(P[0], grid, KX, KY)
    P2x, P2y = _spectral_grad_xy(P[1], grid, KX, KY)
    P3x, P3y = _spectral_grad_xy(P[2], grid, KX, KY)
    fg = (0.5 * G11 * (P1x**2 + P2y**2) + G12 * P1x * P2y
          + 0.5 * (G44 + G44p) * (P3x**2 + P3y**2)
          + 0.5 * G44 * (P1y + P2x)**2 + 0.5 * G44p * (P1y - P2x)**2)
    a = operators.grad_nyquist_coefficients(G11, G44, G44p, 2)
    return operators.add_nyquist_correction(fg, P, grid, a, 2)


def grad_force(P, ops):
    """delta F_grad / delta P = A(k) . P, evaluated via the same A(k) as in M (SPEC G0)."""
    s = (ops["Ny"], ops["Nx"])
    Pk = _rfft2(P)
    F1 = ops["A11"] * Pk[0] + ops["A12"] * Pk[1]
    F2 = ops["A12"] * Pk[0] + ops["A22"] * Pk[1]
    F3 = ops["A33"] * Pk[2]
    Fk = np.stack([F1, F2, F3])
    return _irfft2(Fk, s)


def _elec_Ed_core(source, ops):
    """Homogeneous-kappa0 depolarization-field solve, SPEC §1.3: -E_d,i =
    (1/kappa0) D_ij(k) source_j(k). This is the L1 formula, parameterized on
    an arbitrary real-space vector `source` instead of hardcoding P -- reused
    unchanged as elec_Ed's fast path (source=P) AND as the elliptic solve
    _spm_iterate below repeats each round (source=P+dkappa*E, WP5 §11.1)."""
    s = (ops["Ny"], ops["Nx"])
    Sk = _rfft2(source)
    kappa0 = ops["kappa"]
    Ed1k = -(ops["D11"] * Sk[0] + ops["D12"] * Sk[1]) / kappa0
    Ed2k = -(ops["D12"] * Sk[0] + ops["D22"] * Sk[1]) / kappa0
    Ed3k = np.zeros_like(Sk[2])
    Edk = np.stack([Ed1k, Ed2k, Ed3k])
    return _irfft2(Edk, s)


# --------------------------------------------------------------------------
# WP5 §11.1: electrostatic SPM (Wang et al., Acta Mater 61 (2013) 7591, §2),
# deck units, isotropic scalar kappa(x), no free charge. Poisson:
#   d_i[kappa(x) d_i phi] = d_i P_i,  kappa(x) = kappa0 + dkappa(x)
# Fixed point (their Eq 2.10/2.16-2.20, specialized): each iterate solves the
# SAME homogeneous-kappa0 elliptic problem (_elec_Ed_core above) with an
# extra source dkappa(x)*E^(n-1) (E^(n-1) = E_ext + E_d^(n-1), the TOTAL
# field from the previous iterate); zero order (n=0) has no extra source --
# it IS _elec_Ed_core(P, ops), i.e. the old L1 formula exactly. Converges on
# max|E^n-E^(n-1)|/max|E| <= tol; capped at max_iter (SPEC: cap 200, warn
# >80 -- both defaults below).
# --------------------------------------------------------------------------
def _spm_iterate(P, ops, tol=1e-10, max_iter=None, warn_iter=None, record_history=False):
    """Returns (Ed_converged, Ed_uniform, n_iter, resid[, history]). Ed_uniform
    is the n=0 (homogeneous-kappa0) term -- callers that need the SPM
    CORRECTION dE = Ed_converged - Ed_uniform (the hybrid time-stepping term,
    SPEC §11.1) get both without a redundant second solve.

    E_ext is deliberately NOT part of this iteration (SPEC §11.1: "the
    operator is linear in P, so the self-field 1/2 stands" + "External E_ext
    unchanged" -- read together as: E_d is a map of P ALONE, kept exactly
    linear/self-adjoint, same as the homogeneous-kappa L1 formula; E_ext
    stays the separate, un-redistributed additive term it always was,
    handled outside this function by elec_energy_density/elec_force/
    _explicit_N). Wang et al. 2013's own Eq (2.10) folds E_ext INTO the
    dkappa(x)*(E_ext+E_d) source, which makes their converged E_d an AFFINE
    (not linear) function of P once both E_ext!=0 and dkappa(x)!=0 -- gate
    G8c caught exactly this: with E_ext folded in, f_elec=-1/2 P.Ed(P)-P.Eext
    mismatched its FD derivative by a clean, tol-independent ~5e-4 (traced
    to the missing factor of 1/2 on the P-independent, Eext-sourced part of
    Ed -- NOT SPM iteration noise, confirmed by tightening spm_tol 1000x
    with zero change). Dropping E_ext from the source restores an exactly
    linear/self-adjoint P->Ed map (verified: G8c now agrees to ~1e-10);
    V2 (SPEC §11.3) runs at E_ext=0 throughout, where the two formulations
    are identical, so nothing in this WP's actual deliverables depends on
    which convention is used -- this is the one that makes SPEC's own
    "linear in P" / "External E_ext unchanged" claims literally true.

    max_iter / warn_iter: None means SPEC's 200 / 80 (the basic defaults).

    ops built with build_operators(..., spm_method="cg") (ops["spm_method"]
    == "cg") are solved by the preconditioned CG _spm_pcg instead -- same
    fixed point, same return contract; n counts CG iterations, resid is the
    CG relative residual, and max_iter=None means ops["spm_cg_max_iter"]
    (docs/CG_SPM.md)."""
    if ops.get("spm_method") == "cg":
        return _spm_pcg(P, ops["dkappa"], lambda src: _elec_Ed_core(src, ops), 2, tol=tol,
                        max_iter=max_iter if max_iter is not None else ops["spm_cg_max_iter"],
                        warn_iter=warn_iter, record_history=record_history,
                        where="solver._spm_iterate")
    max_iter = 200 if max_iter is None else max_iter
    warn_iter = 80 if warn_iter is None else warn_iter
    dkappa = ops["dkappa"]
    Ed_uniform = _elec_Ed_core(P, ops)
    Ed = Ed_uniform
    E_prev = Ed
    history = [] if record_history else None
    resid = float("inf")
    n = 0
    for n in range(1, max_iter + 1):
        S = dkappa * E_prev
        Ed = _elec_Ed_core(P + S, ops)
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
        print(f"[solver._spm_iterate] WARNING: SPM ran {n} iterations (> warn_iter={warn_iter}), "
              f"final resid={resid:.3e} (tol={tol:.0e}).")
    if resid > tol:
        print(f"[solver._spm_iterate] WARNING: SPM did NOT converge within max_iter={max_iter} "
              f"(final resid={resid:.3e} > tol={tol:.0e}).")
    if record_history:
        return Ed, Ed_uniform, n, resid, history
    return Ed, Ed_uniform, n, resid


# --------------------------------------------------------------------------
# Opt-in preconditioned CONJUGATE GRADIENT for the same kappa(x) problem:
# build_operators(..., spm_method="cg") here, build_operators_3d(...,
# spm_method="cg") in bulk3d (whose _spm_iterate_3d calls _spm_pcg with its
# own 3D core). The default never reaches this section. docs/CG_SPM.md has
# the derivation and the benchmark; in short:
#
# Write core(s) = _elec_Ed_core(s, ops) = -T s with T = D(k)/kappa0 per
# mode: real symmetric positive SEMI-definite, eigenvalues in [0, 1/kappa0]
# (D(k) is the k k^T/k^2 projector, or on an even-axis Nyquist line the
# diagonal remnant the D12 convention leaves). On E in range(T) the basic
# fixed point E = core(P + dkappa*E) IS the linear system
#     B E = c,   B = T^+ + Pi dkappa Pi,   c = -Pi P
# (T^+ the pseudo-inverse, Pi the projection onto range(T)). Because
# T^+ >= kappa0 Pi, B >= Pi kappa(x) Pi: symmetric positive-definite for ANY
# kappa(x) > 0, whatever kappa0 -- where the basic iteration needs
# max|dkappa| < kappa0. On every projector mode T^+ = kappa0 Pi and B is the
# Galerkin operator of div(kappa grad .) itself; on the Nyquist lines it
# keeps the basic discretization's own kappa0 term, so both methods solve
# the SAME discrete equations.
#
# Preconditioner M = T^+, i.e. M^-1 = T = the homogeneous-kappa0 solve the
# basic scheme repeats; kappa(M^-1 B) <= max(1, kmax/kappa0) /
# min(1, kmin/kappa0) -- the contrast, so ~sqrt(contrast) iterations.
# T^+ is never formed: every vector v of range(T) is carried with a
# preimage v~ (v = T v~, hence M v = Pi v~), and the projection drops out of
# every inner product against range(T). ONE core solve per iteration.
#
# Stopping: resid = sqrt((r, M^-1 r) / (c, M^-1 c)) <= tol -- the relative
# residual in the kappa0-energy norm ((c, M^-1 c) = -<P, Ed_uniform>, twice
# the homogeneous-kappa0 electrostatic energy), re-checked on a recomputed
# residual before returning. Inner product: the plain sum over components
# and grid points (film3d_spm would pass its quadrature-weighted one).
# --------------------------------------------------------------------------
SPM_METHODS = elastic.SPM_METHODS                     # ("basic", "cg"), one definition
SPM_CG_DEFAULT_MAX_ITER = elastic.CG_DEFAULT_MAX_ITER


def _spm_pcg(P, dkappa, core, nsp, tol=1e-10, max_iter=SPM_CG_DEFAULT_MAX_ITER,
             warn_iter=None, record_history=False, warm=None, where="solver._spm_iterate",
             inner=None):
    """The section comment's PCG. core(src) -> E: the homogeneous-kappa0
    solve (2D _elec_Ed_core, 3D bulk3d._elec_Ed_core_3d); nsp = number of
    space axes; P (3,[S,]*space) -- an ensemble axis runs S independent
    CGs in lockstep (per-member alpha/beta, converged members frozen).
    warm: optional previous E (the basic warm-start contract): the start is
    then ONE basic step from it, E0 = core(P + dkappa*warm), whose preimage
    is known exactly. inner(a, b): the inner product (default: sum over
    components and grid; per member if batched).

    Returns _spm_iterate's contract (Ed, Ed_uniform, n_iter, resid[,
    history]): Ed the CG solution (exactly core() of its accumulated
    preimage), Ed_uniform = core(P), n_iter the CG iteration count, resid
    the final freshly recomputed relative residual (max over members).
    warn_iter=None: warn only when tol is not reached (the count itself is
    expected to exceed the basic warn_iter=80 at high contrast)."""
    batched = P.ndim == 2 + nsp
    ex = (Ellipsis,) + (None,) * nsp
    if inner is None:
        def inner(a, b):
            tot = 0.0
            for c in range(a.shape[0]):
                tot = tot + elastic._field_dot(a[c], b[c], nsp)
            return np.asarray(tot, dtype=float)

    def bc(c):
        return c[ex] if batched else c

    def T(y):                                    # M^-1 = T = -core
        return -core(y)

    Ed_uniform = core(P)
    nb2 = -inner(P, Ed_uniform)                  # (c, M^-1 c) = <P, T P>
    if warm is None:
        Et = np.zeros_like(P)                    # preimage of E (E = T Et)
        E = np.zeros_like(Ed_uniform)
        r = -P                                   # preimage of the residual
        z = Ed_uniform                           # M^-1 r = T r
    else:
        src = P + dkappa * warm
        Et = -src
        E = core(src)                            # == T(Et), one basic step from warm
        r = dkappa * (warm - E)                  # -P - Et - dkappa*E
        z = T(r)
    rz = inner(r, z)
    den = np.where(nb2 > 0, nb2, rz)             # P == 0: relative to the start

    def rel(rz_):
        return np.sqrt(elastic._safe_ratio(np.maximum(rz_, 0.0), den))

    history = [] if record_history else None
    p, pt = z, r.copy()
    n, fresh = 0, True
    while True:
        resid = rel(rz)
        conv = resid <= tol
        if np.all(conv) or n >= max_iter:
            if not fresh:                        # verify on the TRUE residual
                E = T(Et)
                r = -P - Et - dkappa * E
                z = T(r)
                rz = inner(r, z)
                resid = rel(rz)
                conv = resid <= tol
                fresh = True
            if np.all(conv) or n >= max_iter:
                break
            p, pt = z, r.copy()                  # restart from the true residual
        active = ~conv
        q = pt + dkappa * p                      # preimage of B p
        pq = inner(p, q)
        if np.any((pq <= 0) & (rz > 0) & active):
            raise RuntimeError(
                f"[{where}] SPM (cg): (p, B p) <= 0 on a nonzero search direction after {n} "
                "CG iterations -- the discrete operator is not positive definite: kappa(x) "
                "is not strictly positive, or (film lane) a nodal dkappa sits below the "
                "face-averaged kappa0(z) reference next to a z-graded/dead layer. The basic "
                "fixed point has no convergent iteration on such an operator either.")
        alpha = elastic._safe_ratio(rz, pq) * active
        Et += bc(alpha) * pt
        r -= bc(alpha) * q
        z = T(r)
        rz_new = inner(r, z)
        beta = elastic._safe_ratio(rz_new, rz) * active
        p = z + bc(beta) * p
        pt = r + bc(beta) * pt
        rz = rz_new
        n += 1
        fresh = False
        if record_history:
            history.append(float(np.max(rel(rz))))
    resid = float(np.max(resid))
    if warn_iter is not None and n > warn_iter:
        print(f"[{where}] WARNING: SPM (cg) ran {n} iterations (> warn_iter={warn_iter}), "
              f"final resid={resid:.3e} (tol={tol:.0e}).")
    if resid > tol:
        print(f"[{where}] WARNING: SPM (cg) did NOT converge within max_iter={max_iter} "
              f"(final resid={resid:.3e} > tol={tol:.0e}).")
    if record_history:
        return E, Ed_uniform, n, resid, history
    return E, Ed_uniform, n, resid


def elec_Ed(P, ops, E_ext=(0., 0., 0.), spm_tol=1e-10):
    """Depolarization field E_d -- a function of P ALONE in both regimes
    (SPEC §1.3 homogeneous-kappa L1 path, ops['dkappa'] is None, bit-
    identical to pre-WP5; or SPEC §11.1 WP5 electrostatic-SPM path,
    ops['dkappa'] is an array). E_ext is accepted for call-site uniformity
    with elec_energy_density/elec_force (which DO use it, as the separate,
    un-redistributed "-P.E_ext" term) but is NOT used here -- see
    _spm_iterate's docstring for why E_ext is kept OUT of the SPM iteration
    itself. spm_tol: SPM convergence tolerance (default 1e-10 = SPEC's
    production value, used unchanged by every stepping call); FD-style
    consistency checks (gate G8c) tighten it for a cleaner comparison."""
    if ops.get("dkappa") is None:
        return _elec_Ed_core(P, ops)
    Ed, _Ed_uniform, _n, _resid = _spm_iterate(P, ops, tol=spm_tol)
    return Ed


def elec_energy_density(P, ops, E_ext, spm_tol=1e-10):
    """f_elec = -0.5*P.E_d - P.E_ext (SPEC §1.3), shape (Ny,Nx); E_d goes through the SPM iteration when ops carries dkappa."""
    Ed = elec_Ed(P, ops, E_ext, spm_tol=spm_tol)
    Eext = _reshape_vec3(E_ext, P.ndim, ensemble=P.ndim == 4)
    return -0.5 * (P[0] * Ed[0] + P[1] * Ed[1] + P[2] * Ed[2]) - (P[0] * Eext[0] + P[1] * Eext[1] + P[2] * Eext[2])


# (3,) E_ext -> (3,1,...,1); a field-shaped E_ext (WP9 §15.1 ch3) passes
# through unbatched and gets the ensemble axis inserted / checked when
# ensemble=True (P.ndim == 4 here, 5 in bulk3d). ONE implementation for every
# engine and sector: operators.vec3_field (docstring there); this name is an
# alias of it, kept for every caller that imports it.
_reshape_vec3 = operators.vec3_field


def elec_force(P, ops, E_ext, spm_tol=1e-10):
    """delta F_elec / delta P_i = -E_d,i - E_ext,i (the 1/2 disappears, SPEC §1.3;
    SPEC §11.1 WP5: still holds with the SPM-converged E_d -- the SPM map
    P -> E_d is linear and self-adjoint at convergence [exact-elliptic-
    operator argument], so the same envelope-free identity applies; verified
    numerically by gate G8c)."""
    Ed = elec_Ed(P, ops, E_ext, spm_tol=spm_tol)
    Eext = _reshape_vec3(E_ext, P.ndim, ensemble=P.ndim == 4)
    return -Ed - Eext


def total_energy(P, alpha, ops, E_ext, grid, elastic_params=None, spm_tol=1e-10):
    """Return (densities dict, totals dict) for bulk/grad/elec[/elastic]/total.
    elastic_params=None (WP1 default) omits the f_el term entirely -- SPEC WP1
    scope has no elasticity; WP2 passes elastic.setup_elastic(...)'s output."""
    fb = bulk_energy_density(P, alpha)
    fg = grad_energy_density(P, grid, ops)
    fe = elec_energy_density(P, ops, E_ext, spm_tol=spm_tol)
    dens = dict(bulk=float(fb.mean()), grad=float(fg.mean()), elec=float(fe.mean()))
    if elastic_params is not None:
        sol = elastic.elastic_solve(P, elastic_params)
        dens["elastic"] = float(elastic.elastic_energy_density(sol).mean())
    dens["total"] = sum(dens.values())
    tot = {k: v * grid["Volume"] for k, v in dens.items()}
    return dens, tot


def total_force(P, alpha, ops, E_ext, elastic_params=None, spm_tol=1e-10):
    """Return (Fb, Fg, Fe[, Fel], sum), each shape (3,Ny,Nx)."""
    Fb = bulk_force(P, alpha)
    Fg = grad_force(P, ops)
    Fe = elec_force(P, ops, E_ext, spm_tol=spm_tol)
    if elastic_params is None:
        return Fb, Fg, Fe, Fb + Fg + Fe
    Fel, _ = elastic.elastic_force(P, elastic_params)
    return Fb, Fg, Fe, Fel, Fb + Fg + Fe + Fel


# --------------------------------------------------------------------------
# §2 Time stepping -- first-order semi-implicit (Chen & Shen)
# --------------------------------------------------------------------------
def build_Minv(grid, ops, dt, L, stab=0.0):
    """M(k) = I3 + dt*L*[A(k) + D(k)/kappa], inverted per k-point (3x3, generic
    batched inverse -- not hand-decoupled, per the spirit of SPEC §1.4).
    Shape (Ny,Nxr,3,3): half-spectrum rfft2 grid (WP4 §10b).

    stab=S > 0 (opt-in linear stabilization, docs/ADAPTIVE_DT.md): M(k) gains
    + dt*L*S*I3 and the returned inverse remembers S; step() must then be
    called with the SAME stab (it refuses a mismatch either way). stab=0 is
    the original code path, bit for bit."""
    Ny, Nx = grid["Ny"], grid["Nxr"]
    M = np.zeros((Ny, Nx, 3, 3))
    M[..., 0, 0] = 1.0 + dt * L * (ops["A11"] + ops["D11"] / ops["kappa"])
    M[..., 1, 1] = 1.0 + dt * L * (ops["A22"] + ops["D22"] / ops["kappa"])
    M[..., 2, 2] = 1.0 + dt * L * ops["A33"]
    off = dt * L * (ops["A12"] + ops["D12"] / ops["kappa"])
    M[..., 0, 1] = off
    M[..., 1, 0] = off
    if stab:
        from fftpf import timestep_adaptive as adaptive
        stab = adaptive.check_stab(stab)
        for i in range(3):
            M[..., i, i] += dt * L * stab
        return adaptive.tag_operator(np.linalg.inv(M), stab)
    return np.linalg.inv(M)


def _explicit_N(P, alpha, ops, E_ext, elastic_params):
    """N(P) = d f_bulk/dP - E_ext [- dE, WP5 §11.1 hybrid stepping] [+ d F_el/dP
    if elastic_params given] -- the explicit (real-space) RHS term shared by
    both step() and step_sbdf2().

    WP5 §11.1 hybrid time-stepping: M(k) (build_Minv/build_Minv2) keeps the
    HOMOGENEOUS D(k)/kappa0 implicit operator unchanged (big-dt property
    preserved); the difference between the true kappa(x) field and what M(k)
    implicitly assumes, dE = Ed_spm_converged(P) - Ed_uniform_kappa0(P), is
    added here explicitly. Derivation: true elec force = -Ed_spm - E_ext =
    -(Ed_uniform + dE) - E_ext; the -Ed_uniform part is exactly the term M(k)
    already handles implicitly (same structure as the L1 D(k)/kappa term), so
    the remaining explicit piece is -E_ext - dE (was just -E_ext pre-WP5).
    ops['dkappa'] is None (homogeneous kappa, every pre-WP5 caller) => this
    whole branch is skipped, N is bit-identical to before (G8d)."""
    N = bulk_force(P, alpha) - _reshape_vec3(E_ext, P.ndim, ensemble=P.ndim == 4)
    if ops.get("dkappa") is not None:
        Ed, Ed_uniform, _n, _resid = _spm_iterate(P, ops)
        N = N - (Ed - Ed_uniform)
    if elastic_params is not None:
        Fel, _ = elastic.elastic_force(P, elastic_params)
        N = N + Fel
    return N


def step(P, alpha, ops, Minv, dt, L, E_ext, elastic_params=None, stab=0.0):
    """One semi-implicit step: P^(n+1) = Minv(k) . [P^n_hat - dt*L*FFT(N(P^n))].
    N(P) = d f_bulk/dP - E_ext [+ d F_el/dP if elastic_params given -- SPEC §2:
    the elastic driving force is explicit, not part of the M(k) operator].

    stab=S > 0: (P^(n+1)-P^n)/dt = -L(A+D/kappa+S)P^(n+1) - L[N(P^n) - S*P^n],
    i.e. RHS = (1+dt*L*S)*P^n_hat - dt*L*FFT(N); Minv must come from
    build_Minv(..., stab=S) with the same S (refused otherwise). Fixed points
    are unchanged. stab=0 is the original arithmetic, bit for bit."""
    if stab:
        from fftpf import timestep_adaptive as adaptive
        Minv = adaptive.stabilized_operator(Minv, stab, "solver.step")
    elif getattr(Minv, "stab", 0.0):
        raise ValueError("solver.step: Minv was built with stab > 0; pass the same stab to step")
    if _FUSED is not None:
        P_new = _FUSED.periodic_step("sbdf1", P, None, alpha, ops, Minv, dt, L, E_ext,
                                     elastic_params, stab, FFT_WORKERS, 2)
        if P_new is not None:
            return P_new
    N = _explicit_N(P, alpha, ops, E_ext, elastic_params)
    s = (ops["Ny"], ops["Nx"])
    Pk = _rfft2(P)
    Nk = _rfft2(N)
    if stab:
        RHS = (1.0 + dt * L * stab) * Pk - dt * L * Nk
    else:
        RHS = Pk - dt * L * Nk                          # (3,[S,]Ny,Nxr)
    # replaces np.moveaxis(np.einsum("...ij,...j->...i", Minv,
    #   np.moveaxis(RHS,0,-1)), -1, 0) -- SPEC §39.1 A5 (0.63 -> 0.27 ms/apply
    # at 32^3; the two moveaxis views go away with it). Minv[...,i,j] is a
    # strided (Ny,Nxr) view of the PUBLIC (Ny,Nxr,3,3) return, which stays
    # exactly as it was (29 gate files read it).
    # Ensemble axis (WP4 §10e, gate G7e): RHS[j] is (S,Ny,Nxr) and
    # Minv[...,i,j] is (Ny,Nxr), so numpy broadcasts Minv in as if it had an
    # implicit leading S=1 -- the same thing the einsum's "..." did.
    Pk_new = np.empty_like(RHS)
    Pk_new[0] = Minv[..., 0, 0] * RHS[0] + Minv[..., 0, 1] * RHS[1] + Minv[..., 0, 2] * RHS[2]
    Pk_new[1] = Minv[..., 1, 0] * RHS[0] + Minv[..., 1, 1] * RHS[1] + Minv[..., 1, 2] * RHS[2]
    Pk_new[2] = Minv[..., 2, 0] * RHS[0] + Minv[..., 2, 1] * RHS[1] + Minv[..., 2, 2] * RHS[2]
    return _irfft2(Pk_new, s)


# --------------------------------------------------------------------------
# WP4 §10(c): SBDF2 -- second-order semi-implicit (Chen & Shen, same source
# paper as the §2 first-order scheme). Two-level (needs P^(n-1) and P^n);
# bootstraps with one step() call (SPEC: "bootstrap step 1 with the existing
# 1st-order scheme").
#   (3P^(n+1) - 4P^n + P^(n-1))/(2dt) = -L[A+D/kappa]P^(n+1) - L*FFT(N(2P^n-P^(n-1)))
# => [3*I + 2*dt*L*(A+D/kappa)] P^(n+1)_hat = 4P^n_hat - P^(n-1)_hat - 2*dt*L*FFT(N(2P^n-P^(n-1)))
# --------------------------------------------------------------------------
def build_Minv2(grid, ops, dt, L, stab=0.0):
    """M2(k) = 3*I3 + 2*dt*L*[A(k) + D(k)/kappa], SBDF2's implicit operator.
    stab=S > 0 adds 2*dt*L*S*I3 (same contract as build_Minv's stab)."""
    Ny, Nx = grid["Ny"], grid["Nxr"]
    M = np.zeros((Ny, Nx, 3, 3))
    M[..., 0, 0] = 3.0 + 2 * dt * L * (ops["A11"] + ops["D11"] / ops["kappa"])
    M[..., 1, 1] = 3.0 + 2 * dt * L * (ops["A22"] + ops["D22"] / ops["kappa"])
    M[..., 2, 2] = 3.0 + 2 * dt * L * ops["A33"]
    off = 2 * dt * L * (ops["A12"] + ops["D12"] / ops["kappa"])
    M[..., 0, 1] = off
    M[..., 1, 0] = off
    if stab:
        from fftpf import timestep_adaptive as adaptive
        stab = adaptive.check_stab(stab)
        for i in range(3):
            M[..., i, i] += 2 * dt * L * stab
        return adaptive.tag_operator(np.linalg.inv(M), stab)
    return np.linalg.inv(M)


def step_sbdf2(P_n, P_nm1, alpha, ops, Minv2, dt, L, E_ext, elastic_params=None, stab=0.0):
    """One SBDF2 step (see module-section comment above for the derivation).
    N is evaluated at the explicit 2nd-order extrapolation P* = 2P^n - P^(n-1).

    stab=S > 0: S*P is implicit at n+1 and explicit at the SAME extrapolant
    P* (as N is): RHS gains + 2*dt*L*S*FFT(P*); Minv2 must come from
    build_Minv2(..., stab=S). stab=0 is the original arithmetic, bit for bit."""
    if stab:
        from fftpf import timestep_adaptive as adaptive
        Minv2 = adaptive.stabilized_operator(Minv2, stab, "solver.step_sbdf2")
    elif getattr(Minv2, "stab", 0.0):
        raise ValueError("solver.step_sbdf2: Minv2 was built with stab > 0; pass the same stab")
    if _FUSED is not None:
        P_new = _FUSED.periodic_step("sbdf2", P_n, P_nm1, alpha, ops, Minv2, dt, L, E_ext,
                                     elastic_params, stab, FFT_WORKERS, 2)
        if P_new is not None:
            return P_new
    P_star = 2 * P_n - P_nm1
    N = _explicit_N(P_star, alpha, ops, E_ext, elastic_params)
    s = (ops["Ny"], ops["Nx"])
    Pk_n = _rfft2(P_n)
    Pk_nm1 = _rfft2(P_nm1)
    Nk = _rfft2(N)
    RHS = 4 * Pk_n - Pk_nm1 - 2 * dt * L * Nk           # (3,[S,]Ny,Nxr)
    if stab:
        RHS = RHS + (2 * dt * L * stab) * (2 * Pk_n - Pk_nm1)
    # replaces np.moveaxis(np.einsum("...ij,...j->...i", Minv2,
    #   np.moveaxis(RHS,0,-1)), -1, 0) -- SPEC §39.1 A5. Identical form to
    # step()'s Minv apply (see its comment for the ensemble-broadcast
    # argument, which holds here unchanged): ONE idiom for the per-mode 3x3
    # apply everywhere in the tree.
    Pk_new = np.empty_like(RHS)
    Pk_new[0] = Minv2[..., 0, 0] * RHS[0] + Minv2[..., 0, 1] * RHS[1] + Minv2[..., 0, 2] * RHS[2]
    Pk_new[1] = Minv2[..., 1, 0] * RHS[0] + Minv2[..., 1, 1] * RHS[1] + Minv2[..., 1, 2] * RHS[2]
    Pk_new[2] = Minv2[..., 2, 0] * RHS[0] + Minv2[..., 2, 1] * RHS[1] + Minv2[..., 2, 2] * RHS[2]
    return _irfft2(Pk_new, s)


def _probe_run(P0, alpha, ops, grid, L, E_ext, elastic_params, dt, n_probe, p_cap, scheme):
    """Run n_probe steps of `scheme` from P0; return True if it stayed finite
    and within p_cap throughout (shared by probe_stable_dt's sbdf1/sbdf2 paths)."""
    Minv = build_Minv(grid, ops, dt, L)
    if scheme == "sbdf1":
        P = P0.copy()
        for _ in range(n_probe):
            P = step(P, alpha, ops, Minv, dt, L, E_ext, elastic_params=elastic_params)
            if not np.all(np.isfinite(P)) or np.max(np.abs(P)) > p_cap:
                return False
        return True
    elif scheme == "sbdf2":
        Minv2 = build_Minv2(grid, ops, dt, L)
        P_nm1 = P0.copy()
        P_n = step(P_nm1, alpha, ops, Minv, dt, L, E_ext, elastic_params=elastic_params)  # bootstrap
        if not np.all(np.isfinite(P_n)) or np.max(np.abs(P_n)) > p_cap:
            return False
        for _ in range(n_probe - 1):
            P_new = step_sbdf2(P_n, P_nm1, alpha, ops, Minv2, dt, L, E_ext, elastic_params=elastic_params)
            if not np.all(np.isfinite(P_new)) or np.max(np.abs(P_new)) > p_cap:
                return False
            P_nm1, P_n = P_n, P_new
        return True
    raise ValueError(f"unknown scheme {scheme!r}, expected 'sbdf1' or 'sbdf2'")


def probe_stable_dt(P0, alpha, ops, grid, L, E_ext=(0., 0., 0.), elastic_params=None,
                     dt0=0.02, n_probe=200, growth=2.0, p_cap=None, max_doublings=30,
                     scheme="sbdf1"):
    """SPEC §2 Δt protocol: start dt0, run n_probe steps, require finite F and
    no growth of max|P| beyond a physical cap; double dt until failure; return
    half the largest stable dt. scheme="sbdf2" (WP4 §10c) probes the 2nd-order
    stepper instead (bootstrapped the same way relax()'s sbdf2 path is).

    p_cap=None => max(10*max|P0|, 5.0): a blow-up detector, not a physics
    bound -- 10x headroom over the seed so an ordinary growth-phase excursion
    is not mistaken for divergence, floored at 5.0 so a near-zero P0 (the
    usual small-noise seed) does not cap at ~0. Documented headroom choice,
    not a SPEC number, same status as relax()'s dt_min_frac below."""
    if p_cap is None:
        p_cap = max(10.0 * np.max(np.abs(P0)), 5.0)
    dt = dt0
    last_stable = None
    for _ in range(max_doublings):
        if not _probe_run(P0, alpha, ops, grid, L, E_ext, elastic_params, dt, n_probe, p_cap, scheme):
            break
        last_stable = dt
        dt *= growth
    if last_stable is None:
        raise RuntimeError(f"probe_stable_dt: dt0={dt0} already unstable ({n_probe} steps, scheme={scheme})")
    return last_stable / 2.0


def relax(P0, alpha, ops, grid, L, E_ext=(0., 0., 0.), elastic_params=None, dt=None, tol_F=1e-8,
          n_max=20000, check_every=10, n_plateau=20, dt0=0.02, n_probe=200, post_step=None,
          scheme="sbdf1", dt_adapt=False, tol_P=None, dt_grow=1.2, dt_cut=2.0,
          dt_min_frac=1e-3, dt_max_frac=32.0):
    """Run the stepper to convergence. Termination: |dF/dt| < tol_F for
    n_plateau consecutive checks (every check_every steps), or n_max steps.
    post_step(P) -> P, if given, is applied after every step (WP2 G4 uses this
    to enforce the Py=Pz Ising/SPP-branch tie, mirroring gld1d.py's own `tie`).
    scheme="sbdf1" (default, SPEC §2, byte-identical to the pre-WP4 behavior)
    or "sbdf2" (SPEC WP4 §10c: bootstrap step 1 with sbdf1, then loop the
    2-level SBDF2 update).

    WP4 §10(d), both opt-in (default off => byte-identical to pre-§10d
    behavior for every existing caller):
      dt_adapt=True: after each check window, grow dt by dt_grow (default
        x1.2) if F dropped smoothly (dF<0), cut it by /dt_cut (default x2) if
        dF>0 (SPEC's grow/cut rule). dt is clamped to
        [dt0_used*dt_min_frac, dt0_used*dt_max_frac] where dt0_used is
        whatever dt this call started with (explicit `dt` or probe_stable_dt's
        result) -- a documented headroom choice, not a SPEC number.
      tol_P (not None): plateau now requires BOTH |dF/dt|<tol_F AND
        max|dP/dt|<tol_P (P-change since the last check / elapsed time) --
        SPEC: "kills the growth-phase false plateau" (near a tiny-P unstable
        seed, |dF/dt| ~ P^2 is quadratically suppressed while max|dP/dt| ~ P
        stays a reliable still-evolving signal; see gates/g7d for the
        constructed test).
    Returns (P_final, dt_used, n_steps_run, history=[(step,F),...]) -- with
    dt_adapt, dt_used is the LAST dt in force (history logs (step,F) only,
    not the dt trace; g7d prints it separately if needed).

    SPEC V2-02.1 E11 (2026-09-16): dt_adapt with scheme="sbdf2" raises.
    SBDF2 is a CONSTANT-COEFFICIENT 2-level recurrence -- its stencil
    weights and Minv2 are derived for ONE fixed dt, so changing dt inside
    the loop silently evaluates a scheme nobody wrote down. Use a fixed dt,
    or adaptive SBDF1."""
    if scheme == "sbdf2" and dt_adapt:
        raise ValueError("adaptive SBDF2 is unsupported: use fixed dt or adaptive SBDF1")
    if dt is None:
        dt = probe_stable_dt(P0, alpha, ops, grid, L, E_ext=E_ext, elastic_params=elastic_params,
                              dt0=dt0, n_probe=n_probe, scheme=scheme)
    dt_min = dt * dt_min_frac
    dt_max = dt * dt_max_frac

    cur = dict(dt=dt, Minv=build_Minv(grid, ops, dt, L),
               Minv2=build_Minv2(grid, ops, dt, L) if scheme == "sbdf2" else None)

    def _rebuild(new_dt):
        new_dt = min(max(new_dt, dt_min), dt_max)
        if new_dt == cur["dt"]:
            return
        cur["dt"] = new_dt
        cur["Minv"] = build_Minv(grid, ops, new_dt, L)
        if scheme == "sbdf2":
            cur["Minv2"] = build_Minv2(grid, ops, new_dt, L)

    hist = []
    F_prev_box = [None]
    P_prev_check_box = [P0.copy()]
    plateau_box = [0]

    def _record(n, P):
        # _record fires exactly every check_every steps (the `n % check_every
        # == 0` guard at each call site), and dt only changes here (at the
        # END, via _rebuild), so cur["dt"] was constant over this whole
        # interval -- elapsed = check_every*cur["dt"] exactly (same
        # denominator the pre-§10d code used, just via cur["dt"]).
        elapsed = check_every * cur["dt"]
        _, tot = total_energy(P, alpha, ops, E_ext, grid, elastic_params=elastic_params)
        F = tot["total"]
        hist.append((n, F))
        stop = False
        if F_prev_box[0] is not None:
            dF = F - F_prev_box[0]
            f_ok = abs(dF) / elapsed < tol_F
            p_ok = True
            if tol_P is not None:
                dPdt = float(np.max(np.abs(P - P_prev_check_box[0]))) / elapsed
                p_ok = dPdt < tol_P
            if f_ok and p_ok:
                plateau_box[0] += 1
                if plateau_box[0] >= n_plateau:
                    stop = True
            else:
                plateau_box[0] = 0
            if dt_adapt:
                _rebuild(cur["dt"] * dt_grow if dF < 0 else cur["dt"] / dt_cut)
        F_prev_box[0] = F
        P_prev_check_box[0] = P.copy()
        return stop

    if scheme == "sbdf1":
        P = P0.copy()
        n_run = 0
        for n in range(1, n_max + 1):
            P_new = step(P, alpha, ops, cur["Minv"], cur["dt"], L, E_ext, elastic_params=elastic_params)
            if not np.all(np.isfinite(P_new)):
                print(f"[solver.relax] WARNING: non-finite P at step {n}; stopping, "
                      f"returning last finite state (step {n_run}).")
                return P, cur["dt"], n_run, hist
            P = P_new
            if post_step is not None:
                P = post_step(P)
            n_run = n
            if n % check_every == 0 and _record(n, P):
                break
        return P, cur["dt"], n_run, hist

    elif scheme == "sbdf2":
        P_nm1 = P0.copy()
        P_n = step(P_nm1, alpha, ops, cur["Minv"], cur["dt"], L, E_ext, elastic_params=elastic_params)  # bootstrap
        if not np.all(np.isfinite(P_n)):
            print("[solver.relax] WARNING: non-finite P at SBDF2 bootstrap step 1.")
            return P_nm1, cur["dt"], 0, hist
        if post_step is not None:
            P_n = post_step(P_n)
        n_run = 1
        if n_run % check_every == 0 and _record(n_run, P_n):
            return P_n, cur["dt"], n_run, hist
        for n in range(2, n_max + 1):
            P_new = step_sbdf2(P_n, P_nm1, alpha, ops, cur["Minv2"], cur["dt"], L, E_ext,
                                elastic_params=elastic_params)
            if not np.all(np.isfinite(P_new)):
                print(f"[solver.relax] WARNING: non-finite P at step {n}; stopping, "
                      f"returning last finite state (step {n_run}).")
                return P_n, cur["dt"], n_run, hist
            if post_step is not None:
                P_new = post_step(P_new)
            P_nm1, P_n = P_n, P_new
            n_run = n
            if n % check_every == 0 and _record(n, P_n):
                break
        return P_n, cur["dt"], n_run, hist

    raise ValueError(f"unknown scheme {scheme!r}, expected 'sbdf1' or 'sbdf2'")


# --------------------------------------------------------------------------
# WP4 §10(e): ensemble batching. Every field-shaped function above (bulk_
# force, grad_force, elec_Ed/elec_force, step/step_sbdf2, and elastic.py's
# elastic_solve/elastic_force) already accepts an OPTIONAL ensemble axis
# with ZERO code changes beyond the einsum-ellipsis fix WP4 made, PROVIDED
# it is inserted as axis 1 -- shape (3,S,Ny,Nx) -- i.e. right after the
# existing component axis 0, not before it: every P[0]/P[1]/P[2]-style
# indexing already scattered through this module and elastic.py then keeps
# meaning "component c, ALL realizations, ALL space" with no changes, and
# every FFT (axes=(-2,-1)) and every elementwise/broadcast operator still
# targets exactly the axes it always did. SPEC's own prose lists the shape
# as "(S,3,Ny,Nx)" (ensemble axis first); these two converters make that the
# supported INTERCHANGE format without forcing the internal axis order (a
# documented, deliberate deviation -- see WP4_REPORT.md -- chosen because
# rewriting every component index to the S-first convention was a much
# larger, higher-risk change for the same physics).
# WP28 P6/P7/P7c note: step(), step_sbdf2() and elastic._green_delta_eps no
# longer HAVE those einsums -- their explicit per-component sums broadcast
# the (Ny,Nxr) operator against an (S,Ny,Nxr) spectrum by numpy's own
# left-padding rule, which is what the ellipsis was buying. G7e is unchanged
# and still the anchor (step_sbdf2's own ensemble path is additionally
# covered batched-vs-serial by gates/wp28_witness/wp28_p7c_witness.py).
# The left-padding rule does NOT help a P-independent per-realization FIELD
# -- a field-shaped E_ext (3,Ny,Nx) or eps_star_static (3,3,Ny,Nx): padded
# against (3,S,Ny,Nx) it lines up with the component axes, not with space.
# _reshape_vec3(..., ensemble=True) and elastic.elastic_solve therefore
# insert S into such a field (elastic._ensemble_field), accept an already
# (...,S,Ny,Nx) one, and raise on anything else.
# --------------------------------------------------------------------------
def ensemble_to_internal(P):
    """(S,3,Ny,Nx) -> (3,S,Ny,Nx): SPEC's listed axis order -> this engine's."""
    return np.moveaxis(P, 0, 1)


def ensemble_from_internal(P):
    """(3,S,Ny,Nx) -> (S,3,Ny,Nx): inverse of ensemble_to_internal."""
    return np.moveaxis(P, 1, 0)


# --------------------------------------------------------------------------
# IC / test-field helpers
# --------------------------------------------------------------------------
def random_smooth_field(grid, n_comp, amplitude, kmax_frac=0.3, seed=0):
    """n_comp independent real fields, band-limited to |k| < kmax_frac*max|k|,
    each rescaled to the given amplitude (std). kmax_frac=1.0 => full-band noise."""
    rng = np.random.default_rng(seed)
    KX, KY = grid["KX"], grid["KY"]
    kmag = np.sqrt(KX**2 + KY**2)
    kmax = kmax_frac * kmag.max()
    out = np.zeros((n_comp, grid["Ny"], grid["Nx"]))
    for c in range(n_comp):
        raw = rng.standard_normal((grid["Ny"], grid["Nx"]))
        rawk = np.fft.fft2(raw)
        rawk[kmag > kmax] = 0.0
        field = np.fft.ifft2(rawk).real
        std = field.std()
        if std > 0:
            field = field / std
        out[c] = amplitude * field
    return out


# --------------------------------------------------------------------------
# WP4 §10(f): shared-realization resolution machinery. Ideal spectral
# resampling (down OR up) of a real periodic 2D field to a NEW resolution,
# SAME physical box -- used both by the disorder generator's master-grid
# mode (cases/disorder.py's resample_strain_dict, so 128^2/256^2/512^2 all
# see the SAME physical disorder field, per SPEC: "never regenerate per N:
# same seed at different N is a different realization") and by
# upsample_state (SPEC: "Fourier zero-pad upsample restart for states
# (N->2N)", item (h)'s 128^2->256^2->512^2 relaxation chain).
# --------------------------------------------------------------------------
def spectral_resample_2d(field2d, Ny_new, Nx_new):
    """fftshift to center DC, crop or zero-pad the LOWEST |k|
    min(Ny,Ny_new) x min(Nx,Nx_new) block (symmetric about the center),
    ifftshift, inverse transform. Unnormalized-FFT convention means the
    result must be rescaled by (Ny_new*Nx_new)/(Ny*Nx) to preserve
    amplitude (the borrowed spectrum was produced by a size-(Ny,Nx) forward
    transform but is now being fed to a size-(Ny_new,Nx_new) inverse one).
    Exact (up to float roundoff) for a field already band-limited to the
    smaller of the two resolutions -- e.g. downsample-then-upsample back to
    the original N (gate G7f's round trip).

    ALIGNMENT: after fftshift, k=0 sits at index N//2 for EITHER parity
    (k = i - N//2), so the block of the n = min(n_src, n_dst) lowest modes
    starts at n_src//2 - n//2 in the source and n_dst//2 - n//2 in the
    destination. That equals the old (n_src - n)//2, (n_dst - n)//2 when the
    two sizes have the SAME parity (every same-parity resample is
    byte-identical to before); for mixed parity (63<->128, 7<->8) the old
    offsets shifted every mode by one bin, the mean included.

    NYQUIST: downsampling to an EVEN n keeps the coarse bin -n/2, which on
    the coarse grid is the same mode as +n/2 -- so the source's +n/2 bin is
    folded into it (per axis; both at a corner). Without the fold that mode
    came back halved, e.g. (-1)^x through 8->16->8 returned 0.5*(-1)^x.
    Upsampling keeps the old ONE-sided placement of an even source's
    Nyquist bin at the fine -n/2: the .real taken below already makes that
    exact, and keeps upsampling byte-identical."""
    Ny, Nx = field2d.shape
    Fk = np.fft.fftshift(np.fft.fft2(field2d))
    Gk = np.zeros((Ny_new, Nx_new), dtype=complex)

    def _slices(n_src, n_dst):
        n = min(n_src, n_dst)
        s0 = n_src // 2 - n // 2
        d0 = n_dst // 2 - n // 2
        return slice(s0, s0 + n), slice(d0, d0 + n)

    ys, yd = _slices(Ny, Ny_new)
    xs, xd = _slices(Nx, Nx_new)
    Gk[yd, xd] = Fk[ys, xs]
    # Downsampling to an even size: fold the source's +n/2 line (the index
    # just past the copied block) into the coarse -n/2 line (index 0).
    fold_y = Ny_new < Ny and Ny_new % 2 == 0
    fold_x = Nx_new < Nx and Nx_new % 2 == 0
    if fold_y:
        Gk[0, xd] += Fk[ys.stop, xs]
    if fold_x:
        Gk[yd, 0] += Fk[ys, xs.stop]
    if fold_y and fold_x:
        Gk[0, 0] += Fk[ys.stop, xs.stop]
    scale = (Ny_new * Nx_new) / (Ny * Nx)
    return np.fft.ifft2(np.fft.ifftshift(Gk)).real * scale


def upsample_state(P, Ny_new, Nx_new):
    """P shape (3,Ny,Nx) -> (3,Ny_new,Nx_new): each component independently
    spectrally resampled (works for either direction; named "upsample" for
    SPEC's N->2N restart use case, SPEC WP4 §10f)."""
    return np.stack([spectral_resample_2d(P[c], Ny_new, Nx_new) for c in range(P.shape[0])])


# --------------------------------------------------------------------------
# WP12 (SPEC §18): R1 -- Langevin thermal noise + exact discrete FDT.
# dP = -L(dF/dP)dt + sqrt(2*L*kT)*dW(r,t). Splitting: the k-linear family
# Lin(k)=A(k)+D(k)/kappa (build_Minv's own operator, minus the I+dt*L*
# wrapper) advances by an EXACT per-mode Ornstein-Uhlenbeck update (decay
# exp(-Lambda*dt) + noise, Lambda=L*eigenvalues(Lin(k))); the nonlinear
# force (bulk [+elastic]) enters as an explicit Euler drift ONLY -- all
# stochastic forcing lives in the linear substep (the noise term in the
# original SDE is additive and does not care how the deterministic drift is
# split, so attributing it entirely to the sub-step with a closed-form exact
# solution avoids a second, redundant noise injection). First-order
# stochastic splitting only (SPEC: SBDF2-style 2nd order is OUT of scope).
#
# This REPLACES backward-Euler (I+dt*L*Lin(k))^-1 with an exact matrix
# exponential for the linear-family decay -- a genuinely different (more
# accurate) propagator than build_Minv's, not a reuse of it; "reuse/extend
# the M^-1 precompute machinery" (SPEC) means reusing the PATTERN (batched
# per-k 3x3 dense linear algebra via einsum, precomputed once per (dt,L,kT)
# and applied every step), not the M^-1 formula itself -- confirmed by
# SPEC's own "built from the symmetric 3x3 eigendecomposition", which a
# matrix INVERSE (build_Minv's np.linalg.inv) never needs but an exact
# exponential does. Consequently "G16c kT->0 regression: reproduces the
# deterministic stepper" is read as internal self-consistency of THIS new
# integrator (turning off noise cleanly reduces to its own deterministic,
# exponential-decay skeleton, cross-checked against an independently-coded
# reference in the gate) -- NOT bit-identical to the pre-existing step()
# (backward Euler and exp(-Lambda*dt) are different maps at finite dt by
# construction; requiring 1e-12 agreement between them would be
# mathematically impossible to satisfy, not merely difficult).
# --------------------------------------------------------------------------
def _lin_matrix(ops):
    """Lin(k) = A(k) + D(k)/kappa, RAW (no I, no dt, no L) -- the same 3x3
    symmetric k-space operator build_Minv wraps as I+dt*L*Lin(k), assembled
    standalone here because the OU stepper needs its EIGENdecomposition, not
    its inverse. Deliberately duplicates build_Minv's entries rather than
    sharing a helper -- build_Minv writes 1.0+dt*L*(...) straight into the
    3x3 it inverts, so no raw Lin(k) exists there to reuse.

    SHAPE-GENERIC over the k-grid (WP28 P3, SPEC §38.1 B2 = REVIEW_arch_
    minimality.md F4: this replaces bulk3d._lin_matrix_3d, which was this
    function with all 9 slots written out). Lin has ops["A11"].shape + (3,3),
    so (Ny,Nxr,3,3) for solver.py's 2D ops and (Nz,Ny,Nxr,3,3) for bulk3d's.
    The 3D-only slots are read with ops.get(key, 0.0): a 2D ops dict has no
    A13/A23/D13/D23/D33 (build_operators' own docstring: "A13=A23=0 and
    D33=D13=D23=0 identically (k3==0)"), so those lines add exact 0.0 to
    already-zeroed entries and the 2D result is byte-identical to the WP12
    form. Gates: G16a-e (2D OU), G27a-e + G30d (3D)."""
    Lin = np.zeros(ops["A11"].shape + (3, 3))
    Lin[..., 0, 0] = ops["A11"] + ops["D11"] / ops["kappa"]
    Lin[..., 1, 1] = ops["A22"] + ops["D22"] / ops["kappa"]
    Lin[..., 2, 2] = ops["A33"] + ops.get("D33", 0.0) / ops["kappa"]
    off12 = ops["A12"] + ops["D12"] / ops["kappa"]
    off13 = ops.get("A13", 0.0) + ops.get("D13", 0.0) / ops["kappa"]
    off23 = ops.get("A23", 0.0) + ops.get("D23", 0.0) / ops["kappa"]
    Lin[..., 0, 1] = Lin[..., 1, 0] = off12
    Lin[..., 0, 2] = Lin[..., 2, 0] = off13
    Lin[..., 1, 2] = Lin[..., 2, 1] = off23
    return Lin


def _component_first_complex(M):
    """(...,3,3) real per-mode operator -> (3,3,...) contiguous complex128.
    This is the layout the explicit 3x3 applies below multiply the (complex)
    spectra by. COMPLEX, not real, because that is what np.einsum's own
    dtype promotion produced from (float64 operator, complex128 spectrum) --
    matching it is what makes the explicit sums byte-identical to the
    einsums they replace (SPEC §39.1 A5). Costs one extra copy of the
    operator at 2x its bytes (1.2 MB per operator on a 128^2 grid, 2.5 MB
    at 32^3), paid once at setup instead of per step."""
    return np.ascontiguousarray(np.moveaxis(M, (-2, -1), (0, 1))).astype(np.complex128)


def _eig_roundoff_floor(lam):
    """SPEC §10.1(1): the roundoff floor on eigenvalues lam (...,n) of one
    symmetric operator per leading index. Per mode, tol = 1e3*eps64*max_i|lam_i|;
    an entry in [-tol, 0) is set to 0 by the very np.clip(lam, 0.0, None) the
    OU builders used before §10 (so signed zeros come out exactly as they did),
    an entry below -tol is a GENUINE negative eigenvalue and is KEPT. On an
    operator with no entry below -tol the result is byte-identical to that clip
    (K2 `ou_psd_byte`). Shared by _ou_from_lin and film3d_langevin._z_eigen."""
    tol = 1e3 * np.finfo(np.float64).eps * np.max(np.abs(lam), axis=-1, keepdims=True)
    return np.where(lam < -tol, lam, np.clip(lam, 0.0, None))


def _ou_from_lin(Lin, dt, L, kT):
    """The exact per-mode OU propagator OF a raw Lin(k) -- extracted from
    build_ou_operators (WP28 P3, SPEC §38.1 B2) because bulk3d.
    build_ou_operators_3d's body was these same 10 lines. See
    build_ou_operators' docstring for the formulas, the eigh/real-symmetric
    argument and the returned-key meanings; NOTHING here is grid-shape aware
    (np.linalg.eigh batches over whatever leading axes Lin carries), so the
    2D and 3D engines share it byte-for-byte.

    NEGATIVE EIGENVALUES (SPEC §10). Lin(k) is NOT PSD in general: on the H11
    lane (grad4, G < 0 < H11) it is genuinely negative near k*. Such a lambda
    is kept and the exact formulas below propagate it unchanged -- decay
    exp(-Lambda*dt) > 1 (the mode grows), var = 2*kT*L*dt*phi1(2*Lambda*dt) > 0
    (_phi1 is valid for negative arguments). Only roundoff-negative values are
    floored to 0 (_eig_roundoff_floor). Before §10 this line was
    np.clip(lam, 0.0, None), which made every lambda < 0 mode NEUTRAL: Langevin
    runs on G < 0 ops evolved without the k* instability."""
    lam, V = np.linalg.eigh(Lin)              # lam:(...,3) ascending; V:(...,3,3), V[...,:,i] = i-th eigvec
    lam = _eig_roundoff_floor(lam)             # SPEC §10: floor roundoff-negatives, keep genuine lambda < 0
    Lambda = L * lam
    decay_eig = np.exp(-Lambda * dt)
    var_eig = 2.0 * kT * L * dt * _phi1(2.0 * Lambda * dt)
    std_eig = np.sqrt(var_eig)
    decay = np.einsum("...ij,...j,...kj->...ik", V, decay_eig, V)
    noise_amp = np.einsum("...ij,...j,...kj->...ik", V, std_eig, V)
    return dict(decay=decay, noise_amp=noise_amp, lam=lam,
                decay_cf=_component_first_complex(decay),          # WP28 P7, SPEC §39.1 A5
                noise_amp_cf=_component_first_complex(noise_amp))  # (same numbers, step layout)


def _noise_cutoff_mask(k2, noise_cutoff_nm):
    """The Hohenberg-Halperin Lambda made explicit (SPEC V2-04.0, D16): 1.0 on
    every mode with |k| < 2*pi/l_c, 0.0 on every mode with |k| >= 2*pi/l_c.
    `k2` is the engine's OWN kx^2+ky^2(+kz^2) on the same spectral wavevector
    arrays build_operators was given, so the comparison is made on SQUARES
    (k2 >= (2*pi/l_c)^2) and never takes a sqrt. Shape-generic: the 2D
    (Ny,Nxr), 3D (Nz,Ny,Nxr) and film (Nz,Ny,Nxr) grids all go through it."""
    lc = float(noise_cutoff_nm)
    if not (lc > 0.0):
        raise ValueError(f"noise_cutoff_nm must be a positive length in nm, got {lc!r}")
    k_cut = 2.0 * np.pi / lc
    return (np.asarray(k2) < k_cut * k_cut).astype(float)


def _apply_noise_cutoff(ou, k2, noise_cutoff_nm):
    """Zero the STORED noise amplitude on the excluded modes -- decay, lam and
    every other key untouched, so the step functions do not change at all and
    the excluded modes simply relax deterministically. noise_amp_cf is REBUILT
    from the masked noise_amp (not masked separately), which is what keeps the
    redundant-by-construction pair in step, exactly as _ou_from_lin builds it.
    Three provenance keys are added so a gate can print what was kept; they
    appear ONLY when a cutoff was asked for, so noise_cutoff_nm=None leaves the
    OU dict bit-identical to the no-kwarg call (K2 `noise_cutoff_none_byte`)."""
    keep = _noise_cutoff_mask(k2, noise_cutoff_nm)
    ou["noise_amp"] = ou["noise_amp"] * keep[..., None, None]
    ou["noise_amp_cf"] = _component_first_complex(ou["noise_amp"])
    ou["noise_cutoff_nm"] = float(noise_cutoff_nm)
    ou["noise_k_cut"] = 2.0 * np.pi / float(noise_cutoff_nm)
    ou["noise_modes_kept"] = int(keep.sum())
    ou["noise_modes_total"] = int(keep.size)
    return ou


def _phi1(x):
    """ETD "phi_1" function (1-exp(-x))/x, stable at x=0 (->1.0), array-safe
    (np.where evaluates both branches so `x==0` never reaches a 0/0 division:
    the `small` branch's Taylor series is used there instead). Writing the
    OU noise variance as 2*kT*L*dt*phi1(2*Lambda*dt) instead of the
    algebraically-equal kT/lambda*(1-exp(-2*Lambda*dt)) cancels the
    1/lambda singularity SYMBOLICALLY before evaluation, so lambda=0 (the
    k=0 mode, where A(k)=D(k)=0 identically -- no gradient/depol restoring
    force for the uniform mode) needs no separate branch anywhere else."""
    small = np.abs(x) < 1e-8
    xs = np.where(small, 1.0, x)
    reg = -np.expm1(-xs) / xs
    ser = 1.0 - x / 2.0 + x * x / 6.0
    return np.where(small, ser, reg)


def build_ou_operators(ops, dt, L, kT, noise_cutoff_nm=None, grid=None):
    """Precompute the exact per-mode OU propagator for Lin(k)=A(k)+D(k)/kappa
    (SPEC §18): decay(k) = V(k) diag(exp(-Lambda_i*dt)) V(k)^T, noise_amp(k)
    = V(k) diag(std_i) V(k)^T, std_i = sqrt(2*kT*L*dt*phi1(2*Lambda_i*dt)),
    Lambda_i=L*lambda_i(k), (lambda_i(k),V(k)) = eigh(Lin(k)).

    kT IS AN ENERGY DENSITY: units.kT_langevin(T, hx, hy, Dz) (SPEC V2-04.0,
    D16). The stationary law of the returned propagator, in step_langevin's
    own UNNORMALISED rfft2 mode variable, is <|m_i(k)|^2> = N_cells*kT/chi_i(k)
    with chi = lambda_i(k) + the bulk curvature. NOT units.si_to_deck_kT.

    NOISE CUTOFF (optional, SPEC V2-04.0 D16). noise_cutoff_nm = l_c sets the
    stored noise amplitude to zero on every mode with |k| >= 2*pi/l_c -- the
    Hohenberg-Halperin Lambda made explicit. The DECAY is untouched, so the
    excluded modes relax deterministically while the retained ones are exactly
    Boltzmann at T; l_c = None is the full discrete FDT and returns an OU dict
    bit-identical to the pre-V2-04 one. `grid` supplies |k|^2 = KXr^2 + KYr^2
    -- the very arrays build_operators itself read -- and is REQUIRED when
    noise_cutoff_nm is given (the ops dict does not carry the wavevectors).
    step_langevin is unchanged: the whole cutoff lives in noise_amp.

    Lin(k) is REAL symmetric (real entries at real k1,k2 -- ordinary
    quadratic-in-k combinations, same as elastic.py's Omega(k)), so eigh
    gives real eigenvalues/orthogonal eigenvectors directly: NO Wirtinger
    subtlety here. (WP7B's Hermitian-vs-symmetric lesson concerned a
    DIFFERENT construction route -- a formal real-symbol Hessian substituted
    with complex Fourier dof AFTER the fact, which silently drops the
    conjugation a per-mode COMPLEX sesquilinear form needs; Lin(k) here, like
    elastic.py's Omega(k), is built the trap-free way, "each derivative's i
    applied once, directly in the strong-form PDE", so it is an ordinary
    real matrix of real numbers, not a complex per-mode Hessian.)

    INDEFINITE Lin (SPEC §10). lambda_i(k) < 0 -- G < 0 with an H11
    regulariser, grad4.build_operators_hog -- is propagated exactly: that
    mode GROWS by exp(-Lambda*dt) per step and its noise variance stays
    positive. Only roundoff-negative eigenvalues are floored to 0. Before §10
    every lambda < 0 was clipped to 0, so G < 0 Langevin runs lost the k*
    instability (the deterministic build_Minv/step path never clipped).

    Batches over the whole (Ny,Nxr) k-grid via np.linalg.eigh's native
    trailing-(...,3,3) batching (no explicit loop), same "generic per-k 3x3
    dense linear algebra" idiom build_Minv uses for np.linalg.inv.
    RETURNED-KEY CONTRACT: decay=(Ny,Nxr,3,3), noise_amp=(Ny,Nxr,3,3),
    lam=(Ny,Nxr,3) eigenvalues (kept for gates that need the analytic
    stiffness directly), plus decay_cf/noise_amp_cf = (3,3,Ny,Nxr) complex
    copies of the same two operators in the layout step_langevin's explicit
    applies use (WP28 P7, SPEC §39.1 A5; see _component_first_complex). The
    _cf pair is redundant-by-construction, never read by a gate, and must
    stay in step with decay/noise_amp -- which it does, being built from
    them in _ou_from_lin.
    The two lines below are the whole function: _lin_matrix and _ou_from_lin
    are both shape-generic, and bulk3d.build_ou_operators_3d is the same two
    lines on a 3D ops dict (WP28 P3, SPEC §38.1 B2)."""
    Lin = _lin_matrix(ops)
    ou = _ou_from_lin(Lin, dt, L, kT)
    if noise_cutoff_nm is None:
        return ou
    if grid is None:
        raise ValueError("build_ou_operators: noise_cutoff_nm needs grid= "
                         "(the KXr/KYr arrays the operators were built from)")
    return _apply_noise_cutoff(ou, operators.k_squared(grid, 2), noise_cutoff_nm)


def step_langevin(P, alpha, ops, ou, dt, L, E_ext, rng, elastic_params=None):
    """One first-order stochastic-splitting step (SPEC §18). Nonlinear force
    (bulk [+elastic]) via `_explicit_N`, applied as an explicit Euler drift
    (no noise of its own); then the exact per-mode OU update (`ou`, from
    build_ou_operators) advances the k-linear family, decay + noise.

    NOISE NORMALISATION (SPEC V2-04.0, ruling D16, 2026-09-16 -- a BEHAVIOR
    change). The rfft2 of the real-space draw is used AS IS: the
    `/sqrt(Ny*Nx)` this line used to carry is gone, exactly as
    film3d_langevin.step_langevin_film never had one. `kT` is therefore an
    energy DENSITY, units.kT_langevin(T, hx, hy, Dz) = k_B*T*1e18/V_cell, and
    the per-mode stationary law in this unnormalised mode variable is
    <|m_i(k)|^2> = N_cells*kT/chi_i(k) (normalise the mode by sqrt(N_cells)
    and it is the film's own kT/chi). With the old pairing --
    units.si_to_deck_kT into this stepper -- the physical temperature actually
    sampled was T*V_cell/N_cells, a grid-dependent number with no continuum
    limit; K5 `kt_convention` now gates the absolute SI temperature, and
    REVIEW_v2_03c_kt.md carries the forensics. An optional noise cutoff lives
    in build_ou_operators' noise_amp, so nothing here knows about it.

    Hermitian noise discipline (SPEC): ONE array of standard normals is
    drawn per step from a SINGLE caller-supplied numpy Generator (`rng`) --
    "single-process full-array draws...(seed,step) bitwise reproducible" --
    then rfft2'd. This makes the k-space noise EXACTLY Hermitian-consistent
    (correct conjugate-pair/self-conjugate variance split, the "rfft weight
    lesson") FOR FREE: rfft2 of a genuine i.i.d. real field automatically
    has total variance Nx*Ny at EVERY stored k (split 50/50 between real/
    imag at a generic k; entirely real, i.e. imag=0, at a self-conjugate k
    -- DC, and the Nyquist row/col on even grids), a Parseval/orthogonality
    fact verified numerically (not just assumed) at implementation time --
    see WP12_REPORT.md. No per-mode conjugate-pairing bookkeeping is
    hand-coded anywhere, sidestepping exactly the class of bug WP7b's
    Hermitian lesson warns about (a hand-built per-mode construction that is
    symmetric but not Hermitian, or here: not correctly variance-weighted)."""
    if _FUSED is not None:
        P_new = _FUSED.periodic_step("langevin", P, None, alpha, ops, ou, dt, L, E_ext,
                                     elastic_params, 0.0, FFT_WORKERS, 2, rng=rng,
                                     noise_shape=(3,) + P.shape[1:])
        if P_new is not None:
            return P_new
    N = _explicit_N(P, alpha, ops, E_ext, elastic_params)
    s = (ops["Ny"], ops["Nx"])
    Pk = _rfft2(P)
    Nk = _rfft2(N)
    Pk_star = Pk - dt * L * Nk                             # (3,Ny,Nxr) explicit nonlinear drift

    # single full-array draw (SPEC seeding), shaped like P as in bulk3d's
    # step_langevin_3d: an unbatched P draws exactly (3,Ny,Nx) as before
    # (bit-identical); a batched (3,S,Ny,Nx) P draws independent noise per
    # member (S=1 consumes the same numbers in the same order as unbatched).
    # A fixed (3,Ny,Nx) draw used to broadcast onto the S axis (S=3: member s
    # got noise component s on every component; S=1: output grew a dimension).
    eta = rng.standard_normal((3,) + P.shape[1:])
    eta_k = _rfft2(eta)                                     # NO 1/sqrt(N): kT is a density (V2-04.0)

    dc, nc = ou["decay_cf"], ou["noise_amp_cf"]             # (3,3,Ny,Nxr) complex
    # replaces np.einsum("...ij,...j->...i", ou["decay"], moveaxis(Pk_star,0,-1))
    # -- SPEC §39.1 A5 (0.30 -> 0.06 ms/apply at 128^2)
    decay_term = np.empty_like(Pk_star)
    decay_term[0] = dc[0, 0] * Pk_star[0] + dc[0, 1] * Pk_star[1] + dc[0, 2] * Pk_star[2]
    decay_term[1] = dc[1, 0] * Pk_star[0] + dc[1, 1] * Pk_star[1] + dc[1, 2] * Pk_star[2]
    decay_term[2] = dc[2, 0] * Pk_star[0] + dc[2, 1] * Pk_star[1] + dc[2, 2] * Pk_star[2]
    # replaces np.einsum("...ij,...j->...i", ou["noise_amp"], moveaxis(eta_k,0,-1))
    noise_term = np.empty_like(eta_k)
    noise_term[0] = nc[0, 0] * eta_k[0] + nc[0, 1] * eta_k[1] + nc[0, 2] * eta_k[2]
    noise_term[1] = nc[1, 0] * eta_k[0] + nc[1, 1] * eta_k[1] + nc[1, 2] * eta_k[2]
    noise_term[2] = nc[2, 0] * eta_k[0] + nc[2, 1] * eta_k[1] + nc[2, 2] * eta_k[2]
    Pk_new = decay_term + noise_term                        # (3,Ny,Nxr), same grouping as before
    return _irfft2(Pk_new, s)


def block_stationary(values, n_blocks_check=4, tol=0.1, *, scale=None):
    """New runs must pass an explicit physical `scale` (zero-offset invariant).
    scale=None retains the historical mean-magnitude normalization ONLY for
    old archived protocols. casekit.anneal supplies explicit scales and real
    block means. (SPEC V2-02.1 E15, adopted 2026-09-16: the scale=None path
    is outcome-identical to the pre-V2-02 function -- gates/core/k2_degeneracy
    item `block_stationary_scale_none`.)

    SPEC §18: "dF-plateau criteria are VOID [under noise]; implement
    stationarity detection -- block-averaged observables ... stationary
    across >=4 consecutive blocks within tolerance". `values`: 1D sequence
    of an observable ALREADY block-averaged by the caller (e.g. block-mean
    of F, of <|P|^2>, or of a per-mode variance estimate -- this function is
    observable-agnostic, callers choose the block length and what to
    average). Returns True once the most recent `n_blocks_check` block-means
    mutually agree to within `tol` (relative to their own mean magnitude) --
    a plain, transparent range/mean test, not a trend/slope fit (a noisy
    stochastic trajectory's block-means fluctuate AROUND a constant once
    stationary, they do not smoothly plateau the way a deterministic dF
    does, so a slope-based criterion is the wrong instrument here)."""
    if len(values) < n_blocks_check:
        return False
    recent = np.asarray(values[-n_blocks_check:], dtype=float)
    spread = recent.max() - recent.min()
    if not np.all(np.isfinite(recent)):
        return False
    # Explicit scale is independent of the arbitrary observable zero.
    # None retains legacy behavior for historical cases only.
    scale = np.mean(np.abs(recent)) if scale is None else float(scale)
    if scale < 0 or not np.isfinite(scale):
        raise ValueError("stationarity scale must be finite and nonnegative")
    if scale == 0:
        return spread == 0
    return (spread / scale) <= tol
