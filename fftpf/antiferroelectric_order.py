"""SPEC S23 (WP17/S1): antiferroelectric order parameter q -- new field,
new module. solver.py and elastic.py are NOT modified (SPEC S23.4: "keep
solver.py's stepping contract"; CLAUDE.md Surgical Changes) -- every
function below either (a) REUSES solver.py/elastic.py's existing functions
unchanged (q's own Devonshire polynomial via solver.bulk_energy_density/
bulk_force with a12=a112=a123=0.0; q's eigenstrain via elastic.eigenstrain
fed into elastic.py's EXISTING eps_star_static slot), or (b) is a small,
new, sympy-built addition for the genuinely new physics: the P-q coupling
f_Pq, q's own isotropic gradient f_gradq, and the eigenstrain force's dF/dq
(elastic.py only ever differentiates w.r.t. P).

Binding physics, SPEC S23.1: q(r) is a 3-component AFE order parameter,
Kittel two-sublattice picture P=(Pa+Pb)/2, q=(Pa-Pb)/2. q does NOT source
the depolarization/Poisson field; E couples to q ONLY indirectly through P
via f_Pq (no linear E.q term) -- both load-bearing, gated by G20d.

Deck: params/pbzo_kittel_lum2022.toml, the EXACT transform of Lum, Lim &
Chew, J. Phys.: Condens. Matter 34, 415702 (2022) Table 1 (SPEC S23.5
coordinator ruling). pbzo_kittel_landau_si(T) below is the T-parametrized
SI generator that frozen TOML is one T=298.0 snapshot of -- same
generator/frozen-snapshot relationship literature_params.py already has
between pzt_haun_x(x,T) and params/archive/pzt_Haun_x0.6.toml.

Embedding note (SPEC S23.5(e)): the transform uses a DIAGONAL/component-
decoupled isotropic vector embedding -- each cubic axis independently
reproduces the scalar Kittel transform, summed over axes, giving
alpha12=alpha112=alpha123=0.0 for BOTH sectors (all anisotropy slots
zero-default) and a DIAGONAL coupling menu (sum_i Pi^2*qi^2 etc., not the
norm-product |P|^2*|q|^2). This is the embedding consistent with the
ruling's own worked value t1=3*beta<0 -- a full O(3)-rotational embedding
instead gives t1=beta, t2=2*beta (verified, see WP17_REPORT.md); the
diagonal embedding is what was locked. See gates/g20a's step 0 for the
independent sympy re-derivation that verifies every coefficient below to
1e-12.

GATES: g20a-e (analytic single cell / field thresholds / FD force-energy /
wiring / double hysteresis). All five import this module under its pre-WP24
shim name, `from solvers import afe`.
"""
import os
from fftpf._locations import PARAMS_DIR as _DECK_DIR
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import scipy.fft as spfft
import sympy as sp

from fftpf import pf2d_engine as solver
from fftpf import elastic_base as elastic
from fftpf import spectral_operators as operators
# SPEC V2-03 A3: the q sector is written PER ENGINE (Part II ruling (3): the
# two 3D solvers stay independent, no dimension-generic core), so this module
# now also builds on bulk3d's and film3d's own primitives. Neither imports
# this module, so no cycle; the extra import cost measured 0.040 s.
from fftpf import pf3d_engine as bulk3d
from fftpf import film_engine as film3d
from fftpf import film_elastic
from fftpf.deck_reader import load_params_toml

FFT_WORKERS = -1


def _rfft2(x):
    return spfft.rfft2(x, axes=(-2, -1), workers=FFT_WORKERS)


def _irfft2(xk, s):
    return spfft.irfft2(xk, s=s, axes=(-2, -1), workers=FFT_WORKERS)


# ==========================================================================
# Exact (Pa,Pb) -> (P,q) transform of Lum, Lim & Chew (2022) Table 1
# (SPEC S23.5). Reproduced here as closed-form Python; gates/g20a's step 0
# is the INDEPENDENT sympy re-derivation that verifies these formulas to
# 1e-12 (kept deliberately separate, mirrors gates/g6d_pzt.py's "HAND-CODED
# FRESH BELOW, deliberately NOT calling literature_params" discipline).
# ==========================================================================
PBZO_SI_TABLE1 = dict(alpha0=5e5, beta=-5e8, gamma=8e9, g=1e8, T0=493.0)


def pbzo_kittel_landau_si(T, table1=None):
    """SI dict: P-sector and q-sector (both engine ALPHA_KEYS naming,
    a12=a112=a123=0.0 exactly -- not REQUIRED placeholders, the diagonal
    embedding forces these to zero) and the coupling menu (t1,t2,u1,u2)."""
    p = table1 or PBZO_SI_TABLE1
    alpha0, beta, gamma, g, T0 = p["alpha0"], p["beta"], p["gamma"], p["g"], p["T0"]
    a11 = beta / 2.0
    a111 = gamma / 3.0
    P = dict(a1=alpha0 * (T - T0) + 2 * g, a11=a11, a12=0.0, a111=a111, a112=0.0, a123=0.0)
    q = dict(a1=alpha0 * (T - T0), a11=a11, a12=0.0, a111=a111, a112=0.0, a123=0.0)
    coupling = dict(t1=3 * beta, t2=0.0, u1=5 * gamma, u2=5 * gamma)
    return dict(P=P, q=q, coupling=coupling)


PBZO_TOML = os.path.join(_DECK_DIR, "pbzo_kittel_lum2022.toml")


def load_pbzo_deck(path=None):
    """Load params/pbzo_kittel_lum2022.toml, remapping its collision-safe
    TOML key names (aq1..., coupling_pq's t1..., Gq, Lam11...) onto the
    engine's own ALPHA_KEYS naming for P/q (see solvers/deck_parser.py's
    TOML_SECTIONS addition). The returned deck["Q"] uses the BARE keys
    Q11/Q12/Q44 but holds the deck's Q11_phys/Q12_phys/Q44_phys VALUES (read
    below), i.e. the set elastic.setup_elastic wants -- not the legacy
    /4-hacked block SPEC §1.4 forbids."""
    m = load_params_toml(path or PBZO_TOML)["materials"]
    P = dict(a1=m["alpha1"], a11=m["alpha11"], a12=m["alpha12"],
             a111=m["alpha111"], a112=m["alpha112"], a123=m["alpha123"])
    q = dict(a1=m["aq1"], a11=m["aq11"], a12=m["aq12"],
             a111=m["aq111"], a112=m["aq112"], a123=m["aq123"])
    coupling = dict(t1=m["t1"], t2=m["t2"], u1=m["u1"], u2=m["u2"])
    gradient = dict(G110=m["G110"], G11_G110=m["G11_G110"], G12_G110=m["G12_G110"],
                     G44_G110=m["G44_G110"], G44P_G110=m["G44P_G110"])
    elastic_c = dict(C11=m["C11"], C12=m["C12"], C44=m["C44"])
    Q = dict(Q11=m["Q11_phys"], Q12=m["Q12_phys"], Q44=m["Q44_phys"])
    Lam = dict(Lam11=m["Lam11"], Lam12=m["Lam12"], Lam44=m["Lam44"])
    return dict(P=P, q=q, coupling=coupling, gradient=gradient, Gq=m["Gq"],
                elastic=elastic_c, Q=Q, Lam=Lam, permittivity=m.get("permittivity"))


# ==========================================================================
# f_Pq -- diagonal coupling menu (SPEC S23.5(a)): t1*sum(Pi^2 qi^2) +
# t2*(P.q)^2 + u1*sum(Pi^4 qi^2) + u2*sum(Pi^2 qi^4). Built symbolically
# once at import (solver.py's own _F_BULK/lambdify pattern), lambdified.
# "Force" below means d f/dP (or dq), matching solver.bulk_force's own
# naming convention (its own docstring: "= d f_bulk / d P_i") -- NOT the
# physical -dF/dP; callers add these into N exactly parallel to bulk_force.
# ==========================================================================
_P1s, _P2s, _P3s, _q1s, _q2s, _q3s = sp.symbols("P1 P2 P3 q1 q2 q3", real=True)
_t1s, _t2s, _u1s, _u2s = sp.symbols("t1 t2 u1 u2", real=True)
_Pdotq = _P1s * _q1s + _P2s * _q2s + _P3s * _q3s
_F_COUPLE = (
    _t1s * (_P1s**2 * _q1s**2 + _P2s**2 * _q2s**2 + _P3s**2 * _q3s**2)
    + _t2s * _Pdotq**2
    + _u1s * (_P1s**4 * _q1s**2 + _P2s**4 * _q2s**2 + _P3s**4 * _q3s**2)
    + _u2s * (_P1s**2 * _q1s**4 + _P2s**2 * _q2s**4 + _P3s**2 * _q3s**4)
)
_COUPLE_ARGS = (_P1s, _P2s, _P3s, _q1s, _q2s, _q3s, _t1s, _t2s, _u1s, _u2s)
COUPLE_KEYS = ("t1", "t2", "u1", "u2")
_f_couple_fn = sp.lambdify(_COUPLE_ARGS, _F_COUPLE, "numpy")
_df_couple_dP_fn = [sp.lambdify(_COUPLE_ARGS, sp.diff(_F_COUPLE, v), "numpy") for v in (_P1s, _P2s, _P3s)]
_df_couple_dq_fn = [sp.lambdify(_COUPLE_ARGS, sp.diff(_F_COUPLE, v), "numpy") for v in (_q1s, _q2s, _q3s)]


def couple_energy_density(P, q, coupling):
    """f_couple(P,q) from coupling's t1/t2/u1/u2 keys (SPEC §23.1), real space, shape = P[0].shape."""
    args = [P[0], P[1], P[2], q[0], q[1], q[2]] + [coupling[k] for k in COUPLE_KEYS]
    return _f_couple_fn(*args)


def couple_force_P(P, q, coupling):
    """delta f_couple/delta P_i, shape (3,Ny,Nx)."""
    args = [P[0], P[1], P[2], q[0], q[1], q[2]] + [coupling[k] for k in COUPLE_KEYS]
    return np.stack([fn(*args) for fn in _df_couple_dP_fn])


def couple_force_q(P, q, coupling):
    """delta f_couple/delta q_i, shape (3,Ny,Nx)."""
    args = [P[0], P[1], P[2], q[0], q[1], q[2]] + [coupling[k] for k in COUPLE_KEYS]
    return np.stack([fn(*args) for fn in _df_couple_dq_fn])


# ==========================================================================
# f_gradq -- isotropic gradient (SPEC S23.1 "isotropic first"): SIMPLER
# than solver.py's P tensor (no D(k) depolarization coupling for q --
# binding, SPEC S23.1; no G12/G44/G44p cross terms -- isotropic scope).
# Reuses solver._spectral_grad_xy (solver.py's private spectral-derivative
# helper -- deliberately reused, not duplicated, it is exact and q-agnostic).
# ==========================================================================
def build_ops_q(grid, Gq):
    """Aq(k) = Gq*(kx^2+ky^2), scalar, SAME for all 3 components -- unlike
    solver.build_operators's A(k), no off-diagonal term, no D(k)."""
    return dict(Aq=operators.isotropic_symbol(grid, Gq, 2), Nx=grid["Nx"], Ny=grid["Ny"])


def grad_energy_density_q(q, grid, Gq):
    """f_gradq = 0.5*Gq*|grad q|^2 summed over the 3 components (isotropic, SPEC §23.1), shape (Ny,Nx).

    Even-grid Nyquist convention (solver.grad_energy_density's): collocation
    local-gradient density with REAL skew-adjoint derivatives (each zeroed on
    its own even-axis Nyquist line, solver._modified_wavevectors_2d), plus the
    alternating mode those derivatives drop, added back per even axis j as
    0.5*Gq*(pi/h_j)^2*(NyquistProject_j q_i)^2. Its grid mean is then the
    energy grad_force_q (Aq = Gq*k^2) is the gradient of. On an odd axis
    there is no correction, so odd grids are bit-identical to the raw form."""
    KX, KY = operators.masked_wavevectors_2d(grid)
    q1x, q1y = solver._spectral_grad_xy(q[0], grid, KX, KY)
    q2x, q2y = solver._spectral_grad_xy(q[1], grid, KX, KY)
    q3x, q3y = solver._spectral_grad_xy(q[2], grid, KX, KY)
    fg = 0.5 * Gq * (q1x**2 + q1y**2 + q2x**2 + q2y**2 + q3x**2 + q3y**2)
    return operators.add_nyquist_correction(fg, q, grid, Gq, 2)


def grad_force_q(q, ops_q):
    """d f_gradq / dq = Aq(k) . q (diagonal-only mirror of solver.grad_force)."""
    s = (ops_q["Ny"], ops_q["Nx"])
    qk = _rfft2(q)
    return _irfft2(ops_q["Aq"] * qk, s)


def build_Minv_q(ops_q, dt, Lq):
    """M_q(k) = 1 + dt*Lq*Aq(k), scalar per k-point (isotropic/diagonal --
    no 3x3 batched inverse needed, unlike solver.build_Minv)."""
    return 1.0 + dt * Lq * ops_q["Aq"]


# ==========================================================================
# f_elq -- eigenstrain eps0_ij += Lam_ijkl q_k q_l (SPEC S23.1), SAME
# cubic-symmetric functional form as P's own electrostriction. q's
# contribution is wired into elastic.py's EXISTING eps_star_static slot
# (SPEC S6.2b) -- elastic.py is untouched. dF_el/dq needs a new mirror of
# elastic.elastic_force_explicit (elastic.py only differentiates w.r.t. P).
#
# q's eigenstrain is ADDED to whatever static eigenstrain the base already
# carries (a quenched disorder.make_random_strain field, a Vegard strain),
# never substituted for it. The base's own static is remembered under
# _STATIC0_KEY, so a with_q_eigenstrain* output fed back in as the base
# rebuilds from that original static instead of stacking q's eigenstrain
# twice (no caller in the tree chains like this; the guard keeps a chained
# call equal to the unchained one, as it was when the slot was overwritten).
# ==========================================================================
_STATIC0_KEY = "_eps_star_static_without_q"


def _add_q_static(base, eps_q):
    """Shallow copy of `base` whose eps_star_static is the base's own static
    eigenstrain plus q's `eps_q` (same shape). No base static (None or key
    absent): the slot is `eps_q` itself, bit-for-bit."""
    out = dict(base)
    static0 = base[_STATIC0_KEY] if _STATIC0_KEY in base else base.get("eps_star_static")
    out[_STATIC0_KEY] = static0
    out["eps_star_static"] = eps_q if static0 is None else static0 + eps_q
    return out


def with_q_eigenstrain(base_elastic_params, q, Lam11, Lam12, Lam44):
    """Shallow-copy base_elastic_params (from elastic.setup_elastic, P's
    own C/Q/Omega/K -- unchanged) with q's eigenstrain added to its
    eps_star_static (see _add_q_static). At q==0 identically,
    elastic.eigenstrain(q,...)==0 exactly (every eps0_ij term is q_k*q_l,
    degree >=2 in q), so the slot holds the base's own static (or, with
    none, the exact zero tensor, and elastic_solve's "eps_star_P + static"
    branch reproduces the static=None branch bit-for-bit, x+0.0==x in
    IEEE754) -- SPEC G20d(i)/(iii)."""
    return _add_q_static(base_elastic_params, elastic.eigenstrain(q, Lam11, Lam12, Lam44))


def elastic_force_explicit_q(q, sigma, Lam11, Lam12, Lam44):
    """dF_el/dq_m -- exact structural mirror of elastic.elastic_force_explicit
    (q,Lam replacing P,Q). Verified fresh below (mirrors elastic.py's own
    _verify_elastic_force_symbolic, SPEC G20c)."""
    q1, q2, q3 = q[0], q[1], q[2]
    s11, s22, s33 = sigma[0, 0], sigma[1, 1], sigma[2, 2]
    s12, s13, s23 = sigma[0, 1], sigma[0, 2], sigma[1, 2]
    f1 = -(2 * q1 * (s11 * Lam11 + (s22 + s33) * Lam12) + Lam44 * (s12 * q2 + s13 * q3))
    f2 = -(2 * q2 * (s22 * Lam11 + (s11 + s33) * Lam12) + Lam44 * (s12 * q1 + s23 * q3))
    f3 = -(2 * q3 * (s33 * Lam11 + (s11 + s22) * Lam12) + Lam44 * (s13 * q1 + s23 * q2))
    return np.stack([f1, f2, f3])


def _verify_elastic_force_symbolic_q():
    q1, q2, q3, Lam11, Lam12, Lam44 = sp.symbols("q1 q2 q3 Lam11 Lam12 Lam44", real=True)
    s11, s22, s33, s12, s13, s23 = sp.symbols("s11 s22 s33 s12 s13 s23", real=True)
    qv = [q1, q2, q3]
    es = {
        (0, 0): Lam11 * q1**2 + Lam12 * (q2**2 + q3**2),
        (1, 1): Lam11 * q2**2 + Lam12 * (q1**2 + q3**2),
        (2, 2): Lam11 * q3**2 + Lam12 * (q1**2 + q2**2),
        (0, 1): sp.Rational(1, 2) * Lam44 * q1 * q2,
        (0, 2): sp.Rational(1, 2) * Lam44 * q1 * q3,
        (1, 2): sp.Rational(1, 2) * Lam44 * q2 * q3,
    }
    es[(1, 0)], es[(2, 0)], es[(2, 1)] = es[(0, 1)], es[(0, 2)], es[(1, 2)]
    sg = {(0, 0): s11, (1, 1): s22, (2, 2): s33,
          (0, 1): s12, (1, 0): s12, (0, 2): s13, (2, 0): s13, (1, 2): s23, (2, 1): s23}
    explicit = [
        -(2 * q1 * (s11 * Lam11 + (s22 + s33) * Lam12) + Lam44 * (s12 * q2 + s13 * q3)),
        -(2 * q2 * (s22 * Lam11 + (s11 + s33) * Lam12) + Lam44 * (s12 * q1 + s23 * q3)),
        -(2 * q3 * (s33 * Lam11 + (s11 + s22) * Lam12) + Lam44 * (s13 * q1 + s23 * q2)),
    ]
    for m in range(3):
        generic = -sum(sp.diff(es[(i, j)], qv[m]) * sg[(i, j)] for i in range(3) for j in range(3))
        if sp.simplify(generic - explicit[m]) != 0:
            raise AssertionError(f"q eigenstrain force mismatch at component {m}: "
                                  f"generic={generic} vs explicit={explicit[m]}")


_verify_elastic_force_symbolic_q()


# ==========================================================================
# Combined (P,q) explicit driving terms + one semi-implicit step. P's own
# implicit operator (Minv, from solver.build_Minv) is UNCHANGED -- only its
# EXPLICIT term N(P) gains the new coupling force, exactly the same pattern
# elastic force / SPM correction already use in solver._explicit_N.
# ==========================================================================
def _explicit_N_P(P, q, alpha_P, coupling, ops, E_ext, elastic_params):
    """P's explicit driving term = solver._explicit_N's own formula PLUS
    the new f_Pq coupling force. At q==0 identically, couple_force_P(...)
    is EXACTLY 0.0 for ANY coupling coefficients (every f_Pq term has
    degree >=2 in q) -- SPEC G20d(i) holds by this construction, not a
    special-cased branch."""
    N = solver.bulk_force(P, alpha_P) - solver._reshape_vec3(E_ext, P.ndim, ensemble=P.ndim == 4)
    N = N + couple_force_P(P, q, coupling)
    if ops.get("dkappa") is not None:
        Ed, Ed_uniform, _n, _resid = solver._spm_iterate(P, ops)
        N = N - (Ed - Ed_uniform)
    if elastic_params is not None:
        Fel, _ = elastic.elastic_force(P, elastic_params)
        N = N + Fel
    return N


def _explicit_N_q(P, q, alpha_q, coupling, Lam=None, elastic_params=None):
    """q's explicit driving term: its own Devonshire polynomial (reusing
    solver.bulk_force with q's diagonal alpha dict) + coupling force +
    [eigenstrain force, if Lam/elastic_params given]. At P==0 identically,
    couple_force_q(...) is EXACTLY 0.0 for any coupling coefficients (same
    even-in-P argument) -- SPEC G20d(ii)."""
    N = solver.bulk_force(q, alpha_q) + couple_force_q(P, q, coupling)
    if Lam is not None and elastic_params is not None:
        sol = elastic.elastic_solve(P, elastic_params)  # elastic_params already carries q's eps_star_static
        N = N + elastic_force_explicit_q(q, sol["sigma"], Lam["Lam11"], Lam["Lam12"], Lam["Lam44"])
    return N


def step_pq(P, q, alpha_P, alpha_q, coupling, ops, Minv, dt, L, E_ext,
            ops_q=None, Minv_q=None, Lq=None, base_elastic_params=None, Lam=None):
    """One semi-implicit step for (P,q) together. P's implicit operator
    (Minv) is UNCHANGED from the pure-P engine; q's own implicit operator
    (Minv_q, scalar) handles its isotropic gradient the same way solver.py
    handles P's anisotropic one. ops_q=None => q is not evolved (returned
    unchanged, e.g. G20d(i)'s q-frozen-at-0 check reuses this by simply not
    passing ops_q at all, which also makes elastic_params fall through
    UNMODIFIED when Lam is None -- the exact-reproduction path, see
    with_q_eigenstrain's docstring for the Lam-given-but-q=0 case)."""
    elastic_params = base_elastic_params
    if base_elastic_params is not None and Lam is not None:
        elastic_params = with_q_eigenstrain(base_elastic_params, q, Lam["Lam11"], Lam["Lam12"], Lam["Lam44"])

    N_P = _explicit_N_P(P, q, alpha_P, coupling, ops, E_ext, elastic_params)
    s = (ops["Ny"], ops["Nx"])
    Pk = _rfft2(P)
    Nk = _rfft2(N_P)
    RHS = Pk - dt * L * Nk                              # (3,Ny,Nxr)
    # replaces np.moveaxis(np.einsum("...ij,...j->...i", Minv,
    #   np.moveaxis(RHS,0,-1)), -1, 0) -- SPEC §39.1 A5. Same explicit form,
    # same strided views of the public (Ny,Nxr,3,3) Minv, as solver.step:
    # ONE idiom for the per-mode 3x3 apply everywhere in the tree.
    Pk_new = np.empty_like(RHS)
    Pk_new[0] = Minv[..., 0, 0] * RHS[0] + Minv[..., 0, 1] * RHS[1] + Minv[..., 0, 2] * RHS[2]
    Pk_new[1] = Minv[..., 1, 0] * RHS[0] + Minv[..., 1, 1] * RHS[1] + Minv[..., 1, 2] * RHS[2]
    Pk_new[2] = Minv[..., 2, 0] * RHS[0] + Minv[..., 2, 1] * RHS[1] + Minv[..., 2, 2] * RHS[2]
    P_new = _irfft2(Pk_new, s)

    if ops_q is None:
        return P_new, q

    N_q = _explicit_N_q(P, q, alpha_q, coupling, Lam=Lam, elastic_params=elastic_params)
    sq = (ops_q["Ny"], ops_q["Nx"])
    qk = _rfft2(q)
    Nqk = _rfft2(N_q)
    qk_new = (qk - dt * Lq * Nqk) / Minv_q
    q_new = _irfft2(qk_new, sq)
    return P_new, q_new


def total_energy_pq(P, q, alpha_P, alpha_q, coupling, ops, grid, Gq, E_ext=(0., 0., 0.),
                     elastic_params=None, Lam=None):
    """Same shape as solver.total_energy (densities dict, totals dict),
    extended with q's own bulk/grad and the f_Pq coupling; elastic folds
    q's eigenstrain in via with_q_eigenstrain, consistent with step_pq."""
    fb_P = solver.bulk_energy_density(P, alpha_P)
    fb_q = solver.bulk_energy_density(q, alpha_q)
    fg_P = solver.grad_energy_density(P, grid, ops)
    fg_q = grad_energy_density_q(q, grid, Gq)
    fc = couple_energy_density(P, q, coupling)
    fe = solver.elec_energy_density(P, ops, E_ext)
    dens = dict(bulk_P=float(fb_P.mean()), bulk_q=float(fb_q.mean()),
                grad_P=float(fg_P.mean()), grad_q=float(fg_q.mean()),
                coupling=float(fc.mean()), elec=float(fe.mean()))
    ep = elastic_params
    if ep is not None and Lam is not None:
        ep = with_q_eigenstrain(ep, q, Lam["Lam11"], Lam["Lam12"], Lam["Lam44"])
    if ep is not None:
        sol = elastic.elastic_solve(P, ep)
        dens["elastic"] = float(elastic.elastic_energy_density(sol).mean())
    dens["total"] = sum(dens.values())
    tot = {k: v * grid["Volume"] for k, v in dens.items()}
    return dens, tot


# ==========================================================================
# SPEC V2-03 A3 -- the q sector on the TWO 3D ENGINES (bulk3d: fully
# periodic rfftn box; film3d: 2D-periodic + banded z). Part II ruling (3)
# keeps the two solvers INDEPENDENT and the WP26/27 "no dimension-generic
# core" constraint stands, so everything below is written ONCE PER ENGINE
# on that engine's own primitives. The 2D functions above are untouched.
#
# WHAT IS SHARED, and why it is legitimate: ONLY the genuinely LOCAL
# (elementwise) pieces -- couple_energy_density / couple_force_P /
# couple_force_q above (sympy-lambdified in P[0..2], q[0..2], no grid-shape
# assumption anywhere in their bodies) and solver.bulk_energy_density /
# bulk_force. Both already work on a (3,Nz,Ny,Nx) field with zero code
# changes -- exactly the reuse bulk3d.py itself documents for bulk_force.
# Every NON-local object (the gradient operator, the implicit operator, the
# transform pair, the elastic solve) is rebuilt per engine below.
#
# TRANSFORMS: the code below calls bulk3d._rfftn/_irfftn and
# film3d._rfft2/_irfft2 -- the ENGINE's own transforms, deliberately NOT
# this module's _rfft2 -- so that a caller which sets bulk3d.FFT_WORKERS /
# film3d.FFT_WORKERS (gates/core/_kit.fft_workers_one, casekit.single_thread)
# gets the same setting for the q channel it gets for the P channel.
#
# LAM CONVENTION (the one named trap in this section, SPEC S37.5). The deck
# prints Lam11/Lam12/Lam44 in VOIGT form, i.e. elastic.eigenstrain's own
# convention: eps0_12 = 0.5*Lam44*q1*q2. bulk3d.eigenstrain_3d instead takes
# the FULL-TENSOR triple Q1111/Q1122/Q1212 with eps0_12 = 2*Q1212*q1*q2.
# Equating the two: 2*Lam1212 = 0.5*Lam44, so
#       Lam1111 = Lam11,  Lam1122 = Lam12,  Lam1212 = Lam44/4
# -- the SAME mapping bulk3d's own docstring states for a Voigt-sourced Q
# deck ("Q1212 = Q44_voigt/4 ... only the shear-shear slot carries it").
# `lam_tensor_3d` below is that mapping, applied ONLY on the bulk3d side.
# The film lane needs NO conversion: film_elastic builds its eigenstrain
# with elastic.eigenstrain (the Voigt convention), so with_q_eigenstrain_film
# passes Lam11/Lam12/Lam44 straight through, and the film's dF_el/dq is the
# EXISTING 2D elastic_force_explicit_q, reused unchanged.
# ==========================================================================

# --------------------------------------------------------------------------
# A3(a) -- bulk3d (fully periodic)
# --------------------------------------------------------------------------
def build_ops_q_3d(grid, Gq):
    """Aq(k) = Gq*(kx^2+ky^2+kz^2), scalar, SAME for all 3 components -- the
    3D mirror of build_ops_q on bulk3d's rfftn half-spectrum grid
    (KXr,KYr,KZr, shape (Nz,Ny,Nxr)). Same dict pattern, one more axis:
    isotropic, so no off-diagonal term and no D(k) (q does not source the
    depolarization field -- SPEC S23.1, unchanged in 3D).

    At Nz=1 bulk3d's KZr is identically 0.0, so this Aq is BIT-FOR-BIT
    build_ops_q's own Aq on the matching 2D grid (K2 `afe_nz1_vs_2d`)."""
    return dict(Aq=operators.isotropic_symbol(grid, Gq, 3),
                Nx=grid["Nx"], Ny=grid["Ny"], Nz=grid["Nz"])


def grad_energy_density_q_3d(q, grid, Gq):
    """f_gradq = 0.5*Gq*|grad q|^2 summed over the 3 components, shape
    (Nz,Ny,Nx). Uses bulk3d._spectral_grad_xyz (that module's own exact
    ik-then-ifft helper) the way the 2D version uses solver._spectral_grad_xy.
    Same even-grid Nyquist convention as grad_energy_density_q (and
    bulk3d.grad_energy_density_3d): real skew-adjoint derivatives
    (bulk3d._modified_wavevectors_3d) plus, per even axis j,
    0.5*Gq*(pi/h_j)^2*(NyquistProject_j q_i)^2; odd grids bit-identical."""
    KX, KY, KZ = operators.masked_wavevectors_3d(grid)
    q1x, q1y, q1z = bulk3d._spectral_grad_xyz(q[0], grid, KX, KY, KZ)
    q2x, q2y, q2z = bulk3d._spectral_grad_xyz(q[1], grid, KX, KY, KZ)
    q3x, q3y, q3z = bulk3d._spectral_grad_xyz(q[2], grid, KX, KY, KZ)
    fg = 0.5 * Gq * (q1x**2 + q1y**2 + q1z**2
                     + q2x**2 + q2y**2 + q2z**2
                     + q3x**2 + q3y**2 + q3z**2)
    return operators.add_nyquist_correction(fg, q, grid, Gq, 3)


def grad_force_q_3d(q, ops_q):
    """d f_gradq / dq = Aq(k) . q, diagonal-only (mirrors bulk3d.grad_force_3d
    with the two off-diagonal rows dropped -- q's gradient is isotropic)."""
    s = (ops_q["Nz"], ops_q["Ny"], ops_q["Nx"])
    qk = bulk3d._rfftn(q)
    return bulk3d._irfftn(ops_q["Aq"] * qk, s)


def build_Minv_q_3d(ops_q, dt, Lq):
    """M_q(k) = 1 + dt*Lq*Aq(k), scalar per k-point (isotropic/diagonal -- no
    3x3 batched inverse, unlike bulk3d.build_Minv_3d)."""
    return 1.0 + dt * Lq * ops_q["Aq"]


def lam_tensor_3d(Lam):
    """(Lam1111, Lam1122, Lam1212) for bulk3d.eigenstrain_3d from the deck's
    VOIGT triple -- see this section's LAM CONVENTION block. Returns
    (Lam11, Lam12, Lam44/4)."""
    return Lam["Lam11"], Lam["Lam12"], Lam["Lam44"] / 4.0


def with_q_eigenstrain_3d(base_elastic_params, q, Lam):
    """Shallow-copy bulk3d.setup_elastic_3d's output with q's eigenstrain in
    bulk3d's TENSOR convention added to `eps_star_static` (the A2 slot; the
    base's own static is kept, see _add_q_static).
    Shallow copy, not a fresh setup_elastic_3d call: the A2 slot is a plain
    dict key that elastic_solve_3d reads per solve, so rebuilding Omega once
    per step would be pure waste -- this is with_q_eigenstrain's own idiom,
    unchanged. At q==0 identically the eigenstrain is the EXACT zero tensor
    (every eps0_ij term is q_k*q_l), so with no base static elastic_solve_3d's
    "eps_star + static" branch reproduces the static=None branch bit-for-bit
    (x+0.0==x)."""
    L1111, L1122, L1212 = lam_tensor_3d(Lam)
    return _add_q_static(base_elastic_params, bulk3d.eigenstrain_3d(q, L1111, L1122, L1212))


def elastic_force_explicit_q_3d(q, sigma, Lam1111, Lam1122, Lam1212):
    """dF_el/dq_m = -sigma_ij * d(eps0_ij)/dq_m, FULL TENSOR sum over
    i,j,k,l=1..3 -- the structural mirror of bulk3d.elastic_force_explicit_3d
    with (q,Lam) for (P,Q), NOT of the 2D elastic_force_explicit_q above
    (different eigenstrain convention: the 4*Lam1212 here is the Voigt
    Lam44). Verified fresh at import below, exactly as the 2D one is."""
    q1, q2, q3 = q[0], q[1], q[2]
    s11, s22, s33 = sigma[0, 0], sigma[1, 1], sigma[2, 2]
    s12, s13, s23 = sigma[0, 1], sigma[0, 2], sigma[1, 2]
    f1 = -(2 * q1 * (s11 * Lam1111 + (s22 + s33) * Lam1122) + 4 * Lam1212 * (s12 * q2 + s13 * q3))
    f2 = -(2 * q2 * (s22 * Lam1111 + (s11 + s33) * Lam1122) + 4 * Lam1212 * (s12 * q1 + s23 * q3))
    f3 = -(2 * q3 * (s33 * Lam1111 + (s11 + s22) * Lam1122) + 4 * Lam1212 * (s13 * q1 + s23 * q2))
    return np.stack([f1, f2, f3])


def _verify_elastic_force_symbolic_q_3d():
    """Import-time self-check: bulk3d.eigenstrain_3d's OWN full k,l=1..3
    tensor eps0 (transcribed here with Lam for Q), differentiated
    symbolically, must equal elastic_force_explicit_q_3d's closed form."""
    q1, q2, q3 = sp.symbols("q1 q2 q3", real=True)
    L1111, L1122, L1212 = sp.symbols("Lam1111 Lam1122 Lam1212", real=True)
    s11, s22, s33, s12, s13, s23 = sp.symbols("s11 s22 s33 s12 s13 s23", real=True)
    qv = [q1, q2, q3]
    es = {
        (0, 0): L1111 * q1**2 + L1122 * (q2**2 + q3**2),
        (1, 1): L1111 * q2**2 + L1122 * (q1**2 + q3**2),
        (2, 2): L1111 * q3**2 + L1122 * (q1**2 + q2**2),
        (0, 1): 2 * L1212 * q1 * q2,
        (0, 2): 2 * L1212 * q1 * q3,
        (1, 2): 2 * L1212 * q2 * q3,
    }
    es[(1, 0)], es[(2, 0)], es[(2, 1)] = es[(0, 1)], es[(0, 2)], es[(1, 2)]
    sg = {(0, 0): s11, (1, 1): s22, (2, 2): s33,
          (0, 1): s12, (1, 0): s12, (0, 2): s13, (2, 0): s13, (1, 2): s23, (2, 1): s23}
    explicit = [
        -(2 * q1 * (s11 * L1111 + (s22 + s33) * L1122) + 4 * L1212 * (s12 * q2 + s13 * q3)),
        -(2 * q2 * (s22 * L1111 + (s11 + s33) * L1122) + 4 * L1212 * (s12 * q1 + s23 * q3)),
        -(2 * q3 * (s33 * L1111 + (s11 + s22) * L1122) + 4 * L1212 * (s13 * q1 + s23 * q2)),
    ]
    for m in range(3):
        generic = -sum(sp.diff(es[(i, j)], qv[m]) * sg[(i, j)] for i in range(3) for j in range(3))
        if sp.simplify(generic - explicit[m]) != 0:
            raise AssertionError(f"3D q eigenstrain force mismatch at component {m}: "
                                  f"generic={generic} vs explicit={explicit[m]}")
    # and the Voigt<->tensor bridge itself, so lam_tensor_3d's /4 is CHECKED,
    # not asserted (SPEC S5: a factor-of-2 shear claim is never asserted):
    Lam44 = sp.symbols("Lam44", real=True)
    if sp.simplify(2 * (Lam44 / 4) * q1 * q2 - sp.Rational(1, 2) * Lam44 * q1 * q2) != 0:
        raise AssertionError("lam_tensor_3d: Voigt->tensor shear mapping is not Lam44/4")


_verify_elastic_force_symbolic_q_3d()


def _explicit_N_P_3d(P, q, alpha_P, coupling, ops, E_ext, elastic_params):
    """P's explicit driving term on bulk3d = bulk3d._explicit_N_3d's own
    formula PLUS the f_Pq coupling force. At q==0 identically couple_force_P
    is EXACTLY 0.0 for ANY coupling coefficients (every f_Pq term has degree
    >=2 in q), so this reproduces bulk3d._explicit_N_3d bit-for-bit -- by
    construction, not a branch (K2 `afe_wiring_3d` leg (i)).

    SPEC V2-03.C carry-over (d.2): since B2 gave bulk3d its own kappa(r) SPM
    (build_operators_3d now accepts an array kappa and sets ops["dkappa"]),
    the 2D _explicit_N_P's `dkappa` branch has a 3D twin and it is MIRRORED
    HERE VERBATIM -- same placement (after the coupling force, before the
    elastic force), same expression `N -= (Ed - Ed_uniform)`, and the same
    solve bulk3d._explicit_N_3d itself calls (_spm_solve_3d, so the warm-start
    cache in ops is read and written in exactly ONE place for both steppers).
    WP5 SS11.1's hybrid split is unchanged by the q sector: Minv still carries
    the HOMOGENEOUS D(k)/kappa0 operator, and the difference between the true
    kappa(r) depolarization field and what Minv implicitly assumes rides in
    the explicit term. ops["dkappa"] is None (scalar kappa, every pre-V2-03.C
    caller) skips the branch, so N is bit-identical to before -- and with
    q == 0 AND an array kappa, step_pq_3d now equals bulk3d.step_3d
    bit-for-bit as well (K2 `afe_spm_hybrid_3d` leg (ii))."""
    N = bulk3d.bulk_force(P, alpha_P) - bulk3d._reshape_vec3_3d(E_ext, P.ndim, ensemble=P.ndim == 5)
    N = N + couple_force_P(P, q, coupling)
    if ops.get("dkappa") is not None:
        Ed, Ed_uniform, _n, _resid = bulk3d._spm_solve_3d(P, ops)
        N = N - (Ed - Ed_uniform)
    if elastic_params is not None:
        Fel, _ = bulk3d.elastic_force_3d(P, elastic_params)
        N = N + Fel
    return N


def _explicit_N_q_3d(P, q, alpha_q, coupling, Lam=None, elastic_params=None):
    """q's explicit driving term on bulk3d: its own Devonshire polynomial
    (solver.bulk_force with q's alpha dict -- elementwise, reused) + the
    coupling force + [the eigenstrain force, if Lam/elastic_params given].
    At P==0 identically couple_force_q is EXACTLY 0.0 (same even-in-P
    argument as the 2D module's)."""
    N = bulk3d.bulk_force(q, alpha_q) + couple_force_q(P, q, coupling)
    if Lam is not None and elastic_params is not None:
        sol = bulk3d.elastic_solve_3d(P, elastic_params)   # el already carries q's eps_star_static
        L1111, L1122, L1212 = lam_tensor_3d(Lam)
        N = N + elastic_force_explicit_q_3d(q, sol["sigma"], L1111, L1122, L1212)
    return N


def step_pq_3d(P, q, alpha_P, alpha_q, coupling, ops, Minv, dt, L, E_ext,
               ops_q=None, Minv_q=None, Lq=None, base_elastic_params=None, Lam=None):
    """One semi-implicit step for (P,q) together on bulk3d -- step_pq's 3D
    mirror, built on bulk3d.step_3d's own primitives (rfftn pair, the full
    (Nz,Ny,Nxr,3,3) Minv apply). P's implicit operator is UNCHANGED from the
    pure-P 3D engine; q's own (scalar Minv_q) handles its isotropic gradient.
    ops_q=None => q is not evolved (returned unchanged), which also leaves
    elastic_params UNMODIFIED when Lam is None -- the exact-reproduction path
    of K2 `afe_wiring_3d` leg (i)."""
    elastic_params = base_elastic_params
    if base_elastic_params is not None and Lam is not None:
        elastic_params = with_q_eigenstrain_3d(base_elastic_params, q, Lam)

    N_P = _explicit_N_P_3d(P, q, alpha_P, coupling, ops, E_ext, elastic_params)
    s = (ops["Nz"], ops["Ny"], ops["Nx"])
    Pk = bulk3d._rfftn(P)
    Nk = bulk3d._rfftn(N_P)
    RHS = Pk - dt * L * Nk                              # (3,Nz,Ny,Nxr)
    # Same explicit per-mode 3x3 apply, same strided views of the public
    # (Nz,Ny,Nxr,3,3) Minv, as bulk3d.step_3d -- SPEC §39.1 A5's ONE idiom.
    Pk_new = np.empty_like(RHS)
    Pk_new[0] = Minv[..., 0, 0] * RHS[0] + Minv[..., 0, 1] * RHS[1] + Minv[..., 0, 2] * RHS[2]
    Pk_new[1] = Minv[..., 1, 0] * RHS[0] + Minv[..., 1, 1] * RHS[1] + Minv[..., 1, 2] * RHS[2]
    Pk_new[2] = Minv[..., 2, 0] * RHS[0] + Minv[..., 2, 1] * RHS[1] + Minv[..., 2, 2] * RHS[2]
    P_new = bulk3d._irfftn(Pk_new, s)

    if ops_q is None:
        return P_new, q

    N_q = _explicit_N_q_3d(P, q, alpha_q, coupling, Lam=Lam, elastic_params=elastic_params)
    sq = (ops_q["Nz"], ops_q["Ny"], ops_q["Nx"])
    qk = bulk3d._rfftn(q)
    Nqk = bulk3d._rfftn(N_q)
    qk_new = (qk - dt * Lq * Nqk) / Minv_q
    q_new = bulk3d._irfftn(qk_new, sq)
    return P_new, q_new


def total_energy_pq_3d(P, q, alpha_P, alpha_q, coupling, ops, grid, Gq, E_ext=(0., 0., 0.),
                       elastic_params=None, Lam=None):
    """total_energy_pq's 3D mirror: same (densities, totals) contract and the
    same seven keys, every term evaluated with bulk3d's own 3D functions."""
    fb_P = bulk3d.bulk_energy_density(P, alpha_P)
    fb_q = bulk3d.bulk_energy_density(q, alpha_q)
    fg_P = bulk3d.grad_energy_density_3d(P, grid, ops)
    fg_q = grad_energy_density_q_3d(q, grid, Gq)
    fc = couple_energy_density(P, q, coupling)
    fe = bulk3d.elec_energy_density_3d(P, ops, E_ext)
    dens = dict(bulk_P=float(fb_P.mean()), bulk_q=float(fb_q.mean()),
                grad_P=float(fg_P.mean()), grad_q=float(fg_q.mean()),
                coupling=float(fc.mean()), elec=float(fe.mean()))
    ep = elastic_params
    if ep is not None and Lam is not None:
        ep = with_q_eigenstrain_3d(ep, q, Lam)
    if ep is not None:
        sol = bulk3d.elastic_solve_3d(P, ep)
        dens["elastic"] = float(bulk3d.elastic_energy_density_3d(sol).mean())
    dens["total"] = sum(dens.values())
    tot = {k: v * grid["Volume"] for k, v in dens.items()}
    return dens, tot


# --------------------------------------------------------------------------
# A3(b) -- film3d (2D-periodic in-plane + banded z, natural BC)
# --------------------------------------------------------------------------
def build_q_factors_film(grid, Gq, dt, Lq):
    """M_q = I + dt*Lq*[q's gradient-force operator], factorized once per
    (grid,Gq,dt,Lq) -- the q twin of film3d.build_grad_factors.

    DELEGATED, not re-typed: f_gradq = 0.5*Gq*|grad q|^2 is the SAME
    isotropic functional film3d's P gradient is (f_grad = 0.5*G11*|grad P|^2,
    the paper's Eq. 7 form this engine is built on), so film3d's own
    face-based finite-volume z-bands + spectral in-plane k^2, evaluated at
    Gq and Lq, ARE q's implicit operator -- natural (Neumann) BC included,
    with the outermost off-diagonals doubled exactly as _grad_z_bands does.
    Re-typing those bands here would be a byte-copy of a validated operator
    (the g10e energy-monotonicity derivation), not an independent one."""
    return film3d.build_grad_factors(grid, Gq, dt, Lq)


def grad_force_q_film(q, grid, Gq):
    """delta F_gradq/delta q_i in film geometry, applied EXPLICITLY (matrix
    action) and already divided by the trapezoidal node weight qw(z), the
    convention film3d's N(P) uses for every nodal term. Same delegation
    argument as build_q_factors_film: film3d.grad_force_3d IS this operator
    at Gq (its body reads only the grid geometry and one isotropic
    coefficient; it makes no P-specific assumption)."""
    return film3d.grad_force_3d(q, grid, Gq)


def grad_energy_q_film(q, grid, Gq):
    """Total film gradient energy for q, using the same operator as its force."""
    return film3d.grad_energy_3d(q, grid, Gq)


def with_q_eigenstrain_film(base_el, q, Lam):
    """Shallow-copy film_elastic.setup_film_elastic's output with q's
    eigenstrain over the FULL STACK added to the A2 `eps_star_static` slot
    (the base's own static is kept, see _add_q_static), q's part exactly
    zero in the SUBSTRATE block (this lane sources face eigenstrain from
    the film block only -- setup_film_elastic refuses anything else, and the
    zeros written here are hard zeros, never a near-zero fill).

    NO Voigt->tensor conversion here (unlike with_q_eigenstrain_3d):
    film_elastic builds eps*(P) with elastic.eigenstrain, the same convention
    the deck's Lam11/Lam12/Lam44 are printed in -- see the section's LAM
    CONVENTION block."""
    S = np.zeros((3, 3, base_el["Nz"], base_el["Ny"], base_el["Nx"]))
    S[:, :, base_el["film_lo"]:] = elastic.eigenstrain(q, Lam["Lam11"], Lam["Lam12"], Lam["Lam44"])
    return _add_q_static(base_el, S)


def _explicit_N_P_film(P, q, alpha_P, coupling, grid, poisson_factors, E_ext, el):
    """P's explicit driving term in film geometry = film3d.explicit_N's own
    (1/qw)-consistent sum (bulk + electrostatic [+ elastic, with that
    function's validated hz/qw rescale]) PLUS the f_Pq coupling force, which
    is POINTWISE and therefore (1/qw)-consistent for free, exactly like
    solver.bulk_force. At q==0 identically the addition is EXACTLY 0.0, so
    this reproduces film3d.explicit_N bit-for-bit."""
    N = film3d.explicit_N(P, alpha_P, grid, poisson_factors, E_ext, el=el)
    return N + couple_force_P(P, q, coupling)


def _explicit_N_q_film(P, q, alpha_q, coupling, grid, Lam=None, el=None):
    """q's explicit driving term in film geometry, in the same (1/qw)
    convention: Devonshire + coupling (both pointwise) + [the rotostriction-
    style eigenstrain force]. The elastic piece needs film3d.explicit_N's
    OWN hz/qw rescale (film_elastic integrates against a UNIFORM hz per node,
    1x interior / 2x at the two z-boundaries) -- the same correction, for the
    same reason, applied to the q channel.

    The force itself is the EXISTING 2D elastic_force_explicit_q, unchanged:
    film_elastic's sigma_node carries the exact d(discrete energy)/d(node)
    chain rule, and q's eigenstrain enters through the same face source in
    the same Voigt convention, so -sigma_node : d eps*(q)/dq is that
    function verbatim."""
    N = solver.bulk_force(q, alpha_q) + couple_force_q(P, q, coupling)
    if Lam is not None and el is not None:
        sol = film_elastic.elastic_solve_film(P, el)   # el already carries q's eps_star_static
        Fel_q = elastic_force_explicit_q(q, sol["sigma_node"],
                                         Lam["Lam11"], Lam["Lam12"], Lam["Lam44"])
        qw_ratio = (grid["hz"] / grid["qw"])[None, :, None, None]
        N = N + Fel_q * qw_ratio
    return N


def step_pq_film(P, q, alpha_P, alpha_q, coupling, grid, grad_factors, poisson_factors,
                 dt, L, E_ext=(0.0, 0.0, 0.0), q_factors=None, Lq=None,
                 base_el=None, Lam=None):
    """One semi-implicit step for (P,q) together in film geometry --
    film3d.step's own split (gradient implicit per in-plane mode through the
    banded z solve, everything else explicit), run once for P and once for q.
    q_factors=None => q is not evolved (returned unchanged), and
    elastic_params falls through UNMODIFIED when Lam is None: the
    exact-reproduction path of K2 `afe_wiring_film` leg (i)."""
    el = base_el
    if base_el is not None and Lam is not None:
        el = with_q_eigenstrain_film(base_el, q, Lam)

    N_P = _explicit_N_P_film(P, q, alpha_P, coupling, grid, poisson_factors, E_ext, el)
    Pk = film3d._rfft2(P)
    Nk = film3d._rfft2(N_P)
    RHS = np.moveaxis(Pk - dt * L * Nk, 1, 0)      # (Nz,3,Ny,Nxr): z first for the tridiagonal solve
    Pk_new = np.moveaxis(film3d.solve_tridiag_factored(grad_factors, RHS), 0, 1)
    P_new = film3d._irfft2(Pk_new, (grid["Ny"], grid["Nx"]))

    if q_factors is None:
        return P_new, q

    N_q = _explicit_N_q_film(P, q, alpha_q, coupling, grid, Lam=Lam, el=el)
    qk = film3d._rfft2(q)
    Nqk = film3d._rfft2(N_q)
    RHSq = np.moveaxis(qk - dt * Lq * Nqk, 1, 0)
    qk_new = np.moveaxis(film3d.solve_tridiag_factored(q_factors, RHSq), 0, 1)
    q_new = film3d._irfft2(qk_new, (grid["Ny"], grid["Nx"]))
    return P_new, q_new


def total_energy_pq_film(P, q, alpha_P, alpha_q, coupling, grid, kappa_b, G11, Gq,
                         poisson_factors=None, E_ext=(0.0, 0.0, 0.0), el=None, Lam=None):
    """(densities, totals) in film geometry, keys as total_energy_pq.
    P's bulk/grad/elec [+ elastic] come from film3d.total_energy_3d itself
    (its quadrature conventions are the validated ones -- qw for the nodal
    densities, the face total for grad_z, film_elastic's own uniform-hz total
    for elastic); q's bulk and the coupling are nodal, so qw-weighted the
    same way; grad_q uses grad_energy_q_film. Densities are totals/Volume3D,
    film3d's own pairing."""
    if poisson_factors is None:
        poisson_factors = film3d.poisson_factor(grid, kappa_b)
    ep = el
    if el is not None and Lam is not None:
        ep = with_q_eigenstrain_film(el, q, Lam)
    _densP, totP = film3d.total_energy_3d(P, alpha_P, grid, kappa_b, G11,
                                          poisson_factors=poisson_factors, E_ext=E_ext, el=ep)
    area = grid["Lx"] * grid["Ly"]
    qw = grid["qw"]
    fb_q = solver.bulk_energy_density(q, alpha_q)
    fc = couple_energy_density(P, q, coupling)
    tot = dict(bulk_P=totP["bulk"],
               bulk_q=area * float(np.sum(qw * fb_q.mean(axis=(-2, -1)))),
               grad_P=totP["grad"], grad_q=grad_energy_q_film(q, grid, Gq),
               coupling=area * float(np.sum(qw * fc.mean(axis=(-2, -1)))),
               elec=totP["elec"])
    if "elastic" in totP:
        tot["elastic"] = totP["elastic"]
    tot["total"] = sum(tot.values())
    dens = {k: v / grid["Volume3D"] for k, v in tot.items()}
    return dens, tot
