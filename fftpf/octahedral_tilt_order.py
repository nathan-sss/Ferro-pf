"""SPEC S28 (WP18/S2): octahedral tilt order parameter theta -- new field,
new module, mirroring solvers/kittel_afe.py's own (P,q) architecture exactly
(SPEC S28 preamble: "mirror the S1 pattern -- a NEW standalone module
solvers/afd.py reusing solver.py primitives; solver.py/afe.py themselves
untouched" -- afd.py/afe.py are this module's and kittel_afe.py's pre-WP24
names, SPEC S35). Every function below either (a) REUSES solver.py's existing
functions unchanged (theta's own Devonshire polynomial via
solver.bulk_energy_density/bulk_force with b111=b112=b123=0.0, exactly the
trick kittel_afe.py already uses for q with a112=a123=0), or (b) is a small, new,
sympy-built addition for the genuinely new physics: the P-theta coupling
f_Ptheta, theta's own isotropic gradient f_gradtheta, and the (parked)
rotostriction eigenstrain force dF/dtheta.

Binding physics, SPEC S28.1 + checkpoint-1 ruling S28.5:
- theta(r): 3-component axial-vector AFD order parameter (oxygen-octahedron
  rotation). theta does NOT source the depolarization/Poisson field; E
  couples to theta ONLY indirectly through P via f_Ptheta -- same
  "binding, gated" pattern as q (SPEC S28.1, mirrors S23.1).
- f_Ptheta uses Chen's PRINTED MINUS-SIGN-PREFIXED convention (S28.5 R2,
  SUPERSEDES S28.1's bare "+"): f_Ptheta = -t11*sum(Pi^2*thetai^2)
  - t12*sum_{i!=j}(Pi^2*thetaj^2) - t44*(P1P2theta1theta2 +
  P2P3theta2theta3 + P3P1theta3theta1). The t44 slot (P_i P_j theta_i
  theta_j family) EXTENDS S28.1's original two-term menu to match Rabe
  App A Table 3's own closure (S28.5 R2, WP17 S23.5(a) precedent for
  "extend the menu to match the source").

Deck: params/sto_afd_rabeA.toml, a T=298K frozen snapshot of
sto_afd_landau_si(T) below -- same generator/frozen-snapshot relationship
literature_params.py already has between pzt_haun_x(x,T) and
params/archive/pzt_Haun_x0.6.toml, and kittel_afe.py has between pbzo_kittel_landau_si(T) and
pbzo_kittel_lum2022.toml.

Source: Rabe, Ahn & Triscone (eds.), "Physics of Ferroelectrics: A Modern
Perspective" (Springer 2007), Appendix A (L.-Q. Chen), Table 3 "Landau-
Devonshire coefficients for SrTiO3" (book p.367), citing refs [10]-[15]
undifferentiated per-row (reports/WP18_STAGE0_NOTE.md; checkpoint-1 ruling
SPEC S28.5 spot-checked this directly against the local PDF). Candidate-
spread policy (S28.5 R3): rows with 2-4 undifferentiated values use the
FIRST listed value as the deck default; the full spread is recorded in
params/sto_afd_rabeA.toml's [meta], not swept (no compute spent on
bracket sweeps this WP).

Units (S28.5 R2): theta is Chen's own "linear oxygen displacement"
(meters), NOT an angle -- units.si_to_deck_theta/si_to_deck_theta_landau
(new, additive) handle its own length rescaling (deck theta = SI theta in
the engine's existing nm length unit), documented there in full.

Book typo on record (S28.5 R2): Table 3 prints "N.m^-4" on the beta11 row
(copy-pasted from the beta1 row above it); dimensional analysis (beta11
multiplies theta^4 in an energy density, and the beta12 row -- same
quartic degree -- correctly prints N.m^-6) requires N.m^-6, used here.

GATES: g22a-f (analytic single cell / pure-theta anchor / FD force-energy /
wiring / PZT R3c / PZT-Haun quantitative). All six import this module under
its pre-WP24 shim name, `from solvers import afd`, so grepping "octa_tilt"
in gates/ finds nothing.
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
# SPEC V2-03 A4: the theta sector is written PER ENGINE (Part II ruling (3):
# the two 3D solvers stay independent, no dimension-generic core), so this
# module now also builds on bulk3d's and film3d's own primitives -- exactly
# as solvers/kittel_afe.py does for q. Neither imports this module, so no cycle.
from fftpf import pf3d_engine as bulk3d
from fftpf import film_engine as film3d
from fftpf import film_elastic
from fftpf import units
from fftpf.deck_reader import load_params_toml

FFT_WORKERS = -1


def _rfft2(x):
    return spfft.rfft2(x, axes=(-2, -1), workers=FFT_WORKERS)


def _irfft2(xk, s):
    return spfft.irfft2(xk, s=s, axes=(-2, -1), workers=FFT_WORKERS)


# ==========================================================================
# Rabe App A Table 3 (SrTiO3), SI units, T in K. Reproduced here as
# closed-form Python -- the T-parametrized generator that the frozen TOML
# snapshot is one T=298.0 K instance of.
# ==========================================================================
# P-sector alpha1(T): FOUR stacked literature forms in the source table
# (refs [10]-[15] undifferentiated); S28.5 R3 "first listed value" default
# is this unrestricted-range form. The other three (range-restricted:
# T<50K, T>70K, T>100K) are the R3 bracket, recorded in the TOML [meta],
# not implemented as extra branches here (they are ALTERNATE
# determinations, not pieces of one function).
ALPHA1_STO_BRACKET_SI = {
    "default_405coth": lambda T: 405e5 * (1.0 / np.tanh(54.0 / T) - 1.0 / np.tanh(54.0 / 30.0)),
    "alt_263p5coth_Tlt50": lambda T: 263.5e5 * (1.0 / np.tanh(42.0 / T) - 0.90476),
    "alt_737_Tgt70": lambda T: 7.37e5 * (T - 28.0),
    "alt_706_Tgt100": lambda T: 7.06e5 * (T - 35.5),
}
ALPHA11_STO_SI = 1.70e9   # T-independent, Table 3
ALPHA12_STO_SI = 1.37e9   # T-independent, Table 3

# theta-sector beta1(T): SINGLE unambiguous coth form (no range restriction
# printed), coth(145/T)-coth(145/105) so beta1(105)=0 identically --
# reports/WP18_STAGE0_NOTE.md's G22b target, S28.5 R4 "CLOSURE-BY-
# CONSTRUCTION" label.
BETA1_STO_SI = lambda T: 1.32e29 * (1.0 / np.tanh(145.0 / T) - 1.0 / np.tanh(145.0 / 105.0))
TC_STO_K = 105.0  # exact, by construction of BETA1_STO_SI

# beta11/beta12: 4 undifferentiated candidates each (S28.5 R3: first = default).
BETA11_STO_BRACKET_SI = (1.69e50, 1.58e50, 1.20e50, 0.996e50)
BETA12_STO_BRACKET_SI = (3.88e50, 3.78e50, 2.88e50, 2.73e50)

# t11/t12: 2 undifferentiated candidates each; t44: single value (Table 3).
# Units confirmed 10^29 C^-2.N by dimensional analysis: -t11*P^2*theta^2 is
# an energy density [N/m^2], P^2*theta^2 is [C^2/m^2] (P in C/m^2, theta in
# m), so t11 must be [N/m^2]/[C^2/m^2] = N/C^2 -- matches the printed unit
# exactly (cross-check against the OTHER Table-3 row with "10^18 m^-2"
# units, which is Lambda -- the rotostriction row, S28.5 R6 parked, NOT
# t -- confirmed by the SAME dimensional check: Lambda*theta^2 must be a
# DIMENSIONLESS strain, Lambda*m^2=1 => Lambda~m^-2, matching that row).
T11_STO_BRACKET_SI = (-1.74e29, -2.10e29)
T12_STO_BRACKET_SI = (-0.755e29, -0.845e29)
T44_STO_SI = 5.85e29  # single value

# Lambda (rotostriction), S28.5 R6: parked verbatim, not wired into the
# main science path this WP. 4 candidates each (Table 3).
LAM11_THETA_STO_BRACKET_SI = (8.7e18, 14.7e18, 9.32e18, 8.35e18)
LAM12_THETA_STO_BRACKET_SI = (-7.8e18, -7.34e18, -6.4e18, -5.54e18)
LAM44_THETA_STO_BRACKET_SI = (-9.2e18, -9.88e18, -6.93e18, -7.56e18)

# c_ij (elastic stiffness), S28.5 R6: parked verbatim, 4 candidates each.
C11_STO_BRACKET_SI = (3.156e11, 3.181e11, 3.36e11, 3.48e11)
C12_STO_BRACKET_SI = (1.01e11, 1.025e11, 1.027e11, 1.07e11)
C44_STO_BRACKET_SI = (1.19e11, 1.215e11, 1.236e11, 1.27e11)
# Q_ij (P-electrostriction), parked with the elastic sector: 3 candidates each.
Q11_STO_BRACKET_SI = (0.0457, 0.08, 0.066)
Q12_STO_BRACKET_SI = (-0.0135, -0.015, -0.013)
Q44_STO_SI = 0.00957  # single value


def sto_afd_landau_si(T):
    """SI dict: P-sector, theta-sector (both engine ALPHA_KEYS naming) and
    the (t11,t12,t44) coupling menu, at temperature T [K]. First-listed
    default per bracket row (S28.5 R3)."""
    a1 = ALPHA1_STO_BRACKET_SI["default_405coth"](T)
    P = dict(a1=a1, a11=ALPHA11_STO_SI, a12=ALPHA12_STO_SI, a111=0.0, a112=0.0, a123=0.0)
    b1 = BETA1_STO_SI(T)
    theta = dict(a1=b1, a11=BETA11_STO_BRACKET_SI[0], a12=BETA12_STO_BRACKET_SI[0],
                 a111=0.0, a112=0.0, a123=0.0)
    coupling = dict(t11=T11_STO_BRACKET_SI[0], t12=T12_STO_BRACKET_SI[0], t44=T44_STO_SI)
    return dict(P=P, theta=theta, coupling=coupling)


def sto_afd_landau_deck(T):
    """SI -> deck units: P-sector via units.si_to_deck_landau (P unscaled,
    same x1e-9 rule as every other Landau coefficient in this engine);
    theta-sector and coupling via units.si_to_deck_theta_landau with the
    correct theta_power for each term (2 for a1/t's, 4 for a11/a12 --
    theta^4 terms)."""
    si = sto_afd_landau_si(T)

    def conv_P(d):
        return {k: units.si_to_deck_landau(v) for k, v in d.items()}

    def conv_theta(d):
        return dict(
            a1=units.si_to_deck_theta_landau(d["a1"], 2),
            a11=units.si_to_deck_theta_landau(d["a11"], 4),
            a12=units.si_to_deck_theta_landau(d["a12"], 4),
            a111=0.0, a112=0.0, a123=0.0,
        )

    def conv_coupling(d):
        # each t multiplies P^2*theta^2: theta_power=2 (P contributes no factor, unscaled)
        return {k: units.si_to_deck_theta_landau(v, 2) for k, v in d.items()}

    return dict(P=conv_P(si["P"]), theta=conv_theta(si["theta"]), coupling=conv_coupling(si["coupling"]))


STO_TOML = os.path.join(_DECK_DIR, "sto_afd_rabeA.toml")


def load_sto_deck(path=None):
    """Load params/sto_afd_rabeA.toml, remapping its collision-safe TOML
    key names (b1..., t11/t12/t44, Gtheta, Lam11_theta...) onto the
    engine's own ALPHA_KEYS naming for P/theta (solvers/deck_parser.py's
    TOML_SECTIONS additions, S28 block). The returned deck["Q"] uses the BARE
    keys Q11/Q12/Q44 but holds the deck's Q11_phys/Q12_phys/Q44_phys VALUES
    (read below), i.e. the set elastic.setup_elastic wants -- not the legacy
    /4-hacked block SPEC §1.4 forbids; None when the deck has no such block."""
    m = load_params_toml(path or STO_TOML)["materials"]
    P = dict(a1=m["alpha1"], a11=m["alpha11"], a12=m["alpha12"],
              a111=m["alpha111"], a112=m["alpha112"], a123=m["alpha123"])
    theta = dict(a1=m["b1"], a11=m["b11"], a12=m["b12"],
                 a111=m["b111"], a112=m["b112"], a123=m["b123"])
    coupling = dict(t11=m["t11"], t12=m["t12"], t44=m["t44"])
    gradient = dict(G110=m["G110"], G11_G110=m["G11_G110"], G12_G110=m["G12_G110"],
                     G44_G110=m["G44_G110"], G44P_G110=m["G44P_G110"])
    # elastic/Q/Lam_theta are OPTIONAL (SPEC S28.5 R6: elastic stays OFF this
    # WP; a deck that never wires eigenstrain -- e.g. the G22e model
    # composite, which borrows PZT's P-sector and has no x=0.2-specific
    # elastic data anyway -- need not carry them at all).
    elastic_c = (dict(C11=m["C11"], C12=m["C12"], C44=m["C44"])
                 if "C11" in m else None)
    Q = (dict(Q11=m["Q11_phys"], Q12=m["Q12_phys"], Q44=m["Q44_phys"])
         if "Q11_phys" in m else None)
    Lam_theta = (dict(Lam11=m["Lam11_theta"], Lam12=m["Lam12_theta"], Lam44=m["Lam44_theta"])
                 if "Lam11_theta" in m else None)
    return dict(P=P, theta=theta, coupling=coupling, gradient=gradient, Gtheta=m["Gtheta"],
                elastic=elastic_c, Q=Q, Lam_theta=Lam_theta, permittivity=m.get("permittivity"))


# ==========================================================================
# f_Ptheta -- biquadratic coupling menu (SPEC S28.1, sign convention
# SUPERSEDED by S28.5 R2 to Chen's printed minus-prefix form):
# -t11*sum(Pi^2*thetai^2) - t12*sum_{i!=j}(Pi^2*thetaj^2)
# - t44*(P1*P2*theta1*theta2 + P2*P3*theta2*theta3 + P3*P1*theta3*theta1)
# Built symbolically once at import (solver.py's own _F_BULK/lambdify
# pattern, mirrors kittel_afe.py's _F_COUPLE), lambdified. "Force" = d f/dP (or
# dtheta), matching solver.bulk_force's own naming convention -- NOT the
# physical -dF/dP; callers add these into N exactly parallel to bulk_force.
# ==========================================================================
_P1s, _P2s, _P3s, _t1s, _t2s, _t3s = sp.symbols("P1 P2 P3 th1 th2 th3", real=True)
_t11s, _t12s, _t44s = sp.symbols("t11 t12 t44", real=True)
_F_COUPLE = (
    -_t11s * (_P1s**2 * _t1s**2 + _P2s**2 * _t2s**2 + _P3s**2 * _t3s**2)
    - _t12s * (_P1s**2 * _t2s**2 + _P1s**2 * _t3s**2
               + _P2s**2 * _t1s**2 + _P2s**2 * _t3s**2
               + _P3s**2 * _t1s**2 + _P3s**2 * _t2s**2)
    - _t44s * (_P1s * _P2s * _t1s * _t2s + _P2s * _P3s * _t2s * _t3s + _P3s * _P1s * _t3s * _t1s)
)
_COUPLE_ARGS = (_P1s, _P2s, _P3s, _t1s, _t2s, _t3s, _t11s, _t12s, _t44s)
COUPLE_KEYS = ("t11", "t12", "t44")
_f_couple_fn = sp.lambdify(_COUPLE_ARGS, _F_COUPLE, "numpy")
_df_couple_dP_fn = [sp.lambdify(_COUPLE_ARGS, sp.diff(_F_COUPLE, v), "numpy") for v in (_P1s, _P2s, _P3s)]
_df_couple_dtheta_fn = [sp.lambdify(_COUPLE_ARGS, sp.diff(_F_COUPLE, v), "numpy") for v in (_t1s, _t2s, _t3s)]


def couple_energy_density(P, theta, coupling):
    """f_couple(P,theta) from coupling's t11/t12/t44 keys (SPEC §28.1), real space, shape = P[0].shape."""
    args = [P[0], P[1], P[2], theta[0], theta[1], theta[2]] + [coupling[k] for k in COUPLE_KEYS]
    return _f_couple_fn(*args)


def couple_force_P(P, theta, coupling):
    """delta f_couple/delta P_i, shape (3,Ny,Nx)."""
    args = [P[0], P[1], P[2], theta[0], theta[1], theta[2]] + [coupling[k] for k in COUPLE_KEYS]
    return np.stack([fn(*args) for fn in _df_couple_dP_fn])


def couple_force_theta(P, theta, coupling):
    """delta f_couple/delta theta_i, shape (3,Ny,Nx)."""
    args = [P[0], P[1], P[2], theta[0], theta[1], theta[2]] + [coupling[k] for k in COUPLE_KEYS]
    return np.stack([fn(*args) for fn in _df_couple_dtheta_fn])


# ==========================================================================
# f_gradtheta -- isotropic gradient (SPEC S28.1 "isotropic first"): exact
# mirror of kittel_afe.py's f_gradq (no D(k) depolarization coupling for theta --
# binding, SPEC S28.1; no G12/G44/G44p cross terms -- isotropic scope).
# ==========================================================================
def build_ops_theta(grid, Gtheta):
    """Atheta(k) = Gtheta*(kx^2+ky^2), scalar, SAME for all 3 components."""
    return dict(Atheta=operators.isotropic_symbol(grid, Gtheta, 2), Nx=grid["Nx"], Ny=grid["Ny"])


def grad_energy_density_theta(theta, grid, Gtheta):
    """f_gradtheta = 0.5*Gtheta*|grad theta|^2 summed over the 3 components (isotropic, SPEC §28.1), shape (Ny,Nx).

    Even-grid Nyquist convention (solver.grad_energy_density's): collocation
    local-gradient density with REAL skew-adjoint derivatives (each zeroed on
    its own even-axis Nyquist line, solver._modified_wavevectors_2d), plus the
    alternating mode those derivatives drop, added back per even axis j as
    0.5*Gtheta*(pi/h_j)^2*(NyquistProject_j theta_i)^2 -- so the grid mean is
    the energy grad_force_theta (Atheta = Gtheta*k^2) is the gradient of. On
    an odd axis there is no correction: odd grids are bit-identical."""
    KX, KY = operators.masked_wavevectors_2d(grid)
    t1x, t1y = solver._spectral_grad_xy(theta[0], grid, KX, KY)
    t2x, t2y = solver._spectral_grad_xy(theta[1], grid, KX, KY)
    t3x, t3y = solver._spectral_grad_xy(theta[2], grid, KX, KY)
    fg = 0.5 * Gtheta * (t1x**2 + t1y**2 + t2x**2 + t2y**2 + t3x**2 + t3y**2)
    return operators.add_nyquist_correction(fg, theta, grid, Gtheta, 2)


def grad_force_theta(theta, ops_theta):
    """d f_gradtheta / dtheta = Atheta(k) . theta (diagonal-only mirror of solver.grad_force)."""
    s = (ops_theta["Ny"], ops_theta["Nx"])
    tk = _rfft2(theta)
    return _irfft2(ops_theta["Atheta"] * tk, s)


def build_Minv_theta(ops_theta, dt, Ltheta):
    """Mtheta(k) = 1 + dt*Ltheta*Atheta(k), scalar per k-point."""
    return 1.0 + dt * Ltheta * ops_theta["Atheta"]


# ==========================================================================
# f_eltheta -- rotostriction eigenstrain eps0_ij += Lam^theta_ijkl theta_k
# theta_l (SPEC S28.1), PARKED per S28.5 R6 (elastic stays OFF this WP;
# built here only so G22c/G22d(iii) can exercise it as an isolated wiring
# check, mirroring kittel_afe.py's f_elq exactly with Lam_theta replacing Lam).
# Like q's, theta's eigenstrain is ADDED to the base's own static eigenstrain
# (quenched random strain, Vegard), never substituted for it; the base's own
# static is remembered under _STATIC0_KEY so a with_theta_eigenstrain* output
# fed back in as the base does not stack theta's eigenstrain twice.
# ==========================================================================
_STATIC0_KEY = "_eps_star_static_without_theta"


def _add_theta_static(base, eps_theta):
    """Shallow copy of `base` whose eps_star_static is the base's own static
    eigenstrain plus theta's `eps_theta` (same shape). No base static (None
    or key absent): the slot is `eps_theta` itself, bit-for-bit."""
    out = dict(base)
    static0 = base[_STATIC0_KEY] if _STATIC0_KEY in base else base.get("eps_star_static")
    out[_STATIC0_KEY] = static0
    out["eps_star_static"] = eps_theta if static0 is None else static0 + eps_theta
    return out


def with_theta_eigenstrain(base_elastic_params, theta, Lam11, Lam12, Lam44):
    """Copy of base_elastic_params with the rotostriction eigenstrain of theta added to its eps_star_static (SPEC §28.1, PARKED per §28.5 R6)."""
    return _add_theta_static(base_elastic_params, elastic.eigenstrain(theta, Lam11, Lam12, Lam44))


def elastic_force_explicit_theta(theta, sigma, Lam11, Lam12, Lam44):
    """dF_el/dtheta_m -- exact structural mirror of elastic.elastic_force_explicit
    (theta,Lam_theta replacing P,Q). Verified fresh below (SPEC G22c)."""
    t1, t2, t3 = theta[0], theta[1], theta[2]
    s11, s22, s33 = sigma[0, 0], sigma[1, 1], sigma[2, 2]
    s12, s13, s23 = sigma[0, 1], sigma[0, 2], sigma[1, 2]
    f1 = -(2 * t1 * (s11 * Lam11 + (s22 + s33) * Lam12) + Lam44 * (s12 * t2 + s13 * t3))
    f2 = -(2 * t2 * (s22 * Lam11 + (s11 + s33) * Lam12) + Lam44 * (s12 * t1 + s23 * t3))
    f3 = -(2 * t3 * (s33 * Lam11 + (s11 + s22) * Lam12) + Lam44 * (s13 * t1 + s23 * t2))
    return np.stack([f1, f2, f3])


def _verify_elastic_force_symbolic_theta():
    t1, t2, t3, Lam11, Lam12, Lam44 = sp.symbols("t1 t2 t3 Lam11 Lam12 Lam44", real=True)
    s11, s22, s33, s12, s13, s23 = sp.symbols("s11 s22 s33 s12 s13 s23", real=True)
    tv = [t1, t2, t3]
    es = {
        (0, 0): Lam11 * t1**2 + Lam12 * (t2**2 + t3**2),
        (1, 1): Lam11 * t2**2 + Lam12 * (t1**2 + t3**2),
        (2, 2): Lam11 * t3**2 + Lam12 * (t1**2 + t2**2),
        (0, 1): sp.Rational(1, 2) * Lam44 * t1 * t2,
        (0, 2): sp.Rational(1, 2) * Lam44 * t1 * t3,
        (1, 2): sp.Rational(1, 2) * Lam44 * t2 * t3,
    }
    es[(1, 0)], es[(2, 0)], es[(2, 1)] = es[(0, 1)], es[(0, 2)], es[(1, 2)]
    sg = {(0, 0): s11, (1, 1): s22, (2, 2): s33,
          (0, 1): s12, (1, 0): s12, (0, 2): s13, (2, 0): s13, (1, 2): s23, (2, 1): s23}
    explicit = [
        -(2 * t1 * (s11 * Lam11 + (s22 + s33) * Lam12) + Lam44 * (s12 * t2 + s13 * t3)),
        -(2 * t2 * (s22 * Lam11 + (s11 + s33) * Lam12) + Lam44 * (s12 * t1 + s23 * t3)),
        -(2 * t3 * (s33 * Lam11 + (s11 + s22) * Lam12) + Lam44 * (s13 * t1 + s23 * t2)),
    ]
    for m in range(3):
        generic = -sum(sp.diff(es[(i, j)], tv[m]) * sg[(i, j)] for i in range(3) for j in range(3))
        if sp.simplify(generic - explicit[m]) != 0:
            raise AssertionError(f"theta eigenstrain force mismatch at component {m}: "
                                  f"generic={generic} vs explicit={explicit[m]}")


_verify_elastic_force_symbolic_theta()


# ==========================================================================
# Combined (P,theta) explicit driving terms + one semi-implicit step. P's
# own implicit operator (Minv, from solver.build_Minv) is UNCHANGED -- only
# its EXPLICIT term N(P) gains the new coupling force (mirrors kittel_afe.py's
# step_pq exactly, q->theta).
# ==========================================================================
def _explicit_N_P(P, theta, alpha_P, coupling, ops, E_ext, elastic_params):
    """At theta==0 identically, couple_force_P(...) is EXACTLY 0.0 for ANY
    coupling coefficients (every f_Ptheta term has degree >=2 in theta) --
    SPEC G22d(i) holds by this construction, not a special-cased branch."""
    N = solver.bulk_force(P, alpha_P) - solver._reshape_vec3(E_ext, P.ndim, ensemble=P.ndim == 4)
    N = N + couple_force_P(P, theta, coupling)
    if ops.get("dkappa") is not None:
        Ed, Ed_uniform, _n, _resid = solver._spm_iterate(P, ops)
        N = N - (Ed - Ed_uniform)
    if elastic_params is not None:
        Fel, _ = elastic.elastic_force(P, elastic_params)
        N = N + Fel
    return N


def _explicit_N_theta(P, theta, alpha_theta, coupling, Lam_theta=None, elastic_params=None):
    """At P==0 identically, couple_force_theta(...) is EXACTLY 0.0 for any
    coupling coefficients (same even-in-P argument) -- SPEC G22d(ii)."""
    N = solver.bulk_force(theta, alpha_theta) + couple_force_theta(P, theta, coupling)
    if Lam_theta is not None and elastic_params is not None:
        sol = elastic.elastic_solve(P, elastic_params)  # elastic_params already carries theta's eps_star_static
        N = N + elastic_force_explicit_theta(theta, sol["sigma"],
                                              Lam_theta["Lam11"], Lam_theta["Lam12"], Lam_theta["Lam44"])
    return N


def step_ptheta(P, theta, alpha_P, alpha_theta, coupling, ops, Minv, dt, L, E_ext,
                 ops_theta=None, Minv_theta=None, Ltheta=None, base_elastic_params=None, Lam_theta=None):
    """One semi-implicit step for (P,theta) together. ops_theta=None => theta
    is not evolved (returned unchanged; G22d(i)'s theta-frozen-at-0 check)."""
    elastic_params = base_elastic_params
    if base_elastic_params is not None and Lam_theta is not None:
        elastic_params = with_theta_eigenstrain(base_elastic_params, theta,
                                                 Lam_theta["Lam11"], Lam_theta["Lam12"], Lam_theta["Lam44"])

    N_P = _explicit_N_P(P, theta, alpha_P, coupling, ops, E_ext, elastic_params)
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

    if ops_theta is None:
        return P_new, theta

    N_theta = _explicit_N_theta(P, theta, alpha_theta, coupling, Lam_theta=Lam_theta, elastic_params=elastic_params)
    st = (ops_theta["Ny"], ops_theta["Nx"])
    tk = _rfft2(theta)
    Nthk = _rfft2(N_theta)
    tk_new = (tk - dt * Ltheta * Nthk) / Minv_theta
    theta_new = _irfft2(tk_new, st)
    return P_new, theta_new


def total_energy_ptheta(P, theta, alpha_P, alpha_theta, coupling, ops, grid, Gtheta, E_ext=(0., 0., 0.),
                         elastic_params=None, Lam_theta=None):
    """Same shape as solver.total_energy, extended with theta's own
    bulk/grad and the f_Ptheta coupling; elastic folds theta's eigenstrain
    in via with_theta_eigenstrain, consistent with step_ptheta."""
    fb_P = solver.bulk_energy_density(P, alpha_P)
    fb_theta = solver.bulk_energy_density(theta, alpha_theta)
    fg_P = solver.grad_energy_density(P, grid, ops)
    fg_theta = grad_energy_density_theta(theta, grid, Gtheta)
    fc = couple_energy_density(P, theta, coupling)
    fe = solver.elec_energy_density(P, ops, E_ext)
    dens = dict(bulk_P=float(fb_P.mean()), bulk_theta=float(fb_theta.mean()),
                grad_P=float(fg_P.mean()), grad_theta=float(fg_theta.mean()),
                coupling=float(fc.mean()), elec=float(fe.mean()))
    ep = elastic_params
    if ep is not None and Lam_theta is not None:
        ep = with_theta_eigenstrain(ep, theta, Lam_theta["Lam11"], Lam_theta["Lam12"], Lam_theta["Lam44"])
    if ep is not None:
        sol = elastic.elastic_solve(P, ep)
        dens["elastic"] = float(elastic.elastic_energy_density(sol).mean())
    dens["total"] = sum(dens.values())
    tot = {k: v * grid["Volume"] for k, v in dens.items()}
    return dens, tot


# ==========================================================================
# SPEC V2-03 A4 -- the theta sector on the TWO 3D ENGINES (bulk3d: fully
# periodic rfftn box; film3d: 2D-periodic + banded z). The exact mirror of
# solvers/kittel_afe.py's V2-03 A3 block with theta for q -- same Part II
# ruling (3) (two independent solvers, no dimension-generic core), so the
# sector is written ONCE PER ENGINE on that engine's own primitives, and the
# only things reused across dimensions are the genuinely LOCAL (elementwise)
# pieces: couple_energy_density / couple_force_P / couple_force_theta above
# and solver.bulk_energy_density/bulk_force. The 2D functions above are
# untouched. Transforms are the ENGINE's own (bulk3d._rfftn/_irfftn,
# film3d._rfft2/_irfft2), so a caller's bulk3d.FFT_WORKERS/film3d.FFT_WORKERS
# setting reaches the theta channel too.
#
# LAM CONVENTION (same named trap as A3's, SPEC S37.5): the deck prints
# Lam11_theta/Lam12_theta/Lam44_theta in VOIGT form, elastic.eigenstrain's
# convention (eps0_12 = 0.5*Lam44*t1*t2). bulk3d.eigenstrain_3d takes the
# FULL-TENSOR triple with eps0_12 = 2*Lam1212*t1*t2, so
#       Lam1111 = Lam11,  Lam1122 = Lam12,  Lam1212 = Lam44/4
# -- bulk3d's own Voigt-sourced-deck mapping ("Q1212 = Q44_voigt/4"), applied
# ONLY on the bulk3d side (`lam_tensor_theta_3d`, checked symbolically at
# import, not asserted). The film lane needs NO conversion: film_elastic
# builds eps*(P) with elastic.eigenstrain, so with_theta_eigenstrain_film
# passes the Voigt triple straight through and the film's dF_el/dtheta is
# the EXISTING 2D elastic_force_explicit_theta, reused unchanged.
#
# ROTOSTRICTION STAYS PARKED (S28.5 R6): as in 2D, the elastic path below is
# built so the wiring gates can exercise it in isolation, not because the
# STO science path turns it on.
# ==========================================================================

# --------------------------------------------------------------------------
# A4(a) -- bulk3d (fully periodic)
# --------------------------------------------------------------------------
def build_ops_theta_3d(grid, Gtheta):
    """Atheta(k) = Gtheta*(kx^2+ky^2+kz^2), scalar, SAME for all 3 components
    -- build_ops_theta's 3D mirror on bulk3d's rfftn half-spectrum grid. At
    Nz=1 bulk3d's KZr is identically 0.0, so this is BIT-FOR-BIT
    build_ops_theta's own Atheta on the matching 2D grid (K2 `afd_nz1_vs_2d`)."""
    return dict(Atheta=operators.isotropic_symbol(grid, Gtheta, 3),
                Nx=grid["Nx"], Ny=grid["Ny"], Nz=grid["Nz"])


def grad_energy_density_theta_3d(theta, grid, Gtheta):
    """f_gradtheta = 0.5*Gtheta*|grad theta|^2 over the 3 components, shape
    (Nz,Ny,Nx), via bulk3d._spectral_grad_xyz. Same even-grid Nyquist
    convention as grad_energy_density_theta (and bulk3d.
    grad_energy_density_3d): real skew-adjoint derivatives
    (bulk3d._modified_wavevectors_3d) plus, per even axis j,
    0.5*Gtheta*(pi/h_j)^2*(NyquistProject_j theta_i)^2; odd grids
    bit-identical."""
    KX, KY, KZ = operators.masked_wavevectors_3d(grid)
    t1x, t1y, t1z = bulk3d._spectral_grad_xyz(theta[0], grid, KX, KY, KZ)
    t2x, t2y, t2z = bulk3d._spectral_grad_xyz(theta[1], grid, KX, KY, KZ)
    t3x, t3y, t3z = bulk3d._spectral_grad_xyz(theta[2], grid, KX, KY, KZ)
    fg = 0.5 * Gtheta * (t1x**2 + t1y**2 + t1z**2
                         + t2x**2 + t2y**2 + t2z**2
                         + t3x**2 + t3y**2 + t3z**2)
    return operators.add_nyquist_correction(fg, theta, grid, Gtheta, 3)


def grad_force_theta_3d(theta, ops_theta):
    """d f_gradtheta / dtheta = Atheta(k) . theta (diagonal-only mirror of
    bulk3d.grad_force_3d)."""
    s = (ops_theta["Nz"], ops_theta["Ny"], ops_theta["Nx"])
    tk = bulk3d._rfftn(theta)
    return bulk3d._irfftn(ops_theta["Atheta"] * tk, s)


def build_Minv_theta_3d(ops_theta, dt, Ltheta):
    """Mtheta(k) = 1 + dt*Ltheta*Atheta(k), scalar per k-point."""
    return 1.0 + dt * Ltheta * ops_theta["Atheta"]


def lam_tensor_theta_3d(Lam_theta):
    """(Lam1111, Lam1122, Lam1212) for bulk3d.eigenstrain_3d from the deck's
    VOIGT triple -- see this section's LAM CONVENTION block. Returns
    (Lam11, Lam12, Lam44/4)."""
    return Lam_theta["Lam11"], Lam_theta["Lam12"], Lam_theta["Lam44"] / 4.0


def with_theta_eigenstrain_3d(base_elastic_params, theta, Lam_theta):
    """Shallow-copy bulk3d.setup_elastic_3d's output with theta's
    rotostriction eigenstrain in bulk3d's TENSOR convention added to
    `eps_star_static` (the A2 slot; the base's own static is kept, see
    _add_theta_static). Shallow copy, not a fresh setup_elastic_3d call: the
    slot is a plain dict key elastic_solve_3d reads per solve, so rebuilding
    Omega once per step would be pure waste (with_theta_eigenstrain's own
    idiom). At theta==0 identically the eigenstrain is the EXACT zero tensor,
    so with no base static the "eps_star + static" branch reproduces the
    static=None branch bit-for-bit."""
    L1111, L1122, L1212 = lam_tensor_theta_3d(Lam_theta)
    return _add_theta_static(base_elastic_params, bulk3d.eigenstrain_3d(theta, L1111, L1122, L1212))


def elastic_force_explicit_theta_3d(theta, sigma, Lam1111, Lam1122, Lam1212):
    """dF_el/dtheta_m = -sigma_ij * d(eps0_ij)/dtheta_m, FULL TENSOR sum --
    the structural mirror of bulk3d.elastic_force_explicit_3d with
    (theta,Lam_theta) for (P,Q), NOT of the 2D elastic_force_explicit_theta
    above (different eigenstrain convention: the 4*Lam1212 here is the Voigt
    Lam44). Verified fresh at import below."""
    t1, t2, t3 = theta[0], theta[1], theta[2]
    s11, s22, s33 = sigma[0, 0], sigma[1, 1], sigma[2, 2]
    s12, s13, s23 = sigma[0, 1], sigma[0, 2], sigma[1, 2]
    f1 = -(2 * t1 * (s11 * Lam1111 + (s22 + s33) * Lam1122) + 4 * Lam1212 * (s12 * t2 + s13 * t3))
    f2 = -(2 * t2 * (s22 * Lam1111 + (s11 + s33) * Lam1122) + 4 * Lam1212 * (s12 * t1 + s23 * t3))
    f3 = -(2 * t3 * (s33 * Lam1111 + (s11 + s22) * Lam1122) + 4 * Lam1212 * (s13 * t1 + s23 * t2))
    return np.stack([f1, f2, f3])


def _verify_elastic_force_symbolic_theta_3d():
    """Import-time self-check: bulk3d.eigenstrain_3d's OWN full k,l=1..3
    tensor eps0 (transcribed here with Lam for Q), differentiated
    symbolically, must equal elastic_force_explicit_theta_3d's closed form;
    plus the Voigt->tensor shear bridge lam_tensor_theta_3d relies on."""
    t1, t2, t3 = sp.symbols("t1 t2 t3", real=True)
    L1111, L1122, L1212 = sp.symbols("Lam1111 Lam1122 Lam1212", real=True)
    s11, s22, s33, s12, s13, s23 = sp.symbols("s11 s22 s33 s12 s13 s23", real=True)
    tv = [t1, t2, t3]
    es = {
        (0, 0): L1111 * t1**2 + L1122 * (t2**2 + t3**2),
        (1, 1): L1111 * t2**2 + L1122 * (t1**2 + t3**2),
        (2, 2): L1111 * t3**2 + L1122 * (t1**2 + t2**2),
        (0, 1): 2 * L1212 * t1 * t2,
        (0, 2): 2 * L1212 * t1 * t3,
        (1, 2): 2 * L1212 * t2 * t3,
    }
    es[(1, 0)], es[(2, 0)], es[(2, 1)] = es[(0, 1)], es[(0, 2)], es[(1, 2)]
    sg = {(0, 0): s11, (1, 1): s22, (2, 2): s33,
          (0, 1): s12, (1, 0): s12, (0, 2): s13, (2, 0): s13, (1, 2): s23, (2, 1): s23}
    explicit = [
        -(2 * t1 * (s11 * L1111 + (s22 + s33) * L1122) + 4 * L1212 * (s12 * t2 + s13 * t3)),
        -(2 * t2 * (s22 * L1111 + (s11 + s33) * L1122) + 4 * L1212 * (s12 * t1 + s23 * t3)),
        -(2 * t3 * (s33 * L1111 + (s11 + s22) * L1122) + 4 * L1212 * (s13 * t1 + s23 * t2)),
    ]
    for m in range(3):
        generic = -sum(sp.diff(es[(i, j)], tv[m]) * sg[(i, j)] for i in range(3) for j in range(3))
        if sp.simplify(generic - explicit[m]) != 0:
            raise AssertionError(f"3D theta eigenstrain force mismatch at component {m}: "
                                  f"generic={generic} vs explicit={explicit[m]}")
    Lam44 = sp.symbols("Lam44", real=True)
    if sp.simplify(2 * (Lam44 / 4) * t1 * t2 - sp.Rational(1, 2) * Lam44 * t1 * t2) != 0:
        raise AssertionError("lam_tensor_theta_3d: Voigt->tensor shear mapping is not Lam44/4")


_verify_elastic_force_symbolic_theta_3d()


def _explicit_N_P_3d(P, theta, alpha_P, coupling, ops, E_ext, elastic_params):
    """P's explicit driving term on bulk3d = bulk3d._explicit_N_3d's own
    formula PLUS the f_Ptheta coupling force. At theta==0 identically
    couple_force_P is EXACTLY 0.0 for ANY coupling coefficients, so this
    reproduces bulk3d._explicit_N_3d bit-for-bit (K2 `afd_wiring_3d` leg
    (i)).

    SPEC V2-03.C carry-over (d.2): B2 gave bulk3d its own kappa(r)
    electrostatic SPM, so the 2D _explicit_N_P's `dkappa` branch now has a 3D
    twin and it is MIRRORED HERE VERBATIM -- same placement (after the
    coupling force, before the elastic force), same expression
    `N -= (Ed - Ed_uniform)`, and the same bulk3d._spm_solve_3d call
    bulk3d._explicit_N_3d makes (one warm-start cache, one reader). The
    hybrid split of WP5 SS11.1 is untouched: Minv keeps the HOMOGENEOUS
    D(k)/kappa0 operator, the kappa(r) correction is explicit. ops["dkappa"]
    is None (scalar kappa, every pre-V2-03.C caller) skips the branch, so N
    is bit-identical to before -- and at theta == 0 with an array kappa,
    step_ptheta_3d equals bulk3d.step_3d bit-for-bit (K2 `afd_spm_hybrid_3d`
    leg (ii))."""
    N = bulk3d.bulk_force(P, alpha_P) - bulk3d._reshape_vec3_3d(E_ext, P.ndim, ensemble=P.ndim == 5)
    N = N + couple_force_P(P, theta, coupling)
    if ops.get("dkappa") is not None:
        Ed, Ed_uniform, _n, _resid = bulk3d._spm_solve_3d(P, ops)
        N = N - (Ed - Ed_uniform)
    if elastic_params is not None:
        Fel, _ = bulk3d.elastic_force_3d(P, elastic_params)
        N = N + Fel
    return N


def _explicit_N_theta_3d(P, theta, alpha_theta, coupling, Lam_theta=None, elastic_params=None):
    """theta's explicit driving term on bulk3d: its own Devonshire polynomial
    + coupling force + [rotostriction force, if Lam_theta/elastic_params
    given]. At P==0 identically couple_force_theta is EXACTLY 0.0."""
    N = bulk3d.bulk_force(theta, alpha_theta) + couple_force_theta(P, theta, coupling)
    if Lam_theta is not None and elastic_params is not None:
        sol = bulk3d.elastic_solve_3d(P, elastic_params)   # el already carries theta's eps_star_static
        L1111, L1122, L1212 = lam_tensor_theta_3d(Lam_theta)
        N = N + elastic_force_explicit_theta_3d(theta, sol["sigma"], L1111, L1122, L1212)
    return N


def step_ptheta_3d(P, theta, alpha_P, alpha_theta, coupling, ops, Minv, dt, L, E_ext,
                   ops_theta=None, Minv_theta=None, Ltheta=None,
                   base_elastic_params=None, Lam_theta=None):
    """One semi-implicit step for (P,theta) together on bulk3d -- the 3D
    mirror of step_ptheta, built on bulk3d.step_3d's own primitives.
    ops_theta=None => theta is not evolved (returned unchanged), which also
    leaves elastic_params UNMODIFIED when Lam_theta is None."""
    elastic_params = base_elastic_params
    if base_elastic_params is not None and Lam_theta is not None:
        elastic_params = with_theta_eigenstrain_3d(base_elastic_params, theta, Lam_theta)

    N_P = _explicit_N_P_3d(P, theta, alpha_P, coupling, ops, E_ext, elastic_params)
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

    if ops_theta is None:
        return P_new, theta

    N_theta = _explicit_N_theta_3d(P, theta, alpha_theta, coupling,
                                   Lam_theta=Lam_theta, elastic_params=elastic_params)
    st = (ops_theta["Nz"], ops_theta["Ny"], ops_theta["Nx"])
    tk = bulk3d._rfftn(theta)
    Nthk = bulk3d._rfftn(N_theta)
    tk_new = (tk - dt * Ltheta * Nthk) / Minv_theta
    theta_new = bulk3d._irfftn(tk_new, st)
    return P_new, theta_new


def total_energy_ptheta_3d(P, theta, alpha_P, alpha_theta, coupling, ops, grid, Gtheta,
                           E_ext=(0., 0., 0.), elastic_params=None, Lam_theta=None):
    """total_energy_ptheta's 3D mirror: same (densities, totals) contract and
    the same keys, every term via bulk3d's own 3D functions."""
    fb_P = bulk3d.bulk_energy_density(P, alpha_P)
    fb_theta = bulk3d.bulk_energy_density(theta, alpha_theta)
    fg_P = bulk3d.grad_energy_density_3d(P, grid, ops)
    fg_theta = grad_energy_density_theta_3d(theta, grid, Gtheta)
    fc = couple_energy_density(P, theta, coupling)
    fe = bulk3d.elec_energy_density_3d(P, ops, E_ext)
    dens = dict(bulk_P=float(fb_P.mean()), bulk_theta=float(fb_theta.mean()),
                grad_P=float(fg_P.mean()), grad_theta=float(fg_theta.mean()),
                coupling=float(fc.mean()), elec=float(fe.mean()))
    ep = elastic_params
    if ep is not None and Lam_theta is not None:
        ep = with_theta_eigenstrain_3d(ep, theta, Lam_theta)
    if ep is not None:
        sol = bulk3d.elastic_solve_3d(P, ep)
        dens["elastic"] = float(bulk3d.elastic_energy_density_3d(sol).mean())
    dens["total"] = sum(dens.values())
    tot = {k: v * grid["Volume"] for k, v in dens.items()}
    return dens, tot


# --------------------------------------------------------------------------
# A4(b) -- film3d (2D-periodic in-plane + banded z, natural BC)
# --------------------------------------------------------------------------
def build_theta_factors_film(grid, Gtheta, dt, Ltheta):
    """M_theta = I + dt*Ltheta*[theta's gradient-force operator], factorized
    once per (grid,Gtheta,dt,Ltheta) -- the theta twin of
    film3d.build_grad_factors.

    DELEGATED, not re-typed: f_gradtheta = 0.5*Gtheta*|grad theta|^2 is the
    SAME isotropic functional film3d's P gradient is, so film3d's own
    face-based finite-volume z-bands + spectral in-plane k^2, evaluated at
    Gtheta and Ltheta, ARE theta's implicit operator -- natural (Neumann) BC
    included, outermost off-diagonals doubled exactly as _grad_z_bands does.
    Re-typing them here would be a byte-copy of a validated operator (the
    g10e energy-monotonicity derivation), not an independent one."""
    return film3d.build_grad_factors(grid, Gtheta, dt, Ltheta)


def grad_force_theta_film(theta, grid, Gtheta):
    """delta F_gradtheta/delta theta_i in film geometry, applied EXPLICITLY
    and already divided by the trapezoidal node weight qw(z) -- the
    convention film3d's N(P) uses for every nodal term. Same delegation
    argument as build_theta_factors_film."""
    return film3d.grad_force_3d(theta, grid, Gtheta)


def grad_energy_theta_film(theta, grid, Gtheta):
    """TOTAL F_gradtheta (not a density) in film geometry, mirroring
    film3d.total_energy_3d's own F_grad block with Gtheta: in-plane spectral
    derivatives integrated with the trapezoidal qw(z), plus the FACE-based
    z total (G/(2*hz))*sum_faces (t_{j+1}-t_j)^2 -- the exact discrete
    functional grad_force_theta_film/build_theta_factors_film are the
    gradient of."""
    area = grid["Lx"] * grid["Ly"]
    qw, hz = grid["qw"], grid["hz"]
    Ny, Nx = grid["Ny"], grid["Nx"]
    tk = film3d._rfft2(theta)
    dtdx = film3d._irfft2(1j * grid["KXr"][None, None, :, :] * tk, (Ny, Nx))
    dtdy = film3d._irfft2(1j * grid["KYr"][None, None, :, :] * tk, (Ny, Nx))
    fg_inplane = 0.5 * Gtheta * (dtdx**2 + dtdy**2).sum(axis=0)     # (Nz,Ny,Nx)
    F_inplane = area * float(np.sum(qw * fg_inplane.mean(axis=(-2, -1))))
    dtz = theta[:, 1:] - theta[:, :-1]
    F_z = (Gtheta * area / (2 * hz)) * float(np.sum((dtz**2).mean(axis=(-2, -1))))
    return F_inplane + F_z


def with_theta_eigenstrain_film(base_el, theta, Lam_theta):
    """Shallow-copy film_elastic.setup_film_elastic's output with theta's
    rotostriction eigenstrain over the FULL STACK added to the A2
    `eps_star_static` slot (the base's own static is kept, see
    _add_theta_static), theta's part exactly zero in the SUBSTRATE block
    (this lane sources face eigenstrain from the film block only --
    setup_film_elastic refuses anything else, and the zeros written here
    are hard zeros).

    NO Voigt->tensor conversion (unlike with_theta_eigenstrain_3d):
    film_elastic builds eps*(P) with elastic.eigenstrain, the convention the
    deck's Lam11_theta/Lam12_theta/Lam44_theta are printed in."""
    S = np.zeros((3, 3, base_el["Nz"], base_el["Ny"], base_el["Nx"]))
    S[:, :, base_el["film_lo"]:] = elastic.eigenstrain(
        theta, Lam_theta["Lam11"], Lam_theta["Lam12"], Lam_theta["Lam44"])
    return _add_theta_static(base_el, S)


def _explicit_N_P_film(P, theta, alpha_P, coupling, grid, poisson_factors, E_ext, el):
    """P's explicit driving term in film geometry = film3d.explicit_N's own
    (1/qw)-consistent sum PLUS the f_Ptheta coupling force, which is
    POINTWISE and therefore (1/qw)-consistent for free. At theta==0
    identically the addition is EXACTLY 0.0, so this reproduces
    film3d.explicit_N bit-for-bit."""
    N = film3d.explicit_N(P, alpha_P, grid, poisson_factors, E_ext, el=el)
    return N + couple_force_P(P, theta, coupling)


def _explicit_N_theta_film(P, theta, alpha_theta, coupling, grid, Lam_theta=None, el=None):
    """theta's explicit driving term in film geometry, same (1/qw)
    convention: Devonshire + coupling (both pointwise) + [the rotostriction
    force]. The elastic piece needs film3d.explicit_N's OWN hz/qw rescale
    (film_elastic integrates against a UNIFORM hz per node, 1x interior / 2x
    at the two z-boundaries). The force itself is the EXISTING 2D
    elastic_force_explicit_theta, unchanged: film_elastic's sigma_node
    carries the exact d(discrete energy)/d(node) chain rule and theta's
    eigenstrain enters the same face source in the same Voigt convention."""
    N = solver.bulk_force(theta, alpha_theta) + couple_force_theta(P, theta, coupling)
    if Lam_theta is not None and el is not None:
        sol = film_elastic.elastic_solve_film(P, el)   # el already carries theta's eps_star_static
        Fel_t = elastic_force_explicit_theta(theta, sol["sigma_node"], Lam_theta["Lam11"],
                                             Lam_theta["Lam12"], Lam_theta["Lam44"])
        qw_ratio = (grid["hz"] / grid["qw"])[None, :, None, None]
        N = N + Fel_t * qw_ratio
    return N


def step_ptheta_film(P, theta, alpha_P, alpha_theta, coupling, grid, grad_factors,
                     poisson_factors, dt, L, E_ext=(0.0, 0.0, 0.0),
                     theta_factors=None, Ltheta=None, base_el=None, Lam_theta=None):
    """One semi-implicit step for (P,theta) together in film geometry --
    film3d.step's own split (gradient implicit per in-plane mode through the
    banded z solve, everything else explicit), run once for P and once for
    theta. theta_factors=None => theta is not evolved (returned unchanged)
    and elastic_params falls through UNMODIFIED when Lam_theta is None."""
    el = base_el
    if base_el is not None and Lam_theta is not None:
        el = with_theta_eigenstrain_film(base_el, theta, Lam_theta)

    N_P = _explicit_N_P_film(P, theta, alpha_P, coupling, grid, poisson_factors, E_ext, el)
    Pk = film3d._rfft2(P)
    Nk = film3d._rfft2(N_P)
    RHS = np.moveaxis(Pk - dt * L * Nk, 1, 0)      # (Nz,3,Ny,Nxr): z first for the tridiagonal solve
    Pk_new = np.moveaxis(film3d.solve_tridiag_factored(grad_factors, RHS), 0, 1)
    P_new = film3d._irfft2(Pk_new, (grid["Ny"], grid["Nx"]))

    if theta_factors is None:
        return P_new, theta

    N_theta = _explicit_N_theta_film(P, theta, alpha_theta, coupling, grid,
                                     Lam_theta=Lam_theta, el=el)
    tk = film3d._rfft2(theta)
    Nthk = film3d._rfft2(N_theta)
    RHSt = np.moveaxis(tk - dt * Ltheta * Nthk, 1, 0)
    tk_new = np.moveaxis(film3d.solve_tridiag_factored(theta_factors, RHSt), 0, 1)
    theta_new = film3d._irfft2(tk_new, (grid["Ny"], grid["Nx"]))
    return P_new, theta_new


def total_energy_ptheta_film(P, theta, alpha_P, alpha_theta, coupling, grid, kappa_b,
                             G11, Gtheta, poisson_factors=None, E_ext=(0.0, 0.0, 0.0),
                             el=None, Lam_theta=None):
    """(densities, totals) in film geometry, keys as total_energy_ptheta.
    P's bulk/grad/elec [+ elastic] come from film3d.total_energy_3d itself
    (its quadrature conventions are the validated ones); theta's bulk and the
    coupling are nodal, so qw-weighted the same way; grad_theta uses
    grad_energy_theta_film. Densities are totals/Volume3D."""
    if poisson_factors is None:
        poisson_factors = film3d.poisson_factor(grid, kappa_b)
    ep = el
    if el is not None and Lam_theta is not None:
        ep = with_theta_eigenstrain_film(el, theta, Lam_theta)
    _densP, totP = film3d.total_energy_3d(P, alpha_P, grid, kappa_b, G11,
                                          poisson_factors=poisson_factors, E_ext=E_ext, el=ep)
    area = grid["Lx"] * grid["Ly"]
    qw = grid["qw"]
    fb_theta = solver.bulk_energy_density(theta, alpha_theta)
    fc = couple_energy_density(P, theta, coupling)
    tot = dict(bulk_P=totP["bulk"],
               bulk_theta=area * float(np.sum(qw * fb_theta.mean(axis=(-2, -1)))),
               grad_P=totP["grad"], grad_theta=grad_energy_theta_film(theta, grid, Gtheta),
               coupling=area * float(np.sum(qw * fc.mean(axis=(-2, -1)))),
               elec=totP["elec"])
    if "elastic" in totP:
        tot["elastic"] = totP["elastic"]
    tot["total"] = sum(tot.values())
    dens = {k: v / grid["Volume3D"] for k, v in tot.items()}
    return dens, tot
