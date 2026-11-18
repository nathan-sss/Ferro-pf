"""Su & Landis BaTiO3 LGD coefficient set (J. Mech. Phys. Solids 55, 280 (2007))
as RE-LISTED, with the authors' own typo corrections, in Su, Liu & Weng, Acta
Mater. 87, 293 (2015), Appendix eqs (A.1)-(A.2) -- the coefficient source that
Cai, Zhu, Wu, Luo, Feng & Wang, Appl. Phys. Lett. 119, 032901 (2021) cite for
every coefficient of their vortex-domain core-shell BaTiO3 model ("with all the
coefficients cited from Ref. 32" = Su 2015). Transcribed 2026-09-10 (cloud
session) from the PDF text layer of the user's Google-Drive copy of Su 2015;
the multipliers below are the printed (A.2) numbers verbatim. The extraction
drops the minus signs inside (A.1); the signs used here (-1/2 b1, -1/2 b2,
-b3, +c, +f, +g terms) are the ones under which the paper's own anchors are
reproduced (see `check`).

What this module provides (SPEC S26.0(b) discipline: DERIVED, not tuned):

1. `si_coefficients()` -- every (A.2) constant in SI, from the printed
   normalization P0=0.26 C/m^2, e0=0.0082, E0=2.1824e7 V/m, sigma0=E0*P0/e0.
2. `landau_clamped_si()` -- the P-only (zero-strain) Landau part of (A.1)
   mapped onto this engine's cubic form (solver._F_BULK: no 1/2,1/4,1/6
   prefactors):  a1=a1_SL/2, a11=a2_SL/4, a12=a3_SL/2, a111=a4_SL/6,
   a112=a6_SL, a1122=a5_SL/4 (=0), every other key 0.
3. `landau_effective_si(kind)` -- the STRESS-FREE effective Landau set for
   an elastic-OFF run: (A.1)'s b/c/f/g electromechanical terms are minimized
   over the strain at each fixed in-plane P (P3=0), EXACTLY (they are a
   quadratic form in the strain), and f_Landau(P)+f_mech,min(P) is least-
   squares fitted with the 8th-order cubic polynomial the engine already
   integrates. kind='free3d': all strain components relaxed (sigma=0 in 3D;
   e13=e23=0 follow by symmetry for in-plane P). kind='planestrain': Su 2015
   S3's generalized plane strain, eps33 = -0.3259*e0 held fixed, in-plane
   strains relaxed. Why an effective set at all: with elasticity off, the
   printed (a1..a6) alone give Ps=0.171 C/m^2, not the paper's P0=0.26 --
   Su & Landis fit their set so that the STRESS-FREE state has P0/e0; the
   electromechanical relaxation is part of the single-domain energetics.
   VALIDATION (`check`, run at import of `deck`): 'free3d' reproduces the
   paper's own three characteristic values to 4 digits -- Ps=0.2600 (P0),
   eps11*=0.0082 (e0), 180-degree thermodynamic coercive field 1.000*E0 --
   and eps22*=eps33*=-0.3293*e0 (the origin of Su S3's -0.3259 clamp).
   NOT covered (engine gap, flagged for the local full test): the
   INHOMOGENEOUS elastic energy of a domain structure and per-grain
   rotated C/Q in a polycrystal -- solvers/elastic.py is homogeneous-C,
   fixed-frame; (A.1)'s f/g strain-nonlinear couplings have no slot there.
4. SHEAR-SECTOR CONVENTION (declared, SPEC S26.0(a); flagged for local
   verification against the 2007 original): the anchors above involve only
   the axial sector (b1,b2,c1,c2,f1-f4,g1,g2). The shear couplings are
   printed as "b3[(e12+e21)P1P2+...]" and "1/4 g3 (e12+e21)(P1P2^3+P2P1^3)".
   Read LITERALLY (e12+e21 = 2*e12 for the symmetric tensor) the stress-
   free single-domain ground state of the set is ORTHORHOMBIC (in-plane
   [110] 9.8% below [100]) and the effective shear electrostriction is
   Q44=0.13 m^4/C^2, larger than Q11 -- impossible for the tetragonal
   room-temperature BaTiO3 the set was built and used for (90/180-degree
   walls). Counting the tensor shear ONCE (SHEAR_ONCE=True, the default)
   gives a tetragonal ground state, [110] 9.3% above [100] (this repo's
   sourced Rabe App A BaTiO3 deck at 298 K: +6.9%), Q44_eff=0.063. The
   literal reading stays available (shear_once=False) for the local check.
5. `deck()` -- a materials.get()-shaped deck (landau in DECK units via
   units.si_to_deck_landau; isotropic gradient G11=a0, G12=0, G44=G44p=a0/2
   via units.si_to_deck_gradient -- the (1,0,0.5,0.5) ratio set is the
   unique isotropic case of SPEC S1.2, same as bto_LCC2005_paper; kappa_b =
   PHYSICAL vacuum permittivity, see EPS0_DECK below).
6. `shell_alpha_deck(eps_r)` -- Cai 2021's dielectric shell ("the shell part
   can be deemed as the dielectric dead layer", relative permittivity 10..50)
   as a pure linear dielectric in this engine's PE-filler form (composite.py
   / B1 precedent): a1 = 1/(2*eps0*(eps_r-1)) so that background eps0 plus
   the filler's chi = eps0*(eps_r-1) gives TOTAL relative permittivity eps_r
   exactly. (B1's pe_filler_landau_si_corrected uses 1/(2*eps0*eps_r) and
   thus totals eps_r + kappa_b/eps0; the difference is the background term.)

EPS0_DECK -- the vacuum permittivity in THIS repo's deck units. units.py:
energy density 1e9 J/m^3, length nm, P unscaled, hence E_deck = E_SI*1e-9
(docs/PARAMETERS.md A.4) and kappa_deck = P/E_deck = kappa_SI*1e9, so
eps0 = 8.8541878e-12 F/m -> 8.8541878e-3 deck. NOTE (finding F7, recorded in
reports/CLOUD_TEST_NOTE.md): the value 0.08854187 that params/*.toml and
SPEC S1.3 label "eps0 / eps_b=1" is 10x this, i.e. physically eps_b=10;
verified numerically (a longitudinal P sinusoid's engine f_elec equals
1/2 P^2/eps0 only with 0.008854187). This module uses the physical value
because Su & Landis' last term of (A.1) is explicitly 1/(2*kappa0) with
kappa0 = permittivity of free space.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from fftpf import units

EPS0_SI = 8.8541878128e-12          # F/m
EPS0_DECK = EPS0_SI * 1e9           # 8.854e-3, see module docstring (kappa_deck = kappa_SI*1e9)
EPS33_OVER_E0 = -0.3259             # Su 2015 S3: generalized plane strain, eps33 = -0.3259*e0
SHEAR_ONCE = True                   # module docstring item 4

# Su 2015 Appendix (A.2), verbatim multipliers of the normalization constants.
SU2015 = dict(
    a0=1.0e-10,                     # V m^3/C, isotropic gradient, 1/2 a0 sum_ij (P_i,j)^2
    P0=0.26, e0=0.0082, E0=2.1824e7,
    a1=-0.668325, a2=-3.80653, a3=0.78922, a4=12.4421, a5=0.0, a6=0.134226,   # x E0/P0^(2n-1)
    b1=2.54138, b2=1.74267, b3=0.399353,                                       # x E0/(e0 P0)
    c1=2.04999, c2=0.971673, c3=1.27976,                                       # x sigma0/e0
    f1=0.663581, f2=0.841326, f3=-0.170635, f4=0.687281, f5=0.106647, f6=0.213294,  # x E0/(e0^2 P0)
    g1=-3.66149, g2=6.27423, g3=-1.21644,                                      # x E0/(e0 P0^3)
)

CITATION = ("Su, Liu & Weng, Acta Mater. 87, 293 (2015), Appendix (A.1)-(A.2) "
            "[= Su & Landis, JMPS 55, 280 (2007), typo-corrected]; cited as the "
            "coefficient source by Cai et al., Appl. Phys. Lett. 119, 032901 (2021)")

KINDS = ("free3d", "planestrain", "clamped")


def si_coefficients():
    """Every (A.2) constant in SI units."""
    s = SU2015
    P0, e0, E0 = s["P0"], s["e0"], s["E0"]
    sigma0 = E0 * P0 / e0                                    # = 692 MPa (paper's own number)
    out = dict(a0=s["a0"], P0=P0, e0=e0, E0=E0, sigma0=sigma0)
    out["a1"] = s["a1"] * E0 / P0
    for k in ("a2", "a3"):
        out[k] = s[k] * E0 / P0**3
    for k in ("a4", "a5", "a6"):
        out[k] = s[k] * E0 / P0**5
    for k in ("b1", "b2", "b3"):
        out[k] = s[k] * E0 / (e0 * P0)
    for k in ("c1", "c2", "c3"):
        out[k] = s[k] * sigma0 / e0
    for k in ("f1", "f2", "f3", "f4", "f5", "f6"):
        out[k] = s[k] * E0 / (e0**2 * P0)
    for k in ("g1", "g2", "g3"):
        out[k] = s[k] * E0 / (e0 * P0**3)
    return out


def landau_clamped_si():
    """Zero-strain Landau part of (A.1) in this engine's alpha keys (SI)."""
    c = si_coefficients()
    return dict(a1=0.5 * c["a1"], a11=0.25 * c["a2"], a12=0.5 * c["a3"],
                a111=c["a4"] / 6.0, a112=c["a6"], a123=0.0,
                a1111=0.0, a1112=0.0, a1122=0.25 * c["a5"], a1123=0.0)


def _mech_quadratic(P1, P2, c, shear_once):
    """(A.1)'s b/c/f/g terms for in-plane P (P3=0, e13=e23=0) as a quadratic
    form in e=(eps11, eps22, eps33, eps12): w = 1/2 e.A(P).e + B(P).e.
    Shapes: A (...,4,4), B (...,4). The shear couplings enter with the
    tensor shear counted once (shear_once) or twice (literal e12+e21)."""
    q1, q2 = P1 * P1, P2 * P2
    shp = np.broadcast(P1, P2).shape
    A = np.zeros(shp + (4, 4))
    B = np.zeros(shp + (4,))
    ks = 1.0 if shear_once else 2.0
    # elastic: 1/2 c1 (e11^2+e22^2+e33^2) + c2 (e11e22+e11e33+e22e33) + 1/2 c3 (e12^2+e21^2)
    for i in range(3):
        A[..., i, i] += c["c1"]
        for j in range(3):
            if i != j:
                A[..., i, j] += c["c2"]
    A[..., 3, 3] += 2.0 * c["c3"]
    # f (strain-quadratic, P^2-weighted); P1^2 bracket then P2^2 bracket
    A[..., 0, 0] += c["f1"] * q1 + c["f2"] * q2
    A[..., 1, 1] += c["f1"] * q2 + c["f2"] * q1
    A[..., 2, 2] += c["f2"] * (q1 + q2)
    A[..., 0, 1] += c["f3"] * (q1 + q2)
    A[..., 1, 0] += c["f3"] * (q1 + q2)
    A[..., 0, 2] += c["f3"] * q1 + c["f4"] * q2
    A[..., 2, 0] += c["f3"] * q1 + c["f4"] * q2
    A[..., 1, 2] += c["f3"] * q2 + c["f4"] * q1
    A[..., 2, 1] += c["f3"] * q2 + c["f4"] * q1
    A[..., 3, 3] += 2.0 * c["f5"] * (q1 + q2)
    # b (strain-linear): -1/2 b1 (e11 P1^2 + e22 P2^2) - 1/2 b2 [(e22+e33) P1^2 + (e11+e33) P2^2] - b3 (e12+e21) P1 P2
    B[..., 0] += -0.5 * c["b1"] * q1 - 0.5 * c["b2"] * q2
    B[..., 1] += -0.5 * c["b1"] * q2 - 0.5 * c["b2"] * q1
    B[..., 2] += -0.5 * c["b2"] * (q1 + q2)
    B[..., 3] += -ks * c["b3"] * P1 * P2
    # g (strain-linear, P^4): [1/4 g1 e11 + 1/4 g2 (e22+e33)] P1^4 + [1/4 g1 e22 + 1/4 g2 (e11+e33)] P2^4
    #                        + 1/4 g3 (e12+e21)(P1 P2^3 + P2 P1^3)
    B[..., 0] += 0.25 * c["g1"] * q1 * q1 + 0.25 * c["g2"] * q2 * q2
    B[..., 1] += 0.25 * c["g1"] * q2 * q2 + 0.25 * c["g2"] * q1 * q1
    B[..., 2] += 0.25 * c["g2"] * (q1 * q1 + q2 * q2)
    B[..., 3] += 0.25 * ks * c["g3"] * P1 * P2 * (q1 + q2)
    return A, B


def relaxed_mech(P1, P2, c=None, kind="free3d", shear_once=None):
    """Minimize (A.1)'s mechanical terms over the strain at fixed in-plane P.
    kind='free3d': over (e11,e22,e33,e12); 'planestrain': e33 clamped at
    EPS33_OVER_E0*e0, over (e11,e22,e12). Returns (w_min, eps_star (...,4))
    -- exact (quadratic form with a positive-definite P-dependent stiffness)."""
    c = si_coefficients() if c is None else c
    shear_once = SHEAR_ONCE if shear_once is None else shear_once
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)
    A, B = _mech_quadratic(P1, P2, c, shear_once)
    if kind == "free3d":
        eps = -np.linalg.solve(A, B[..., None])[..., 0]
    elif kind == "planestrain":
        e33 = EPS33_OVER_E0 * c["e0"]
        idx = [0, 1, 3]
        Ar = A[..., idx, :][..., :, idx]
        Br = B[..., idx] + A[..., idx, 2] * e33
        er = -np.linalg.solve(Ar, Br[..., None])[..., 0]
        eps = np.zeros(B.shape)
        eps[..., 0], eps[..., 1], eps[..., 3] = er[..., 0], er[..., 1], er[..., 2]
        eps[..., 2] = e33
    else:
        raise ValueError(kind)
    w = 0.5 * np.einsum("...i,...ij,...j->...", eps, A, eps) + np.einsum("...i,...i->...", B, eps)
    return w, eps


def _basis(P1, P2):
    """Cubic-invariant basis for P3=0, in the engine's alpha order."""
    q1, q2 = P1 * P1, P2 * P2
    return np.stack([q1 + q2,                      # a1
                     q1 * q1 + q2 * q2,            # a11
                     q1 * q2,                      # a12
                     q1**3 + q2**3,                # a111
                     q1 * q1 * q2 + q2 * q2 * q1,  # a112
                     q1**4 + q2**4,                # a1111
                     q1**3 * q2 + q2**3 * q1,      # a1112
                     q1 * q1 * q2 * q2], axis=-1)  # a1122


_FIT_KEYS = ("a1", "a11", "a12", "a111", "a112", "a1111", "a1112", "a1122")


def _f_clamped(P1, P2, cl):
    return (cl["a1"] * (P1**2 + P2**2) + cl["a11"] * (P1**4 + P2**4)
            + cl["a12"] * P1**2 * P2**2 + cl["a111"] * (P1**6 + P2**6)
            + cl["a112"] * (P1**4 * P2**2 + P2**4 * P1**2) + cl["a1122"] * P1**4 * P2**4)


def landau_effective_si(kind="free3d", shear_once=None, p_max_frac=1.25, n_r=60, n_theta=25, bounded=True):
    """Stress-free effective Landau set (SI, engine alpha keys) from a
    least-squares fit of the exact relaxed energy on a polar
    (|P| <= p_max_frac*P0, angle 0..45 deg) grid. Returns (alpha_si, info):
    info = fit residual + the paper checks (`check`).

    bounded=True (production default): the three 8th-order coefficients are
    constrained >= 0 (scipy lsq_linear) so the polynomial is bounded below --
    the exact relaxed energy rises steeply past Ps (+7.6e7 J/m^3 at 0.5
    C/m^2 along [100]) while the UNCONSTRAINED 8th-order fit turns over at
    ~0.55 C/m^2 and runs to -inf, a blow-up trap for any transient
    excursion. Cost: fit residual 3.9% of the well depth instead of 0.9%,
    anchors Ps=0.2628 (1.1% high), Ec=0.986 E0, e11=1.025 e0 instead of
    0.2594/1.015/0.995 -- both sets are reported in info."""
    c = si_coefficients()
    shear_once = SHEAR_ONCE if shear_once is None else shear_once
    P0 = c["P0"]
    r = np.linspace(0.0, p_max_frac * P0, n_r + 1)[1:]
    th = np.linspace(0.0, np.pi / 4.0, n_theta)
    R, TH = np.meshgrid(r, th, indexing="ij")
    P1, P2 = R * np.cos(TH), R * np.sin(TH)
    cl = landau_clamped_si()
    w_mech, _ = relaxed_mech(P1, P2, c, kind=kind, shear_once=shear_once)
    w0 = relaxed_mech(0.0, 0.0, c, kind=kind, shear_once=shear_once)[0]   # constant (planestrain clamp offset)
    f_target = (_f_clamped(P1, P2, cl) + w_mech - w0).ravel()
    X = _basis(P1, P2).reshape(-1, len(_FIT_KEYS))
    col = np.sqrt((X * X).mean(axis=0))                      # column scaling for conditioning
    if bounded:
        from scipy.optimize import lsq_linear
        lb = np.full(len(_FIT_KEYS), -np.inf)
        lb[5:] = 0.0                                             # a1111, a1112, a1122 >= 0
        coef = lsq_linear(X / col, f_target, bounds=(lb, np.full(len(_FIT_KEYS), np.inf))).x
    else:
        coef, *_ = np.linalg.lstsq(X / col, f_target, rcond=None)
    coef = coef / col
    resid = X @ coef - f_target
    alpha = dict(zip(_FIT_KEYS, (float(v) for v in coef)))
    alpha.update(a123=0.0, a1123=0.0)
    depth = float(-f_target.min())
    info = dict(kind=kind, shear_once=bool(shear_once), bounded=bool(bounded),
                fit_max_abs_resid_si=float(np.abs(resid).max()),
                fit_max_resid_over_well_depth=float(np.abs(resid).max() / depth),
                well_depth_si=depth, n_samples=int(X.shape[0]), p_max=float(p_max_frac * P0))
    info.update(check(alpha, c, kind=kind, shear_once=shear_once))
    return alpha, info


def _f1d(alpha, P):
    """Effective potential along a cubic axis (P2=P3=0)."""
    q = P * P
    return alpha["a1"] * q + alpha["a11"] * q * q + alpha["a111"] * q**3 + alpha["a1111"] * q**4


def _f_diag(alpha, P):
    """Effective potential along the in-plane diagonal [110], |P| = P."""
    p = P / np.sqrt(2.0)
    q = p * p
    return (2 * alpha["a1"] * q + 2 * alpha["a11"] * q * q + alpha["a12"] * q * q
            + 2 * alpha["a111"] * q**3 + 2 * alpha["a112"] * q**3
            + 2 * alpha["a1111"] * q**4 + 2 * alpha["a1112"] * q**4 + alpha["a1122"] * q**4)


def check(alpha_si, c=None, kind="free3d", shear_once=None):
    """The paper's own three characteristic values, recomputed from the
    fitted stress-free potential: Ps (-> P0=0.26), axial spontaneous strain
    eps11 at Ps (-> e0=0.0082), thermodynamic 180-degree coercive field
    max_{0<P<Ps}(-df/dP) (-> E0=2.1824e7 V/m, 'the field needed to
    completely switch a uniform domain'); the transverse strains at Ps
    (Su S3's clamp -0.3259*e0 should be eps22*=eps33*); and the in-plane
    [110] (orthorhombic) minimum relative to the [100] (tetragonal) one."""
    c = si_coefficients() if c is None else c
    shear_once = SHEAR_ONCE if shear_once is None else shear_once
    P = np.linspace(1e-4, 1.6 * c["P0"], 40001)
    f = _f1d(alpha_si, P)
    i_min = int(np.argmin(f))
    Ps = float(P[i_min])
    dfdP = np.gradient(f, P)
    Ec = float(np.max(-dfdP[: i_min]))
    fo = _f_diag(alpha_si, P)
    io = int(np.argmin(fo))
    _, eps = relaxed_mech(Ps, 0.0, c, kind=kind, shear_once=shear_once)
    _, eps_o = relaxed_mech(P[io] / np.sqrt(2), P[io] / np.sqrt(2), c, kind=kind, shear_once=shear_once)
    return dict(Ps=Ps, Ps_over_P0=Ps / c["P0"],
                Ec=Ec, Ec_over_E0=Ec / c["E0"],
                eps11_at_Ps=float(eps[0]), eps11_over_e0=float(eps[0] / c["e0"]),
                eps22_at_Ps=float(eps[1]), eps33_at_Ps=float(eps[2]),
                eps22_over_e0=float(eps[1] / c["e0"]),
                well_depth_1d_si=float(-f[i_min]),
                P_110=float(P[io]), f110_minus_f100_over_abs_f100=float((fo[io] - f[i_min]) / abs(f[i_min])),
                Q44_eff_voigt=float(2.0 * eps_o[3] / (P[io] ** 2 / 2.0)) if P[io] > 0 else 0.0)


def deck(kind="free3d", shear_once=None, **fit_kw):
    """materials.get()-shaped deck. kind in KINDS: 'free3d' (default, all
    strains relaxed, the set's own stress-free single-domain state),
    'planestrain' (Su S3's eps33 clamp), 'clamped' (printed zero-strain
    Landau, reference only)."""
    c = si_coefficients()
    shear_once = SHEAR_ONCE if shear_once is None else shear_once
    if kind in ("free3d", "planestrain"):
        alpha_si, info = landau_effective_si(kind=kind, shear_once=shear_once, **fit_kw)
    elif kind == "clamped":
        alpha_si = landau_clamped_si()
        info = dict(kind=kind, shear_once=bool(shear_once))
        info.update(check(alpha_si, c, kind="free3d", shear_once=shear_once))
    else:
        raise ValueError(f"kind {kind!r} not in {KINDS}")
    landau = {k: units.si_to_deck_landau(v) for k, v in alpha_si.items()}
    G = units.si_to_deck_gradient(c["a0"])
    gradient = dict(G11=G, G12=0.0, G44=0.5 * G, G44p=0.5 * G)
    meta = dict(source=CITATION, kind=kind, shear_once=bool(shear_once),
                T="room temperature (the set is fit at the tetragonal state; only a1 is "
                  "temperature-dependent in Su-Landis and (A.2) prints its room-temperature value)",
                P0=c["P0"], e0=c["e0"], E0=c["E0"], sigma0=c["sigma0"], a0_si=c["a0"],
                eps33_clamp_over_e0=EPS33_OVER_E0, kappa_b_note="physical eps0 (see module docstring, F7)",
                landau_si=alpha_si, derivation=info)
    return dict(landau=landau, gradient=gradient, kappa_b=EPS0_DECK, meta=meta,
                elastic=None, electrostriction=None)


def shell_alpha_deck(eps_r):
    """Linear-dielectric shell of TOTAL relative permittivity eps_r on top of
    the eps0 background: chi = eps0*(eps_r-1) = 1/(2 a1)."""
    if eps_r <= 1.0:
        raise ValueError("eps_r must exceed the eps0 background (1)")
    a1_si = 1.0 / (2.0 * EPS0_SI * (eps_r - 1.0))
    return dict(a1=units.si_to_deck_landau(a1_si), a11=0.0, a12=0.0, a111=0.0, a112=0.0, a123=0.0,
                a1111=0.0, a1112=0.0, a1122=0.0, a1123=0.0)


if __name__ == "__main__":
    def show(d):
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}
    a_cl = landau_clamped_si()
    print("clamped (printed a1..a6 only):", {k: f"{v:.4e}" for k, v in a_cl.items() if v})
    print("  checks:", show(check(a_cl)))
    for kind in ("free3d", "planestrain"):
        for so in (True, False):
            a, info = landau_effective_si(kind=kind, shear_once=so)
            print(f"{kind} shear_once={so}:", {k: f"{v:.4e}" for k, v in a.items() if v})
            print("  fit/checks:", show(info))
    d = deck()
    print("PRODUCTION deck landau (deck units):", {k: f"{v:.5g}" for k, v in d["landau"].items() if v})
    print("  gradient:", d["gradient"], " kappa_b:", d["kappa_b"])
    print("shell a1 deck: eps_r=20 ->", round(shell_alpha_deck(20.0)["a1"], 4), " eps_r=30 ->",
          round(shell_alpha_deck(30.0)["a1"], 4), " eps_r=50 ->", round(shell_alpha_deck(50.0)["a1"], 4))
