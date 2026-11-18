"""S1b = WP20 fitter (SPEC S26.2/S27, checkpoint-1 ruling S27.1 R5).

Single-axis {alpha1, alpha11, alpha111} <-> (E_F, E_A, P_max[, chi_AFE]) for the
enlarged-cell single-P triple-well AFE model (SPEC S1.1's f_bulk restricted to
one active component, P2=P3=0): f(P) = alpha1*P^2 + alpha11*P^4 + alpha111*P^6,
field-tilted f(P,E) = f(P) - E*P. ENGINE-INDEPENDENT: this module imports only
numpy/scipy -- no `from solvers import solver` / `elastic`, no reference to the
stepping path (step/relax/build_Minv/...). It has no opinion on off-axis
alpha12/alpha112/alpha123 (SPEC S27.1 R5: those stay FROZEN to a named base
deck by the caller, e.g. params/pbzo_triplewell_yang2025.toml -- this module
never reads or writes them).

Forward relations (closed-form, exact polynomial roots -- see
reports/WP20_STAGE0_NOTE.md for the derivation):
  E(P)  = f'(P) = 2*a1*P + 4*a11*P^3 + 6*a111*P^5      (equilibrium field at P)
  Spinodal (dE/dP=0, u=P^2): 15*a111*u^2 + 6*a11*u + a1 = 0 -> u1<u2
    E_F = E(sqrt(u1))   (first/upper turning field)
    E_A = E(sqrt(u2))   (second/lower turning field)
  P_max(E_max) = largest real root of E(P) = E_max
  chi_AFE       = 1/(2*a1)                               (small-signal, P=0)

AFE-global-minimum constraint (SPEC S27.1 R1/R5): P=0 (AFE) must be the
GLOBAL zero-field minimum, else the fit describes an inverted double-well, not
the intended triple-well. Checked as f(sqrt(u2)) > 0 = f(0) (P=0 below the
upper spinodal point) AND, since that alone misses an FE well beyond P2 when
E_A<0, a111>0 with f>0 at every zero-field stationary point P>0. EXPLICIT,
checked after every fit; violation raises TripleWellConstraintError (loud, not
silent) naming the violating root and its energy.
"""
import numpy as np
from scipy.optimize import root, least_squares

# --------------------------------------------------------------------------
# Forward closed-form relations
# --------------------------------------------------------------------------
def f_energy(P, a1, a11, a111):
    """0-D sextic Landau energy density f(P) = a1*P^2 + a11*P^4 + a111*P^6, deck units."""
    return a1 * P**2 + a11 * P**4 + a111 * P**6


def e_field(P, a1, a11, a111):
    """E(P) = f'(P) -- the equilibrium field at polarization P."""
    return 2 * a1 * P + 4 * a11 * P**3 + 6 * a111 * P**5


def chi_afe(a1):
    """Small-signal susceptibility at P=0, E=0: 1/f''(0) = 1/(2*a1)."""
    return 1.0 / (2.0 * a1)


class TripleWellConstraintError(RuntimeError):
    """Raised when a candidate (a1,a11,a111) fails the AFE-global-minimum
    constraint (P=0 the global zero-field minimum: f(sqrt(u2)) > 0 = f(0) and
    f > 0 at every zero-field stationary point) -- SPEC S27.1 R1/R5, must fail
    LOUDLY."""


def spinodal_points(a1, a11, a111):
    """Real, positive roots (P1<P2) of dE/dP=0 (u=P^2 quadratic
    15*a111*u^2+6*a11*u+a1=0). Returns (P1, P2, u1, u2, discriminant)."""
    D = 9.0 * a11**2 - 15.0 * a111 * a1
    if D < 0 or a111 == 0:
        return None
    sq = np.sqrt(D)
    u1 = (-3.0 * a11 - sq) / (15.0 * a111)
    u2 = (-3.0 * a11 + sq) / (15.0 * a111)
    if u1 <= 0 or u2 <= 0:
        return None
    return float(np.sqrt(u1)), float(np.sqrt(u2)), float(u1), float(u2), float(D)


def thresholds(a1, a11, a111):
    """(E_F, E_A) at the two spinodal turning points, or None if the deck has
    no real S-shaped E(P) (a1<=0, or a11/a111 don't support two turning
    points -- e.g. a plain convex/double-well shape, not this model class)."""
    sp = spinodal_points(a1, a11, a111)
    if sp is None:
        return None
    P1, P2, *_ = sp
    return float(e_field(P1, a1, a11, a111)), float(e_field(P2, a1, a11, a111))


def p_max_at_field(a1, a11, a111, E_max):
    """Largest real root of E(P)=E_max (quintic, companion-matrix via
    np.roots -- exact for a polynomial, no iteration/seed needed)."""
    coeffs = [6 * a111, 0.0, 4 * a11, 0.0, 2 * a1, -E_max]
    roots = np.roots(coeffs)
    real = [r.real for r in roots if abs(r.imag) < 1e-8 * max(1.0, abs(r.real))]
    if not real:
        raise ValueError(f"p_max_at_field: no real root for E_max={E_max} at "
                          f"a1={a1},a11={a11},a111={a111} (roots={roots})")
    return float(max(real))


def _zero_field_stationary_points(a1, a11, a111):
    """Real P>0 where f'(P)=0 at E=0, with f there: [(P, f(P)), ...].
    With u=P^2 these solve 3*a111*u^2+2*a11*u+a1=0 (f'(P)=2P times that).
    For the triple-well sign pattern the larger root is the zero-field FE
    minimum. It lies BEYOND the upper spinodal point P2 whenever E_A<0, so
    there f(FE minimum) < f(P2)."""
    if a111 == 0:
        return []
    disc = a11**2 - 3.0 * a111 * a1
    if disc < 0:
        return []
    sq = np.sqrt(disc)
    roots = ((-a11 - sq) / (3.0 * a111), (-a11 + sq) / (3.0 * a111))
    return [(float(np.sqrt(u)), float(f_energy(np.sqrt(u), a1, a11, a111))) for u in roots if u > 0]


def check_global_min_constraint(a1, a11, a111, label=""):
    """SPEC S27.1 R1/R5: f(sqrt(u2)) > 0 = f(0), i.e. P=0 stays the GLOBAL
    minimum below the upper spinodal point, and nothing beyond it is lower:
    P=0 is the global zero-field minimum. Raises TripleWellConstraintError
    (loud) on violation -- prints the violating root and energies first.

    f(P2) > 0 alone does not make P=0 the global minimum: when E_A<0 the
    zero-field FE minimum lies beyond P2 and is lower. Example: (0.06,
    -0.4932, 1) gives f(P2)=+7.7e-4 but f=-2.0e-4 at P=0.498. So this also
    requires a111>0 (f bounded below) and f>0 at every zero-field
    stationary point P>0 (_zero_field_stationary_points). The f(P2) check
    runs first with its own message, so decks it already rejected fail as
    before, and a passing deck prints the same single line as before."""
    sp = spinodal_points(a1, a11, a111)
    if sp is None:
        print(f"[loopfit] CONSTRAINT VIOLATION{' ('+label+')' if label else ''}: "
              f"no real positive spinodal points at a1={a1:.10g}, a11={a11:.10g}, "
              f"a111={a111:.10g} -- not an S-shaped triple-well candidate at all.")
        raise TripleWellConstraintError(
            f"no real spinodal points (a1={a1}, a11={a11}, a111={a111})")
    P1, P2, u1, u2, D = sp
    f0 = f_energy(0.0, a1, a11, a111)
    f2 = f_energy(P2, a1, a11, a111)
    lower = sorted(((P, f) for P, f in _zero_field_stationary_points(a1, a11, a111)
                    if not f > f0), key=lambda pf: pf[1])
    ok = f2 > f0 and a111 > 0 and not lower
    print(f"[loopfit] global-min constraint{' ('+label+')' if label else ''}: "
          f"P2=sqrt(u2)={P2:.9f}  f(0)={f0:.6e}  f(P2)={f2:.6e}  "
          f"{'PASS' if ok else 'VIOLATION'}")
    if not f2 > f0:
        raise TripleWellConstraintError(
            f"f(sqrt(u2))={f2:.6e} <= f(0)={f0:.6e} at a1={a1}, a11={a11}, a111={a111} "
            f"(violating root P2={P2:.9f}) -- this is an inverted double-well, not a "
            f"triple-well AFE deck; SPEC S27.1 R1/R5 requires this to fail loudly.")
    if not ok:
        where = (f"zero-field stationary point P={lower[0][0]:.9f} has f={lower[0][1]:.6e} "
                 f"<= f(0)={f0:.6e}" if lower else f"a111={a111} <= 0, f unbounded below")
        print(f"[loopfit]   P=0 is not the global minimum: {where}")
        raise TripleWellConstraintError(
            f"{where} at a1={a1}, a11={a11}, a111={a111} although f(sqrt(u2))={f2:.6e} > 0 "
            f"-- the FE well is the global minimum, not P=0 (an inverted double-well, not a "
            f"triple-well AFE deck); SPEC S27.1 R1/R5 requires this to fail loudly.")
    return P2, f2


# --------------------------------------------------------------------------
# Inverse problem: targets -> (a1, a11, a111)
# --------------------------------------------------------------------------
# Six hand-picked (a1, a11, a111) starting points, unsourced: all carry the
# triple-well sign pattern (a1>0, a11<0, a111=-2*a11) at six magnitudes.
# fit() has no RNG, so THIS LIST is its determinism -- it returns the
# lowest-residual constraint-satisfying root over these seeds, so editing or
# reordering it can change which root a given target set fits to (G21a).
_DEFAULT_SEEDS = [
    (0.3, -2.0, 4.0), (0.2, -1.0, 2.0), (0.1, -0.5, 1.0),
    (0.5, -3.0, 6.0), (0.3, -1.5, 3.0), (1.0, -5.0, 10.0),
]


def _residual_3(x, E_F, E_A, P_max, E_max):
    a1, a11, a111 = x
    th = thresholds(a1, a11, a111)
    if th is None:
        # Push the solver away from the infeasible region with a smooth,
        # large-but-finite penalty (no real spinodal points here at all).
        return np.array([1e3, 1e3, 1e3])
    e_f, e_a = th
    try:
        pmax = p_max_at_field(a1, a11, a111, E_max)
    except ValueError:
        return np.array([1e3, 1e3, 1e3])
    return np.array([e_f - E_F, e_a - E_A, pmax - P_max])


def _residual_4(x, E_F, E_A, P_max, E_max, chi_target):
    r3 = _residual_3(x, E_F, E_A, P_max, E_max)
    a1 = x[0]
    r_chi = chi_afe(a1) - chi_target if a1 != 0 else 1e3
    return np.concatenate([r3, [r_chi]])


def fit(E_F, E_A, P_max, E_max, chi_AFE=None, seeds=None, tol=1e-13, verbose=True):
    """Deterministic constrained fit: targets (E_F, E_A, P_max at E_max[,
    chi_AFE]) -> (a1, a11, a111). SPEC S26.0(a): method/targets/residuals are
    all in the returned dict, for the caller to carry into deck [meta].

    E_F/E_A/P_max are MANDATORY (SPEC S26.2); chi_AFE is optional and, if
    given, makes the system 4-eq/3-unknown (least-squares, generally nonzero
    chi residual) rather than the exactly-determined 3-eq/3-unknown case.

    Deterministic: fixed seed list (no RNG), scipy 'hybr'/'trf' (also
    deterministic given a fixed x0). Every candidate solution is checked
    against the AFE-global-minimum constraint (check_global_min_constraint)
    before being accepted; a solution that fails it is DISCARDED (not
    returned) -- if every seed's converged point (or its best least-squares
    point) is a genuine target-consistent constraint violator, this function
    raises TripleWellConstraintError (the SPEC S27.1 R5 rejection path)."""
    seeds = seeds if seeds is not None else _DEFAULT_SEEDS
    overdetermined = chi_AFE is not None
    best = None  # (residual_norm, x)
    last_violation = None

    for x0 in seeds:
        if overdetermined:
            res = least_squares(_residual_4, x0, args=(E_F, E_A, P_max, E_max, chi_AFE),
                                 method="trf", xtol=tol, ftol=tol, gtol=tol, max_nfev=20000)
            resid = _residual_4(res.x, E_F, E_A, P_max, E_max, chi_AFE)
        else:
            sol = root(_residual_3, x0, args=(E_F, E_A, P_max, E_max), method="hybr",
                       tol=tol, options=dict(maxfev=20000))
            res = sol
            resid = _residual_3(sol.x, E_F, E_A, P_max, E_max)
        rnorm = float(np.linalg.norm(resid))
        a1, a11, a111 = (float(v) for v in res.x)
        if not np.all(np.isfinite(res.x)) or rnorm > 1e-6:
            continue  # this seed did not converge to a target-consistent point at all
        try:
            check_global_min_constraint(a1, a11, a111, label=f"seed{seeds.index(x0)}")
        except TripleWellConstraintError as exc:
            last_violation = (a1, a11, a111, rnorm, str(exc))
            continue  # SPEC S27.1 R5: reject, do not silently accept
        if best is None or rnorm < best[0]:
            best = (rnorm, (a1, a11, a111))

    if best is None:
        if last_violation is not None:
            a1, a11, a111, rnorm, msg = last_violation
            print(f"[loopfit] FIT REJECTED: every target-consistent seed violated the "
                  f"AFE-global-minimum constraint (closest: a1={a1:.6g}, a11={a11:.6g}, "
                  f"a111={a111:.6g}, target-residual-norm={rnorm:.3e}). {msg}")
            raise TripleWellConstraintError(
                f"fit(E_F={E_F}, E_A={E_A}, P_max={P_max}, E_max={E_max}, chi_AFE={chi_AFE}): "
                f"no constraint-satisfying solution found across {len(seeds)} deterministic "
                f"seeds -- targets are only consistent with an inverted double-well. {msg}")
        raise RuntimeError(
            f"fit(E_F={E_F}, E_A={E_A}, P_max={P_max}, E_max={E_max}, chi_AFE={chi_AFE}): "
            f"no seed converged to a target-consistent point across {len(seeds)} deterministic seeds.")

    rnorm, (a1, a11, a111) = best
    residuals = dict(zip(
        ("E_F", "E_A", "P_max") + (("chi_AFE",) if overdetermined else ()),
        (_residual_4 if overdetermined else _residual_3)(
            (a1, a11, a111), E_F, E_A, P_max, E_max, *((chi_AFE,) if overdetermined else ()))
    ))
    result = dict(
        a1=a1, a11=a11, a111=a111,
        targets=dict(E_F=E_F, E_A=E_A, P_max=P_max, E_max=E_max, chi_AFE=chi_AFE),
        residuals=residuals, residual_norm=rnorm,
        method="least_squares(trf)" if overdetermined else "root(hybr)",
        overdetermined=overdetermined,
    )
    if verbose:
        print(f"[loopfit] fit: a1={a1:.10g}  a11={a11:.10g}  a111={a111:.10g}  "
              f"residual_norm={rnorm:.3e}  method={result['method']}")
    return result
