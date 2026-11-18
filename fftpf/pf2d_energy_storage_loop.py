"""WP21/E1 instrument (SPEC S29): unipolar energy-storage protocol. Drives
E: 0 -> E_max -> 0, state-continuous, records the (E,<P1>) trace, computes
W_store/W_rec (Yang's own Ue = int E dP definition, main text p.1104,
reports/LIT_yang2025_main.md S7), W_loss, eta. Reuses solver.py's stepping
contract exactly (solver.step/build_Minv unmodified) -- same pattern as
gates/g21b_loop_closure.py's run_sweep, generalized to unipolar + two
DECLARED protocols + energy quadrature (g21b never integrated the loop).
"""
import os
from fftpf._locations import PARAMS_DIR as _DECK_DIR
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from fftpf import pf2d_engine as solver
from fftpf import mat_registry as materials
from fftpf import units

# SPEC S27.3(d)/S29.1(ii): Yang main-model nucleation, "randomized noise
# within 0.001 C m^-2 in magnitude" -- P is UNSCALED (deck=SI numerically,
# units.si_to_deck_landau's own documented premise), so this amplitude
# applies to deck units directly, no conversion factor.
YANG_NOISE_AMPLITUDE_SI = 0.001

PROTOCOL_CLEAN = "clean_relaxation"
PROTOCOL_NOISE = "yang_noise_ic"

# SPEC S36.3 2C / S35.4: DEPENDENCY INVERSION. build_case used to import
# cases/wp20_triplewell_case.py for its deck/grid machinery -- the actual
# g21c circular-import regression root cause (SPEC S35.4: a solvers/
# library module reaching back into a cases/ script for its own
# configuration). These three constants are WP20's own working defaults
# (cases/wp20_triplewell_case.py's KAPPA_B_DEFAULT/PARAMS_TOML/
# EPS0_DECK_BASE, SPEC S27.1 R2 -- that module keeps its own name for it),
# duplicated here as PLAIN NUMBERS/PATHS
# (not imported from cases/) so build_case's DEFAULT behavior stays
# bit-for-bit identical while the dependency direction is inverted
# (solvers/ no longer imports cases/, SPEC S36.2 R5).
_PARAMS_TOML_DEFAULT = os.path.join(_DECK_DIR, "pbzo_triplewell_yang2025.toml")
_KAPPA_B_DEFAULT = 50.0
# SPEC V2-02.1 (Lens 2 U2, ruling D7): one legacy unit = 10*eps0 in deck units;
# kappa_for(m) -> eps_b = 10*m (V2-02.1). It is NOT deck eps0 -- the derived
# mapping is kappa_deck = 0.0088541878128*eps_b (units.relative_to_deck_permittivity),
# so this constant is that mapping evaluated at eps_b = 10, the Ferret original.
# RENAMED from `_EPS0_DECK_BASE` (which asserted eps0); VALUE FROZEN by D7.
_KAPPA_UNIT_EPSB10 = 0.08854187  # WP15/16 convention: kappa_b * this = deck permittivity


def build_case(N=64, delta_nm=1.0, kappa_b=None, params_path=None, *,
               eps_b=None, permittivity_deck=None):
    """Explicit new inputs: eps_b is SI-relative background permittivity;
    permittivity_deck is already converted and used exactly once.
    kappa_b remains the historical 0.08854187 multiplier (not SI).

    SPEC V2-02.1 E14 (2026-09-16): the three inputs are mutually exclusive and
    the LEGACY DEFAULT PATH IS BYTE-IDENTICAL (kappa_b=None -> 50.0 ->
    kappa_for(50.0) = 4.4270935, i.e. eps_b = 500 under the ruled mapping); the
    returned dict only GAINS a `permittivity_convention` key naming which of the
    three was used. K7.unipolar_short / K7.loop_closure pin that.

    Grid + k-space operators for an N x N box at spacing delta_nm, SAME
    deck/grid machinery WP20 uses (SPEC S29's own "basis material") --
    default params_path/kappa_b resolve to WP20's own deck
    (params/pbzo_triplewell_yang2025.toml) and working default (50.0).

    SPEC S36.3 2C: DEPENDENCY INVERSION -- this no longer imports
    cases/wp20_triplewell_case.py (the g21c regression's actual root
    cause, SPEC S35.4); it takes a deck TOML PATH and reads it directly
    via solvers.materials.from_toml (SPEC S36.2, the new unified deck
    reader), exactly mirroring what cases/wp20_triplewell_case.py's own
    load_params()+build_case() computed. kappa_b/params_path default to
    None (resolved to the module-level constants above INSIDE the
    function body, not at def-time in the signature) for the SAME reason
    the previous version deferred to wp20case's attributes at call time:
    defaults that read another module's attribute at DEF time can crash on
    a half-initialized module during solvers/'s lazy import; a call-time
    default has no such window. kappa_b is applied via kappa_for()
    (kappa_b*_KAPPA_UNIT_EPSB10), NOT read from the TOML's own `permittivity`
    field/deck["kappa_b"] -- SPEC S27.3(b)'s documented name-collision: the
    TOML's permittivity value happens to equal kappa_for(50.0) numerically
    for the WP20 deck specifically, but kappa_b is a caller-overridable
    working default (WP20 R2's [25,100] bracket), not always derivable
    from whatever the TOML prints. Verified (gates/g29e_timestep_equivalence.py
    and a direct comparison against cases/wp20_triplewell_case.build_case)
    to reproduce the exact same grid/ops/alpha/kappa, bit-for-bit."""
    if sum(v is not None for v in (kappa_b, eps_b, permittivity_deck)) > 1:
        raise ValueError("choose legacy kappa_b, SI-relative eps_b, or permittivity_deck")
    if permittivity_deck is not None:
        kappa = float(permittivity_deck)
        convention = "deck_absolute"
    elif eps_b is not None:
        kappa = units.relative_to_deck_permittivity(eps_b)
        convention = "SI_relative"
    else:
        kappa_b = _KAPPA_B_DEFAULT if kappa_b is None else kappa_b
        kappa = kappa_for(kappa_b)
        convention = "legacy_0.08854187"
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("permittivity must be finite and positive")
    if params_path is None:
        params_path = _PARAMS_TOML_DEFAULT
    deck = materials.from_toml(params_path)
    L = N * delta_nm
    grid = solver.make_grid(Nx=N, Ny=N, Lx=L, Ly=L)
    ops = solver.build_operators(grid, **deck["gradient"], kappa=kappa)
    return dict(grid=grid, ops=ops, alpha=deck["landau"], kappa_b=kappa_b, kappa=kappa, params=deck,
                permittivity_convention=convention)


def kappa_for(kappa_b):
    """deck permittivity = kappa_b * _KAPPA_UNIT_EPSB10 (WP15/16 convention,
    mirrors cases/wp20_triplewell_case.py's own kappa_for exactly).

    SPEC V2-02.1 (Lens 2 U2): the base unit is 10*eps0 in deck units, so
    kappa_for(m) is eps_b = 10*m, NOT eps_b = m -- kappa_for(50) = 4.4270935 is
    eps_b = 500. For an SI-stated background use
    units.relative_to_deck_permittivity(eps_b) (or build_case(eps_b=...))
    instead; this function is the LEGACY multiplier, kept because every stored
    WP15/16/20/21 number was produced through it (D7 freezes the values)."""
    return kappa_b * _KAPPA_UNIT_EPSB10


def ic_clean(grid, alpha, ops, dt, seed=0, n_equil=2000):
    """Protocol (i), clean relaxation (WP20 G21b precedent, SPEC S29.1(i)):
    small-noise seed, EQUILIBRATED AT E=0 before the ramp starts (mirrors
    gates/g21b_loop_closure.py's own `equilibrate`). This deck has no
    nonzero E=0 stationary point (WP20 R1 triple_well_sanity finding), so
    the equilibrated state is P=0 regardless of seed -- the seed only
    matters insofar as it must be SOME nonzero symmetry-breaking
    perturbation for the stepper's first move, not because the outcome is
    seed-sensitive."""
    P0 = solver.random_smooth_field(grid, 3, amplitude=0.01, kmax_frac=0.3, seed=seed)
    Minv0 = solver.build_Minv(grid, ops, dt, 1.0)
    P = P0.copy()
    for _ in range(n_equil):
        P = solver.step(P, alpha, ops, Minv0, dt, 1.0, (0., 0., 0.))
    return P


def ic_noise_yang(grid, seed=0):
    """Protocol (ii), Yang's noise-IC (SPEC S27.3(d)/S29.1(ii)): ONE-TIME
    randomized noise IS the t=0 state the ramp starts from directly (Yang's
    own words: 'initial polarization nuclei consisted of randomized noise')
    -- NOT pre-equilibrated, unlike protocol (i)."""
    return solver.random_smooth_field(grid, 3, amplitude=YANG_NOISE_AMPLITUDE_SI, kmax_frac=0.3, seed=seed)


def e_path_unipolar(n1, e_max):
    """E values for 0 -> e_max (n1 points) -> 0 (n1 points); the E=0 START
    point is NOT included here (caller's run_unipolar prepends the IC's own
    E=0 state) -- mirrors gates/g21b_loop_closure.py's e_path_half exactly."""
    return np.concatenate([np.linspace(0.0, e_max, n1 + 1)[1:], np.linspace(e_max, 0.0, n1 + 1)[1:]])


def run_unipolar(alpha, ops, Minv, dt, n_dwell, P_init, e_path, record_fields=False):
    """State-continuous unipolar drive: hold each E in e_path for n_dwell
    steps (the 'dwell ladder' rate knob -- larger n_dwell = more
    quasi-static), record (E,<P1>) after each dwell. E=0's own (already-
    equilibrated-or-noise-seeded) state is index 0. record_fields=True also
    keeps the FULL P snapshot at the E_max turning point and the final
    (E=0 return) state, for real-space panels -- off by default (memory)."""
    P = P_init.copy()
    E_full = [0.0]
    P1_full = [float(P[0].mean())]
    n1 = len(e_path) // 2
    P_at_turn = None
    for i, E in enumerate(e_path):
        E_ext = (float(E), 0.0, 0.0)
        for _ in range(n_dwell):
            P = solver.step(P, alpha, ops, Minv, dt, 1.0, E_ext)
        E_full.append(float(E))
        P1_full.append(float(P[0].mean()))
        if record_fields and i == n1 - 1:
            P_at_turn = P.copy()
    out = dict(E_full=np.asarray(E_full), P1_full=np.asarray(P1_full), P_final=P.copy(), n1=n1)
    if record_fields:
        out["P_at_turn"] = P_at_turn
    return out


def energy_integrals(E_full, P1_full, n1):
    """Trapezoid quadrature (SPEC S29.1), branch bookkeeping explicit
    (gated in G23a): E_full/P1_full = [E=0] + charge(n1 pts) + discharge(n1
    pts), length 2*n1+1. W_store = int E dP over the CHARGE branch
    (P increasing 0->P_max as E increases 0->E_max -- np.trapezoid(E,P)
    integrates E "dP" directly, treating P as the independent variable).
    W_rec = Yang's Ue = int[Pr->Pm] E dP (main text p.1104) -- LIMITS
    REVERSED from the actual discharge trajectory's own Pm->Pr direction
    (P decreasing as E decreases), so the raw np.trapezoid(E,P) along the
    trajectory (negative, since dP<0 while E>=0) is NEGATED to match Yang's
    own limit convention and read as a positive recoverable energy."""
    E_chg, P_chg = E_full[:n1 + 1], P1_full[:n1 + 1]
    E_dis, P_dis = E_full[n1:], P1_full[n1:]
    W_store = float(np.trapezoid(E_chg, P_chg))
    W_rec = -float(np.trapezoid(E_dis, P_dis))
    W_loss = W_store - W_rec
    eta = W_rec / (W_rec + W_loss) if (W_rec + W_loss) != 0.0 else float("nan")
    return dict(W_store=W_store, W_rec=W_rec, W_loss=W_loss, eta=eta)


def verified_dt(alpha, ops, grid, e_max, dt0, n1_probe=8, n_dwell_probe=200, T_total=None,
                tol=0.01, max_halvings=10, P_init_fn=None):
    """SPEC S28.6(a)/S29.3: FIXED-TOTAL-PHYSICAL-TIME dt/2 agreement check
    (a fixed-step-count halving loop was found to give a FALSE convergence
    signal in WP18 -- smaller dt integrating less total time, asymptoting
    back toward the unrelaxed seed, not the true trajectory; the amended
    program-wide law, SPEC S28.6(a), forbids that pattern). T_total =
    n1_probe * n_dwell_probe * dt0 by default (the SAME total physical time
    the probe dt integrates, held fixed across every dt tried). Runs a
    SMALL unipolar probe ladder (n1_probe field points each way) at dt vs
    dt/2, comparing final <P1> after the full probe path.

    PARAMETER-BINDING SHIM (SPEC S36.2 R4, WP25 Phase 2C, solvers/timestep.py):
    this signature and every default are UNCHANGED; the body now calls the
    unified solvers.timestep.verified_dt with `skip_probe=True` (this
    implementation never called probe_stable_dt itself -- it starts its own
    halving ladder directly at dt0, exactly reproduced by skip_probe) and a
    custom `run_fn` that reproduces `probe_run` verbatim, including its
    (final_P1, max_P1) two-value return (normalizes the agreement tolerance
    by the trajectory's own max, `eps=1e-9`, matching this function's
    original division-floor exactly -- solvers.timestep.verified_dt's own
    default eps is 1e-12, cases/wp20_triplewell_case.py's shim's own
    convention, NOT this one)."""
    from fftpf import timestep_verify as timestep

    if P_init_fn is None:
        def P_init_fn(seed):
            return solver.random_smooth_field(grid, 3, amplitude=0.01, kmax_frac=0.3, seed=seed)
    if T_total is None:
        T_total = n1_probe * n_dwell_probe * dt0

    def probe_run(dt_try):
        n_dwell = max(1, int(round(T_total / (2 * n1_probe * dt_try))))
        e_path = e_path_unipolar(n1_probe, e_max)
        Minv = solver.build_Minv(grid, ops, dt_try, 1.0)
        P0 = P_init_fn(0)
        out = run_unipolar(alpha, ops, Minv, dt_try, n_dwell, P0, e_path)
        if not np.all(np.isfinite(out["P_final"])):
            return None
        return float(out["P1_full"][-1]), float(np.max(out["P1_full"]))

    dt_v, info = timestep.verified_dt(
        solver, P_init_fn(0), alpha, ops, grid, L=1.0, E_ext=(0., 0., 0.),
        T_total=T_total, dt0=dt0, tol=tol, max_halvings=max_halvings, eps=1e-9,
        skip_probe=True, run_fn=probe_run)
    return dt_v, dict(last_dt=info["last_checked_dt"], verified_dt=info["verified_dt"],
                       agree_relerr=info["agree_relerr"], T_total=info["T_total"])
