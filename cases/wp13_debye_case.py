"""WP13 (SPEC.md §19, R2) shared case: quadratic-only alpha (a1=0.06, all
higher orders 0 -- G16a's own precedent), real bto_298K.toml grad/depol
params, N=32/h=0.2nm/dt=0.015/T=298K/L=1. Bulk force is EXACTLY linear
(2*a1*P), so the k=0 (homogeneous) mode -- A(0)=D(0)=0 identically, SPEC
§1.2/1.3 -- is an EXACTLY solvable, single-relaxation-time (Debye) scalar
Ornstein-Uhlenbeck process with stiffness=2*a1, tau=1/(L*stiffness),
chi0=1/stiffness: the analytic reference gates G17a/b/c all validate the
chi(omega) instrument against.

Shared by gates/g17a_two_route_agreement.py, g17b_debye_limit.py,
g17c_kk_consistency.py (SPEC §3 cases/ convention: "one file per case,
params extracted... + IC builder" -- reused rather than tripled because
the ensemble/multiprocessing/stationarity-gating/normalization-convention
code is identical and delicate; the three gates differ only in what they
DO with the same reference system's data, exactly the WP12 g16a/d
precedent of several gates sharing one physical setup).

Observable convention (reports/WP13_REPORT.md "Design notes" has the full
derivation): "the homogeneous (k=0) polarization" is tracked as the RAW
rfft2 DC coefficient Pk_raw = P[0].sum() (real; DC of a real field's FFT
has zero imaginary part) -- the SAME convention gate G16a validated
Var(Pk_raw)=kT/stiffness against (NOT the real-space mean P[0].mean(),
which differs by a factor of (Nx*Ny) and would need a rescaled "effective
kT" to satisfy the same FDT formula -- confirmed by direct derivation and
numerical test during WP13 pilot work, see the report). For the AC route,
a spatially uniform E_ext couples to this SAME raw coefficient with an
effective amplitude of Nx*Ny*E0 (traced directly through _explicit_N's
real-space "-E_ext" term and its FFT), so `run_ac_seed` divides the
lock-in projection by Nx*Ny*E0, not E0.

Ensemble sizing (empirically calibrated, not guessed -- see
reports/WP13_REPORT.md "Design notes" for the pilot studies this was
derived from): a SINGLE long trajectory's Welch-segment averaging alone
was far short of the SPEC's 2% tolerances (15-80% raw errors at 720 tau);
what closes the gap is ensemble-averaging across N_SEEDS independent
trajectories (genuinely independent draws, cheap to parallelize across
cores) on top of within-trajectory segment averaging and a modest
frequency-neighborhood pool. 14 seeds x 9e6 steps (16200 tau/seed) is the
configuration used, N_SEEDS matching this machine's 14 physical cores
(one seed/core, no oversubscription).

RECOVERY-SESSION REVISION (this run of WP13, see WP13_REPORT.md "Design
notes"): the figure "14 seeds x 6e6 steps" and "MAX_LAG=60*TAU/DT" quoted
above/below in an earlier uncommitted draft were an in-progress calibration
whose own G17a run was never completed before that session was recycled --
i.e. NOT independently verified against the actual SPEC gate tolerances.
Running G17a for real in the recovery session found chi'' solid everywhere
(<3.5%, confirms machinery/signs/normalization) but chi' failing at
X>~0.5 (5-22% vs the 2% tol) -- a genuine, diagnosed dynamic-range issue
(chi'~1/X^2 falls off faster than chi''~1/X, so a roughly flat estimator
noise floor hurts chi' proportionally more at high X), COMPOUNDED by an
actual estimator flaw: fl_chi's tau-integral ran out to 60*TAU, but the
true correlation has decayed to <1e-25 by then, so ~45*TAU of that window
was integrating pure post-decay estimator noise into chi', worst at high
test frequencies. Diagnosed and the fix sized via a cheap synthetic-OU
proxy (the k=0 mode IS exactly a scalar OU process -- swapping the 32x32
field solve for the equivalent scalar recursion, same TAU/DT/sigma2,
reusing dynsus.py's real functions unmodified, made iteration seconds
instead of hours) BEFORE spending more real compute: truncating the
integration window helps in expectation but showed high realization-to-
realization variance at N_SEEDS=14 in the proxy (a truncation-length scan
was non-monotonic), so MAX_LAG=20*TAU/DT below is a principled, moderate
choice (still >>1 decay times of margin, cuts most of the unnecessary
noise-tail) rather than a cherry-picked "optimal" value; the RMS-vs-
n_steps power-law fit from the same proxy was noisy (fitted exponent
-0.88 against a theoretical -0.5 for pure sqrt-N averaging) and not
trusted to better than roughly a factor of 2, so N_STEPS_PER_SEED below is
a bounded, moderate 1.5x increase (not the fit's own point estimate)
pending the next gate result.

ERRATUM (SPEC V2-03.C.1 / V2-04.0): the recorded runs used the pre-V2-04.0
Langevin normalisation, whose noise was low by N_cells/V_cell = 2.56e4; the
nominal T = 298 K was T_equivalent = T*V_cell/N_cells = 0.01164 K on this
32^2 / 0.2 nm solver grid. The code now uses the convention of record
(units.kT_langevin), so re-running does NOT reproduce the recorded numbers.
Status: INTERPRETATION -- this is linear response: chi'' =
omega*S(omega)/(2kT) with S proportional to kT, so the noise amplitude
cancels; tau = 1/(L*lambda(k)) is noise-free and the AC route is
independent. The Debye lineshape, the KK check and the two-route agreement
all stand; only the "298 K" label is void (see SPEC V2-03.C.1(a) and the
impact table of reports/REVIEW_v2_03c_kt.md).
"""
import os

import numpy as np

from solvers import solver
from solvers import units
from solvers import dynsus
from deck_parser import load_params_toml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- case parameters (G16a precedent: quadratic-only alpha, real bto_298K
# grad/depol, though A(0)=D(0)=0 makes grad/depol irrelevant to the k=0
# mode specifically -- kept for physical-realism/precedent continuity) ---
A1 = 0.06
N = 32
H = 0.2
DT = 0.015
T_KELVIN = 298.0
L_MOB = 1.0

# --- ensemble sizing (see module docstring, "RECOVERY-SESSION REVISION") ---
N_SEEDS = 14
N_STEPS_PER_SEED = 9_000_000
NPERSEG = 300_000
POOL = 15                 # Welch chi'' frequency-neighborhood pool width (bins)
BLOCK_LEN = 2800          # stationarity-diagnostic block length (~5 tau/dt; see below)
N_BURN_BLOCKS = 40        # fixed burn-in length in units of BLOCK_LEN (~200 tau)
SEED0_FDT = 5000

m = load_params_toml(os.path.join(_ROOT, "params", "bto_298K.toml"))["materials"]
G11 = m["G110"] * m["G11_G110"]
G12 = m["G110"] * m["G12_G110"]
G44 = m["G110"] * m["G44_G110"]
G44p = m["G110"] * m["G44P_G110"]
KAPPA = m["permittivity"]
ALPHA = dict(a1=A1, a11=0.0, a12=0.0, a111=0.0, a112=0.0, a123=0.0)
GRID = solver.make_grid(Nx=N, Ny=N, Lx=N * H, Ly=N * H)
OPS = solver.build_operators(GRID, G11, G12, G44, G44p, KAPPA)
# fluctuation-slab thickness Dz = hx (SPEC V2-04.0)
KT = units.kT_langevin(T_KELVIN, GRID["hx"], GRID["hy"], GRID["hx"])
STIFFNESS = 2.0 * A1
TAU = 1.0 / (L_MOB * STIFFNESS)
CHI0 = 1.0 / STIFFNESS
NXNY = GRID["Nx"] * GRID["Ny"]
OU = solver.build_ou_operators(OPS, DT, L_MOB, KT)
MAX_LAG = int(20 * TAU / DT)  # see module docstring "RECOVERY-SESSION REVISION"


def _run_one_fdt_seed(seed):
    """One equilibrium (E_ext=0) trajectory. Burn-in is FIXED at
    N_BURN_BLOCKS*BLOCK_LEN steps (~200 tau, generous relative to G16a/d's
    own 20-36 tau burn-ins) -- the SAME "fixed, analytically-sized window"
    choice WP12's G16a/d made, for the SAME reason (WP12_REPORT.md:
    block_stationary's own range/mean-agreement test needs the tracked
    observable's block-to-block noise to be modest enough for 4 consecutive
    blocks to plausibly agree; a SINGLE seed's SINGLE k=0-mode-squared
    block-mean does not have that property -- confirmed empirically during
    WP13 pilot work, some seeds never satisfied the default tol=0.1 within
    a 400-block cap; see reports/WP13_REPORT.md "Design notes"). During
    this fixed burn-in, block-means of the observable's square are
    collected and returned so run_fdt_ensemble can run block_stationary
    ONCE on the ENSEMBLE-AVERAGED sequence (14 independent seeds' block-
    means averaged together at each block index -- sqrt(14) lower noise,
    the SAME ensemble-averaging trick the main estimate itself relies on)
    -- a statistically meaningful use of the SAME detector, CONFIRMING
    (not gating pass/fail on) that the fixed window is adequate. This is
    SPEC §19's "stationarity detector (WP12) must pass before any window
    enters the estimate" applied at the level where the detector is
    actually well-posed, rather than mechanically per lone seed.

    Then N_STEPS_PER_SEED sampling steps of Pk_raw=P[0].sum().
    Returns (burn_block_means, omega_w, chi_pp_w, C)."""
    rng = np.random.default_rng(seed)
    Ny, Nx = GRID["Ny"], GRID["Nx"]
    P = np.zeros((3, Ny, Nx))

    def step_fn():
        nonlocal P
        P = solver.step_langevin(P, ALPHA, OPS, OU, DT, L_MOB, (0., 0., 0.), rng)
        return P[0].sum()

    burn_block_means = []
    for _ in range(N_BURN_BLOCKS):
        vals = np.empty(BLOCK_LEN)
        for i in range(BLOCK_LEN):
            vals[i] = step_fn()
        burn_block_means.append(float(np.mean(vals ** 2)))

    traj = np.empty(N_STEPS_PER_SEED)
    for i in range(N_STEPS_PER_SEED):
        traj[i] = step_fn()
    omega_w, chi_pp_w = dynsus.welch_chi_pp(traj, DT, KT, nperseg=NPERSEG, window="hann")
    C = dynsus.sample_correlation(traj, MAX_LAG)
    return burn_block_means, omega_w, chi_pp_w, C


def run_fdt_ensemble(seed0=SEED0_FDT, n_seeds=N_SEEDS):
    """Run the N_SEEDS-member parallel FDT-route ensemble (multiprocessing,
    one process/core per seed). Returns dict with omega_w, chi_pp_mean,
    chi_pp_pooled (dynsus.pool_spectrum, width=POOL), tau_arr, C_mean, C_all
    (per-seed correlation arrays, shape (n_seeds, MAX_LAG) -- exposed so a
    caller can compute a per-seed chi'(omega) via dynsus.fl_chi and get the
    estimator's actual measured noise floor from the across-seed spread,
    coordinator ruling on WP13 §19 G17a's resolvable-domain restriction, see
    gates/g17a_two_route_agreement.py), plus the stationarity DIAGNOSTIC
    (stationarity_confirmed, stationarity_first_block, ensemble_burn_means)
    -- see _run_one_fdt_seed's docstring."""
    import multiprocessing as mp
    with mp.Pool(n_seeds) as pool:
        results = pool.map(_run_one_fdt_seed, list(range(seed0, seed0 + n_seeds)))

    burn_means_all = np.array([r[0] for r in results])   # (n_seeds, N_BURN_BLOCKS)
    ensemble_burn_means = burn_means_all.mean(axis=0)
    stationarity_confirmed, stationarity_first_block = False, None
    for b in range(4, N_BURN_BLOCKS + 1):
        if solver.block_stationary(list(ensemble_burn_means[:b])):
            stationarity_confirmed, stationarity_first_block = True, b
            break

    omega_w = results[0][1]
    chi_pp_all = np.array([r[2] for r in results])
    chi_pp_mean = chi_pp_all.mean(axis=0)
    chi_pp_pooled = dynsus.pool_spectrum(omega_w, chi_pp_mean, POOL)
    C_all = np.array([r[3] for r in results])
    C_mean = C_all.mean(axis=0)
    tau_arr = np.arange(C_mean.shape[0]) * DT
    return dict(omega_w=omega_w, chi_pp_mean=chi_pp_mean, chi_pp_pooled=chi_pp_pooled,
                tau_arr=tau_arr, C_mean=C_mean, C_all=C_all,
                stationarity_confirmed=stationarity_confirmed,
                stationarity_first_block=stationarity_first_block,
                ensemble_burn_means=ensemble_burn_means)


def run_ac_seed(args):
    """One AC-driven trajectory: after its own FIXED E_ext=0 equilibration
    (N_BURN_BLOCKS*BLOCK_LEN steps -- same fixed window as the FDT route,
    see _run_one_fdt_seed's docstring for why a per-trajectory gated
    detector is not used here), drive E_ext=(E0*sin(omega*t),0,0) from a
    LOCAL t=0, lock-in demodulate. args=(omega, E0, n_periods, seed).
    Effective driving amplitude on Pk_raw is Nx*Ny*E0 (module docstring)
    -- divides by that, not the bare field amplitude. Returns
    (chi_p, chi_pp)."""
    omega, E0, n_periods, seed = args
    rng = np.random.default_rng(seed)
    Ny, Nx = GRID["Ny"], GRID["Nx"]
    P = np.zeros((3, Ny, Nx))

    def step_fn():
        nonlocal P
        P = solver.step_langevin(P, ALPHA, OPS, OU, DT, L_MOB, (0., 0., 0.), rng)
        return P[0].sum()

    for _ in range(N_BURN_BLOCKS * BLOCK_LEN):
        step_fn()
    period = 2 * np.pi / omega
    n_steps = int(n_periods * period / DT)
    traj = np.empty(n_steps)
    for i in range(n_steps):
        t = i * DT
        Eext = (E0 * np.sin(omega * t), 0., 0.)
        P = solver.step_langevin(P, ALPHA, OPS, OU, DT, L_MOB, Eext, rng)
        traj[i] = P[0].sum()
    E0_eff = NXNY * E0
    return dynsus.lockin_chi(traj, DT, omega, E0_eff)
