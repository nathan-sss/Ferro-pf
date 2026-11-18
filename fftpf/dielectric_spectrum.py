"""fftpf.dielectric_spectrum (until 2026-11-13 pf2d_dynamic_susceptibility; that name and the
solvers.susceptibility / solvers.dynsus names are forwarding aliases of this module).

WP13 (SPEC.md §19, R2): dynamic susceptibility chi(omega,T) INSTRUMENT.

Builds on WP12's Langevin/FDT machinery (solver.build_ou_operators,
solver.step_langevin, solver.block_stationary). This module holds the
REUSABLE numerical primitives of the instrument -- spectral estimation and
linear-response bookkeeping -- deliberately separate from solver.py (a
signal-processing/estimator concern, not PDE physics) and from gates/ (which
own the actual "run the engine, build a trajectory" loops, same split WP12
established: solver.py/here = machinery, gates/ = case + acceptance test).

Two independent routes to chi(omega) = chi'(omega) + i*chi''(omega):

1. FDT route (equilibrium, E_ext=0): long stationary Langevin trajectory of
   an observable P(t) (SPEC: "the homogeneous (k=0) polarization... and
   optionally selected k-modes"). Classical fluctuation-dissipation theorem:
     chi''(omega) = omega * S(omega) / (2 kT)          [SPEC's own formula]
   S(omega) = TWO-SIDED power spectral density of P(t) (Wiener-Khinchin:
   FT of the equilibrium autocorrelation C(tau)=<P(0)P(tau)>), estimated via
   Welch's method (`welch_chi_pp`) -- "Welch/multitaper-style block
   averaging with documented window choice" (SPEC).
   chi'(omega): SPEC offers two options, "Kramers-Kronig or the direct
   Fourier-Laplace of C(t) -- implementer picks, documents, and checks
   KK-consistency." This module picks the DIRECT Fourier-Laplace route
   (`fl_chi`) as primary (numerically better-behaved -- see its docstring),
   and provides `kk_chi_prime` as the independent cross-check machinery
   gate G17c uses.

   Derivation of `fl_chi` (documented here since it is not copied verbatim
   from a textbook): classical FDT in the time domain (Onsager regression),
   phi(t) = -(1/kT) * theta(t) * dC/dt, is the causal response function;
   Fourier-Laplace transforming and integrating by parts (boundary term at
   t->infinity vanishes since C decays) gives
     chi(omega) = [C(0) + i*omega * Integral_0^inf C(t) e^{i*omega*t} dt] / kT
   Splitting the integral into cos/sin parts recovers SPEC's chi'' formula
   exactly (chi''=omega*S(omega)/(2kT), S(omega)=2*Integral_0^inf C(t)
   cos(omega t)dt by evenness of C) -- an internal consistency check on the
   sign/factor conventions used throughout this module, done once at
   WP13 pilot time (see WP13_REPORT.md), not re-derived per call.

2. AC route (small-signal driven, cross-check): E_ext(t) = E0*sin(omega*t)
   applied via the engine's existing explicit-force channel (solver's
   `_explicit_N`/`step_langevin` E_ext argument -- no engine change needed).
   Steady-state response extracted by lock-in demodulation (`lockin_chi`).
   Sign/normalization derived directly from THIS project's own equation of
   motion (not assumed from a textbook convention) -- see `lockin_chi`
   docstring.

HONEST LIMIT (SPEC §19, restated verbatim-in-substance): direct simulation
resolves a FINITE frequency window (roughly 4-6 decades, set by dt at the
high end and total trajectory length at the low end -- see WP13_REPORT.md
for this run's actual numbers). Downstream deliverables (R4) are
Vogel-Fulcher activation parameters and trends, never a claim of covering
the full experimental frequency window (10+ decades, Hz to GHz).

GATES: g17a-c (two-route agreement / Debye limit / KK consistency) and
g19a_curie_weiss_pilot. All four import this module under its pre-WP24 shim
name, `from solvers import dynsus`. The estimators are 2D-agnostic: they act on a time series.
"""
import os

import numpy as np
import scipy.signal as sig


def welch_chi_pp(traj, dt, kT, nperseg, window="hann", noverlap=None):
    """chi''(omega) = omega*S(omega)/(2kT) (SPEC §19), S(omega) the TWO-SIDED
    PSD (Wiener-Khinchin FT of the equilibrium autocorrelation), estimated by
    Welch's method: segment `traj` into (possibly overlapping) blocks of
    `nperseg` samples, apply `window` to each, average the periodograms.

    Convention check (done explicitly, not assumed -- SPEC §5 warns FFT
    normalization is a classic unexplained-factor trap): scipy.signal.welch
    with scaling='density' returns a ONE-SIDED PSD Pxx(f), f in [0, fs/2],
    satisfying Pxx(f) = 2*S_two_sided(f) for f>0 (the standard one-/two-sided
    relation). S(omega) evaluated AT omega=2*pi*f is numerically the SAME
    function as S_two_sided(f) (pure substitution of the frequency variable,
    no extra 2*pi factor -- verified against the analytic Debye/OU case
    before use, WP13_REPORT.md pilot section). Hence
      chi''(omega) = omega * (Pxx(f)/2) / (2*kT) = omega * Pxx(f) / (4*kT).

    traj: 1D real array, an ALREADY-STATIONARY segment (caller burns in and
    passes solver.block_stationary first -- SPEC §19: "stationarity detector
    (WP12) must pass before any window enters the estimate"). detrend=
    'constant' removes any small residual sample mean (expected ~0 by
    symmetry, not assumed exactly 0).

    Returns (omega, chi_pp), omega[0] excluded (omega=0 is not meaningful
    for this ratio; chi''(0)=0 identically for any physical response)."""
    f, Pxx = sig.welch(traj, fs=1.0 / dt, window=window, nperseg=nperseg,
                        noverlap=noverlap, detrend="constant", scaling="density")
    omega = 2.0 * np.pi * f
    chi_pp = omega * Pxx / (4.0 * kT)
    return omega[1:], chi_pp[1:]


def sample_correlation(traj, max_lag):
    """C[lag] = mean_t[(traj[t]-mean)*(traj[t+lag]-mean)], lag=0..max_lag-1
    -- the direct lagged-product estimator (same estimator gate G16d used
    for its decay-rate fit; here the whole array is returned, not just a
    fitted slope, for use as `fl_chi`'s input)."""
    x = np.asarray(traj, dtype=float) - np.mean(traj)
    n = len(x)
    C = np.empty(max_lag)
    for lag in range(max_lag):
        C[lag] = np.mean(x[:n - lag] * x[lag:]) if lag > 0 else np.mean(x * x)
    return C


def fl_chi(omega_array, tau_array, C_array, kT):
    """Direct Fourier-Laplace route (see module docstring for derivation):
      chi(omega) = [C(0) + i*omega * Integral_0^inf C(t) e^{i*omega*t} dt] / kT
    tau_array/C_array: sample_correlation's output and its matching lag-time
    axis (tau_array = lag*dt), integrated (trapezoid) up to whatever max lag
    the caller supplies -- should reach several decay times so the true
    signal's tail contribution is negligible (documented per-call in the
    gate that uses this).

    TAPER (WP13 recovery-session finding, see reports/WP13_REPORT.md
    "Design notes"): the integral above is truncated at a FINITE tau_max
    (the caller's array length), approximating the true tau_max=infinity
    integral. A HARD cutoff of an oscillatory integrand (C(t)*e^{i*omega*t})
    is a textbook source of spectral-leakage/Gibbs ringing -- a SYSTEMATIC,
    omega-dependent bias that does NOT shrink with more ensemble averaging
    (unlike random noise), found empirically: an early recovery-session gate
    run showed chi' at high omega*tau confidently non-zero (>5 standard
    errors of an ensemble mean) yet still ~8% off the known true value.
    welch_chi_pp (this module's OTHER FDT-route estimator, for chi'') already
    applies window="hann" for exactly this reason (standard STFT/periodogram
    practice) -- fl_chi had no analogous taper, an unintentional asymmetry
    between the two routes. Fixed here with a half-cosine (Hann-shape) taper
    on C(tau): 1.0 at tau=0 (no information thrown away at the most
    important lag, where the true signal is largest) smoothly -> 0.0 at
    tau_max (kills the ringing). Cross-checked against a synthetic
    Ornstein-Uhlenbeck proxy with a KNOWN analytic answer before use here:
    cut a -83%/+49%/-110% raw error at the 3 highest tested omega*tau down
    to -22%/+9%/+8%, and incidentally reduced the per-seed standard error by
    2-3x (a shorter *effective* noise-integrating tail) -- both a bias fix
    and a free variance reduction, not a tradeoff.

    Returns complex array chi = chi' + i*chi''."""
    tau_array = np.asarray(tau_array, dtype=float)
    C_array = np.asarray(C_array, dtype=float)
    taper = 0.5 * (1.0 + np.cos(np.pi * tau_array / tau_array[-1]))
    C_tapered = C_array * taper
    C0 = C_array[0]   # untapered C(0): taper[0]=1.0 exactly, so this is the same value either way
    out = np.empty(len(omega_array), dtype=complex)
    for i, w in enumerate(omega_array):
        integrand = C_tapered * np.exp(1j * w * tau_array)
        Lval = np.trapezoid(integrand, tau_array)
        out[i] = (C0 + 1j * w * Lval) / kT
    return out


def kk_chi_prime(omega_grid, chi_pp_grid, omega_eval):
    """Numerical Kramers-Kronig transform, principal-value quadrature with
    singularity subtraction:
      chi'(w) = (2/pi) * P-Integral_0^inf  w'*chi''(w') / (w'^2 - w^2)  dw'
    Writing g(w') = w'*chi''(w')/(w'+w) (pole only at w'=w, since w'+w>0 for
    w',w>0):
      P-Integral_0^inf g(w')/(w'-w) dw' =
        Integral [g(w')-g(w)]/(w'-w) dw'  +  g(w)*ln|(wmax-w)/(wmin-w)|
    evaluated over the FINITE resolved band [omega_grid[0], omega_grid[-1]]
    (algebraically exact for that finite range; the only approximation is
    truncating the true 0..infinity integral to this band).

    TRUNCATION-ERROR CAVEAT (found and quantified during WP13 pilot testing,
    see WP13_REPORT.md): because chi''(w') decays only as ~1/w' at large w'
    (generic relaxational response), the neglected tail contributes an
    ABSOLUTE error that shrinks only as ~1/wmax, not exponentially -- so the
    RELATIVE error at a test frequency omega_eval grows quickly once
    omega_eval approaches/exceeds a modest multiple of wmax's own scale.
    This function does NOT self-check that omega_eval is "far enough" inside
    [wmin,wmax] -- callers (gate G17c) must restrict to a band verified safe
    for their own omega_grid/wmax (documented per-call, not silently
    assumed)."""
    omega_grid = np.asarray(omega_grid, dtype=float)
    chi_pp_grid = np.asarray(chi_pp_grid, dtype=float)
    wmin, wmax = omega_grid[0], omega_grid[-1]
    out = np.empty(len(omega_eval))
    for i, w in enumerate(omega_eval):
        gw = 0.5 * np.interp(w, omega_grid, chi_pp_grid)   # g(w) = w*chi''(w)/(2w)
        g = omega_grid * chi_pp_grid / (omega_grid + w)
        with np.errstate(divide="ignore", invalid="ignore"):
            reg = np.where(np.abs(omega_grid - w) > 1e-12, (g - gw) / (omega_grid - w), 0.0)
        idx = int(np.argmin(np.abs(omega_grid - w)))
        if np.abs(omega_grid[idx] - w) < 1e-9:
            lo, hi = max(idx - 1, 0), min(idx + 1, len(omega_grid) - 1)
            reg[idx] = 0.5 * (reg[lo] + reg[hi])            # fill the removable singularity point
        integral = np.trapezoid(reg, omega_grid) + gw * np.log(np.abs((wmax - w) / (wmin - w)))
        out[i] = (2.0 / np.pi) * integral
    return out


def lockin_chi(traj, dt, omega, E0):
    """AC cross-check route: lock-in extraction of (chi', chi'') from a
    driven trajectory `traj` (caller has ALREADY discarded the startup
    transient -- traj[0] corresponds to local time t=0 for this function's
    own sin/cos clock, which must match the phase of the E_ext(t)=
    E0*sin(omega*t) actually fed to solver.step_langevin from that same
    t=0 origin).

    Sign/normalization derivation (this project's own equation of motion,
    not assumed): the k=0 mode of solver.step_langevin's deterministic part
    obeys dP/dt = -L*(stiffness*P - E_ext(t)) + noise. For E_ext=E0*sin(wt)
    the steady-state particular solution is P(t)=A*sin(wt)+B*cos(wt) with
    (matching coefficients, Gamma=L*stiffness):
      A = L*E0*Gamma/(Gamma^2+w^2) = E0*chi'(w),
      B = -L*E0*w/(Gamma^2+w^2)    = -E0*chi''(w)
    (chi'=L*Gamma/(Gamma^2+w^2), chi''=L*w/(Gamma^2+w^2) reproduce the
    Debye form chi0/(1+i*w*tau), chi0=1/stiffness, tau=1/Gamma -- checked
    algebraically in WP13_REPORT.md). Orthogonality over an integer number
    of periods gives A,B as plain projections:
      A = (2/T) Integral P(t) sin(wt) dt,  B = (2/T) Integral P(t) cos(wt) dt
    so chi'=A/E0, chi''=-B/E0.

    Returns (chi_p, chi_pp)."""
    x = np.asarray(traj, dtype=float)
    n = len(x)
    t = np.arange(n) * dt
    T = t[-1] if n > 1 else dt
    A = (2.0 / T) * np.trapezoid(x * np.sin(omega * t), t)
    B = (2.0 / T) * np.trapezoid(x * np.cos(omega * t), t)
    return A / E0, -B / E0


def pool_spectrum(omega, values, pool):
    """Smooth `values` (e.g. Welch chi'') along the frequency axis with a
    `pool`-wide moving average (odd `pool` recommended so the window is
    centered). SPEC calls for "Welch/multitaper-style block averaging" --
    segment-averaging (inside welch_chi_pp, via nperseg/noverlap) already
    does part of this; WP13 pilot testing (see WP13_REPORT.md) found that a
    single long trajectory's segment-averaging alone was NOT enough to
    reach the SPEC gates' 2% tolerances, and that an ADDITIONAL, modest,
    fixed-bin-count frequency-neighborhood average (this function) -- on
    top of ensemble-averaging across independent-seed trajectories --
    closes the remaining gap without materially biasing a slowly-varying
    response function. `pool` is a plain bin count (not a relative
    bandwidth): callers document the specific value used and why (it
    trades bias, worse for a wider pool near a sharply-curving feature
    like the Debye peak, against variance, better for a wider pool) for
    their own omega grid spacing."""
    if pool <= 1:
        return values.copy()
    kernel = np.ones(pool) / pool
    return np.convolve(values, kernel, mode="same")


def debye_chi(omega, chi0, tau):
    """Analytic single-relaxation-time (Debye) susceptibility, the exact
    reference form for gate G17b -- returned as a complex array chi'+i*chi''.

    Sign convention (checked explicitly, not textbook-assumed -- caught by a
    pre-gate unit test during WP13 pilot work, see WP13_REPORT.md "Design
    notes"): this module's chi''>0 for omega>0 (SPEC's own chi''=omega*S
    (omega)/(2kT) is manifestly positive; `fl_chi`/`lockin_chi` are BOTH
    derived directly from this engine's equation of motion and agree with
    that sign). The textbook Debye form is often written chi0/(1+i*omega*
    tau), but THAT expression's imaginary part is NEGATIVE for omega>0 (the
    "physicist" chi'-i*chi'' convention, e.g. epsilon=eps'-i*eps'') --
    the opposite of this module's convention. The algebraically equivalent
    chi0/(1-i*omega*tau) has the SAME real part (chi'=chi0/(1+(omega*tau)^2))
    and the correctly-signed positive imaginary part
    (chi''=chi0*omega*tau/(1+(omega*tau)^2)), so that is what this function
    returns."""
    return chi0 / (1.0 - 1j * omega * tau)


def measure_chi_p_floor(C_all, tau_arr, omega_grid, kT, k_sigma=3.0):
    """WP13 recovery-session addition (coordinator ruling on G17a, extended
    to G17b/G17c since SPEC §19 phrases THEIR criteria "over/where...
    resolved" -- an explicit resolved-window concept, not a fixed decade
    count like G17a's). Measures the FDT route's chi'(omega) estimator
    noise floor directly from ENSEMBLE SPREAD: chi' computed independently
    per seed (fl_chi applied to each seed's OWN correlation array, C_all,
    shape (n_seeds, max_lag) -- from wp13_debye_case.run_fdt_ensemble's
    C_all, not the ensemble mean C_mean), giving N_SEEDS independent draws
    of chi'(omega) at every point of omega_grid; the standard error of
    their mean is a real, measured (not assumed/guessed) per-frequency
    noise floor -- no new simulation, reuses the ensemble already paid
    for. A point is "resolved" if |chi'_mean| exceeds k_sigma standard
    errors (default 3, a standard, non-cherry-picked Gaussian detection
    threshold, ~99.7% confidence per point).

    Returns (chi_p_mean, chi_p_sem, resolvable) at every point of
    omega_grid, each shape (len(omega_grid),)."""
    chi_p_seeds = np.array([fl_chi(omega_grid, tau_arr, C_seed, kT).real
                             for C_seed in C_all])
    n_seeds = C_all.shape[0]
    chi_p_mean = chi_p_seeds.mean(axis=0)
    chi_p_sem = chi_p_seeds.std(axis=0, ddof=1) / np.sqrt(n_seeds)
    resolvable = np.abs(chi_p_mean) > k_sigma * chi_p_sem
    return chi_p_mean, chi_p_sem, resolvable


def contiguous_resolved_range(X_cand, resolvable):
    """Largest contiguous resolvable run of `resolvable` (a boolean array,
    e.g. measure_chi_p_floor's output) starting from X_cand[0]. chi'(X)
    falls off monotonically with X (Debye form, or generic relaxational
    response) while the estimator noise floor is roughly flat, so
    resolvability is expected to hold below some cutoff and fail above it
    -- this searches for that cutoff rather than assuming its location.
    Returns (i_max, X_res_lo, X_res_hi): i_max is the index of the last
    resolvable candidate in the contiguous run (X_cand[i_max]=X_res_hi);
    i_max=-1, X_res_lo=X_res_hi=None if X_cand[0] itself is not resolvable
    (empty window -- caller's STOP-AND-REPORT case)."""
    i = 0
    while i < len(resolvable) and resolvable[i]:
        i += 1
    i_max = i - 1
    if i_max < 0:
        return -1, None, None
    return i_max, X_cand[0], X_cand[i_max]


def _ou_seed_correlation(args):
    """Multiprocessing worker for calibrate_fl_chi_bias: one synthetic
    scalar Ornstein-Uhlenbeck trajectory (EXACT discretization, matching
    solver.build_ou_operators' own per-mode exact-OU convention) and its
    sample correlation. Module-level (not a closure) so it is picklable."""
    tau, dt, sigma2, max_lag, n_steps, seed = args
    decay = np.exp(-dt / tau)
    std = np.sqrt(sigma2 * (1.0 - decay ** 2))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n_steps)
    x = np.empty(n_steps)
    cur = 0.0
    for i in range(n_steps):
        cur = cur * decay + std * noise[i]
        x[i] = cur
    return sample_correlation(x, max_lag)


_DEFAULT_BIAS_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "reports", "wp13_bias_calibration_cache.npz")


def calibrate_fl_chi_bias(tau, kT, dt, max_lag, omega_array, n_seeds=14,
                           n_steps=40_000_000, n_draws=4, seed0=90000,
                           cache_path=_DEFAULT_BIAS_CACHE):
    """WP13 recovery-session addition (coordinator requirement): measures
    fl_chi's SYSTEMATIC bias -- NOT random noise, which measure_chi_p_floor
    already covers -- at each point of omega_array. Random noise (an
    ensemble's standard error) shrinks as more seeds are averaged in; a
    finite-window truncation bias (fl_chi's own docstring) does NOT, so no
    amount of averaging the REAL ensemble can reveal it from the real data
    alone -- it needs an INDEPENDENT reference with a KNOWN true answer.

    That reference is a synthetic scalar Ornstein-Uhlenbeck proxy: the
    engine's own k=0 mode IS exactly this process (established throughout
    WP13's case/gates -- A(0)=D(0)=0, so Lin(0) reduces to the bulk-only
    stiffness=1/tau), so simulating the scalar recursion directly (no 32x32
    field solve -- orders of magnitude cheaper) at the SAME tau, dt, kT
    (hence sigma2=kT*tau via stiffness=1/tau, L=1) and the SAME max_lag the
    real gate uses reproduces the real estimator's truncation/taper
    behavior exactly, against an analytically-known chi0/(1-i*omega*tau)
    truth.

    CONVERGENCE (recovery-session finding, see reports/WP13_REPORT.md
    "Design notes"): an EARLIER version of this function ran a SINGLE draw
    at n_steps=6e6 -- found, by direct comparison, to be itself noise-
    dominated: two independent 6e6-step draws at the same candidate
    frequencies disagreed by >5x at the worst point (e.g. 22.7% vs 4.3%),
    proof the single-draw bias number was not a converged measurement.
    Fixed by running n_draws INDEPENDENT calibration ensembles (each
    n_seeds x n_steps, disjoint seed ranges: draw d uses seeds
    seed0+d*n_seeds .. +n_seeds-1) and reporting BOTH the across-draw mean
    AND the across-draw spread -- the spread is a real, quantified
    uncertainty on the bias measurement itself, not asserted away.

    Returns (bias_mean, bias_spread), each shape (len(omega_array),):
    bias_mean = mean over draws of |chi_p_measured/chi_p_true - 1|;
    bias_spread = std over draws (ddof=1; 0 if n_draws==1) of the same
    per-draw quantity -- the CALLER compares bias_mean against a
    tolerance (e.g. half of SPEC's 2% gate tolerance) and should treat a
    point as trustworthy only where bias_spread is also comfortably
    smaller than that tolerance (documented per-call, not assumed here).

    CACHING (economy -- this WP's recovery session found the calibration
    is by far the most expensive PER-GATE-INDEPENDENT step, ~70min, yet
    its result depends ONLY on tau/kT/dt/max_lag/n_seeds/n_steps/n_draws/
    seed0/omega_array -- all identical across G17a/b/c, which share
    cases/wp13_debye_case.py's constants): if `cache_path` exists and its
    stored parameters match EXACTLY (the case module's constants are
    literal, not independently re-derived floats, so exact equality is
    the right check, not an approximate one), the cached (bias_mean,
    bias_spread) is returned with NO new simulation; otherwise this
    computes fresh and writes the cache. Pass cache_path=None to disable.

    PROCESS COUNT: on a cache miss this opens mp.Pool(n_seeds) -- 14 worker
    processes by default, the ONLY multiprocessing anywhere in solvers/. The
    caller's single-thread env (ENGINE_GUIDE §1.2) is inherited by each
    worker and so bounds its BLAS threads, but NOT the 14 processes; lower
    n_seeds if the box is shared.
    CACHE LOCATION: cache_path defaults beside THIS FILE, i.e.
    solvers/reports/wp13_bias_calibration_cache.npz -- not the repo-root
    reports/ (a second directory of the same name; do not "fix" the path,
    that silently discards the cached ~70-min result).
    n_seeds=14 / n_steps=40_000_000 / seed0=90000 are declared dials with no
    external source (only n_draws=4 is justified, above); they are part of
    the cache key, so changing any of them forces a fresh ~70-min run."""
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path)
        params_match = (float(d["tau"]) == tau and float(d["kT"]) == kT and
                         float(d["dt"]) == dt and int(d["max_lag"]) == max_lag and
                         int(d["n_seeds"]) == n_seeds and int(d["n_steps"]) == n_steps and
                         int(d["n_draws"]) == n_draws and int(d["seed0"]) == seed0 and
                         d["omega_array"].shape == np.asarray(omega_array).shape and
                         np.array_equal(d["omega_array"], omega_array))
        if params_match:
            return d["bias_mean"], d["bias_spread"]

    stiffness = 1.0 / tau
    chi0 = 1.0 / stiffness
    sigma2 = kT / stiffness
    import multiprocessing as mp
    tau_arr = np.arange(max_lag) * dt
    chi_p_true = (chi0 / (1.0 - 1j * omega_array * tau)).real
    draw_biases = []
    for d in range(n_draws):
        jobs = [(tau, dt, sigma2, max_lag, n_steps, seed0 + d * n_seeds + i)
                for i in range(n_seeds)]
        with mp.Pool(n_seeds) as pool:
            C_list = pool.map(_ou_seed_correlation, jobs)
        C_mean = np.mean(C_list, axis=0)
        chi_p_meas = fl_chi(omega_array, tau_arr, C_mean, kT).real
        draw_biases.append(np.abs(chi_p_meas / chi_p_true - 1.0))
    draw_biases = np.array(draw_biases)   # (n_draws, len(omega_array))
    bias_mean = draw_biases.mean(axis=0)
    bias_spread = (draw_biases.std(axis=0, ddof=1) if n_draws > 1
                   else np.zeros_like(bias_mean))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, tau=tau, kT=kT, dt=dt, max_lag=max_lag,
                 n_seeds=n_seeds, n_steps=n_steps, n_draws=n_draws, seed0=seed0,
                 omega_array=np.asarray(omega_array), bias_mean=bias_mean,
                 bias_spread=bias_spread)
    return bias_mean, bias_spread
