"""SPEC S41.C / S41.1 A3(i) (WP29) -- SUGAR over the raw engine API for the
six things nearly every script in cases/ re-derives by hand: the FFT thread
knobs, grid+ops, the 3D elastic Q trap, the anneal loop with its stopping
rule, the output folder, and the save set.

NOT A LAYER, NOT THE ONLY WAY. Every function below is a 2-8 line
composition of calls a case can equally well type itself, and the raw API
stays the reference: ENGINE_GUIDE.md S3 keeps templates (1)/(3)/(5) written
against it verbatim, and S3.6 keeps the raw-API version of the two
templates (2)/(4) that were rewritten with this module. NO GATE DEPENDS ON
THIS MODULE -- deleting it breaks no acceptance instrument, it only makes
cases longer.

TWO CONCERNS IT DELIBERATELY DOES NOT ABSORB, because it cannot:
  (1) the four *_NUM_THREADS env vars must take effect BEFORE numpy is
      imported (ENGINE_GUIDE S1.2), and `from solvers import casekit`
      imports numpy -- by the time any function here could run, the BLAS
      pools are already built. Those lines stay at the top of the case.
  (2) sys.path must already contain the repo root before `from solvers
      import ...` resolves; a module inside solvers/ cannot put it there
      on behalf of its own importer.
  Both DO disappear for a job-launched case: tools/job.py builds the
  child's environment (the four vars + PYTHONPATH=<repo root>) before
  exec, so a case only ever started by `job.py submit` needs neither.

LAYERING (SPEC S36.2 R5): this module imports the engines; nothing under
solvers/ imports it and nothing has to. It sits at the top of the library,
one step below cases/. It is deliberately absent from solvers/__init__.py's
__all__ -- `from solvers import casekit` resolves through Python's ordinary
submodule mechanism, the same way `from solvers import afd` does (see that
file's own docstring).
"""
import json
import os
from fftpf._locations import default_run_root
import subprocess
import sys
import time
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from fftpf import ENGINE_NAME, ENGINE_VERSION
from fftpf import pf3d_engine as bulk3d
from fftpf import state_io as io_utils
from fftpf import pf2d_engine as solver
from fftpf import input_validation as validate

# Run records and restart snapshots name the engine by its historical key; keep
# writing those keys so existing checkpoints still load (resume_anneal reads them).
_ENGINE_KEY = {"pf2d_engine": "solver", "pf3d_engine": "bulk3d", "film_engine": "film3d"}

# The four env vars ENGINE_GUIDE S1.2 requires; read, never written (writing
# them here would be too late to matter -- see (1) in the module docstring).
# The fourth is Apple Accelerate's VECLIB_MAXIMUM_THREADS -- NOT *_NUM_THREADS,
# which Accelerate ignores (WP29 C6; the name the profiling harness under
# gates/wp26_profiling and gates/wp27_profiling has always exported).
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def single_thread(*modules):
    """Set FFT_WORKERS = 1 on each module passed and return the one-line
    thread-settings string for the log header.

    Absorbs: the per-module `X.FFT_WORKERS = 1` lines of ENGINE_GUIDE S1.2
    (six independent knobs -- solver/elastic/film3d/kittel_afe/octa_tilt/
    bulk3d -- setting one does NOT set the others, which is why this takes
    *modules), plus the thread-settings line S1.2 says every production
    script should print and which exactly one file in the repo
    (cases/w0_relaxor_demo.py:162) actually does.

    The four env values are REPORTED as they stand, not set: they had to be
    in place before numpy was imported. "unset" in the returned string
    means that variable was never exported -- a real finding, not a
    formatting artifact. `nice` is included because S1.2 counts it part of
    "single-thread env"; os.getpriority reads the current value
    without changing it (SPEC V2-02.1 E16: os.nice(0) is a
    side-effecting call that returns the same number)."""
    for module in modules:
        module.FFT_WORKERS = 1
    env = " ".join(f"{v.split('_')[0]}={os.environ.get(v, 'unset')}" for v in _THREAD_ENV)
    names = ",".join(_ENGINE_KEY.get(m.__name__.rsplit(".", 1)[-1], m.__name__.rsplit(".", 1)[-1]) for m in modules) or "none"
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    return f"threads: {env} | FFT_WORKERS=1 in [{names}] | nice {priority}"


def build_2d(deck, Nx, Ny, h):
    """2D periodic grid + operators from a materials.get() deck, cell size h
    (deck length units). Returns (grid, ops).

    Absorbs: ENGINE_GUIDE S3 template (2)'s make_grid + build_operators
    pair, including the `**deck["gradient"]` expansion (those values are
    already absolute, i.e. multiplied through by G110 -- hand-multiplying
    again is FAQ #1's wiring bug) and `kappa=deck["kappa_b"]`."""
    validate.build_2d(deck, Nx, Ny, h)
    grid = solver.make_grid(Nx=Nx, Ny=Ny, Lx=Nx * h, Ly=Ny * h)
    ops = solver.build_operators(grid, **deck["gradient"], kappa=deck["kappa_b"])
    return grid, ops


def build_3d(deck, N, d, *, elastic=False, k0="stress_free"):
    """Cubic N^3 fully-periodic grid + operators (+ elastic_params when
    elastic=True) from a materials.get() deck, cell size d. Returns
    (grid, ops, el); el is None when elastic=False.

    Absorbs: ENGINE_GUIDE S3 template (4)'s make_grid_3d + build_operators_3d
    + setup_elastic_3d block, with the SPEC S37.5 NAMED TRAP done here once:
    a deck's electrostriction sector is VOIGT (Q11_phys/Q12_phys/Q44_phys)
    while setup_elastic_3d wants the TENSOR triple, and the shear component
    converts as Q1212 = Q44_phys / 4 (see bulk3d.eigenstrain_3d's docstring
    for why -- Q1212 populates all four equivalent shear slots). Passing
    Q44_phys straight through is a factor-16 error in the shear eigenstrain
    energy that nothing downstream will flag for you.

    k0: "stress_free" (SPEC S37.3 default) or "clamped" -- forwarded
    unchanged to bulk3d.setup_elastic_3d."""
    validate.build_3d(deck, N, d, elastic)
    grid = bulk3d.make_grid_3d(Nx=N, Ny=N, Nz=N, Lx=N * d, Ly=N * d, Lz=N * d)
    ops = bulk3d.build_operators_3d(grid, **deck["gradient"], kappa=deck["kappa_b"])
    if not elastic:
        return grid, ops, None
    C, Qp = deck["elastic"], deck["electrostriction"]
    el = bulk3d.setup_elastic_3d(
        grid, C=(C["C11"], C["C12"], C["C44"]),
        Q=(Qp["Q11_phys"], Qp["Q12_phys"], Qp["Q44_phys"] / 4.0), k0=k0)
    return grid, ops, el


def anneal(engine, P, alpha, ops, grid, *, dt, L=1.0, E_ext=(0., 0., 0.),
           kT=None, noise_cutoff_nm=None, elastic_params=None, seed=0, n_max, sample_every,
           tol=0.01, n_blocks=4, criterion="legacy", block_size=8,
           energy_scale=None, polarization_scale=None, residual_tol=1e-6,
           scheme="sbdf1", restart=None, return_state=False):
    """Run `engine` (solver or bulk3d) from P until its stopping rule fires,
    n_max steps are spent, or the state goes non-finite. Returns
    (P, trace, stop_reason) with stop_reason in {"stationary", "n_max",
    "nonfinite"} and trace = the sampled TOTAL energy (not a density), one
    entry every `sample_every` steps; with return_state=True, a fourth
    element, the restart certificate (below).

    kT=None runs the deterministic semi-implicit stepper (build_Minv/step,
    or the two-step SBDF2 recurrence when scheme="sbdf2" -- DETERMINISTIC
    only, on either engine: bulk3d's build_Minv2_3d/step_sbdf2_3d landed in
    SPEC V2-03 A1 and this function picks them up by the same `_3d` suffix
    it uses for every other bulk3d primitive);
    a kT runs Langevin (build_ou_operators/step_langevin, per-mode exact OU,
    seeded by `seed`). Build that kT with units.kT_langevin(T, hx, hy, hz) --
    ONE converter for every engine since SPEC V2-04.0 (D16), an energy
    DENSITY. In 2D pass hz = Dz, the declared fluctuation-slab thickness
    (legacy surrogate Dz = grid["hx"]). units.si_to_deck_kT and
    bulk3d.kT_deck_3d are unit conversions of k_B*T into F_total units and
    are NOT the FDT coefficient; K5 `kt_convention` is the authority.

    `noise_cutoff_nm` = l_c is forwarded to the engine's OU builder together
    with the `grid` it needs (SPEC V2-04.0, D16): the noise -- never the
    decay -- is zeroed on every mode with |k| >= 2*pi/l_c, so the retained
    modes stay exactly Boltzmann at T and the excluded ones relax
    deterministically. It is meaningless without noise and is REFUSED when
    kT is None. None is the full discrete FDT and is byte-identical to the
    pre-V2-04.0 call (K2 `anneal_noise_cutoff_none_byte`). It is recorded in
    the restart certificate's `model` next to `kT`, and a restart refuses a
    changed l_c for the same reason it refuses a changed kT.

    TWO STOPPING RULES (SPEC V2-02.1 E1). `criterion="legacy"` is the
    DEFAULT and is the rule this function has had since WP29, unchanged and
    bit-for-bit: solver.block_stationary on the RAW sampled energy list --
    the last n_blocks entries mutually within `tol` relative to their own
    mean magnitude -- first tested once 2*n_blocks samples exist. Its known
    weakness is the one the 2026-09-10 assessment named (claim A, verified
    TRUE): the samples are instantaneous, not block means, and the
    mean-magnitude normalization makes the verdict depend on the arbitrary
    ZERO of the energy. `criterion="block_v2"` is the replacement: REAL
    block means (block_size samples per block, n_blocks blocks) of five
    observables -- energy density, the three components of <P>, and <P^2> --
    each tested against an EXPLICIT physical scale, so shifting the energy
    zero cannot change the answer; and for a deterministic run stationarity
    additionally requires max|L*dF/dP| <= residual_tol, which a flat energy
    trace alone does not prove. block_v2 therefore DEMANDS energy_scale and
    polarization_scale: 1.0 is not a physical scale for a deck whose
    f ~ 1e-2 and P ~ 0.26, and defaulting them to 1.0 is what turned the 2D
    demo into a 6000-step `n_max` run on the codex branch (Lens 1 (a)).

    `tol < 0` disables the stop test under either criterion -- the way a
    fixed-horizon run, or a restart check, asks for exactly n_max steps.

    Absorbs: ENGINE_GUIDE S3 template (4)'s 16-line Langevin loop and the 44
    lines of stopping rule the six audited production cases re-derive
    between them (REVIEW_wp29_inventory.md A3(i)); under noise a dF-plateau
    criterion is VOID (SPEC S18), which is why both rules are block tests.

    RESTART (V2-02.1 E8, additive). return_state=True also returns
    dict(step, t, trace, observations, P_previous, rng_state, model,
    stop_reason, residual): everything needed to continue the SAME
    recurrence -- the SBDF2 previous field, the RNG bit-generator state, the
    full sample history (so a split between two sample points loses none of
    it) and the model inputs. Hand it straight back as `restart=` (then
    `n_max` is the number of ADDITIONAL steps and sampling follows the
    ABSOLUTE step number), or save it with io_utils.save_state_npz(restart=)
    and reopen it with resume_anneal. A restart REFUSES a changed
    recurrence: dt, L, scheme, sample_every, block_size, n_blocks, kT and
    noise_cutoff_nm must be what they were. (A certificate written before
    V2-04.0 carries no noise_cutoff_nm key; it is read as None, which is the
    only value it could have had.)

    NON-FINITE: returns the offending state with reason "nonfinite" rather
    than raising -- the caller decides whether a blown-up trajectory is a dt
    bug or an expected boundary probe. "legacy" checks only at its own
    sample points (so up to sample_every-1 steps of NaN may precede the
    report, exactly as before); "block_v2" checks every step."""
    validate.anneal(engine, P, alpha, grid, E_ext)
    suffix = "_3d" if hasattr(engine, "step_3d") else ""
    if not hasattr(engine, "step_langevin" + suffix):
        raise ValueError(f"casekit.anneal: {engine.__name__} is not solver or bulk3d")
    if criterion not in ("legacy", "block_v2"):
        raise ValueError(f"casekit.anneal: unknown criterion {criterion!r}")
    if scheme not in ("sbdf1", "sbdf2") or (scheme == "sbdf2" and kT is not None):
        raise ValueError("casekit.anneal: SBDF2 is deterministic-only (no Langevin)")
    if dt <= 0 or sample_every < 1 or n_max < 0:
        raise ValueError("casekit.anneal: invalid dt, step count or sampling interval")
    if noise_cutoff_nm is not None and kT is None:
        raise ValueError("casekit.anneal: noise_cutoff_nm is a NOISE knob and needs a kT "
                         "(SPEC V2-04.0); a deterministic run has no noise to cut")
    if criterion == "block_v2":
        if energy_scale is None or polarization_scale is None:
            raise ValueError(
                "casekit.anneal: criterion='block_v2' requires an explicit energy_scale "
                "and polarization_scale -- 1.0 is not a physical scale (V2-02.1 E1)")
        if energy_scale <= 0 or polarization_scale <= 0 or residual_tol < 0:
            raise ValueError("casekit.anneal: scales must be positive, residual_tol nonnegative")
        if block_size < 2 or n_blocks < 4:
            raise ValueError("casekit.anneal: block_v2 needs block_size >= 2, n_blocks >= 4")
    total_energy = getattr(engine, "total_energy" + suffix)
    P = np.asarray(P).copy()
    state = {} if restart is None else restart
    model = dict(engine=_ENGINE_KEY.get(engine.__name__.split(".")[-1], engine.__name__.split(".")[-1]), alpha=alpha, ops=ops, dt=dt, L=L,
                 E_ext=E_ext, kT=kT, noise_cutoff_nm=noise_cutoff_nm,
                 elastic_params=elastic_params, seed=seed, scheme=scheme,
                 criterion=criterion, sample_every=sample_every, tol=tol, n_blocks=n_blocks,
                 block_size=block_size, energy_scale=energy_scale,
                 polarization_scale=polarization_scale, residual_tol=residual_tol)
    # resume_anneal binds these archived values; a raw caller must not change
    # the recurrence or the sampling grid underneath an accumulated history.
    if restart is not None and (
            any(state["model"][k] != model[k]
                for k in ("dt", "L", "scheme", "sample_every", "block_size", "n_blocks", "kT"))
            # .get: a pre-V2-04.0 certificate has no such key and means None
            or state["model"].get("noise_cutoff_nm") != noise_cutoff_nm):
        raise ValueError("casekit.anneal: restart recurrence/sampling parameters changed")
    trace = list(state.get("trace", []))
    observations = list(state.get("observations", []))
    start = int(state.get("step", 0))
    t0 = float(state.get("t", 0.0))
    previous = state.get("P_previous")
    rng = None
    if kT is None:
        step = getattr(engine, "step" + suffix)
        Minv = getattr(engine, "build_Minv" + suffix)(grid, ops, dt, L)
        Minv2 = (getattr(engine, "build_Minv2" + suffix)(grid, ops, dt, L)
                 if scheme == "sbdf2" else None)
    else:
        step = getattr(engine, "step_langevin" + suffix)
        build_ou = getattr(engine, "build_ou_operators" + suffix)
        # None goes through the SAME positional call as before V2-04.0, so a
        # cutoff-free anneal is byte-identical (K2 `anneal_noise_cutoff_none_byte`).
        ou = (build_ou(ops, dt, L, kT) if noise_cutoff_nm is None else
              build_ou(ops, dt, L, kT, noise_cutoff_nm=noise_cutoff_nm, grid=grid))
        rng = np.random.default_rng(seed)
        if state.get("rng_state") is not None:
            rng.bit_generator.state = state["rng_state"]
    reason, residual, n = "n_max", None, start
    for n in range(start + 1, start + n_max + 1):
        if kT is not None:
            new = step(P, alpha, ops, ou, dt, L, E_ext, rng, elastic_params=elastic_params)
        elif scheme == "sbdf2" and previous is not None:
            new = getattr(engine, "step_sbdf2" + suffix)(P, previous, alpha, ops, Minv2, dt, L,
                                                        E_ext, elastic_params=elastic_params)
        else:
            new = step(P, alpha, ops, Minv, dt, L, E_ext, elastic_params=elastic_params)
        previous, P = P, new
        if criterion == "block_v2" and not np.all(np.isfinite(P)):
            reason = "nonfinite"
            break
        if n % sample_every:
            continue
        if criterion == "legacy" and not np.all(np.isfinite(P)):
            reason = "nonfinite"           # legacy looks only at its own sample points
            break
        dens, tot = total_energy(P, alpha, ops, E_ext, grid, elastic_params=elastic_params)
        trace.append(float(tot["total"]))
        observations.append([float(dens["total"]), *P.reshape(3, -1).mean(axis=1),
                             float(np.mean(np.sum(P * P, axis=0)))])
        if tol < 0:
            continue
        if criterion == "legacy":
            if len(trace) >= 2 * n_blocks and solver.block_stationary(
                    trace, n_blocks_check=n_blocks, tol=tol):
                reason = "stationary"
                break
            continue
        if len(observations) < n_blocks * block_size or len(observations) % block_size:
            continue
        blocks = np.asarray(observations[-n_blocks * block_size:]).reshape(
            n_blocks, block_size, 5).mean(axis=1)
        scales = [energy_scale, *([polarization_scale] * 3), polarization_scale ** 2]
        stationary = all(solver.block_stationary(blocks[:, j], n_blocks, tol, scale=scales[j])
                         for j in range(5))
        if kT is None:
            force = getattr(engine, "total_force" + suffix)(
                P, alpha, ops, E_ext, elastic_params=elastic_params)[-1]
            residual = float(np.max(np.abs(L * force)))
            stationary = stationary and residual <= residual_tol
        if stationary:
            reason = "stationary"
            break
    result = (P, trace, reason)
    if return_state:
        state = dict(step=n, t=t0 + (n - start) * dt, trace=trace, observations=observations,
                     P_previous=previous,
                     rng_state=None if rng is None else rng.bit_generator.state,
                     model=model, stop_reason=reason, residual=residual)
        return (*result, state)
    return result


def anneal_adaptive(engine, P, alpha, ops, grid, *, dt0, L=1.0, E_ext=(0., 0., 0.),
                    kT=None, noise_cutoff_nm=None, elastic_params=None, stab=0.0,
                    err_tol=1e-3, k_min=-8, k_max=6, k_start=0, dt_max=None, grow_after=10,
                    energy_check="sample", energy_rtol=1e-12, n_max, t_end=None,
                    sample_every=10, tol=0.01, n_blocks=4, criterion="legacy", block_size=8,
                    energy_scale=None, polarization_scale=None, residual_tol=1e-6,
                    dPdt_tol=np.inf, rate_tol=np.inf, n_windows=5, restart=None,
                    callback=None):
    """anneal() with an ADAPTIVE dt, for DETERMINISTIC relaxation only
    (solver, bulk3d or film3d). Returns (P, trace, stop_reason, state):
    trace = [(t, E_total, dt), ...] (initial state first, then one entry
    every `sample_every` accepted steps), stop_reason in {"stationary",
    "n_max", "t_end", "dt_min"}, state = the restart certificate plus the
    step log and counters (accepted/rejected steps, operator builds, energy
    and force evaluations, seconds). docs/ADAPTIVE_DT.md is the user guide.

    SBDF2 on the ladder dt0*2**k (k_min..k_max, dt <= dt_max), variable-step
    SBDF2 right after a dt change, error estimate from the scheme's own
    linear-extrapolation predictor (no extra force evaluation), reject and
    halve when err > err_tol (polarization units), double after grow_after
    consecutive steps with err < err_tol/4; an energy rise beyond round-off
    is rejected per sampling window (energy_check="sample") or per step
    ("step"). `stab` is the engines' linear stabilization S (it changes the
    time accuracy -- relaxation only, re-audit dynamics). Full description:
    solvers/adaptive.py.

    STOPPING: anneal's own rules -- criterion="legacy" (block_stationary on
    the sampled energies, tol) or "block_v2" (block means of the five
    observables against explicit energy_scale/polarization_scale, plus
    max|L*dF/dP| <= residual_tol) -- or "plateau" (the film cases' rule:
    residual <= residual_tol, max|dP|/dt <= dPdt_tol, |d f|/dt <= rate_tol
    at n_windows consecutive samples) or None; plus n_max ACCEPTED steps and
    t_end. Samples are every `sample_every` ACCEPTED steps, so their spacing
    in time follows dt. For film3d pass ops={"G11" (or "G"): ...,
    "kappa_b": ..., "H11": optional}.

    REFUSED: kT / noise_cutoff_nm (Langevin). An adaptive dt has no place in
    an OU/Langevin run -- the noise amplitude and the discrete FDT are tied
    to one fixed dt -- and a fixed, certified dt (timestep.verified_dt) is
    what such runs need."""
    if kT is not None or noise_cutoff_nm is not None:
        raise ValueError("casekit.anneal_adaptive: deterministic runs only -- Langevin "
                         "(kT/noise_cutoff_nm) needs a fixed dt; use casekit.anneal")
    validate.anneal(engine, P, alpha, grid, E_ext)     # solver/bulk3d inputs; film passes
    from fftpf import timestep_adaptive as adaptive
    return adaptive.integrate(
        engine, P, alpha, ops, grid, dt0=dt0, L=L, E_ext=E_ext, elastic_params=elastic_params,
        stab=stab, err_tol=err_tol, k_min=k_min, k_max=k_max, k_start=k_start, dt_max=dt_max,
        grow_after=grow_after, energy_check=energy_check, energy_rtol=energy_rtol,
        n_max=n_max, t_end=t_end, sample_every=sample_every, criterion=criterion, tol=tol,
        n_blocks=n_blocks, block_size=block_size, energy_scale=energy_scale,
        polarization_scale=polarization_scale, residual_tol=residual_tol, dPdt_tol=dPdt_tol,
        rate_tol=rate_tol, n_windows=n_windows, restart=restart, callback=callback)


def resume_anneal(path, *, n_max):
    """Continue an anneal from a checkpoint written with
    io_utils.save_state_npz(restart=anneal's state), using the ARCHIVED
    inputs rather than this caller's defaults -- same engine, alpha, ops,
    grid, dt, scheme, criterion, scales, seed, the SBDF2 previous field and
    the RNG state. `n_max` is the number of ADDITIONAL steps; the return is
    anneal's 4-tuple (return_state=True), so a run can be split again.

    The certificate this makes possible: 200 steps + resume(200) is
    BIT-IDENTICAL to one uninterrupted 400-step anneal, deterministic or
    Langevin -- gates/core/k7_protocols.py item `anneal_semantics` leg (e),
    gates/core/k8_smoke.py item `checkpoint_roundtrip`."""
    data = io_utils.load_state_npz(path)
    checkpoint = data.get("checkpoint", {})
    state = checkpoint.get("restart")
    if state is None:
        raise ValueError("casekit.resume_anneal: snapshot carries no restart state")
    model = dict(state["model"])
    engine = {"solver": solver, "bulk3d": bulk3d}[model.pop("engine")]
    alpha, ops = model.pop("alpha"), model.pop("ops")
    return anneal(engine, data["P"], alpha, ops, checkpoint["grid"],
                  n_max=n_max, restart=state, return_state=True, **model)


def relax(engine, P, alpha, ops, grid, *, E_ext=(0., 0., 0.), elastic_params=None,
          method="lbfgs", tau=None, gtol=1e-6, ftol=1e-10, max_iter=10000, memory=10,
          sample_every=10, max_step=None, precondition=True, energy_scale=None,
          exact_energy=False, callback=None, kT=None, noise_cutoff_nm=None):
    """Drive `engine` (solver or bulk3d) from P straight to a STATIONARY
    state by minimizing its own F_total (solvers/minimize.py,
    docs/MINIMIZE.md) instead of integrating the gradient flow. Returns
    (P, trace, stop_reason, info): trace = F_total at iteration 0, every
    `sample_every` iterations and at the end; stop_reason in {"converged",
    "max_iter", "roundoff_floor", "line_search_failed"} (minimize.minimize
    says what each means); info carries n_iter, n_energy_evals,
    n_grad_evals, max_force (final max|dF/dP| density, the engine's force
    units), energy (the engine's total_energy 'total' of the returned P),
    wall_time, tau and the rest of minimize.minimize's record.

    STATIONARY STATES ONLY. No time, no kinetics, no Langevin: kT and
    noise_cutoff_nm are accepted only so that a call ported from anneal fails
    loudly instead of silently dropping the noise. From the same P, relax and
    anneal may end in DIFFERENT local minima (domain configurations); from a
    small-noise start, warm up with anneal (default stop rule) and relax its
    result to stay in the flow's basin (docs/MINIMIZE.md).

    method: "lbfgs" (preconditioned L-BFGS, strong-Wolfe, monotone energy)
    or "fire". tau: preconditioner scale, (I + tau*A)^-1 with A the engine's
    implicit operator A(k) + D(k)/kappa -- None picks minimize.default_tau
    (~ the inverse Landau stiffness); your production dt is a fine choice.
    Stops when max|force| < gtol AND the last relative energy change < ftol,
    or after max_iter iterations. max_step caps max|dP| per iteration.
    SPM kappa(r) / C(r) and ensemble-batched P are refused (ValueError)."""
    if kT is not None or noise_cutoff_nm is not None:
        raise ValueError("casekit.relax: a minimizer has no Langevin mode -- it finds "
                         "stationary states of F_total; use casekit.anneal for noise")
    from fftpf import energy_minimize as minimize          # lazy: casekit imports work without it
    problem = minimize.PeriodicProblem(engine, alpha, ops, grid, E_ext=E_ext,
                                       elastic_params=elastic_params, tau=tau,
                                       precondition=precondition, exact_energy=exact_energy)
    return minimize.minimize(problem, P, method=method, gtol=gtol, ftol=ftol,
                             max_iter=max_iter, memory=memory, sample_every=sample_every,
                             energy_scale=energy_scale, max_step=max_step, callback=callback)


def relax_film(P, alpha, grid, G, kappa_b, *, E_ext=(0., 0., 0.), el=None, H11=None,
               poisson_factors=None, method="lbfgs", tau=None, gtol=1e-6, ftol=1e-10,
               max_iter=10000, memory=10, sample_every=10, max_step=None,
               precondition="depolarization", energy_scale=None, exact_energy=False,
               callback=None, kT=None):
    """casekit.relax for the film (film3d + optional film_elastic `el`), with
    film3d's own argument set (G scalar or {G11,G12,G44,G44p}, kappa_b scalar
    or a (Nz,) profile, H11, a prebuilt poisson_factors). Same return and the
    same caveats as relax. info["max_force"] is the quantity
    cases/bto_film_biaxial.py logs as force_max_deck.

    precondition: "depolarization" (default) -- per in-plane mode, the exact
    3Nz x 3Nz symbol of gradient + H11 + short-circuit depolarization, so the
    explicit 1/kappa_b stiffness that bounds the film's stable dt is
    preconditioned too (memory 16*Ny*(Nx//2+1)*(3*Nz)^2 bytes: 53 MB at
    40x40x21); "gradient" -- the stepper's own banded factors
    (film3d.build_grad_factors), no extra memory; False -- none.
    The film's SPM (film3d_spm factors, film elastic SPM) is refused."""
    if kT is not None:
        raise ValueError("casekit.relax_film: a minimizer has no Langevin mode; use "
                         "film3d_langevin for noise")
    from fftpf import energy_minimize as minimize
    problem = minimize.FilmProblem(alpha, grid, G, kappa_b, E_ext=E_ext, el=el, H11=H11,
                                   poisson_factors=poisson_factors, tau=tau,
                                   precondition=precondition, exact_energy=exact_energy)
    return minimize.minimize(problem, P, method=method, gtol=gtol, ftol=ftol,
                             max_iter=max_iter, memory=memory, sample_every=sample_every,
                             energy_scale=energy_scale, max_step=max_step, callback=callback)


def run_dir(case_name):
    """Create and return reports/runs/<YYYYMMDD-HHMM>_<case_name>/ (SPEC
    S41.1 A3(iii)). Under tools/job.py the whole folder name comes from the
    FFT_RUN_ID env var instead, so the case and the job registry agree on
    one directory; `case_name` is then unused. FFT_RUN_ROOT optionally
    overrides reports/runs, including when job.py uses --run-root.

    Absorbs: the module-level REPORTS_DIR/RAW_PATH constants 69 of 98 cases
    carry. MIGRATION RULE, unchanged: NEW RUNS ONLY. Nothing existing moves,
    and a case whose output a gate reads keeps writing its legacy
    reports/<name> path -- it may additionally copy into the run folder.

    COLLISIONS (SPEC V2-02.1 E13): the minute-resolution id is DOCUMENTED
    (README.md:137, docs/API_INDEX.md, ENGINE_GUIDE.md) and stays; two
    foreground runs of the same case inside one minute no longer share a
    folder -- the second gets `_2`, the third `_3`. (The branch's
    time+uuid8 id was rejected: it breaks every documented example, and the
    FFT_RUN_ID handshake already de-duplicates job-launched runs.)"""
    run_id = os.environ.get("FFT_RUN_ID")
    root = os.environ.get("FFT_RUN_ROOT", default_run_root())
    if run_id:
        path = os.path.join(root, run_id)
        os.makedirs(path, exist_ok=True)      # job.py created it before exec
        return path
    base = os.path.join(root, f"{time.strftime('%Y%m%d-%H%M')}_{case_name}")
    path, k = base, 2
    while os.path.exists(path):
        path, k = f"{base}_{k}", k + 1
    os.makedirs(path)
    return path


def save_run(rundir, *, params, P, grid, trace, thread_line, extra=None,
             state=None, input_paths=()):
    """Write the standard run-folder set into `rundir`: params.json (params
    + the thread line + the git commit + the engine name/version + the
    source provenance), state.npz, energy_trace.csv, an extra.npz if `extra`
    is given, and quicklook.png last.

    Absorbs: the io_utils.save_state_npz + save_quicklook_png + hand-rolled
    provenance block cases repeat. Order is deliberate -- the PNG is written
    last so a machine without matplotlib (optional per ENGINE_GUIDE S1.1)
    still gets every number.

    params: JSON-serializable run record; two optional keys are read for
        save_state_npz's provenance stamps, "n_steps" and "t".
    state: anneal(return_state=True)'s restart certificate. Pass it and the
        record's n_steps/t/stop_reason come from the RUN instead of the
        caller's arithmetic, and state.npz becomes resumable
        (casekit.resume_anneal). Without it -- and without n_steps/t in
        params -- this WARNS and stamps 0, as it always has: a missing step
        count is a hole in the record, not a reason to lose a finished run
        (SPEC V2-02.1 E3; the branch raised here).
    input_paths: extra files whose SHA256 belongs in the provenance stamp
        (a deck TOML, a quenched-field npz) on top of every solvers/*.py.
    extra: {name: array} of extra fields worth keeping (the quenched E_d,
        a disordered alpha1 map) -> extra.npz, and the same arrays into
        state.npz's metadata tree. Arrays only; scalars belong in params.
    A 4D P (3,Nz,Ny,Nx) is quicklooked at its mid-z plane; the npz holds
    the full state either way."""
    record = dict(params, thread_line=thread_line, commit=_git_commit(),
                  engine=f"{ENGINE_NAME} {ENGINE_VERSION}",
                  provenance=io_utils.provenance(input_paths))
    if state is not None:
        record.update(n_steps=state["step"], t=state["t"], stop_reason=state["stop_reason"])
    missing = [k for k in ("n_steps", "t") if k not in record]
    if missing:
        warnings.warn(f"casekit.save_run({os.path.basename(rundir)}): no true "
                      f"{'/'.join(missing)} -- state.npz stamps 0. Pass anneal's "
                      f"state= (or the key in params) for the real value.", stacklevel=2)
    with open(os.path.join(rundir, "params.json"), "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True, default=str)
    io_utils.save_state_npz(os.path.join(rundir, "state.npz"), P, grid,
                            step=int(record.get("n_steps", 0)),
                            t=float(record.get("t", 0.0)), prm=record,
                            alpha=None if state is None else state["model"]["alpha"],
                            restart=state, fields=extra)
    np.savetxt(os.path.join(rundir, "energy_trace.csv"), np.asarray(trace, dtype=float),
               header="total energy, one row per sample", comments="# ")
    if extra:
        np.savez_compressed(os.path.join(rundir, "extra.npz"), **extra)
    P = np.asarray(P)
    zmid = P.shape[1] // 2 if P.ndim == 4 else None
    io_utils.save_quicklook_png(
        os.path.join(rundir, "quicklook.png"), P if zmid is None else P[:, zmid], grid,
        title=os.path.basename(rundir) + ("" if zmid is None else f"  (z={zmid})"))


def _git_commit():
    """Short HEAD hash, "+dirty" if the working tree has changes, "unknown"
    if git is unavailable (a tarball copy of the repo, no .git)."""
    # An installed wheel can live inside another project's virtual environment.
    if not os.path.exists(os.path.join(_ROOT, ".git")):
        return "unknown"
    try:
        out = subprocess.run(["git", "-C", _ROOT, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", _ROOT, "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"
    return (out + "+dirty") if (out and dirty) else (out or "unknown")


# --------------------------------------------------------------------------
# Ensemble sweeps over the batch axis (WP4 §10e / SPEC V2-03 C3 layout
# (3,S,*space)). One process advances S independent DETERMINISTIC members
# with the ordinary steppers; on small and medium grids that amortises the
# per-call overhead (docs/CPU_PARALLELISM.md, "Ensembles"). Langevin members
# share ONE kT and ONE Generator: their draws are independent, but the
# temperature cannot differ between members, and a batched run reproduces the
# per-member runs in distribution only (bulk3d.step_langevin_3d's note).
# The 2D Langevin stepper draws an unbatched (3,Ny,Nx) field -- do not batch
# it. film3d has no ensemble axis.
# --------------------------------------------------------------------------
def ensemble_initial(engine, grid, seeds, amplitude, kmax_frac=0.3):
    """(3,S,*space) initial batch whose member s is exactly the single-member
    random_smooth_field (solver) / random_smooth_field_3d (bulk3d) at
    seeds[s], bit for bit."""
    seeds = list(seeds)
    if not seeds:
        raise ValueError("casekit.ensemble_initial: give at least one seed")
    if hasattr(engine, "step_3d"):
        return bulk3d.random_smooth_field_3d(grid, 3, amplitude, kmax_frac, seeds=seeds)
    return np.stack([solver.random_smooth_field(grid, 3, amplitude, kmax_frac, s)
                     for s in seeds], axis=1)


def ensemble_alpha(alphas, space_ndim):
    """Merge one Landau dict per member (e.g. materials.get(deck, T=T)["landau"]
    for a temperature sweep) into a single alpha for a (3,S,*space) batch.

    A coefficient equal in every member is kept as given (so a scalar stays a
    scalar); one that differs becomes an array whose leading axis is S,
    padded to broadcast against P.shape[1:] = (S,*space): (S,1,1) in 2D,
    (S,1,1,1) in bulk3d, or (S,*field) for spatial coefficient fields.
    A key some members lack is read as 0.0 (the 8th-order convention)."""
    alphas = list(alphas)
    if not alphas:
        raise ValueError("casekit.ensemble_alpha: give at least one member")
    out = {}
    for key in dict.fromkeys(k for a in alphas for k in a):
        values = [np.asarray(a.get(key, 0.0), dtype=float) for a in alphas]
        if all(v.shape == values[0].shape and np.array_equal(v, values[0]) for v in values):
            out[key] = alphas[0].get(key, 0.0)
            continue
        shape = np.broadcast_shapes(*(v.shape for v in values))
        if len(shape) > space_ndim:
            raise ValueError(f"casekit.ensemble_alpha: {key} has shape {shape}, more than "
                             f"{space_ndim} spatial axes")
        padded = (1,) * (space_ndim - len(shape)) + shape
        out[key] = np.stack([np.broadcast_to(v, shape).reshape(padded) for v in values])
    return out
