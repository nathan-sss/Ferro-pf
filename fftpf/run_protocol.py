"""RUN-ENTRY packages E3a / E3b (SPEC § 16.5): the stages and the run loop.

Every stage steps through the Adapter only (no engine call of its own for stepping); readings come from
run_measure.  Stage kinds:

  relax, anneal   deterministic / Langevin stepping to a stop rule (E3a, unchanged numerics);
  ac, fdt       dielectric spectrum (E3c): small-signal lock-in tones and the equilibrium-fluctuation route, definitions in
                  run_measure ("dielectric spectrum") and the two functions below;
  cool, hold, pe_loop   scheduled drive (E3b): ONE stepping loop, ``drive``, fed by one schedule function per kind
                  that returns (total steps, T(s), E(s), event(s), record_every) in the manner of the frozen driver
                  Runs/NBT6BT/two_well_20261104/run_nbt6bt.py (``schedule``): s = 0 is the state on entry, step s runs at
                  temperature T(s) under the applied field E(s) [kV/cm] along the unit axis n, on top of the static
                  [[field]] E_ext (E(s) = 0 -> the static field alone).

A stage function takes the built dict ``b`` (run_builder.build), its config table ``st``, the start field ``P`` and
the elapsed deck time ``t0`` (and ``ctx``) and returns a dict: P, step, t, stop_reason, trace (sampled total energies),
samples (every reading with its step and t, the end of the stage included), P_previous, rng_state.

RNG rule: ONE ``numpy.random.default_rng(run.seed)`` for the whole run, carried through every stage and every
checkpoint; a stage that sets its own ``seed`` starts a new generator from it, which the later stages then carry on.

``run(cfg_or_path)`` is the in-process entry the CLI uses; it resumes from a checkpoint of the same config found in
output.dir (run_output).
"""
import copy
import json
import os

import numpy as np

from fftpf import backend_config, run_builder, run_input, run_measure, run_output, units

# relax stop rule: the film relax loop of cases/bto_film_biaxial.py.
CHECK_EVERY = 51           # a check every 51 steps (and at step n_max)
PLATEAU_WINDOWS = 5        # consecutive good checks that end the stage ("converged")
DELTA_P_TOL = 1e-5         # max|P - P at the previous check|
F_RATE_TOL = 1e-8          # |f - f at the previous check| / elapsed deck time
DEFAULT_RECORD = 100       # record_every of cool / hold; a pe_loop takes the largest divisor of its quarter cycle <= this




def new_ctx(cfg):
    """The state carried from stage to stage: the run's one RNG, steps done, the applied field E [kV/cm] along the unit axis n."""
    return dict(rng=np.random.default_rng(cfg["run"]["seed"]), step0=0, E=0.0, n=None, E_entry=0.0)


def _reading(b, E, P, step, t, ctx=None, E_kV=0.0, axis=None):
    s = run_measure.sample(b["adapter"], P, E, b["cfg"]["measure"], axis)
    s.update(step=step, step_total=(0 if ctx is None else ctx["step0"]) + step, t=t,
             T_K=float("nan") if b["T"] is None else float(b["T"]), E_kV_cm=float(E_kV))
    return s


def anneal(b, st, P, t0=0.0, ctx=None):
    """Fixed-T stage (Langevin when ``st["langevin"]``), reproducing casekit.anneal bit for bit.

    One step per loop turn from P; a sample (total energy appended to ``trace``) every
    ``sample_every`` steps, or only at the last step when sample_every = 0.  Stop rules:
    ``n_steps`` -> exactly that many steps, no stop test (casekit tol < 0);
    ``block_stationary`` = tol -> casekit's "legacy" rule: the last ``n_blocks`` (default 4)
    sampled energies mutually within tol of their mean magnitude, first tested once
    2 * n_blocks samples exist (pf2d_engine.block_stationary), at most ``n_max`` steps.
    A non-finite field at a sample point stops with reason "nonfinite".  The Langevin RNG is the run's
    carried generator (ctx["rng"]; ``numpy.random.default_rng(st["seed"])`` when the stage sets a seed).
    """
    from fftpf.pf2d_engine import block_stationary
    ctx = ctx or new_ctx(b["cfg"])
    ad, E, dt = b["adapter"], b["E_static"], b["dt"]
    stop = st["stop"]
    tol = stop.get("block_stationary", -1.0)
    n_max = stop["n_steps"] if "n_steps" in stop else stop["n_max"]
    n_blocks, every = stop.get("n_blocks", 4), st["sample_every"]
    rng = ctx["rng"] if st["langevin"] else None
    P = np.asarray(P).copy()
    previous, trace, samples, reason = None, [], [], "n_max"
    n = 0
    for n in range(1, n_max + 1):
        new = ad.step(P, E, rng=rng, P_prev=previous)
        previous, P = P, new
        if (n % every if every else n != n_max):
            continue
        if not np.all(np.isfinite(P)):
            reason = "nonfinite"
            break
        s = _reading(b, E, P, n, t0 + n * dt, ctx)
        samples.append(s)
        trace.append(s["E_total"])
        if tol < 0:
            continue
        if len(trace) >= 2 * n_blocks and block_stationary(trace, n_blocks_check=n_blocks, tol=tol):
            reason = "stationary"
            break
    if reason != "nonfinite" and (not samples or samples[-1]["step"] != n):    # stage end off the sample grid
        samples.append(_reading(b, E, P, n, t0 + n * dt, ctx))
    return dict(P=P, step=n, t=t0 + n * dt, stop_reason=reason, trace=trace, samples=samples,
                P_previous=previous, rng_state=None if rng is None else rng.bit_generator.state)


def relax(b, st, P, t0=0.0, ctx=None):
    """Deterministic stepping (numerics.scheme) until the film relax rule or ``stop.n_max`` steps.

    Rule (cases/bto_film_biaxial.py): every CHECK_EVERY = 51 steps and at the last step,
    a check is good when max|L dF/dP| <= ``stop.tol`` AND max|P - P at the previous check| < 1e-5
    AND |f - f at the previous check| / elapsed deck time < 1e-8; PLATEAU_WINDOWS = 5 consecutive
    good checks end the stage ("converged").  The initial state is read first (it is the
    reference of the first check) and is the first sample.  ``trace`` is the sampled E_total.
    """
    ctx = ctx or new_ctx(b["cfg"])
    ad, E, dt = b["adapter"], b["E_static"], b["dt"]
    tol, n_max = st["stop"]["tol"], st["stop"]["n_max"]
    P = np.asarray(P).copy()
    first = _reading(b, E, P, 0, t0, ctx)
    samples, trace = [first], [first["E_total"]]
    previous_P, previous_f, last, good_run = P.copy(), first["f_total"], 0, 0
    previous, reason, n = None, "n_max", 0
    for n in range(1, n_max + 1):
        new = ad.step(P, E, P_prev=previous)
        previous, P = P, new
        if not np.all(np.isfinite(P)):
            reason = "nonfinite"
            break
        if n % CHECK_EVERY and n != n_max:
            continue
        s = _reading(b, E, P, n, t0 + n * dt, ctx)
        samples.append(s)
        trace.append(s["E_total"])
        delta_P = float(np.max(np.abs(P - previous_P)))
        f_rate = abs(s["f_total"] - previous_f) / ((n - last) * dt)
        good = run_measure.force_max(ad, P, E) <= tol and delta_P < DELTA_P_TOL and f_rate < F_RATE_TOL
        good_run = good_run + 1 if good else 0
        previous_P, previous_f, last = P.copy(), s["f_total"], n
        if good_run >= PLATEAU_WINDOWS:
            reason = "converged"
            break
    return dict(P=P, step=n, t=t0 + n * dt, stop_reason=reason, trace=trace, samples=samples,
                P_previous=previous, rng_state=None)


# --------------------------------------------------------------------------- scheduled stages (E3b)

def deck_field(E_kV_cm):
    """kV/cm -> deck field (1 kV/cm = 1e5 V/m; units.si_to_deck_field)."""
    return units.si_to_deck_field(E_kV_cm * 1e5)


def drive_field(b, amp, n):
    """Static E_ext plus the uniform deck field ``amp`` along the unit axis n."""
    static = b["E_static"]
    if isinstance(static, tuple):                       # no static field: a plain 3-vector, as run_unipolar passes it
        return tuple(float(amp * x) for x in n)
    return static + amp * np.reshape(n, (3,) + (1,) * (static.ndim - 1))


def efield(b, E_kV_cm, n):
    """Total deck field E(t) n + static E_ext (driver B ``Model.field``); E = 0 gives the static field alone."""
    if E_kV_cm == 0.0:
        return b["E_static"]
    return drive_field(b, deck_field(E_kV_cm), n)


def unit_axis(axis):
    """The axis normalised as driver B normalises n."""
    return [float(x) for x in np.asarray(axis, dtype=float) / np.linalg.norm(axis)]


def triangle(s, sps):
    """+1 at s = sps, 0 at 2 sps, -1 at 3 sps, 0 at 4 sps; exact at the turning points and zero crossings (driver B)."""
    q, r = divmod(s, sps)
    x = r / sps
    return (x, 1. - x, 0. - x, x - 1.)[q % 4]


def default_every(quarter):
    """Largest divisor of the quarter cycle that is <= DEFAULT_RECORD."""
    return max(d for d in range(1, min(DEFAULT_RECORD, quarter) + 1) if quarter % d == 0)


def stage_T(b, st):
    """The temperature a stage starts at (None: keep the current one)."""
    if st["kind"] == "cool":
        return st["rungs"][0][0]
    return st.get("T_K", b["cfg"]["material"].get("temperature_K"))


def schedule(b, st, ctx):
    """A scheduled stage as functions of its step s: (total, T(s), E(s) [kV/cm], event(s) -> name or None, record_every, axis).

    ``axis`` is the new unit axis or None (keep ctx["n"]).  Event names follow driver B."""
    kind = st["kind"]
    axis = unit_axis(st["axis"]) if "axis" in st else None
    E = (lambda s: 0.)
    if kind == "cool":
        rungs = st["rungs"]
        ends = np.cumsum([n for _, n in rungs])
        total = int(ends[-1])
        T = lambda s: rungs[int(np.searchsorted(ends, s))][0]
        event = lambda s: f"T{T(s):g}" if s in ends else None
        return total, T, E, event, st.get("record_every", DEFAULT_RECORD), axis
    T0 = stage_T(b, st)
    T = (lambda s: T0)
    if kind == "hold":
        total, ramp, target, E0 = st["n_steps"], st["ramp_steps"], st["E_kV_cm"], ctx["E_entry"]
        if ramp:
            E = lambda s: target + (E0 - target) * (1. - s / ramp) if s < ramp else target
        else:
            E = lambda s: target
        event = lambda s: "end" if s == total else "ramped" if s == ramp else None
        return total, T, E, event, st.get("record_every", DEFAULT_RECORD), axis
    Emax = st["E_max_kV_cm"]                                          # pe_loop
    if st["waveform"] == "unipolar":
        from fftpf.pf2d_energy_storage_loop import e_path_unipolar
        n1, nd = st["n_points"], st["n_dwell"]
        path = e_path_unipolar(n1, Emax)
        total = 2 * n1 * nd
        E = lambda s: float(path[(s - 1) // nd]) if s else 0.0
        event = lambda s: "Emax" if s == n1 * nd else "end" if s == total else None
        return total, T, E, event, nd, axis
    sps = st["steps_per_cycle"] // 4
    total = int(round(4 * st.get("cycles", 1.0))) * sps
    E = lambda s: Emax * triangle(s, sps)
    names = {1: "Emax", 2: "E0down", 0: "E0up"}                       # +Emax; E = 0 falling; E = 0 rising

    def event(s):
        q, r = divmod(s, sps)
        return f"c{(s - 1) // (4 * sps) + 1}_{names[q % 4]}" if s and not r and q % 4 in names else None
    return total, T, E, event, st.get("record_every") or default_every(sps), axis


def drive(b, st, P, ctx, i, out, resume=None):
    """The one stepping loop of cool / hold / pe_loop (driver B ``_run``).

    resume: dict(stage_step, P_prev, samples) -- continue a stage from a checkpoint.  Rows go to ``out`` as they are
    recorded; states at events; a checkpoint every output.checkpoint_every steps of the stage (the stage end is
    checkpointed by ``run``)."""
    ad, dt, kind = b["adapter"], b["dt"], st["kind"]
    ce = b["cfg"]["output"]["checkpoint_every"]
    first, previous, samples = (0, None, []) if resume is None else (resume["stage_step"], resume["P_prev"], resume["samples"])
    if first == 0:
        ctx["E_entry"] = ctx["E"]                       # the field on entry stays in the checkpoint (a ramp needs it)
    total, T_of, E_of, event, every, axis = schedule(b, st, ctx)
    if axis is not None:
        ctx["n"] = axis
    n, rng, step0, reason = ctx["n"], ctx["rng"] if st["langevin"] else None, ctx["step0"], "n_steps"

    def record(s, E, field):
        smp = _reading(b, field, P, s, (step0 + s) * dt, ctx, E, n)
        samples.append(smp)
        out.add_row(i, kind, smp)

    s = first
    if first == 0:
        run_builder.set_T(b, T_of(0))
        ctx["E"] = E_of(0)
        record(0, ctx["E"], efield(b, ctx["E"], n))
    for s in range(first + 1, total + 1):
        T, E = T_of(s), E_of(s)
        run_builder.set_T(b, T)
        field = efield(b, E, n)
        new = ad.step(P, field, rng=rng, P_prev=previous)
        previous, P = P, new
        ctx["E"] = E
        name = event(s)
        if (s % every == 0 or s == total or name) and not np.all(np.isfinite(P)):
            reason = "nonfinite"
            break
        if s % every == 0 or s == total or name:
            record(s, E, field)
        if name:
            out.write_event(i, kind, name, P, previous, ctx, s, T, E, step0 + s, (step0 + s) * dt)
        if ce and s % ce == 0 and s != total:
            out.checkpoint(P, previous, ctx, i, s, out.summaries)
    return dict(P=P, step=s, t=(step0 + s) * dt, stop_reason=reason, trace=[x["E_total"] for x in samples], samples=samples,
                P_previous=previous, rng_state=rng.bit_generator.state if rng is not None else None, streamed=True)



# --------------------------------------------------------------------------- spectrum stages (E3c)

def _cell_data(b):
    """(N_cells, cell volume in nm^3 or nan, kT or nan) of the build at the present temperature."""
    g, hz = b["grid"], b["hz"]
    vol = float("nan") if hz is None else float(g["hx"] * g["hy"] * hz)
    return int(np.prod(b["space"])), vol, float("nan") if b["kT"] is None else float(b["kT"])


def _eps_b(b):
    """eps_b: physics.electrostatics.eps_b when given, else the deck's kappa_b / eps0_deck."""
    given = b["cfg"]["physics"]["electrostatics"].get("eps_b")
    return float(given) if given is not None else float(b["kappa"] / run_measure.eps0_deck())


def _end(b, P, step, t, ctx):
    return [_reading(b, b["E_static"], P, step, t, ctx)]


def ac(b, st, P, t0=0.0, ctx=None):
    """Small-signal lock-in tones (a stage of run_measure's "dielectric spectrum" definitions).

    Each tone of ``omega`` (rad per deck time) starts from the state the stage was entered with (a copy; the field is not
    carried from tone to tone) and runs  n_skip + n_keep  steps, n_skip = int(discard_periods * T / dt), n_keep =
    int((periods - discard_periods) * T / dt), T = 2 pi / omega (wp13's ``int(n_periods * period / DT)`` for the kept part).
    Step i is taken under  E = (dc_bias + E0 sin(omega i dt)) n  [deck] on top of the static E_ext -- the clock starts at 0 at
    the first driven step -- and records the raw series s_i = sum(P . n) of the field after the step.  Langevin at the
    stage temperature when ``langevin``, one carried generator (the tones draw in order); deterministic otherwise.  The
    lock-in sees the kept samples only (run_measure.lockin_deck turns the clock back).  The stage ends in the entry
    state: it is a measurement, and the carried generator has moved on."""
    ctx = ctx or new_ctx(b["cfg"])
    ad, dt = b["adapter"], b["dt"]
    n = unit_axis(st["axis"])
    E0, dc = deck_field(st["E0_kV_cm"]), deck_field(st["dc_bias_kV_cm"])
    rng = ctx["rng"] if st["langevin"] else None
    cells, vol, kT = _cell_data(b)
    P_in = np.asarray(P).copy()
    series, omega, chi_p, chi_pp, total, reason = {}, [], [], [], 0, "n_steps"
    for w in st["omega"]:
        period = 2 * np.pi / w
        n_skip = int(st["discard_periods"] * period / dt)
        n_keep = int((st["periods"] - st["discard_periods"]) * period / dt)
        x, P, previous = np.empty(n_skip + n_keep), P_in.copy(), None
        for i in range(n_skip + n_keep):
            t = i * dt
            new = ad.step(P, drive_field(b, dc + E0 * np.sin(w * t), n), rng=rng, P_prev=previous)
            previous, P = P, new
            x[i] = run_measure.total_along(P, n)
        total += n_skip + n_keep
        if not np.all(np.isfinite(P)):
            reason = "nonfinite"
            break
        cp, cpp = run_measure.lockin_deck(x, dt, w, E0, cells, n_skip)
        omega.append(w), chi_p.append(cp), chi_pp.append(cpp)
        series[f"ac_w{w:.6g}"] = dict(series=x, dt=dt, omega=w, E0=E0, dc_bias=dc, n_skip=n_skip, kT=kT, N_cells=cells,
                                      cell_volume_nm3=vol, axis=np.array(n))
    spectrum = None
    if reason != "nonfinite":
        spectrum = dict(lockin=dict(omega=np.array(omega), chi_p=np.array(chi_p), chi_pp=np.array(chi_pp),
                                    eps_b=_eps_b(b), E0=E0, N_cells=cells, cell_volume_nm3=vol, kT=kT))
    summary = dict(omega=omega, chi_p=[float(x) for x in chi_p], chi_pp=[float(x) for x in chi_pp], N_cells=cells,
                   cell_volume_nm3=vol, kT=kT, E0_deck=E0, units="chi per volume in deck units (P / E_deck)")
    return dict(P=P_in, step=total, t=t0 + total * dt, stop_reason=reason, trace=[], samples=_end(b, P_in, total, t0 + total * dt, ctx),
                P_previous=None, rng_state=None if rng is None else rng.bit_generator.state, series=series,
                spectrum=spectrum, summary=summary)


def fdt(b, st, P, t0=0.0, ctx=None):
    """Equilibrium fluctuation route: ``burn_steps`` steps, then ``n_steps`` steps recording the raw series s_i = sum(P . n)
    under the static fields alone (E = 0 otherwise), Langevin at the stage temperature with the carried generator.  The
    series gives run_measure.fdt_deck (nperseg, max_lag).  The stage ends in the last state of the trajectory."""
    ctx = ctx or new_ctx(b["cfg"])
    ad, dt, E, rng = b["adapter"], b["dt"], b["E_static"], ctx["rng"]
    n = unit_axis(st["axis"])
    cells, vol, kT = _cell_data(b)
    P, previous, reason = np.asarray(P).copy(), None, "n_steps"
    for _ in range(st["burn_steps"]):
        new = ad.step(P, E, rng=rng, P_prev=previous)
        previous, P = P, new
    x = np.empty(st["n_steps"])
    for i in range(st["n_steps"]):
        new = ad.step(P, E, rng=rng, P_prev=previous)
        previous, P = P, new
        x[i] = run_measure.total_along(P, n)
    total = st["burn_steps"] + st["n_steps"]
    spectrum, summary = None, dict(N_cells=cells, cell_volume_nm3=vol, kT=kT, nperseg=st["nperseg"], max_lag=st["max_lag"],
                                   units="chi per volume in deck units (P / E_deck)")
    if np.all(np.isfinite(P)):
        est = run_measure.fdt_deck(x, dt, kT, cells, st["nperseg"], st["max_lag"])
        spectrum = dict(fdt=dict(est, eps_b=_eps_b(b), N_cells=cells, cell_volume_nm3=vol, kT=kT))
        summary.update(chi_static_deck=float(est["chi_static"]), variance_series=float(np.var(x)))
    else:
        reason = "nonfinite"
    series = {"fdt": dict(series=x, dt=dt, omega=np.nan, E0=0.0, kT=kT, N_cells=cells, cell_volume_nm3=vol, axis=np.array(n))}
    return dict(P=P, step=total, t=t0 + total * dt, stop_reason=reason, trace=[], samples=_end(b, P, total, t0 + total * dt, ctx),
                P_previous=previous, rng_state=rng.bit_generator.state, series=series, spectrum=spectrum, summary=summary)

STAGES = {"relax": relax, "anneal": anneal, "ac": ac, "fdt": fdt}
DRIVEN = ("cool", "hold", "pe_loop")


def summary(i, res, st, cfg, fractions_at=None):
    """The measures.json entry of a finished stage (plus its loop measures).

    With measure.phase_fractions: ``fractions_end`` (the stage's last reading) and ``fractions_at`` ({event: fractions}
    of the event states the stage saved: run_output.RunDir.event_fractions)."""
    d = dict(stage=i, kind=res["kind"], steps=res["step"], t=res["t"], stop_reason=res["stop_reason"],
             end=res["samples"][-1] if res["samples"] else None)
    if cfg["measure"].get("phase_fractions") and d["end"] and "f_R" in d["end"]:
        d["fractions_end"] = {k: d["end"][f"f_{k}"] for k in run_measure.fraction_keys(cfg["measure"]["phase_fractions"])}
        d["fractions_at"] = fractions_at or {}
    if res.get("summary"):
        d["spectrum"] = res["summary"]
    if res["kind"] == "pe_loop" and cfg["measure"].get("loop") and res["stop_reason"] != "nonfinite":
        every = st["n_dwell"] if st["waveform"] == "unipolar" else (st.get("record_every") or default_every(st["steps_per_cycle"] // 4))
        d["loop"] = run_measure.loop(res["samples"], st, cfg["measure"]["loop"], every)
    return d


def check(cfg):
    """What the CLI ``check`` and ``explain`` do first: every refusal ``run`` raises before its first step (the dry build of
    run_builder, no array and no step) and an output.dir that cannot be written.  Creates nothing."""
    run_builder.check(cfg)
    run_output.check_dir(cfg)


def run(cfg_or_path, stop_after_checkpoints=None):
    """Run the file (path) or resolved config dict: run_one, or run_replicas when run.replicas > 1.

    With NumPy already imported in a process whose BLAS thread variables are not 1, backend_config.configure refuses; that
    is turned into a ConfigError that names the variables."""
    input_path = None
    if isinstance(cfg_or_path, (str, os.PathLike)):
        input_path = os.path.abspath(cfg_or_path)
        cfg = run_input.load(input_path)
    else:
        cfg = cfg_or_path
    run_output.check_dir(cfg)
    try:
        if cfg["run"]["replicas"] > 1:
            return run_replicas(cfg, input_path, stop_after_checkpoints)
        return run_one(cfg, input_path, stop_after_checkpoints)
    except RuntimeError as exc:
        if "before importing NumPy" not in str(exc):
            raise
        names = ", ".join(f"{k}=1" for k in backend_config.BLAS_ENV[:3])
        raise run_input.ConfigError(
            f"numerics.threads: NumPy was imported before the thread variables were set ({exc}); set {names} in the environment "
            "before importing NumPy (os.environ.update(fftpf.backend_config.environment(n)) as the first lines of the script), "
            "or start the run with `python -m fftpf run <file>`, which sets them for you") from None


def run_one(cfg, input_path=None, stop_after_checkpoints=None):
    """Run one resolved config; returns dict(run_dir, build, stages, P, n_checkpoints).

    ``stages`` holds one stage result per [[stage]] run by THIS call (see the module docstring) plus ``kind``.
    The run directory (run_output) is created after the build, so a refusal leaves nothing behind.
    A non-finite stage is written out, then raises FloatingPointError.
    An output.dir holding a checkpoint of the same config resumes from it.  ``stop_after_checkpoints``: test hook,
    run_output.RunInterrupted after that many checkpoints written in this call.
    """
    ident = run_output.identity(cfg)
    ck = run_output.find_checkpoint(cfg, ident)
    saved = None
    if ck is not None:
        saved = run_output.load_quenched(ck["dir"])
        if cfg["field"] and saved is None:
            raise run_input.ConfigError(f"output.dir: {ck['dir']} holds a checkpoint but no quenched.npz")
    b = run_builder.build(cfg, saved)
    out = run_output.RunDir(cfg, b, input_path, ident, resume=ck, stop_after=stop_after_checkpoints)
    ctx, results, P, i0, resume = new_ctx(cfg), [], None, 0, None
    out.summaries = []
    if ck is not None:
        m = ck["meta"]
        ctx.update(step0=m["step0"], E=m["E"], n=m["n"], E_entry=m["E_entry"])
        ctx["rng"].bit_generator.state = m["rng"]
        P, i0, out.summaries = ck["P"], m["stage_index"], list(m["summaries"])
        out.event_fractions = {i0: m.get("fractions_at", {})}
        if m["stage_step"] > 0:
            resume = dict(stage_step=m["stage_step"], P_prev=ck["P_prev"], samples=out.stage_samples(i0))
    for i in range(i0, len(cfg["stage"])):
        st = cfg["stage"][i]
        if resume is None:
            if "seed" in st:
                ctx["rng"] = np.random.default_rng(st["seed"])
            start = st["start"]
            if start["kind"] != "previous":
                P = run_builder.make_start(b["adapter"], start, f"stage[{i}]", b["space"])
            T = stage_T(b, st)
            if T is not None:
                run_builder.set_T(b, T)
        if st["kind"] in DRIVEN:
            res = drive(b, st, P, ctx, i, out, resume)
        else:
            ctx["E"] = 0.0                              # relax / anneal run under the static field alone
            res = STAGES[st["kind"]](b, st, P, ctx["step0"] * b["dt"], ctx)
        resume = None
        res["kind"] = st["kind"]
        P = res["P"]
        ctx["step0"] += res["step"]
        results.append(res)
        out.summaries.append(summary(i, res, st, cfg, out.event_fractions.get(i)))
        out.write_stage(i, res)
        if res["stop_reason"] == "nonfinite":
            raise FloatingPointError(f"stage[{i}] {st['kind']}: non-finite field by step {res['step']} "
                                     f"(run directory {out.path})")
        if cfg["output"]["checkpoint_every"]:
            out.checkpoint(P, None, ctx, i + 1, 0, out.summaries)
    out.finish(out.summaries, P)
    return dict(run_dir=out.path, build=b, stages=results, P=P, n_checkpoints=out.n_checkpoints)


# --------------------------------------------------------------------------- replicas (E3c)

def replica_cfg(cfg, r, top, dt=None):
    """The config of replica r: run.seed + r, and r added to every stage seed and every random / white start seed (those
    are what derive from run.seed); [[field]] seeds are the sample's quenched disorder and stay; replicas = 1; the run
    directory <top>/rep_<r>; ``dt`` replaces numerics.dt = "verified" (one dt for the whole ensemble)."""
    c = copy.deepcopy(cfg)
    c["run"]["seed"] += r
    c["run"]["replicas"] = 1
    for st in c["stage"]:
        if "seed" in st:
            st["seed"] += r
        if st["start"]["kind"] in ("random", "white"):
            st["start"]["seed"] += r
    c["output"]["dir"] = os.path.join(top, f"rep_{r}")
    if dt is not None:
        c["numerics"]["dt"] = dt
    return c


def run_replicas(cfg, input_path=None, stop_after_checkpoints=None):
    """R = run.replicas independent complete runs, one after another, replica r in <top>/rep_<r>/ with the seeds of
    replica_cfg; then the pooled spectrum (mean and standard error over the replicas) in the top directory.

    The top directory holds resolved.toml, replicas.json (identity of the config, seeds, the one dt), and at the end
    spectrum.csv, figures/spectrum.png and measures.json.  Re-running the same command resumes: a replica whose
    measures.json exists is skipped, the unfinished one resumes from its own checkpoint.  numerics.dt = "verified" is
    resolved once, on replica 0's build, and stored in replicas.json.  ``stop_after_checkpoints`` counts over all replicas.
    Returns dict(run_dir, replicas=[dict(run_dir, skipped, stages)])."""
    R, ident = cfg["run"]["replicas"], run_output.identity(cfg)
    top, info_path = run_output.existing_dir(cfg), None
    if top is not None:
        info_path = os.path.join(top, "replicas.json")
        if not os.path.isfile(info_path) or json.load(open(info_path))["identity"] != ident:
            raise run_input.ConfigError(f"output.dir: {top} exists and is not a replicated run of this configuration; "
                                        "use a new directory")
        meta = json.load(open(info_path))
    else:
        top = run_output.make_dir(cfg)
        info_path = os.path.join(top, "replicas.json")
        dt = None
        if cfg["numerics"]["dt"] == "verified":
            dt = run_builder.build(replica_cfg(cfg, 0, top))["dt"]
        meta = dict(identity=ident, replicas=R, seeds=[cfg["run"]["seed"] + r for r in range(R)], dt=dt)
        with open(os.path.join(top, "resolved.toml"), "w") as fh:
            fh.write(run_input.dumps(cfg))
        with open(info_path, "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
    left, results, dirs = stop_after_checkpoints, [], []
    for r in range(R):
        rc = replica_cfg(cfg, r, top, meta["dt"])
        d = rc["output"]["dir"]
        dirs.append(d)
        if os.path.isfile(os.path.join(d, "measures.json")):
            results.append(dict(run_dir=d, skipped=True))
            continue
        res = run_one(rc, input_path, left)
        left = None if left is None else left - res["n_checkpoints"]
        results.append(dict(run_dir=d, skipped=False, stages=res["stages"]))
    pooled = run_output.write_spectrum(top, cfg, dirs)
    with open(os.path.join(top, "measures.json"), "w") as fh:
        json.dump(dict(replicas=dict(n=R, seeds=meta["seeds"], dirs=[os.path.basename(d) for d in dirs], dt=meta["dt"]),
                       spectrum=pooled), fh, indent=2, sort_keys=True, default=str)
    return dict(run_dir=top, replicas=results)
