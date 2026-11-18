"""SPEC S36.3 (WP25 Phase 2, stage 2C) -- the unified dt-verification
discipline. Answers ONE question: "given this deck/IC/forcing, what dt is
safe to run production at, and what is the audit trail for that decision?"

Merges THREE generations of ad hoc dt-selection code (ENGINE_GUIDE.md S3.4)
into one engine-agnostic implementation that satisfies all three dt laws
SIMULTANEOUSLY, in a single call:

  (1) SPEC S27.2(b), first generation: `probe_stable_dt` only checks
      BOUNDEDNESS (finite, no blow-up) -- a bounded dt can still converge to
      a numerically-spurious fixed point instead of the true one (the k=0
      mode steps by plain explicit Euler under the semi-implicit split,
      since A(0)=D(0)=0 make M(0)=I identically). The fix: verify the
      probed dt against dt/2 on a REPRESENTATIVE relaxation, halving until
      two successive answers agree to `tol`.
  (2) SPEC S28.6(a), second generation (amends (1), program-wide): the
      dt-vs-dt/2 agreement check must hold TOTAL PHYSICAL TIME FIXED
      (T_total = n_steps*dt, constant across the ladder), never fixed step
      count -- a fixed step count shrinks the integrated time as dt shrinks
      and can signal FALSE convergence near an unrelaxed seed (caught in
      WP18: smaller dt asymptoting back toward the initial condition, not
      the true equilibrium).
  (3) SPEC S32.5, third generation (supplements (1)+(2)): the probe -- both
      the boundedness doubling AND the dt-vs-dt/2 agreement check -- MUST
      include the production forcing terms (E_ext, and any quenched
      disorder already baked into `alpha`). An E_ext=0 probe once missed
      the true binding stability constraint (a random field E_rand) by 64x
      (NaN blow-ups in production, caught, re-probed).

ENGINE ADAPTER: `solver`/`bulk3d`/`film3d` name their stepping primitives
differently (`probe_stable_dt` vs `probe_stable_dt_3d`; an `ops` dict vs
bare `G11`/`kappa_b` scalars; `build_Minv` vs `build_grad_factors` +
`poisson_factor`). `_get_adapter` normalizes all three to one calling
convention (duck-typed on the module's own name) so the algorithm itself
is written ONCE. The H11 lane solvers/grad4.py (and its permanent `hog`
shim) has only the SBDF1 pair `build_Minv`/`step`, each a one-line pass to
solver's own with the same arguments, and `build_operators_hog`'s `ops`
dict is a solver-compatible superset -- but no `probe_stable_dt` and no
SBDF2 pair, so as an `engine` of its own it used to die on AttributeError.
`_resolve_engine` therefore maps engine=grad4 to engine=solver, which is
EXACTLY equivalent (same functions on the same ops; verified bit-for-bit
against engine=solver, which is how gates/g25_verified_dt.py's WP23 deck --
H11-augmented `ops`, `engine=solver` -- has always been run). grad4's 3D
H11 ops (build_operators_hog_3d) step through engine=bulk3d. The film
adapter reads an optional "H11" from its small `ops` dict and passes it to
film3d's probe and both grad-factor builders.

RUN_FN ESCAPE HATCH: the default relaxation (hold `E_ext` constant for
`round(T_total/dt)` steps, compare mean|P|) covers equilibration/uniform-
field-response/pattern-formation checks. A caller whose historical
protocol is not "relax under a constant field" -- e.g.
solvers/energy_protocol.py's unipolar E-ramp (dwell-ladder through a whole
0->e_max->0 path) -- supplies its OWN `run_fn(dt_try) -> value | (value,
norm_ref) | None`; the shared machinery here (probe, blow-up backoff,
fixed-T_total-implied halving ladder, tol comparison, audit trail) is
IDENTICAL either way. This is what lets cases/wp20_triplewell_case.py's
fixed-step-count first-generation check and solvers/energy_protocol.py's
own second-generation dwell-ladder check both become thin parameter-
binding shims around this ONE implementation (SPEC S36.2 R4) while
reproducing their historical numeric decisions exactly (`skip_probe=True`
lets a shim start its halving ladder at its own dt0 instead of re-probing,
matching an implementation that never called `probe_stable_dt` itself;
`eps` matches a shim's own historical division-floor convention).

`info` returns the audit trail (probe dt, each rung's dt/values/relerr,
the adopted verified dt) that callers used to print by hand.

V2-02.1 E2/E12 (adapted from the codex branch's rewrite, Lens 1 E2/E12).
FOUR changes, all on the DEFAULT path -- every custom-`run_fn` caller above
is numerically untouched:
  (a) `scheme` now BINDS the comparison. Before, the default relaxation
      always stepped SBDF1, so `scheme="sbdf2"` reached only the boundedness
      probe and sbdf1/sbdf2 returned the SAME dt with byte-identical relerr
      -- the check certified an integrator nobody ran. The default run_fn
      now goes through the new public `trajectory()`, which steps the
      REQUESTED integrator (every engine now HAS one: bulk3d gained
      build_Minv2_3d/step_sbdf2_3d in SPEC V2-03 A1).
  (b) dt is ALIGNED to T_total/ceil(T_total/dt_probe) before the ladder, so
      `n_steps*dt == T_total` exactly at every rung instead of `round()`
      silently integrating a slightly different total time per rung (law
      (2) held only to one step). `info` gains `n_steps`.
  (c) The blow-up backoff is BOUNDED by `max_halvings` (it was a `while r is
      None` loop that could not terminate), and a blown-up rung now RESETS
      the comparison, so only ADJACENT dts are ever compared -- before, a
      blow-up at dt/2 left the dt rung's values cached and the next finite
      rung was compared against a non-adjacent one.
  (d) The comparison quantity is SELECTABLE: `observable="mean_abs_P"`
      (the default, unchanged -- V2-02.1 keeps default-path callers'
      decisions) or `observable="full_field_L2"`, the branch's own choice,
      which compares the full final field instead of a mean that is blind
      to morphology. Opt-in, and recorded in `info["observable"]`.
`verified_ensemble_dt` (bottom) is additive: the same ladder for a
STOCHASTIC protocol, decided on paired seed replicas instead of one run.
"""
import numpy as np


class _Adapter:
    """Normalizes one engine module's (probe, build_minv, step) primitives
    to a single calling convention. `grid` is bound in at construction
    time (it never changes across a dt ladder, only `dt`/`ops` do)."""
    __slots__ = ("probe", "build_minv", "step")

    def __init__(self, probe, build_minv, step):
        self.probe = probe
        self.build_minv = build_minv
        self.step = step


def _resolve_engine(engine):
    """engine=grad4 (or its `hog` shim) -> solver; anything else unchanged.
    grad4.build_Minv(grid, ops, dt, L) / grad4.step(P, alpha, ops, Minv, dt,
    L, E_ext, elastic_params) ARE solver.build_Minv / solver.step with the
    same arguments, so stepping "through grad4" is stepping through solver;
    solver additionally supplies the probe_stable_dt and build_Minv2/
    step_sbdf2 that grad4 does not re-export (module docstring)."""
    if getattr(engine, "__name__", "").split(".")[-1] in ("grad4", "hog", "gradient4"):
        from fftpf import pf2d_engine as solver
        return solver
    return engine


def _get_adapter(engine, grid, stab=0.0):
    name = getattr(engine, "__name__", "")
    # stab (opt-in, docs/ADAPTIVE_DT.md) reaches the builder and the step
    # together; 0 passes NO keyword, so every existing call is unchanged.
    kw = {"stab": stab} if stab else {}

    if name.endswith((".bulk3d", ".pf3d_engine")) or name == "bulk3d":
        def probe(P0, alpha, ops, L, E_ext, elastic_params, dt0, n_probe, growth, p_cap, max_doublings, scheme):
            # probe_stable_dt_3d gained `scheme` with bulk3d's SBDF2 (V2-03 A1).
            return engine.probe_stable_dt_3d(P0, alpha, ops, grid, L, E_ext=E_ext,
                                              elastic_params=elastic_params, dt0=dt0, n_probe=n_probe,
                                              growth=growth, p_cap=p_cap, max_doublings=max_doublings,
                                              scheme=scheme)

        def build_minv(ops, dt, L):
            return engine.build_Minv_3d(grid, ops, dt, L, **kw)

        def step(P, alpha, ops, Minv, dt, L, E_ext, elastic_params):
            return engine.step_3d(P, alpha, ops, Minv, dt, L, E_ext, elastic_params=elastic_params,
                                  **kw)

        return _Adapter(probe, build_minv, step)

    if name.endswith((".film3d", ".film_engine")) or name == "film3d":
        # film3d has no single `ops` dict (SPEC S13.0's per-mode-banded
        # design uses bare G11/kappa_b scalars everywhere instead) -- for
        # THIS engine, verified_dt's `ops` argument is instead a small dict
        # {"G11": ..., "kappa_b": ...} (documented in verified_dt's own
        # docstring). `Minv` here is the (grad_factors, poisson_factors)
        # pair film3d.step actually consumes, rebuilt fresh at every dt
        # tried (poisson_factors is dt-independent but cheap to rebuild;
        # correctness over micro-optimization). An optional "H11" in that
        # dict (SPEC V2-03 B5, implicit pentadiagonal) is passed to the probe
        # and to BOTH grad-factor builders (trajectory's SBDF2 one too) --
        # it lives entirely in those factors, so leaving it out stepped,
        # probed and certified an H11 film as H11=0. Absent -> None, the
        # film's own default: byte-identical to before.
        def probe(P0, alpha, ops, L, E_ext, elastic_params, dt0, n_probe, growth, p_cap, max_doublings, scheme):
            return engine.probe_stable_dt(P0, alpha, grid, ops["G11"], ops["kappa_b"], L, E_ext=E_ext,
                                           dt0=dt0, n_probe=n_probe, growth=growth, p_cap=p_cap,
                                           max_doublings=max_doublings, scheme=scheme, el=elastic_params,
                                           H11=ops.get("H11"))

        def build_minv(ops, dt, L):
            poisson_factors = engine.poisson_factor(grid, ops["kappa_b"])
            grad_factors = engine.build_grad_factors(grid, ops["G11"], dt, L, H11=ops.get("H11"),
                                                     **kw)
            return (grad_factors, poisson_factors)

        def step(P, alpha, ops, Minv, dt, L, E_ext, elastic_params):
            grad_factors, poisson_factors = Minv
            return engine.step(P, alpha, grid, grad_factors, poisson_factors, dt, L, E_ext, el=elastic_params,
                               **kw)

        return _Adapter(probe, build_minv, step)

    # Default: "solver-shaped" -- solver.py itself, which is also what
    # engine=grad4 resolves to (_resolve_engine: grad4.py's
    # build_operators_hog augments solver.build_operators's A11/A22/A33
    # with H11*(kx^4+ky^4) but keeps every key solver.step/build_Minv
    # actually read; kittel_afe.py/octa_tilt.py's P-sector is likewise
    # solver-compatible). A module reaching this branch must itself
    # provide probe_stable_dt and build_Minv/step (+ build_Minv2/
    # step_sbdf2 for scheme="sbdf2").
    def probe(P0, alpha, ops, L, E_ext, elastic_params, dt0, n_probe, growth, p_cap, max_doublings, scheme):
        return engine.probe_stable_dt(P0, alpha, ops, grid, L, E_ext=E_ext, elastic_params=elastic_params,
                                       dt0=dt0, n_probe=n_probe, growth=growth, p_cap=p_cap,
                                       max_doublings=max_doublings, scheme=scheme)

    def build_minv(ops, dt, L):
        return engine.build_Minv(grid, ops, dt, L, **kw)

    def step(P, alpha, ops, Minv, dt, L, E_ext, elastic_params):
        return engine.step(P, alpha, ops, Minv, dt, L, E_ext, elastic_params=elastic_params, **kw)

    return _Adapter(probe, build_minv, step)


def _as_pairs(r):
    """Normalize one run_fn result to a list of (value, norm_ref) pairs.
    r may be:
      - a plain float: one pair, compared against/normalized by itself.
      - a (value, norm_ref) 2-tuple: one pair, normalized by norm_ref
        instead -- solvers/energy_protocol.py's historical convention
        (normalize by the trajectory's own max, not its final value, to
        stay well-defined near a zero crossing).
      - a list of (value, norm_ref) pairs: one comparison quantity PER
        FIELD of a coupled multi-field system (e.g. a P-theta run_fn
        returning [(meanP, meanP), (meanTheta, meanTheta)]) -- see
        `_relerr` below for how these combine."""
    if isinstance(r, list):
        return r
    if isinstance(r, tuple):
        return [r]
    return [(r, r)]


def _l2(v):
    """L2 norm of one observable value. For a SCALAR this is exactly
    `abs(v)` -- computed as such, not as sqrt(v*v) -- so `_relerr` below
    reduces bit-for-bit to the scalar criterion the three historical shims
    were decided with, with no floating-point argument required."""
    a = np.asarray(v)
    return abs(float(a)) if a.ndim == 0 else float(np.linalg.norm(a.ravel()))


def _relerr(pairs, pairs_half, eps):
    """The dt-vs-dt/2 agreement criterion: the MAX relative error across
    every parallel (value, norm_ref) pair -- for a single pair this is
    just that pair's own relative error; for several (a coupled system's
    several fields) this is the "worst of" rule a hand-rolled
    max(relerr_a, relerr_b, ...) loop would use, so a multi-field system
    converges only once EVERY field has agreed to `tol`, not just one.

    A pair's value may be an ARRAY (observable="full_field_L2" compares
    whole fields): the error is then ||v - v_half|| / max(||ref_half||, eps),
    which for scalars IS `abs(v - v_half) / max(abs(ref_half), eps)`, the
    pre-V2-02 formula. Non-finite anywhere -> inf (never an accidental pass);
    a changed observable count/shape between two rungs is a caller bug and
    raises rather than comparing incomparable things."""
    if not pairs or len(pairs) != len(pairs_half):
        raise ValueError("timestep: observable count changed between dt and dt/2")
    errors = []
    for (v, _ref), (v_half, ref_half) in zip(pairs, pairs_half):
        v, v_half, ref_half = np.asarray(v), np.asarray(v_half), np.asarray(ref_half)
        if v.shape != v_half.shape:
            raise ValueError("timestep: observable shape changed between dt and dt/2")
        if not all(np.all(np.isfinite(x)) for x in (v, v_half, ref_half)):
            return float("inf")
        errors.append(_l2(v - v_half) / max(_l2(ref_half), eps))
    return max(errors)


def _summary(pairs):
    """One rung's values as they go into `info["trace"]`. Scalars go in
    verbatim (the audit trail prints exactly what it always printed); an
    array observable is summarized (shape/l2/mean) so the trace stays small
    and JSON-able even when the comparison used every field voxel."""
    def one(v):
        a = np.asarray(v)
        return float(a) if a.ndim == 0 else dict(shape=list(a.shape),
                                                 l2=float(np.linalg.norm(a.ravel())),
                                                 mean=float(a.mean()))
    return [(one(v), one(ref)) for v, ref in pairs]


def trajectory(engine, P0, alpha, ops, grid, *, dt, n_steps, L=1.0,
               E_ext=(0., 0., 0.), elastic_params=None, scheme="sbdf1",
               P_previous=None, stab=0.0):
    """Fixed-step deterministic trajectory; returns (P, previous P).

    The one place `verified_dt`'s default path integrates, and a public
    primitive in its own right: `n_steps` steps of `engine`'s own stepper at
    a FIXED dt under a FIXED E_ext, no adaptation and no stopping rule.

    SBDF2 bootstraps with ONE SBDF1 step, and only when `P_previous` is
    None -- pass the stored (P, P_previous) pair plus the same dt to resume
    the identical two-step recurrence. All three engines have an SBDF2 pair
    (bulk3d's build_Minv2_3d/step_sbdf2_3d landed in SPEC V2-03 A1, which
    lifted this function's bulk3d-is-SBDF1-only guard). Stops early on a
    non-finite state (the caller checks). Noisy/coupled protocols keep their
    own production runner. engine=grad4 steps through solver
    (_resolve_engine); a film `ops` "H11" enters both grad-factor builds.

    stab=S > 0 (opt-in): the engines' linear stabilization, passed to the
    builders AND the steps together (docs/ADAPTIVE_DT.md); 0 is the
    original call, bit for bit."""
    engine = _resolve_engine(engine)
    bulk = hasattr(engine, "step_3d")
    film = getattr(engine, "__name__", "").split(".")[-1] in ("film3d", "film_engine")
    if scheme not in ("sbdf1", "sbdf2"):
        raise ValueError(f"timestep.trajectory: unsupported scheme {scheme!r} "
                         f"for {getattr(engine, '__name__', engine)}")
    if not np.isfinite(dt) or dt <= 0 or n_steps < 0 or int(n_steps) != n_steps:
        raise ValueError("timestep.trajectory: dt must be positive and n_steps "
                         "a nonnegative integer")
    adapter = _get_adapter(engine, grid, stab)
    kw = {"stab": stab} if stab else {}
    m1 = adapter.build_minv(ops, dt, L)
    if scheme == "sbdf2":
        m2 = (engine.build_grad_factors_sbdf2(grid, ops["G11"], dt, L, H11=ops.get("H11"), **kw)
              if film
              else engine.build_Minv2_3d(grid, ops, dt, L, **kw) if bulk
              else engine.build_Minv2(grid, ops, dt, L, **kw))
    P = np.asarray(P0).copy()
    previous = None if P_previous is None else np.asarray(P_previous).copy()
    for _ in range(int(n_steps)):
        if scheme == "sbdf1" or previous is None:
            new = adapter.step(P, alpha, ops, m1, dt, L, E_ext, elastic_params)
        elif film:
            new = engine.step_sbdf2(P, previous, alpha, grid, m2, m1[1], dt, L,
                                    E_ext, el=elastic_params, **kw)
        elif bulk:
            new = engine.step_sbdf2_3d(P, previous, alpha, ops, m2, dt, L, E_ext,
                                       elastic_params=elastic_params, **kw)
        else:
            new = engine.step_sbdf2(P, previous, alpha, ops, m2, dt, L, E_ext,
                                    elastic_params=elastic_params, **kw)
        previous, P = P, new
        if not np.all(np.isfinite(P)):
            break
    return P, previous


def verified_dt(engine, P0, alpha, ops, grid, *, L=1.0, E_ext=None, T_total,
                 elastic_params=None, dt0=0.02, n_probe=200, growth=2.0,
                 p_cap=None, max_doublings=30, scheme="sbdf1",
                 tol=0.01, max_halvings=10, eps=1e-12,
                 run_fn=None, skip_probe=False, observable=None, stab=0.0):
    """Verify a stable, converged dt for `engine` on this deck/IC/forcing.

    engine: the stepping module -- `solver`, `bulk3d`, or `film3d`.
        `grad4` (or `hog`) is accepted and runs as `solver` -- its
        build_Minv/step ARE solver's on the same H11-augmented ops, so the
        result is identical to passing engine=solver (_resolve_engine).
        Another module must provide the solver-style probe_stable_dt,
        build_Minv/step and, for scheme="sbdf2", build_Minv2/step_sbdf2.
    P0: seed state for the boundedness probe (and the default run_fn's own
        relaxation, if no custom `run_fn` is supplied).
    alpha: Landau coefficients (engine-native a1..a123; may itself be a
        per-point array for a quenched/disordered deck).
    ops: engine's own k-space operator dict (solver.build_operators/
        bulk3d.build_operators_3d's return). For engine=film3d, which has
        no single `ops` dict, pass {"G11": ..., "kappa_b": ...} instead,
        plus "H11": ... for an H11 film (forwarded to film3d's probe and
        grad-factor builders; absent means H11=None).
    L, E_ext: mobility and external field, threaded into BOTH the
        boundedness probe and the relaxation check (SPEC S32.5 -- E_ext
        MUST be the production forcing, never a 0-field placeholder, for
        any deck where it can matter). E_ext=None means (0.,0.,0.).
    T_total: fixed total physical time integrated at EVERY dt tried
        (SPEC S28.6(a) -- never a fixed step count). On the default path the
        starting dt is ALIGNED to T_total/ceil(T_total/dt_probe), so every
        rung lands EXACTLY on T_total (V2-02.1 E2(b)); n_steps =
        round(T_total/dt). Informational only (not read by the algorithm
        itself) when a custom `run_fn` supplies its own notion of "how
        long", e.g. energy_protocol's dwell-ladder -- a custom run_fn's dt
        is NEVER aligned, that is its own business.
    run_fn: optional custom trajectory runner, `run_fn(dt_try) -> value |
        (value, norm_ref) | [(value, norm_ref), ...] | None` (None means
        the trajectory blew up / went non-finite -- triggers dt-halving
        backoff before the comparison ladder even starts; a LIST of pairs
        is one comparison quantity PER FIELD of a coupled multi-field
        system, e.g. a P-theta stepper's own [(meanP,meanP),
        (meanTheta,meanTheta)] -- the agreement criterion becomes the MAX
        relative error across every pair, see `_relerr`). Default (None):
        relax P0 under E_ext held constant for round(T_total/dt_try) steps
        through `trajectory` (i.e. with the REQUESTED `scheme`, V2-02.1
        E2(a)), returns the `observable` below.
    observable: what the default path compares, "mean_abs_P" (None, the
        default: mean(|P_final|), the pre-V2-02 quantity -- V2-02.1 E2(f)
        keeps it so no existing default-path caller is silently re-decided)
        or "full_field_L2" (opt-in: the whole final field, compared as
        ||P_dt - P_dt/2|| / ||P_dt/2||, which unlike a mean is not blind to
        morphology -- a domain pattern that moves at half the dt fails it).
        Mutually exclusive with `run_fn` (a custom runner owns its own
        observable). Recorded in info["observable"].
    skip_probe: if True, skip the `engine.probe_stable_dt*` boundedness
        doubling entirely and start the halving ladder directly at `dt0`
        -- matches a legacy implementation (energy_protocol.verified_dt)
        that never called probe_stable_dt itself, only ran its own
        dt0-anchored halving loop. False (default): the canonical,
        fully law-(1)-(2)-(3)-compliant path used by all NEW code.
    dt0, n_probe, growth, p_cap, max_doublings, scheme: forwarded to the
        boundedness probe (engine.probe_stable_dt*'s own parameters).
    tol: relative-agreement tolerance for the dt-vs-dt/2 comparison. The
        0.01 default is a declared dial, not a sourced number: it is the
        value the three historical implementations G29e replays were run at
        (SPEC §36.4 G29e). Callers needing tighter convergence pass their own.
    max_halvings: give up (RuntimeError) after this many halvings without
        reaching `tol`.
    eps: floor on the comparison denominator (avoids division by ~0 near a
        zero crossing) -- 1e-12 matches cases/wp20_triplewell_case.py's
        own historical convention; solvers/energy_protocol.py's shim
        passes 1e-9 to match ITS own historical convention exactly.

    Returns (dt, info). info: probe_dt, T_total, tol, eps, trace (one
    dict per dt tried: dt, dt_half, values, values_half (each a list of
    (value, norm_ref) pairs, see `run_fn` and `_summary`), agree_relerr, or
    a blow-up marker), verified_dt (== the returned dt), last_checked_dt,
    agree_relerr (the final one), n_halvings, plus (V2-02.1 E2) scheme,
    scope ("deterministic" / "custom_protocol"), observable and n_steps
    (None for a custom run_fn, which owns its own step count).

    stab: opt-in linear stabilization S (docs/ADAPTIVE_DT.md), forwarded to
    the default path's `trajectory` so the ladder certifies the STABILIZED
    integrator (S changes the time accuracy, so it must be audited with the
    same S it runs with). The boundedness probe (probe_stable_dt*) has no S
    and certifies only the unstabilized dt -- with S, pass skip_probe=True
    and your dt0. A custom run_fn owns its own stepping and S is refused
    there. Recorded in info["stab"] when nonzero; 0 is the original call.
    """
    custom = run_fn is not None
    if custom and observable is not None:
        raise ValueError("timestep.verified_dt: pass run_fn or observable, not both")
    obs = observable if observable is not None else "mean_abs_P"
    if obs not in ("mean_abs_P", "full_field_L2"):
        raise ValueError(f"timestep.verified_dt: unknown observable {observable!r}")
    if scheme not in ("sbdf1", "sbdf2"):
        raise ValueError(f"timestep.verified_dt: unsupported scheme {scheme!r} "
                         f"for {getattr(engine, '__name__', engine)}")
    if E_ext is None:
        E_ext = (0.0, 0.0, 0.0)
    if stab and custom:
        raise ValueError("timestep.verified_dt: stab applies to the default trajectory; "
                         "a custom run_fn owns its own stepping")
    engine = _resolve_engine(engine)
    adapter = _get_adapter(engine, grid)

    if skip_probe:
        dt_probe = dt0
    else:
        dt_probe = adapter.probe(P0, alpha, ops, L, E_ext, elastic_params,
                                  dt0, n_probe, growth, p_cap, max_doublings, scheme)

    dt = float(dt_probe)
    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError("timestep.verified_dt: the ladder cannot start at "
                           f"dt={dt_probe!r} ({'dt0' if skip_probe else 'the boundedness probe'})")

    if not custom:
        # E2(b): land exactly on T_total at every rung (dt, dt/2, dt/4, ...
        # all divide T_total once the first one does).
        dt = T_total / max(1, int(np.ceil(T_total / dt)))

        def run_fn(dt_try):
            P, _prev = trajectory(engine, P0, alpha, ops, grid, dt=dt_try,
                                   n_steps=max(1, int(round(T_total / dt_try))),
                                   L=L, E_ext=E_ext, elastic_params=elastic_params,
                                   scheme=scheme, **({"stab": stab} if stab else {}))
            if not np.all(np.isfinite(P)):
                return None
            return float(np.mean(np.abs(P))) if obs == "mean_abs_P" else P

    trace = [dict(stage="probe", dt=float(dt_probe), aligned_dt=dt, skip_probe=skip_probe)]

    # ONE run per rung, comparing each finite rung against the PREVIOUS one
    # (E12): a blown-up rung drops the cached values, so a comparison is
    # only ever between ADJACENT dts, and the backoff cannot outlive
    # max_halvings. Rung i is dt_aligned/2**i; the ladder makes at most
    # max_halvings + 1 run_fn calls in total.
    pairs = None
    n_blown = 0
    for i in range(max_halvings + 1):
        r = run_fn(dt)
        current = None if r is None else _as_pairs(r)
        finite = current is not None and all(
            np.all(np.isfinite(v)) and np.all(np.isfinite(ref)) for v, ref in current)
        if not finite:
            trace.append(dict(dt=dt, status="blew_up"))
            pairs = None
            n_blown += 1
        elif pairs is not None:
            rel = _relerr(pairs, current, eps)
            trace.append(dict(dt=2.0 * dt, dt_half=dt, values=_summary(pairs),
                              values_half=_summary(current), agree_relerr=rel))
            if rel <= tol:
                info = dict(probe_dt=float(dt_probe), T_total=T_total, tol=tol, eps=eps,
                            trace=trace, verified_dt=dt, last_checked_dt=2.0 * dt,
                            agree_relerr=rel, n_halvings=i, scheme=scheme,
                            scope="custom_protocol" if custom else "deterministic",
                            observable="custom" if custom else obs,
                            n_steps=None if custom else max(1, int(round(T_total / dt))))
                if stab:
                    info["stab"] = stab
                return dt, info
            pairs = current
        else:
            pairs = current
        dt /= 2.0

    raise RuntimeError(f"timestep.verified_dt: no dt-vs-dt/2 agreement within "
                       f"{max_halvings} halvings (last dt tried {2.0 * dt}, "
                       f"{n_blown} of {max_halvings + 1} rungs blew up)")


def verified_ensemble_dt(run_fn, *, dt0, seeds, scales, T_total,
                         tol=0.05, max_halvings=8, confidence_z=3.0):
    """The same ladder for a STOCHASTIC protocol (V2-02.1 E2, additive).

    `verified_dt` above decides on ONE trajectory, which certifies nothing
    about an OU/Langevin run: a single noisy replica differs at dt and dt/2
    by sampling error alone. Here `run_fn(dt, seed) -> observable vector`
    is run over independent replica `seeds` at the SAME T_total and forcing,
    and two adjacent rungs agree only when the PAIRED seed difference is
    small against declared physical `scales`: |mean(delta)| + z*SEM(delta)
    <= tol*scales for EVERY observable. Because the bound is an upper
    confidence limit, a WIDE, under-resolved ensemble cannot pass merely by
    overlapping -- it fails until more replicas or a smaller dt shrink it.
    `scales` are explicit, so a near-zero mean is still a valid comparison.
    The normal/CLT estimate is approximate: check autocorrelation and
    replica convergence yourself. No seconds/Hz interpretation is implied.

    Returns (dt, info) like `verified_dt`; RuntimeError if unresolved."""
    seeds = tuple(seeds)
    scales = np.atleast_1d(np.asarray(scales, dtype=float))
    if len(seeds) < 8 or len(set(seeds)) != len(seeds):
        raise ValueError("timestep.verified_ensemble_dt: use at least 8 distinct "
                         "independent replica seeds")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("timestep.verified_ensemble_dt: observable scales must be "
                         "finite and positive")
    if not all(np.isfinite(x) and x > 0 for x in (dt0, T_total, tol, confidence_z)):
        raise ValueError("timestep.verified_ensemble_dt: dt0/T_total/tol/confidence_z "
                         "must be positive")
    if max_halvings < 1:
        raise ValueError("timestep.verified_ensemble_dt: max_halvings must be >= 1")
    dt = T_total / max(1, int(np.ceil(T_total / dt0)))
    previous = None
    trace = []
    for rung in range(max_halvings + 1):
        rows = [run_fn(dt, seed) for seed in seeds]
        values = None if any(row is None for row in rows) else np.asarray(rows, dtype=float)
        if values is not None and values.ndim == 1:
            values = values[:, None]
        if values is None or not np.all(np.isfinite(values)):
            trace.append(dict(dt=dt, status="blew_up"))
            previous = None
        else:
            if values.shape != (len(seeds), scales.size):
                raise ValueError("timestep.verified_ensemble_dt: run_fn observable "
                                 "vector must match scales")
            if previous is not None:
                delta = previous - values
                difference = delta.mean(axis=0)
                sem = delta.std(axis=0, ddof=1) / np.sqrt(len(seeds))
                upper = (abs(difference) + confidence_z * sem) / scales
                trace.append(dict(dt=2.0 * dt, dt_half=dt, mean_delta=difference.tolist(),
                                  sem_delta=sem.tolist(), error_upper=upper.tolist()))
                if np.all(upper <= tol):
                    return dt, dict(verified_dt=dt, T_total=T_total,
                                    scope="stochastic_ensemble", seeds=list(seeds),
                                    scales=scales.tolist(), tol=tol,
                                    confidence_z=confidence_z, trace=trace,
                                    n_halvings=rung, error_upper=upper.tolist())
            previous = values
        dt /= 2.0
    raise RuntimeError("timestep.verified_ensemble_dt: sampling error did not resolve "
                       f"tol={tol} within {max_halvings} halvings -- reduce dt0 and/or "
                       "use more replicas")
