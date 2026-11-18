"""Adaptive time stepping for DETERMINISTIC gradient-flow runs, plus the
helpers behind the engines' opt-in linear stabilization (`stab=`).
User guide: docs/ADAPTIVE_DT.md. Case-level entry point: casekit.anneal_adaptive.

LINEAR STABILIZATION (stab=S >= 0). The three engines' builders and steps
(solver.build_Minv/build_Minv2/step/step_sbdf2, their bulk3d `_3d` twins,
film3d.build_grad_factors[_sbdf2]/step/step_sbdf2) accept `stab=S`:
    SBDF1: (P+ - P)/dt          = -L(A+S)P+ - L[N(P)  - S*P ]
    SBDF2: (3P+ - 4P + P-)/(2dt) = -L(A+S)P+ - L[N(P*) - S*P*],  P* = 2P - P-
S*P is added implicitly and subtracted explicitly (at the SAME point N is
evaluated), so the two cancel at a fixed point: fixed points are exactly the
S=0 ones. S >~ the Lipschitz bound of the explicit force (landau_lipschitz)
lets SBDF1 take dt beyond the explicit limit dt*L*ell < 2 (and keeps SBDF2
bounded there). It CHANGES the time accuracy -- an extra O(dt*L*S) term in
SBDF1, O(dt^2*L*S*P'') in SBDF2, which accumulates over the slow modes -- so
dynamics must be re-audited at dt/2. The builder records S on what it returns (an ndarray
view, StabilizedInverse, or a "stab" key in the film factors dict) and the
step refuses an operator built with a different S, in both directions.
S=0 never reaches this module and is the original arithmetic bit for bit.

ADAPTIVE dt (integrate). dt lives on a geometric ladder dt_k = dt0*2**k,
k_min <= k <= k_max (and dt <= dt_max); implicit operators are built lazily
and cached per (dt, S). Stepping is SBDF2. Constant-dt steps call the
engine's own step_sbdf2; a step right after a dt change uses the
variable-step SBDF2 (VSBDF2, Wang & Ruuth 2008) coefficients with
omega = dt_n/dt_(n-1):
    [(1+2w)/(1+w) P+ - (1+w) P + w^2/(1+w) P-] / dt = -L(A+S)P+ - L[N(P*) - S P*],
    P* = (1+w)P - wP-,
whose implicit operator (1+2w)/(1+w)*I + dt*L*(A+S) is the engine's SBDF1
operator at dt_eff = dt*(1+w)/(1+2w), scaled; w = 1 is exactly SBDF2. Only
the very first step (no history) is SBDF1.

ERROR ESTIMATE (no extra force evaluation). P* is the scheme's own explicit
linear-extrapolation predictor; D = P+ - P* ~ dt^2 P''. The leading SBDF2
local error is the IMEX splitting term
    delta ~ dt_eff*L*[I + dt_eff*L*(A+S)]^-1 (dN/dP + S) D,
so the estimate is D filtered through the implicit operator (Shampine's
filter: stiff, implicitly damped components -- e.g. the high-k noise of a
random start -- are accurate even though D is large there):
    err = (1 + dt_eff*L*S) * max|[I + dt_eff*L*(A+S)]^-1 D|
(= max|D| on the slow, low-k modes; two FFTs, no force evaluation; the
unknown dN/dP factor is absorbed into err_tol). Units: polarization (C/m^2
for the decks here). A rejected step right after the SBDF1 bootstrap also
discards the bootstrap (its first-order error is part of what was rejected).

CONTROLLER. Reject the step and halve dt when err > err_tol or the state is
non-finite; double dt after `grow_after` consecutive accepted steps with
err < err_tol/4; never exceed dt_max/k_max. Gradient flow: a total-energy
rise beyond round-off (energy_rtol * sum of |energy terms|) is rejected too,
per step (energy_check="step") or per sampling window (energy_check=
"sample", the default: the energy is evaluated at the sample points only and
a rise rolls the whole window back -- an energy evaluation costs 3-8 steps on
these engines). An energy or non-finite rejection also caps the level below
the failing one until `ceiling_hold` clean samples have passed.

STOPPING: the same criteria casekit.anneal uses ("legacy" block_stationary
on the sampled energies, "block_v2" block means + force residual), the
film-case "plateau" rule (force residual, |dP|/dt and |df|/dt over
`n_windows` consecutive samples), or None; plus n_max accepted steps and
t_end (the last step lands exactly on t_end).

Deterministic only: there is no noise term anywhere in this module and
casekit.anneal_adaptive refuses kT.
"""
import itertools
import time
from collections import OrderedDict

import numpy as np


# ==========================================================================
# Linear stabilization helpers (imported lazily by the engines' stab branch)
# ==========================================================================
class StabilizedInverse(np.ndarray):
    """An implicit-operator inverse built with stab=S>0 (a plain ndarray view
    that remembers S). The S=0 builders never return one."""
    stab = 0.0

    def __array_finalize__(self, obj):
        self.stab = getattr(obj, "stab", 0.0)


def check_stab(stab):
    """S as a float; S must be finite and >= 0."""
    value = float(stab)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"stab must be finite and >= 0, got {stab!r}")
    return value


def tag_operator(op, stab):
    """Record the S an operator was built with (dict key or ndarray view)."""
    if isinstance(op, dict):
        op["stab"] = stab
        return op
    out = np.asarray(op).view(StabilizedInverse)
    out.stab = stab
    return out


def operator_stab(op):
    """The S an operator was built with (0.0 for every untagged operator)."""
    if isinstance(op, dict):
        return float(op.get("stab", 0.0))
    return float(getattr(op, "stab", 0.0))


def stabilized_operator(op, stab, where):
    """Refuse an operator built with a different S; return it untagged."""
    stab = check_stab(stab)
    built = operator_stab(op)
    if built != stab:
        raise ValueError(f"{where}: the implicit operator was built with stab={built!r} but "
                         f"the step was called with stab={stab!r}; the builder and the step "
                         "must receive the same S")
    return op if isinstance(op, dict) else np.asarray(op)


def landau_lipschitz(alpha, p_max, n=33):
    """Estimate ell = sup ||d^2 f_bulk/dP^2|| (spectral norm of the 3x3 Landau
    Hessian) over the cube |P_i| <= p_max: the Lipschitz bound of the
    explicit Landau force, the scale for `stab` (S >~ ell/2 makes stabilized
    SBDF1 energy-stable for that force; S ~ ell is the conservative choice).
    Central differences of solver.bulk_force on an n^3 grid. Array-valued
    coefficients (quenched alpha(r)) are taken at their extremes. Elastic and
    electrostatic explicit terms are NOT included: add their bound yourself
    when they are stiff."""
    from fftpf import pf2d_engine as solver
    axis = np.linspace(-p_max, p_max, n)
    P = np.stack(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1)
    h = 1e-6 * max(p_max, 1e-3)
    keys = [k for k, v in alpha.items() if np.ndim(v) > 0]
    best = 0.0
    for combo in itertools.product(*[(np.min(alpha[k]), np.max(alpha[k])) for k in keys]):
        a = dict(alpha)
        a.update(zip(keys, (float(c) for c in combo)))
        H = np.empty((P.shape[1], 3, 3))
        for j in range(3):
            e = np.zeros((3, 1))
            e[j] = h
            H[:, :, j] = ((solver.bulk_force(P + e, a) - solver.bulk_force(P - e, a)) / (2 * h)).T
        lam = np.linalg.eigvalsh(0.5 * (H + np.transpose(H, (0, 2, 1))))
        best = max(best, float(np.max(np.abs(lam))))
    return best


# ==========================================================================
# Engine adapter: one calling convention for solver / bulk3d / film3d
# ==========================================================================
_KIND_OF_MODULE = {"pf2d_engine": "solver", "pf3d_engine": "bulk3d", "film_engine": "film3d"}


def engine_kind(engine):
    """'solver' | 'bulk3d' | 'film3d' (by module name), else ValueError."""
    name = getattr(engine, "__name__", "").split(".")[-1]
    name = _KIND_OF_MODULE.get(name, name)  # fftpf.* names -> the historical kind labels
    if name not in ("solver", "bulk3d", "film3d"):
        raise ValueError(f"adaptive: unsupported engine {name or engine!r} "
                         "(solver, bulk3d or film3d; deterministic stepping only)")
    return name


class Engine:
    """Binds (alpha, ops, grid, L, E_ext, elastic_params, stab) to one engine
    and exposes build/step/energy/force with a single signature. For film3d,
    `ops` is {"G11" or "G": ..., "kappa_b": ..., "H11": optional} (the same
    convention timestep.verified_dt uses for that engine)."""

    def __init__(self, engine, alpha, ops, grid, *, L=1.0, E_ext=(0., 0., 0.),
                 elastic_params=None, stab=0.0):
        self.kind = engine_kind(engine)
        self.mod = engine
        self.alpha, self.ops, self.grid, self.L = alpha, ops, grid, L
        self.E_ext, self.el = E_ext, elastic_params
        self.stab = check_stab(stab)
        self.kw = {"stab": self.stab} if self.stab else {}
        if self.kind == "film3d":
            self.G = ops["G"] if "G" in ops else ops["G11"]
            self.kappa_b = ops["kappa_b"]
            self.H11 = ops.get("H11")
            self.pf = engine.poisson_factor(grid, self.kappa_b)

    # -- operators ---------------------------------------------------------
    def build1(self, dt):
        m, kw = self.mod, self.kw
        if self.kind == "solver":
            return m.build_Minv(self.grid, self.ops, dt, self.L, **kw)
        if self.kind == "bulk3d":
            return m.build_Minv_3d(self.grid, self.ops, dt, self.L, **kw)
        return m.build_grad_factors(self.grid, self.G, dt, self.L, H11=self.H11, **kw)

    def build2(self, dt):
        m, kw = self.mod, self.kw
        if self.kind == "solver":
            return m.build_Minv2(self.grid, self.ops, dt, self.L, **kw)
        if self.kind == "bulk3d":
            return m.build_Minv2_3d(self.grid, self.ops, dt, self.L, **kw)
        return m.build_grad_factors_sbdf2(self.grid, self.G, dt, self.L, H11=self.H11, **kw)

    # -- steps -------------------------------------------------------------
    def step1(self, P, op, dt):
        m, kw = self.mod, self.kw
        if self.kind == "solver":
            return m.step(P, self.alpha, self.ops, op, dt, self.L, self.E_ext,
                          elastic_params=self.el, **kw)
        if self.kind == "bulk3d":
            return m.step_3d(P, self.alpha, self.ops, op, dt, self.L, self.E_ext,
                             elastic_params=self.el, **kw)
        return m.step(P, self.alpha, self.grid, op, self.pf, dt, self.L, self.E_ext,
                      el=self.el, **kw)

    def step2(self, P, P_prev, op2, dt):
        m, kw = self.mod, self.kw
        if self.kind == "solver":
            return m.step_sbdf2(P, P_prev, self.alpha, self.ops, op2, dt, self.L, self.E_ext,
                                elastic_params=self.el, **kw)
        if self.kind == "bulk3d":
            return m.step_sbdf2_3d(P, P_prev, self.alpha, self.ops, op2, dt, self.L,
                                   self.E_ext, elastic_params=self.el, **kw)
        return m.step_sbdf2(P, P_prev, self.alpha, self.grid, op2, self.pf, dt, self.L,
                            self.E_ext, el=self.el, **kw)

    def explicit_N(self, P):
        m = self.mod
        if self.kind == "solver":
            return m._explicit_N(P, self.alpha, self.ops, self.E_ext, self.el)
        if self.kind == "bulk3d":
            return m._explicit_N_3d(P, self.alpha, self.ops, self.E_ext, self.el)
        return m.explicit_N(P, self.alpha, self.grid, self.pf, self.E_ext, el=self.el)

    # -- k-space plumbing (the engines' own FFT wrappers) --------------------
    def _fwd(self, X):
        m = self.mod
        return m._rfftn(X) if self.kind == "bulk3d" else m._rfft2(X)

    def _inv(self, Xk):
        m, g = self.mod, self.grid
        if self.kind == "bulk3d":
            return m._irfftn(Xk, (g["Nz"], g["Ny"], g["Nx"]))
        return m._irfft2(Xk, (g["Ny"], g["Nx"]))

    def _apply(self, op, Xk):
        """op (an SBDF1/SBDF2 implicit inverse) applied to a k-space field."""
        if self.kind == "film3d":
            return self.mod._solve_grad_factored(op, Xk)
        M = np.asarray(op)
        out = np.empty_like(Xk)
        for i in range(3):
            out[i] = M[..., i, 0] * Xk[0] + M[..., i, 1] * Xk[1] + M[..., i, 2] * Xk[2]
        return out

    def filtered(self, op, X):
        """op applied to a real field: the implicit filter of the error
        estimate (two FFTs, no force evaluation)."""
        return self._inv(self._apply(op, self._fwd(X)))

    def vstep2(self, P, P_prev, op_eff, dt, omega):
        """One variable-step SBDF2 step (module docstring); `op_eff` is
        build1(dt*(1+w)/(1+2w)) with this Engine's S. Returns (P+, P*)."""
        w = omega
        c0 = (1.0 + 2.0 * w) / (1.0 + w)
        P_star = (1.0 + w) * P - w * P_prev
        X = (1.0 + w) * P - (w * w / (1.0 + w)) * P_prev
        if self.stab:
            X = X + (dt * self.L * self.stab) * P_star
        N = self.explicit_N(P_star)
        RHS = (self._fwd(X) - dt * self.L * self._fwd(N)) / c0
        return self._inv(self._apply(op_eff, RHS)), P_star

    # -- diagnostics -------------------------------------------------------
    def energy(self, P):
        """(densities, totals) -- the engine's own total_energy*."""
        m = self.mod
        if self.kind == "solver":
            return m.total_energy(P, self.alpha, self.ops, self.E_ext, self.grid,
                                  elastic_params=self.el)
        if self.kind == "bulk3d":
            return m.total_energy_3d(P, self.alpha, self.ops, self.E_ext, self.grid,
                                     elastic_params=self.el)
        return m.total_energy_3d(P, self.alpha, self.grid, self.kappa_b, self.G,
                                 poisson_factors=self.pf, E_ext=self.E_ext, el=self.el,
                                 H11=self.H11)

    def force(self, P):
        """dF/dP in the engine's rate convention (dP/dt = -L * force)."""
        m = self.mod
        if self.kind == "solver":
            return m.total_force(P, self.alpha, self.ops, self.E_ext, elastic_params=self.el)[-1]
        if self.kind == "bulk3d":
            return m.total_force_3d(P, self.alpha, self.ops, self.E_ext,
                                    elastic_params=self.el)[-1]
        return (m.grad_force_3d(P, self.grid, self.G, self.H11)
                + m.explicit_N(P, self.alpha, self.grid, self.pf, self.E_ext, el=self.el))


# ==========================================================================
# The driver
# ==========================================================================
CRITERIA = ("legacy", "block_v2", "plateau", None)
ENERGY_CHECKS = ("sample", "step", "off")
_LISTS = ("trace", "energies", "observations", "log_t", "log_dt", "log_err")
# loop state restored when a rejected second step also discards the bootstrap
_BOOT_KEYS = ("n", "t", "E", "f", "E_scale", "last_sample", "snapshot", "windows",
              "clean_samples", "residual", "window_max_k")


def integrate(engine, P, alpha, ops, grid, *, dt0, L=1.0, E_ext=(0., 0., 0.),
              elastic_params=None, stab=0.0, err_tol=1e-3, k_min=-8, k_max=6, k_start=0,
              dt_max=None, grow_after=10, energy_check="sample", energy_rtol=1e-12,
              ceiling_hold=4, n_max, t_end=None, sample_every=10, criterion="legacy",
              tol=0.01, n_blocks=4, block_size=8, energy_scale=None,
              polarization_scale=None, residual_tol=1e-6, dPdt_tol=np.inf,
              rate_tol=np.inf, n_windows=5, max_cache=24, restart=None, callback=None):
    """Adaptive-dt SBDF2 relaxation of a deterministic gradient flow (see the
    module docstring for the scheme, error estimate and controller).

    engine: solver | bulk3d | film3d; ops as in Engine (film: a small dict).
    dt0: the ladder's base step (dt_k = dt0*2**k); start at level k_start.
    err_tol: local-error tolerance in polarization units (max-norm).
    n_max: cap on ACCEPTED steps; t_end: stop exactly at this time.
    sample_every: accepted steps between samples (trace, stopping test and,
        with energy_check="sample", the energy-rise check).
    criterion/tol/n_blocks/block_size/energy_scale/polarization_scale/
        residual_tol: casekit.anneal's stopping rules ("legacy" | "block_v2");
        "plateau" = residual <= residual_tol and max|dP|/dt <= dPdt_tol and
        |d f_density|/dt <= rate_tol at n_windows consecutive samples; None
        or tol < 0 = no stationarity test.
    restart: the `state` a previous call returned -- continues the identical
        recurrence (P must be that call's returned P).
    callback(info_dict) is called at every accepted sample.

    Returns (P, trace, stop_reason, state). trace: list of (t, E_total, dt)
    -- the initial state first (dt=0), then one entry per sample.
    stop_reason in {"stationary", "n_max", "t_end", "dt_min"}; "dt_min" means
    a step was rejected at the floor level k_min (P is the last accepted
    state). state: the restart certificate (P_previous, dt_previous, level,
    counters, sample history, model) plus the accepted-step log (log_t,
    log_dt, log_err), the work counts -- attempts = n + rejected_err +
    rejected_energy + rejected_nonfinite (per-step rejections) +
    rolled_back_steps (accepted steps discarded by the `rollbacks` of
    energy_check="sample") + bootstrap_redos (SBDF1 bootstraps redone at a
    lower dt) -- builds, energy_evals, force_evals, residual and seconds."""
    if criterion not in CRITERIA:
        raise ValueError(f"adaptive.integrate: unknown criterion {criterion!r}")
    if energy_check not in ENERGY_CHECKS:
        raise ValueError(f"adaptive.integrate: energy_check must be one of {ENERGY_CHECKS}")
    if not (np.isfinite(dt0) and dt0 > 0) or not (err_tol > 0) or n_max < 0:
        raise ValueError("adaptive.integrate: need dt0 > 0, err_tol > 0, n_max >= 0")
    if not (k_min <= k_start <= k_max) or grow_after < 1 or sample_every < 1:
        raise ValueError("adaptive.integrate: need k_min <= k_start <= k_max, "
                         "grow_after >= 1, sample_every >= 1")
    if criterion == "block_v2" and (energy_scale is None or polarization_scale is None):
        raise ValueError("adaptive.integrate: criterion='block_v2' requires explicit "
                         "energy_scale and polarization_scale (as casekit.anneal)")
    if t_end is not None and not t_end > 0:
        raise ValueError("adaptive.integrate: t_end must be positive")
    eng = Engine(engine, alpha, ops, grid, L=L, E_ext=E_ext, elastic_params=elastic_params,
                 stab=stab)
    S = eng.stab
    k_top = k_max
    if dt_max is not None:
        k_top = min(k_max, int(np.floor(np.log2(dt_max / dt0) + 1e-12)))
        if k_top < k_start:
            raise ValueError("adaptive.integrate: dt_max is below the starting dt")
    model = dict(engine=eng.kind, dt0=dt0, L=L, stab=S, err_tol=err_tol, k_min=k_min,
                 k_max=k_top, grow_after=grow_after, energy_check=energy_check,
                 energy_rtol=energy_rtol, ceiling_hold=ceiling_hold,
                 sample_every=sample_every, criterion=criterion, tol=tol)
    cache = OrderedDict()
    counters = dict(builds=0, energy_evals=0, force_evals=0)

    def operator(kind, dt):
        key = (kind, float(dt))
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        op = eng.build1(dt) if kind == "sbdf1" else eng.build2(dt)
        counters["builds"] += 1
        cache[key] = op
        if len(cache) > max_cache:
            cache.popitem(last=False)
        return op

    def energy(Pf):
        dens, tot = eng.energy(Pf)
        counters["energy_evals"] += 1
        scale = sum(abs(v) for k, v in tot.items() if k != "total")
        return float(dens["total"]), float(tot["total"]), scale

    def residual(Pf):
        counters["force_evals"] += 1
        return float(np.max(np.abs(L * eng.force(Pf))))

    P = np.asarray(P, dtype=float).copy()
    if restart is None:
        f0, E0, sc0 = energy(P)
        st = dict(n=0, t=0.0, k=k_start, ceiling=k_top, clean_samples=0, small=0,
                  P_previous=None, dt_previous=None, E=E0, f=f0, E_scale=sc0,
                  trace=[(0.0, E0, 0.0)], energies=[], observations=[], windows=0,
                  last_sample=dict(t=0.0, f=f0, P=P.copy()), log_t=[], log_dt=[], log_err=[],
                  attempts=0, rejected_err=0, rejected_energy=0, rejected_nonfinite=0,
                  rollbacks=0, rolled_back_steps=0, bootstrap_redos=0, residual=None,
                  window_max_k=k_start, snapshot=None)
    else:
        st = dict(restart)
        for key in _LISTS:
            st[key] = list(st[key])
        old = st["model"]
        if any(old[k] != model[k] for k in model):
            raise ValueError("adaptive.integrate: restart model parameters changed")
    st["model"] = model

    def take_snapshot():
        st["snapshot"] = dict(P=P.copy(), P_previous=None if st["P_previous"] is None
                              else st["P_previous"].copy(),
                              **{k: st[k] for k in ("n", "t", "k", "small", "dt_previous",
                                                    "E", "f", "E_scale")},
                              n_log=len(st["log_t"]))

    if st["snapshot"] is None and energy_check == "sample":
        take_snapshot()

    def stationary_test(Pf, f_now):
        """casekit.anneal's rules (+ 'plateau'); returns True to stop."""
        if criterion is None or tol < 0:
            return False
        if criterion == "legacy":
            return (len(st["energies"]) >= 2 * n_blocks and
                    _block_stationary(st["energies"], n_blocks, tol))
        if criterion == "block_v2":
            obs = st["observations"]
            if len(obs) < n_blocks * block_size or len(obs) % block_size:
                return False
            blocks = np.asarray(obs[-n_blocks * block_size:]).reshape(
                n_blocks, block_size, 5).mean(axis=1)
            scales = [energy_scale, *([polarization_scale] * 3), polarization_scale ** 2]
            if not all(_block_stationary(blocks[:, j], n_blocks, tol, scale=scales[j])
                       for j in range(5)):
                return False
            st["residual"] = residual(Pf)
            return st["residual"] <= residual_tol
        last = st["last_sample"]                       # "plateau"
        span = st["t"] - last["t"]
        st["residual"] = residual(Pf)
        good = (st["residual"] <= residual_tol and span > 0
                and float(np.max(np.abs(Pf - last["P"]))) / span <= dPdt_tol
                and abs(f_now - last["f"]) / span <= rate_tol)
        st["windows"] = st["windows"] + 1 if good else 0
        return st["windows"] >= n_windows

    reason = None
    t0_wall = time.perf_counter()
    start_n = st["n"]
    while reason is None:
        if st["n"] - start_n >= n_max:
            reason = "n_max"
            break
        if t_end is not None and st["t"] >= t_end * (1 - 1e-12):
            reason = "t_end"
            break
        k = min(st["k"], st["ceiling"])
        dt = dt0 * 2.0 ** k
        if t_end is not None and st["t"] + dt >= t_end * (1 - 1e-12):
            dt = t_end - st["t"]                       # land exactly on t_end
        st["attempts"] += 1
        P_prev, dt_prev = st["P_previous"], st["dt_previous"]
        if P_prev is None:                             # bootstrap: SBDF1, no estimate
            new = eng.step1(P, operator("sbdf1", dt), dt)
            err = 0.0
        else:
            if dt == dt_prev:                          # the engine's own SBDF2
                op2 = operator("sbdf2", dt)
                new = eng.step2(P, P_prev, op2, dt)
                P_star = 2 * P - P_prev
                dt_eff, filt, c = 2.0 * dt / 3.0, op2, 3.0   # [I+dt_eff L(A+S)]^-1 = 3*Minv2
            else:                                      # variable-step SBDF2
                w = dt / dt_prev
                dt_eff = dt * (1.0 + w) / (1.0 + 2.0 * w)
                filt, c = operator("sbdf1", dt_eff), 1.0
                new, P_star = eng.vstep2(P, P_prev, filt, dt, w)
            # err = (1 + dt_eff L S) * max|[I + dt_eff L (A+S)]^-1 (P+ - P*)|
            D = new - P_star
            if not np.all(np.isfinite(D)):
                err = np.inf
            elif np.isinf(err_tol):                    # fixed-dt use: no estimate
                err = 0.0
            else:
                err = c * (1.0 + dt_eff * L * S) * float(np.max(np.abs(eng.filtered(filt, D))))
        finite = bool(np.all(np.isfinite(new))) and np.isfinite(err)
        verdict = "ok"
        if not finite:
            verdict = "nonfinite"
        elif err > err_tol:
            verdict = "err"
        elif energy_check == "step":
            f_new, E_new, sc_new = energy(new)
            if E_new > st["E"] + energy_rtol * max(sc_new, st["E_scale"]):
                verdict = "energy"
        if verdict != "ok":
            st["rejected_" + verdict] += 1
            st["small"] = 0
            if verdict in ("nonfinite", "energy"):
                st["ceiling"] = min(st["ceiling"], k - 1)
                st["clean_samples"] = 0
            st["k"] = k - 1
            if st.get("boot") is not None and P_prev is not None:
                # the only history is the SBDF1 bootstrap: its own (first-order)
                # error is part of what was rejected, so redo it at the lower dt
                boot = st.pop("boot")
                st["bootstrap_redos"] += 1
                P = boot.pop("P")
                for key, size in boot.pop("lens").items():
                    del st[key][size:]
                st.update(boot, P_previous=None, dt_previous=None)
            if st["k"] < k_min or st["ceiling"] < k_min:
                reason = "dt_min"
            continue
        # ---- accept ----
        st["boot"] = (dict(P=P.copy(), lens={key: len(st[key]) for key in _LISTS},
                           **{key: st[key] for key in _BOOT_KEYS})
                      if P_prev is None else None)
        st["P_previous"], P = P, new
        st["dt_previous"] = dt
        st["t"] += dt
        st["n"] += 1
        st["window_max_k"] = max(st["window_max_k"], k)
        st["log_t"].append(st["t"])
        st["log_dt"].append(dt)
        st["log_err"].append(err)
        if energy_check == "step":
            st["f"], st["E"], st["E_scale"] = f_new, E_new, sc_new
        if P_prev is not None and err < err_tol / 4:
            st["small"] += 1
        else:
            st["small"] = 0
        if st["small"] >= grow_after and k < min(k_top, st["ceiling"]):
            st["k"], st["small"] = k + 1, 0
        else:
            st["k"] = k
        done_n = st["n"] - start_n >= n_max
        done_t = t_end is not None and st["t"] >= t_end * (1 - 1e-12)
        # samples follow the ABSOLUTE accepted-step count (so a restart split
        # anywhere continues the identical recurrence); t_end adds one final
        # sample, n_max (a budget, as in casekit.anneal) does not
        if st["n"] % sample_every and not done_t:
            if done_n:
                reason = "n_max"
            continue
        # ---- sample ----
        if energy_check == "step":
            f_s, E_s = st["f"], st["E"]
        else:
            f_s, E_s, sc_s = energy(P)
            if (energy_check == "sample" and
                    E_s > st["E"] + energy_rtol * max(sc_s, st["E_scale"])):
                snap = st["snapshot"]                  # roll the window back
                st["rollbacks"] += 1
                st["rolled_back_steps"] += st["n"] - snap["n"]
                st["ceiling"] = min(st["ceiling"], st["window_max_k"] - 1)
                st["clean_samples"] = 0
                P = snap["P"].copy()
                st["P_previous"] = None if snap["P_previous"] is None else snap["P_previous"].copy()
                for key in ("n", "t", "small", "dt_previous", "E", "f", "E_scale"):
                    st[key] = snap[key]
                st["k"] = min(snap["k"], st["ceiling"])
                st["small"] = 0
                for key in ("log_t", "log_dt", "log_err"):
                    del st[key][snap["n_log"]:]
                st["window_max_k"] = st["k"]
                st["boot"] = None
                if st["ceiling"] < k_min:
                    reason = "dt_min"
                continue
            st["f"], st["E"], st["E_scale"] = f_s, E_s, sc_s
        st["clean_samples"] += 1
        if st["clean_samples"] >= ceiling_hold and st["ceiling"] < k_top:
            st["ceiling"], st["clean_samples"] = st["ceiling"] + 1, 0
        st["trace"].append((st["t"], E_s, dt))
        st["energies"].append(E_s)
        st["observations"].append([f_s, *P.reshape(3, -1).mean(axis=1),
                                   float(np.mean(np.sum(P * P, axis=0)))])
        if stationary_test(P, f_s):
            reason = "stationary"
        st["last_sample"] = dict(t=st["t"], f=f_s, P=P.copy())
        st["window_max_k"] = st["k"]
        if energy_check == "sample":
            take_snapshot()
        if callback is not None:
            callback(dict(n=st["n"], t=st["t"], E=E_s, dt=dt, level=k, err=err,
                          residual=st["residual"], attempts=st["attempts"],
                          seconds=time.perf_counter() - t0_wall))
        if reason is None and done_t:
            reason = "t_end"
        elif reason is None and done_n:
            reason = "n_max"
    st["stop_reason"] = reason
    st.update(builds=st.get("builds", 0) + counters["builds"],
              energy_evals=st.get("energy_evals", 0) + counters["energy_evals"],
              force_evals=st.get("force_evals", 0) + counters["force_evals"],
              seconds=st.get("seconds", 0.0) + time.perf_counter() - t0_wall)
    return P, list(st["trace"]), reason, st


def _block_stationary(values, n_blocks, tol, scale=None):
    from fftpf import pf2d_engine as solver
    return solver.block_stationary(values, n_blocks_check=n_blocks, tol=tol, scale=scale)
