# Adaptive dt and linear stabilization (opt-in)

Two opt-in features for **deterministic** runs:

- **`stab=S`**: linear stabilization on the deterministic builders and steps of all three engines
  (`solver`, `bulk3d`, `film3d`). It lets a semi-implicit step stay bounded beyond the explicit stability limit.
- **`casekit.anneal_adaptive`** (core in `solvers/adaptive.py`): adaptive-dt SBDF2 for relaxation and
  coarsening, using `casekit.anneal`'s stopping rules.

Neither changes any existing code path. `stab=0` (the default) runs the original arithmetic bit for bit, and
nothing calls the adaptive driver unless you do (see "Bit-identity" below).

## When to use it, and when not

| Use | Do not use |
|---|---|
| Deterministic relaxation to a stationary state (the end state matters, the path does not) | **Langevin / kT runs**. `anneal_adaptive` refuses `kT` and `noise_cutoff_nm`: the OU noise and the discrete FDT are tied to one fixed dt. Use `casekit.anneal` at a `verified_dt` / `verified_ensemble_dt` dt |
| Coarsening and domain dynamics where you want error control rather than a hand-picked dt (`stab=0`, with `err_tol` as the accuracy dial) | Runs whose protocol needs a **certified fixed dt** (dt-vs-dt/2 audit, `timestep.verified_dt`), e.g. field ramps, dwell ladders, χ(ω), anything compared step-by-step with an archived run |
| Long plateaus that a fixed dt crawls through because the explicit Landau (and, in the film, electrostatic) stiffness caps dt (`stab=S`) | Dynamics with `stab>0`: S changes the time accuracy (below). If you need dynamics with S, re-audit at dt/2 with the same S (`timestep.verified_dt(..., stab=S, skip_probe=True)`) |

**Multi-basin warning (ENGINE_GUIDE §13).** A different time integrator can end in a different local minimum
from the same noise seed, and so can a different `err_tol` or S. Adaptive dt is a cheaper way to reach *a*
stationary state. It is not a basin-preserving replacement for a certified run: compare final energies across
seeds as usual.

## Quick start

```python
from solvers import casekit, materials, solver, adaptive

deck = materials.get("bto_lcc2005_8th", T=298.0)
grid, ops = casekit.build_2d(deck, Nx=64, Ny=64, h=0.4)
P0 = solver.random_smooth_field(grid, 3, amplitude=0.005, seed=0)

P, trace, why, state = casekit.anneal_adaptive(
    solver, P0, deck["landau"], ops, grid, dt0=0.32,      # dt0: e.g. timestep.verified_dt's answer
    err_tol=1e-3,                                         # local error, polarization units (max-norm)
    stab=0.0,                                             # or S, see below (relaxation only)
    n_max=20000, sample_every=10,
    criterion="block_v2", energy_scale=1e-3, polarization_scale=0.26, residual_tol=1e-6)
# trace = [(t, E_total, dt), ...]; why in {"stationary", "n_max", "t_end", "dt_min"}
# state: restart certificate + work counters (attempts, rejected_*, builds, energy_evals, seconds, log_dt ...)
```

`bulk3d` takes the same call. For `film3d` pass `ops={"G11": G, "kappa_b": kappa}` (optionally `"H11"`), which
is the same small dict `timestep.verified_dt` uses for the film, and the film's `el` as `elastic_params`.
`restart=state` continues the identical recurrence: a split run is bit-identical to an unsplit one.

## The scheme

**Ladder.** `dt_k = dt0·2^k`, `k_min ≤ k ≤ k_max` (defaults −8 and 6), capped by `dt_max`. Implicit operators
are built lazily and cached per (dt, S), up to 24 entries. Start level `k_start=0`.

**Steps.** SBDF2 throughout. A constant-dt step calls the engine's own `step_sbdf2`, so pinned to one level the
driver reproduces `timestep.trajectory(scheme="sbdf2")` bit for bit (tested). A step right after a dt change uses
the **variable-step SBDF2** coefficients, with ω = dt_n/dt_{n−1}:

    [(1+2ω)/(1+ω) P⁺ − (1+ω) P + ω²/(1+ω) P⁻] / dt = −L(A+S)P⁺ − L[N(P*) − S·P*],   P* = (1+ω)P − ωP⁻

Its implicit operator is the engine's SBDF1 operator at `dt·(1+ω)/(1+2ω)`, so no new builder is needed. At ω=1
it equals the engine's SBDF2 to ~1e-16, and it is second order on a non-uniform mesh (both tested). Only the very
first step is SBDF1 (the bootstrap). If the step after it is rejected, the bootstrap is discarded and redone at
the lower dt.

**Error estimate (no extra force evaluation).** P* is the scheme's own explicit linear-extrapolation predictor,
so D = P⁺ − P* ≈ dt²P''. The leading SBDF2 local error is the IMEX splitting term
`dt_eff·L·[I+dt_eff·L(A+S)]⁻¹(∂N/∂P + S)·D`. The controller therefore uses

    err = (1 + dt_eff·L·S) · max |[I + dt_eff·L(A+S)]⁻¹ D|,     dt_eff = dt(1+ω)/(1+2ω)

This is D with the stiff, implicitly damped components filtered out (Shampine's filter, two FFTs). The filter
matters because the high-k noise of a random start makes D large while being integrated accurately; without it
dt collapses to the floor. On the slow, low-k modes `err ≈ max|D|`.

**Controller.**
- Reject and halve when `err > err_tol` or the state is non-finite.
- Double after `grow_after` (default 10) consecutive accepted steps with `err < err_tol/4`.
- Gradient flow: an energy rise beyond round-off (`energy_rtol`·Σ|energy terms|, default 1e-12) is rejected.
  - `energy_check="sample"` (default): the energy is evaluated at the sample points only, and a rise rolls the
    whole window back to the previous sample.
  - `"step"`: the energy is checked every step. It costs one energy evaluation per step, which is 3–8 steps'
    worth on these engines because the lambdified bulk energy dominates: 8.4 ms vs 1.0 ms per 2D 64² step;
    0.12 s vs 0.043 s on the 40×40×21 film.
  - An energy or non-finite rejection also caps the level below the failing one until `ceiling_hold` (4) clean
    samples have passed.
- `stop_reason="dt_min"` means a step was rejected at `k_min`. P is the last accepted state; lower `k_min` or
  loosen `err_tol`.

**Stopping.** `casekit.anneal`'s rules, unchanged in meaning:
- `criterion="legacy"`: `block_stationary` on the sampled energies, with `tol`.
- `"block_v2"`: block means of the five observables against explicit `energy_scale`/`polarization_scale`, plus
  `max|L·∂F/∂P| ≤ residual_tol`.
- The film cases' **`"plateau"`** rule: residual ≤ `residual_tol`, `max|ΔP|/Δt ≤ dPdt_tol`, `|Δf|/Δt ≤ rate_tol`,
  at `n_windows` consecutive samples.
- `None`.

On top of the criterion, `n_max` (accepted steps) and `t_end` (the last step lands exactly on it). Samples are
every `sample_every` *accepted* steps, counted absolutely, so their spacing in time follows dt. `legacy` is loose
for a deterministic run (on the 2D demo it fires at its 400-step minimum with residual 8e-4). Use `block_v2` or
`plateau` with a residual when the end state matters.

## The `stab` option

```python
Minv  = solver.build_Minv(grid, ops, dt, L, stab=S)        # also build_Minv2, *_3d, film3d.build_grad_factors[_sbdf2]
P_new = solver.step(P, alpha, ops, Minv, dt, L, E_ext, stab=S)
```

The two schemes become:

    SBDF1: (P⁺−P)/dt          = −L(A+S)P⁺ − L[N(P)  − S·P ]
    SBDF2: (3P⁺−4P+P⁻)/(2dt) = −L(A+S)P⁺ − L[N(P*) − S·P*],   P* = 2P − P⁻   (S·P extrapolated like N)

- **Builder and step must receive the same S.** The builder records S on its return value (an ndarray view,
  `adaptive.StabilizedInverse`, or a `"stab"` key in the film factors). The step refuses a mismatch in both
  directions with `ValueError`. `stab=0` adds no tag and no key.
- **Fixed points are unchanged.** The two S terms cancel at a fixed point. Tested: the exact uniform tetragonal
  state stays put to <1e-12 at dt=5 with S=ℓ on all three engines, and a perturbed S=0 relaxed state
  re-relaxes with S=ℓ to the same field (<1e-6) and energy (<1e-10 relative).
- **Stability.** 2D 32² BTO at dt=16, where S=0 SBDF1 and SBDF2 blow up in a few steps: S=ℓ/2 and S=ℓ keep
  SBDF1 bounded with strictly decreasing energy (stabilized SBDF1 is energy-stable for S ≥ ℓ/2), and S=ℓ keeps
  SBDF2 bounded. SBDF2+S is not monotone in energy during start-up; the adaptive energy check handles that.
- **Choosing S.** `adaptive.landau_lipschitz(alpha, p_max)` returns ℓ = sup‖∂²f_bulk/∂P²‖ over |P_i| ≤ p_max.
  For BTO at 298 K, ℓ is 4.6 / 11.5 / 25.8 at p_max = 0.30 / 0.35 / 0.40. Add the other **explicit** stiffness
  yourself:
  - in the **film** the electrostatic force is explicit, adding about 1/κ_b (11.3 at ε_b=10);
  - electrostrictive elastic forces add a smaller term.

  Too large an S slows the slow modes: SBDF1+S advances them by at most ~1/(L·S) per step. So take the smallest
  S that removes the dt cap. S = 8 is what the benchmarks below use: it pays off on the film and in 2D,
  and on bulk3d it relaxes into a different basin.
- **S changes the time accuracy.** SBDF1+S adds an O(dt·L·S) splitting term. SBDF2+S stays second order (measured
  convergence ratios 3.8–3.9 per halving), but its error constant grows with S: about 18× at S=13 on the 2D test,
  because `(2/3)·dt·L·S·(P⁺−P*)` accumulates coherently over the slow modes. The adaptive estimate cannot see
  this amplification (it is ∝ S/λ_slow), so with S the measured error is far above `err_tol` (table below).
  **Use S for relaxation. For dynamics, re-audit at dt/2 with the same S.**
  `timestep.trajectory(..., stab=S)` and `timestep.verified_dt(..., stab=S)` pass S through. The boundedness
  probe has no S, so use `skip_probe=True` with your dt0.
- The film's anisotropic 3×3 block gradient operator (dict `G` that is not isotropic) has no `stab` and raises
  `NotImplementedError`. The tri/penta (scalar `G`, optional `H11`) operators support it.

## Accuracy: the error tracks `err_tol` (S=0)

**Setup.** The `cases/demo_casekit_2d.py` case: BTO 298 K, 64², h=0.4, quenched `E_d`, noise start amplitude
0.005, dt0 = 0.32 (that case's `verified_dt`), integrated to t_end = 40.

**Reference.** Fixed SBDF2 at dt0/8 = 0.04, energy every step. Its own error, measured against dt0/16, is
max|ΔP| = 1.4e-6 and E(t) = 3e-5.

**Columns.**
- `max|ΔP(t_end)|`: polarization error at t_end.
- `E(t)`: max |ΔE| over the trace for t ≥ 2, relative to the total energy drop (9.08). The first stiff steps
  are excluded: the decay of the high-k noise is not resolved at any of these dt, the reference included.
- `E(t_end)`: same normalization, at t_end.

| run | steps (attempts) | max\|ΔP(t_end)\| | E(t) | E(t_end) |
|---|---|---|---|---|
| fixed SBDF2 dt=0.32 | 125 | 1.2e-4 | 2.4e-3 | 2.1e-7 |
| fixed SBDF2 dt=0.16 | 250 | 2.7e-5 | 5.9e-4 | 4.9e-8 |
| fixed SBDF2 dt=0.08 | 500 | 5.4e-6 | 1.2e-4 | 9.7e-9 |
| adaptive err_tol=1e-2 | 94 (98) | 1.9e-4 | 2.1e-3 | 4.2e-7 |
| adaptive err_tol=3e-3 | 112 (116) | 6.1e-5 | 1.8e-3 | 3.0e-7 |
| adaptive err_tol=1e-3 | 186 (196) | 3.2e-5 | 1.1e-4 | 1.1e-7 |
| adaptive err_tol=3e-4 | 291 (304) | 2.0e-5 | 2.5e-5 | 3.4e-8 |
| adaptive err_tol=1e-4 | 485 (499) | 8.6e-6 | 4.0e-5 ≈ ref floor | 9.2e-9 |
| adaptive+S=8 err_tol=1e-2 | 76 (77) | **3.8e-2** | **1.6e-1** | 5.4e-5 |
| adaptive+S=8 err_tol=1e-3 | 174 (181) | **1.0e-2** | 4.0e-3 | 2.9e-5 |
| adaptive+S=8 err_tol=1e-4 | 487 (500) | **1.9e-3** | 4.5e-4 | 7.1e-6 |

**What the table shows.**
- With S=0 the error falls monotonically with `err_tol` and stays below it: |ΔP| ≤ 0.09·err_tol.
- At equal work the adaptive run matches fixed dt on P and beats it on E(t). For example, `err_tol=1e-3` takes
  196 attempts and has an E(t) error 5× smaller than fixed dt=0.16 at 250 steps.
- With S>0 the time error is 10²–10³ × `err_tol`. This is the documented cost of the stabilization; it does not
  matter for the end state of a relaxation.

**Late-phase floor.** Over a longer horizon, to t=100, an `err_tol`-independent floor of about 2e-5 in P appears
once dt reaches the explicit stability limit and chatters between two levels.

`tests/test_adaptive.py::test_adaptive_error_tracks_tol` pins the same property on a 24² dynamics case: the
error is monotone in tol, below tol, and falls ×20 over two decades.

## Benchmark (2026-11-04)

**Conditions.**
- Single-thread FFT/BLAS on a 4-core container, each run alone.
- 2D wall time is the median of 3 repeats; the film and bulk3d runs are single (minutes long).
- "steps" = accepted steps; "attempts" also counts rejected and rolled-back steps.
- `err_tol=1e-3` and `energy_check="sample"` throughout.
- Scripts are not in the repo; they call `casekit.anneal` / `casekit.anneal_adaptive` exactly as described.

**(a) BTO film, `Runs/BTO Film under strain/submit_film_B.sh` physics.**
- Geometry and material: 40×40×21, 16×16×8 nm, 12 ss316_rt substrate layers, ε_b=10, 25 °C, strain 0, seed 0.
  The IC is that case's layerwise noise.
- Stop rule: the case's plateau rule, i.e. 5 consecutive checks every 51 steps with force ≤ 1e-6,
  max|ΔP| < 1e-5 per window and |Δf|/Δt < 1e-8.
- Fixed run: the case's own loop at dt0=0.08 SBDF2. The case's `--fixed-dt-ladder` audit is skipped; it would
  adopt 0.04 and double the cost.
- Adaptive runs: the same thresholds through `criterion="plateau"` (ΔP threshold as a rate,
  1e-5/(51·0.08)), samples every 51 accepted steps.
- Fixed and adaptive S=0 were capped at 840 s wall.

| run | steps (attempts) | wall | t reached | final energy density f | residual | stop |
|---|---|---|---|---|---|---|
| fixed SBDF2 dt=0.08 | 21,770 | 840 s (cap) | 1742 | −1.93080725e-3 | 2.1e-6 | cap; residual ×0.54 per 1020 steps ⇒ plateau ≈ 23,200 steps ≈ 900 s |
| adaptive, S=0 | 16,932 (18,276) | 842 s (cap) | 2294 | −1.93080731e-3 | 2.0e-6 | cap (dt chatters 0.08–0.32: the film's explicit electrostatics + Landau cap it) |
| **adaptive + S=8** | **1,989 (2,111)** | **97 s** | 2751 | −1.93080728e-3 | 6.1e-8 | **plateau** (dt median 1.28, max 5.12; 18 err rejections, 2 window rollbacks) |

The three runs reach the same state: f agrees to 3e-8 relative. That makes stabilized adaptive dt about
**9× faster** in wall time than the fixed run's extrapolated plateau, and ~11× fewer step evaluations. Adaptive
dt alone gains nothing here: without S the step stays at the explicit stability limit.

**(b) 2D, `cases/demo_casekit_2d.py` setup.** 64², BTO 298 K, quenched E_d, dt0 = 0.32 (the case's
`verified_dt`), S=8.

| criterion | run | steps (attempts) | wall | E_total | residual |
|---|---|---|---|---|---|
| demo's own (`legacy`, every 50) | fixed SBDF1 (the demo) | 400 | 0.25 s | −8.97083 | 8.2e-4 |
| | fixed SBDF2 | 400 | 0.45 s | −8.97083 | 8.2e-4 |
| | adaptive | 400 (428) | 0.58 s | −8.97085 | 8.2e-4 |
| | adaptive + S=8 | 400 (457) | 0.71 s | −8.97093 | 1.4e-4 |
| `block_v2` + residual ≤ 1e-6 (every 10) | fixed SBDF2 | 1,680 | 2.56 s | −8.9709352873 | 8.5e-7 |
| | adaptive | 1,200 (1,346) | 2.51 s | −8.9709352873 | 9.4e-7 |
| | **adaptive + S=8** | **480 (547)** | **1.19 s** | −8.9709352873 | 8.2e-7 |

The demo's legacy rule always fires at its 400-step minimum (2·n_blocks samples), which is far from stationary
(residual 8e-4), so nothing can be gained there. With a residual criterion, S=8 is 2.1× faster and S=0 breaks
even.

**(c) bulk3d.** 32³, d=1 nm, elasticity (stress-free), noise-free deterministic relaxation from a 0.005-amplitude
start, `block_v2` + residual ≤ 1e-6, every 10 steps.

| run | steps (attempts) | wall | E_total | residual | stop |
|---|---|---|---|---|---|
| fixed SBDF2 dt=1.28 (`probe_stable_dt_3d`'s answer) | 20,000 | 538 s | −54.5928 | 5.1e-3 | **n_max: bounded but never converges** |
| fixed SBDF2 dt=0.64 | 1,600 | 43.2 s | −54.5932667 | 7.9e-7 | stationary |
| fixed SBDF2 dt=0.32 | 3,200 | 83.9 s | −54.5932667 | 7.4e-7 | stationary |
| **adaptive**, dt0=1.28 | **1,280 (1,347)** | **39.6 s** | −54.5932667 | 3.0e-7 | stationary |
| adaptive + S=8 | 1,360 (1,420) | 38.6 s | **−54.2143** | 7.0e-7 | stationary, **another basin** |

Started from the probed dt, which is bounded but parks SBDF2 in a persistent oscillation, the adaptive run finds
a working dt by itself and matches the best hand-picked fixed dt. S=8 relaxed just as fast but into a different,
higher-energy local minimum: the multi-basin caveat in practice.

## Bit-identity of the default path

Every new argument defaults to `stab=0.0`, guarded by `if stab:`. The S=0 branch is the original expression
(the only addition is a refusal if a stab-tagged operator is passed without its S). Evidence:

- a 138-array snapshot (2D/bulk3d/film; `build_Minv*`/grad factors; SBDF1/SBDF2 steps; elastic on/off; film
  tri/penta/block; energies; `relax`, `casekit.anneal` deterministic and Langevin; `timestep.trajectory`/
  `verified_dt`; bulk3d ensemble steps; film H11 and grad4 through `timestep`) is **byte-identical** (`cmp`)
  between the base tree and this one;
- `tests/test_adaptive.py::test_stab_zero_is_the_default_path` pins `stab=0.0` == omitted (same bytes, plain
  `ndarray`, no `"stab"` key).

## API

| Where | What |
|---|---|
| `solver.build_Minv/build_Minv2/step/step_sbdf2`, `bulk3d.*_3d` twins, `film3d.build_grad_factors[_sbdf2]/step/step_sbdf2` | `stab=0.0` keyword |
| `timestep.trajectory(..., stab=0.0)`, `timestep.verified_dt(..., stab=0.0)` | pass-through (`info["stab"]` when nonzero) |
| `casekit.anneal_adaptive(engine, P, alpha, ops, grid, *, dt0, err_tol=1e-3, stab=0.0, k_min=-8, k_max=6, k_start=0, dt_max=None, grow_after=10, energy_check="sample", n_max, t_end=None, sample_every=10, criterion="legacy", ...)` | returns `(P, trace, stop_reason, state)` |
| `adaptive.integrate(...)` | the same driver (casekit is sugar over it) |
| `adaptive.Engine` | one calling convention for build/step/variable-step SBDF2/energy/force on the three engines |
| `adaptive.landau_lipschitz(alpha, p_max, n=33)` | ℓ for choosing S |

## Limitations

- Deterministic only, by design.
- The error estimate is first-order-predictor based (≈ dt²P''), so error tracks `err_tol` during the dynamics.
  Late in a run, where dt sits at the explicit stability limit (S=0), dt chatters between two levels and the
  error has a small floor independent of `err_tol` (about 2e-5 on the 2D demo at t=100).
- With `stab>0` the estimate does not bound the global time error (see above).
- `energy_check="sample"` rejects windows, not single steps. A rise inside a window that recovers before the
  sample point goes unnoticed. Use `"step"` if that matters and you can pay one energy per step.
- Samples are per accepted step, so `legacy`/`block_v2` blocks span more physical time as dt grows.
- The film block (anisotropic `G`) operator has no `stab`. `L` must be a scalar.
