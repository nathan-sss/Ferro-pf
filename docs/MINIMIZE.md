# Direct free-energy minimization (`solvers/minimize.py`, `casekit.relax`)

**What it is for.** Many runs only want the *relaxed* state: they integrate the gradient flow
`dP/dt = −L δF/δP` at a fixed dt for ~10⁴ steps and keep the last frame. `casekit.relax` /
`casekit.relax_film` minimize the engine's **own discrete F_total** directly (preconditioned
L-BFGS), and stop at a certified stationary point: `max|δF/δP| < gtol` in the engine's force
units **and** a relative energy change `< ftol`.

**What it is not.**

- **Not dynamics.** The iterates are not a trajectory and the iteration count is not a time.
  Nothing time-dependent (field ramps, dwell protocols, kinetics, hysteresis rates, dP/dt) can be
  read from it.
- **Not Langevin.** There is no noise and no finite-temperature sampling; `kT=` is accepted only
  to raise a `ValueError` for a call ported from `casekit.anneal`.
- **Not the same basin as the flow.** A minimizer and a gradient flow started from the **same**
  P0 may end in **different local minima** (different domain configurations): both only go
  downhill, but along different paths, and a multi-domain landscape has many minima. Started
  from the usual small random noise (a saddle at P ≈ 0), L-BFGS leaves the saddle without the
  flow's selection of the fastest-growing wavelengths and can settle in a different, higher
  domain arrangement: in the benchmarks below +1.5% in F_total on the −1% film and +1.5% in the
  32³ bulk cell. Which arrangement it picks is even sensitive to round-off (a change in the
  summation order of the inner products moved the −1% film result from F = 17.333 to 17.342);
  a given code and input are deterministic. Where the landscape has one relevant minimum it finds the flow's state: the 2D demo
  (ΔF/F = 3e-12), the 0% film (single monoclinic variant, ΔF/F = 5e-7 against a flow that had
  not finished), uniform tetragonal starts (`tests/test_minimize.py`, ≤ 1e-8).

## When to use it, and the recipe for random starts

- **Single-domain / unique-minimum problems, continuation** (a converged state re-relaxed after a
  small change of strain, field, temperature: quasi-static sweeps): call `relax` directly. On the
  2D demo, re-relaxing after each ΔE_z = 0.002 step takes 50–66 iterations.
- **Domain configurations from random noise:** let the flow nucleate the pattern, then let the
  minimizer finish. The flow's early phase is cheap and is what selects the domain topology; the
  long tail (wall motion, coarsening, the last decades of the residual) is where the flow is
  slow and the minimizer is fast:

  ```python
  P, trace, why = casekit.anneal(bulk3d, P0, alpha, ops, grid, dt=dt, elastic_params=el,
                                 n_max=6000, sample_every=50)          # default (legacy) stop
  P, trace, why, info = casekit.relax(bulk3d, P, alpha, ops, grid, elastic_params=el)
  ```

  On the 32³ bulk cell this lands **in the flow's own minimum** (ΔF/F = −4e-10, max|ΔP| = 3e-5)
  at 40% of the flow's cost; on the film (which has no `anneal`; use `film3d.step`/`step_sbdf2`
  for the warm-up) a 2000-step warm-up + `relax_film` reaches the flow's domain fractions and an
  energy 0.26% *below* where the fixed-dt run got in 840 s, in ~220 s.
- **Do not** use it for anything whose answer depends on the path in time (switching kinetics,
  rate-dependent loops, aging) or on temperature.

## Calling it

```python
from solvers import casekit
# 2D (solver) or bulk3d -- same arguments as casekit.anneal, minus dt/kT/sampling of time
P, trace, why, info = casekit.relax(engine, P0, alpha, ops, grid,
                                    E_ext=E_d, elastic_params=el,       # optional
                                    gtol=1e-6, ftol=1e-10, max_iter=10000)
# film (film3d + film_elastic): film3d's own argument set
P, trace, why, info = casekit.relax_film(P0, alpha, grid, G, kappa_b, el=el,   # E_ext, H11,
                                         gtol=1e-6, ftol=1e-10)                # poisson_factors
print(why, info["n_iter"], info["n_energy_evals"], info["max_force"], info["energy"])
```

- `why` ∈ `converged` (both tolerances met), `max_iter`, `roundoff_floor` (gtol is below what an
  energy-monotone search can resolve: the decrease a step promises is under the round-off of F;
  `info["max_force"]` says where it stopped), `line_search_failed`.
- `trace`: F_total at iteration 0, every `sample_every` iterations and at the end
  (`info["trace_iterations"]`). It is non-increasing: every accepted step satisfies sufficient
  decrease.
- `info`: `n_iter`, `n_energy_evals`, `n_grad_evals` (each evaluation computes both in one pass),
  `max_force` (final `max|δF/δP|` density — for the film exactly what `cases/bto_film_biaxial.py`
  logs as `force_max_deck`), `energy` (the engine's own `total_energy*` "total" of the returned P),
  `wall_time`, `tau`, `rel_energy_change`, `n_resets`.
- `gtol` is in the engine's force units (deck energy density per polarization unit), the same
  number as `casekit.anneal(criterion="block_v2", residual_tol=...)` and the film production
  plateau's `--residual-tol`, so it does not depend on grid size or cell volume.
- Threads: as for every engine run, set the four `*_NUM_THREADS=1` before importing numpy and
  `casekit.single_thread(...)` / `execution.configure(1, ...)`. (The minimizer's own inner products
  avoid BLAS, so an unpinned BLAS pool cannot stall it.)

### Options

| option | default | meaning |
|---|---|---|
| `method` | `"lbfgs"` | preconditioned L-BFGS (two-loop, memory `memory=10`, γ = sᵀy / yᵀH₀y, strong-Wolfe line search). `"fire"`: preconditioned FIRE with energy-monotone backtracking — simpler, usually slower. |
| `tau` | `None` | preconditioner scale, H₀ = τ (I + τA)⁻¹ W⁻¹. Periodic engines: A = A(k) + D(k)/κ, the stepper's own implicit operator, so the first unit step **is one SBDF1 step with dt = τ**. `None` → `minimize.default_tau` = 1/(largest Landau curvature at the uniform minima), capped at 1 (1.0 for BTO at 298 K). |
| `precondition` | `True` / `"depolarization"` | film: `"depolarization"` (default) uses, per in-plane mode, the exact 3Nz×3Nz symbol of gradient + H11 + short-circuit depolarization (memory 16·Ny·(Nx/2+1)·(3Nz)² bytes, 53 MB at 40×40×21); `"gradient"` uses the stepper's banded factors (no memory; then τ defaults to κ_b, the explicit depolarization's stability scale); `False` none. |
| `max_step` | `None` | cap on `max|ΔP|` per iteration. |
| `exact_energy` | `False` | `True`: every internal energy is bit-for-bit the engine's (the lambdified Landau density); `False`: the pow-free `minimize.bulk_energy_density_fast` (symbolically identical, round-off equal, ~2× cheaper evaluation on the film). `energy(P)` and `info["energy"]` are always the engine's own number. |
| `energy_scale` | `None` | floor for the denominator of the relative energy change (the energy zero is arbitrary). |

### Supported physics and refusals

Supported (gradient verified against central finite differences of the engine's own energy,
≤ 1e-8 relative, on even and odd grids — `tests/test_minimize.py`): Landau 6th/8th order (scalar
or spatial α), isotropic and anisotropic G, H11 (grad4 ops in 2D/3D, `H11=` on the film), scalar
κ_b (film: also a κ(z) node profile), homogeneous elasticity (2D stress-free, misfit `eps_m`,
`film_full`, `eps_star_static`; bulk3d `stress_free`/`clamped`, per-label Q; film with substrate,
`C_substrate`, misfit `e0`), uniform or field `E_ext`.

Refused with a `ValueError`: SPM κ(r) (2D/3D `ops["dkappa"]`, `film3d_spm` factors) and SPM
elasticity C(r) (any `el` with `"spm"`) — their fields are iterated to a tolerance, so energy and
force are not an exact pair; Langevin (`kT`, `noise_cutoff_nm`); ensemble-batched P; an indefinite
film preconditioner (negative gradient coefficients — pass `precondition=False`). A 2D/3D H11
operator with G11 < 0 (indefinite) is shifted so the preconditioner stays positive definite.

## How it works (one paragraph each)

**Energy and gradient.** `energy(P)` calls the engine's `total_energy` / `total_energy_3d` /
`grad4.total_energy_hog[_3d]` / `film3d.total_energy_3d`. The gradient is the exact nodal
derivative `∂F_total/∂P = W · force`, with W the quadrature weight of a node: `Volume/N` on the
periodic grids, `(Lx·Ly/(Nx·Ny))·qw(z)` on the film (trapezoidal: hz inside, hz/2 at both
surfaces; the film's elastic force is first brought to the (1/qw) convention exactly as
`film3d.explicit_N` does). One fused evaluation shares the depolarization and elastic solves, so
it costs about one force evaluation plus the energy-only terms.

**Preconditioner.** H₀ = τ (I + τA)⁻¹ W⁻¹ = τ (W + τK)⁻¹ with K the symmetric nodal stiffness of
the linear terms — symmetric positive definite in the Euclidean metric of the gradient, i.e.
consistent with the film's weighted metric. On the film, A is read off by probing film3d's own
gradient and Poisson operators with a nodal delta (they are translation invariant in x/y, so the
response's rfft2 is a column of A(k) for every in-plane k at once).

**Stopping and round-off.** An energy-monotone search cannot certify forces below roughly
`sqrt(ε·|F| / (Volume·c))` (ε the float epsilon, c a curvature); for the 40×40×21 film that is
~1e-9, far below the production `residual_tol = 1e-6`.

## Benchmarks

Machine: 4-core container shared with other jobs (load average 2–10 during these runs), so wall
times are noisy; step and evaluation counts are the robust numbers. One "evaluation" = energy +
gradient in one pass. All runs single-threaded (FFT workers 1, BLAS 1), gtol = 1e-6, ftol = 1e-10.

### (a) 2D demo — `cases/demo_casekit_2d.py` setup

64×64, h = 0.4 nm, `bto_lcc2005_8th` at 298 K, quenched charge-disorder field E_d, no elasticity,
verified dt = 0.32 (SBDF1, as `casekit.anneal`). "Flow" = the same stepper continued until
`max|force| < 1e-6` (checked every 25 steps).

| run | steps / iterations | evaluations | wall | F_total | max force |
|---|---|---|---|---|---|
| demo `anneal` (legacy stop rule) | 400 steps | — | 0.34 s | −8.970828860 | 8.2e-4 |
| fixed-dt flow to max force < 1e-6 | 1700 steps | — | 1.44 s | −8.970935287304 | 8.5e-7 |
| `casekit.relax` (L-BFGS, τ = 1) | 85 iterations | 94 | 0.37 s (median of 5) | −8.970935287330 | 5.0e-7 |
| `casekit.relax(method="fire")` | 246 iterations | 251 | 1.0 s | −8.970935287316 | < 1e-6 |

relax vs flow: ΔF/F = −2.9e-12, max|ΔP| = 3.2e-5, ‖ΔP‖/‖P‖ = 5.9e-6 — the same minimum
(the demo's own legacy stop is 1.2e-5 above it, ‖ΔP‖/‖P‖ = 1.7e-2). 18× fewer evaluations,
~4× wall: on a 64² grid one evaluation (≈ 4 ms) is Python-overhead bound, ~7 steps' worth.

### (b) BTO film 40×40×21 — `Runs/BTO Film under strain/submit_film_B.sh` physics

16×16 nm, 8 nm thick (21 nodes), 12 `ss316_rt` substrate layers, ε_b = 10, 25 °C, the canonical
seed-0 layerwise noise of `scaled_initial_film.py`. Reference: the `bto_film_biaxial.py` loop and
plateau rule (5 consecutive checks every 51 steps with force ≤ 1e-6, ΔP < 1e-5, |ΔF|/Δt < 1e-8)
at a **fixed dt = 0.08, SBDF2**, with the production `--elastic-workers 1` force solver; the dt
ladder was not run (it would compare 0.08 with 0.04 and adopt the finer passing step, doubling the
step count). **Capped at 840 s**; neither strain reached the plateau. Per-operation medians on a
quiet machine: SBDF2 step 32 ms (53 ms with the reference elastic solve), one minimizer
evaluation 66 ms, dense preconditioner apply 4.8 ms, its setup 0.8 s (53 MB). Comparisons are
against the fixed-dt end state; correlation = ⟨P_a,P_b⟩/‖P_a‖‖P_b‖ with the qw weights.

| strain | run | steps / iterations | evaluations | wall | F_total | max force | vs fixed-dt end state |
|---|---|---|---|---|---|---|---|
| 0 | fixed-dt SBDF2, capped | 19 302 steps (t = 1544) | — | 840 s* | −3.954291425 | 9.5e-6 | single variant (P₁<0, P₂>0, P₃>0) |
| 0 | `relax_film` from P0 | 215 | 252 | 20.3 s | −3.954293258 | 8.4e-7 | ΔF/F = −4.6e-7, corr 0.993 (0.9996 at the best in-plane shift), same single variant |
| 0 | capped flow → `relax_film` | +11 | +17 | +1.4 s | −3.954293379 | 9.6e-7 | the stationary state the flow was approaching |
| −1% | fixed-dt SBDF2, capped | 20 685 steps (t = 1655) | — | 840 s* | 17.087690 | 4.4e-3 | +z 0.595 / −z 0.404, 6 mid-plane domains |
| −1% | `relax_film` from P0 | 1312 | 1616 | 127 s | 17.341847 | 9.5e-7 | **ΔF/F = +1.49%**, corr 0.827, +z 0.588 / −z 0.412, 7 domains |
| −1% | same, banded `"gradient"` preconditioner | 2616 | 2770 | — | 17.451992 | 9.2e-7 | ΔF/F = +2.1% |
| −1% | 2000 flow steps → `relax_film` | 2000 + 1479 | 1826 | 77 + 144 s* | 17.042979 | 8.7e-7 | ΔF/F = −0.26%, corr 0.975, +z 0.596 / −z 0.404 |
| −1% | capped flow → `relax_film` | +652 | +807 | +64 s | 17.039330 | 9.4e-7 | ΔF/F = −0.28%, corr 0.985, +z 0.599 / −z 0.401 |

\* measured under load average 7–10 (the fixed-dt runs: 36–40 ms per step instead of 32 ms; at
32 ms the 25 000 steps of production's end-time 2000 take ~800 s and also end on the time limit,
not the plateau). Reading: at 0% the minimizer certifies the flow's state (force < 1e-6, which the
flow had not reached after 19 302 steps) ≥ 30× faster; at −1% from noise it is ≥ 5× faster but
lands in a different, +1.5% domain arrangement; with a flow warm-up it finishes the flow's
arrangement and goes below the capped flow's energy.

### (c) bulk3d 32³ with elasticity

N = 32, d = 1 nm (`demo_casekit_3d.py`'s production size), `bto_lcc2005_8th` at 298 K,
`k0 = "stress_free"` elasticity, P0 = band-limited noise 0.005 (seed 0), verified dt = 0.637
(SBDF1). Measured at load average ~5.

| run | steps / iterations | evaluations | wall | F_total | max force | vs flow |
|---|---|---|---|---|---|---|
| `anneal` (legacy stop) | 500 steps | — | 10.9 s | −54.483674 | 1.7e-2 | — |
| fixed-dt flow to max force < 1e-6 | 1825 steps | — | 39.8 s (21 ms/step) | −54.608294977 | 8.0e-7 | — |
| `casekit.relax` from P0 | 194 | 244 | 10.9 s (median of 3) | −53.766307 | 3.7e-7 | **ΔF/F = +1.54%**, corr 0.930: another minimum |
| `anneal` (legacy, 500 steps) → `relax` | 500 + 62 | 78 | 12.1 + 3.3 s | −54.608295001 | 6.0e-7 | ΔF/F = −4.3e-10, max\|ΔP\| = 2.6e-5: the flow's minimum |

One evaluation costs ≈ 2.1 steps here (45 vs 21 ms).

Reproduce: (a)/(c) are `casekit.relax` with defaults against the engine's `step`/`step_3d` at
the verified dt, looped until `max|total_force| < 1e-6` (checked every 25 steps); (b) is
`casekit.relax_film` with defaults against the `cases/bto_film_biaxial.py` stepping loop and
plateau rule at dt = 0.08 (SBDF2, `execution.configure(1, "numpy", el, elastic_workers=1)`), the
case built with that script's `setup()` and `initial_state(grid, 0)`.
