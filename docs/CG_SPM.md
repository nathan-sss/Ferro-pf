# CG solvers for the inhomogeneous C(r) / κ(r) problems (opt-in)

The engine solves inhomogeneous elasticity C(r) and permittivity κ(r) with the
basic Moulinec–Suquet fixed point ("SPM", Wang et al., Acta Mater 61 (2013)
7591): each iteration applies the Green's operator of a uniform reference
(C0 or κ0) to a source that contains the perturbation C − C0 (or κ − κ0). That
iteration converges only while the reference bounds every local modulus
(C0 ≳ max C / 2, κ0 > max|κ − κ0|). Its iteration count also grows roughly
in proportion to the contrast.

This page describes an **opt-in** alternative that solves the *same discrete
equations* by (preconditioned) conjugate gradients:

- It converges for any positive-definite C(r) and any κ(r) > 0, whatever the
  reference.
- The iteration count grows like √contrast instead.
- One iteration costs the same as one basic iteration: one Green or
  homogeneous-κ0 apply.

The default ("basic") paths are unchanged, bit for bit. A basic setup dict
does not gain any key.

## 1. Selecting it

| problem | entry point | switch | default |
|---|---|---|---|
| 2D elastic C(r) | `elastic.setup_elastic_spm(...)` | `method="cg"`, `cg_precond=True` | `"basic"` |
| 3D elastic C(r) | `bulk3d_spm.setup_elastic_spm_3d(...)` | `method="cg"`, `cg_precond=True` | `"basic"` |
| 2D electrostatic κ(r) | `solver.build_operators(..., kappa_array)` | `spm_method="cg"`, `spm_cg_max_iter=None` | `"basic"` |
| 3D electrostatic κ(r) | `bulk3d.build_operators_3d(..., kappa_array)` | `spm_method="cg"`, `spm_cg_max_iter=None` | `"basic"` |
| film electrostatic κ(x,y,z) | `film3d_spm.poisson_factor_spm(...)` | `method="cg"`, `spm_cg_max_iter=None` | `"basic"` |

```python
el  = elastic.setup_elastic_spm(grid, C11f, C12f, C44f, Q11, Q12, Q44, method="cg")
el3 = bulk3d_spm.setup_elastic_spm_3d(grid3, C, Q, k0="clamped", method="cg")
ops = solver.build_operators(grid, G11, G12, G44, G44p, kappa_field, spm_method="cg")
ops3 = bulk3d.build_operators_3d(grid3, G11, G12, G44, G44p, kappa_field, spm_method="cg")
fac = film3d_spm.poisson_factor_spm(film_grid, kappa_field, method="cg")
```

Nothing downstream changes, because dispatch happens inside the existing
solve functions:

- `elastic_solve_spm`, `elastic_solve_spm_3d`;
- `solver._spm_iterate`, `bulk3d._spm_iterate_3d`;
- `film3d_spm.solve_poisson_spm`.

So `elastic_force`, `total_energy`, `_explicit_N`, the steppers,
`film3d.solve_poisson_factored`, etc. pick the CG up automatically. Builders
that construct these dicts themselves (composite, polycrystal, grad4's HOG
builder, …) do not forward the switch. To use CG there, set the keys on the
built dict:

- elastic: `el["spm_method"] = "cg"`, and raise `el["spm_max_iter"]`, since a
  basic el carries basic's cap of 100;
- electrostatic: `ops["spm_method"] = "cg"` and `ops["spm_cg_max_iter"] = 1000`.

For `cg_precond=False` the elastic dict also needs the `cg_*` Green keys, so
rebuild it with the setup function instead.

**Knobs.**

- `spm_tol` (elastic) / `tol` (electrostatic) is the CG relative residual
  (§2.3).
- `spm_max_iter=None` means 100 for basic and `elastic.CG_DEFAULT_MAX_ITER` =
  1000 CG iterations for cg. The electrostatic `max_iter=None` means 200
  (basic) or `ops["spm_cg_max_iter"]` (cg).
- `outer_tol` keeps its meaning, max|⟨σ⟩|/max|σ| (§2.1).
- `outer_max_iter` is unused by CG, which has no outer loop.
- `C0_override` / `kappa0_override` choose the preconditioner's reference.
  Unlike for basic, this is never needed for convergence.

## 2. Method

### 2.1 Elastic (`elastic._spm_cg_elastic`, shared by 2D and 3D)

The approach is the Galerkin form of Zeman et al., J. Comput. Phys. 229
(2010) 8065, and Vondřejc, Zeman & Marek, Comput. Math. Appl. 68 (2014) 156.

**Unknown and equations.**

- The unknown is the total strain ε = ε̄ + δε.
- δε is compatible and zero-mean on the code's modified-wavevector grid, which
  is exactly the space `_green_delta_eps` produces. Both methods therefore
  share one discrete problem and one fixed point.
- Equilibrium is G[C(r):(ε − ε*)] = 0, where G is the L²-orthogonal projection
  onto that space.

**The k = 0 block.**

- `k0="stress_free"` (2D always, 3D default): the constant strains belong to
  the space, so G[σ] = 0 also says ⟨σ⟩ = 0. One CG solve returns ε̄ as well,
  with no outer ε̄ loop.
- `k0="clamped"` (3D): ε̄ = 0 is fixed and only δε is unknown. ⟨σ⟩ is reported,
  not driven to zero, exactly as in basic.

**Inner product.** The inner product is (a, b) = ⟨a_ij b_ij⟩.

- It is the full 9-component contraction of the code's symmetric (3,3,…)
  storage, so shear pairs count twice. This is the same contraction that
  ½σ:e uses.
- The CG energy norm is therefore the elastic energy, and no Voigt factor is
  hand-coded. It is evaluated on the 6 independent slots with weight 2 on the
  off-diagonals.
- ε ↦ G[C:ε] is self-adjoint and positive definite in it.

**Preconditioner.** The preconditioner is M = G C_ref G with a uniform C_ref.

- M⁻¹s = Γ_ref[s] + C_ref⁻¹:⟨s⟩, where Γ_ref is the basic SPM's own Green apply
  built on C_ref. The second term is used only while ε̄ is free.
- With `cg_precond=True` (default), C_ref is the el's C0: ⟨C⟩, or
  `C0_override`.
- With `cg_precond=False`, C_ref is the identity on symmetric tensors. This is
  plain Galerkin CG with the compatibility projection.
- κ(M⁻¹A) is bounded by the eigenvalue spread of C_ref^-½ C(r) C_ref^-½ over
  the grid, which is about the stiffness contrast.

**Residual.**

- The residual is kept unprojected: s = C:(ε* − ε) = −σ, updated as
  s ← s − α C:p.
- Every inner product pairs s with a vector of the compatible space, where G
  is the identity, so G is never applied.
- The cost is one Green apply (12 FFTs in 3D) plus one C(r) contraction per
  iteration. Per solve there are about 2 extra applies: one for ‖b‖, one for
  the final true-residual check.

**Stopping.** Two conditions must both hold:

1. resid = √((r, M⁻¹r)/(b, M⁻¹b)) ≤ `spm_tol`. This is the relative residual in
   the preconditioner norm, where b is the residual of the zero start.
2. While ε̄ is free, max|⟨σ⟩|/max|σ| ≤ `outer_tol`, the basic outer test's own
   measure. This is also counted as met once |⟨σ⟩| is at the round-off floor
   of the load stress (64 ulp × max|C:ε*|). See the uniform-ε* note in §5.

Both conditions are re-checked on a freshly recomputed residual, and CG
restarts from it if recursion drift made the updated residual optimistic.

**Start and warm start.**

- The cold start is basic's: ε = ⟨ε*⟩ (stress_free) or ε = ε̄_fixed (clamped).
  A uniform ε* is therefore solved in 0 iterations with σ = 0 to round-off.
- The warm start reads and writes basic's `_warm_ebar` / `_warm_delta_eps`
  cache.

**Returned dict.** The keys are the same as basic, with these meanings for
CG:

| key | value |
|---|---|
| `n_inner` (and `n_inner_total` in 3D) | CG iteration count |
| `n_outer` | 1 |
| `resid_inner` | final CG relative residual |
| `resid_outer` | max\|⟨σ⟩\|/max\|σ\| of the returned state, computed exactly as basic does |

`_spm_convergence_check` runs on the final state:

- It warns if a tolerance is unmet.
- It raises if resid_inner > 1e-3 or σ is non-finite. For CG that means the
  cap ran out or C(r) is not positive definite, and the message says so.
- A non-positive curvature (p, C:p) ≤ 0 raises immediately.

**Ensembles.** A batched P (3,S,…) runs S independent CGs in lockstep, with
per-member α and β. A converged member is frozen. This works on the cubic and
the full-tensor 3D paths, where C_ref⁻¹ is a batched 6×6 symmetric solve.

### 2.2 Electrostatic (`solver._spm_pcg`, shared by 2D, 3D and film)

**The linear system.** Write the homogeneous-κ0 solve as core(s) = −T s, with
T = D(k)/κ0 per mode.

- T is symmetric positive semi-definite with eigenvalues in [0, 1/κ0].
- On the basic SPM's range, the fixed point E = core(P + δκ E) *is* the linear
  system B E = −Π P, with B = T⁺ + Π δκ Π (T⁺ the pseudo-inverse, Π the
  projection onto range T).
- Since T⁺ ≥ κ0 Π, B ≥ Π κ(r) Π. B is therefore SPD for any κ(r) > 0 and any
  κ0, whereas basic needs max|δκ| < κ0.

**Preconditioner.** The preconditioner is M = T⁺, so M⁻¹ = T, which is
basic's own homogeneous solve. κ(M⁻¹B) ≤ max(1, κmax/κ0)/min(1, κmin/κ0).

**Preimages.** T⁺ is never formed. Every vector v ∈ range T carries a
preimage ṽ with v = Tṽ, so M v = Π ṽ, and Π drops out of every inner product.
The cost is one core solve per iteration.

**Stopping.**

- resid = √((r, M⁻¹r)/(c, M⁻¹c)) ≤ `tol`, with c = −ΠP.
- (c, M⁻¹c) = −⟨P, E_uniform⟩ is twice the homogeneous-κ0 electrostatic
  energy, so this is a relative residual in the κ0-energy norm.
- It is re-checked on a recomputed residual.
- The returned E is exactly core() of its accumulated preimage.

**Output contract.** The output contract is unchanged: (Ed, Ed_uniform, n,
resid[, history]).

| output | value for CG |
|---|---|
| `Ed_uniform` | core(P), bit-identical to basic's |
| `n` | CG iterations |
| `resid` | final CG relative residual |

E_ext stays outside the iteration, as in basic. With `warm=` (bulk3d
`spm_warm=True`, film `_warm_E`), CG starts from one basic step taken from the
warm field, whose preimage is known exactly. It prints the same
"[module.function] WARNING …" lines as basic when `tol` is not met. Ensembles
run lockstep per member, as in §2.1.

**Film.** The film uses the same `_spm_pcg`:

- The core is film3d's banded κ0(z) solve.
- The inner product is Σ_z qw[z] Σ_xy, in which that solve is self-adjoint.

### 2.3 Tolerance semantics

`spm_tol` / `tol` measure different things for the two methods:

| method | stopping measure | worst-case error at tolerance tol |
|---|---|---|
| basic | relative max-norm step between iterates | can be ≈ contrast × tol, because the contraction factor is ≈ 1 − 1/contrast |
| CG | relative residual in the preconditioner's energy norm (§2.1, §2.2) | ≤ √(κ(M⁻¹A)) × tol, in the energy norm |

The `err` columns below measure the relative max-norm error of the solution
at the production tol = 1e-10, against a CG reference at 1e-13:

- Preconditioned CG stays between 2e-11 and 1e-9 at every contrast (plain CG
  up to 2e-9).
- Basic electrostatics degrades from ~4e-11 at contrast 2 to 5e-8 to 1e-7 at
  contrast 1000, the contraction-factor effect above.

## 3. Correctness evidence (`tests/test_cg_spm.py`, 24 tests, ≈ 5 s)

**Same fixed point as basic, wherever basic converges.**

- Elastic: sigma, e, δε, ε̄, energy and force agree to < 1e-9 (observed
  1e-13 to 1e-12), for 2D and 3D, both C(r) input paths (cubic / full tensor),
  stress_free and clamped, with a static eigenstrain, with and without the
  preconditioner.
- Electrostatic: E_d agrees to < 1e-10 at the same κ0 in 2D, 3D and film;
  E_uniform is bit-identical.

**Central-FD energy–force consistency with CG.**

- 2D elastic (even and odd grids, contrast 50), relative error < 1e-7.
- 3D elastic (stress_free and clamped, contrast 100), relative error < 1e-7.
- 2D electrostatic with E_ext ≠ 0 (contrast 10 and 1000), relative error
  < 1e-8.

**High contrast.**

- Elastic contrasts 10/100/1000: CG converges with no warning. The
  equilibrium residual, measured with the preconditioner-independent identity
  projection, is < 1e-8·max|σ|. Plain and preconditioned CG agree to 1e-8.
  Basic with its default C0 raises DIVERGED.
- 3D thin high-κ layer (max|δκ| > κ0 at the default mean κ0): basic diverges,
  while CG converges and matches basic run at the midpoint κ0.

**Ensembles.** Ensemble members equal their unbatched solves:

- 2D elastic with a shared static field;
- 3D cubic and full tensor;
- 2D electrostatic, including members 1000× smaller than others.

**Warm starts.** Elastic 2D and bulk3d `spm_warm` take fewer iterations and
give the same answer. Film `_warm_E`, through `film3d.solve_poisson_factored`,
gives the same answer.

**Contract.**

- A basic dict gains no key.
- Unknown method names are refused.
- A too-small CG cap raises the CG-worded error and leaves the warm cache
  untouched.
- A uniform ε* returns at once without a warning.

**Bit identity of the default paths** (not in the suite, because it needs
the pre-change tree):

- A 465-entry hash battery is identical between `feat/engine-optimization`
  (9a8a52c) and this change.
- It covers the 2D/3D elastic SPM (setup dicts, cold/warm solves, static,
  ensemble, cubic/tensor, both k0) and the 2D/3D/film electrostatic SPM (ops
  dicts, histories, warm cache, forces).
- It includes stdout, so the printed warnings are identical too.

## 4. Benchmark

**Setup.**

- Single-threaded FFTs, on a shared 4-core container, so wall times are
  indicative only.
- Each cell reads **Green (or core) applies / wall time / err**. Applies are
  the machine-independent cost: basic counts every inner iteration over all
  outer iterations, CG counts its iterations plus about 2–3 overhead applies.
- err is the relative max-norm error of σ (elastic) or E_d (electrostatic) vs
  a CG solution at tol 1e-13 with the same reference.
- Tolerance is the production 1e-10 (`spm_tol`, `outer_tol`, `tol`).
- Basic is given generous caps (20000 inner × 500 outer), but is stopped at an
  apply budget: 40000 for 2D, 6000 for 3D elastic, 20000 for 3D
  electrostatic.

**Geometries.**

- stiff/soft: one centred square/cube inclusion of side N/4 (6.25 % / 1.6 %
  volume), with modulus c× or 1/c× the matrix.
- random: a Gaussian-smoothed noise field thresholded to a 30 % stiff phase
  (σ = 2 px in 2D, 1.5 px in 3D).
- C(r) = C_cubic(275, 179, 54.3) × profile. P is a smooth random field with
  std 0.2.

**References.** "default" is the setup's own reference:

- elastic C0 = ⟨C⟩;
- 2D electrostatic κ0 = (min+max)/2, which is SPEC's 2D default and so equals
  the midpoint;
- 3D electrostatic κ0 = mean.

"mid" is the min/max midpoint, which is the best reference for basic.

**Labels.**

- "diverges (raises)" is the elastic `_spm_convergence_check` RuntimeError.
- "diverges" for the electrostatic basic is a blow-up. At its default cap of
  200 it prints the non-convergence warning and returns a finite but wrong
  E_d. With a larger cap it overflows, and `resid = num/denom if denom > 0
  else 0.0` then reports resid = 0 for a NaN field (§5).

### 4.1 Elastic, 2D 64²

| geometry | contrast | basic C0=<C> | basic C0=mid | cg | cg noprec |
|---|---|---|---|---|---|
| stiff | 2 | 366 / 1.35 s / 1e-10 | 53 / 0.25 s / 5e-11 | 16 / 0.02 s / 5e-11 | 28 / 0.03 s / 1e-10 |
| stiff | 5 | diverges (raises) | 194 / 0.70 s / 1e-10 | 26 / 0.03 s / 8e-11 | 40 / 0.04 s / 8e-11 |
| stiff | 10 | diverges (raises) | 522 / 2.10 s / 7e-11 | 35 / 0.08 s / 1e-10 | 55 / 0.15 s / 1e-10 |
| stiff | 100 | diverges (raises) | 18855 / 67.66 s / 3e-10 | 100 / 0.20 s / 8e-11 | 150 / 0.32 s / 1e-10 |
| stiff | 1000 | diverges (raises) | not converged in 40000 | 250 / 0.46 s / 2e-10 | 380 / 0.85 s / 3e-10 |
| soft | 2 | 74 / 0.17 s / 1e-10 | 52 / 0.14 s / 1e-10 | 15 / 0.02 s / 1e-10 | 28 / 0.03 s / 5e-11 |
| soft | 5 | 284 / 0.84 s / 1e-10 | 179 / 0.49 s / 1e-10 | 24 / 0.04 s / 2e-10 | 39 / 0.13 s / 1e-10 |
| soft | 10 | 676 / 1.86 s / 2e-10 | 403 / 1.23 s / 1e-10 | 33 / 0.04 s / 2e-10 | 54 / 0.15 s / 1e-10 |
| soft | 100 | 7478 / 15.20 s / 2e-10 | 4329 / 6.38 s / 1e-10 | 91 / 0.12 s / 4e-10 | 142 / 0.33 s / 3e-10 |
| soft | 1000 | not converged in 40000 | 35156 / 40.38 s / 7e-10 | 239 / 0.38 s / 7e-10 | 343 / 0.62 s / 2e-09 |
| random | 2 | 131 / 0.14 s / 3e-11 | 79 / 0.08 s / 2e-11 | 16 / 0.02 s / 8e-11 | 29 / 0.03 s / 1e-10 |
| random | 5 | diverges (raises) | 385 / 0.37 s / 1e-10 | 27 / 0.05 s / 4e-11 | 43 / 0.10 s / 1e-10 |
| random | 10 | diverges (raises) | 1214 / 1.22 s / 3e-10 | 37 / 0.04 s / 6e-11 | 59 / 0.08 s / 9e-11 |
| random | 100 | diverges (raises) | not converged in 40000 | 106 / 0.11 s / 1e-10 | 170 / 0.28 s / 1e-10 |
| random | 1000 | diverges (raises) | not converged in 40000 | 313 / 0.42 s / 2e-10 | 494 / 0.81 s / 2e-10 |

### 4.2 Elastic, 3D 24³

| geometry | contrast | basic C0=<C> | basic C0=mid | cg | cg noprec |
|---|---|---|---|---|---|
| stiff | 2 | 1093 / 14.59 s / 2e-10 | 42 / 0.53 s / 1e-10 | 16 / 0.06 s / 6e-11 | 28 / 0.14 s / 1e-10 |
| stiff | 5 | diverges (raises) | 133 / 2.27 s / 2e-10 | 26 / 0.25 s / 2e-10 | 42 / 0.26 s / 2e-10 |
| stiff | 10 | diverges (raises) | 316 / 4.95 s / 2e-10 | 37 / 0.13 s / 1e-10 | 59 / 0.36 s / 1e-10 |
| stiff | 100 | diverges (raises) | not converged in 6000 | 105 / 0.49 s / 3e-10 | 168 / 0.88 s / 3e-10 |
| stiff | 1000 | diverges (raises) | not converged in 6000 | 289 / 1.78 s / 9e-10 | 429 / 2.03 s / 1e-09 |
| soft | 2 | 60 / 0.58 s / 7e-11 | 41 / 0.48 s / 4e-11 | 15 / 0.05 s / 2e-10 | 27 / 0.08 s / 2e-10 |
| soft | 5 | 202 / 1.21 s / 1e-10 | 123 / 0.57 s / 3e-11 | 25 / 0.08 s / 8e-11 | 39 / 0.12 s / 9e-11 |
| soft | 10 | 440 / 2.22 s / 1e-10 | 252 / 1.38 s / 8e-11 | 34 / 0.11 s / 1e-10 | 50 / 0.20 s / 2e-10 |
| soft | 100 | 4311 / 21.26 s / 3e-10 | 2385 / 7.68 s / 1e-10 | 93 / 0.31 s / 3e-10 | 141 / 0.45 s / 4e-10 |
| soft | 1000 | not converged in 6000 | not converged in 6000 | 256 / 0.82 s / 8e-10 | 354 / 1.12 s / 7e-10 |
| random | 2 | 111 / 0.47 s / 1e-10 | 69 / 0.27 s / 1e-10 | 17 / 0.09 s / 2e-11 | 29 / 0.18 s / 4e-10 |
| random | 5 | diverges (raises) | 291 / 0.90 s / 2e-10 | 27 / 0.10 s / 8e-11 | 44 / 0.17 s / 6e-10 |
| random | 10 | diverges (raises) | 868 / 2.57 s / 4e-10 | 38 / 0.12 s / 9e-11 | 60 / 0.21 s / 2e-10 |
| random | 100 | diverges (raises) | not converged in 6000 | 110 / 0.35 s / 2e-10 | 175 / 0.70 s / 2e-10 |
| random | 1000 | diverges (raises) | not converged in 6000 | 327 / 1.25 s / 5e-10 | 519 / 2.10 s / 5e-10 |

### 4.3 Electrostatic, 2D 64²

| geometry | contrast | basic k0=mean | basic k0=mid | cg k0=default |
|---|---|---|---|---|
| stiff | 2 | 183 / 0.26 s / 4e-11 | 22 / 0.03 s / 3e-11 | 15 / 0.04 s / 2e-10 |
| stiff | 5 | diverges | 55 / 0.08 s / 2e-10 | 25 / 0.06 s / 1e-10 |
| stiff | 10 | diverges | 108 / 0.10 s / 4e-10 | 35 / 0.05 s / 1e-10 |
| stiff | 100 | diverges | 958 / 0.94 s / 5e-09 | 105 / 0.19 s / 6e-11 |
| stiff | 1000 | diverges | 8395 / 9.47 s / 5e-08 | 296 / 0.42 s / 9e-11 |
| soft | 2 | 32 / 0.03 s / 8e-11 | 22 / 0.02 s / 3e-11 | 15 / 0.02 s / 3e-10 |
| soft | 5 | 93 / 0.11 s / 3e-10 | 56 / 0.07 s / 2e-10 | 25 / 0.05 s / 2e-10 |
| soft | 10 | 188 / 0.24 s / 8e-10 | 109 / 0.18 s / 3e-10 | 34 / 0.08 s / 3e-10 |
| soft | 100 | 1718 / 1.86 s / 9e-09 | 965 / 0.97 s / 4e-09 | 93 / 0.19 s / 2e-10 |
| soft | 1000 | 15294 / 18.11 s / 9e-08 | 8540 / 9.77 s / 4e-08 | 220 / 0.41 s / 6e-11 |
| random | 2 | 38 / 0.07 s / 3e-11 | 22 / 0.04 s / 4e-11 | 16 / 0.04 s / 2e-10 |
| random | 5 | diverges | 56 / 0.10 s / 2e-10 | 27 / 0.08 s / 1e-10 |
| random | 10 | diverges | 108 / 0.17 s / 4e-10 | 38 / 0.08 s / 7e-11 |
| random | 100 | diverges | 956 / 1.29 s / 5e-09 | 117 / 0.25 s / 3e-11 |
| random | 1000 | diverges | 8425 / 6.74 s / 5e-08 | 366 / 0.54 s / 2e-11 |

### 4.4 Electrostatic, 3D 24³

| geometry | contrast | basic k0=mean | basic k0=mid | cg k0=default |
|---|---|---|---|---|
| stiff | 2 | 680 / 3.35 s / 5e-11 | 22 / 0.09 s / 4e-11 | 15 / 0.08 s / 2e-10 |
| stiff | 5 | diverges | 56 / 0.37 s / 1e-10 | 25 / 0.20 s / 1e-10 |
| stiff | 10 | diverges | 108 / 0.71 s / 4e-10 | 34 / 0.27 s / 2e-10 |
| stiff | 100 | diverges | 953 / 4.76 s / 5e-09 | 99 / 0.62 s / 6e-11 |
| stiff | 1000 | diverges | 8388 / 41.50 s / 5e-08 | 272 / 1.32 s / 5e-11 |
| soft | 2 | 32 / 0.10 s / 9e-11 | 23 / 0.11 s / 1e-11 | 15 / 0.08 s / 3e-10 |
| soft | 5 | 94 / 0.18 s / 4e-10 | 58 / 0.14 s / 4e-11 | 24 / 0.08 s / 7e-10 |
| soft | 10 | 193 / 0.71 s / 8e-10 | 115 / 0.39 s / 7e-11 | 33 / 0.13 s / 1e-09 |
| soft | 100 | 1778 / 4.75 s / 1e-08 | 1033 / 4.41 s / 8e-10 | 89 / 0.62 s / 4e-10 |
| soft | 1000 | 15756 / 69.30 s / 1e-07 | 9299 / 41.11 s / 7e-09 | 182 / 0.72 s / 1e-09 |
| random | 2 | 37 / 0.06 s / 3e-11 | 22 / 0.06 s / 4e-11 | 17 / 0.06 s / 3e-11 |
| random | 5 | diverges | 56 / 0.21 s / 1e-10 | 28 / 0.10 s / 3e-11 |
| random | 10 | diverges | 108 / 0.29 s / 4e-10 | 39 / 0.19 s / 4e-11 |
| random | 100 | diverges | 958 / 2.75 s / 5e-09 | 120 / 0.62 s / 3e-11 |
| random | 1000 | diverges | 8381 / 20.62 s / 5e-08 | 376 / 0.95 s / 2e-11 |

### 4.5 Inside a stepping loop (warm start)

These are 10 consecutive elastic solves on 2D 64², random two-phase, with P
drifting by 2e-3·noise per call and the warm start on (the default). They
show the per-solve cost in a time stepper:

| contrast | method | Green applies / solve | ms / solve |
|---|---|---|---|
| 10 | basic C0=mid | 664 | 592 |
| 10 | cg | 31 | 39 |
| 100 | basic C0=mid | 24215 | 24640 |
| 100 | cg | 87 | 96 |

The warm start saves CG relatively little: 31 against 37 iterations cold at
contrast 10. Its tolerance is relative to the right-hand side, so a drifted P
still needs about 7 of the 10 decades. Basic's cost per solve stays dominated
by its outer ε̄ loop.

### 4.6 Reading the numbers

**Basic.**

- With the default reference, basic diverges for every stiff-inclusion and
  random case from contrast 5 on, in both elastic and electrostatic. Soft
  inclusions keep max C ≤ 2·⟨C⟩, so they converge, only slowly.
- With the midpoint reference, its applies grow ∝ contrast: ≈ 8400–9300 core
  solves at κ contrast 1000.
- For the elastic stress-free case it is much worse, because the outer ε̄ loop
  multiplies the inner count: 18855 applies at stiff contrast 100 in 2D, and
  more than the 40000 (2D) / 6000 (3D) budget for the random and 3D stiff
  cases from contrast 100 on.

**CG.**

- CG needs 15–40 applies at contrast 2–10, 90–120 at 100, and 180–380 at 1000
  (≈ 10·√c).
- Compared with the best basic in applies:
  - contrast 2: 1.3–1.5× fewer (electrostatic), 2.5–5× fewer (elastic);
  - contrast 10: ≈ 3× fewer (electrostatic), 7–33× fewer (elastic);
  - contrast 1000: 20–50× fewer for κ, and > 100× for elastic
    (35156 → 239, 2D soft).
- Wall time follows the applies: one CG iteration costs about one basic
  iteration plus two inner products.

**Preconditioner.** For the elastic case, the C0 = ⟨C⟩ preconditioner saves
≈ 40 % of the iterations over plain Galerkin CG. The cubic anisotropy of this
C alone gives an eigenvalue spread of ≈ 6.6 between the bulk mode
(C11+2C12 = 633) and the tetragonal mode (C11−C12 = 96), and the
preconditioner removes it.

## 5. Limitations and findings

**Scope.**

- Opt-in only; no default changes.
- The 2D elastic CG is stress_free only, the same scope as basic 2D. The
  film-misfit/eps_m modes are homogeneous-C and do not use the SPM at all.
- Not covered: `film_elastic.setup_film_elastic_spm`, the film-lane elastic
  SPM, which is a separate fixed point in film_elastic.py and outside this
  change.
- Not covered: the higher-level builders, which must opt in per §1.

**Film definiteness.** The film CG needs the *discrete* film operator to be
positive definite.

- That holds for a z-uniform reference. Measured cases (16²×9 and 12²×7,
  lateral inclusion, contrast 10–1000) match basic to ≤ 3e-10, and converge
  where basic at the in-plane-mean reference blows up.
- It does **not** hold in general next to a dead or graded layer at high
  contrast. There the nodal δκ meets the harmonic face average of κ0(z), and
  the effective coefficient goes negative.
- In those cases basic diverges or stalls as well (16²×9, dead layer,
  contrast 10, midpoint reference: resid 6e-2 after 5000 iterations).
- CG raises a RuntimeError at the first non-positive curvature rather than
  returning garbage.

**Round-off floor.** The attainable residual is ≈ eps × contrast.
`tol` = 1e-13 at κ contrast 1000 stalls at ≈ 3e-13 and runs to the cap
(printing the warning). Production tolerances of 1e-10 are far from it.

**Memory.** Elastic CG keeps about 6 strain-sized fields (ε, s, z, p, q and the
eigenstrain) against basic's ≈ 4. Electrostatic CG keeps 6 vector fields.

**Findings about the pre-existing basic paths.** These are not changed here,
because the defaults must stay bit-identical.

1. **Even-grid κ0 dependence (2D, 3D electrostatic).**
   - D(k) = k k/k² is built from raw wavevectors, and only D12/D13/D23 are
     zeroed on even-axis Nyquist lines. There D is a diagonal remnant, not a
     projector.
   - The converged E_d of the basic SPM therefore depends on κ0. Measured:
     5 % (max-norm) for white-noise P on 12³, 0.5 % for a smooth P with a
     sharp inclusion, 1 % on 16² in 2D. On odd grids it is 1e-15.
   - CG reproduces basic at the *same* κ0 to 1e-14, so it solves the same
     discrete problem, but the dependence remains. So 2D (κ0 = midpoint) and
     3D (κ0 = mean) answers differ on even grids, and `kappa0_override`
     changes the answer, not just the speed.
   - The elastic Green operator uses the modified wavevectors in Ω and K
     alike, is a true projector, and is C0-independent.
2. **A blown-up electrostatic basic SPM can report resid = 0.**
   - `resid = num/denom if denom > 0 else 0.0` evaluates to 0.0 when the field
     is NaN, so a diverged run with a large `max_iter` returns NaN as
     "converged".
   - This applies to `solver._spm_iterate`, `bulk3d._spm_iterate_3d` and
     `film3d_spm.solve_poisson_spm`.
   - At the default cap of 200 the field is still finite garbage, and the
     printed non-convergence warning is the only signal.
3. **Uniform P with an inhomogeneous C(r) (stress_free basic elastic).**
   - The exact answer is σ = 0, but basic's outer test max|⟨σ⟩|/max|σ|
     compares two round-off-level numbers. It runs to `outer_max_iter` = 50
     and warns "did NOT converge" (observed ratio 0.4–0.9).
   - CG treats a round-off-level ⟨σ⟩ as converged and returns in 0
     iterations.
