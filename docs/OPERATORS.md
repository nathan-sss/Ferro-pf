# The shared operator layer (`solvers/operators.py`)

The three engines are `solver.py` (2D periodic), `bulk3d.py` (3D periodic) and
`film3d.py` (in-plane rfft2 ⊗ z finite differences). Each keeps its own transforms,
steppers, energies and physics terms. Since this refactor, the k-space *symbols* that they
and the sector modules multiply by are built in one module, `solvers/operators.py`. The
Nyquist masks and the E_ext / static-eigenstrain normalisers live there too.

Before the refactor these were private copies, and the copies drifted apart:

- `film3d._grad_A2d` once failed to mask A12 on even Nyquist lines, while `solver.py`
  masked it.
- The ensemble normaliser had to be patched in five places.

The layer is a leaf module. It uses numpy only, imports no engine and runs no FFT, so each
engine keeps its own `FFT_WORKERS` knob. Each function reproduces, operation for
operation, the code it replaced. Routing an engine through the layer changed no number
bit for bit (checked on a 25,935-entry snapshot of ops, symbols, steps and energies).

## What lives where

| Shared function (`operators.*`) | Replaces / used by |
|---|---|
| `k_axis`, `k_axis_half`, `k_mesh` | wavevector construction in `solver.make_grid` and `bulk3d.make_grid_3d` (the film reuses `solver.make_grid`) |
| `grad_symbols_2d` / `_3d` (via `grad_symbols`) | `solver._grad_symbols` (alias), `bulk3d._grad_symbols_3d` (alias), the body of `film3d._grad_A2d`; called by `solver.build_operators` and `bulk3d.build_operators_3d` |
| `depolarization_symbols_2d` / `_3d` | D(k) inline in `build_operators` / `build_operators_3d` |
| `masked_wavevectors_2d` / `_3d` | `solver._modified_wavevectors_2d`, `elastic._modified_wavevectors_2d`, `bulk3d._modified_wavevectors_3d` (all aliases); the local-gradient energy densities of P, q and θ |
| `derivative_symbols_2d` | `film3d._odd_k` (alias): the Poisson source and recovery, the anisotropic face terms, and `film_elastic` |
| `green_operator(grid, C, dim)` | `elastic.build_Omega`, `bulk3d.build_Omega_3d`, and the zero_k of `setup_elastic` / `setup_elastic_spm` |
| `k_squared`, `isotropic_symbol` | film scalar-G force, energy and factors; Poisson bands; OU noise cutoffs; `kittel_afe` Aq and `octa_tilt` Aθ (2D and 3D) |
| `hog_symbol`, `add_hog` | `grad4.build_operators_hog[_3d]`, `film3d._grad_A2d(H11)`, `hog_force_film`, `build_grad_factors[_sbdf2]` |
| `nyquist_project_axis`, `add_nyquist_correction`, `grad_nyquist_coefficients` | `solver._nyquist_project_axis` (alias); the even-axis energy correction in `solver`, `bulk3d`, `kittel_afe` and `octa_tilt` (2D and 3D) |
| `vec3_field` | `solver._reshape_vec3`, `bulk3d._reshape_vec3_3d` (both aliases), used by every periodic E_ext caller |
| `ensemble_field` | `elastic._ensemble_field` (alias); the eps_star_static alignment in `elastic`, `bulk3d` and `bulk3d_spm` |
| `vec3_field_checked` | the film's stricter E_ext rule (was inline in `film3d.solve_poisson_factored`) |

Every old private name is still importable. It is either an alias of the shared function
(`solver._grad_symbols is operators.grad_symbols_2d`) or a thin wrapper that calls the
shared function (`film3d._grad_A2d`, `elastic.build_Omega`, `bulk3d.build_Omega_3d`).
No public signature changed.

## Conventions

These are stated once, in the module docstring. In short:

**Layout.** Real fields are component-first with space last: `(3, [S,] [Nz,] Ny, Nx)`.
Spectral arrays are the rfft half-spectrum: x is the last axis and is halved to `Nxr = Nx//2+1`
bins (`rfftfreq`, kx ≥ 0). y and z keep their full length in `fftfreq` order. The grid
carries `KXr, KYr[, KZr]` already broadcast to `(Ny, Nxr)` or `(Nz, Ny, Nxr)`. `dim` is the
number of Fourier axes: 2 for `solver` and for the film (whose z is a node grid), 3 for
`bulk3d`.

**Nyquist rule.** On an even axis of N points, the bin at index `N//2` is the single
self-conjugate wavenumber ±π/h. This holds in both the full and the half layout. A symbol
that is *odd* in k_j is set to zero on that line or plane whenever N_j is even. This applies
to:

- the first-derivative wavevectors, each on its own axis;
- the mixed A_ij and D_ij (i ≠ j), on both of their axes;
- the film's i·k terms.

*Even* symbols are never masked: A_ii, D_ii, |k|², Σk_j⁴ and −k_j². Odd axes have no
Nyquist bin, so odd grids never depend on the rule. On the energy side,
`add_nyquist_correction` restores the alternating mode that a collocation density built
from masked derivatives drops. With it, the grid mean equals the quadratic form whose
gradient is the k² force. In films, the q and θ gradient energies use the same
in-plane quadratic form and thickness-direction face differences as the polarization.

**Shear convention.** Stiffness is always the cubic Voigt triple. It is expanded with
C_1212 = C44, so σ12 = 2·C44·e12 on the tensor strain. The Green operator takes the full
tensor. The eigenstrain is *not* shared, deliberately:

- `elastic` and `film_elastic` use the Voigt-style Q44: e12 = ½·Q44·P1·P2.
- `bulk3d` uses the tensor Q1212: e12 = 2·Q1212·P1·P2.
- So Q44 = 4·Q1212 (SPEC §37.5).

**Ensemble axis.** S sits right after the component axes. A per-realisation field
(`(3,…)` or `(3,3,…)`) that meets a batched quantity gets S inserted. A field that already
carries S is kept. Any other shape is refused. The film has no ensemble axis, and its
`vec3_field_checked` validates a field-shaped E_ext up front.

## What is deliberately not shared

- **Transforms** (`_rfft2`, `_rfftn`, `_spectral_grad_xy[z]`, grad4's second partials).
  Each engine owns a different `FFT_WORKERS` knob (bulk3d uses 1 on purpose). The FFT
  execution layer is not part of this module either.
- **The implicit operators** (`build_Minv*`, `build_grad_factors*`, `_lin_matrix`) and the
  steppers. They *read* the shared symbols but assemble them per engine.
- **Eigenstrain and its force** (`elastic.eigenstrain` vs `bulk3d.eigenstrain_3d`). These
  use different Q conventions on purpose (see above), and they carry the `cpu_kernels` hooks.
- **The film's z operators** (Neumann bands, Poisson bands, face element). They are finite
  differences, not symbols.
- **The real-space random-field generators** (`disorder`, `composite`, `quenched_fields`,
  `random_smooth_field*`). These are setup-time IC tools on the full spectrum, and they
  deliberately import no engine.

## Adding a symbol

1. Write it once in `operators.py`, generic in `dim` where the math is. Give it thin
   `_2d` / `_3d` wrappers if the engines want tuples in their own order. Build it from
   `half_spectrum(grid, dim)`. If it is odd in some k_j, pass it through
   `mask_nyquist(sym, grid, axes)` with exactly those axes. Never mask an even symbol.
2. Call it from each engine as `operators.name(...)`. Use an attribute lookup at call
   time, not `from solvers.operators import name`, so the delegation guard can see the call.
   If an engine needs the old private name, alias it: `_old = operators.name`.
3. Extend `tests/test_operators.py`:
   - add the symbol to the Nyquist property test, against an independent raw formula;
   - add it to the cross-solver test (bulk3d at Nz = 1 against 2D, film in-plane against 2D);
   - add it to the delegation guard (an identity alias, or a spy whose output must be the
     very object the engine returns).
4. If the symbol replaces existing code, prove bit-identity: snapshot the affected
   ops, steps and energies before and after on even, odd and mixed grids, and compare bytes.
