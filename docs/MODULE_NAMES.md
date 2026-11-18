# Module names: `fftpf` and the permanent `solvers` aliases

Ruling 2026-11-11 (user): the engine lives in the package `fftpf/` under descriptive module
names. The old package `solvers/` is kept whole as forwarding aliases.

## Naming rule

| prefix | meaning |
|---|---|
| `pf2d_` | 2D-periodic only |
| `pf3d_` | 3D-periodic only |
| `film_` | thin-film geometry only |
| `mat_` | material-specific parameters |
| none | shared by the geometries |

A module name must say what the module does. Python identifiers cannot start with a digit, hence
`pf2d_` rather than `2d_`.

## Forwarding policy

Every `solvers/<old>.py` is a three-line file that sets `sys.modules[__name__]` to the
`fftpf` module, so `solvers.<old>` **is** `fftpf.<new>` (same object, not a copy): module
state such as `FFT_WORKERS` or the backend hooks `configure` sets is shared. Old imports keep
working unchanged, and the aliases are permanent. New code imports `fftpf.*`; `fftpf` never
imports `solvers`. `python solvers/x.py` no longer runs the module's `__main__` block: use
`python fftpf/<new>.py`. `tests/test_package_forwarding.py` pins the table below.

## Backends

Three geometry engines (`pf2d_engine`, `pf3d_engine`, `film_engine`) times three backends,
selected by `backend_config.configure`: the NumPy reference; Numba per-term kernels
(`kernels_per_term`); and the fused single-pass step (`kernels_fused`, opt-in with
`fused=True`).

## Old -> new

| old (`solvers`) | new (`fftpf`) | what it does |
|---|---|---|
| `solver` | `pf2d_engine` | 2D periodic semi-implicit TDGL solver (Landau bulk, gradient, electrostatics, Langevin noise) |
| `energy_protocol` | `pf2d_energy_storage_loop` | 2D unipolar energy-storage protocol (E: 0 -> Emax -> 0, W_store / W_rec) |
| `susceptibility` | `dielectric_spectrum` | dynamic susceptibility chi(omega, T) from Langevin/FDT trajectories (estimators on a time series; was `pf2d_dynamic_susceptibility`, which stays as a forwarding alias in `fftpf`) |
| `bulk3d` | `pf3d_engine` | 3D fully periodic (rfftn) engine, same stepping contract as pf2d_engine |
| `bulk3d_spm` | `pf3d_elastic_inhomogeneous` | Full-tensor inhomogeneous C(r) elasticity (elastic SPM) for pf3d_engine |
| `film3d` | `film_engine` | 3D thin-film engine: in-plane FFT x thickness finite difference, short-circuit electrodes |
| `film3d_langevin` | `film_langevin` | Exact per-mode OU/Langevin stepper for the film geometry |
| `film3d_spm` | `film_electrostatic_inhomogeneous` | kappa(x,y,z) electrostatic SPM over the film banded solve |
| `film_elastic` | `film_elastic` | Film + substrate elasticity: per-mode block-tridiagonal finite-difference solve |
| `film_elastic_fast` | `film_elastic_force_numpy` | Optional force-only direct film elasticity (NumPy slab pool) |
| `film_elastic_numba` | `film_elastic_force_numba` | Optional Numba direct film elastic force kernels |
| `elastic` | `elastic_base` | Homogeneous-C Khachaturyan elasticity (eigenstrain, strain, stress, energy) |
| `grad4` | `gradient4` | Higher-order gradient (H11) sector for the 2D engine |
| `kittel_afe` | `antiferroelectric_order` | Antiferroelectric order parameter q (P, q) sector and its loop observables |
| `octa_tilt` | `octahedral_tilt_order` | Octahedral tilt order parameter theta sector |
| `polycrystal` | `polycrystal` | 2D periodic polycrystal layer (Voronoi grains, grain-boundary shell) |
| `disorder` | `disorder_fields` | Spatial Landau disorder and static eigenstrain disorder generators |
| `composite` | `composite_labels` | Static composite phase-label field (host / filler) from the disorder machinery |
| `execution` | `backend_config` | CPU workers and backend selection (configure: numpy / numba / fused) |
| `cpu_kernels` | `kernels_per_term` | Numba per-term CPU kernels (Landau force etc.) |
| `cpu_fused` | `kernels_fused` | Numba fused single-pass step kernels for all three geometries |
| `operators` | `spectral_operators` | Shared k-space symbols, Nyquist rule and E_ext normalisers |
| `timestep` | `timestep_verify` | Unified dt verification and audit trail for deterministic runs |
| `adaptive` | `timestep_adaptive` | Adaptive time stepping and linear stabilization (stab=) |
| `minimize` | `energy_minimize` | Direct free-energy minimization (L-BFGS / FIRE) for the deterministic engines |
| `validate` | `input_validation` | Setup-time validation of decks, fields and arguments |
| `drive` | `aging_hold_protocol` | Polarization / defect-aging hold protocol |
| `defects` | `defect_bias_field` | Prescribed defect bias field (energy -b.P via E_ext) |
| `crystallinity` | `crystallinity_landau` | Prescribed crystallinity term f_X = A (1 - Xc) |P|^2 / 2 |
| `quenched_fields` | `random_field_mechanisms` | Quenched random-field disorder mechanisms |
| `composition` | `composition_field` | Spatial composition x(r) geometry and mixing helpers |
| `landau0d` | `landau_0d_diagnostics` | 0-D homogeneous Landau energy, gradient and Hessian diagnostics |
| `loopfit` | `afe_loop_landau_fit` | Fit of single-axis Landau coefficients to antiferroelectric loop features |
| `wall_metrics` | `wall_metrics` | Domain-wall energy and position metrics |
| `io_utils` | `state_io` | State npz save/load, quick-look PNG, provenance stamp |
| `deck_parser` | `deck_reader` | Ferret .i deck extractor and the runtime TOML deck reader |
| `paths` | `ferret_paths` | Location of the Ferret source repo |
| `casekit` | `casekit` | Case-writing helpers (single_thread, build_2d/3d, anneal, run_dir, save_run) |
| `units` | `units` | SI to deck-unit conversion |
| `materials` | `mat_registry` | Material coefficient registry (Landau, gradient, elastic, electrostriction) |
| `materials_archive` | `mat_registry_archive` | Archived registry entries no computation calls |
| `literature_params` | `mat_chen2007_decks` | Parameter sets from Rabe/Ahn/Triscone 2007 Appendix A (Chen) |
| `pmn_field` | `mat_pmn_charge_disorder` | PMN heterovalent chemical-disorder statistics |
| `pzt_field` | `mat_pzt_coefficient_field` | PZT Haun coefficient fields from a composition field |
| `pan2019_field` | `mat_pan2019_landau` | Pan 2019 BFO/BTO/STO Landau coefficient generators |
| `su_landis` | `mat_bto_su_landis_landau` | Su-Landis BaTiO3 Landau coefficient set |
| `afd` | `octahedral_tilt_order` | legacy shim name (WP24, SPEC S35), now an alias of `octahedral_tilt_order` |
| `afe` | `antiferroelectric_order` | legacy shim name (WP24, SPEC S35), now an alias of `antiferroelectric_order` |
| `hog` | `gradient4` | legacy shim name (WP24, SPEC S35), now an alias of `gradient4` |
| `dynsus` | `dielectric_spectrum` | legacy shim name (WP24, SPEC S35), now an alias of `dielectric_spectrum` |
| `wall_utils` | `wall_metrics` | legacy shim name (WP24, SPEC S35), now an alias of `wall_metrics` |
| `walls` | `wall_metrics` | legacy shim name (WP24, SPEC S35), now an alias of `wall_metrics` |

## Input-file entry point (RUN-ENTRY, SPEC § 16)

New in 2026-11, without an old `solvers` name (so there is no alias): flat modules with **no geometry prefix**, because they serve all three
geometries. They import the engines; no engine imports them. Manual: `docs/RUN_INPUT.md`.

| module | what it does |
|---|---|
| `__main__` | `python -m fftpf run \| check \| explain case.toml` |
| `run_input` | TOML schema, defaults, units in key names, cross-key refusals, `resolved.toml` writer |
| `run_builder` | resolved input -> deck, grid, operators, elastic setup, quenched fields, backend, adapter (no step taken) |
| `run_adapter` | one calling convention (`build`, `step`, `mean_P`, `energy`, `random_start`) over `pf2d_engine`, `pf3d_engine`, `film_engine` |
| `run_protocol` | the stages `relax`, `anneal`, `cool`, `hold`, `pe_loop`, `ac`, `fdt`, the run loop and replicas |
| `run_measure` | readings: mean P, energy terms, loop measures, phase fractions, dielectric-spectrum estimators and tables |
| `run_output` | the run directory: trace, states, checkpoints, `measures.json`, `spectrum.csv` |
| `run_figures` | domain maps (direction, phase) and the spectrum figure |
