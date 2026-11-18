# `fftpf/` API index

GENERATED FILE -- do not hand-edit. Regenerate after any signature or
docstring change:

```bash
python3 tools/api_index.py
```

Every public (non-`_`) module-level function of `fftpf/`, in source order
so this doubles as a table of contents for each file. The description is the
function's own first docstring line, read by AST -- a trailing ` ...` means
that line was cut by the source's own ~72-column wrap, so open the function
for the rest. If a description reads wrong, fix the docstring in `fftpf/`
and rerun, never this file. Prose, decks, traps and worked examples stay in
`docs/ENGINE_GUIDE.md`, and what each script-facing parameter MEANS (physics,
units, literature range) in `docs/PARAMETERS.md`; this is the lookup table.
The old `solvers.*` names are permanent aliases of these modules: see
`docs/MODULE_NAMES.md` for the old -> new table.

54 canonical modules -- 563 public functions, 157 public constants.

*package-level constants* (`fftpf/__init__.py`, SPEC §V2-01.1(a)): `ENGINE_NAME`, `ENGINE_VERSION`.

## Canonical modules

### `afe_loop_landau_fit.py`

| public name | signature | what it does |
|---|---|---|
| `f_energy` | `(P, a1, a11, a111)` | 0-D sextic Landau energy density f(P) = a1*P^2 + a11*P^4 + a111*P^6, deck units. |
| `e_field` | `(P, a1, a11, a111)` | E(P) = f'(P) -- the equilibrium field at polarization P. |
| `chi_afe` | `(a1)` | Small-signal susceptibility at P=0, E=0: 1/f''(0) = 1/(2*a1). |
| `spinodal_points` | `(a1, a11, a111)` | Real, positive roots (P1<P2) of dE/dP=0 (u=P^2 quadratic ... |
| `thresholds` | `(a1, a11, a111)` | (E_F, E_A) at the two spinodal turning points, or None if the deck has ... |
| `p_max_at_field` | `(a1, a11, a111, E_max)` | Largest real root of E(P)=E_max (quintic, companion-matrix via ... |
| `check_global_min_constraint` | `(a1, a11, a111, label='')` | SPEC S27.1 R1/R5: f(sqrt(u2)) > 0 = f(0), i.e. P=0 stays the GLOBAL ... |
| `fit` | `(E_F, E_A, P_max, E_max, chi_AFE=None, seeds=None, tol=1e-13, verbose=True)` | Deterministic constrained fit: targets (E_F, E_A, P_max at E_max[, ... |

### `aging_hold_protocol.py`

No public function.

### `antiferroelectric_order.py`

| public name | signature | what it does |
|---|---|---|
| `pbzo_kittel_landau_si` | `(T, table1=None)` | SI dict: P-sector and q-sector (both engine ALPHA_KEYS naming, ... |
| `load_pbzo_deck` | `(path=None)` | Load params/pbzo_kittel_lum2022.toml, remapping its collision-safe ... |
| `couple_energy_density` | `(P, q, coupling)` | f_couple(P,q) from coupling's t1/t2/u1/u2 keys (SPEC §23.1), real space, shape = P[0].shape. |
| `couple_force_P` | `(P, q, coupling)` | delta f_couple/delta P_i, shape (3,Ny,Nx). |
| `couple_force_q` | `(P, q, coupling)` | delta f_couple/delta q_i, shape (3,Ny,Nx). |
| `build_ops_q` | `(grid, Gq)` | Aq(k) = Gq*(kx^2+ky^2), scalar, SAME for all 3 components -- unlike ... |
| `grad_energy_density_q` | `(q, grid, Gq)` | f_gradq = 0.5*Gq*\|grad q\|^2 summed over the 3 components (isotropic, SPEC §23.1), shape (Ny,Nx). |
| `grad_force_q` | `(q, ops_q)` | d f_gradq / dq = Aq(k) . q (diagonal-only mirror of solver.grad_force). |
| `build_Minv_q` | `(ops_q, dt, Lq)` | M_q(k) = 1 + dt*Lq*Aq(k), scalar per k-point (isotropic/diagonal -- ... |
| `with_q_eigenstrain` | `(base_elastic_params, q, Lam11, Lam12, Lam44)` | Shallow-copy base_elastic_params (from elastic.setup_elastic, P's ... |
| `elastic_force_explicit_q` | `(q, sigma, Lam11, Lam12, Lam44)` | dF_el/dq_m -- exact structural mirror of elastic.elastic_force_explicit ... |
| `step_pq` | `(P, q, alpha_P, alpha_q, coupling, ops, Minv, dt, L, E_ext, ops_q=None, Minv_q=None, Lq=None, base_elastic_params=None, Lam=None)` | One semi-implicit step for (P,q) together. P's implicit operator ... |
| `total_energy_pq` | `(P, q, alpha_P, alpha_q, coupling, ops, grid, Gq, E_ext=(0.0, 0.0, 0.0), elastic_params=None, Lam=None)` | Same shape as solver.total_energy (densities dict, totals dict), ... |
| `build_ops_q_3d` | `(grid, Gq)` | Aq(k) = Gq*(kx^2+ky^2+kz^2), scalar, SAME for all 3 components -- the ... |
| `grad_energy_density_q_3d` | `(q, grid, Gq)` | f_gradq = 0.5*Gq*\|grad q\|^2 summed over the 3 components, shape ... |
| `grad_force_q_3d` | `(q, ops_q)` | d f_gradq / dq = Aq(k) . q, diagonal-only (mirrors bulk3d.grad_force_3d ... |
| `build_Minv_q_3d` | `(ops_q, dt, Lq)` | M_q(k) = 1 + dt*Lq*Aq(k), scalar per k-point (isotropic/diagonal -- no ... |
| `lam_tensor_3d` | `(Lam)` | (Lam1111, Lam1122, Lam1212) for bulk3d.eigenstrain_3d from the deck's ... |
| `with_q_eigenstrain_3d` | `(base_elastic_params, q, Lam)` | Shallow-copy bulk3d.setup_elastic_3d's output with q's eigenstrain in ... |
| `elastic_force_explicit_q_3d` | `(q, sigma, Lam1111, Lam1122, Lam1212)` | dF_el/dq_m = -sigma_ij * d(eps0_ij)/dq_m, FULL TENSOR sum over ... |
| `step_pq_3d` | `(P, q, alpha_P, alpha_q, coupling, ops, Minv, dt, L, E_ext, ops_q=None, Minv_q=None, Lq=None, base_elastic_params=None, Lam=None)` | One semi-implicit step for (P,q) together on bulk3d -- step_pq's 3D ... |
| `total_energy_pq_3d` | `(P, q, alpha_P, alpha_q, coupling, ops, grid, Gq, E_ext=(0.0, 0.0, 0.0), elastic_params=None, Lam=None)` | total_energy_pq's 3D mirror: same (densities, totals) contract and the ... |
| `build_q_factors_film` | `(grid, Gq, dt, Lq)` | M_q = I + dt*Lq*[q's gradient-force operator], factorized once per ... |
| `grad_force_q_film` | `(q, grid, Gq)` | delta F_gradq/delta q_i in film geometry, applied EXPLICITLY (matrix ... |
| `grad_energy_q_film` | `(q, grid, Gq)` | TOTAL F_gradq (not a density) in film geometry, mirroring ... |
| `with_q_eigenstrain_film` | `(base_el, q, Lam)` | Shallow-copy film_elastic.setup_film_elastic's output with q's ... |
| `step_pq_film` | `(P, q, alpha_P, alpha_q, coupling, grid, grad_factors, poisson_factors, dt, L, E_ext=(0.0, 0.0, 0.0), q_factors=None, Lq=None, base_el=None, Lam=None)` | One semi-implicit step for (P,q) together in film geometry -- ... |
| `total_energy_pq_film` | `(P, q, alpha_P, alpha_q, coupling, grid, kappa_b, G11, Gq, poisson_factors=None, E_ext=(0.0, 0.0, 0.0), el=None, Lam=None)` | (densities, totals) in film geometry, keys as total_energy_pq. |

*public constants:* `FFT_WORKERS`, `PBZO_SI_TABLE1`, `PBZO_TOML`, `COUPLE_KEYS`

### `backend_config.py`

| public name | signature | what it does |
|---|---|---|
| `environment` | `(workers)` | Thread limits for a fresh process; FFT and Numba run in separate phases. |
| `set_fft_workers` | `(workers)` |  |
| `close` | `(el)` | Detach and release an optional film backend; reference mode needs none. |
| `configure` | `(workers=1, backend='numpy', el=None, *, elastic_workers=None, fused=False)` | Select reference NumPy or fused Numba kernels and an explicit CPU budget. |

*public constants:* `BLAS_ENV`, `FFT_MODULES`

### `casekit.py`

| public name | signature | what it does |
|---|---|---|
| `single_thread` | `(*modules)` | Set FFT_WORKERS = 1 on each module passed and return the one-line ... |
| `build_2d` | `(deck, Nx, Ny, h)` | 2D periodic grid + operators from a materials.get() deck, cell size h ... |
| `build_3d` | `(deck, N, d, *, elastic=False, k0='stress_free')` | Cubic N^3 fully-periodic grid + operators (+ elastic_params when ... |
| `anneal` | `(engine, P, alpha, ops, grid, *, dt, L=1.0, E_ext=(0.0, 0.0, 0.0), kT=None, noise_cutoff_nm=None, elastic_params=None, seed=0, n_max, sample_every, tol=0.01, n_blocks=4, criterion='legacy', block_size=8, energy_scale=None, polarization_scale=None, residual_tol=1e-06, scheme='sbdf1', restart=None, return_state=False)` | Run `engine` (solver or bulk3d) from P until its stopping rule fires, ... |
| `anneal_adaptive` | `(engine, P, alpha, ops, grid, *, dt0, L=1.0, E_ext=(0.0, 0.0, 0.0), kT=None, noise_cutoff_nm=None, elastic_params=None, stab=0.0, err_tol=0.001, k_min=-8, k_max=6, k_start=0, dt_max=None, grow_after=10, energy_check='sample', energy_rtol=1e-12, n_max, t_end=None, sample_every=10, tol=0.01, n_blocks=4, criterion='legacy', block_size=8, energy_scale=None, polarization_scale=None, residual_tol=1e-06, dPdt_tol=np.inf, rate_tol=np.inf, n_windows=5, restart=None, callback=None)` | anneal() with an ADAPTIVE dt, for DETERMINISTIC relaxation only ... |
| `resume_anneal` | `(path, *, n_max)` | Continue an anneal from a checkpoint written with ... |
| `relax` | `(engine, P, alpha, ops, grid, *, E_ext=(0.0, 0.0, 0.0), elastic_params=None, method='lbfgs', tau=None, gtol=1e-06, ftol=1e-10, max_iter=10000, memory=10, sample_every=10, max_step=None, precondition=True, energy_scale=None, exact_energy=False, callback=None, kT=None, noise_cutoff_nm=None)` | Drive `engine` (solver or bulk3d) from P straight to a STATIONARY ... |
| `relax_film` | `(P, alpha, grid, G, kappa_b, *, E_ext=(0.0, 0.0, 0.0), el=None, H11=None, poisson_factors=None, method='lbfgs', tau=None, gtol=1e-06, ftol=1e-10, max_iter=10000, memory=10, sample_every=10, max_step=None, precondition='depolarization', energy_scale=None, exact_energy=False, callback=None, kT=None)` | casekit.relax for the film (film3d + optional film_elastic `el`), with ... |
| `run_dir` | `(case_name)` | Create and return reports/runs/<YYYYMMDD-HHMM>_<case_name>/ (SPEC ... |
| `save_run` | `(rundir, *, params, P, grid, trace, thread_line, extra=None, state=None, input_paths=())` | Write the standard run-folder set into `rundir`: params.json (params ... |
| `ensemble_initial` | `(engine, grid, seeds, amplitude, kmax_frac=0.3)` | (3,S,*space) initial batch whose member s is exactly the single-member ... |
| `ensemble_alpha` | `(alphas, space_ndim)` | Merge one Landau dict per member (e.g. materials.get(deck, T=T)["landau"] ... |

### `composite_labels.py`

| public name | signature | what it does |
|---|---|---|
| `pe_filler_landau_si` | `(eps_r)` | PE (non-polar) filler: pure linear dielectric. alpha1=5.645e8/eps_r ... |
| `pe_filler_landau_si_corrected` | `(eps_r)` | B1 coordinator-review rider (SPEC S26.0(b), physics-derived, NOT ... |
| `si_to_deck_alpha` | `(alpha_si)` | {alpha_name: SI value} -> {alpha_name: deck value}, uniform units.si_to_deck_landau for every P-power. |
| `make_label_field` | `(grid, x_AN, xi_AN, seed)` | Static binary phase-label field, EXACT target volume fraction x_AN ... |
| `alpha_field` | `(L, host_alpha_deck, filler_alpha_deck)` | alpha_i(r) = host_i*(1-L) + filler_i*L for every ALPHA_KEYS entry. |
| `half_slab_label` | `(grid, axis=1)` | Deterministic (non-random) step-function label field for G24d's ... |
| `circular_label` | `(grid, cx_nm, cy_nm, radius_nm)` | Deterministic (non-random) SINGLE circular inclusion label field -- ... |
| `make_label_field_multi` | `(grid, fractions, xi, seed)` | Static K-ary phase-label field, EXACT target fractions (generalizes ... |
| `alpha_field_multi` | `(labels, material_alpha_decks)` | alpha_i(r) = material_alpha_decks[labels[r]][i] for every ALPHA_KEYS ... |
| `layered_dielectric_closed_form` | `(E_ext, f_filler, chi_host, chi_filler, kappa)` | Independent closed-form derivation (SPEC S30.2 G24d), periodic ... |
| `make_label_field_multi_3d` | `(grid, fractions, xi, seed)` | 3D generalization of make_label_field_multi -- SAME rank-and-cut K-ary ... |

*public constants:* `ALPHA_KEYS`, `FE_FILLER_LANDAU_SI`

### `composition_field.py`

| public name | signature | what it does |
|---|---|---|
| `formula_units_per_cell` | `(hx, hy, a_lattice, hz=None)` | N_f = formula units per coarse-grain cell = cell volume / a_lattice^3 ... |
| `composition_std` | `(x_bar, N_f)` | Random-substitutional-alloy mixing limit: variance of the mean ... |
| `composition_std_correlated` | `(x_bar, xi_x, a_lattice, t_eff=None)` | The mean over a CORRELATION VOLUME, not a grid cell: std(x_bar over ... |
| `composition_field` | `(grid, x_bar, std, a_lattice, xi_x=None, seed=0)` | x(r) = x_bar + a zero-mean fluctuation of the given std, via ... |
| `composition_field_3d` | `(grid, x_bar, std, a_lattice, xi_x=None, seed=0)` | SPEC V2-03 A6. 3D twin of composition_field: the SAME one-lattice- ... |
| `vegard_eigenstrain_field` | `(x, x_bar, dlna_dx)` | eps_chem = dlna_dx*(x-x_bar)*I3 -- isotropic chemical (Vegard) strain ... |
| `mix_decks` | `(fractions, decks, rule='linear')` | N-ary LINEAR mixing of K flat coefficient dicts: out[key] = ... |
| `deck_field` | `(x_field, material_fn, *args, **kwargs)` | Apply an already ARRAY-AWARE material function pointwise over a ... |

*public constants:* `PRESET_PZT`

### `crystallinity_landau.py`

| public name | signature | what it does |
|---|---|---|
| `effective_alpha` | `(alpha, Xc=None, A=0.0, *, shape=None)` | Return a shallow copy with only a1 changed; never mutate input arrays. |
| `energy_density` | `(P, Xc=None, A=0.0)` | Diagnostic f_X on each site, with the component axis reduced. |
| `force` | `(P, Xc=None, A=0.0)` | Positive energy derivative d f_X/dP; TDGL applies its negative. |

### `deck_reader.py`

| public name | signature | what it does |
|---|---|---|
| `parse_materials` | `(text)` | Return {prop_name: float value} merged over all GenericConstantMaterial blocks. |
| `parse_mesh` | `(text)` | Return dict with nx,ny,xmin,xmax,ymin,ymax from GeneratedMeshGenerator. |
| `polar_components_present` | `(text)` | Which of polar_x/polar_y/polar_z are declared as [Variables] blocks. |
| `parse_functions` | `(text)` | Return {function_name: expression_string} for [Functions] ParsedFunction blocks. |
| `parse_polar_ic` | `(text)` | For each polar_x/y/z [Variables] block, return ('constant', value) or ... |
| `parse_deck` | `(path)` | MOOSE .i deck -> dict(path, materials, mesh, polar_present, functions, polar_ic); half (a)'s only external entry point. |
| `materials_to_toml` | `(materials, source_deck_rel, extracted_date)` | Format a materials dict (as returned by parse_materials) as TOML text ... |
| `write_params_toml` | `(deck_path, out_path, ferret_root, extracted_date)` | Extract materials from deck_path and write a SPEC §6.1 TOML file. |
| `load_params_toml` | `(path)` | Read a SPEC §6.1 TOML params file back into {'materials': {...flat...}, 'meta': {...}}. |
| `generate_default_params` | `(ferret_root, out_dir, extracted_date)` | Generator mode entry point (SPEC §6.1): write params/bto_298K.toml and ... |

*public constants:* `TOML_SECTIONS`

### `defect_bias_field.py`

| public name | signature | what it does |
|---|---|---|
| `compose_field` | `(applied=(0.0, 0.0, 0.0), *, quenched=None, bias=None, spatial_shape=None)` | Compose applied, quenched and defect fields for E_ext without double counting. |
| `bias_energy_density` | `(P, bias)` | Return -b.P in deck energy-density units (1e9 J/m3); diagnostic only. |
| `bias_force` | `(P, bias)` | Return energy derivative -b, broadcast to P's component-first shape. |
| `aging_state` | `(P, g, tau, bias=None)` | Initialize an independent bias array, validating g>=0 and tau>0. |
| `aging_self_energy_density` | `(bias, g)` | Return \|b\|^2/(2g), zero on inactive sites, in deck energy density. |
| `aging_energy_density` | `(P, bias, g, tau=np.inf)` | Full local aging energy for diagnostics: self energy minus b.P. |
| `aging_force_bias` | `(P, bias, g, tau=np.inf)` | Derivative with respect to b on g>0: b/g-P. |
| `aging_update` | `(P, bias, g, tau, dt)` | Exact db/dt=(gP-b)/tau update for polarization held fixed over dt. |

### `dielectric_spectrum.py`

| public name | signature | what it does |
|---|---|---|
| `welch_chi_pp` | `(traj, dt, kT, nperseg, window='hann', noverlap=None)` | chi''(omega) = omega*S(omega)/(2kT) (SPEC §19), S(omega) the TWO-SIDED ... |
| `sample_correlation` | `(traj, max_lag)` | C[lag] = mean_t[(traj[t]-mean)*(traj[t+lag]-mean)], lag=0..max_lag-1 ... |
| `fl_chi` | `(omega_array, tau_array, C_array, kT)` | Direct Fourier-Laplace route (see module docstring for derivation): |
| `kk_chi_prime` | `(omega_grid, chi_pp_grid, omega_eval)` | Numerical Kramers-Kronig transform, principal-value quadrature with ... |
| `lockin_chi` | `(traj, dt, omega, E0)` | AC cross-check route: lock-in extraction of (chi', chi'') from a ... |
| `pool_spectrum` | `(omega, values, pool)` | Smooth `values` (e.g. Welch chi'') along the frequency axis with a ... |
| `debye_chi` | `(omega, chi0, tau)` | Analytic single-relaxation-time (Debye) susceptibility, the exact ... |
| `measure_chi_p_floor` | `(C_all, tau_arr, omega_grid, kT, k_sigma=3.0)` | WP13 recovery-session addition (coordinator ruling on G17a, extended ... |
| `contiguous_resolved_range` | `(X_cand, resolvable)` | Largest contiguous resolvable run of `resolvable` (a boolean array, ... |
| `calibrate_fl_chi_bias` | `(tau, kT, dt, max_lag, omega_array, n_seeds=14, n_steps=40000000, n_draws=4, seed0=90000, cache_path=_DEFAULT_BIAS_CACHE)` | WP13 recovery-session addition (coordinator requirement): measures ... |

### `disorder_fields.py`

| public name | signature | what it does |
|---|---|---|
| `piecewise_regions_1d` | `(grid, axis, boundaries, values)` | Piecewise-constant field along one axis (e.g. dielectric layers with ... |
| `random_alpha1` | `(grid, mean, amplitude, kmax_frac=0.3, seed=0)` | Smooth random alpha1(x) field (thin wrapper on solver.random_smooth_field, ... |
| `random_field_gaussian` | `(grid, rms, xi, seed)` | Single zero-mean field with power spectrum ~exp(-k^2 xi^2/4) (real-space ... |
| `random_field_gaussian_3d` | `(grid, rms, xi, seed)` | SPEC V2-03 A6. 3D twin of random_field_gaussian: a single zero-mean ... |
| `make_random_strain` | `(grid, eps_rms, xi, seed, components=('xx', 'yy', 'xy'))` | SPEC §6.2b generator. Returns {comp: (Ny,Nx) array}, one independently- ... |
| `assemble_eps_star_static` | `(grid, comp_dict)` | {comp in xx/yy/zz/xy/xz/yz: (Ny,Nx) or (S,Ny,Nx)} -> full symmetric ... |
| `random_field_gaussian_batch` | `(grid, rms, xi, seeds)` | Batched random_field_gaussian: returns (S,Ny,Nx), slice s independently ... |
| `make_random_strain_batch` | `(grid, eps_rms, xi, seeds, components=('xx', 'yy', 'xy'))` | Batched make_random_strain (WP4 §10e): returns {comp: (S,Ny,Nx)}, with ... |
| `radial_autocorrelation` | `(field, grid, nbins, rmax)` | Radially-averaged (Wiener-Khinchin) autocorrelation, normalized to 1 at r=0. |
| `fit_correlation_length` | `(field, grid, xi_guess, fit_mult=0.75, nbins=30)` | Gaussian fit C(r)=exp(-r^2/xi^2) to the radially-averaged autocorrelation, ... |
| `make_random_strain_master` | `(N_max, Lx, Ly, eps_rms, xi, seed, components=('xx', 'yy', 'xy'))` | Generate the master realization at N_max x N_max (same physical box ... |
| `resample_strain_dict` | `(strain_master, N_new, eps_rms)` | Spectrally resample each component of a master-grid strain dict ... |

*public constants:* `FFT_WORKERS`

### `elastic_base.py`

| public name | signature | what it does |
|---|---|---|
| `build_C_tensor` | `(C11, C12, C44)` | Cubic stiffness C_ijkl built from deck C11/C12/C44 (SPEC §1.4), shape (3,3,3,3). |
| `contract_C` | `(C, e)` | sigma_ij = C_ijkl e_kl, e/sigma shape (3,3,Ny,Nx) or (3,3). |
| `eigenstrain` | `(P, Q11, Q12, Q44)` | eps*_ij(P), SPEC §1.4, in this module's VOIGT-flavoured Q convention: |
| `elastic_force_explicit` | `(P, sigma, Q11, Q12, Q44)` | delta F_el / delta P_m, SPEC §1.4 explicit formula (matches ... |
| `build_Omega` | `(grid, C)` | Omega(k) on the rfft2 half-spectrum grid (WP4 §10b), shape (Ny,Nxr,3,3), ... |
| `setup_elastic` | `(grid, C11, C12, C44, Q11, Q12, Q44, eps_star_static=None, eps_m=None, film_full=False)` | eps_star_static: optional (3,3,Ny,Nx) static eigenstrain disorder field ... |
| `homogeneous_strain_misfit` | `(eps_star_mean, eps_m, C11, C12)` | SPEC §1.4 mode (b) / §6.2c: ebar_11=ebar_22=eps_m, ebar_12=ebar_13=ebar_23=0, ... |
| `homogeneous_strain_film_full` | `(eps_star_mean, e0, C11, C12)` | SPEC WP6 §12.1: FULL film constraint -- ebar_11=ebar_22=e0, ebar_12=0; ... |
| `green_precompute` | `(K, Omega)` | The P-independent complex operands _green_delta_eps multiplies: |
| `elastic_solve` | `(P, el)` | Returns dict with eps_star (total, incl. static disorder if any), ... |
| `elastic_energy_density` | `(sol)` | f_el = 1/2 sigma:e (full double contraction -> shear factor 2 falls out of einsum). |
| `elastic_force` | `(P, el)` | delta F_el / delta P (re-solves u at the given P -- the '1 elastic solve per ... |
| `contract_C_cubic` | `(C11, C12, C44, e)` | sigma_ij = C(r):e(r) for a CUBIC (possibly spatially-varying) C given ... |
| `contract_S_cubic` | `(s11, s12, s44_voigt, sigma)` | e_ij = S(r):sigma(r), the COMPLIANCE contraction for a cubic tensor ... |
| `setup_elastic_spm` | `(grid, C11_field, C12_field, C44_field, Q11, Q12, Q44, eps_star_static=None, spm_tol=1e-10, spm_max_iter=None, outer_tol=1e-10, outer_max_iter=50, C0_override=None, method='basic', cg_precond=True)` | WP10 SPEC §16: inhomogeneous-C(r) elasticity via the Wang et al. 2013 ... |
| `elastic_solve_spm` | `(P, el, warm_start=True)` | WP10 SPEC §16 elastic SPM solve (Wang et al. 2013 Appendix A): inner ... |

*public constants:* `FFT_WORKERS`, `SPM_DIVERGED_RESID`, `SPM_METHODS`, `CG_DEFAULT_MAX_ITER`

### `energy_minimize.py`

| public name | signature | what it does |
|---|---|---|
| `bulk_energy_density_fast` | `(P, alpha)` | f_bulk(P), pow-free; alpha as in solver.bulk_energy_density. |
| `default_tau` | `(alpha, kappa_b=None)` | Default preconditioner scale ~ the engine's typical stable dt, i.e. the ... |
| `minimize` | `(problem, P0, *, method='lbfgs', gtol=1e-06, ftol=1e-10, max_iter=10000, memory=10, sample_every=1, energy_scale=None, max_step=None, callback=None)` | Minimize problem's F_total from P0. Returns (P, trace, stop_reason, info). |

### `ferret_paths.py`

No public function.

*public constants:* `FERRET_ROOT`

### `film_elastic.py`

| public name | signature | what it does |
|---|---|---|
| `make_stack_grid` | `(film_grid, hs_dx=PAPER_HS_DX)` | Extend a film3d z-grid downward by hs_dx substrate nodes; adds Nz_film/Nz_s/hs/film_lo, film node indices unchanged. |
| `paper_stack_grid` | `(hs_dx=PAPER_HS_DX, **kwargs)` | make_stack_grid over film3d.paper_grid(**kwargs) -- SPEC §13.0's film+substrate stack at the paper's own size. |
| `homogeneous_strain_film_e0` | `(e0, C11, C12)` | Li 2002 substrate-clamped homogeneous strain for misfit e0: diag(e0, e0, -2*(C12/C11)*e0), shape (3,3). |
| `factor_block_tridiag_batched` | `(sub, diag, sup)` | sub,sup: (n-1,...,3,3). diag: (n,...,3,3). |
| `solve_block_tridiag_factored` | `(factors, rhs)` | rhs: (n,...,3[,extra broadcastable trailing axes]). Returns same shape. |
| `setup_film_elastic` | `(stack_grid, C11, C12, C44, Q11, Q12, Q44, e0, eps_star_static=None, C_substrate=None)` | Assemble+factorize the stack's per-k FE stiffness once per (grid, C, Q, e0) -> the `el` dict elastic_solve_film consumes. |
| `elastic_solve_film` | `(P, el)` | Six stages, in order: (1) stack eigenstrain eps*(P), zero in the ... |
| `elastic_energy_density_film` | `(sol, el)` | Film-only elastic energy density (SPEC film3d.py convention: total / ... |
| `total_elastic_energy_film` | `(sol, el, grid)` | hz*sum(dens)/Volume3D, area-averaged first (matching film3d.py's own ... |
| `elastic_force_film` | `(P, el)` | delta F_el/delta P (film-only, matches P's own shape) -- re-solves ... |
| `setup_film_elastic_spm` | `(stack_grid, C_film, Q, e0, eps_star_static=None, C0_override=None, spm_tol=1e-10, spm_max_iter=100, outer_tol=1e-10, outer_max_iter=50)` | SPEC V2-03 C2: the `el` dict elastic_solve_film dispatches to for an ... |
| `elastic_solve_film_spm` | `(P, el, warm_start=True)` | SPEC V2-03 C2's fixed point (section comment above for the derivation). |

*public constants:* `PAPER_HS_DX`

### `film_elastic_force_numba.py`

No public function.

### `film_elastic_force_numpy.py`

No public function.

### `film_electrostatic_inhomogeneous.py`

| public name | signature | what it does |
|---|---|---|
| `kappa0_inplane_mean` | `(kappa_field)` | The default reference profile: the in-plane mean of kappa(x,y,z) at ... |
| `poisson_factor_spm` | `(grid, kappa_field, kappa0_profile=None, method='basic', spm_cg_max_iter=None)` | Factorize the SPM's reference operator once per (grid, kappa_field). |
| `solve_poisson_spm` | `(P, grid, factors, E_ext=(0.0, 0.0, 0.0), tol=1e-10, max_iter=None, warn_iter=None, warm=None, record_history=False)` | The fixed point of the module docstring: each iterate is ONE ... |
| `elec_force_film_spm` | `(P, grid, factors, E_ext=(0.0, 0.0, 0.0), tol=1e-10, warm=None)` | delta F_elec/delta P_i = -E_d,i - E_ext,i -- film3d.elec_force_3d's ... |
| `elec_energy_film_spm` | `(P, grid, factors, E_ext=(0.0, 0.0, 0.0), tol=1e-10, warm=None)` | f_elec = -0.5*P.E_d - P.E_ext, nodal density (Nz,Ny,Nx) -- film3d. |
| `total_energy_film_spm` | `(P, alpha, grid, factors, G11, E_ext=(0.0, 0.0, 0.0), tol=1e-10)` | (densities, totals) for bulk/grad/elec/total -- film3d.total_energy_3d's ... |

### `film_engine.py`

| public name | signature | what it does |
|---|---|---|
| `make_grid_3d` | `(Nx, Ny, Lx, Ly, Nz, hf)` | NAME COLLISION: bulk3d has its own make_grid_3d(Nx, Ny, Nz, Lx, Ly, ... |
| `paper_grid` | `(Nx=PAPER_NXY, Ny=PAPER_NXY, Nz=PAPER_NZ, dx_nm=PAPER_DX_NM)` | Convenience: the paper's own in-plane size/spacing (SPEC §13.0), ... |
| `paper_mechanics` | `()` | SPEC §13.0's paper parameter set: literature_params.bto_lcc2005_paper(T=298 K, dx_nm=PAPER_DX_NM). |
| `paper_alpha` | `(T_C)` | SPEC WP6-style alpha dict at temperature T_C (deg C) -- same pattern ... |
| `factor_tridiag_batched` | `(sub, diag, sup)` | Precompute the forward-elimination coefficients ONCE for a fixed ... |
| `solve_tridiag_factored` | `(factors, rhs)` | rhs: (n,)+batch_shape[+extra trailing axes broadcastable against ... |
| `solve_tridiag_batched` | `(sub, diag, sup, rhs)` | One-shot (factor+solve) entry point -- used where the operator isn't ... |
| `poisson_factor` | `(grid, kappa_b)` | Factorize the film's z-tridiagonal Poisson operator once per (grid, ... |
| `solve_poisson_factored` | `(P, grid, factors, E_ext=(0.0, 0.0, 0.0))` | Solve for phi given P, return conjugate (E, E_ext_broadcast) as real- ... |
| `physical_electric_field_3d` | `(P, grid, factors, E_ext=(0.0, 0.0, 0.0))` | Pointwise field diagnostic with second-order one-sided surface E3. |
| `elec_force_3d` | `(P, grid, poisson_factors, E_ext=(0.0, 0.0, 0.0))` | delta F_elec/delta P_i = -E_d,i - E_ext,i (self-adjoint envelope ... |
| `elec_energy_density_3d` | `(P, grid, poisson_factors, E_ext=(0.0, 0.0, 0.0))` | f_elec = -0.5*P.E_d - P.E_ext with the film-geometry Poisson solve, shape (Nz,Ny,Nx). |
| `factor_pentadiag_batched` | `(sub2, sub1, diag, sup1, sup2)` | sub2,sup2: 1D (n-2,); sub1,sup1: 1D (n-1,); diag: (n,)+batch_shape. |
| `solve_pentadiag_factored` | `(factors, rhs)` | rhs: (n,)+batch_shape[+trailing axes]; returns the same shape. |
| `factor_block_pentadiag_batched` | `(sub2, sub1, diag, sup1, sup2)` | 3x3-block twin of factor_pentadiag_batched, in film_elastic's block ... |
| `solve_block_pentadiag_factored` | `(factors, rhs)` | rhs: (n,...,3); returns the same shape. |
| `hog_energy_film` | `(P, grid, H11)` | F_hog (a total, deck-energy units): area*sum_j qw_j <in-plane part> + ... |
| `hog_energy_density_film` | `(P, grid, H11)` | hog_energy_film / Volume3D -- the module's own dens convention (see ... |
| `hog_force_film` | `(P, grid, H11)` | delta F_hog/delta P_i in this module's (1/qw) convention, (3,Nz,Ny,Nx). |
| `grad_energy_3d` | `(P, grid, G, H11=None)` | F_grad, a TOTAL (not a density) -- the exact discrete functional whose ... |
| `grad_energy_density_3d` | `(P, grid, G, H11=None)` | grad_energy_3d / Volume3D -- the module's dens convention; see ... |
| `grad_force_3d` | `(P, grid, G, H11=None)` | delta F_grad/delta P_i, applied EXPLICITLY (matrix action, not a ... |
| `build_grad_factors` | `(grid, G, dt, L, H11=None, stab=0.0)` | M(k) = I + dt*L*[grad force operator] -- SPEC's implicit gradient, ... |
| `build_grad_factors_hog` | `(grid, G, H11, dt, L)` | SPEC V2-03 B5's own spelling of build_grad_factors(..., H11=H11). |
| `build_grad_factors_sbdf2` | `(grid, G, dt, L, H11=None, stab=0.0)` | M2 = 3I + 2*dt*L*[grad force operator] (SBDF2's implicit operator, ... |
| `explicit_N` | `(P, alpha, grid, poisson_factors, E_ext=(0.0, 0.0, 0.0), el=None)` | N(P) is this module's mass-matrix-weighted TDGL RHS: qw(z)*dP(z)/dt = ... |
| `step` | `(P, alpha, grid, grad_factors, poisson_factors, dt, L, E_ext=(0.0, 0.0, 0.0), el=None, stab=0.0)` | One semi-implicit step, gradient implicit in z+k: P^(n+1) = grad_factors^-1 . [P^n_hat - dt*L*FFT(N(P^n))]. |
| `step_sbdf2` | `(P_n, P_nm1, alpha, grid, grad_factors2, poisson_factors, dt, L, E_ext=(0.0, 0.0, 0.0), el=None, stab=0.0)` | One SBDF2 step from (P_n, P_nm1), N evaluated at the extrapolant P* = 2*P_n - P_nm1; grad_factors2 from build_grad_factors_sbdf2. |
| `probe_stable_dt` | `(P0, alpha, grid, G, kappa_b, L, E_ext=(0.0, 0.0, 0.0), dt0=0.02, n_probe=200, growth=2.0, p_cap=None, max_doublings=30, scheme='sbdf1', el=None, H11=None)` | SPEC §2's protocol, adapted (same as solver.py's probe_stable_dt). |
| `total_energy_3d` | `(P, alpha, grid, kappa_b, G, poisson_factors=None, E_ext=(0.0, 0.0, 0.0), el=None, H11=None)` | Returns (densities dict, totals dict): bulk/grad/elec[/hog][/elastic]/total. |
| `relax` | `(P0, alpha, grid, G, kappa_b, L, E_ext=(0.0, 0.0, 0.0), dt=None, tol_F=1e-08, n_max=20000, check_every=10, n_plateau=20, dt0=0.02, n_probe=200, scheme='sbdf1', el=None, H11=None)` | Minimal relax loop (no dt_adapt/tol_P/post_step -- not needed by any ... |

*public constants:* `FFT_WORKERS`, `PAPER_NXY`, `PAPER_DX_NM`, `PAPER_HF_DX`, `PAPER_NZ`, `KAPPA_B_DECK`, `KAPPA_B_PAPER_LINEAGE`

### `film_langevin.py`

| public name | signature | what it does |
|---|---|---|
| `node_weights` | `(grid)` | Dimensionless FD node weights w_j = qw_j/hz: exactly 1.0 at interior ... |
| `build_ou_operators_film` | `(grid, G11, dt, L, kT, noise_cutoff_nm=None)` | Exact per-(j,k) OU propagator of the film's IMPLICIT operator ... |
| `step_langevin_film` | `(P, alpha, grid, ou, poisson_factors, dt, L, E_ext, rng, el=None)` | One first-order stochastic-splitting step of the film TDGL (SPEC §18, ... |
| `mode_amplitudes` | `(P, grid, ou)` | The film's FDT observable: y_j(k) = V^T W_hat rfft2(P) / sqrt(Nx*Ny), ... |
| `kT_deck_film` | `(T_kelvin, hx_nm, hy_nm, hz_nm, k_B=units.K_BOLTZMANN_SI)` | kT for step_langevin_film, in deck units, derived in the module ... |

### `gradient4.py`

| public name | signature | what it does |
|---|---|---|
| `build_operators_hog` | `(grid, G11, G12, G44, G44p, kappa, H11)` | Isotropic first-gradient note (params/nnbm_wang2026.toml convention, ... |
| `hog_energy_density` | `(P, grid, H11)` | f_hog = 0.5*H11*sum_i[(d^2Pi/dx^2)^2+(d^2Pi/dy^2)^2], i=1,2,3 ... |
| `grad_energy_density_hog` | `(P, grid, ops)` | Total gradient energy = solver.py's own (unmodified) G-term [reads ... |
| `total_energy_hog` | `(P, alpha, ops, E_ext, grid, elastic_params=None, spm_tol=1e-10)` | Mirrors solver.total_energy's (dens, tot) contract exactly, with ... |
| `total_force_hog` | `(P, alpha, ops, E_ext, elastic_params=None, spm_tol=1e-10)` | Fb+Fg+Fe[+Fel]. Fg is JUST solver.grad_force(P, ops) -- ops's ... |
| `build_Minv` | `(grid, ops, dt, L)` | Delegates to solver.build_Minv -- the H11 sector needs no stepping math of its own (ARCHITECTURE note above). |
| `step` | `(P, alpha, ops, Minv, dt, L, E_ext, elastic_params=None)` | Deterministic semi-implicit step, H11-augmented ops -- delegates to ... |
| `build_operators_hog_3d` | `(grid, G11, G12, G44, G44p, kappa, H11, kappa0_override=None, spm_warm=False)` | bulk3d.build_operators_3d(...) + H11*(kx^4+ky^4+kz^4) added to the ... |
| `hog_energy_density_3d` | `(P, grid, H11)` | f_hog = 0.5*H11*sum_i[(d^2Pi/dx^2)^2+(d^2Pi/dy^2)^2+(d^2Pi/dz^2)^2], ... |
| `grad_energy_density_hog_3d` | `(P, grid, ops)` | bulk3d's own (unmodified) G-term energy + this module's H11 piece ... |
| `total_energy_hog_3d` | `(P, alpha, ops, E_ext, grid, elastic_params=None)` | Mirrors bulk3d.total_energy_3d's (dens, tot) contract exactly, with ... |
| `total_force_hog_3d` | `(P, alpha, ops, E_ext, elastic_params=None)` | Fb+Fg+Fe[+Fel], bulk3d.total_force_3d's own return-arity convention. |
| `dispersion_rate` | `(alpha1, G11, H11, kx, ky)` | Lambda(k) = 2*alpha1 + G11*(kx^2+ky^2) + H11*(kx^4+ky^4) -- the ... |
| `k_star` | `(G11, H11)` | k* = sqrt(-G11/(2*H11)), the minimum of dispersion_rate ALONG A ... |
| `lambda_star` | `(G11, H11)` | lambda* = 2*pi/k* (nm, same length unit as the grid): the period of ... |
| `load_nnbm_deck` | `(path=None)` | Returns dict(alpha={a1..a123}, gradient={G110,G11_G110,...}, H11=, ... |

*public constants:* `NNBM_TOML`

### `input_validation.py`

| public name | signature | what it does |
|---|---|---|
| `landau_coefficients` | `(alpha, *, field_shape=None, require=True, where='alpha')` | Check a Landau coefficient dict in the engine's a1..a1123 naming. |
| `grid_geometry` | `(grid, ndim, *, where='grid')` | Check a solver.make_grid (ndim=2) or bulk3d.make_grid_3d (ndim=3) dict: |
| `cubic_stiffness` | `(C11, C12, C44, *, where='C', names=('C11', 'C12', 'C44'))` | Finite cubic stiffness that is positive-definite: C11 > \|C12\|, ... |
| `applied_field` | `(E_ext, P_shape, *, where='E_ext')` | E_ext must be finite and either a 3-vector (3,) or a field with P's ... |
| `polarization` | `(P, grid_shape, *, where='P')` | P must be finite real, shaped (3, *grid_shape) or (3, S, *grid_shape). |
| `deck` | `(d, *, where='deck')` | Check a materials.get()/from_toml() deck (materials.py R1 schema) and ... |
| `deck_sectors` | `(d, sectors, *, where='deck')` | The deck has each of `sectors`, complete: exactly the schema's keys ... |
| `toml_deck` | `(path)` | Check a params/*.toml deck BEFORE deck_parser reads it, for the cases ... |
| `build_operators` | `(grid, G11, G12, G44, G44p, kappa)` | Setup check for solver.build_operators (2D). |
| `build_operators_3d` | `(grid, G11, G12, G44, G44p, kappa, kappa0_override=None)` | Setup check for bulk3d.build_operators_3d. |
| `setup_elastic` | `(grid, C11, C12, C44, Q11, Q12, Q44, eps_star_static=None, eps_m=None)` | Setup check for elastic.setup_elastic (2D, homogeneous C). |
| `setup_elastic_3d` | `(grid, C, Q, labels=None, eps_star_static=None)` | Setup check for bulk3d.setup_elastic_3d (C and Q are tensor triples). |
| `setup_film_elastic` | `(stack_grid, C11, C12, C44, Q11, Q12, Q44, e0, eps_star_static=None, C_substrate=None)` | Setup check for film_elastic.setup_film_elastic (eps_star_static's ... |
| `film_make_grid_3d` | `(Nx, Ny, Lx, Ly, Nz, hf)` | Setup check for film3d.make_grid_3d (Nz nodes span [0, hf] inclusive). |
| `poisson_factor` | `(grid, kappa_b)` | Setup check for film3d.poisson_factor: kappa_b scalar or (Nz,) profile. |
| `build_2d` | `(d, Nx, Ny, h)` | Setup check for casekit.build_2d. |
| `build_3d` | `(d, N, d_cell, elastic=False)` | Setup check for casekit.build_3d. |
| `anneal` | `(engine, P, alpha, grid, E_ext)` | Setup check for casekit.anneal: P finite and shaped for the engine's ... |

*public constants:* `LANDAU_REQUIRED`, `LANDAU_OPTIONAL`, `LANDAU_KEYS`, `DECK_SECTORS`, `DECK_KEYS`

### `kernels_fused.py`

| public name | signature | what it does |
|---|---|---|
| `workspace` | `(name, shape, dtype=np.float64)` | A per-thread scratch array, reused across steps of the same shape. |
| `release_workspaces` | `()` | Drop this thread's scratch arrays (configure() calls it on switches). |
| `periodic_step` | `(kind, P, P_prev, alpha, ops, op, dt, L, E_ext, el, stab, workers, ndim, rng=None, noise_shape=None)` | One fused periodic step, or None when the inputs are outside coverage. |
| `film_force_solver` | `(el, workers)` | A film_elastic_numba.ForceSolver whose evaluation uses the fused passes. |
| `film_step` | `(kind, P, P_prev, alpha, grid, factors, poisson_factors, dt, L, E_ext, el, stab)` | One fused film3d step ("sbdf1" or "sbdf2"), or None outside coverage ... |

*public constants:* `PER_COMPONENT_2D`

### `kernels_per_term.py`

| public name | signature | what it does |
|---|---|---|
| `landau_force` | `(P, alpha)` | Evaluate the existing Landau derivative using independent CPU sites. |
| `green_delta_eps` | `(tau, K, Om, zero, shape, fwd, inv)` | Fuse Green arithmetic between the existing six forward/inverse FFTs. |
| `eigenstrain` | `(P, Q11, Q12, Q44)` | Voigt-sourced Q44 convention: eps12 = 0.5 * Q44 * P1 * P2. |
| `contract_cubic` | `(C11, C12, C44, e)` | Cubic stiffness contraction for symmetric tensor fields. |
| `elastic_force` | `(P, sigma, Q11, Q12, Q44)` | Contract symmetric stress with the derivative of eigenstrain. |

### `landau_0d_diagnostics.py`

| public name | signature | what it does |
|---|---|---|
| `evaluate` | `(P, alpha, E=(0.0, 0.0, 0.0))` | Return energy density, dF/dP, Hessian for one homogeneous P vector. |
| `radial_coefficients` | `(alpha, direction)` | Coefficients b_n for F(r*direction)=sum b_n*(r^2)^n. |
| `symmetry_stationary_points` | `(alpha)` | All real nonnegative radial extrema on C/T/O/R branches, with full Hessians. |
| `local_response` | `(alpha, P0, E, *, tolerance=1e-12, max_iterations=40)` | Newton solve in a stable homogeneous basin for a small applied deck field. |

*public constants:* `POWERS`, `DIRECTIONS`

### `mat_bto_su_landis_landau.py`

| public name | signature | what it does |
|---|---|---|
| `si_coefficients` | `()` | Every (A.2) constant in SI units. |
| `landau_clamped_si` | `()` | Zero-strain Landau part of (A.1) in this engine's alpha keys (SI). |
| `relaxed_mech` | `(P1, P2, c=None, kind='free3d', shear_once=None)` | Minimize (A.1)'s mechanical terms over the strain at fixed in-plane P. |
| `landau_effective_si` | `(kind='free3d', shear_once=None, p_max_frac=1.25, n_r=60, n_theta=25, bounded=True)` | Stress-free effective Landau set (SI, engine alpha keys) from a ... |
| `check` | `(alpha_si, c=None, kind='free3d', shear_once=None)` | The paper's own three characteristic values, recomputed from the ... |
| `deck` | `(kind='free3d', shear_once=None, **fit_kw)` | materials.get()-shaped deck. kind in KINDS: 'free3d' (default, all ... |
| `shell_alpha_deck` | `(eps_r)` | Linear-dielectric shell of TOTAL relative permittivity eps_r on top of ... |

*public constants:* `EPS0_SI`, `EPS0_DECK`, `EPS33_OVER_E0`, `SHEAR_ONCE`, `SU2015`, `CITATION`, `KINDS`

### `mat_chen2007_decks.py`

| public name | signature | what it does |
|---|---|---|
| `bto_lcc2005_8th` | `(T=298.0)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._bto_lcc2005_8th_flat, reshaped to this module's frozen alpha1..-keyed dict. |
| `bto_lcc2005_paper` | `(T=298.0, dx_nm=0.4)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._bto_lcc2005_paper_flat, reshaped to this module's frozen alpha1..-keyed dict. |
| `pto_haun1987` | `(T=298.0)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pto_haun1987_flat, reshaped to this module's frozen alpha1..-keyed dict. |
| `pzt_haun_x_landau_si` | `(x, T)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials.pzt_haun_x_landau_si, pass-through (SI values). |
| `pzt_haun_x_Q` | `(x)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials.pzt_haun_x_Q, pass-through (electrostriction Q, no unit conversion). |
| `pzt_haun_x_elastic_si` | `(x, mechanics=True)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials.pzt_haun_x_elastic_si, pass-through (SI values). |
| `pzt_haun_x` | `(x=0.6, T=298.0, mechanics=True)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pzt_haun_x_flat, reshaped to this module's frozen alpha1..-keyed dict. |
| `write_toml` | `(params, out_path)` | Write a legacy alpha1..-keyed params dict to a SPEC §6.1 TOML file; None-valued REQUIRED fields are omitted with a comment, never faked. |
| `generate_all_toml` | `(out_dir)` | Write the concrete (default-parameter) TOML for each WP4 §9.1 set ... |

### `mat_pan2019_landau.py`

| public name | signature | what it does |
|---|---|---|
| `bfo_alpha1_si` | `(T)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bfo_alpha1_si. BFO alpha1(T) in SI. |
| `bfo_landau_si` | `(T)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bfo_landau_si. BFO alpha1..alpha123 dict in SI. |
| `bto_alpha1_si` | `(T)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bto_alpha1_si. BTO alpha1(T) in SI. |
| `bto_alpha11_si` | `(T)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bto_alpha11_si. BTO alpha11(T) in SI. |
| `bto_alpha111_si` | `(T)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bto_alpha111_si. BTO alpha111(T) in SI, Pan-literal constant. |
| `bto_landau_si` | `(T)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bto_landau_si. BTO alpha1..alpha123 dict in SI. |
| `bto_alpha111_si_sluka_literal` | `(T)` | ALTERNATE, NOT used by bto_landau_si/bto_landau_deck (production path ... |
| `sto_alpha1_si` | `(T)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_sto_alpha1_si. STO alpha1(T) in SI, Pan-literal 36.5 K. |
| `sto_landau_si` | `(T)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_sto_landau_si. STO alpha1..alpha123 dict in SI. |
| `si_to_deck_landau_dict` | `(d_si)` | {alpha_name: SI value} -> {alpha_name: deck value}, uniform x1e-9 ... |
| `bfo_landau_deck` | `(T=T_REF_K)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bfo_landau_si, then si_to_deck_landau_dict -- BFO alphas in deck units. |
| `bto_landau_deck` | `(T=T_REF_K)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_bto_landau_si, then si_to_deck_landau_dict -- BTO alphas in deck units. |
| `sto_landau_deck` | `(T=T_REF_K)` | PARAMETER-BINDING SHIM (SPEC §36.2 R4) -> materials._pan2019_sto_landau_si, then si_to_deck_landau_dict -- STO alphas in deck units. |

*public constants:* `T_REF_K`, `G110_SI`, `G11_G110`, `G12_G110`, `G44_G110`, `G44P_G110`, `G110_DECK`, `EPS0_DECK_BASE`, `KAPPA_B_MULT`, `KAPPA_B_DECK`

### `mat_pmn_charge_disorder.py`

| public name | signature | what it does |
|---|---|---|
| `derive_ed_stats_substitutional` | `(lambda_s_nm=LAMBDA_S_DEFAULT_NM, eps_r=EPS_R_DEFAULT, a_lattice_nm=A_LATTICE_PMN_NM)` | DERIVE (charge statistics -> screening -> field), SPEC S21.1: returns ... |
| `defect_field_substitutional` | `(grid, lambda_s_nm=LAMBDA_S_DEFAULT_NM, eps_r=EPS_R_DEFAULT, seed=0, a_lattice_nm=A_LATTICE_PMN_NM)` | Layer 1 E_d(r), DECK units, shape (3,Ny,Nx) -- ready for solver's ... |
| `derive_ed_stats_cor` | `(f_cor=F_COR_DEFAULT, xi_cor_nm=XI_COR_DEFAULT_NM, lambda_s_nm=LAMBDA_S_DEFAULT_NM, eps_r=EPS_R_DEFAULT, a_lattice_nm=A_LATTICE_PMN_NM)` | SPEC S21.3 (2026-08-23): historical/sensitivity option, DEFAULT OFF -- ... |
| `defect_field_cor` | `(grid, f_cor=F_COR_DEFAULT, xi_cor_nm=XI_COR_DEFAULT_NM, lambda_s_nm=LAMBDA_S_DEFAULT_NM, eps_r=EPS_R_DEFAULT, seed=0, a_lattice_nm=A_LATTICE_PMN_NM)` | SPEC S21.3: historical/sensitivity option, DEFAULT OFF -- see ... |
| `bond_field_cor_alpha1` | `(grid, T, f_cor=F_COR_DEFAULT, xi_cor_nm=XI_COR_DEFAULT_NM, dT0_cor_K=DT0_COR_DEFAULT_K, T0_pmn_K=None, C_cw_K=None, seed=0)` | DEFAULT COR model (SPEC S21.3 ruling 1): alpha1(r), DECK units, shape ... |
| `pmn_clean_alpha1` | `(T, T0_K=T0_PMN_CW_K, C_cw_K=C_CW_PMN_K)` | Pure-PMN clean-limit alpha1(T), DECK units -- SPEC S21.3 ruling 2, ... |

*public constants:* `MEAN_DQ`, `VAR_DQ`, `DQ_COR`, `E_CHARGE_SI`, `EPS0_SI`, `A_LATTICE_PMN_NM`, `LAMBDA_S_RANGE_NM`, `LAMBDA_S_DEFAULT_NM`, `EPS_R_BRACKET`, `EPS_R_DEFAULT`, `F_COR_DEFAULT`, `XI_COR_DEFAULT_NM`, `DT0_COR_BRACKET_K`, `DT0_COR_DEFAULT_K`, `T0_PMN_CW_K`, `T0_PMN_CW_RANGE_K`, `C_CW_PMN_K`, `C_CW_PMN_RANGE_K`, `ALPHA11_PMN_BRACKET`, `ALPHA11_PMN_NOTE`

### `mat_pzt_coefficient_field.py`

| public name | signature | what it does |
|---|---|---|
| `landau_field` | `(x, T)` | SPEC §15.1: alpha1(x,T) via T0(x),C0(x); alpha11/111/112 via the App A ... |
| `Q_field` | `(x)` | Q11,Q12,Q44(x) -- SPEC §15.1. x scalar or array; materials. |
| `elastic_C_field` | `(x, mechanics=True)` | WP10 SPEC §16: C11,C12,C44 (deck units) as a SPATIAL field -- x may be ... |
| `elastic_C_at` | `(x, mechanics=True)` | Elastic C11,C12,C44 (deck units) at a SINGLE scalar composition x, via ... |
| `vegard_eigenstrain_field` | `(x, x_bar, dlna_dx=DLNA_DX_PZT)` | eps_chem = dlna_dx*(x-x_bar)*I3. x may be an (Ny,Nx) array or a ... |
| `formula_units_per_cell` | `(hx, hy, a_lattice=A_LATTICE_PZT_NM, hz=None)` | N_f = formula units per coarse-grain cell = cell volume / a^3 (SPEC: |
| `composition_std` | `(x_bar, N_f)` | Random-substitutional-alloy mixing limit (SPEC §15.1): variance of ... |
| `composition_std_correlated` | `(x_bar, xi_x, a_lattice=A_LATTICE_PZT_NM, t_eff=None)` | SPEC §15.5 coordinator ruling (2026-08-21), BINDING for WP10's G14e ... |
| `composition_field` | `(grid, x_bar, std, xi_x=None, seed=0, a_lattice=A_LATTICE_PZT_NM)` | x(r) = x_bar + a zero-mean fluctuation of the given std (SPEC §15.1 ... |
| `defect_field` | `(grid, rms, xi, seed)` | Returns (3,Ny,Nx): a field-valued E_ext, consumed by solver.py's ... |

*public constants:* `DLNA_DX_PZT`, `A_LATTICE_PZT_NM`

### `mat_registry.py`

| public name | signature | what it does |
|---|---|---|
| `bto_lcc2005_8th` | `(T=298.0)` | BaTiO3 8th-order (Rabe/Ahn/Triscone 2007 App A, Table 1 'Eight-order'; mechanics from bto_298K.toml) -> R1 deck. |
| `bto_lcc2005_paper` | `(T=298.0, dx_nm=0.4)` | BaTiO3 with Li & Chen APL 88, 072905 (2006) Ref.18 mechanics and that paper's own gradient -> R1 deck. |
| `pto_haun1987` | `(T=298.0)` | PbTiO3 (Rabe App A, Table 5) -> R1 deck; gradient and permittivity unsourced. |
| `pzt_haun_x_landau_si` | `(x, T)` | The p.367 formula set, in SI units. Returns ... |
| `pzt_haun_x_Q` | `(x)` | Q11,Q12,Q44(x), unscaled SI==deck. Q44 carries the source's own ... |
| `pzt_haun_x_elastic_si` | `(x, mechanics=True)` | C11,C12,C44 (SI Pa) via linear interpolation of Table 4's compliances ... |
| `pzt_haun_x` | `(x=0.6, T=298.0, mechanics=True)` | Pb(Zr1-xTix)O3 Haun x-dependent set (Rabe App A p.367 + Table 4) -> R1 deck; x scalar or array. |
| `pmn_clean_alpha1` | `(T, T0_K=_PMN_T0_CW_K_DEFAULT, C_cw_K=_PMN_C_CW_K_DEFAULT)` | Pure-PMN clean-limit alpha1(T), DECK units -- SPEC S21.3 ruling 2. |
| `get` | `(name, **kw)` | materials.get(name, T=..., **kw) -> deck (SPEC S36.2 R1 schema, ... |
| `list` | `()` | [(name, one-line provenance), ...], sorted by name. |
| `from_toml` | `(path)` | Read a params/*.toml deck (deck_parser's flat TOML_SECTIONS schema) ... |

*public constants:* `PARAMS_DIR`, `PAN2019_T_REF_K`, `PAN2019_G110_SI`, `PAN2019_G11_G110`, `PAN2019_G12_G110`, `PAN2019_G44_G110`, `PAN2019_G44P_G110`, `PAN2019_G110_DECK`, `PAN2019_EPS0_DECK_BASE`, `PAN2019_KAPPA_B_MULT`, `PAN2019_KAPPA_B_DECK`

### `mat_registry_archive.py`

| public name | signature | what it does |
|---|---|---|
| `bto_bc1984_6th` | `(T=298.0)` | BaTiO3 6th-order (Rabe App A, Table 1 'Sixth order') -> R1 deck; reference only, not fit to k_t180.i. |
| `bto_bc1984_6th_legacy` | `(T=298.0)` | Was literature_params.bto_bc1984_6th: _bto_bc1984_6th_flat reshaped to the frozen alpha1..-keyed, "_meta"-suffixed dict that literature_params.write_toml serializes. |
| `bfo_pan2019` | `(T=PAN2019_T_REF_K)` | BiFeO3 end member of Pan 2019's ternary region mixture -> R1 deck; numeric source untraceable, flagged. |
| `bto_pan2019` | `(T=PAN2019_T_REF_K)` | BaTiO3 end member of Pan 2019's ternary region mixture -> R1 deck; alpha111 Pan-literal 2.76e8. |
| `sto_pan2019` | `(T=PAN2019_T_REF_K)` | SrTiO3 end member of Pan 2019's ternary region mixture -> R1 deck; alpha1 Pan-literal 36.5 K. |
| `pmn_clean_x0` | `(T=298.0, T0_K=_PMN_T0_CW_K_DEFAULT, C_cw_K=_PMN_C_CW_K_DEFAULT)` | Pure PMN (x_PT=0) clean-limit Curie-Weiss alpha1(T) -> R1 deck; alpha11.. unsourced and absent. |
| `pmn_pt_placeholder` | `(x_pt=0.3, T=298.0)` | 0.70PMN-0.30PT: citation confirmed (Hangbo Zhang et al., PRB 96, ... |
| `pmn_pt_landau_placeholder` | `(x_pt=0.3, T=298.0)` | Structure-only placeholder (see pmn_field.py's PMN-PT comment block): |
| `get` | `(name, **kw)` | materials_archive.get(name, T=..., **kw) -> deck, the same R1 schema materials.get() returns; KeyError for a name that is not archived. |
| `list` | `()` | [(name, one-line provenance), ...] over the archived entries, sorted by name. |
| `generate_archive_toml` | `(out_dir=ARCHIVE_PARAMS_DIR)` | Rewrite the archived snapshot whose generator lives here (bto_BC1984_6th.toml, T=298 K) via literature_params.write_toml; literature_params.generate_all_toml covers the others. |

*public constants:* `ARCHIVE_PARAMS_DIR`, `ARCHIVED_REGISTRY`

### `octahedral_tilt_order.py`

| public name | signature | what it does |
|---|---|---|
| `sto_afd_landau_si` | `(T)` | SI dict: P-sector, theta-sector (both engine ALPHA_KEYS naming) and ... |
| `sto_afd_landau_deck` | `(T)` | SI -> deck units: P-sector via units.si_to_deck_landau (P unscaled, ... |
| `load_sto_deck` | `(path=None)` | Load params/sto_afd_rabeA.toml, remapping its collision-safe TOML ... |
| `couple_energy_density` | `(P, theta, coupling)` | f_couple(P,theta) from coupling's t11/t12/t44 keys (SPEC §28.1), real space, shape = P[0].shape. |
| `couple_force_P` | `(P, theta, coupling)` | delta f_couple/delta P_i, shape (3,Ny,Nx). |
| `couple_force_theta` | `(P, theta, coupling)` | delta f_couple/delta theta_i, shape (3,Ny,Nx). |
| `build_ops_theta` | `(grid, Gtheta)` | Atheta(k) = Gtheta*(kx^2+ky^2), scalar, SAME for all 3 components. |
| `grad_energy_density_theta` | `(theta, grid, Gtheta)` | f_gradtheta = 0.5*Gtheta*\|grad theta\|^2 summed over the 3 components (isotropic, SPEC §28.1), shape (Ny,Nx). |
| `grad_force_theta` | `(theta, ops_theta)` | d f_gradtheta / dtheta = Atheta(k) . theta (diagonal-only mirror of solver.grad_force). |
| `build_Minv_theta` | `(ops_theta, dt, Ltheta)` | Mtheta(k) = 1 + dt*Ltheta*Atheta(k), scalar per k-point. |
| `with_theta_eigenstrain` | `(base_elastic_params, theta, Lam11, Lam12, Lam44)` | Copy of base_elastic_params with the rotostriction eigenstrain of theta added to its eps_star_static (SPEC §28.1, PARKED per §28.5 R6). |
| `elastic_force_explicit_theta` | `(theta, sigma, Lam11, Lam12, Lam44)` | dF_el/dtheta_m -- exact structural mirror of elastic.elastic_force_explicit ... |
| `step_ptheta` | `(P, theta, alpha_P, alpha_theta, coupling, ops, Minv, dt, L, E_ext, ops_theta=None, Minv_theta=None, Ltheta=None, base_elastic_params=None, Lam_theta=None)` | One semi-implicit step for (P,theta) together. ops_theta=None => theta ... |
| `total_energy_ptheta` | `(P, theta, alpha_P, alpha_theta, coupling, ops, grid, Gtheta, E_ext=(0.0, 0.0, 0.0), elastic_params=None, Lam_theta=None)` | Same shape as solver.total_energy, extended with theta's own ... |
| `build_ops_theta_3d` | `(grid, Gtheta)` | Atheta(k) = Gtheta*(kx^2+ky^2+kz^2), scalar, SAME for all 3 components ... |
| `grad_energy_density_theta_3d` | `(theta, grid, Gtheta)` | f_gradtheta = 0.5*Gtheta*\|grad theta\|^2 over the 3 components, shape ... |
| `grad_force_theta_3d` | `(theta, ops_theta)` | d f_gradtheta / dtheta = Atheta(k) . theta (diagonal-only mirror of ... |
| `build_Minv_theta_3d` | `(ops_theta, dt, Ltheta)` | Mtheta(k) = 1 + dt*Ltheta*Atheta(k), scalar per k-point. |
| `lam_tensor_theta_3d` | `(Lam_theta)` | (Lam1111, Lam1122, Lam1212) for bulk3d.eigenstrain_3d from the deck's ... |
| `with_theta_eigenstrain_3d` | `(base_elastic_params, theta, Lam_theta)` | Shallow-copy bulk3d.setup_elastic_3d's output with theta's ... |
| `elastic_force_explicit_theta_3d` | `(theta, sigma, Lam1111, Lam1122, Lam1212)` | dF_el/dtheta_m = -sigma_ij * d(eps0_ij)/dtheta_m, FULL TENSOR sum -- ... |
| `step_ptheta_3d` | `(P, theta, alpha_P, alpha_theta, coupling, ops, Minv, dt, L, E_ext, ops_theta=None, Minv_theta=None, Ltheta=None, base_elastic_params=None, Lam_theta=None)` | One semi-implicit step for (P,theta) together on bulk3d -- the 3D ... |
| `total_energy_ptheta_3d` | `(P, theta, alpha_P, alpha_theta, coupling, ops, grid, Gtheta, E_ext=(0.0, 0.0, 0.0), elastic_params=None, Lam_theta=None)` | total_energy_ptheta's 3D mirror: same (densities, totals) contract and ... |
| `build_theta_factors_film` | `(grid, Gtheta, dt, Ltheta)` | M_theta = I + dt*Ltheta*[theta's gradient-force operator], factorized ... |
| `grad_force_theta_film` | `(theta, grid, Gtheta)` | delta F_gradtheta/delta theta_i in film geometry, applied EXPLICITLY ... |
| `grad_energy_theta_film` | `(theta, grid, Gtheta)` | TOTAL F_gradtheta (not a density) in film geometry, mirroring ... |
| `with_theta_eigenstrain_film` | `(base_el, theta, Lam_theta)` | Shallow-copy film_elastic.setup_film_elastic's output with theta's ... |
| `step_ptheta_film` | `(P, theta, alpha_P, alpha_theta, coupling, grid, grad_factors, poisson_factors, dt, L, E_ext=(0.0, 0.0, 0.0), theta_factors=None, Ltheta=None, base_el=None, Lam_theta=None)` | One semi-implicit step for (P,theta) together in film geometry -- ... |
| `total_energy_ptheta_film` | `(P, theta, alpha_P, alpha_theta, coupling, grid, kappa_b, G11, Gtheta, poisson_factors=None, E_ext=(0.0, 0.0, 0.0), el=None, Lam_theta=None)` | (densities, totals) in film geometry, keys as total_energy_ptheta. |

*public constants:* `FFT_WORKERS`, `ALPHA1_STO_BRACKET_SI`, `ALPHA11_STO_SI`, `ALPHA12_STO_SI`, `BETA1_STO_SI`, `TC_STO_K`, `BETA11_STO_BRACKET_SI`, `BETA12_STO_BRACKET_SI`, `T11_STO_BRACKET_SI`, `T12_STO_BRACKET_SI`, `T44_STO_SI`, `LAM11_THETA_STO_BRACKET_SI`, `LAM12_THETA_STO_BRACKET_SI`, `LAM44_THETA_STO_BRACKET_SI`, `C11_STO_BRACKET_SI`, `C12_STO_BRACKET_SI`, `C44_STO_BRACKET_SI`, `Q11_STO_BRACKET_SI`, `Q12_STO_BRACKET_SI`, `Q44_STO_SI`, `STO_TOML`, `COUPLE_KEYS`

### `pf2d_dynamic_susceptibility.py`

No public function.

### `pf2d_energy_storage_loop.py`

| public name | signature | what it does |
|---|---|---|
| `build_case` | `(N=64, delta_nm=1.0, kappa_b=None, params_path=None, *, eps_b=None, permittivity_deck=None)` | Explicit new inputs: eps_b is SI-relative background permittivity; ... |
| `kappa_for` | `(kappa_b)` | deck permittivity = kappa_b * _KAPPA_UNIT_EPSB10 (WP15/16 convention, ... |
| `ic_clean` | `(grid, alpha, ops, dt, seed=0, n_equil=2000)` | Protocol (i), clean relaxation (WP20 G21b precedent, SPEC S29.1(i)): |
| `ic_noise_yang` | `(grid, seed=0)` | Protocol (ii), Yang's noise-IC (SPEC S27.3(d)/S29.1(ii)): ONE-TIME ... |
| `e_path_unipolar` | `(n1, e_max)` | E values for 0 -> e_max (n1 points) -> 0 (n1 points); the E=0 START ... |
| `run_unipolar` | `(alpha, ops, Minv, dt, n_dwell, P_init, e_path, record_fields=False)` | State-continuous unipolar drive: hold each E in e_path for n_dwell ... |
| `energy_integrals` | `(E_full, P1_full, n1)` | Trapezoid quadrature (SPEC S29.1), branch bookkeeping explicit ... |
| `verified_dt` | `(alpha, ops, grid, e_max, dt0, n1_probe=8, n_dwell_probe=200, T_total=None, tol=0.01, max_halvings=10, P_init_fn=None)` | SPEC S28.6(a)/S29.3: FIXED-TOTAL-PHYSICAL-TIME dt/2 agreement check ... |

*public constants:* `YANG_NOISE_AMPLITUDE_SI`, `PROTOCOL_CLEAN`, `PROTOCOL_NOISE`

### `pf2d_engine.py`

| public name | signature | what it does |
|---|---|---|
| `bulk_energy_density` | `(P, alpha)` | f_bulk(P), the lambdified 6th/8th-order Landau density (SPEC §1.1); 8th-order alpha keys default to 0. |
| `bulk_force` | `(P, alpha)` | delta F_bulk / delta P_i = d f_bulk / d P_i, real space, shape (3,...) ... |
| `make_grid` | `(Nx, Ny, Lx, Ly)` | 2D periodic grid dict: hx/hy, full (KX,KY) shape (Ny,Nx), rfft2 half-spectrum (KXr,KYr) shape (Ny,Nxr), Volume=Lx*Ly. |
| `build_operators` | `(grid, G11, G12, G44, G44p, kappa, spm_method='basic', spm_cg_max_iter=None)` | A(k) (grad) and D(k) (depolarization) -- SPEC §1.2 / §1.3, built on the ... |
| `grad_energy_density` | `(P, grid, ops)` | f_grad, SPEC §1.2, as the collocation (pointwise) local-gradient ... |
| `grad_force` | `(P, ops)` | delta F_grad / delta P = A(k) . P, evaluated via the same A(k) as in M (SPEC G0). |
| `elec_Ed` | `(P, ops, E_ext=(0.0, 0.0, 0.0), spm_tol=1e-10)` | Depolarization field E_d -- a function of P ALONE in both regimes ... |
| `elec_energy_density` | `(P, ops, E_ext, spm_tol=1e-10)` | f_elec = -0.5*P.E_d - P.E_ext (SPEC §1.3), shape (Ny,Nx); E_d goes through the SPM iteration when ops carries dkappa. |
| `elec_force` | `(P, ops, E_ext, spm_tol=1e-10)` | delta F_elec / delta P_i = -E_d,i - E_ext,i (the 1/2 disappears, SPEC §1.3; ... |
| `total_energy` | `(P, alpha, ops, E_ext, grid, elastic_params=None, spm_tol=1e-10)` | Return (densities dict, totals dict) for bulk/grad/elec[/elastic]/total. |
| `total_force` | `(P, alpha, ops, E_ext, elastic_params=None, spm_tol=1e-10)` | Return (Fb, Fg, Fe[, Fel], sum), each shape (3,Ny,Nx). |
| `build_Minv` | `(grid, ops, dt, L, stab=0.0)` | M(k) = I3 + dt*L*[A(k) + D(k)/kappa], inverted per k-point (3x3, generic ... |
| `step` | `(P, alpha, ops, Minv, dt, L, E_ext, elastic_params=None, stab=0.0)` | One semi-implicit step: P^(n+1) = Minv(k) . [P^n_hat - dt*L*FFT(N(P^n))]. |
| `build_Minv2` | `(grid, ops, dt, L, stab=0.0)` | M2(k) = 3*I3 + 2*dt*L*[A(k) + D(k)/kappa], SBDF2's implicit operator. |
| `step_sbdf2` | `(P_n, P_nm1, alpha, ops, Minv2, dt, L, E_ext, elastic_params=None, stab=0.0)` | One SBDF2 step (see module-section comment above for the derivation). |
| `probe_stable_dt` | `(P0, alpha, ops, grid, L, E_ext=(0.0, 0.0, 0.0), elastic_params=None, dt0=0.02, n_probe=200, growth=2.0, p_cap=None, max_doublings=30, scheme='sbdf1')` | SPEC §2 Δt protocol: start dt0, run n_probe steps, require finite F and ... |
| `relax` | `(P0, alpha, ops, grid, L, E_ext=(0.0, 0.0, 0.0), elastic_params=None, dt=None, tol_F=1e-08, n_max=20000, check_every=10, n_plateau=20, dt0=0.02, n_probe=200, post_step=None, scheme='sbdf1', dt_adapt=False, tol_P=None, dt_grow=1.2, dt_cut=2.0, dt_min_frac=0.001, dt_max_frac=32.0)` | Run the stepper to convergence. Termination: \|dF/dt\| < tol_F for ... |
| `ensemble_to_internal` | `(P)` | (S,3,Ny,Nx) -> (3,S,Ny,Nx): SPEC's listed axis order -> this engine's. |
| `ensemble_from_internal` | `(P)` | (3,S,Ny,Nx) -> (S,3,Ny,Nx): inverse of ensemble_to_internal. |
| `random_smooth_field` | `(grid, n_comp, amplitude, kmax_frac=0.3, seed=0)` | n_comp independent real fields, band-limited to \|k\| < kmax_frac*max\|k\|, ... |
| `spectral_resample_2d` | `(field2d, Ny_new, Nx_new)` | fftshift to center DC, crop or zero-pad the LOWEST \|k\| ... |
| `upsample_state` | `(P, Ny_new, Nx_new)` | P shape (3,Ny,Nx) -> (3,Ny_new,Nx_new): each component independently ... |
| `build_ou_operators` | `(ops, dt, L, kT, noise_cutoff_nm=None, grid=None)` | Precompute the exact per-mode OU propagator for Lin(k)=A(k)+D(k)/kappa ... |
| `step_langevin` | `(P, alpha, ops, ou, dt, L, E_ext, rng, elastic_params=None)` | One first-order stochastic-splitting step (SPEC §18). Nonlinear force ... |
| `block_stationary` | `(values, n_blocks_check=4, tol=0.1, *, scale=None)` | New runs must pass an explicit physical `scale` (zero-offset invariant). |

*public constants:* `FFT_WORKERS`, `ALPHA_KEYS`, `ALPHA_KEYS_8TH`, `SPM_METHODS`, `SPM_CG_DEFAULT_MAX_ITER`

### `pf3d_elastic_inhomogeneous.py`

| public name | signature | what it does |
|---|---|---|
| `cubic_tensor_field` | `(C11, C12, C44, shape)` | The cubic C_ijkl of elastic.build_C_tensor, with (Nz,Ny,Nx)-shaped ... |
| `contract_C_field` | `(C_field, e)` | sigma_ij(r) = C_ijkl(r) e_kl(r), the full rank-4 einsum. e/sigma ... |
| `solve_sym_tensor` | `(A, b)` | The symmetric x solving A_ijkl x_kl = b_ij, for a (3,3,3,3) A and a ... |
| `setup_elastic_spm_3d` | `(grid, C, Q, k0='stress_free', labels=None, eps_star_static=None, C0_override=None, spm_tol=1e-10, spm_max_iter=None, outer_tol=1e-10, outer_max_iter=50, method='basic', cg_precond=True)` | Build the elastic_params dict for an INHOMOGENEOUS C(r) (this ... |
| `elastic_solve_spm_3d` | `(P, el, warm_start=True)` | The SPM solve: inner fixed point on delta_eps(r) at fixed ebar (module ... |

### `pf3d_engine.py`

| public name | signature | what it does |
|---|---|---|
| `make_grid_3d` | `(Nx, Ny, Nz, Lx, Ly, Lz)` | Mirrors solver.make_grid, one axis added. NAME COLLISION: film3d has ... |
| `build_operators_3d` | `(grid, G11, G12, G44, G44p, kappa, kappa0_override=None, spm_warm=False, spm_method='basic', spm_cg_max_iter=None, depol_N=None)` | A(k) (grad, full symmetric 3x3) and D(k) (3D transverse/longitudinal- ... |
| `grad_energy_density_3d` | `(P, grid, ops)` | f_grad, full 3D form, as the collocation (pointwise) local-gradient ... |
| `grad_force_3d` | `(P, ops)` | delta F_grad / delta P = A(k) . P, full symmetric 3x3 A(k). |
| `elec_Ed_3d` | `(P, ops, E_ext=(0.0, 0.0, 0.0), spm_tol=1e-10)` | Depolarization field E_d -- a function of P ALONE in both regimes ... |
| `elec_energy_density_3d` | `(P, ops, E_ext, spm_tol=1e-10)` | f_elec = -0.5*P.E_d - P.E_ext, real space, shape (Nz,Ny,Nx); 3D mirror ... |
| `elec_force_3d` | `(P, ops, E_ext, spm_tol=1e-10)` | delta F_elec/delta P_i = -E_d,i - E_ext,i, shape (3,Nz,Ny,Nx); 3D ... |
| `build_Omega_3d` | `(grid, C)` | Omega(k) = [C_ijkl k_j k_l]^-1 on the rfftn half-spectrum grid, full ... |
| `eigenstrain_3d` | `(P, Q1111, Q1122, Q1212)` | eps0_ij(r) = Q_ijkl(r) P_k(r) P_l(r), FULL TENSOR FORM (SPEC S37.5's ... |
| `elastic_force_explicit_3d` | `(P, sigma, Q1111, Q1122, Q1212)` | delta F_el / delta P_m = -sigma_ij * d(eps0_ij)/dP_m, full tensor sum -- ... |
| `setup_elastic_3d` | `(grid, C, Q, k0='stress_free', labels=None, eps_star_static=None, z_mode='generalized_plane_strain')` | Build the resolved elastic_params dict every public function in this ... |
| `elastic_solve_3d` | `(P, el)` | Returns dict(eps_star, ebar, delta_eps, e, sigma) -- SAME keys/roles as ... |
| `elastic_force_3d` | `(P, el)` | delta F_el / delta P (re-solves u at the given P, "1 elastic solve per ... |
| `total_energy_3d` | `(P, alpha, ops, E_ext, grid, elastic_params=None, spm_tol=1e-10)` | Mirrors solver.total_energy's (dens, tot) contract. elastic_params=None ... |
| `total_force_3d` | `(P, alpha, ops, E_ext, elastic_params=None, spm_tol=1e-10)` | elastic_params=None (default): returns (Fb,Fg,Fe,sum), BYTE-IDENTICAL ... |
| `build_Minv_3d` | `(grid, ops, dt, L, stab=0.0)` | M(k) = I3 + dt*L*[A(k)+D(k)/kappa], full 3x3 (off-diagonals A13/A23/ ... |
| `step_3d` | `(P, alpha, ops, Minv, dt, L, E_ext, elastic_params=None, stab=0.0)` | One semi-implicit step, mirrors solver.step exactly (module docstring: |
| `build_Minv2_3d` | `(grid, ops, dt, L, stab=0.0)` | M2(k) = 3*I3 + 2*dt*L*[A(k)+D(k)/kappa], SBDF2's implicit operator -- ... |
| `step_sbdf2_3d` | `(P_n, P_nm1, alpha, ops, Minv2, dt, L, E_ext, elastic_params=None, stab=0.0)` | One SBDF2 step (see the section comment above for the derivation) -- ... |
| `probe_stable_dt_3d` | `(P0, alpha, ops, grid, L, E_ext=(0.0, 0.0, 0.0), elastic_params=None, dt0=0.02, n_probe=200, growth=2.0, p_cap=None, max_doublings=30, scheme='sbdf1')` | Mirrors solver.probe_stable_dt exactly (same doubling protocol); ... |
| `build_ou_operators_3d` | `(ops, dt, L, kT, noise_cutoff_nm=None, grid=None)` | Exact per-mode OU propagator for Lin(k)=A(k)+D(k)/kappa (SPEC §18) on ... |
| `step_langevin_3d` | `(P, alpha, ops, ou, dt, L, E_ext, rng, elastic_params=None)` | One first-order stochastic-splitting step, mirrors solver. |
| `ensemble_to_internal_3d` | `(P)` | (S,3,Nz,Ny,Nx) -> (3,S,Nz,Ny,Nx). solver.ensemble_to_internal REUSED, ... |
| `ensemble_from_internal_3d` | `(P)` | (3,S,Nz,Ny,Nx) -> (S,3,Nz,Ny,Nx): the inverse, solver. |
| `random_smooth_field_3d` | `(grid, n_comp, amplitude, kmax_frac=0.3, seed=0, seeds=None)` | n_comp band-limited (\|k\| < kmax_frac*max\|k\|) unit-variance random fields x amplitude, shape (n_comp,Nz,Ny,Nx). |
| `kT_deck_3d` | `(T_kelvin, k_B=units.K_BOLTZMANN_SI)` | Unit conversion of k_B*T into F_total units -- NOT the Langevin FDT ... |

*public constants:* `FFT_WORKERS`, `elastic_energy_density_3d`, `bulk_energy_density`, `bulk_force`, `block_stationary`

### `polycrystal.py`

| public name | signature | what it does |
|---|---|---|
| `periodic_voronoi` | `(grid, n_side, seed, jitter=0.35, mode='poisson')` | Periodic Voronoi tessellation of the (Lx,Ly) box with n_side^2 seed ... |
| `shell_label` | `(dist_gb, fraction)` | Binary shell label L(r) in {0=core, 1=shell} of EXACT area fraction ... |
| `shell_thickness_nm` | `(dist_gb, L)` | Mean shell half-width proxy: the largest GB distance inside the ... |
| `orientation_fields` | `(grain_id, angles_deg)` | (cos theta, sin theta) fields from per-grain angles (degrees, in-plane ... |
| `make_alpha` | `(core_alpha, shell_alpha, L, cos_th, sin_th)` | alpha_i(r) = core_i*(1-L) + shell_i*L for every 6th+8th-order key ... |
| `bulk_force` | `(P, alpha)` | delta F_bulk/delta P in the GLOBAL frame for per-grain rotated cubic ... |
| `bulk_energy_density` | `(P, alpha)` |  |
| `step_langevin` | `(P, alpha, ops, ou, dt, L, E_ext, rng, elastic_params=None)` | Present so casekit.anneal's engine check (`hasattr(engine, ... |
| `step` | `(P, alpha, ops, Minv, dt, L, E_ext, elastic_params=None)` | One SBDF1 semi-implicit step, solver.step with the rotated Landau force ... |
| `total_force` | `(P, alpha, ops, E_ext, elastic_params=None)` | (Fb, Fg, Fe, sum) in the global frame -- the G0-style consistency ... |
| `total_energy` | `(P, alpha, ops, E_ext, grid, elastic_params=None, spm_tol=1e-10)` | Mirrors solver.total_energy's (dens, tot) contract with the rotated ... |
| `probe_stable_dt` | `(P0, alpha, ops, grid, L, E_ext=(0.0, 0.0, 0.0), elastic_params=None, dt0=0.02, n_probe=200, growth=2.0, p_cap=None, max_doublings=30, scheme='sbdf1')` | solver.probe_stable_dt's doubling protocol on THIS module's step. |
| `winding_number` | `(P, p_min=0.0)` | Winding number of the in-plane P direction around every grid ... |
| `grain_descriptors` | `(P, grain_id, L, n_grains, p_min)` | Per grain (core cells only): mean \|P\|, \|<P>\|/<\|P\|> (net-polarization ... |

*public constants:* `ALPHA_KEYS`, `FFT_WORKERS`, `build_Minv`, `block_stationary`, `random_smooth_field`, `build_ou_operators`

### `random_field_mechanisms.py`

| public name | signature | what it does |
|---|---|---|
| `gaussian_random_field` | `(grid, sigma, xi_nm, seed)` | Single-component (Ny,Nx) zero-mean, spectrally-correlated Gaussian ... |
| `gaussian_random_field_3d` | `(grid, sigma, xi_nm, seed)` | SPEC V2-03 A6. 3D twin of gaussian_random_field: the same DECLARED-DIAL ... |
| `charge_disorder_stats` | `(conc, delta_q, lambda_s_nm, eps_r, a_lattice_nm)` | DERIVE (charge statistics -> screening -> field) substitutional ... |
| `charge_disorder_field` | `(grid, conc, delta_q, lambda_s_nm, eps_r, a_lattice_nm, seed)` | Substitutional-defect E_d(r), DECK units, shape (3,Ny,Nx) -- a ... |
| `charge_disorder_field_3d` | `(grid, conc, delta_q, lambda_s_nm, eps_r, a_lattice_nm, seed)` | SPEC V2-03 A6. 3D twin of charge_disorder_field: IDENTICAL derivation ... |
| `region_charge_disorder_stats` | `(f_region, xi_region_nm, delta_q_region, lambda_s_nm, eps_r, a_lattice_nm)` | DERIVE charged-nanoregion E_d(r) statistics. Regions of volume ... |
| `region_charge_disorder_field` | `(grid, f_region, xi_region_nm, delta_q_region, lambda_s_nm, eps_r, a_lattice_nm, seed)` | Charged-nanoregion E_d(r), DECK units, shape (3,Ny,Nx) -- same ... |
| `region_charge_disorder_field_3d` | `(grid, f_region, xi_region_nm, delta_q_region, lambda_s_nm, eps_r, a_lattice_nm, seed)` | SPEC V2-03 A6. 3D twin of region_charge_disorder_field -- same ... |
| `bond_order_alpha1_field` | `(grid, T, T0_K, C_cw_K, f_region, xi_region_nm, dT0_region_K, seed)` | DERIVE (region geometry -> T0(r) -> alpha1(r) via Curie-Weiss), DECK ... |
| `bond_order_alpha1_field_3d` | `(grid, T, T0_K, C_cw_K, f_region, xi_region_nm, dT0_region_K, seed)` | SPEC V2-03 A6. 3D twin of bond_order_alpha1_field -- the SAME ... |

*public constants:* `E_CHARGE_SI`, `EPS0_SI`, `PRESET_PMN`

### `run_adapter.py`

No public function.

*public constants:* `ENGINES`, `SCHEMES`

### `run_builder.py`

| public name | signature | what it does |
|---|---|---|
| `file_sha256` | `(path)` |  |
| `load_array` | `(path, key, where)` | Array ``key`` of the npz file ``path`` (a ConfigError at ``where`` if it is absent). |
| `precheck` | `(cfg)` | Refusals that need no built array: stage keys the schema allows but this version does not build. |
| `base_T` | `(cfg)` | The temperature the engine objects are first built at: material.temperature_K, else the first stage's T. |
| `stage_temperatures` | `(cfg)` | Every temperature the stages visit (build-time checks run over all of them). |
| `material` | `(cfg, T=None)` | Deck dict (registry layout) at temperature T (default base_T): registry name, params/*.toml path, or inline tables. |
| `make_alpha` | `(cfg, parts, deck, T)` | The alpha dict at temperature T: the deck's Landau coefficients (registry decks regenerated at T), a1 from the ... |
| `set_T` | `(b, T)` | Move the built objects to temperature T (driver B ``Model.set_T``): alpha again; for a Langevin run kT and the OU ... |
| `geometry` | `(cfg, deck, dry=False)` | (grid, ops, el, film_dict, kappa); ops / film as the Adapter wants them for the engine. |
| `fields` | `(cfg, grid, saved=None, dry=False)` | parts = dict(landau={key: ("add"\|"set", array)}, T0=array or None, E_static=zero tuple or the summed E_ext field, ... |
| `make_start` | `(adapter, start, where, space)` | P0 for a ``start`` table: random (the engine's generator) or from_file (an npz array (3, *space)). |
| `thread_line` | `(settings)` |  |
| `check` | `(cfg)` | The refusals of ``run`` that come before its first step, without arrays or steps: ``build(cfg, dry=True)``. |
| `build` | `(cfg, saved=None, dry=False)` | Build everything a run needs from the resolved config; no step is taken.  ``saved``: the per-field arrays of ... |
| `explain` | `(cfg)` | Text: which engine functions the file selects, the dt / kT choices known without stepping, the stages. |

### `run_figures.py`

| public name | signature | what it does |
|---|---|---|
| `signed_rgb` | `(p)` | fig_snap_20261106.signed_rgb: 0.5 + 0.5 p / \|p\| along the first axis, moved to the last. |
| `domain_map` | `(path, kind, P, grid, engine, title, prm=None, fractions=None)` |  |
| `spectrum` | `(path, tables, quantities, title, calibrated=False)` |  |

*public constants:* `PALETTE`, `PALETTE_RTO`, `DPI`

### `run_input.py`

| public name | signature | what it does |
|---|---|---|
| `deck_problem` | `(name, d)` | None when the deck dict ``d`` has what a run reads (the Landau sector with its six sixth-order coefficients, gradient, ... |
| `resolve` | `(raw, base_dir='.')` | Validate ``raw`` (a parsed TOML dict) and return a new fully resolved dict. |
| `load` | `(path)` | Read the TOML file ``path`` and return the resolved config dict. |
| `dumps` | `(cfg)` | Return TOML text of a resolved config; ``resolve(tomllib.loads(dumps(cfg)))`` equals ``cfg``. |

*public constants:* `BUILT_STAGES`, `BUILT_MEASURES`, `BUILT_WAVEFORMS`, `RTO_MAX_ANGLE`, `SCHEMA`, `P_SECTORS`

### `run_measure.py`

| public name | signature | what it does |
|---|---|---|
| `sample` | `(adapter, P, E, measure, axis=None)` | One reading: dict(mean_P=[3 floats] if measure["mean_P"], Pn, E_total, f_total, terms). |
| `scheme_of` | `(prm)` | "rt" or "rto": the resolved phase_fractions table of scheme "rt" carries no ``scheme`` key. |
| `fraction_keys` | `(prm)` |  |
| `describe` | `(prm)` | One line for ``explain``: the scheme and its thresholds. |
| `classes_rto` | `(P, prm)` | Boolean masks R, O, T, other, low of the field P (3, ...) in C/m^2, scheme "rto" (module docstring). |
| `classes` | `(P, prm)` | Boolean masks of the field P (3, ...) in C/m^2.  Scheme "rt": R, T, T15, other, low (readout_nbt6bt.classify, operation ... |
| `fractions` | `(P, prm, qw=None)` | Fractions per class (keys fraction_keys(prm)) of the field P (3, ...) in C/m^2; ``prm`` is the measure.phase_fractions table. |
| `phase_fractions` | `(adapter, P, prm)` | fractions() of a state of ``adapter`` (film: weighted by grid["qw"]). |
| `phase_index` | `(P, prm)` | Class index per cell.  Scheme "rt": 0 R, 1 T15, 2 T not T15, 3 other, 4 low (the precedence of fig_snap_20261106.py); ... |
| `force_max` | `(adapter, P, E)` | max\|L * dF/dP\| on the grid -- casekit.anneal's residual (pf2d / pf3d) and the ... |
| `triangle_cycle` | `(E, Pn, m)` | All triangle measures from one cycle's 4m + 1 points (E in kV/cm, Pn in C/m^2); see the module docstring. |
| `loop` | `(samples, st, names, every)` | The requested ``names`` for a pe_loop stage ``st`` from its recorded ``samples`` (dicts with step, E_kV_cm, Pn). |
| `stage_events` | `(st)` | ([(event name, step in the stage)], total steps) of a cool / hold / pe_loop stage: the states it saves. |
| `saved_states` | `(cfg)` | Every state file a run of ``cfg`` writes, in stage order: dicts(i, kind, event, step, total, stem). |
| `domain_map_targets` | `(cfg)` | The saved states to draw (list of the dicts of saved_states), from measure.domain_map.at. |
| `eps0_deck` | `()` |  |
| `total_along` | `(P, n)` | The raw observable s = sum over cells of P . n (n a unit vector as a list of floats); exactly P[k].sum() for n = e_k. |
| `lockin_deck` | `(series, dt, omega, E0_deck, n_cells, n_skip=0)` | (chi', chi'') per volume in deck units from the raw series of one tone. |
| `fdt_deck` | `(series, dt, kT, n_cells, nperseg, max_lag)` | The FDT route of one equilibrium series: dict(omega, chi_pp (Welch), chi_p (fl_chi), C, C_blocks, tau, kT_eff, ... |
| `spectrum_tables` | `(recs, routes, seconds_per_deck_time=None)` | Spectrum tables from the per-replica records (one dict per replica: route -> dict of arrays as the stages saved them). |

*public constants:* `FRACTION_KEYS`, `FRACTION_KEYS_RTO`, `N_BLOCKS`

### `run_output.py`

| public name | signature | what it does |
|---|---|---|
| `git_commit` | `()` | HEAD hash, with "+dirty" appended when the tree has changes, "unknown" without git. |
| `identity` | `(cfg)` | sha256 of the resolved config without ``output``, plus the sha256 of every from_file input. |
| `existing_dir` | `(cfg)` | The output directory when it is already named and present (a resume candidate), else None. |
| `save_npz` | `(path, **arrays)` | Atomic: a temporary file, fsync, then rename. |
| `find_checkpoint` | `(cfg, ident)` | None, or dict(dir, P, P_prev, meta) of the newest readable checkpoint of this config in output.dir. |
| `load_quenched` | `(d)` | The per-field arrays saved by an earlier build in ``d`` ({'field_<i>': array}), or None if no field was saved. |
| `check_dir` | `(cfg)` | ConfigError (output.dir) when the run directory cannot be created or written; creates nothing. |
| `make_dir` | `(cfg)` | Create and return the run directory named by cfg["output"]["dir"] (a ConfigError on output.dir when it cannot be made). |
| `sample_from_row` | `(row)` | A sample dict (the form run_protocol makes) from one trace.csv row (strings). |
| `load_spectrum` | `(d, cfg)` | {route: dict of arrays} of the spectrum stages saved in run directory ``d`` (one stage per route; ConfigError if missing). |
| `write_spectrum` | `(path, cfg, dirs)` | spectrum.csv and figures/spectrum.png in ``path`` from the spectrum stages of the run directories ``dirs`` (one per ... |

### `run_protocol.py`

| public name | signature | what it does |
|---|---|---|
| `new_ctx` | `(cfg)` | The state carried from stage to stage: the run's one RNG, steps done, the applied field E [kV/cm] along the unit axis n. |
| `anneal` | `(b, st, P, t0=0.0, ctx=None)` | Fixed-T stage (Langevin when ``st["langevin"]``), reproducing casekit.anneal bit for bit. |
| `relax` | `(b, st, P, t0=0.0, ctx=None)` | Deterministic stepping (numerics.scheme) until the film relax rule or ``stop.n_max`` steps. |
| `deck_field` | `(E_kV_cm)` | kV/cm -> deck field (1 kV/cm = 1e5 V/m; units.si_to_deck_field). |
| `drive_field` | `(b, amp, n)` | Static E_ext plus the uniform deck field ``amp`` along the unit axis n. |
| `efield` | `(b, E_kV_cm, n)` | Total deck field E(t) n + static E_ext (driver B ``Model.field``); E = 0 gives the static field alone. |
| `unit_axis` | `(axis)` | The axis normalised as driver B normalises n. |
| `triangle` | `(s, sps)` | +1 at s = sps, 0 at 2 sps, -1 at 3 sps, 0 at 4 sps; exact at the turning points and zero crossings (driver B). |
| `default_every` | `(quarter)` | Largest divisor of the quarter cycle that is <= DEFAULT_RECORD. |
| `stage_T` | `(b, st)` | The temperature a stage starts at (None: keep the current one). |
| `schedule` | `(b, st, ctx)` | A scheduled stage as functions of its step s: (total, T(s), E(s) [kV/cm], event(s) -> name or None, record_every, axis). |
| `drive` | `(b, st, P, ctx, i, out, resume=None)` | The one stepping loop of cool / hold / pe_loop (driver B ``_run``). |
| `ac` | `(b, st, P, t0=0.0, ctx=None)` | Small-signal lock-in tones (a stage of run_measure's "dielectric spectrum" definitions). |
| `fdt` | `(b, st, P, t0=0.0, ctx=None)` | Equilibrium fluctuation route: ``burn_steps`` steps, then ``n_steps`` steps recording the raw series s_i = sum(P . n) ... |
| `summary` | `(i, res, st, cfg, fractions_at=None)` | The measures.json entry of a finished stage (plus its loop measures). |
| `check` | `(cfg)` | What the CLI ``check`` and ``explain`` do first: every refusal ``run`` raises before its first step (the dry build of ... |
| `run` | `(cfg_or_path, stop_after_checkpoints=None)` | Run the file (path) or resolved config dict: run_one, or run_replicas when run.replicas > 1. |
| `run_one` | `(cfg, input_path=None, stop_after_checkpoints=None)` | Run one resolved config; returns dict(run_dir, build, stages, P, n_checkpoints). |
| `replica_cfg` | `(cfg, r, top, dt=None)` | The config of replica r: run.seed + r, and r added to every stage seed and every random / white start seed (those ... |
| `run_replicas` | `(cfg, input_path=None, stop_after_checkpoints=None)` | R = run.replicas independent complete runs, one after another, replica r in <top>/rep_<r>/ with the seeds of ... |

*public constants:* `CHECK_EVERY`, `PLATEAU_WINDOWS`, `DELTA_P_TOL`, `F_RATE_TOL`, `DEFAULT_RECORD`, `STAGES`, `DRIVEN`

### `spectral_operators.py`

| public name | signature | what it does |
|---|---|---|
| `k_axis` | `(N, h)` | Angular wavenumbers of an N-point axis of spacing h, fftfreq order. |
| `k_axis_half` | `(N, h)` | Angular wavenumbers of the halved (rfft) axis: N//2+1 bins, k >= 0. |
| `k_mesh` | `(*k1d)` | Broadcast 1D wavenumber axes given in (x, y[, z]) order to full ... |
| `half_spectrum` | `(grid, dim)` | The grid's half-spectrum wavevector arrays (KXr, KYr[, KZr]) -- the ... |
| `nyquist_index` | `(grid, axis)` | Index of `axis`' Nyquist bin (N//2) if its length N is even, else None. |
| `mask_nyquist` | `(sym, grid, axes)` | Zero `sym` IN PLACE on the Nyquist line/plane of every EVEN axis in ... |
| `masked_wavevectors` | `(grid, dim)` | Real skew-adjoint first-derivative wavevectors: copies of ... |
| `masked_wavevectors_2d` | `(grid)` | masked_wavevectors(grid, 2): (KX, KY) on the rfft2 half-spectrum. |
| `masked_wavevectors_3d` | `(grid)` | masked_wavevectors(grid, 3): (KX, KY, KZ) on the rfftn half-spectrum. |
| `derivative_symbols` | `(grid, dim)` | (i*kx, i*ky[, i*kz]) with the masked wavevectors: the first-derivative ... |
| `derivative_symbols_2d` | `(grid)` | derivative_symbols(grid, 2) -- the film's in-plane odd symbols (its ... |
| `k_squared` | `(grid, dim)` | \|k\|^2 = kx^2 + ky^2 [+ kz^2] on the half-spectrum, unmasked (even). |
| `isotropic_symbol` | `(grid, coeff, dim)` | coeff*\|k\|^2, the scalar isotropic gradient symbol shared by every ... |
| `hog_symbol` | `(grid, dim)` | kx^4 + ky^4 [+ kz^4]: the component-wise H11 multiplier (NC2025 eq.3, ... |
| `add_hog` | `(diagonals, H11, k4)` | (A11 + H11*k4, A22 + H11*k4, A33 + H11*k4): H11 joins the three ... |
| `grad_symbols` | `(grid, G11, G12, G44, G44p, dim)` | The base-G gradient operator A(k) as a dict {(i, j): array}, i <= j, ... |
| `grad_symbols_2d` | `(grid, G11, G12, G44, G44p)` | (A11, A22, A12, A33) on the rfft2 half-spectrum -- the 2D engine's and ... |
| `grad_symbols_3d` | `(grid, G11, G12, G44, G44p)` | (A11, A22, A33, A12, A13, A23) on the rfftn half-spectrum. Was ... |
| `grad_nyquist_coefficients` | `(G11, G44, G44p, dim)` | a[i, j], the coefficient of the (P_i, axis j) Nyquist energy term of ... |
| `depolarization_symbols` | `(grid, dim)` | D_ij = k_i k_j / \|k\|^2 as a dict {(i, j): array}, i <= j < dim, with ... |
| `depolarization_symbols_2d` | `(grid)` | (D11, D22, D12) on the rfft2 half-spectrum (solver.build_operators). |
| `depolarization_symbols_3d` | `(grid)` | (D11, D22, D33, D12, D13, D23) on the rfftn half-spectrum ... |
| `green_operator` | `(grid, C, dim)` | Omega(k) = [C_ijkl k_j k_l]^-1 built from the full (3,3,3,3) tensor C ... |
| `nyquist_project_axis` | `(field, axis, N)` | Project one real field onto the alternating (Nyquist) pattern along a ... |
| `add_nyquist_correction` | `(fg, fields, grid, coeff, dim)` | fg plus, for every EVEN axis j (x, y[, z] order) and component i, ... |
| `ensemble_field` | `(v, n_lead, ndim, what)` | Align a P-INDEPENDENT field with a BATCHED quantity (ensemble axis S ... |
| `vec3_field` | `(v, ndim, ensemble=False)` | The periodic engines' E_ext normaliser (solver, bulk3d, the sectors). |
| `vec3_field_checked` | `(v, shape)` | The film engine's E_ext normaliser -- a STRICTER policy than ... |

*public constants:* `AXES`

### `state_io.py`

| public name | signature | what it does |
|---|---|---|
| `save_state_npz` | `(path, P, grid, step, t, prm=None, alpha=None, eps_star_static=None, *, restart=None, fields=None)` | Save full state to a single .npz (format v2, module docstring): P, ... |
| `load_state_npz` | `(path)` | Inverse of save_state_npz -- returns a plain dict of whatever arrays ... |
| `provenance` | `(input_paths=())` | Versions and SHA256 of inputs plus the current solver source (dirty ... |
| `save_quicklook_png` | `(path, P, grid, title='')` | Quick-look PNG: P1, P2, P3, \|P\| side by side. |
| `save_vtk` | `(path, field_dict, grid)` | Minimal legacy VTK ASCII STRUCTURED_POINTS writer (no new deps). |

### `timestep_adaptive.py`

| public name | signature | what it does |
|---|---|---|
| `check_stab` | `(stab)` | S as a float; S must be finite and >= 0. |
| `tag_operator` | `(op, stab)` | Record the S an operator was built with (dict key or ndarray view). |
| `operator_stab` | `(op)` | The S an operator was built with (0.0 for every untagged operator). |
| `stabilized_operator` | `(op, stab, where)` | Refuse an operator built with a different S; return it untagged. |
| `landau_lipschitz` | `(alpha, p_max, n=33)` | Estimate ell = sup \|\|d^2 f_bulk/dP^2\|\| (spectral norm of the 3x3 Landau ... |
| `engine_kind` | `(engine)` | 'solver' \| 'bulk3d' \| 'film3d' (by module name), else ValueError. |
| `integrate` | `(engine, P, alpha, ops, grid, *, dt0, L=1.0, E_ext=(0.0, 0.0, 0.0), elastic_params=None, stab=0.0, err_tol=0.001, k_min=-8, k_max=6, k_start=0, dt_max=None, grow_after=10, energy_check='sample', energy_rtol=1e-12, ceiling_hold=4, n_max, t_end=None, sample_every=10, criterion='legacy', tol=0.01, n_blocks=4, block_size=8, energy_scale=None, polarization_scale=None, residual_tol=1e-06, dPdt_tol=np.inf, rate_tol=np.inf, n_windows=5, max_cache=24, restart=None, callback=None)` | Adaptive-dt SBDF2 relaxation of a deterministic gradient flow (see the ... |

*public constants:* `CRITERIA`, `ENERGY_CHECKS`

### `timestep_verify.py`

| public name | signature | what it does |
|---|---|---|
| `trajectory` | `(engine, P0, alpha, ops, grid, *, dt, n_steps, L=1.0, E_ext=(0.0, 0.0, 0.0), elastic_params=None, scheme='sbdf1', P_previous=None, stab=0.0)` | Fixed-step deterministic trajectory; returns (P, previous P). |
| `verified_dt` | `(engine, P0, alpha, ops, grid, *, L=1.0, E_ext=None, T_total, elastic_params=None, dt0=0.02, n_probe=200, growth=2.0, p_cap=None, max_doublings=30, scheme='sbdf1', tol=0.01, max_halvings=10, eps=1e-12, run_fn=None, skip_probe=False, observable=None, stab=0.0)` | Verify a stable, converged dt for `engine` on this deck/IC/forcing. |
| `verified_ensemble_dt` | `(run_fn, *, dt0, seeds, scales, T_total, tol=0.05, max_halvings=8, confidence_z=3.0)` | The same ladder for a STOCHASTIC protocol (V2-02.1 E2, additive). |

### `units.py`

| public name | signature | what it does |
|---|---|---|
| `si_to_deck_landau` | `(value_si)` | Any Landau coefficient a_n (multiplies P^(2n) in an energy-density ... |
| `si_to_deck_elastic` | `(value_si_pa)` | Elastic stiffness C_ij [Pa] -> deck units (numerically GPa, since ... |
| `si_to_deck_gradient` | `(value_si)` | SPEC S34.2 (B2-A): SI -> deck conversion for a gradient-energy coefficient ... |
| `cubic_stiffness_to_compliance` | `(c11, c12, c44)` | Inverse of cubic_compliance_to_stiffness -- for cubic symmetry the ... |
| `kT_langevin` | `(T_kelvin, hx_nm, hy_nm, hz_nm, k_B=K_BOLTZMANN_SI)` | THE Langevin FDT coefficient -- ONE convention for all three engines ... |
| `si_to_deck_kT` | `(T_kelvin, hx_nm, k_B=K_BOLTZMANN_SI)` | Unit conversion of k_B*T into F_total units -- NOT the Langevin FDT ... |
| `si_to_deck_theta` | `(value_si_m)` | SPEC S28.1/S28.5 R2: theta is a LINEAR OXYGEN DISPLACEMENT (meters, Chen's own Rabe App A ... |
| `si_to_deck_theta_angle_deg` | `(value_deg)` | WP18 Stage 2 (SPEC S28.7 R2): a SECOND, per-deck theta convention -- ... |
| `si_to_deck_theta_landau` | `(value_si, theta_power)` | Companion rescaling for any Landau-type coefficient multiplying theta^theta_power ... |
| `cubic_compliance_to_stiffness` | `(s11, s12, s44)` | Standard cubic-symmetry Voigt elastic compliance -> stiffness ... |
| `si_to_deck_field` | `(value_v_per_m)` | E [V/m] -> deck field; -E.P rescales like an energy density. |
| `deck_to_si_field` | `(value_deck)` | Deck field -> V/m (inverse of si_to_deck_field, x1e9). |
| `si_to_deck_permittivity` | `(value_f_per_m)` | Absolute permittivity [F/m] -> deck kappa via P^2/(2*kappa). |
| `deck_to_si_permittivity` | `(value_deck)` | Deck kappa -> absolute permittivity [F/m]. |
| `relative_to_deck_permittivity` | `(eps_r)` | Relative background permittivity -> kappa (0.0088541878128*eps_r). |

*public constants:* `SI_TO_DECK_ENERGY`, `K_BOLTZMANN_SI`, `THETA_LEN_SCALE`, `EPSILON_0_SI`

### `wall_metrics.py`

| public name | signature | what it does |
|---|---|---|
| `parse_tanh_params` | `(expr)` | 'A*tanh((H-abs(x))/W)' -> (A, H, W). |
| `periodic_tanh_wall_1d` | `(grid, amplitude, width, wall_frac=0.25)` | Two-wall periodic tanh profile matching k_t180.i/k_r109_*.i's IC convention ... |
| `total_density_field` | `(P, alpha, ops, E_ext, grid, elastic_params=None)` | Per-point total energy density (Ny,Nx), summing solver.py's per-term ... |
| `wall_sigma` | `(P, alpha, ops, E_ext, grid, f_uniform, n_walls=2, elastic_params=None)` | sigma_w in mJ/m^2 (see module docstring for the unit derivation). |
| `fit_wall_width` | `(x, profile, x0, half=2.0)` | curve_fit A*tanh((x-x0)/xi) near x0 (same form/window as wall_metrics.wall()). |
| `extrapolate_1_over_L` | `(Ls, vals)` | Linear least-squares fit vs 1/L; returns (intercept_at_Linf, slope). |
| `setup_case_elastic` | `(grid, prm)` | elastic.setup_elastic from a case prm dict's C11/C12/C44 + Q11_phys/Q12_phys/Q44_phys (elastic.py's Voigt Q, no /4). |
| `make_tie` | `(i, j)` | post_step hook: force P[i]=P[j]=average(P[i],P[j]) every step -- mirrors ... |
| `compute_f_uniform` | `(prm, P0_guess, n_max=20000, tol_F=1e-13, n_plateau=30, scheme='sbdf1', dt_tol=1e-08)` | Relax a tiny homogeneous-IC grid (with elasticity active) to get the ... |
| `run_wall_series` | `(prm, ic_builder, Ls, n_max=8000, tol_F=1e-10, n_plateau=25, check_every=20, post_step=None, dt=None, scheme='sbdf1')` | Relax the periodic 2-wall IC at each box length in Ls (quasi-1D, Ny=1, ... |
| `track_walls` | `(field, grid)` | field: (Ny,Nx) real array (e.g. Pz), periodic in x, sampled at ... |
| `measure_walls` | `(field, grid)` | Top-level SPEC §10(g) entry point: wall count, mean period, roughness ... |
| `fit_sinusoidal_amplitude` | `(track)` | Fit x(y) = x0 + A*sin(2*pi*m*y/Ly + phi) to a tracked wall (m fixed ... |
