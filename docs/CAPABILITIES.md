# Capabilities and Limits (FFT-PF-V2)

This page answers four questions: **what the engine computes, what it does not, which materials it can
take, and what the claims rest on.**

**EVIDENCE RULE (binding).** Every capability below names the **core-gate item(s) that prove it**
(`gates/core/kN_*.py --list` enumerates all 205) and at least one **case that exercises it**
(`cases/*.py`). A claim with no item behind it is written **未验证 (unverified)** — never as a
capability. A statement in an old report does not become a capability by having been written down.

- Account of record: **`CORE 8/8 @ 2408137`, 622.8 s, 205 items** (SPEC V2-SYNC, D18).
- This page is dated 2026-09-28. Where it disagrees with `SPEC.md`, SPEC wins — it is the sole
  authoritative work-order log.
- How to run and how to write a case: `docs/QUICKSTART.md` / `docs/ENGINE_GUIDE.md`. Parameter values and
  provenance: `docs/PARAMETERS.md`. Function signatures: `docs/API_INDEX.md`. Schedule:
  `reports/ROADMAP_FFT-PF-V2.md`.

---

## §1 What the engine is, and the three solvers

**FFT-PF-V2** (`solvers.ENGINE_NAME`, `solvers.__version__ = 2.0.0-dev`) is a pure-numpy
**ferroelectric phase-field (TDGL / Ginzburg–Landau–Devonshire)** solver: semi-implicit Fourier-spectral
gradient-flow relaxation of a three-component polarization field **P(r)** (Chen & Shen splitting —
k-space linear operators implicit, nonlinear force explicit), with a free energy of Landau bulk +
gradient + electrostatic (depolarization) + Khachaturyan elastic terms, optionally extended by an AFE
`q` sector, an octahedral-tilt `θ` sector, a fourth-order gradient `H11`, Langevin thermal noise and
quenched disorder.

By user ruling (SPEC Part II ruling (3)) the **three solvers stay independent — there is no
dimension-generic core**: one physics term means three implementations and three sets of items.

| Solver | Geometry | Boundary conditions | Intended use |
|---|---|---|---|
| `solvers/solver.py` (+`elastic.py`) | 2D periodic box `Nx×Ny`; P keeps all three components; only in-plane wavevectors (no k3) | Periodic; electrostatic short-circuit convention `Ê_d(k=0)=0`; elastic k=0 either `stress_free` (⟨σ⟩=0) or `clamped` (ε̄=0); uniform misfit `eps_m` | Domain walls, domain structures, disorder / relaxor statistics, the energy-storage protocol, anything needing many repeats. **2D is a slab of stated thickness `Dz`** (§4) |
| `solvers/bulk3d.py` (+`bulk3d_spm.py`) | Fully periodic 3D box `Nx×Ny×Nz` | Periodic in all three directions; same two elastic k=0 conventions; k = 0 electrostatics: tin foil by default; optional depolarization factor N (`build_operators_3d(..., depol_N=)`, SPEC § 11); at Nz = 1 optional per-cell plane stress σ_i3 = 0 (`setup_elastic_3d(..., z_mode="plane_stress")`, SPEC § 12; default generalized plane strain, uniform ε_33) | Bulk single-crystal 3D domain morphology, 3D elasticity (full-tensor C(r)), batched ensemble S axis |
| `solvers/film3d.py` (+`film_elastic.py`, `film3d_langevin.py`, `film3d_spm.py`) | True thin film: **in-plane FFT ⊗ thickness finite difference**, film + deformable substrate stack | Periodic in-plane; non-periodic through thickness: free surface + short-circuit electrodes + substrate clamp (the Li & Chen, APL 88, 072905 (2006) geometry) | Film strain phase diagrams, dead layers, surface/thickness effects, C(r) inside the film |

All three engines share **one pointwise** Landau energy/force (`solver.bulk_energy_density` /
`bulk_force` — a hand-written `pow`-free Horner form re-verified symbolically against the sympy free
energy at every import), so 6th- and 8th-order Landau are the same code in all three, not three rewrites.

No build step, no `requirements.txt`, no CLI framework: numpy, scipy and sympy are required, matplotlib
only for figures. One small entry point exists, `python -m fftpf run | check | explain case.toml`, which runs a
whole calculation from an input file (§8; manual `docs/RUN_INPUT.md`). Every case and gate locates the repo root from its own `__file__` and runs from anywhere.

---

## §2 Physics-term matrix

One row per physics term, one column per solver. **✓** = an item backs it (items named as
`K<n>.<item>`); **✗** = the capability does not exist (with the WP where it is planned, or "not planned"
plus the reason); **未验证** = the code path may work but no item has ever exercised it, so it is not a
capability. Across the 60 cells: **53 ✓ / 5 ✗ / 2 未验证**.

| Physics term | `solver.py` (2D) | `bulk3d.py` | `film3d.py` |
|---|---|---|---|
| **Landau 6th / 8th order** (3-component) | ✓ `K1.fd_2d_all_terms`, `K3.homogeneous_ps_scipy`, `K5.bto8th_monodomain`, `K5.pto_monodomain`, `K5.pzt_monodomain`, `K5.landau0d_vs_horner` | ✓ `K1.fd_3d_all_terms`, `K3.homogeneous_equilibrium_3d`, `K2.bulk3d_nz1_vs_2d` | ✓ `K1.fd_film_full_coupling`, `K3.film_uniform_p_vs_tier1`, `K8.film_dynamic_crosscheck` |
| **Gradient energy · isotropic** | ✓ `K1.fd_2d_all_terms`, `K3.wall_t180_gld1d`, `K3.wall_t180_recorded` | ✓ `K1.fd_3d_all_terms`, `K3.dispersion_3d` | ✓ `K1.fd_film_full_coupling`, `K2.film_grad_isotropic_byte` |
| **Gradient energy · anisotropic** (G11/G12/G44/G44′) | ✓ `K1.fd_2d_all_terms`, `K3.wall_r109_recorded` | ✓ `K1.fd_3d_all_terms`, `K3.dispersion_3d` | ✓ `K1.fd_film_grad_aniso`, `K3.film_aniso_dispersion`, `K3.film_aniso_thick_bulk_limit` |
| **Electrostatics · homogeneous κ** | ✓ `K3.depolarization_closed_form`, `K3.linear_media_response` | ✓ `K1.fd_3d_all_terms`, `K3.dispersion_3d`; optional k = 0 depolarization factor N (SPEC § 11): `K1.fd_3d_depol_k0`, `K3.depol_k0_uniform_closed_form`, `K3.depol_k0_mode_rates`, `K2.depol_k0_off_byte`, `K2.depol_k0_ensemble_s1_byte`, refused with κ(x) `K2.depol_k0_spm_refused` | ✓ `K3.film_poisson_analytic`, `K3.film_electrode_screening`, `K3.film_sandwich_closed_form` |
| **Electrostatics · κ(x) inhomogeneous SPM** | ✓ `K1.fd_2d_kappa_x`, `K2.kappa_const_2d`, `K3.layered_dielectric_closed_form`, `K4.spm_residual` | ✓ `K1.fd_3d_kappa_x`, `K2.kappa_const_3d`, `K2.spm_nz1_vs_2d`, `K4.spm_residual_3d` | ✓ `K1.fd_film_kappa_xyz`, `K2.film_spm_kappa_z_only`, `K2.film_spm_thick_bulk_limit`, `K2.film_spm_dispatch_byte`, `K4.film_spm_residual` |
| **Electrostatics · κ(z) layers (dead layer)** | ✓ `K3.layered_dielectric_closed_form` (series-capacitor closed form) | ✓ `K3.layered_dielectric_closed_form_3d` | ✓ `K1.fd_film_kappa_z`, `K2.film_kappa_z_const_byte`, `K3.film_dead_layer_closed_form` |
| **Elasticity · homogeneous C** | ✓ `K1.fd_2d_elastic`, `K3.misfit_mode_closed_form`, `K3.eshelby_inclusion_2d`, `K4.elastic_sigma_residual` | ✓ `K1.fd_3d_elastic`, `K3.elastic_homogeneous_3d_closed_form`, `K3.eshelby_inclusion_3d`, `K2.bulk3d_nz1_vs_2d_elastic`, `K2.elastic_off_byte_baseline`; Nz = 1 plane stress (§ 12): `K1.fd_3d_plane_stress`, `K3.plane_stress_laminate_closed_form`, `K3.plane_stress_sigma_i3_zero`, `K3.plane_stress_uniform_zero`, `K2.plane_stress_off_byte` | ✓ `K1.fd_film_elastic`, `K3.film_surface_traction`, `K3.film_machinery_dense_solve`, `K2.film_uniform_p_thickness_independence` |
| **Elasticity · C(r) inhomogeneous SPM** | ✓ `K1.fd_2d_pzt_c_field`, `K2.dc_zero_degenerate`, `K4.elastic_sigma_residual` | ✓ `K1.fd_3d_elastic_spm`, `K2.elastic_spm_const_3d`, `K2.elastic_spm_nz1_vs_2d`, `K2.elastic_spm_fulltensor_equals_cubic`, `K3.laminate_closed_form_3d`, `K3.eshelby_inhomogeneity_3d`, `K4.elastic_spm_residual_3d` | ✓ `K1.fd_film_elastic_spm`, `K2.film_elastic_spm_off_byte`, `K2.film_elastic_spm_const`, `K3.film_laminate_closed_form`, `K4.film_elastic_spm_residual` |
| **Film misfit e0 + substrate clamp** | ✗ not planned: a periodic box has no surface. 2D has only a **uniform** misfit `eps_m` (✓ `K3.misfit_mode_closed_form`) | ✗ not planned: a fully periodic box has no surface. The two k=0 conventions (`stress_free`/`clamped`) are both run by `K1.fd_3d_elastic` | ✓ `K1.fd_film_elastic`, `K3.film_surface_traction`, `K3.film_sandwich_closed_form`, `K2.film_thick_slab_bulk_limit` |
| **Static eigenstrain `eps_star_static`** | ✓ `K1.fd_2d_disordered` | ✓ `K1.fd_3d_elastic_static`, `K2.eps_star_static_none_3d` | ✓ `K1.fd_film_elastic_static`, `K2.eps_star_static_none_film` |
| **AFE `q` sector (Kittel two-sublattice)** | ✓ `K1.fd_2d_afe`, `K2.afe_wiring`, `K3.afe_analytic_single_cell`, `K3.afe_field_thresholds_lum2022`, `K7.cycle_stationarity_0d` | ✓ `K1.fd_3d_afe`, `K1.fd_3d_afe_kappa_x`, `K2.afe_wiring_3d`, `K2.afe_nz1_vs_2d`, `K2.afe_spm_hybrid_3d`, `K3.afe_uniform_3d` | ✓ `K1.fd_film_afe`, `K2.afe_wiring_film` |
| **Octahedral tilt `θ` sector (AFD)** | ✓ `K1.fd_2d_afd`, `K2.afd_wiring`, `K3.afd_analytic_single_cell`, `K3.afd_pure_theta_anchor`, `K3.pzt_r3c_structural`, `K3.pzt_haun_quantitative` | ✓ `K1.fd_3d_afd`, `K1.fd_3d_afd_kappa_x`, `K2.afd_wiring_3d`, `K2.afd_nz1_vs_2d`, `K2.afd_spm_hybrid_3d`, `K3.afd_uniform_3d` | ✓ `K1.fd_film_afd`, `K2.afd_wiring_film` |
| **Fourth-order gradient `H11`** | ✓ `K1.fd_2d_hog`, `K2.hog_h11_zero_degenerate`, `K3.hog_dispersion_h11`, `K3.hog_stripe_wavelength`; Langevin on G<0 ops (indefinite Lin, SPEC §10 — before it the OU clipped λ<0 and lost the k* instability): `K3.ou_negative_mode_2d`, `K4.ou_negative_energy_monotone` | ✓ `K1.fd_3d_hog`, `K2.hog_h11_zero_degenerate_3d`, `K2.hog_nz1_vs_2d`, `K3.hog_dispersion_3d`; Langevin on G<0 ops: `K3.ou_negative_mode_3d`, `K4.ou_negative_energy_monotone` | ✓ `K1.fd_film_hog`, `K2.film_hog_h11_zero_byte`, `K3.film_hog_dispersion`, `K4.film_hog_stable_dt`; the film OU has no H11 and **refuses** G11<0 (`K2.film_ou_negative_g_refused`) |
| **Langevin thermal noise (+ cutoff ℓ_c)** | ✓ `K3.fdt_mode_variance`, `K3.fdt_equipartition`, `K3.fdt_time_correlation`, `K2.kt_zero_regression`, `K7.reproducibility_independence`, `K5.kt_convention`, `K3.langevin_cutoff_fdt`, `K3.langevin_cutoff_grid_invariance`, `K2.noise_cutoff_none_byte`, `K4.langevin_stable_dt`; negative eigenvalues of Lin kept, only roundoff floored (SPEC §10): `K2.ou_psd_byte`, `K3.ou_negative_mode_2d` | ✓ `K5.kt_convention`, `K3.langevin_cutoff_fdt`, `K3.langevin_cutoff_grid_invariance`, `K2.noise_cutoff_none_byte`, `K2.anneal_noise_cutoff_none_byte`, `K4.langevin_stable_dt`, `K8.demo_3d`; SPEC §10: `K2.ou_psd_byte`, `K3.ou_negative_mode_3d` | ✓ `K3.film_fdt_mode_variance`, `K3.film_fdt_equipartition`, `K2.film_langevin_kt0`, `K2.film_langevin_reproducibility`, `K5.kt_convention`, `K4.film_ou_cost`; SPEC §10: `K2.ou_psd_byte`, G11<0 refused `K2.film_ou_negative_g_refused` |
| **Quenched disorder (random field / composition / labels)** | ✓ `K5.field_statistics`, `K5.pmn_field_statistics`, `K5.label_field_statistics`, `K5.ternary_label_statistics`, `K5.quenched_field_equivalence`, `K5.composition_equivalence`, `K1.fd_2d_disordered`, `K1.fd_2d_pzt_fields`, `K1.fd_2d_composite`, `K1.fd_2d_ternary`, `K2.pzt_channel_wiring`, `K2.pmn_channel_wiring` | ✓ `K5.field_statistics_3d`, `K5.pmn_field_statistics_3d`, `K5.composition_field_statistics_3d`, `K2.generators_nz1_vs_2d`, `K1.fd_3d_elastic` (region-wise Q field) | **未验证**: the generators emit `(Nz,Ny,Nx)` fields and the film channel accepts that shape, but no item has ever taken this path |
| **Ensemble S axis (batched replicas)** | ✓ `K4.ensemble_batching` | ✓ `K2.ensemble_s1_byte_3d`, `K4.ensemble_batching_3d` | ✗ not built, not on the roadmap (not planned) |
| **SBDF1 / SBDF2 time stepping** | ✓ `K4.sbdf2_order`, `K4.timestep_equivalence`, `K4.adaptive_dt` | ✓ `K4.sbdf2_order_3d`, `K2.nz1_sbdf2_vs_2d` | ✓ `K4.film_stepping` (SBDF2 order), `K4.verified_dt_failure_paths` (both schemes) |
| **`verified_dt` (the three dt laws)** | ✓ `K4.verified_dt_default_path`, `K4.verified_dt_failure_paths`, `K4.timestep_equivalence` | ✓ `K8.demo_3d` (through `casekit` into `timestep.verified_dt`) | ✓ `K4.verified_dt_failure_paths`, `K4.film_stepping` |
| **Checkpoint / resume** | ✓ `K8.checkpoint_roundtrip` (sbdf1/sbdf2/Langevin), `K7.anneal_semantics` | ✓ `K8.checkpoint_roundtrip` (sbdf1/Langevin) | **未验证**: `io_utils`/`casekit` are engine-agnostic, but no item has ever taken the film path |
| **Energy-storage protocol E1 (unipolar)** | ✓ `K7.unipolar_short`, `K7.loop_closure`, `K3.storage_linear_dielectric`, `K3.storage_lossless_closure`, `K7.cycle_stationarity_0d` | ✗ V2-08 (energy storage across all engines) | ✗ V2-08 (energy storage across all engines) |

**One case per row** (proof that the path has actually been driven): 2D all terms `cases/t180.py`;
fully periodic 3D `cases/b2c_science_exit_3d.py`; full film coupling `cases/film_tier3_fig1.py`;
κ(x) dead layer `cases/film_dielectric.py`; C(r) elasticity `cases/pzt_elastic_spm_g14e.py`;
`eps_star_static` `cases/demo_disorder.py`; AFE `cases/wp20_triplewell_case.py`; AFD
`cases/wp18_afd_case.py`; H11 `cases/wp23_slush_case.py`; Langevin + disorder
`cases/w0_relaxor_demo.py`, `cases/wp16_pmn_pilot.py`; storage E1 `cases/energy_protocol.py` (the
library lives in `solvers/`); end-to-end casekit `cases/demo_casekit_2d.py` (2D) /
`cases/demo_casekit_3d.py` (bulk3d).

---

## §3 Materials: what can be computed, what cannot

### 3.1 What a material must provide to be computable

Following the free-energy form of SPEC Part I §1, the **minimum computable set** (missing any of these
means the run does not start, or runs only on placeholder numbers):

| Required | Deck key | What is lost without it |
|---|---|---|
| Landau coefficients to 6th order (cubic, 3-component: `a1,a11,a12,a111,a112,a123`); 8th order optional (`a1111..a1123`, default 0) | `[landau]` | No free energy — nothing can be computed |
| Gradient coefficient `G110` + the ratios `G11/G12/G44/G44'` | `[gradient]` | No domain-wall width / wall-energy scale. `G110` is simply not published by most sources (`docs/PARAMETERS.md` §C.3) |
| Background permittivity `permittivity` → engine key `kappa_b` | `[electrostatics]` | The depolarization field cannot be solved |
| Electrostriction `Q11/Q12/Q44` + elastic stiffness `C11/C12/C44` | `[electrostriction]`, `[elastic]` | Elasticity must stay off (`elastic_params=None`); no strain phase diagram, no misfit effects |

**Per-sector extras.** AFE needs the whole Kittel two-sublattice set (`[landau_q]`, `[coupling_pq]`,
`[gradient_q]`, `[eigenstrain_q]`); AFD needs the tilt sector (`[landau_theta]`, `[coupling_Ptheta]`,
`[gradient_theta]`) **plus a declared unit convention for θ** — the two existing decks differ by nine
orders of magnitude (one uses metres, the other degrees); the fourth-order gradient needs
`[higher_order_gradient].H11`; a relaxor needs a **disorder model** (stoichiometry + screening length +
correlation length), not a single number. Temperature enters by exactly one route: a closed form or table
for `a1(T)` (and where needed `a11(T)`, `a111(T)`) — **gradient and elastic coefficients are
T-independent in this engine** (§3.4).

### 3.2 The 10 active decks (`params/*.toml`) and the 9 archived ones (`params/archive/`)

Four validation classes: **closed-form / literature anchor** (a K3 item pins it to a closed form or a
literature number), **behaviour deck** (it only drives machinery; the numbers themselves are dials or
placeholders), **literature transcription only** (sourced numbers, but no item ever compares them to an
independent result), **placeholder / rejected**.

**Archived 2026-11** (user decision: the built-in material parameters are of little value): the nine
decks that no computation loads were moved with `git mv` to `params/archive/`, contents unchanged —
second table below, reasons and restore steps in `params/archive/README.md`. Three of them are TOML
snapshots of generators that stay live in `solvers/materials.py` (`bto_lcc2005_8th`, `pto_haun1987`,
`pzt_haun_x`), so archiving the file removes no capability. The registry entries nothing called —
`bto_bc1984_6th`, the `bfo/bto/sto_pan2019` deck wrappers, `pmn_clean_x0`, `pmn_pt_placeholder` — moved
to `solvers/materials_archive.py` (`materials_archive.get(name)`; nothing in the engine imports it, and
`materials.get(<archived name>)` raises a `KeyError` that points there). Substrate references
(`params/substrates/`, five elastic-only decks) are covered in §7.

Active decks (a case, run, solver path or test reads the file or calls its generator):

| Deck | Material | Phase / class | Source | Validation status | Used by |
|---|---|---|---|---|---|
| `bto_298K.toml` | BaTiO₃ 298 K | tetragonal FE | Ferret deck `k_t180.i` (WP1 import) | **anchor**: `K3.wall_t180_gld1d` (σ_w 0.33 %), `K3.wall_t180_recorded`, `K3.homogeneous_ps_scipy`, `K3.misfit_mode_closed_form` | all three engines; `cases/t180.py`, `demo_disorder.py`, `wp13_debye_case.py` |
| `bto_118K.toml` | BaTiO₃ 118 K | rhombohedral FE | Ferret deck `k_r109_L64.i` | **anchor**: `K3.wall_r109_recorded` (109° wall, σ_w 0.026 %) | 2D; `cases/r109.py` |
| `bto_LCC2005_paper.toml` | BaTiO₃ | film T / O / M | Li & Chen, APL 88, 072905 (2006), its own ref-18 footnote + Eqs 3/7 | **anchor**: `K3.film_uniform_p_vs_tier1`, `K8.film_dynamic_crosscheck`, `K1.fd_film_full_coupling`; `G110` back-derived from dx = 0.4 nm (a declared free parameter) | film3d; `cases/film_tier1_scan.py`, `film_tier3_fig1.py`, `wp19_rt_strain_series.py`; `cases/bto_film_biaxial.py` / `continue_bto_film.py` hash the file into run provenance |
| `bfo_pan2019.toml` | BiFeO₃ end member | rhombohedral (R) | Pan et al., Science 365, 578 (2019) SM; true source Zhang et al., JAP 101, 114105 (2007) | **anchor**: `K3.ternary_endmember_0d` (independent multi-start 0D minimization → R) | 2D/bulk3d; `cases/b2a_r1_binary.py`, `b2c_science_exit_3d.py`; `cases/b2e_pan2019_3d_case.py` reads the `[elastic]`/`[electrostriction]` of all three Pan decks from the files |
| `bto_pan2019.toml` | BaTiO₃ end member | tetragonal (T) | same SM | **anchor**: `K3.ternary_endmember_0d` (→ T); **on-record conflict**: `alpha111` Pan-printed vs the Sluka lineage (`[meta].discrepancy_flag`, traced but not ruled) | 2D/bulk3d; `cases/b2a_r2_ternary.py` |
| `sto_pan2019.toml` | SrTiO₃ end member | cubic paraelectric (C) | same SM | **anchor**: `K3.ternary_endmember_0d` (→ C); **on-record conflict**: Pan's transcription of its own cited source differs by 1.0 K in T₀ (`[meta].stoAlpha1_flag`) | 2D/bulk3d; `cases/b2a_r1_binary.py` |
| `sto_afd_rabeA.toml` | SrTiO₃ | cubic + AFD tilt | Rabe App A, Table 3 (multiple sources, undifferentiated per row) | **anchor**: `K3.afd_analytic_single_cell`, `K3.afd_pure_theta_anchor`; θ is a **linear displacement in metres** | 2D; `cases/wp18_afd_case.py` |
| `pbzo_kittel_lum2022.toml` | PbZrO₃-flavoured | antiferroelectric (Kittel two-sublattice) | Lum, Lim & Chew, JPCM 34, 415702 (2022), Table 1 (open access) | **anchor**: `K3.afe_analytic_single_cell`, `K3.afe_field_thresholds_lum2022` (field thresholds), `K7.cycle_stationarity_0d`; the vector extension is a **diagonal embedding**, not full O(3) | 2D; `cases/archive/science_exits/g20e_double_hysteresis.py` |
| `pbzo_triplewell_yang2025.toml` | PbZrO₃ line | AFE triple well (enlarged cell, single P) | Yang et al., Nature 637, 1104 (2025), SI Table 1 | **behaviour deck + partial closed form**: `K3.triplewell_fitter_roundtrip`, `K7.loop_closure`, `K7.unipolar_short`; **gaps**: `kappa_b = 50` unresolved by the main text (a name collision is recorded), G borrowed from BTO (placeholder) | 2D; `cases/wp20_triplewell_case.py` |
| `nnbm_wang2026.toml` | (1−x)NaNbO₃–xBM5 | relaxor / slush (AFE stripes) | Wang et al., Nano Energy 156, 112129 (2026) SI + NC 2025 Methods | **behaviour deck**: `K3.hog_stripe_wavelength`, `K1.fd_2d_hog` validate the H11 machinery; `[meta]` states that T = 180 K, `G11`, `H11` and ξ are all **dials** — a qualitative/schematic construction, not a fitted NN-BM5 potential | the H11 sector of all three engines; `cases/wp23_slush_case.py`, `w3_tsweep_case.py` |
| *(NBT-6BT)* | Na₀.₅Bi₀.₅TiO₃–6BaTiO₃ | relaxor | the codex side branch's deck | **REJECTED** (SPEC V2-02.1): at its sole held-out point, 323.15 K, P_r 0.2322 vs 0.281 and E_c 1.908 vs 2.80 MV/m; the fit is under-determined and non-unique. **No nbt deck exists anywhere in the repo**; it can enter only through V2-11's re-fit to lens 3's recipe R1–R9 | none |

Archived decks (`params/archive/`; validation status as it stood when archived — the K items named
here live in the local-only `gates/`, which may still read these files by their old `params/` path):

| Deck | Material | Phase / class | Source | Validation status | Live generator / consumer |
|---|---|---|---|---|---|
| `archive/bto_BC1984_6th.toml` | BaTiO₃ | tetragonal (6th-order) | Rabe App A, Table 1 (6th-order column) | **transcription only**: `[meta]` says REFERENCE ONLY, not fit to reproduce `k_t180.i`; only `K5.materials_equivalence` / `K5.deck_schema_dualpath` | none; generator archived too: `materials_archive.get("bto_bc1984_6th")` |
| `archive/bto_LCC2005_8th.toml` | BaTiO₃ | T/O/R (8th-order potential) | Rabe/Ahn/Triscone 2007 App A, Table 1 (8th-order column) | **anchor**: `K5.bto8th_monodomain` (C→T, T→O, Ps(298 K)); **exception**: the potential's own O→R is 202.5 K, not the experimental 183 K — unsuitable for O–R studies (stated in its `[meta]`) | generator live: `materials.get("bto_lcc2005_8th")` (2D; `cases/demo_casekit_2d.py` and others) — the file itself is read by nothing |
| `archive/pto_Haun1987.toml` | PbTiO₃ | tetragonal FE | Rabe App A, Table 5 | **anchor**: `K5.pto_monodomain`; **gap**: `G110`/`permittivity` are None (placeholders), G ratios borrowed from BTO | generator live: `materials.get("pto_haun1987")` / `literature_params.pto_haun1987` (`cases/pzt_g_calibration.py`) |
| `archive/pzt_Haun_x0.6.toml` | PZT (x = 0.6) | tetragonal (Ti-rich side of the MPB) | Rabe App A p.367 formula set + Table 4 | **anchor**: `K5.pzt_monodomain`, `K2.composition_const_degenerate`; **gap**: `G110`/`permittivity` None | generator live: `materials.get("pzt_haun_x", x=…)` (`cases/pzt_g_calibration.py`, `wp11_matrix.py`) |
| `archive/pzt_r3c_haun_x02.toml` | PZT (x = 0.2, Zr-rich) | R3c (FE + tilt) | Haun et al., Ferroelectrics 99, 55 (1989) part IV, Table | **anchor**: `K3.pzt_haun_quantitative` (quantitative tilt anchor), `K1.fd_2d_afd`; θ is in **degrees** | none in `cases/`/`Runs/` (hand-composed; read by path only by the local gates) |
| `archive/pzt_r3c_g22e_model.toml` | PZT Zr-rich (model material) | R3c | **composite model material**: P sector from Haun, θ sector borrowed verbatim from STO | **behaviour deck**: `K3.pzt_r3c_structural` pins structural conclusions only; `[meta]` states it is not a literature PZT-tilt deck | none in `cases/`/`Runs/` (hand-composed; K4 numerics items in the local gates) |
| `archive/pmn_clean_x0.toml` | PMN (x_PT = 0) | relaxor, clean limit | Viehland et al., PRB 46, 8003 (1992) (`alpha1` only) | **partial closed form**: `K3.curie_weiss_clean_control` (instrument chain recovers C_CW), `K5.clean_controls`; **gaps**: `alpha11` is an order-of-magnitude bracket (not PMN-derived), `alpha111/112/123` REQUIRED | the live formula is `pmn_field.pmn_clean_alpha1` (`cases/wp16_clean_control.py`, `wp16_pmn_pilot.py`); the deck wrapper is archived: `materials_archive.get("pmn_clean_x0")` |
| `archive/pmn_disorder_R3.toml` | PMN heterovalent chemical disorder | disorder model (**not a materials deck**) | WP15 derivation record (no `[landau]`/`[elastic]` schema) | **closed-form / statistical anchor**: the fields it derives are validated by `K5.pmn_field_statistics` (+`_3d`), `K2.pmn_channel_wiring` | none — a record; the disorder layer it documents is live in `pmn_field.py`/`quenched_fields.py` (`cases/w0_relaxor_demo.py`) |
| `archive/pmn_pt_landau_x0.30.toml` | 0.70PMN–0.30PT | rhombohedral / MPB (target) | Zhang et al., PRB 96, 054109 (2017) — **citation confirmed, coefficients not transcribed** (paywalled) | **placeholder**: every field REQUIRED; cannot be computed | none (waiting on V2-11 / V2-09); `materials_archive.get("pmn_pt_placeholder")` |

**Counts** — active: closed-form/literature anchor 8; behaviour deck 2 (`pbzo_triplewell_yang2025` also
carries closed-form legs). Archived: closed-form/literature anchor 5 (`bto_LCC2005_8th`, `pto_Haun1987`,
`pzt_Haun_x0.6`, `pzt_r3c_haun_x02`, `pmn_disorder_R3`); behaviour deck 1 (`pzt_r3c_g22e_model`);
partial closed form 1 (`pmn_clean_x0`); literature transcription only 1 (`bto_BC1984_6th`); placeholder
1 (`pmn_pt_landau_x0.30`). Plus NBT-6BT, **rejected**, with no file in the repo. Per-key values, units
and provenance grades are in `docs/PARAMETERS.md` §K (generated from `params/*.toml` and
`params/archive/*.toml` by `tools/param_tables.py`; archived decks are marked there).

### 3.3 Material classes that can be computed

| Class | What backs it | Decks / cases on hand |
|---|---|---|
| **Proper ferroelectrics**: tetragonal, orthorhombic, rhombohedral | Wall and monodomain anchors `K3.wall_t180_gld1d`, `K3.wall_r109_recorded`, `K5.bto8th_monodomain`, `K5.pto_monodomain`, `K5.pzt_monodomain` | BTO 298 K / 118 K, PTO, PZT; `cases/t180.py`, `r109.py` |
| **Antiferroelectric PbZrO₃** (two independent routes) | (a) Kittel two-sublattice `q`: `K3.afe_analytic_single_cell`, `K3.afe_field_thresholds_lum2022`; (b) enlarged-cell single-P triple well: `K3.triplewell_fitter_roundtrip`, `K7.loop_closure` | `pbzo_kittel_lum2022`, `pbzo_triplewell_yang2025`; `cases/wp20_triplewell_case.py` |
| **AFD-coupled perovskites** (SrTiO₃, PZT R3c) | `K3.afd_analytic_single_cell`, `K3.afd_pure_theta_anchor`, `K3.pzt_haun_quantitative` | `sto_afd_rabeA`, `pzt_r3c_haun_x02` (archived deck, `params/archive/`); `cases/wp18_afd_case.py` |
| **PMN-like relaxors** (via quenched disorder) | Two **physically independent channels that are never mixed**: charge mismatch → a quenched **E field** `E_d(r)` (fed to `E_ext`); chemical short-range order → a quenched **α₁(r)** (fed to `deck["landau"]["a1"]`). `K5.pmn_field_statistics`, `K2.pmn_channel_wiring`, `K3.curie_weiss_clean_control` | `pmn_field.pmn_clean_alpha1` + the `pmn_field`/`quenched_fields` disorder layer (their decks `pmn_clean_x0` / `pmn_disorder_R3` are archived in `params/archive/`); `cases/wp16_pmn_pilot.py`, `w0_relaxor_demo.py`. **Read §6.5 before using any finite-T conclusion** |
| **Thin films on substrates** | `K3.film_sandwich_closed_form`, `K3.film_surface_traction`, `K8.film_dynamic_crosscheck`. **Recorded result (not gated)**: 7 of the 8 Li & Chen Fig-1 points agree; point (e)'s multi-basin root cause is in `reports/archive/v1/M2F_DISCREPANCY.md`, and the ε_b that run actually used is discussed in §6.6 | `bto_LCC2005_paper`; `cases/film_tier1_scan.py`, `film_tier3_fig1.py` |
| **Solid solutions (composition fields)** | Continuous Vegard-type x(r): `K1.fd_2d_pzt_fields`, `K2.composition_const_degenerate`, `K5.composition_equivalence`; whole-cell label fields (ternary): `K1.fd_2d_ternary`, `K3.ternary_endmember_0d` | PZT x(r); BFO–BTO–STO; `cases/wp11_matrix.py`, `b2a_r2_ternary.py` |
| **Energy storage (2D, E1 unipolar protocol)** | `K7.unipolar_short`, `K3.storage_linear_dielectric`, `K3.storage_lossless_closure` | `pbzo_triplewell_yang2025`; `solvers/energy_protocol.py` |

### 3.4 What cannot be computed, or is unverified for lack of an item

Each line names **the WP where it lands**, or "not planned" with the reason. Schedule:
`reports/ROADMAP_FFT-PF-V2.md`.

| Not available | Lands in |
|---|---|
| Polycrystal **elasticity** (per-grain rotated C/Q, eigenstrains), 3D polycrystals, and a gated/validated 2D status. The 2D geometry itself — periodic Voronoi grains, a grain-boundary shell label and per-grain rotated Landau force — merged 2026-09-22/23 (`solvers/polycrystal.py`) and is **unverified (no core-gate item)**: exercised only by `cases/cloud_cai2021_vortex_relax.py` / `cases/cloud_cai2021_figures.py` (elasticity off) | **V2-05** |
| Film AFE/AFD gradient energy exact on **even** in-plane grids: `kittel_afe.grad_energy_q_film` / `octa_tilt.grad_energy_theta_film` still drop the Nyquist mode there (FD relerr 1.40; odd grids exact, 3.5e-10); production uses odd in-plane grids | **film lane carry-over** (SPEC § 8.2.1(b)) |
| Free charge / semiconductor screening (Poisson–Boltzmann, λ_s linear screening), **finite** electrode screening | **V2-06** (today there are only ideal short-circuit electrodes) |
| Defect dipoles, ageing, imprint (E_bias(r), τ_age) | **V2-07** |
| Rhombohedral 3D poling with a **validated** PMN-PT deck | **V2-09** (the deck itself waits on V2-11) |
| AC/DC poling science package (frequency ladders, lock-in χ₃₃, d33 proxy) | **V2-10** (the input-file layer has a small-signal lock-in `ac` stage and a fluctuation `fdt` stage for the dielectric spectrum, §8; not a poling package) |
| Materials with no Landau coefficients | **V2-11** fit lane (NBT-6BT re-fit per R1–R9; PMN-PT pending literature) |
| Field waveforms beyond the E1 unipolar protocol: the bipolar **triangle**, DC hold, field ramp and temperature ladders are built in the input-file layer (§8, portable tests only, no gate item); **sine, staircase**, a decaying envelope and field-cooling sequences are not built | V2-04.1 was taken over by the input-file layer (SPEC § 16.9-7); the rest has no work package yet |
| The energy-storage protocol on bulk3d / film3d | **V2-08** |
| Quenched disorder and checkpoint/resume on the film lane | **未验证** (no gate item exists; the input-file layer runs a film resume only as a smoke test, §8, and accepts quenched disorder on the film only from a file); the film ensemble S axis is **not built** |
| Improper / hybrid-improper ferroelectrics, magnetoelectric coupling, flexoelectricity | **not planned**: no such terms in the free energy and nothing scheduled |
| Finite strain (geometric nonlinearity), inertial / piezoelectric wave dynamics | **not planned**: elasticity is small-strain Khachaturyan, dynamics are **overdamped** TDGL gradient flow — there is no ∂²P/∂t² |
| Temperature-dependent gradient or elastic coefficients | **not planned**: T enters only through `a1(T)` (and where needed `a11`/`a111`) |
| Non-periodic bulk boundary conditions | **not planned**: only the film has surfaces; `solver.py`/`bulk3d.py` are periodic in every direction |
| Treating 2D as a three-dimensional body | **a convention, not a capability**: 2D is a slab of stated thickness `Dz` (§4) |
| Continuum-limit thermal fluctuations | **a convention**: the noise is coarse-grained at ℓ_c (§4); `noise_cutoff_nm=None` is the pure discrete FDT |
| Converting deck time into seconds | **V2-11** (`seconds_per_deck_time` is not calibrated; today `L = 1` and the time unit is undefined) |

---

## §4 Units and conventions (short)

The full statement is in `SPEC.md` §2 "conventions of record" and `docs/PARAMETERS.md` §A.

- **Deck units**: energy density `1e9 J/m³`, length `nm`, **P unscaled** (the SI number is used directly).
- **Permittivity**: `κ_deck = 0.0088541878128 · ε_b`, `E_deck = E_SI × 1e-9`. The historical deck value
  `0.08854187` is **ε_b = 10**, not 1 (SPEC V2-02.1, ruling D7: the labels were wrong by 10×, the engine
  arithmetic was always right). Authority: `K5.permittivity_convention`.
- **Langevin FDT coefficient**: `kT = k_B·T·1e18/(hx·hy·hz)` — an **energy density**
  (`units.kT_langevin`). In 2D, `hz` is the **fluctuation-slab thickness `Dz`**, a stated modelling
  parameter (legacy `Dz = hx`), now written explicitly at every 2D call site. Authority:
  `K5.kt_convention`.
- **Noise cutoff `noise_cutoff_nm = ℓ_c`**: modes with `|k| ≥ 2π/ℓ_c` get no noise (their decay is
  unchanged). The engine primitives default to `None` = the pure discrete FDT (guarded bit-identical by
  `K2.noise_cutoff_none_byte`); the **production/demo default is 2.0 nm** (user ruling V2-04.0.2(b)) —
  landing it is the first item of V2-04.1 and **has not landed yet**.
- **Mobility `L = 1`**, and the time unit is **not calibrated to seconds** (`seconds_per_deck_time` is
  scheduled for V2-11).
- **Q notation**: the engine consumes the **tensor** form; decks store Voigt, so
  `Q1212 = Q44_phys/4` (`casekit.build_3d` does this for you).

---

## §5 Validation: the eight core gates

The acceptance instrument is the eight files under `gates/core/`, each a set of named **ITEMs**, 205 in
total: **K1** force-energy consistency (central FD vs analytic force) | **K2** degenerate limits and
byte baselines | **K3** closed forms and literature anchors | **K4** numerics and per-step timing |
**K5** materials, field statistics and unit conventions | **K6** hygiene (imports, dependency direction,
freshness of generated files) | **K7** protocols | **K8** smoke (end-to-end scripts, checkpoints).

```bash
/opt/miniconda3/bin/python3 gates/core/k1_consistency.py --list          # every item in this gate
/opt/miniconda3/bin/python3 gates/core/k1_consistency.py --only fd_2d_all_terms   # one item, <1 s
/opt/miniconda3/bin/python3 gates/g8e_full_suite.py                      # all eight (the account), ~10 min
```

**How to read one ITEM line**:

```
ITEM K1.fd_2d_all_terms PASS 0.006s grid=48x48 eps=1e-05 tol=1e-05 bulk_relerr=3.27961635231e-10 ...
```

`ITEM` | `K<gate>.<item name>` | `PASS`/`FAIL` | wall-clock seconds | the key=value pairs that item
prints — **the tolerance and the measured number are both on the line**, so what was proved is readable
without opening the code. `--json` emits the machine-readable form (the zero-regression audit is a
key-by-key diff of those dumps). Each item also carries a risk class: `BYTE` (bit-identical), `TOL`
(within tolerance), `ANALYTIC` (against a closed form), `REGRESSION` (against a stored baseline). A few
items are **report-only** (always `ok=True`, they only print numbers — `K4.timing_budget`,
`K4.langevin_stable_dt`, `K4.film_ou_cost`): they are instruments, not criteria.

**Account of record**: `CORE 8/8 PASS @ 2408137`, **622.8 s, 205 items** (box not idle, budget 900 s;
K1 57.63 / K2 34.04 / K3 239.60 / K4 71.58 / K5 52.97 / K6 117.39 / K7 28.22 / K8 21.41 s, SPEC
V2-SYNC, D18). By program law `gates/` is **local only** (gitignored, never pushed to GitHub) while staying
fully runnable on disk; it is the sole acceptance instrument. The archive tier (`gates/archive/`,
`cases/archive/science_exits/`) runs under `--archive` and **never counts toward the account**.

---

## §6 Known exceptions and edges

These are **exceptions of record**, not a to-do list. Each carries its ruling.

### 6.1 Even-grid Nyquist convention, and the film q/θ carry-over

**2D `solver` / `bulk3d` (V2-SYNC, D18, SPEC § 8.2).** The gradient energy density of P, the AFE `q`
sector (`kittel_afe`) and the AFD `θ` sector (`octa_tilt`), and the homogeneous elastic Green operator,
now use one convention on **even** grids: collocation local-gradient density plus a positive even-axis
Nyquist correction, with mixed A/D symbols masked on their own odd Nyquist lines, on the same real
skew-adjoint wavevectors. **Odd grids are unchanged bit-for-bit.** Evidence: the ten `K1.even_grid_*`
items (`K1.even_grid_{2d,3d}_{grad,q,theta,elastic}`, `K1.even_grid_{2d,3d}_elastic_spm`;
`gates/core/k1_consistency.py --list`).

**Film C(r) elastic SPM — formerly "standing finding 3", now retired.** On the current tree the film
C(r) elastic SPM force is an exact energy gradient on **both** even and odd in-plane grids (≈`1e-10`,
`K1.fd_film_elastic_spm`). The former even-grid residual (`1e-4..1e-2` on the `kx = Nyquist` column, ruled a
documented limitation in SPEC V2-03.C.1(f)) no longer applies to this term.

**New finding — film q/θ gradient energy, even in-plane grids (film lane carry-over).** Unlike the P
sector above, `kittel_afe.grad_energy_q_film` and `octa_tilt.grad_energy_theta_film` still drop the
Nyquist mode on **even** in-plane grids: FD relative error **1.40** (odd grids exact, `3.5e-10`).
`K1.fd_film_afe` / `K1.fd_film_afd` PASS only because their fields are band-limited — they do not
exercise this limitation. Until fixed, film AFE/AFD production decks use **odd** in-plane grids,
mirroring the elastic-SPM rule above (SPEC § 8.2.1(b)).

A V1-era boundary of the same kind — `solver.py`'s off-diagonal k-operators inexact at the
self-conjugate Nyquist column/row — is **resolved** by the masking above (V2-SYNC); the film solvers
zero their odd first-derivative symbols on their own Nyquist axis since `0c6491c` / `bf9df2c`.

### 6.2 The SPM reference (κ₀ / C₀) is not a free choice

`κ₀ = mean` is not always an admissible SPM reference: the contraction bound `max|δκ|/κ₀` is minimized at
the **midpoint**, and `g8a`'s own stack diverges with the mean. Ruling (SPEC V2-03.B.1(b)): the default
stays the mean (for 2D/3D byte compatibility), `kappa0_override` / `kappa0_profile` are the documented
escape hatches, and the converged field's invariance to the reference (`9e-12`) is the pinned guarantee.
The film elastic side adds one more: the substrate keeps `C0` by construction, so **changing
`C0_override` changes the substrate material** (SPEC V2-03.C.1(g)).

### 6.3 `clamped` + ensemble: refused loudly

bulk3d's elastic `k0="clamped"` branch is missing one broadcast axis, so **using it together with the
ensemble S axis raises loudly** rather than silently computing the wrong thing (SPEC V2-03.C.1(h); the
one-line fix is carry-over ⑥ and has not been done).

### 6.4 0.4 nm decks at 298 K with no cutoff need a smaller dt

Under the corrected Langevin convention `kT/well-depth` is **27.1** on **0.4 nm** cells (1.74 on 1.0 nm)
— cell-scale fluctuations really are that large; the old `1/√N_cells` was hiding them. Consequence
(`K4.langevin_stable_dt`, SPEC V2-04.0.1 ruling (7)): at 0.4 nm, 298 K and **no cutoff** the 2D stepper
needs **5 halvings** of dt (2.56 → 0.08); with `ℓ_c = 2 nm` it returns to 0.32 (8×), and bulk3d needs no
reduction. **So**: a production deck at 0.4 nm resolution needs either an ℓ_c or a coarser cell.
`ℓ_c = 1 nm` is inert for `h > 0.5 nm` (ℓ_c < 2h, every mode kept).

### 6.5 Where the relaxor claims stand after the D17 re-runs

The Part-I WP16/W0 finite-temperature conclusions were produced under the **pre-correction** noise
normalisation (nominal 298 K was actually `T·V_cell/N_cells`). D17 re-ran WP16 Stages 1–4 and W0 at
`ℓ_c = 2 nm` (7/7 stages, 1.17 h, measured equivalent temperature **299.0 K** on the run's own grid,
0.34 % off). **Current state** (SPEC V2-04.D17.1; full table in `reports/WP-V2-04_D17_REPORT.md`):

- **Vogel–Fulcher is WITHDRAWN**, structurally: the T_m spacing shrinks with ω where VF requires it to
  grow, so no finite (ω₀, E_a, T_VF) exists. The dispersion itself **survives** and is **Arrhenius**
  (ω₀ 2.94e4, E_a ≈ 4962 K).
- Freezing temperature **T_f moves from (338,358) K to (298,318) K**; the FC−ZFC splitting survives.
- The 298 K loop shape and saturation survive, but the split is **−30 %** and the area **−40 %** — the
  recorded numbers really were upper bounds.
- The PNR cluster-area peak at T₀ is **withdrawn**; `q_EA` 1.211 → 1.015; the diffuse peak at
  `k = 0.833 nm⁻¹` **survives exactly** (it is set by the quenched-disorder geometry, not by noise).
- **W0's "topology locks, quenched-disorder pinning" is WITHDRAWN**: 19.9 % of cells reverse sign
  between step 300 and step 100 000 at 298 K.
- The ageing `t_w` trend now has the right sign and size but is resolved only to 1.2 SEM at 8 replicas —
  **no conclusion**.
- The re-run used **one** ℓ_c. Sensitivity at 1.5 / 3 nm is **untested** — a carry-over, not a claim.

### 6.6 Two standing warnings

- **The engine returns a local minimum** (semi-implicit gradient flow, not global optimization). The same
  parameters with a different initial condition can land in a different basin; near a close multi-variant
  competition (Tier-1 energy margins ~1e-4) a single anneal is a **lower bound, not a conclusion** — use
  multi-start / multi-seed protocols (the WP8 lesson, `reports/archive/v1/M2F_DISCREPANCY.md`).
- **Under noise the dF-plateau criterion is void**: stop on block-mean differences
  (`solver.block_stationary`), not on slope or trend; the threshold depends on deck stiffness, so a new
  deck needs it re-validated rather than copied.
- **What the two unit incidents actually cost.** (i) The κ_b labels were wrong by 10× (D7) — the
  arithmetic was not, but the **recorded** film paper-lineage runs actually ran at ε_b = 660 (not the 66
  the paper lineage intended) and the B1 panels at 500; user ruling D15 is **no re-run**, so stored
  results are read with corrected ε_b labels (SPEC V2-02.1 / V2-02.1a). (ii) The Langevin kT was weak by
  `N_cells/V_cell` — that one was an **engine change** (V2-04.0); Part-I Langevin results are D7-frozen
  and relabelled with their equivalent temperature, the AT-RISK rows were re-run under D17 (§6.5), and
  the WP12/WP13 FDT work and the B2 series are restated rather than re-run. Each convention now has an
  authority gate (`K5.permittivity_convention` / `K5.kt_convention`) — consistency gates cannot catch a
  unit error, which is the lesson itself.

---

## §7 Features merged 2026-09-22/23 (crystallinity, defects, substrates, polycrystal)

These arrived with the `codex/crystallinity-defects` / `codex/performance-kernels` merges and the
cloud-domain-examples merge (SPEC § 8.0(a)). Under the EVIDENCE RULE: checked against every
`gates/core/k1..k8 --list` roster (2026-09-28) — no core-gate item exercises prescribed crystallinity,
defects, aging or the drive protocol layer, or the 2D polycrystal/core-shell geometry, so each is
written **unverified (portable tests only)**, naming the test file, per SPEC § 8.4.

| Feature | Module | Evidence |
|---|---|---|
| Prescribed crystallinity (`effective_alpha`) | `solvers/crystallinity.py` | **unverified (portable tests only)**: `tests/crystallinity_defects/check_crystallinity.py`, `tests/run_crystallinity_defects.py` |
| Static defect bias (`compose_field`, `bias_force`) | `solvers/defects.py` | **unverified (portable tests only)**: `tests/crystallinity_defects/check_defects_static.py`, `tests/run_crystallinity_defects.py` |
| Local aging (`aging_state`/`aging_update`) and the `AgingDrive` protocol runner | `solvers/defects.py`, `solvers/drive.py` | **unverified (portable tests only)**: `tests/crystallinity_defects/check_defects_aging.py`, `tests/crystallinity_defects/check_defects_drive.py`, `tests/run_crystallinity_defects.py` |
| Prerequisite film electrostatic/elastic adjoint correction (odd-Nyquist handling, gradient quadratic-form consistency) | `solvers/film3d.py`, `solvers/film_elastic.py` | **unverified (portable tests only)** for the adjoint re-check itself: `tests/check_film_variational.py`; the film C(r) elastic SPM term it touches is separately core-gated (`K1.fd_film_elastic_spm`, §6.1) |
| Optional Numba CPU backend for direct film elasticity | `solvers/film_elastic_fast.py`, `solvers/film_elastic_numba.py` | ✓ core-gated: `K1.film_fast_force` (reference-force/FD cross-check, odd/even/static eigenstrain, 1/2/4 workers) |
| Heterogeneous substrate stiffness (`C_substrate`) | `solvers/film_elastic.py`, `solvers/materials.py` (substrate registry) | machinery + the `ss316_rt` deck ✓ core-gated: `K1.film_fast_force`; the other four decks (`ss304_rt`, `silicon_cubic_rt`, `fused_silica_hpfs_rt`, `mgo_cubic_rt`, `params/substrates/*.toml`) are **unverified (portable check only, no core-gate item)**: `cases/check_substrate_parameters.py` |
| 2D polycrystal (periodic Voronoi grains, per-grain rotated Landau, elasticity off) + core-shell dead-layer geometry (Cai 2021) | `solvers/polycrystal.py` | **unverified (no core-gate item)**: `cases/cloud_cai2021_vortex_relax.py`, `cases/cloud_cai2021_figures.py` (mirrors Cai et al., APL 119, 032901 (2021) Fig. 5 panels) |

Feature-level guide and units: `docs/CRYSTALLINITY_DEFECTS.md`; portable-check inventory:
`tests/crystallinity_defects/README.md`.

---

## §8 Running from an input file (RUN-ENTRY, SPEC § 16, 2026-11-13)

`python -m fftpf run | check | explain case.toml` reads one TOML file (material, geometry, physics, stages, measures), builds
the objects the existing engines need, runs the stages and writes a run directory. Manual: `docs/RUN_INPUT.md`. Modules:
`fftpf/run_input.py`, `run_builder.py`, `run_adapter.py`, `run_protocol.py`, `run_measure.py`, `run_output.py`, `run_figures.py`,
`__main__.py` (`docs/MODULE_NAMES.md`).

**Evidence status.** These are **portable tests** in `tests/run_entry/` (156 items plus `test_docs.py`), run by
`python3 -m pytest tests -q`. **No CORE gate item covers the entry layer**, and no engine file was changed to build it; under
the EVIDENCE RULE every claim below is therefore **unverified (portable tests only)**, not a capability in the sense of §2.
The tests compare the entry path with the old path in one process (`numpy.array_equal` on the final P and on every recorded
number) wherever an old path exists. Tests marked † skip when the local `Runs/` folder (NBT6BT driver) or
`cases/wp13_debye_case.py` is missing.

| Claim | Tests (`tests/run_entry/`) | Example file (`docs/run_examples/`) |
|---|---|---|
| The file is parsed, defaulted and validated; an unknown key, a wrong type and every cross-key combination of the schema is refused with the key path before the first step | `test_input.py` (all), `test_run.py::test_refusals`, `test_drive.py::test_input_refusals`, `::test_temperature_law_refusals`, `::test_field_key_refusals`, `test_spectrum.py::test_refusals`, `test_measure.py::test_domain_map_refusals`, `test_measure_rto.py::test_refusals`; `resolved.toml` re-parses to the same object (`test_input.py::test_round_trip_*`) | all five (`check`) |
| The builder makes the arrays that `casekit.build_2d`, `casekit.build_3d` and `cases/bto_film_biaxial.setup` make; a Gaussian field equals `disorder_fields` with the same seed; `physics.noise.cutoff_nm = 0` is no cutoff | `test_run.py::test_builder_equals_casekit_2d_and_3d_and_film`, `::test_gaussian_fields_equal_disorder_fields`, `::test_no_cutoff_and_inline_material` | `01`, `05` |
| `anneal` reproduces `casekit.anneal` bit for bit on `pf2d` and `pf3d` (fixed and verified dt, with and without the noise cutoff); `relax` reproduces the film relax loop of `cases/bto_film_biaxial.py` | `test_run.py::test_demo_casekit_2d_anchor`, `::test_demo_casekit_3d_anchor`, `::test_film_relax_anchor`; adapter equals the direct engine calls for 3 engines × {SBDF1, SBDF2, Langevin}: `test_adapter.py::test_bit_for_bit` | `01`, `05` |
| `cool`, `hold` and `pe_loop` (triangle) follow the NBT6BT driver's stage semantics (T ladder, field ramp, one carried generator, T0 and E_ext fields, plane-stress `pf3d`); the unipolar loop equals `run_unipolar` | `test_drive.py::test_nbt6bt_anchor_driver_b` †, `::test_nbt6bt_anchor_numpy_backend` †, `::test_unipolar_anchor_run_unipolar`, `::test_two_rung_cool_equals_manual_segments`, `::test_rng_is_one_generator_for_the_run`, `::test_hold_ramp_is_driver_b_remanent_formula`, `::test_field_orders_seed_and_seeds` | `02`, `04` |
| Loop measures (`Pr`, `Ec`, `Pmax`, `W_store`, `W_rec`, `W_loss`, `eta`) equal the stated definitions on a known series and the `energy_integrals` of the old module on a real run | `test_drive.py::test_loop_measures_on_a_known_series`, `::test_unipolar_series_measures`, `::test_real_triangle_loop_measures_json_equals_trace` | `02` |
| Checkpoint and resume reproduce the uninterrupted run byte for byte (trace, measures, states, figure names), reload quenched fields, refuse another configuration and a busy directory | `test_drive.py::test_resume_equals_uninterrupted`, `::test_resume_is_mid_stage_and_rng_carried`, `::test_checkpoint_of_another_config_and_busy_dir`, `::test_quenched_fields_are_reloaded_not_regenerated`, `::test_cli_stop_hook_and_resume`, `test_measure.py::test_resume_gives_the_same_trace_measures_and_figure_names` | — (`output.checkpoint_every`) |
| Phase fractions (`rt`) equal the frozen NBT6BT readout; scheme `rto` partitions the cells and equals a brute-force angle test against the 26 directions; film fractions are weighted by `qw` | `test_measure.py::test_fractions_equal_frozen_readout_random_and_boundary` †, `::test_nbt6bt_anchor_trace_fraction_columns` †, `::test_hand_built_fields_have_the_known_classes`, `::test_partition_identity_and_t15_inside_t_on_random_fields`, `::test_film_fractions_are_weighted_by_qw_by_layers`, `test_measure_rto.py` (all) | `02` (`rto`), `04` (`rt`) |
| Domain maps (`direction`, `phase`) are written for the selected states | `test_measure.py::test_pf2d_domain_maps_end_to_end`, `::test_pf3d_domain_map_is_the_mid_plane`, `::test_saved_states_are_exactly_the_state_files_of_a_run` | `02`, `04` |
| `ac` (lock-in) and `fdt` (fluctuation) stages equal the WP13 case bit for bit and give the Debye closed form; the per-volume χ does not depend on the cell count; ε and tan δ are formed as stated; replicas pool a mean and a standard error and resume | `test_spectrum.py::test_ac_bit_for_bit_wp13` †, `::test_fdt_bit_for_bit_wp13` †, `::test_deterministic_lockin_matches_the_debye_form_and_the_error_shrinks_with_dt`, `::test_fdt_replicas_match_the_debye_relaxator`, `::test_per_volume_chi_is_independent_of_the_grid`, `::test_pf3d_deterministic_lockin_is_debye`, `::test_replicas_write_rep_dirs_and_the_pooled_mean`, `::test_interrupted_replicated_run_resumes_to_the_same_spectrum` | `03` |
| The film runs `relax` (anchored), and `anneal`, `cool`, `hold`, `pe_loop` as smoke runs | `test_run.py::test_film_relax_anchor`, `test_measure.py::test_film_run_uses_grid_qw`, `test_drive.py::test_film_cool_hold_pe_loop_run` | `05` |
| The manual cannot drift from the code | `test_docs.py` | all |

**Not claimed** (details in `docs/RUN_INPUT.md` § 10): waveforms `sine` and `staircase`; `ac` and `fdt` on the film; wall metrics;
Gaussian fields on the film; mid-stage checkpoints in `relax`, `anneal`, `ac`, `fdt`; backends `numba` and `fused` through the entry
point beyond the NBT6BT `pf3d` anchor; `threads` above 1; a unipolar loop on `pf3d` and the film; `rto` maps on `pf3d` and the film.
