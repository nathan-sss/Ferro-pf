# RUN_INPUT — running the engine from one input file

`python -m fftpf run | check | explain case.toml` (SPEC § 16, RUN-ENTRY). Written 2026-11-13 from the code of
`fftpf/run_input.py`, `run_builder.py`, `run_protocol.py`, `run_measure.py`, `run_output.py`, `run_figures.py`,
`run_adapter.py`, `__main__.py` and the tests in `tests/run_entry/`. The key tables are generated from the input schema
(`tools/run_input_table.py`); everything else is hand-written and is checked against the code as described in § 13.

**Evidence status.** The entry layer has portable tests (`tests/run_entry/`) and
no CORE gate item. Under the EVIDENCE RULE of SPEC § 5 its claims are therefore *unverified (portable tests only)*;
`docs/CAPABILITIES.md` § 8 lists the claims with the tests that prove them. § 10 of this page states what is not built or
not verified.

## 1 What it is

1. You write one TOML file: a material, a geometry, the physics switches, and a list of stages (relax, anneal, cool, hold,
   P-E loop, small-signal ac, fluctuation fdt).
2. `python -m fftpf run case.toml` builds the engine objects, runs the stages one after the other on one of the three
   engines (`pf2d`, `pf3d`, `film`), and writes a run directory with the input as resolved, a trace, saved states,
   figures and `measures.json`.
3. The layer calls the existing engines and estimators; it has no numerics of its own. Where an old script
   exists, the entry path is tested bit for bit against it (§ 2.2); where none exists the stage is only smoke-tested.
4. A combination that is not implemented stops before the first step and names the key and the reason (§ 9); it never
   runs different physics.
5. One-off research constructions still need a script; the layer can be called from Python and takes your arrays through
   `kind = "from_file"` (§ 11).

The smallest complete file (a deterministic relaxation; every other key has a default):

```toml complete
[run]
name = "minimal"

[material]
deck = "bto_lcc2005_8th"

[geometry]
engine = "pf2d"
N = [16, 16]

[[stage]]
kind = "relax"
```

## 2 Commands

| command | does | prints | exit code |
|---|---|---|---|
| `run case.toml` | reads and validates the file, builds everything, runs all stages, writes the run directory | the run directory path (last line) | 0 |
| `check case.toml` | reads and validates the file, then runs the dry build: every refusal `run` raises before its first step (§ 9); builds no array, takes no step, creates no directory | `case.toml: ok` | 0 |
| `explain case.toml` | the same checks as `check` first (a file `check` refuses is refused with the same message and exit code 2); then prints the engine functions the file selects, the Langevin kT and cutoff, the dt choice, the stage list | the description | 0 |

Exit codes: **0** success; **2** the file or the command line is wrong (usage, file missing, or any `ConfigError`; the message
`error: <key.path>: <reason>` goes to standard error); **3** only when the test hook `--stop-after-checkpoints k` stops a run
right after its k-th checkpoint. Any other failure is an uncaught Python exception (traceback, exit 1): a non-finite field
(`FloatingPointError`, raised after the stage is written out), and the items listed under "Failures that are not refusals"
in § 9.

`check` is the same code path as `run` with a `dry` switch (`run_builder.build(cfg, dry=True)`): it reads and validates the file, builds the
material, checks the engine/elastic/field/start combinations and reads every `from_file` array (shape and key), checks that
`output.dir` (or `FFT_RUN_ROOT`) can be created, and stops there. What it skips is what costs time or steps the engine: the grid
operators and elastic setup, the generation of Gaussian fields, the random start, and the `dt = "verified"` probe. So every
refusal in § 9 is raised by `check`; what `check` cannot know is whether the run stays finite (`FloatingPointError`) and whether
the files in `output.dir` are still as they were when the run is resumed. `explain` prints no list of "what would be refused if
you changed a key": it only refuses consistently with `check`.

Environment: `FFT_RUN_ROOT` replaces `reports/runs` for `output.dir = "auto"`; `FFT_RUN_ID` replaces the folder name (and makes
`auto` resumable, § 5). Thread variables are set from `numerics.threads` before NumPy is imported.

### 2.1 Data flow

```
 case.toml ──► run_input ──► run_builder ──► run_protocol ──► run_measure ──► run_output
               parse,        deck, grid,     stage list        readings per    run directory
               defaults,     alpha fields,   over one          step or event:  resolved.toml
               validate,     E_ext fields,   adapter:          mean P, energy  provenance.json
               refuse        operators,      relax, anneal,    loop measures   trace.csv
                 │           elastic,        cool, hold,       phase fractions state_*.npz
                 │           backend         pe_loop, ac,      chi(w), eps     measures.json
                 │               │           fdt                               figures/*.png
                 │               ▼               │                             checkpoint.npz
                 │           run_adapter ◄───────┘                             spectrum.csv
                 │           step(P, E, rng) · mean_P · energy
                 │               │ call-through, no numerics of its own
                 ▼               ▼
        input_validation   pf2d_engine │ pf3d_engine │ film_engine       (unchanged)
```

The layer imports the engines; no engine imports the layer. `resolved.toml` is the input with every default written out;
running it reproduces the run.

### 2.2 Stage × engine

| stage | `pf2d` | `pf3d` | `film` |
|---|---|---|---|
| `relax` | built, run by hand only | built, run by hand only | anchored (`test_film_relax_anchor`) |
| `anneal` | anchored (`test_demo_casekit_2d_anchor`) | anchored (`test_demo_casekit_3d_anchor`) | smoke only (deterministic: `test_film_run_uses_grid_qw`; Langevin built, run by hand only) |
| `cool` | anchored (`test_two_rung_cool_equals_manual_segments`) | anchored to the frozen NBT6BT driver (`test_nbt6bt_anchor_driver_b`, needs the local `Runs/` folder) | smoke only (`test_film_cool_hold_pe_loop_run`) |
| `hold` | anchored (same tests) | anchored (same) | smoke only |
| `pe_loop`, triangle | built; loop measures tested on a known series and on a real run | anchored (NBT6BT driver, `Runs/` needed) | smoke only |
| `pe_loop`, unipolar | anchored (`test_unipolar_anchor_run_unipolar`) | not tested | not tested |
| `ac` | anchored (`test_ac_bit_for_bit_wp13`, Debye closed form) | Debye closed form (`test_pf3d_deterministic_lockin_is_debye`) | refused (no anchor case) |
| `fdt` | anchored (`test_fdt_bit_for_bit_wp13`, Debye closed form) | built, run by hand only | refused (no anchor case) |

"Anchored": a test runs the old path and the entry path in one process and asserts equal floats (`numpy.array_equal` on
the final P unless the test says otherwise). "Smoke only": the run completes and a few numbers are checked, nothing is
compared with a reference. "Run by hand only": it ran without error during the writing of this manual (2026-11-13); no test
covers it. Backends: `numba` is anchored on `pf3d` by the NBT6BT test (`Runs/` needed); `fused` and `numba` on `pf2d` ran by hand.

## 3 The run directory

`output.dir = "auto"` gives `reports/runs/<YYYYMMDD-HHMM>_<run.name>/` (a second folder in the same minute gets `_2`, `_3`).
Any other value is the directory itself; it must be new or empty, or hold a checkpoint of the same configuration (§ 5). The
directory is created after the build, so a refusal leaves nothing behind. A run never overwrites.

| file | when | content |
|---|---|---|
| `resolved.toml` | always | the input with every default written out (`cell_nm` as a list, the start table, `field = []`, …). Re-running it reproduces the run. |
| `provenance.json` | always | `commit` (HEAD, `+dirty` if the tree has changes), `engine`, `thread_line`, `config_sha256`, `input_file` and its SHA-256, `from_file_arrays` (path and SHA-256 of every `from_file` input), `seed`, `fields` (target, kind, seeds), `dt`, `dt_verified` (`agree_relerr`, `T_total`, or null), `kT`, `noise_cutoff_nm`, and `provenance` (SHA-256 of every source file, library versions, platform). |
| `trace.csv` | always | one row per record. Columns: `stage`, `kind`, `step` (in the stage), `step_total` (all stages), `t` (deck time = `step_total · dt`), `T_K`, `E_kV_cm` (applied drive, without the static E_ext), `Pn` (mean P · axis, C/m²; `nan` without an axis), `Px`, `Py`, `Pz` (volume-mean P, C/m²; unless `measure.mean_P = false`), `E_total` (engine total energy = `f_total` × grid volume), `f_total` (mean energy density, deck units = 10⁹ J/m³), `E_bulk`, `E_grad`, `E_elec`, `E_elastic` (only with `measure.energy_terms`), `f_R`, `f_T`, `f_T15`, `f_other`, `f_low` or `f_R`, `f_O`, `f_T`, `f_other`, `f_low` (only with `measure.phase_fractions`, by scheme). `relax`/`anneal` rows are written at the end of the stage; `cool`/`hold`/`pe_loop` rows as they are recorded; `ac` and `fdt` write one row at the stage end. Floats are written with `repr`, so they read back exactly. |
| `state_<i>_<kind>.npz` | end of each `relax`/`anneal`/`ac`/`fdt` stage | the field P (3, Ny, Nx) or (3, Nz, Ny, Nx) in C/m², the grid, step and time, and a `metadata_json` restart record. Read it with `fftpf.state_io.load_state_npz`. |
| `state_<i>_<kind>_<event>.npz` | events of `cool`, `hold`, `pe_loop` | same. Events: `cool`: `T<rung temperature>` at each rung end (`T298`); `hold`: `ramped` (when 0 < `ramp_steps` < `n_steps`) and `end`; triangle loop: `c<k>_Emax` (+Emax), `c<k>_E0down` (E = 0, falling: positive remanent state), `c<k>_E0up` (E = 0, rising: negative remanent state) for each cycle k; unipolar loop: `Emax`, `end`. The −Emax point is not saved. |
| `measures.json` | end of run | `dt`, `kT`, and `stages`: one entry per stage with `stage`, `kind`, `steps`, `t`, `stop_reason` (`n_max`, `n_steps`, `converged`, `stationary`, `nonfinite`), `end` (the last reading: `E_total`, `f_total`, `Pn`, `T_K`, `E_kV_cm`, `mean_P`, `terms`, `f_*`), `loop` (§ 8.1), `fractions_end` and `fractions_at` ({event: fractions}), and for `ac`/`fdt` a `spectrum` summary. With `measure.spectrum` also `spectrum` (per route: `n_points`, `eps_b`, and for `fdt` the resolved range). |
| `quicklook.png` | end of run | the final field (mid-z plane for `pf3d` and film). Written last. Needs matplotlib. |
| `figures/<state stem>_<phase|direction>.png` | `measure.domain_map` | one map per selected state and kind (§ 8.2). |
| `quenched.npz` | when there is a `[[field]]` | one array per field (`field_<i>`), written once; a resume reloads it and never regenerates it. |
| `checkpoint.npz`, `checkpoint_prev.npz` | `output.checkpoint_every > 0` | P, the previous P (SBDF2), the generator state, stage and step, the byte count of `trace.csv`, the config hash (§ 5). |
| `series_<i>_ac_w<omega>.npz`, `series_<i>_fdt.npz` | `ac`, `fdt` | the raw series s(t) = Σ over cells of P·n, with `dt`, `omega`, `E0`, `kT`, `N_cells`, `cell_volume_nm3`, `axis` (§ 8.3). |
| `spectrum_<i>_<kind>.npz` | `ac`, `fdt` | the per-volume estimator output of the stage. |
| `spectrum.csv`, `figures/spectrum.png` | `measure.spectrum` | Columns `route` (`lockin`/`fdt`), `omega` (rad per deck time), `f_Hz` (only with `seconds_per_deck_time`), `chi_real`, `chi_imag` (per volume, deck units), `eps_real`, `eps_imag`, `tan_delta`, and with replicas the standard errors `<name>_se`. `measure.spectrum.quantities` only chooses the panels of the figure; the CSV always holds all columns. |
| `replicas.json`, `rep_<r>/` | `run.replicas > 1` | § 5. |

Without matplotlib every number is still written; the figures are skipped with one warning.

## 4 Conventions

* **Units.** A key carries its unit in its name: `_K` kelvin, `_nm` nanometre, `_kV_cm` kV/cm, `_pct` percent, `_deg` degrees. P is in
  C/m². *Deck units*, used by `landau`, `gradient`, `numerics.dt` and every energy: length nm, energy density 10⁹ J/m³
  (= GPa), field V/nm, so the Landau coefficient α_n (multiplying P^{2n}) is α_SI × 10⁻⁹ and the gradient coefficient is
  G_SI × 10⁹. Fields in kV/cm are converted by 1 kV/cm = 10⁵ V/m = 10⁻⁴ V/nm. Deck time is not calibrated (SPEC § 2):
  frequencies are *reduced* (rad per deck time) unless `measure.seconds_per_deck_time` is given.
* **Langevin noise.** kT is the energy *density* k_B·T / V_cell with V_cell = h_x·h_y·h_z (`units.kT_langevin`; the 2D
  slab thickness `slab_thickness_nm` is the fluctuation thickness h_z, so a `pf2d` thermal result depends on it).
  `physics.noise.cutoff_nm` is the length cutoff l_c of the noise.
* **One generator per run.** `numpy.random.default_rng(run.seed)` is created once, carried through every stage and every
  checkpoint, and is the source of all Langevin noise. A stage with its own `seed` starts a new generator from it, which the
  following stages then carry on. The band-limited `random` start and the `white` start have their own seeds (default
  `run.seed`). `[[field]]` seeds are the sample's quenched disorder and have their own seeds. Replicas shift every seed that derives from `run.seed` (§ 5).
* **Temperature.** A stage runs at its own `T_K`, else at `material.temperature_K`. How the material depends on T:
  *registry deck* (`deck = "bto_lcc2005_8th"` …): the deck is re-read with `T` at every stage temperature; only the Landau
  coefficients may change with T (the builder refuses a deck whose gradient, elastic or electrostriction sector does);
  *fixed deck* (a `params/*.toml` path, inline `landau`, or inline `landau_si` with a plain number for `alpha1`): valid at one
  temperature only; a stage at another temperature is refused; *slope law* (inline `landau_si` with
  `alpha1 = { slope, T0_K }`): a1(r, T) = slope · (T − T0(r)) with T0(r) = `T0_K` plus every `T0` field, converted to deck
  units; T-dependent for any stage temperature.
* **The deck's permittivity.** `physics.electrostatics.eps_b` absent means the deck's own κ_b (registry and `params/*.toml` decks
  have one; an inline material has none and uses 10). Given, it is converted by κ_b = 0.0088541878128 · ε_b.
  This is the default because the legacy deck constant is not exactly 10 · ε_0 (SPEC § 2, D7).
* **Noise cutoff.** `physics.noise.cutoff_nm` defaults to 2.0 nm (the SPEC § 2 ruling). `0` means *no cutoff*.
* **Axes.** `axis` is any non-zero 3-vector; it is normalised. The applied field of `hold` and
  `pe_loop` is E(t) · n on top of the static `E_ext` fields; E = 0 leaves the static field alone.

## 5 Resume and replicas

**Resume.** With `output.checkpoint_every = k > 0` the run writes `checkpoint.npz` every k steps inside `cool`, `hold` and
`pe_loop` stages and at the end of every stage (`relax`, `anneal`, `ac`, `fdt` checkpoint only at their end). Run the same
command again with the same `output.dir` and the run continues from the newest readable checkpoint (`checkpoint_prev.npz` if
the newest is truncated). The result is byte-identical to the uninterrupted run for `trace.csv`, `measures.json` and the state
files (tests: `test_resume_equals_uninterrupted`, `test_resume_is_mid_stage_and_rng_carried`). The checkpoint records the
config hash: the SHA-256 of the resolved file *without `[output]`*, plus the SHA-256 of every `from_file` input. Another
configuration in the same directory is a refusal, not an overwrite. Quenched fields are reloaded from `quenched.npz`.
`output.dir = "auto"` names a new folder every time and therefore never resumes unless `FFT_RUN_ID` is set; to resume, give an
explicit `output.dir` (the path the first launch printed) in the file and run the same command again.

**Replicas.** `run.replicas = R` means R independent complete runs, one after another, not an ensemble axis. Replica r runs in
`rep_<r>/` with `run.seed + r`; the same r is added to every stage `seed` and to every `random`/`white` start seed. `[[field]]`
seeds are not shifted: every replica sees the same quenched sample. With `dt = "verified"` the dt audit runs once, on replica
0, and the value is stored in `replicas.json` for all. The top directory holds `resolved.toml`, `replicas.json` (config hash,
seeds, dt), and at the end the pooled `spectrum.csv`, `figures/spectrum.png` and `measures.json`: ε′, ε″, tan δ are formed per
replica and then averaged; the `_se` columns are the standard errors of the mean (ddof = 1). Re-running resumes: a replica
whose `measures.json` exists is skipped, the unfinished one resumes from its own checkpoint. Replicas need an `ac` or `fdt`
stage and **`measure.spectrum`** (a file with `run.replicas` above 1 and no `measure.spectrum` is refused at load). A single run with an `ac` or `fdt` stage and no `measure.spectrum` is allowed: it writes the series and the per-stage spectrum files (`series_*.npz`, `spectrum_<i>_<kind>.npz`) but no `spectrum.csv`.

## 6 Input file reference

Every unknown key or table is an error (with a "did you mean" hint). Each table below is generated from the schema
(`fftpf/run_input.py`); "—" under Default means the key is absent unless given; "required" means the file is refused without
it. Notes under a table say what the table cannot.

### 6.1 `[run]`

<!-- run_input_table:begin run -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `name` | string | — | required | — | Label of the run. Names the automatic output directory and appears in figure titles. |
| `seed` | integer | — | `0` | — | Seed of the run's one random generator (carried through every stage and checkpoint) and the default seed of `random` / `white` starts. |
| `replicas` | integer | — | `1` | ≥ 1 | Number of independent complete runs (seeds `seed + r`) pooled into one spectrum. More than 1 needs an `ac` or `fdt` stage and `measure.spectrum`, and is refused for the film. |

<!-- run_input_table:end run -->

### 6.2 `[material]`

Either `deck`, or the inline tables `landau` (deck units) or `landau_si` (SI), `gradient` or `gradient_si`, `elastic` and
`electrostriction`. Never both. An inline material is complete only with the six sixth-order Landau coefficients `alpha1 … alpha123` (the engine reads all six;
the file is refused at load if one is missing, § 9), the gradient, all of `C11 C12 C44` and all of `Q11_phys Q12_phys Q44_phys`; the four
eighth-order coefficients are optional and default to 0. A `deck` (registry name or `.toml` path) must have the Landau, gradient, elastic and
electrostriction sectors: a substrate-only deck such as `ss316_rt` is for `physics.elastic.substrate` and is refused as `material.deck`.

<!-- run_input_table:begin material -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `deck` | string | — | — | — | Registry name (`mat_registry`) or the path of a `params/*.toml` deck, relative to the input file. Give this or the inline tables, never both. |
| `temperature_K` | number | K | — | > 0 | Reference temperature. Registry decks are built at it; it is the default `T_K` of every stage and makes `langevin` default to true. The slope law needs it (or a stage `T_K`). |

<!-- run_input_table:end material -->

`deck` is a registry name (`python -c "from fftpf import mat_registry; print(mat_registry.list())"`; archived names are refused)
or a path ending in `.toml` or containing a path separator, read as a `params/*.toml` deck, relative to the input file.
Registry decks that take a temperature are re-read at each stage temperature; the substrate-only decks (`ss304_rt`,
`ss316_rt`, `silicon_cubic_rt`, `fused_silica_hpfs_rt`, `mgo_cubic_rt`) are elastic references for `physics.elastic.substrate`,
not materials (§ 9).

`[material.landau]` (deck units; α_deck = α_SI · 10⁻⁹):

<!-- run_input_table:begin material.landau -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `alpha1` | number | deck | — | — | Coefficient of ΣP_i² (second order). Required, with `alpha11`, `alpha12` and (needed by the engine) `alpha111`, `alpha112`, `alpha123`. |
| `alpha11` | number | deck | — | — | Coefficient of ΣP_i⁴. |
| `alpha12` | number | deck | — | — | Coefficient of P_1²P_2² + P_2²P_3² + P_3²P_1². |
| `alpha111` | number | deck | — | — | Coefficient of ΣP_i⁶. |
| `alpha112` | number | deck | — | — | Coefficient of Σ_i P_i⁴ (P_j² + P_k²), j, k ≠ i. |
| `alpha123` | number | deck | — | — | Coefficient of P_1²P_2²P_3². |
| `alpha1111` | number | deck | — | — | Coefficient of ΣP_i⁸ (eighth order; 0 if absent). |
| `alpha1112` | number | deck | — | — | Coefficient of Σ_i P_i⁶ (P_j² + P_k²). |
| `alpha1122` | number | deck | — | — | Coefficient of P_1⁴P_2⁴ + P_2⁴P_3⁴ + P_3⁴P_1⁴. |
| `alpha1123` | number | deck | — | — | Coefficient of P_1⁴P_2²P_3² + P_2⁴P_1²P_3² + P_3⁴P_1²P_2². |

<!-- run_input_table:end material.landau -->

`[material.landau_si]` (SI; the energy density is Σ α_n P^{2n}, P in C/m²):

<!-- run_input_table:begin material.landau_si -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `alpha1` | number or table | SI; K | — | — | Number (J m / C², temperature independent) or the table `{ slope, T0_K }`: a1 = slope · (T − T0(r)), `slope` in J m / (C² K), `T0_K` in K. |
| `alpha11` | number | SI | — | — | Coefficient of ΣP_i⁴, J m⁵ / C⁴. |
| `alpha12` | number | SI | — | — | Coefficient of P_1²P_2² + …, J m⁵ / C⁴. |
| `alpha111` | number | SI | — | — | Coefficient of ΣP_i⁶, J m⁹ / C⁶. |
| `alpha112` | number | SI | — | — | Coefficient of Σ_i P_i⁴ (P_j² + P_k²), J m⁹ / C⁶. |
| `alpha123` | number | SI | — | — | Coefficient of P_1²P_2²P_3², J m⁹ / C⁶. |
| `alpha1111` | number | SI | — | — | Coefficient of ΣP_i⁸, J m¹³ / C⁸. |
| `alpha1112` | number | SI | — | — | Coefficient of Σ_i P_i⁶ (P_j² + P_k²), J m¹³ / C⁸. |
| `alpha1122` | number | SI | — | — | Coefficient of P_1⁴P_2⁴ + …, J m¹³ / C⁸. |
| `alpha1123` | number | SI | — | — | Coefficient of P_1⁴P_2²P_3² + …, J m¹³ / C⁸. |

<!-- run_input_table:end material.landau_si -->

`[material.gradient]` (deck; G11 = G110 · G11_G110, G12 = G110 · G12_G110, G44 = G110 · G44_G110, G44′ = G110 · G44P_G110):

<!-- run_input_table:begin material.gradient -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `G110` | number | deck | — | — | Reference gradient coefficient; G11 = G110 · G11_G110 and so on. Deck units (energy density × nm² / (C/m²)²). |
| `G11_G110` | number | 1 | — | — | G11 / G110. |
| `G12_G110` | number | 1 | — | — | G12 / G110. |
| `G44_G110` | number | 1 | — | — | G44 / G110. |
| `G44P_G110` | number | 1 | — | — | G44′ / G110. |

<!-- run_input_table:end material.gradient -->

`[material.gradient_si]` (SI, isotropic; replaces `gradient`):

<!-- run_input_table:begin material.gradient_si -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `isotropic` | number | J m^3 / C^2 | — | > 0 | Isotropic gradient coefficient β in SI: G11 = β, G12 = 0, G44 = G44′ = β / 2 (converted by `units.si_to_deck_gradient`). Replaces the `gradient` table. |

<!-- run_input_table:end material.gradient_si -->

`[material.elastic]` (GPa):

<!-- run_input_table:begin material.elastic -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `C11` | number | GPa | — | — | Stiffness C11 (cubic). |
| `C12` | number | GPa | — | — | Stiffness C12. |
| `C44` | number | GPa | — | — | Stiffness C44. |

<!-- run_input_table:end material.elastic -->

`[material.electrostriction]`:

<!-- run_input_table:begin material.electrostriction -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `Q11_phys` | number | m^4 / C^2 | — | — | Electrostrictive coefficient Q11 (physical, strain per P²). |
| `Q12_phys` | number | m^4 / C^2 | — | — | Electrostrictive coefficient Q12. |
| `Q44_phys` | number | m^4 / C^2 | — | — | Electrostrictive coefficient Q44 (the engineering-shear convention of the registry decks). |

<!-- run_input_table:end material.electrostriction -->

```toml
# fragment: an inline SI material with the temperature law, see example 4 for a complete file
[material]
temperature_K = 400.0

[material.landau_si]
alpha1 = { slope = 1.02e5, T0_K = 446.0 }   # a1 = slope * (T - T0(r)), J m / (C^2 K) and K
alpha11 = 4.4e7
alpha12 = 1.2e9
```

### 6.3 `[geometry]`

<!-- run_input_table:begin geometry -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `engine` | string | — | required | `pf2d` · `pf3d` · `film` | Which of the three engines runs. Always stated, never inferred from the grid. |
| `N` | list of 2 or 3 integers | cells | required | ≥ 1 each | Grid points `[Nx, Ny]` (`pf2d`) or `[Nx, Ny, Nz]` (`pf3d`, `film`). |
| `cell_nm` | number or list | nm | `1.0` | — | Cell edge: one number for all directions, or a list per direction (2 for `pf2d` and for the film's in-plane cell, 3 for `pf3d`). The film's z spacing comes from `film_thickness_nm`. |
| `slab_thickness_nm` | number | nm | — | > 0 | Out-of-plane thickness Dz of a `pf2d` cell, or Lz of a `pf3d` run at Nz = 1. It is the fluctuation volume of the Langevin noise. |
| `film_thickness_nm` | number | nm | — | > 0 | Film thickness (film only, required there); the grid spacing is thickness / (Nz − 1). |

<!-- run_input_table:end geometry -->

The engine is always named, never inferred: "2D" runs on two engines in practice (`pf2d`, and `pf3d` at Nz = 1 with plane stress).
The grid is `N` cells of size `cell_nm`; the box is N · cell_nm. `pf2d` is periodic in x and y and has no z; `slab_thickness_nm`
is the thickness the Langevin noise assumes. `pf3d` is periodic in all three directions; at Nz = 1, `slab_thickness_nm` is Lz.
`film` is periodic in x and y and non-periodic through the thickness (free surface, short-circuit electrodes, substrate).

### 6.4 `[physics]`

`[physics.electrostatics]`:

<!-- run_input_table:begin physics.electrostatics -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `eps_b` | number | 1 (relative) | the deck's κ_b (inline: 10) | > 0 | Relative background permittivity. Absent: the deck's own κ_b (an inline material then uses 10). |
| `depol_N` | number or 3×3 list | 1 | — | — | Depolarization factor of the k = 0 mode: a number in [0, 1] or a symmetric 3×3 list of lists (`pf3d` only). |

<!-- run_input_table:end physics.electrostatics -->

`[physics.elastic]`:

<!-- run_input_table:begin physics.elastic -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `k0` | string | — | `"stress_free"` | `stress_free` · `clamped` · `off` | Mean-strain boundary condition: `stress_free` (mean stress zero), `clamped` (mean strain zero; `pf3d` only) or `off` (no elasticity; `pf2d`, `pf3d`). |
| `z_mode` | string | — | `"generalized_plane_strain"` | `generalized_plane_strain` · `plane_stress` | Out-of-plane condition of `pf3d` at Nz = 1: uniform ε33 (generalized plane strain) or σ_i3 = 0 per cell (plane stress). |
| `misfit_strain_pct` | number | % | — | — | Misfit strain of the film on its substrate (film only, required there); negative is compressive. |
| `substrate` | string | — | — | — | Registry name of the substrate whose stiffness is used (film only). Absent: the substrate has the film's stiffness. |
| `substrate_layers` | integer | layers | 12 (film) | ≥ 1 | Substrate layers below the film (film only; default 12). |

<!-- run_input_table:end physics.elastic -->

`k0` is the boundary condition of the homogeneous strain: `stress_free` lets the box deform freely, `clamped` holds the mean
strain at zero (`pf3d` only; refused with replicas). `off` switches elasticity off (`pf2d`, `pf3d`; the film always has it).
`z_mode = "plane_stress"` is available only for `pf3d` at Nz = 1 and needs elasticity.

`[physics.noise]`:

<!-- run_input_table:begin physics.noise -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `cutoff_nm` | number | nm | `2.0` | ≥ 0 | Langevin noise length cutoff l_c. 0 means no cutoff; the default is 2.0 nm. |

<!-- run_input_table:end physics.noise -->

### 6.5 `[numerics]`

<!-- run_input_table:begin numerics -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `scheme` | string | — | `"sbdf1"` | `sbdf1` · `sbdf2` | `sbdf1` (first order) or `sbdf2` (second order, deterministic stages only). |
| `dt` | number or "verified" | deck time | `"verified"` | — | Time step, or `"verified"`: the dt audit (`timestep_verify.verified_dt`) runs before the first step. |
| `dt_verify_time` | number | deck time | `100.0` | > 0 | Total deck time of the dt audit. |
| `L` | number | deck mobility | `1.0` | > 0 | Kinetic coefficient (mobility) of the TDGL equation ∂P/∂t = −L δF/δP. |
| `backend` | string | — | `"numpy"` | `numpy` · `numba` · `fused` | `numpy` reference, `numba` kernels, or `fused` (fused Numba step). |
| `threads` | integer | threads | `1` | ≥ 1 | Worker threads, set before NumPy starts. |

<!-- run_input_table:end numerics -->

`scheme = "sbdf2"` is deterministic only: Langevin stages need `sbdf1`. `dt = "verified"` runs `timestep_verify.verified_dt`
(a dt versus dt/2 agreement test over `dt_verify_time` deck time) on the first stage's start field and static fields before
the first step; for a registry deck it adds time to every run, and a fixed `dt` removes it. `backend = "numba"` or `"fused"`
selects the Numba kernels; the thread count is set before NumPy starts. Parameter set-ups go through the project's parameter
review before any production use (rule of 2026-10-19).

### 6.6 `[[stage]]`

Stages run in file order on one state. Every stage has `kind` and an optional `start` table; the other keys depend on `kind`.

<!-- run_input_table:begin stage -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `kind` | string | — | required | `relax` · `anneal` · `cool` · `hold` · `pe_loop` · `ac` · `fdt` | What the stage does. |

<!-- run_input_table:end stage -->

`start` (a table, or a bare string as shorthand: `start = "zero"`) says where the stage's field comes from:

<!-- run_input_table:begin start -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `kind` | string | — | `random` (first stage), `previous` (others) | `random` · `white` · `zero` · `previous` · `from_file` | Where the stage's field comes from. Absent table: `random` for the first stage, `previous` for the others. A bare string is shorthand for `{ kind = ... }`. |
| `amplitude` (random) | number | C/m^2 | `0.005` | > 0 | Amplitude of the band-limited random field. |
| `kmax_frac` (random) | number | fraction of k_max | `0.3` | > 0; ≤ 1 | Upper wavenumber of the random field as a fraction of the grid's k_max. |
| `seed` (random) | integer | — | `run.seed` | — | Seed of the random field. Default `run.seed`. |
| `std` (white) | number | C/m^2 | required | > 0 | Standard deviation of the Gaussian white-noise field, per component. |
| `seed` (white) | integer | — | `run.seed` | — | Seed of the white-noise field. Default `run.seed`. |
| `file` (from_file) | file path | — | required | — | `.npz` file, relative to the input file; its SHA-256 is recorded. |
| `key` (from_file) | string | — | required | — | Name of the array in the file, shape `(3, *space)`, C/m². |

<!-- run_input_table:end start -->

`previous` continues from the previous stage (refused in the first stage); `random` is the engine's band-limited random field;
`white` is a Gaussian white-noise field `N(0, std)` per component and cell; `zero` is P = 0.

#### `relax`

Deterministic stepping (`numerics.scheme`, no noise) under the static `[[field]]` fields. A check is made every 51 steps and at
the last step; it is good when max|L δF/δP| ≤ `stop.tol`, max|P − P at the previous check| < 10⁻⁵, and |f − f previous| / elapsed
deck time < 10⁻⁸; five good checks in a row end the stage as `converged`, otherwise it stops at `n_max`.

<!-- run_input_table:begin stage.relax -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `T_K` | number | K | — | > 0 | Stage temperature. Absent: `material.temperature_K`. |
| `stop.tol` | number | deck force | `1e-06` | > 0 | A check is good when max\|L δF/δP\| ≤ tol (and P and f have stopped changing); 5 good checks in a row end the stage. |
| `stop.n_max` | integer | steps | `50000` | ≥ 1 | Step limit. |

<!-- run_input_table:end stage.relax -->

#### `anneal`

Fixed-temperature stepping with Langevin noise when `langevin` is true (the default with a temperature), else deterministic.
Stop rule: exactly one of `n_steps` (run that many steps) or `block_stationary` with `n_max` (stop when the last `n_blocks`
sampled total energies agree within the relative tolerance; first tested when 2 · n_blocks samples exist).

<!-- run_input_table:begin stage.anneal -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `T_K` | number | K | — | > 0 | Stage temperature. Absent: `material.temperature_K`. |
| `langevin` | boolean | — | true if `material.temperature_K` is given | — | Thermal noise on. Default: true when `material.temperature_K` is given. |
| `stop.n_steps` | integer | steps | — | ≥ 1 | Run exactly this many steps. Exclusive with `block_stationary`. |
| `stop.block_stationary` | number | relative | — | > 0 | Stop when the last `n_blocks` sampled energies agree within this relative tolerance (needs `n_max`). |
| `stop.n_max` | integer | steps | — | ≥ 1 | Step limit with `block_stationary`. |
| `stop.n_blocks` | integer | samples | 4 | ≥ 2 | Samples compared by the stationarity test (default 4). |
| `sample_every` | integer | steps | `0` | ≥ 0 | Steps between samples. 0 samples the last step only (100, or `n_max` if smaller, when `block_stationary` is used). |
| `seed` | integer | — | — | — | Start a new random generator from this seed at the stage. Absent: the run's generator carries on. |

<!-- run_input_table:end stage.anneal -->

#### `cool`

A temperature ladder: `rungs = [[T_K, n_steps], ...]`; the state is carried from rung to rung, and each rung's end state is
saved. The first sample is taken at the first rung's temperature. Langevin by default.

<!-- run_input_table:begin stage.cool -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `rungs` | list of [T_K, n_steps] | — | required | — | List of `[T_K, n_steps]` pairs: the temperature ladder; the state is carried from rung to rung. |
| `langevin` | boolean | — | true if `material.temperature_K` is given | — | Thermal noise on. Default: true when `material.temperature_K` is given. |
| `seed` | integer | — | — | — | Start a new random generator from this seed at the stage. Absent: the run's generator carries on. |
| `record_every` | integer | steps | 100 | ≥ 1 | Steps between trace rows. |

<!-- run_input_table:end stage.cool -->

#### `hold`

`n_steps` at constant T and constant field `E_kV_cm` along `axis`. With `ramp_steps > 0` the field moves linearly from its
value on entry (the field the previous stage ended with) to `E_kV_cm` over `ramp_steps`, then stays. A hold with a field or a
ramp needs an `axis`, given here or in an earlier `hold` or `pe_loop`.

<!-- run_input_table:begin stage.hold -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `T_K` | number | K | — | > 0 | Stage temperature. Absent: `material.temperature_K`. |
| `n_steps` | integer | steps | required | ≥ 1 | Length of the hold. |
| `E_kV_cm` | number | kV/cm | `0.0` | — | Applied field along `axis`; 0 holds under the static `[[field]]` E_ext alone. |
| `axis` | list of 3 numbers | 1 | — | — | Field direction (any length, normalised). Absent: the axis of an earlier `hold` / `pe_loop`. |
| `ramp_steps` | integer | steps | `0` | ≥ 0 | Steps over which the field ramps linearly from its value on entry to `E_kV_cm`. |
| `langevin` | boolean | — | true if `material.temperature_K` is given | — | Thermal noise on. Default: true when `material.temperature_K` is given. |
| `seed` | integer | — | — | — | Start a new random generator from this seed at the stage. Absent: the run's generator carries on. |
| `record_every` | integer | steps | 100 | ≥ 1 | Steps between trace rows. |

<!-- run_input_table:end stage.hold -->

#### `pe_loop`

A polarization-field loop along `axis`. **Triangle**: E(t) = Emax · tri(t), starting at E = 0 and rising to +Emax, through 0 to
−Emax and back to 0, `steps_per_cycle` steps per cycle (a multiple of 4), `cycles` cycles (a multiple of 0.25, default 1);
records every `record_every` steps starting at stage entry, and `record_every` must divide the quarter cycle, so the extremes
and zero crossings are recorded points. Event names on the saved states: `c<k>_Emax`, `c<k>_E0down`, `c<k>_E0up` (§ 3).
**Unipolar**: E rises 0 → Emax in `n_points` equal steps and returns to 0 in `n_points` steps, `n_dwell` steps at each field point,
one record per point; deterministic by default (`langevin` defaults to false for it), `sbdf1` only. `sine` and `staircase` are in the schema and refused as not built.

<!-- run_input_table:begin stage.pe_loop -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `T_K` | number | K | — | > 0 | Stage temperature. Absent: `material.temperature_K`. |
| `axis` | list of 3 numbers | 1 | required | — | Field direction (any length, normalised). |
| `E_max_kV_cm` | number | kV/cm | required | > 0 | Field amplitude Emax. |
| `waveform` | string | — | `"triangle"` | `triangle` · `sine` · `staircase` · `unipolar` | `triangle` and `unipolar` are built; `sine` and `staircase` are refused. |
| `cycles` | number | cycles | 1 (triangle) | > 0 | Triangle: number of cycles, a multiple of 0.25 (default 1). |
| `steps_per_cycle` | integer | steps | — | ≥ 1 | Triangle: steps per cycle, a multiple of 4 (required there). |
| `n_points` | integer | field points | — | ≥ 1 | Unipolar: field points n1 on the way up (required there). |
| `n_dwell` | integer | steps | — | ≥ 1 | Unipolar: steps at each field point (required there). |
| `langevin` | boolean | — | true if `material.temperature_K` is given | — | Thermal noise on. Default: true when `material.temperature_K` is given. |
| `seed` | integer | — | — | — | Start a new random generator from this seed at the stage. Absent: the run's generator carries on. |
| `record_every` | integer | steps | largest divisor of the quarter cycle ≤ 100 | ≥ 1 | Triangle: steps between records, a divisor of the quarter cycle (default: the largest divisor ≤ 100). |

<!-- run_input_table:end stage.pe_loop -->

#### `ac`

Small-signal dielectric response by lock-in. For each tone ω (rad per deck time) the stage starts from a copy of the entry state and
steps `periods` periods under E(t) = (`dc_bias` + E0 · sin ω t) · n on top of the static fields; the first `discard_periods` periods
are dropped before the lock-in. Tones run in order and share the carried generator; the stage ends in the entry state (it is
a measurement). Definitions of χ and ε: § 8.3.

<!-- run_input_table:begin stage.ac -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `T_K` | number | K | — | > 0 | Stage temperature. Absent: `material.temperature_K`. |
| `langevin` | boolean | — | true if `material.temperature_K` is given | — | Thermal noise on. Default: true when `material.temperature_K` is given. |
| `seed` | integer | — | — | — | Start a new random generator from this seed at the stage. Absent: the run's generator carries on. |
| `axis` | list of 3 numbers | 1 | required | — | Drive and read-out direction. |
| `E0_kV_cm` | number | kV/cm | required | > 0 | Amplitude E0 of the small-signal sine drive. |
| `omega` | list of numbers | rad / deck time | required | > 0 each | Angular frequencies of the tones, reduced (rad per deck time); distinct to 6 significant digits. |
| `periods` | integer | periods | required | ≥ 1 | Periods per tone, including the discarded ones. |
| `discard_periods` | integer | periods | `0` | ≥ 0 | Initial periods dropped before the lock-in (must be < `periods`). |
| `dc_bias_kV_cm` | number | kV/cm | `0.0` | — | Constant field added to the drive. |

<!-- run_input_table:end stage.ac -->

#### `fdt`

The zero-field fluctuation route: `burn_steps` steps discarded, then `n_steps` steps recorded under the static fields alone, with
Langevin noise (required). The series gives χ″ by a Welch estimate and χ′ from the correlation function (§ 8.3). The stage ends
in the last state of the trajectory.

<!-- run_input_table:begin stage.fdt -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `T_K` | number | K | — | > 0 | Stage temperature. Absent: `material.temperature_K`. |
| `langevin` | boolean | — | true if `material.temperature_K` is given | — | Thermal noise on. Default: true when `material.temperature_K` is given. |
| `seed` | integer | — | — | — | Start a new random generator from this seed at the stage. Absent: the run's generator carries on. |
| `n_steps` | integer | steps | required | ≥ 1 | Steps of the recorded equilibrium series. |
| `nperseg` | integer | samples | required | ≥ 2 | Welch segment length (≤ `n_steps`). |
| `burn_steps` | integer | steps | `0` | ≥ 0 | Steps run and discarded before the series. |
| `max_lag` | integer | samples | min(`nperseg`, `n_steps` // 8) | ≥ 2 | Largest lag of the correlation function. Default min(nperseg, n_steps // 8). |
| `axis` | list of 3 numbers | 1 | — | — | Read-out direction. Absent: the axis of the file's `ac` stage. |

<!-- run_input_table:end stage.fdt -->

### 6.7 `[[field]]`

Quenched fields, any number, each seeded and recorded in `provenance.json`. Targets: `E_ext` (adds to the static applied
field), `T0` (adds to the T0 landscape of the slope law), `landau.<coef>` (adds to the deck coefficient, or replaces it with a
`from_file` array; one field per coefficient), `eps_star` (refused, not built).

<!-- run_input_table:begin field -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `target` | string | — | required | — | What the field adds to: `E_ext`, `T0` (needs the slope law), `landau.<coef>` (one field per coefficient), or `eps_star` (refused: not built). |
| `kind` | string | — | required | `gaussian` · `from_file` | `gaussian` (generated, seeded) or `from_file`. |
| `sigma_kV_cm` (gaussian) | number | kV/cm | — | > 0 | Target `E_ext`: rms of each Cartesian component. |
| `sigma` (gaussian) | number | deck (of the coefficient) | — | > 0 | Target `landau.<coef>`: rms added to the deck value, in deck units of the coefficient. |
| `sigma_K` (gaussian) | number | K | — | > 0 | Target `T0`: rms of the T0 shift. |
| `correlation_nm` (gaussian) | number | nm | required | > 0 | Correlation length ξ of the Gaussian field. |
| `seed` (gaussian) | integer | — | — | — | Seed. For `E_ext` the components use `seed`, `seed + 1`, `seed + 2`. |
| `seeds` (gaussian) | list of 3 integers | — | — | — | Target `E_ext` only: one seed per component (three distinct integers); exclusive with `seed`. |
| `file` (from_file) | file path | — | required | — | `.npz` file, relative to the input file; its SHA-256 is recorded. |
| `key` (from_file) | string | — | required | — | Name of the array: shape `(3, *space)` for `E_ext` in deck units (1 deck field unit = 1 V/nm), `space` for the others (K for `T0`, the coefficient in deck units for `landau.<coef>`). |

<!-- run_input_table:end field -->

A Gaussian field is zero-mean with the stated rms and correlation length ξ (`disorder_fields.random_field_gaussian`, `…_3d`).
For `E_ext` the three Cartesian components are independent fields: with `seed` they use `seed`, `seed + 1`, `seed + 2`; with
`seeds = [s1, s2, s3]` component c is built from `seeds[c]` (the NBT6BT driver's order of operations; the two forms differ in the last
bits). Gaussian fields are not built for the film; `kind = "from_file"` is the door for every custom construction:

```toml
# fragment: a quenched field map from a file made with numpy.savez
[[field]]
target = "E_ext"
kind = "from_file"
file = "E_map.npz"          # relative to the input file
key = "E"                   # array of shape (3, Ny, Nx) [pf2d] or (3, Nz, Ny, Nx), in deck units (1 deck field unit = 1 V/nm)
```

### 6.8 `[measure]`

<!-- run_input_table:begin measure -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `mean_P` | boolean | — | `true` | — | Write the volume-mean P (`Px`, `Py`, `Pz`) at every record. |
| `energy_terms` | boolean | — | `false` | — | Write the energy totals by term (`E_bulk`, `E_grad`, `E_elec`, `E_elastic`). |
| `loop` | list of strings | — | — | `Pr` · `Ec` · `Pmax` · `W_store` · `W_rec` · `W_loss` · `eta` | Loop measures read from each `pe_loop` stage (needs one). `Ec` is for the triangle only. |
| `seconds_per_deck_time` | number | s | — | > 0 | Calibration of deck time. Only with it is an `f_Hz` column written; without it frequencies are reduced. |

<!-- run_input_table:end measure -->

`[measure.phase_fractions]` (definitions in § 8.2):

<!-- run_input_table:begin measure.phase_fractions -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `scheme` | string | — | `"rt"` | `rt` · `rto` | `rt`: R / T on ⟨111⟩ / ⟨100⟩ (needs `P_ref`); `rto`: T / O / R cones about ⟨100⟩ / ⟨110⟩ / ⟨111⟩. |
| `P_ref` | number | C/m^2 | — | > 0 | Reference rhombohedral \|P\| (scheme `rt`, required there). |
| `angle_deg` | number | deg | 15 | > 0; ≤ 25 | Half-angle of the cones. Default 15; at most 25 (`rt`) or 17.5 (`rto`). |
| `P_min` | number | C/m^2 | `0.05` | > 0 | Below this \|P\| a cell is `low`. |
| `R_factor` | number | 1 | 0.7 (`rt`) | > 0 | R needs \|P\| ≥ R_factor · P_ref (scheme `rt` only; default 0.7). |

<!-- run_input_table:end measure.phase_fractions -->

`[measure.domain_map]`. A selector is `"<stage kind or index>:<event>"`: the stage is a kind (`pe_loop`, every stage of that kind) or an index
(`2`); the event is `end` (the state at the stage's last step; a triangle loop that stops at −Emax has none), `E0+` (= `E0down`,
E = 0 falling, the positive remanent state), `E0-` (= `E0up`, E = 0 rising, the negative remanent state), `Emax`, a rung name
(`T350`) or any event name of § 3. A triangle event written without its cycle number means the *last* cycle that has it; ask
for `c1_E0down` for another. A selector that matches no saved state is refused.

<!-- run_input_table:begin measure.domain_map -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `at` | "all" or list of strings | — | `"all"` | — | `"all"` or a list of selectors `"<stage kind or index>:<event>"` naming the saved states to draw. |
| `kind` | list of strings | — | `["direction"]` | `phase` · `direction` | Which maps to draw: `direction` (signed RGB of P/\|P\|), `phase` (needs `phase_fractions`). |

<!-- run_input_table:end measure.domain_map -->

`[measure.spectrum]`:

<!-- run_input_table:begin measure.spectrum -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `quantities` | list of strings | — | required | `eps_real` · `eps_imag` · `tan_delta` | Quantities drawn in `figures/spectrum.png` (`spectrum.csv` holds all of them). |
| `routes` | list of strings | — | required | `lockin` · `fdt` | `lockin` reads the `ac` stage, `fdt` the `fdt` stage; each needs exactly one such stage. |

<!-- run_input_table:end measure.spectrum -->

### 6.9 `[output]`

<!-- run_input_table:begin output -->
<!-- GENERATED by tools/run_input_table.py from fftpf/run_input.py SCHEMA -- do not hand-edit inside these markers. Regenerate:  python3 tools/run_input_table.py -->

| Key | Type | Unit | Default | Allowed | Meaning |
|---|---|---|---|---|---|
| `dir` | string | — | `"auto"` | — | `auto` = `reports/runs/<YYYYMMDD-HHMM>_<run.name>/`, or a directory (new or empty, or holding a checkpoint of the same configuration). |
| `checkpoint_every` | integer | steps | `0` | ≥ 0 | Write a checkpoint every this many steps; 0 writes none. Re-running the command then resumes. |

<!-- run_input_table:end output -->

## 7 Run time

Examples 1–5 ran in 2–9 seconds each on one core (§ 12). A stage of N steps costs N engine steps; `dt = "verified"` adds an audit run
(`numerics.dt_verify_time` deck time at dt and at dt/2) before the first step, which a fixed `dt` avoids. No other statement about
scaling or about production sizes is made here.

## 8 Definitions

### 8.1 Loop measures (`measure.loop`)

Read from the recorded (E, Pn) series of each `pe_loop` stage, E in kV/cm along the axis and Pn = mean P · axis in C/m². Triangle: the
**last full cycle** k = ⌊cycles⌋, with m records per quarter cycle; points j = 0 … 4m (j = 0 and 4m: E = 0 rising; m: +Emax;
2m: E = 0 falling; 3m: −Emax).

| name | definition | unit |
|---|---|---|
| `Pmax` | Pn at +Emax (j = m) | C/m² |
| `Pr` | ½ (Pn at the falling E = 0 point − Pn at the rising E = 0 point) = (Pn[2m] − Pn[4m]) / 2 | C/m² |
| `Ec` | ½ (E_rise − E_fall): E_rise is the field where Pn crosses zero going up inside j = 0 … m, E_fall where it crosses zero going down inside j = m … 3m, each by linear interpolation between recorded points; `null` with an `Ec_note` when a branch has no crossing | kV/cm |
| `W_store` | ∫ E dP over the charging branch (j = 0 … m: 0 → +Emax), trapezoid rule | MJ/m³ |
| `W_rec` | −∫ E dP over the discharging branch (j = m … 2m: +Emax → 0), so a recoverable energy is positive | MJ/m³ |
| `W_loss` | `W_store` − `W_rec` | MJ/m³ |
| `eta` | `W_rec` / (`W_rec` + `W_loss`) = `W_rec` / `W_store` | 1 |

The W are computed on the positive half of the cycle only (`pf2d_energy_storage_loop.energy_integrals`); the deck-unit values
are written as `W_*_deck` (deck energy density × 10³ = MJ/m³). A triangle loop needs at least one full cycle (`cycles ≥ 1`).
Unipolar: `Pmax` = Pn at the top point, `Pr` = Pn after the discharge (E = 0), and the W and η as above; `Ec` is not defined
for it and is refused.

### 8.2 Phase fractions and maps (`measure.phase_fractions`, `measure.domain_map`)

P is in C/m². With c111 = Σ|P_i| / (√3 |P|) and c100 = max|P_i| / |P| (the cosines to the nearest ⟨111⟩ and ⟨100⟩):

* **Scheme `rt`** (default; the classifier of the NBT6BT readout, operation for operation). `R`: c111 ≥ cos(`angle_deg`) and
  |P| ≥ `R_factor` · `P_ref`. `T`: c100 > c111 and |P| ≥ `P_min` (closer to ⟨100⟩ than to ⟨111⟩). `T15`: c100 ≥ cos(`angle_deg`)
  and |P| ≥ `P_min` (a subset of `T` for `angle_deg` ≤ 25°). `low`: |P| < `P_min`. `other`: |P| ≥ `P_min` and neither R nor T.
  R, T, other, low partition the cells; T15 is inside T. Defaults: `angle_deg` 15, `P_min` 0.05 C/m², `R_factor` 0.7; `P_ref` is
  required (the reference rhombohedral |P|).
* **Scheme `rto`**. With a1 ≥ a2 ≥ a3 the sorted |P_i|: c100 = a1/|P|, c110 = (a1 + a2) / (√2 |P|), c111 = (a1 + a2 + a3) / (√3 |P|). A cell
  with |P| ≥ `P_min` is `T`, `O` or `R` when the cosine to the nearest ⟨100⟩, ⟨110⟩ or ⟨111⟩ is ≥ cos(`angle_deg`), else `other`;
  |P| < `P_min` is `low`. `angle_deg` is at most 17.5° (half of the 35.26° between ⟨110⟩ and ⟨111⟩ is 17.63°) so that the cones are disjoint
  and T, O, R, other, low partition the cells. `P_ref` and `R_factor` are refused in this scheme.

Fractions are plain cell means for `pf2d` and `pf3d`; for the film each z layer's in-plane mean is weighted by the grid's
trapezoid weights `qw`. The columns of `trace.csv` are `f_R f_T f_T15 f_other f_low` (`rt`) or `f_R f_O f_T f_other f_low` (`rto`).
`fractions_at` holds the fractions of each saved event state.

Maps (`figures/<state stem>_<kind>.png`, mid-z plane for `pf3d` and film, nm axes): `direction` is the signed RGB colour
0.5 + 0.5 P/|P| (⟨111⟩ directions are the eight corner colours, ⟨100⟩ the six face colours; |P| does not enter; a key is drawn);
`phase` colours each class (R dark, T light; `rto`: R navy, O vermillion, T pale yellow) with the box fractions in the legend.
`rto` maps on `pf3d` and film have not been drawn in any test.

### 8.3 Dielectric spectrum (`ac`, `fdt`, `measure.spectrum`)

The observable is the raw series s(t) = Σ over cells of P·n (n the unit axis; for n = x exactly `P[0].sum()`). N is the number
of cells, V_cell the cell volume, kT the Langevin energy *density* of § 4, E0 the amplitude in deck field units.

* **Lock-in (`ac`).** χ_deck(ω) = the complex amplitude of the mean polarization p = s/N divided by E0, obtained as
  `lockin_chi(s, dt, ω, N · E0)` (a uniform E0 couples to the sum s with N · E0). Discarded periods are dropped and the
  phase of the skipped time is restored. χ = χ′ + iχ″ with the convention χ0 / (1 − iωτ), so χ″ > 0.
* **Fluctuation route (`fdt`).** χ″(ω) = ω S_p(ω) N / (2 kT) per volume, with S_p the power spectral density of the mean polarization (Welch
  estimate, `welch_chi_pp`, called with kT_eff = N · kT); χ′ follows from the correlation function (`fl_chi`); the static limit is
  χ(0) = N · Var(p) / kT = V · Var(p) / (k_B T) with V the box volume. This per-volume form is what makes the result independent of the cell count.
  The lowest one or two Welch bins of χ″ read low (existing estimator, unchanged); `measure.spectrum` therefore keeps only the contiguous resolved range of
  the `fdt` route (judged from the spread over the replicas, or over four sub-series of the one trajectory).
* **From χ to ε.** ε′ = ε_b + χ′ / ε0_deck, ε″ = χ″ / ε0_deck, tan δ = ε″ / ε′, with ε0_deck = 0.0088541878128 (the vacuum permittivity in
  deck units: P in C/m² over E in V/nm) and ε_b from `physics.electrostatics.eps_b`, else the deck's κ_b / ε0_deck. χ is the response of the order parameter;
  ε_b adds the background.
* **Frequency.** `omega` is in rad per deck time. A column `f_Hz = omega / (2π · seconds_per_deck_time)` appears only when
  `measure.seconds_per_deck_time` is given; this layer does not determine that calibration.

## 9 Refusals

Every refusal is a `ConfigError` printed as `error: <key.path>: <reason>` with exit code 2. **Check** = raised by `check` and `explain`
(the dry build of § 2, which is also the first thing `run` does), and therefore by `run` before its first step; **during** = found only while the run
directory is being used (it needs the files of an earlier run). Index brackets are written `[i]`. A command line that is wrong
(usage, file missing) is exit 2 without a key path.

**Whole file.** Check: an unknown key or table (with a "did you mean" hint); invalid TOML; a missing required table or key
(`[run]`, `[material]`, `[geometry]`, `run.name`, `geometry.engine`, `geometry.N`, at least one `[[stage]]`); a wrong type, a value outside its range, a value that is not
one of the allowed choices (the tables of § 6 give them).

**`[run]`.** Check: `run.replicas` above 1 for the film; `run.replicas` above 1 without an `ac` or `fdt` stage, or without `measure.spectrum`; `physics.elastic.k0` = `clamped` with `run.replicas` above 1.
During: `output.dir` holding a replicated run of another configuration.

**`[material]`.** Check: `material.deck` together with inline tables; neither of the two; `material.deck` an archived deck, an unknown
registry name (with a suggestion), or a `.toml` path that does not exist; `material.landau` together with `material.landau_si`, `material.gradient` together with
`material.gradient_si`; a missing `gradient`, `elastic`, `electrostriction`, or one of the six sixth-order coefficients `alpha1`, `alpha11`, `alpha12`, `alpha111`, `alpha112`, `alpha123`; a missing key in an inline
`material.<table>`; the `{ slope, T0_K }` form of `alpha1` in `material.landau` (it is
for `landau_si` only) or with a key missing or unknown; `material.temperature_K` missing when the slope law is used and no stage states a temperature;
`stage[i].T_K` (or `stage[i].rungs`) at a temperature other than the deck's with a fixed-temperature deck; `material.deck` that cannot be read, or has no Landau, gradient, elastic or electrostriction sector
(a substrate deck such as `ss316_rt`, or `pto_haun1987`, which has no gradient); `material.deck` whose gradient, elastic or electrostriction sector changes with T.

**`[geometry]`.** Check: `geometry.N` with the wrong number of entries for the engine; `geometry.cell_nm` as a list of the wrong length;
`geometry.slab_thickness_nm` for an engine other than `pf2d` or `pf3d` at Nz = 1; `geometry.film_thickness_nm` missing for the film or given for another engine.

**`[physics]`.** Check: `physics.electrostatics.depol_N` for an engine other than `pf3d`; `physics.elastic.z_mode` = `plane_stress` unless `pf3d` at Nz = 1;
`physics.elastic.misfit_strain_pct` missing for the film or given for another engine; `physics.elastic.substrate` and `substrate_layers` for
another engine, or a substrate that is not a registry deck or has no elastic sector; `physics.elastic.k0` = `clamped` on `pf2d`; `physics.elastic.k0` other than `stress_free`
on the film; `physics.elastic.z_mode` = `plane_stress` with `physics.elastic.k0` = `off`; a substrate the registry cannot build at the temperature.

**`[numerics]`.** Check: `stage[i].langevin` with `numerics.scheme` = `sbdf2`; `numerics.scheme` other than `sbdf1` with a unipolar loop.
`material.gradient` not isotropic with G11 > 0 (G12 = 0, G44 = G44′ = G11/2) for a film Langevin run.

**`[[stage]]`.** Check: `stage[0].start` = `previous`; a `from_file` start whose file does not exist; `stage[i].kind` `ac` or `fdt` on the film;
`stage[i].waveform` = `sine` or `staircase`; `stage[i].langevin` for `fdt` when false, without a temperature, for `pf2d` without `geometry.slab_thickness_nm`, or true for a
unipolar loop; `stage[i].axis` the zero vector, or missing for a `hold` with a field or ramp, or an `fdt` with no `ac` stage to follow;
anneal `stop` with both or neither of `n_steps` and `block_stationary`, or `n_max` / `n_blocks` with the wrong one;
`ramp_steps` above `n_steps`; `ac` `discard_periods` not below `periods`, two tones equal to six significant digits; `fdt` `nperseg` above `n_steps`, `max_lag` out of range;
`pe_loop`: triangle without `steps_per_cycle`, `steps_per_cycle` not a multiple of 4, `cycles` not a multiple of 0.25, `record_every` not dividing the quarter cycle, `n_points` / `n_dwell` on a
triangle, `steps_per_cycle` / `cycles` / `record_every` on a unipolar loop, a unipolar loop without `n_points` or `n_dwell`; a `from_file` start array of the wrong shape or key.

**`[[field]]`.** Check: `field[i].target` unknown, or `landau.<coef>` with an unknown coefficient; the wrong sigma key for the target
(`sigma_kV_cm` for `E_ext`, `sigma_K` for `T0`, `sigma` for `landau.<coef>`); `field[i].seed` / `seeds` missing, both given, `seeds` for another target than `E_ext`, or three seeds that are not distinct;
`field[i].target` = `T0` without the slope law; `landau.alpha1` with the slope law (use `T0`); `field[i].target` = `eps_star` (not built); `field[i].kind` = `gaussian`
for the film; `field[i].file` with an array of the wrong shape, `field[i].key` not in the file; two fields for the same `landau.<coef>`.

**`[measure]`.** Check: `measure.loop` without a `pe_loop` stage, a name the loop's waveform cannot give (`Ec` for a unipolar loop), a triangle with fewer than one full cycle;
`measure.domain_map.kind` = `phase` without `measure.phase_fractions`; `measure.domain_map.at` a selector that is not `"<stage>:<event>"` or matches no saved state;
`measure.spectrum.routes` naming a route without its stage, or more than one stage of that kind; `measure.phase_fractions`: `P_ref` missing in `rt`, `R_factor · P_ref` below `P_min`, `P_ref` or `R_factor` given in `rto`,
`angle_deg` above 17.5 in `rto`. During: `measure.spectrum` finding a spectrum file missing.

**`[output]`.** Check: `output.dir` (or, for `auto`, `FFT_RUN_ROOT`) that cannot be created or written (the message names the key path `output.dir`; `check` creates nothing).
During: `output.dir` existing and not empty without a checkpoint (a run never overwrites); `output.dir` holding a checkpoint of a different configuration;
`output.dir` with a checkpoint but no `quenched.npz`; `trace.csv` missing or shorter than the checkpoint expects.

**In-process use.** Raised by `run_protocol.run` before its first step, not by `check` (it depends on the state of the Python process, not on the file): `numerics.threads` when NumPy was imported before the BLAS thread variables were set. The message names
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` (all to be `1` before NumPy is imported) and says that `python -m fftpf` sets them for you (§ 11).
`check` and `explain` take no backend and do not need them.

**Failures that are not refusals.** These end in a traceback, not in a message with a key path: `FloatingPointError`, when a stage produced a non-finite field (the run
directory is kept and the stage written; exit 1). Whether a combination stays finite is not decidable from the file.

## 10 Not built, not verified

Taken from SPEC § 16.10–16.14 and the code; nothing here is softened.

**Not built.** Waveforms `sine` and `staircase` (refused). Field target `eps_star` (refused). Gaussian fields on the film
(refused; use `from_file`). `ac` and `fdt` on the film (refused: no anchor case exists). Wall metrics (SPEC § 16.5 E4 lists them; no `measure` key exists). Domain maps of other
kinds than `direction` and `phase`. Mid-stage checkpoints in `relax`, `anneal`, `ac` and `fdt` (they checkpoint at the stage end only). A `cluster` / `sweep` table and job submission (E6). Re-expressed
`cases/` drivers (E7). A reduced-frequency calibration (`seconds_per_deck_time` can be stated; this layer does not determine it). Composite region mixtures, polycrystal
orientation fields, charge and defect layers, AFE / tilt sectors, 0D landscape scans. Landau forms beyond the engine polynomial. Substrate-only registry decks as a material (§ 9).

**Not verified (portable tests only; no CORE gate item exists for any of it).** The whole layer: SPEC § 5 gives no capability without a core-gate item, and none covers this
layer. Specifically: backends `numba` and `fused` through the entry point (`numba` on `pf3d` is covered by the NBT6BT anchor, which needs the local `Runs/` folder and is
skipped without it; `fused` and `numba` on `pf2d` ran by hand only; `fused` on `pf3d` and either backend on the film are untested); `numerics.threads` above 1; the film against a direct-engine reference for any stage but `relax`
(`anneal`, `cool`, `hold`, `pe_loop` on the film are smoke runs; film resume is a smoke run); the film with an array field from a file; a unipolar loop on `pf3d` or the film; `T0` read from
a file; the driver comparison at production size; a dc bias or a static `[[field]]` under `ac`; `dt = "verified"` with replicas; the lowest Welch bins of `fdt`; `rto` domain maps on `pf3d` and the film;
`relax`, and `fdt` on `pf3d`, beyond a hand run. Two anchors depend on files outside the repository: the NBT6BT anchors read `Runs/NBT6BT/two_well_20261104/` (found through `FFT_RUNS_ROOT`, else beside the git
common directory), and the WP13 anchors import `cases/wp13_debye_case.py`; without them the tests are skipped. The statistical tests of the spectrum use fixed seeds with a 3-standard-error threshold.
The quadratic Debye model of example 3 is a verification model, not a material.

**Parameter sets.** The numbers in the examples are illustrative. A production set-up passes the project's parameter review (units, conventions, plausibility) before it is run.

## 11 From a script

The layer is ordinary Python. `fftpf.run_input.load(path)` or `resolve(dict, base_dir)` returns the resolved configuration;
`fftpf.run_builder.build(cfg)` makes the grid, operators, elastic setup, fields and the adapter without taking a step;
`fftpf.run_protocol.run(path_or_cfg)` runs it and returns `dict(run_dir, build, stages, P, n_checkpoints)`.

Set the thread variables **before NumPy is imported** (the CLI does it for you):
`os.environ.update(fftpf.backend_config.environment(1))`. Without them `run` stops with a `ConfigError` on `numerics.threads`
that names `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` as soon as NumPy is already loaded;
`run_builder.check(cfg)` (the dry build) works without them.

Custom arrays go in through `kind = "from_file"` (a field) or a `from_file` start. The example writes a quenched E_ext map and runs a
short anneal with it:

```python run
import os
import tempfile
import tomllib

from fftpf import backend_config
os.environ.update(backend_config.environment(1))          # before numpy is imported
import numpy as np
from fftpf import run_input, run_protocol

work = tempfile.mkdtemp()
rng = np.random.default_rng(0)
E = np.zeros((3, 16, 16))                                  # deck units: 1 deck field unit = 1 V/nm
E[0] = 2e-4 * rng.standard_normal((16, 16))                # rms 2e-4 V/nm = 2 kV/cm along x
np.savez(os.path.join(work, "E_map.npz"), E=E)

raw = tomllib.loads(f"""
[run]
name = "from_script"
seed = 1
[material]
deck = "bto_lcc2005_8th"
temperature_K = 298.0
[geometry]
engine = "pf2d"
N = [16, 16]
slab_thickness_nm = 1.0
[numerics]
dt = 0.02
[[field]]
target = "E_ext"
kind = "from_file"
file = "E_map.npz"
key = "E"
[[stage]]
kind = "anneal"
stop = {{ n_steps = 200 }}
[output]
dir = "{os.path.join(work, 'out')}"
""")
cfg = run_input.resolve(raw, base_dir=work)
res = run_protocol.run(cfg)
print(res["run_dir"].endswith("out"), res["P"].shape, [round(float(x), 6) for x in res["build"]["adapter"].mean_P(res["P"])])
```

With the builder alone, `b = run_builder.build(cfg)` gives `b["adapter"]` (`step(P, E, rng=None, P_prev=None)`, `mean_P`, `energy`,
`random_start`), the field `b["E_static"]`, `b["dt"]` and `b["kT"]`; a script that wants its own stepping loop uses exactly these.
For a different field in the same file, the `from_file` start of a stage (`start = { kind = "from_file", file = ..., key = ... }`,
shape `(3, *space)`, C/m²) hands a prepared initial state to a stage.

## 12 Worked examples

The five complete files are in `docs/run_examples/`. Each passes `check`; the numbers quoted below are from a run on 2026-11-13
(macOS, one core, NumPy 1.26.4, `FFT_RUN_ROOT` pointing at a scratch folder) and are reproducible on the same machine, not across
platforms. **The parameters are illustrative: sizes and step counts are chosen so that a run takes seconds, not so that it is converged,
and no result below is a claim about a material.** A production set-up passes the project's parameter review first.

Commands, run from the repository root with the Python that has numpy and scipy:

```
python -m fftpf check   docs/run_examples/01_anneal_pf2d.toml
python -m fftpf explain docs/run_examples/01_anneal_pf2d.toml
python -m fftpf run     docs/run_examples/01_anneal_pf2d.toml
```

### Example 1 — anneal on `pf2d` (`01_anneal_pf2d.toml`, 2 s)

A BaTiO3 registry deck at 298 K on a 32 × 32 periodic box (1 nm cells, 1 nm slab), random start, 4000 Langevin steps at fixed
dt = 0.02, energy terms recorded. It shows the minimal shape of a calculation: material, geometry, one stage, one measure.

`explain` prints the choices before anything runs: the engine functions, `kT = units.kT_langevin(298.0, 1.0, 1.0, 1) = 0.00411433`, the cutoff (2.0 nm),
and `stage[0] anneal: start random, stop {'n_steps': 4000}, langevin True, sample_every 200`. After `run`, look at `trace.csv` (20 rows, energy by term) and `measures.json`:

```
"stop_reason": "n_max", "steps": 4000, "t": 80.0
"end": { "E_total": 3.3636570383620867, "f_total": 0.0032848213265254753,
         "mean_P": [0.02441996879363248, 0.012377592317862007, 0.007740033661268086],
         "terms": { "bulk": -1.2242366527580066, "elastic": 0.5821090431903714,
                    "elec": 1.3760628969303084, "grad": 2.6297217509994137 } }
```

The mean P of a thermal, polydomain box is near zero; the gradient energy is the cost of the domain walls the random start produced.
`stop_reason` is `n_max` for a fixed-length anneal; the stop rule is only a step count here.

### Example 2 — cool, hold, P-E loop, loop measures, domain maps (`02_cool_hold_pe_loop.toml`, 7 s)

The same deck cooled 400 → 350 → 298 K in three rungs of 1500 steps, held 1000 steps, then two triangle cycles to Emax = 500 kV/cm
(6000 steps per cycle) along x. `measure.loop` reads the last cycle; `phase_fractions` uses scheme `rto`; `domain_map` draws phase and
direction maps of the end of the cooling and of the two remanent states of the last cycle.

```
python -m fftpf run docs/run_examples/02_cool_hold_pe_loop.toml
```

`measures.json`, stage 2:

```
"loop": { "cycle": 2, "cycles_run": 2.0, "waveform": "triangle",
          "Pmax": 0.28785060586776356, "Pr": 0.22442120811248498, "Ec": 274.1917038508307,
          "W_rec": 1.2426260695599365, "eta": 0.08945575136139068,
          "W_store_deck": 0.013890957827181773, "W_rec_deck": 0.0012426260695599365, "W_loss_deck": 0.012648331757621837 }
"fractions_end": { "R": 0.0303, "O": 0.2383, "T": 0.1514, "other": 0.5801, "low": 0.0 }
```

(`Pmax`, `Pr` in C/m², `Ec` in kV/cm, `W_rec` in MJ/m³; the `*_deck` entries are the same energies in deck units, 10⁹ J/m³; only the
requested names plus the deck-unit W appear.) A cycle is 6000 steps = 120 deck time units, far faster than a real
measurement, so `Ec` is inflated by the sweep rate and the loop is not a material property. Look at `figures/`: `state_2_pe_loop_c2_E0down_phase.png` and
`..._E0up_phase.png` (the two remanent states), `state_0_cool_T298_direction.png`, and in `trace.csv` the `Pn` against `E_kV_cm` of stage 2 (the loop itself; 161 rows).

### Example 3 — dielectric spectrum, both routes, replicas (`03_dielectric_spectrum.toml`, 6 s)

**A verification model, not a material.** A purely quadratic Landau energy (α1 = 0.6 deck units, all other α zero) on an 8 × 8 box is an exact
Debye relaxator on the k = 0 mode: χ_deck(ω) = χ0 / (1 − iωτ) with χ0 = 1/(2α1) = 0.8333 and τ = 1/(2α1L) = 0.8333 deck time. Three tones are measured by lock-in (`ac`) and the
zero-field fluctuation series (`fdt`) gives the spectrum a second way; `replicas = 3` runs the whole file three times (seeds 21, 22, 23) and pools them.

```
python -m fftpf run docs/run_examples/03_dielectric_spectrum.toml
```

Top directory: `spectrum.csv` (pooled, with `_se` columns), `figures/spectrum.png`, `measures.json`, `replicas.json`, and `rep_0/` … `rep_2/`, each a complete run (with its own `series_*.npz`). The lock-in rows of `spectrum.csv`:

```
route,omega,chi_real,chi_imag,...,eps_real,eps_imag,tan_delta,chi_real_se,chi_imag_se
lockin,0.3,0.8503,0.1959,...,106.03,22.12,0.2110,0.0329,0.0513
lockin,1.2,0.4627,0.4084,...,62.26,46.12,0.7831,0.0621,0.0600
lockin,4.0,0.1657,0.2359,...,28.71,26.64,0.9467,0.0257,0.0637
```

The closed form at ω = 1.2 is χ′ = χ″ = 0.4167, within one standard error of the pooled values; at ω = 0.3 it is χ′ = 0.784, χ″ = 0.196, and the pooled χ′ = 0.850 ± 0.033 is two standard errors above it (three replicas of an 8 × 8 box are a small sample). The `fdt` route gives 13 resolved points between ω = 0.307 and 3.99,
and `measures.json` reports the static limit `chi_static_deck` = 0.849 ± 0.071 (closed form 0.8333). ε′ = ε_b + χ′/ε0_deck with ε_b = 10 and ε0_deck = 0.0088541878128.

### Example 4 — inline SI material, T0 and E_ext fields, `pf3d` at Nz = 1 with plane stress (`04_inline_si_pf3d_plane_stress.toml`, 3 s)

The construction used for the NBT-6BT-type relaxor work: a material written in SI in the file (`landau_si` with the slope law a1 = slope · (T − T0(r)), `gradient_si`, GPa
stiffnesses, electrostriction), a single 32 × 32 layer of 2 nm cells with a 6 nm slab, elasticity with `plane_stress`, two Gaussian T0 landscapes (σ = 40 K, ξ = 60 nm and
σ = 20 K, ξ = 6 nm) and a Gaussian random E_ext field (16.8 kV/cm rms, ξ = 6 nm, a seed per component). The protocol is `cool` (white-noise start) → `hold` → one `pe_loop` along ⟨111⟩, with phase fractions in scheme `rt` and phase maps. **The coefficients are the reduced test set of `tests/run_entry/test_drive.py`, illustrative only.**

```
python -m fftpf run docs/run_examples/04_inline_si_pf3d_plane_stress.toml
```

`quenched.npz` holds the three field arrays; `provenance.json` lists their seeds; `figures/` holds the two phase maps. `measures.json`:

```
cool  : "fractions_end": { "R": 0.0010, "T": 0.5605, "T15": 0.1797, "other": 0.4209, "low": 0.0176 }
hold  : "fractions_end": { "R": 0.0039, "T": 0.5967, "T15": 0.2207, "other": 0.3906, "low": 0.0088 }
pe_loop: "loop": { "Pmax": 0.14145090584610945, "Pr": 0.14627305294106915, "Ec": null,
                   "Ec_note": "no zero crossing of Pn on one branch of the cycle" }
```

`Ec` is `null` with a note because Pn along ⟨111⟩ never crosses zero on one branch in this short run; `rt` fractions sum to 1 over R, T, other, low (T15 is a subset of T). Nothing here is a statement about NBT-6BT.

### Example 5 — film relax (`05_film_relax.toml`, 9 s)

A 16 × 16 × 9 film of 3.2 nm thickness (0.4 nm in-plane cells) under −0.5 % misfit strain on a 6-layer substrate stack, deterministic `sbdf2` relaxation from a small random field, `stop = { tol = 1e-6, n_max = 6000 }`. The deck `bto_lcc2005_paper` is used because it has an
isotropic gradient, which any Langevin stage on the film needs; this relax does not.

```
python -m fftpf run docs/run_examples/05_film_relax.toml
```

```
"kind": "relax", "steps": 6000, "stop_reason": "n_max", "t": 120.0
"end": { "mean_P": [4.3e-05, 4.8e-05, -0.013775], ... }
```

The relaxation does **not** converge in 6000 steps (`stop_reason` is `n_max`, not `converged`): the out-of-plane polarization is still evolving. Raise `n_max` to get further (about 1.5 ms per step at this size); look at `trace.csv`
(the energy by term every 51 steps; `E_elastic` is the misfit contribution) and `quicklook.png`, the mid-z plane.

## 13 How this manual is kept true

* `tools/run_input_table.py` writes every key table between `<!-- run_input_table:begin NAME -->` markers from the schema; `--check` fails when a table is stale, a schema key has no description, or a table names a key the schema lacks.
* `tests/run_entry/test_docs.py` runs that check; checks that every `docs/run_examples/*.toml` loads, passes the builder precheck and the CLI `check`; checks that every fenced block marked `toml complete` in this file parses and passes `check`, that every fenced `toml` block parses, and runs every block marked `python run`; and checks that every key path a `ConfigError` site of the code names occurs in § 9.
* Marker convention: the info string of a code fence. `toml complete` = a whole input file that must pass `check`; `toml` = a fragment that must be valid TOML; `python run` = a script that must run to the end; any other fence (plain, `text`) is not checked.
