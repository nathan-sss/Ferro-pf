# params/archive/ — material decks no computation loads

Archived 2026-11 by user decision: the engine's built-in material parameters are of little value, so the
decks that no code path loads were moved here with `git mv` — same file names, contents byte-for-byte
unchanged (the `[meta]` `registry_name` / `reader` lines inside them are the historical record and were
not edited). "No computation loads it" means: no script under `cases/` or `Runs/`, no `solvers/` code
path and no test reads the file. The infrastructure still reads them: `materials.from_toml(path)` works
on any of them, `tools/param_tables.py` keeps documenting them in `docs/PARAMETERS.md` §K (marked
"(存档)"), and the snapshots whose generator still exists stay checked against it by
`tests/test_regress_materials.py` and `tests/test_materials_archive.py`.

| File | Material | Source | Snapshot of | Why archived |
|---|---|---|---|---|
| `bto_BC1984_6th.toml` | BaTiO₃, 6th-order Landau (Bell & Cross lineage) | Rabe/Ahn/Triscone 2007 App A (Chen), Table 1 "Sixth order" | `materials_archive.bto_bc1984_6th(T=298)` — generator archived too (`solvers/materials_archive.py`; rewrite with `materials_archive.generate_archive_toml()`) | no computation loads it |
| `bto_LCC2005_8th.toml` | BaTiO₃, 8th-order Landau | Rabe/Ahn/Triscone 2007 App A, Table 1 "Eight-order"; mechanics from `bto_298K.toml` | `materials.bto_lcc2005_8th(T=298)` — generator live (cases call it), via `literature_params.generate_all_toml` | no computation loads it (they call the generator) |
| `pto_Haun1987.toml` | PbTiO₃ | Rabe/Ahn/Triscone 2007 App A, Table 5 | `materials.pto_haun1987(T=298)` — generator live (`cases/pzt_g_calibration.py`), via `literature_params.generate_all_toml` | no computation loads it (they call the generator) |
| `pzt_Haun_x0.6.toml` | Pb(Zr₀.₄Ti₀.₆)O₃ | Rabe/Ahn/Triscone 2007 App A, p.367 formula set + Table 4 | `materials.pzt_haun_x(x=0.6, T=298)` — generator live, via `literature_params.generate_all_toml` | no computation loads it (they call the generator) |
| `pmn_clean_x0.toml` | PMN (x_PT = 0), clean-limit α₁ only | Viehland, Jang, Cross & Wuttig, PRB 46, 8003 (1992) | `[landau]` of `materials_archive.pmn_clean_x0(T=298)` — generator archived; the live formula is `pmn_field.pmn_clean_alpha1` | no computation loads it |
| `pmn_pt_landau_x0.30.toml` | 0.70PMN–0.30PT (placeholder, every field REQUIRED) | Zhang et al., PRB 96, 054109 (2017) — citation only, coefficients never transcribed | `materials_archive.pmn_pt_landau_placeholder()` / `pmn_pt_placeholder` — archived | no computation loads it |
| `pmn_disorder_R3.toml` | PMN chemical-disorder derivation record (not a materials deck) | WP15 (R3) derivation, `generated_by = pmn_field.py` | none (a record of `pmn_field.py` inputs/outputs; `cases/w0_relaxor_demo.py` cites one value from it in a comment) | no computation loads it |
| `pzt_r3c_g22e_model.toml` | PZT Zr-rich (x_Ti = 0.2) R3c model composite | P sector Rabe App A formula set at 100 K; θ sector borrowed from the STO generator at 50 K | none (hand-composed); only the local-only `gates/` read it by path | no computation loads it |
| `pzt_r3c_haun_x02.toml` | PZT Zr-rich (x_Ti = 0.2) R3c | P sector Rabe App A at 298 K; θ sector Haun et al., Ferroelectrics 99, 55 (1989) | none (hand-composed); only the local-only `gates/` read it by path | no computation loads it |

**Proposed but kept in `params/`** (a computation reads them): `bfo_pan2019.toml`, `bto_pan2019.toml`,
`sto_pan2019.toml` — `cases/b2e_pan2019_3d_case.py` loads their `[elastic]`/`[electrostriction]` at import
(`f"{short_name}_pan2019.toml"`); `bto_LCC2005_paper.toml` — `cases/bto_film_biaxial.py`,
`continue_bto_film.py`, `certify_bto_backend.py` and `check_scaled_workflow.py` hash it into run
provenance, and `continue_bto_film.py` refuses to resume a checkpoint if it is missing or changed.

## Restore a deck

```bash
git mv params/archive/<file>.toml params/<file>.toml
python3 tools/param_tables.py            # it moves back into the active rows of docs/PARAMETERS.md §K
```

If its snapshot writer is `literature_params.generate_all_toml`, drop the `archive/` prefix from that
entry's path there. If its generator was archived as well (`bto_BC1984_6th`, `pmn_clean_x0`,
`pmn_pt_landau_x0.30`), restore the registry entry as `solvers/materials_archive.py`'s module docstring
describes, and update `tests/test_materials_archive.py`.
