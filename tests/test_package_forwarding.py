"""solvers.* are permanent aliases of fftpf.* (same module object); fftpf never imports solvers."""
import importlib
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (old solvers name, new fftpf name); see docs/MODULE_NAMES.md
NAMES = [
    ("solver", "pf2d_engine"),
    ("energy_protocol", "pf2d_energy_storage_loop"),
    ("susceptibility", "pf2d_dynamic_susceptibility"),
    ("bulk3d", "pf3d_engine"),
    ("bulk3d_spm", "pf3d_elastic_inhomogeneous"),
    ("film3d", "film_engine"),
    ("film3d_langevin", "film_langevin"),
    ("film3d_spm", "film_electrostatic_inhomogeneous"),
    ("film_elastic", "film_elastic"),
    ("film_elastic_fast", "film_elastic_force_numpy"),
    ("film_elastic_numba", "film_elastic_force_numba"),
    ("elastic", "elastic_base"),
    ("grad4", "gradient4"),
    ("kittel_afe", "antiferroelectric_order"),
    ("octa_tilt", "octahedral_tilt_order"),
    ("polycrystal", "polycrystal"),
    ("disorder", "disorder_fields"),
    ("composite", "composite_labels"),
    ("execution", "backend_config"),
    ("cpu_kernels", "kernels_per_term"),
    ("cpu_fused", "kernels_fused"),
    ("operators", "spectral_operators"),
    ("timestep", "timestep_verify"),
    ("adaptive", "timestep_adaptive"),
    ("minimize", "energy_minimize"),
    ("validate", "input_validation"),
    ("drive", "aging_hold_protocol"),
    ("defects", "defect_bias_field"),
    ("crystallinity", "crystallinity_landau"),
    ("quenched_fields", "random_field_mechanisms"),
    ("composition", "composition_field"),
    ("landau0d", "landau_0d_diagnostics"),
    ("loopfit", "afe_loop_landau_fit"),
    ("wall_metrics", "wall_metrics"),
    ("io_utils", "state_io"),
    ("deck_parser", "deck_reader"),
    ("paths", "ferret_paths"),
    ("casekit", "casekit"),
    ("units", "units"),
    ("materials", "mat_registry"),
    ("materials_archive", "mat_registry_archive"),
    ("literature_params", "mat_chen2007_decks"),
    ("pmn_field", "mat_pmn_charge_disorder"),
    ("pzt_field", "mat_pzt_coefficient_field"),
    ("pan2019_field", "mat_pan2019_landau"),
    ("su_landis", "mat_bto_su_landis_landau"),
    ("afd", "octahedral_tilt_order"),
    ("afe", "antiferroelectric_order"),
    ("hog", "gradient4"),
    ("dynsus", "pf2d_dynamic_susceptibility"),
    ("wall_utils", "wall_metrics"),
    ("walls", "wall_metrics"),
]


def _import(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # optional dependency (numba) absent
        if "numba" in str(exc).lower() or getattr(exc, "name", "") == "numba":
            pytest.skip(f"{name} needs numba: {exc}")
        raise


@pytest.mark.parametrize("old,new", NAMES)
def test_old_name_is_the_new_module(old, new):
    assert _import("solvers." + old) is _import("fftpf." + new)


@pytest.mark.parametrize("old,new", NAMES)
def test_from_import_gives_the_same_object(old, new):
    ns = {}
    exec(f"from solvers import {old} as m", ns)
    assert ns["m"] is _import("fftpf." + new)


def test_fftpf_never_imports_solvers():
    pat = re.compile(r"^\s*(from|import)\s+solvers", re.M)
    offenders = [p.name for p in (ROOT / "fftpf").glob("*.py") if pat.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_solvers_files_are_pure_aliases():
    files = [p for p in (ROOT / "solvers").glob("*.py") if p.name != "__init__.py"]
    assert {p.stem for p in files} == {o for o, _ in NAMES}
    assert [p.name for p in files if len(p.read_text(encoding="utf-8").splitlines()) > 12] == []
