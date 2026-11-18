"""solvers -- permanent backward-compatibility package. Every module here is an
alias of the same-named engine module in fftpf (same module object, see
docs/MODULE_NAMES.md); new code imports fftpf.*.
"""
import importlib

from fftpf import ENGINE_NAME, ENGINE_VERSION, __version__  # noqa: F401

_ALIASES = {
    "adaptive": "timestep_adaptive",
    "afd": "octahedral_tilt_order",
    "afe": "antiferroelectric_order",
    "bulk3d": "pf3d_engine",
    "bulk3d_spm": "pf3d_elastic_inhomogeneous",
    "casekit": "casekit",
    "composite": "composite_labels",
    "composition": "composition_field",
    "cpu_fused": "kernels_fused",
    "cpu_kernels": "kernels_per_term",
    "crystallinity": "crystallinity_landau",
    "deck_parser": "deck_reader",
    "defects": "defect_bias_field",
    "disorder": "disorder_fields",
    "drive": "aging_hold_protocol",
    "dynsus": "pf2d_dynamic_susceptibility",
    "elastic": "elastic_base",
    "energy_protocol": "pf2d_energy_storage_loop",
    "execution": "backend_config",
    "film3d": "film_engine",
    "film3d_langevin": "film_langevin",
    "film3d_spm": "film_electrostatic_inhomogeneous",
    "film_elastic": "film_elastic",
    "film_elastic_fast": "film_elastic_force_numpy",
    "film_elastic_numba": "film_elastic_force_numba",
    "grad4": "gradient4",
    "hog": "gradient4",
    "io_utils": "state_io",
    "kittel_afe": "antiferroelectric_order",
    "landau0d": "landau_0d_diagnostics",
    "literature_params": "mat_chen2007_decks",
    "loopfit": "afe_loop_landau_fit",
    "materials": "mat_registry",
    "materials_archive": "mat_registry_archive",
    "minimize": "energy_minimize",
    "octa_tilt": "octahedral_tilt_order",
    "operators": "spectral_operators",
    "pan2019_field": "mat_pan2019_landau",
    "paths": "ferret_paths",
    "pmn_field": "mat_pmn_charge_disorder",
    "polycrystal": "polycrystal",
    "pzt_field": "mat_pzt_coefficient_field",
    "quenched_fields": "random_field_mechanisms",
    "solver": "pf2d_engine",
    "su_landis": "mat_bto_su_landis_landau",
    "susceptibility": "pf2d_dynamic_susceptibility",
    "timestep": "timestep_verify",
    "units": "units",
    "validate": "input_validation",
    "wall_metrics": "wall_metrics",
    "wall_utils": "wall_metrics",
    "walls": "wall_metrics",
}
__all__ = sorted(_ALIASES)


def __getattr__(name):
    """PEP 562: resolve an old module name to its (aliased) module on first access."""
    if name in _ALIASES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_ALIASES))
