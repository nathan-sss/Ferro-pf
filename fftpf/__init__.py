"""fftpf -- the FFT phase-field engine (FFT-PF-V2): 2D periodic, 3D periodic and
3D film geometries, their elasticity/electrostatics/noise layers and material
parameter sets. Submodules import lazily (PEP 562), so `import fftpf` loads no
engine. Module names and the old solvers.* aliases: docs/MODULE_NAMES.md.
"""
import importlib

# Plain module constants: attribute lookup finds them without importing an engine.
ENGINE_NAME = "FFT-ferro"
ENGINE_VERSION = "1.0"
__version__ = ENGINE_VERSION

__all__ = [
    "afe_loop_landau_fit",
    "antiferroelectric_order",
    "backend_config",
    "casekit",
    "composite_labels",
    "composition_field",
    "deck_reader",
    "dielectric_spectrum",
    "disorder_fields",
    "elastic_base",
    "ferret_paths",
    "film_elastic",
    "film_electrostatic_inhomogeneous",   # SPEC V2-03 B3: kappa(x,y,z) electrostatic SPM over film3d's banded solve
    "film_engine",
    "film_langevin",   # SPEC V2-03 B1: exact per-mode OU/Langevin stepper for the film geometry
    "gradient4",
    "landau_0d_diagnostics",
    "mat_chen2007_decks",
    "mat_pan2019_landau",
    "mat_pmn_charge_disorder",
    "mat_pzt_coefficient_field",
    "mat_registry",
    "octahedral_tilt_order",
    "pf2d_dynamic_susceptibility",
    "pf2d_energy_storage_loop",
    "pf2d_engine",
    "pf3d_elastic_inhomogeneous",   # SPEC V2-03 C1: full-tensor inhomogeneous C(r) elastic SPM for bulk3d
    "pf3d_engine",
    "random_field_mechanisms",
    "spectral_operators",   # shared k-space symbols / Nyquist rule / E_ext normalisers (docs/OPERATORS.md)
    "state_io",
    "timestep_verify",
    "units",
    "wall_metrics",
]


def __getattr__(name):
    """PEP 562: import a submodule on first access and cache it as an attribute."""
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
