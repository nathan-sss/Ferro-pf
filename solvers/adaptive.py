"""Permanent alias of fftpf.timestep_adaptive (the same module object, so shared module state
is shared). New code imports fftpf.timestep_adaptive; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.timestep_adaptive")
