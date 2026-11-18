"""Permanent alias of fftpf.timestep_verify (the same module object, so shared module state
is shared). New code imports fftpf.timestep_verify; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.timestep_verify")
