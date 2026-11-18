"""Permanent alias of fftpf.energy_minimize (the same module object, so shared module state
is shared). New code imports fftpf.energy_minimize; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.energy_minimize")
