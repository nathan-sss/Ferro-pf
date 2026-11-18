"""Permanent alias of fftpf.units (the same module object, so shared module state
is shared). New code imports fftpf.units; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.units")
