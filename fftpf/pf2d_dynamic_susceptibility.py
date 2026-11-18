"""Permanent alias of fftpf.dielectric_spectrum (the same module object, so shared module state
is shared). New code imports fftpf.dielectric_spectrum; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.dielectric_spectrum")
