"""Permanent alias of fftpf.spectral_operators (the same module object, so shared module state
is shared). New code imports fftpf.spectral_operators; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.spectral_operators")
