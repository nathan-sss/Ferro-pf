"""Permanent alias of fftpf.kernels_per_term (the same module object, so shared module state
is shared). New code imports fftpf.kernels_per_term; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.kernels_per_term")
