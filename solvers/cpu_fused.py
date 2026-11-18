"""Permanent alias of fftpf.kernels_fused (the same module object, so shared module state
is shared). New code imports fftpf.kernels_fused; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.kernels_fused")
