"""Permanent alias of fftpf.afe_loop_landau_fit (the same module object, so shared module state
is shared). New code imports fftpf.afe_loop_landau_fit; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.afe_loop_landau_fit")
