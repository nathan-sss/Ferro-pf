"""Permanent alias of fftpf.pf2d_dynamic_susceptibility (the same module object, so shared module state
is shared). New code imports fftpf.pf2d_dynamic_susceptibility; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.pf2d_dynamic_susceptibility")
