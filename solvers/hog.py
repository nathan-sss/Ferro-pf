"""Permanent alias of fftpf.gradient4 (the same module object, so shared module state
is shared). New code imports fftpf.gradient4; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.gradient4")
