"""Permanent alias of fftpf.input_validation (the same module object, so shared module state
is shared). New code imports fftpf.input_validation; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.input_validation")
