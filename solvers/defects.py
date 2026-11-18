"""Permanent alias of fftpf.defect_bias_field (the same module object, so shared module state
is shared). New code imports fftpf.defect_bias_field; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.defect_bias_field")
