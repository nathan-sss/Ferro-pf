"""Permanent alias of fftpf.octahedral_tilt_order (the same module object, so shared module state
is shared). New code imports fftpf.octahedral_tilt_order; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.octahedral_tilt_order")
