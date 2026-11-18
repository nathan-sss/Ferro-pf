"""Permanent alias of fftpf.antiferroelectric_order (the same module object, so shared module state
is shared). New code imports fftpf.antiferroelectric_order; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.antiferroelectric_order")
