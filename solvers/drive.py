"""Permanent alias of fftpf.aging_hold_protocol (the same module object, so shared module state
is shared). New code imports fftpf.aging_hold_protocol; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.aging_hold_protocol")
