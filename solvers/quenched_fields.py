"""Permanent alias of fftpf.random_field_mechanisms (the same module object, so shared module state
is shared). New code imports fftpf.random_field_mechanisms; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.random_field_mechanisms")
