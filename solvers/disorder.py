"""Permanent alias of fftpf.disorder_fields (the same module object, so shared module state
is shared). New code imports fftpf.disorder_fields; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.disorder_fields")
