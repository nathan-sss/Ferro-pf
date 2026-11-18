"""Permanent alias of fftpf.backend_config (the same module object, so shared module state
is shared). New code imports fftpf.backend_config; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.backend_config")
