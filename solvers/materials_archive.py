"""Permanent alias of fftpf.mat_registry_archive (the same module object, so shared module state
is shared). New code imports fftpf.mat_registry_archive; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.mat_registry_archive")
