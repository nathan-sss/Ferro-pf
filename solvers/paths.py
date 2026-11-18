"""Permanent alias of fftpf.ferret_paths (the same module object, so shared module state
is shared). New code imports fftpf.ferret_paths; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.ferret_paths")
