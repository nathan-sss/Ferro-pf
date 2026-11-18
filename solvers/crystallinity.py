"""Permanent alias of fftpf.crystallinity_landau (the same module object, so shared module state
is shared). New code imports fftpf.crystallinity_landau; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.crystallinity_landau")
