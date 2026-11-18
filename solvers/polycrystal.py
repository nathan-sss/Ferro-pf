"""Permanent alias of fftpf.polycrystal (the same module object, so shared module state
is shared). New code imports fftpf.polycrystal; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.polycrystal")
