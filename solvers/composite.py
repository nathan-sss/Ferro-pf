"""Permanent alias of fftpf.composite_labels (the same module object, so shared module state
is shared). New code imports fftpf.composite_labels; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.composite_labels")
