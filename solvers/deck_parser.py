"""Permanent alias of fftpf.deck_reader (the same module object, so shared module state
is shared). New code imports fftpf.deck_reader; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.deck_reader")
