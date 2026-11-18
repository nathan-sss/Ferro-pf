"""Permanent alias of fftpf.mat_chen2007_decks (the same module object, so shared module state
is shared). New code imports fftpf.mat_chen2007_decks; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.mat_chen2007_decks")
