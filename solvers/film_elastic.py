"""Permanent alias of fftpf.film_elastic (the same module object, so shared module state
is shared). New code imports fftpf.film_elastic; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.film_elastic")
