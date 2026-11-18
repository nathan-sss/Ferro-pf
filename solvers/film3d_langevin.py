"""Permanent alias of fftpf.film_langevin (the same module object, so shared module state
is shared). New code imports fftpf.film_langevin; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.film_langevin")
