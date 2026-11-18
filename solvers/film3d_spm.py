"""Permanent alias of fftpf.film_electrostatic_inhomogeneous (the same module object, so shared module state
is shared). New code imports fftpf.film_electrostatic_inhomogeneous; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.film_electrostatic_inhomogeneous")
