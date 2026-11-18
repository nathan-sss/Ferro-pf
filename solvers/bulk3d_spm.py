"""Permanent alias of fftpf.pf3d_elastic_inhomogeneous (the same module object, so shared module state
is shared). New code imports fftpf.pf3d_elastic_inhomogeneous; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.pf3d_elastic_inhomogeneous")
