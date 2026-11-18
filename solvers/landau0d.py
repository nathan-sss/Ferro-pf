"""Permanent alias of fftpf.landau_0d_diagnostics (the same module object, so shared module state
is shared). New code imports fftpf.landau_0d_diagnostics; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.landau_0d_diagnostics")
