"""Permanent alias of fftpf.state_io (the same module object, so shared module state
is shared). New code imports fftpf.state_io; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.state_io")
