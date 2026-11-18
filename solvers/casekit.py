"""Permanent alias of fftpf.casekit (the same module object, so shared module state
is shared). New code imports fftpf.casekit; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.casekit")
