"""Permanent alias of fftpf.elastic_base (the same module object, so shared module state
is shared). New code imports fftpf.elastic_base; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.elastic_base")
