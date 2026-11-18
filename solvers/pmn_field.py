"""Permanent alias of fftpf.mat_pmn_charge_disorder (the same module object, so shared module state
is shared). New code imports fftpf.mat_pmn_charge_disorder; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.mat_pmn_charge_disorder")
