"""Permanent alias of fftpf.mat_bto_su_landis_landau (the same module object, so shared module state
is shared). New code imports fftpf.mat_bto_su_landis_landau; see docs/MODULE_NAMES.md."""
import importlib, sys
sys.modules[__name__] = importlib.import_module("fftpf.mat_bto_su_landis_landau")
