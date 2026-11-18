"""Locations of bundled materials and user-created runs."""
from pathlib import Path

_package = Path(__file__).resolve().parent
# The wheel places the source distribution's params/ inside the fftpf package.
PARAMS_DIR = _package / "_decks"
if not PARAMS_DIR.is_dir():
    PARAMS_DIR = _package.parent / "params"


def default_run_root():
    return str(Path.cwd() / "reports" / "runs")
