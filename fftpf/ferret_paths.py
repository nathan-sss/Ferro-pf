"""Location of the optional external Ferret source decks, used by the deck generator."""
import os
from pathlib import Path

FERRET_ROOT = os.environ.get("FERRET_ROOT", str(Path.cwd() / "Ferret"))
