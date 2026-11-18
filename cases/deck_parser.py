"""Permanent forwarding shim: deck_parser.py moved to solvers/deck_parser.py
in WP24/SPEC S35, 2026-08-29 (library code promoted out of cases/ into
solvers/, SPEC S35.1). This file exists ONLY so every existing call site
(`from deck_parser import load_params_toml`, `import deck_parser`, plus
solvers/literature_params.py's/solvers/afd.py's/solvers/afe.py's own bare
`from deck_parser import ...`) keeps resolving forever -- gates/ and cases/
imports were deliberately left UNCHANGED by this refactor, this shim covers
them all. NEW code must import solvers.deck_parser directly (or
`from solvers import deck_parser`); do not add new call sites against this
cases/ name. See docs/ENGINE_GUIDE.md S2.1a for the full old->new mapping
table and the shim policy explanation.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from solvers.deck_parser import *  # noqa: F401,F403 -- re-exports every non-underscore module-level name

# `import *` skips underscore-prefixed names (true regardless of whether the
# source module defines __all__). solvers.deck_parser has at least one
# private module-level helper (_read); forward every remaining non-dunder
# name generically (rather than hand-enumerate, which the solvers/afd.py
# shim's first pass proved error-prone) so this shim's namespace is an
# exact superset of solvers.deck_parser's.
from solvers import deck_parser as _src_mod
for _attr_name in dir(_src_mod):
    if not (_attr_name.startswith("__") and _attr_name.endswith("__")) and _attr_name not in globals():
        globals()[_attr_name] = getattr(_src_mod, _attr_name)
del _attr_name, _src_mod
