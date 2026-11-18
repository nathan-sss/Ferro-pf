#!/usr/bin/env python3
"""SPEC S41.A A4.4 / S41.1 A4 (WP29) -- generate docs/API_INDEX.md: every
public function of fftpf/, with its signature and its first docstring line.

    python3 tools/api_index.py            # rewrite docs/API_INDEX.md
    python3 tools/api_index.py --check    # exit 1 if the file is out of date

WHY IT IS GENERATED. ENGINE_GUIDE S2.1's per-module signature lists are ~366
hand-written lines -- the one part of the guide that drifts silently, because
nothing fails when a signature changes under it. This file cannot drift: it is
read from the source by AST at generation time, so the fix for a wrong entry is
always to fix the docstring in fftpf/ and rerun this script.

METHOD. ast.parse over a fresh os.listdir of fftpf/*.py (never a hand-copied
module list -- the discipline gates/g29f_dependency_direction.py uses for the
same reason). Public = a module-level def whose name does not start with "_".
Nothing is imported, so this runs with no numpy and cannot execute engine code.

fftpf/ holds only real modules. The old solvers/ package is a set of permanent
aliases of them (docs/MODULE_NAMES.md) and is not indexed here.

fftpf/__init__.py is not a canonical MODULE (it is the package header, and its
__all__ is the canonical module list itself) so it gets no module section -- but
since SPEC S V2-01.1(a) it does define public PACKAGE-level constants
(ENGINE_NAME/ENGINE_VERSION), and S V2-01.1(c) requires them to show up here.
They get one line under the intro, read by the same AST pass as every other
constant.
"""
import argparse
import ast
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVERS_DIR = os.path.join(_ROOT, "fftpf")
OUT_PATH = os.path.join(_ROOT, "docs", "API_INDEX.md")


def first_line(doc):
    """The docstring's first line; " ..." marks one cut by the ~72-col wrap.

    Measured over the 292 docstrings at WP29-D1: 179 first lines stop
    mid-clause because the docstring wraps, so an unmarked cell would read as
    a finished sentence that is not one. Only 6 are under 45 characters, so the
    first line does identify the function -- it is the marker that was missing,
    not the text. The line is never re-flowed: joining wrapped lines splices
    identifiers that wrap across a break (`elastic.` + `build_Omega_3d`).
    """
    text = doc.strip().splitlines()[0].strip()
    return text if text.endswith((".", "!", "?", ":")) else text + " ..."


def public_names(tree):
    """(functions, constants) at module level: [(name, signature, desc)], [name]."""
    funcs, consts = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            doc = ast.get_docstring(node)
            funcs.append((node.name, ast.unparse(node.args),
                          first_line(doc) if doc else ""))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    consts.append(t.id)
    return funcs, consts


def cell(text):
    """Escape a value for one Markdown table cell."""
    return text.replace("|", "\\|")


def build():
    """The whole document, as one string."""
    names = sorted(f for f in os.listdir(SOLVERS_DIR)
                   if f.endswith(".py") and not f.startswith("_"))
    canonical = []
    for fname in names:
        with open(os.path.join(SOLVERS_DIR, fname), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        canonical.append((fname, tree))

    out = [
        "# `fftpf/` API index",
        "",
        "GENERATED FILE -- do not hand-edit. Regenerate after any signature or",
        "docstring change:",
        "",
        "```bash",
        "python3 tools/api_index.py",
        "```",
        "",
        "Every public (non-`_`) module-level function of `fftpf/`, in source order",
        "so this doubles as a table of contents for each file. The description is the",
        "function's own first docstring line, read by AST -- a trailing ` ...` means",
        "that line was cut by the source's own ~72-column wrap, so open the function",
        "for the rest. If a description reads wrong, fix the docstring in `fftpf/`",
        "and rerun, never this file. Prose, decks, traps and worked examples stay in",
        "`docs/ENGINE_GUIDE.md`, and what each script-facing parameter MEANS (physics,",
        "units, literature range) in `docs/PARAMETERS.md`; this is the lookup table.",
        "The old `solvers.*` names are permanent aliases of these modules: see",
        "`docs/MODULE_NAMES.md` for the old -> new table.",
        "",
    ]

    with open(os.path.join(SOLVERS_DIR, "__init__.py"), encoding="utf-8") as fh:
        pkg_consts = public_names(ast.parse(fh.read()))[1]

    n_f = sum(len(public_names(t)[0]) for _, t in canonical)
    n_c = sum(len(public_names(t)[1]) for _, t in canonical)
    out += [
        f"{len(canonical)} canonical modules -- {n_f} public functions, "
        f"{n_c} public constants.",
        "",
        "*package-level constants* (`fftpf/__init__.py`, SPEC §V2-01.1(a)): "
        + ", ".join(f"`{c}`" for c in pkg_consts) + ".",
        "",
        "## Canonical modules",
        "",
    ]

    for fname, tree in canonical:
        funcs, consts = public_names(tree)
        out += [f"### `{fname}`", ""]
        if funcs:
            out += ["| public name | signature | what it does |", "|---|---|---|"]
            for name, sig, first in funcs:
                out.append(f"| `{cell(name)}` | `({cell(sig)})` | {cell(first)} |")
        else:
            out.append("No public function.")
        if consts:
            out += ["", "*public constants:* "
                    + ", ".join(f"`{c}`" for c in consts)]
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if docs/API_INDEX.md is stale")
    args = ap.parse_args()

    text = build()
    if args.check:
        try:
            with open(OUT_PATH, encoding="utf-8") as fh:
                current = fh.read()
        except FileNotFoundError:
            current = None
        if current == text:
            print(f"api_index: up to date ({OUT_PATH})")
            return 0
        print(f"api_index: STALE -- rerun tools/api_index.py ({OUT_PATH})")
        return 1
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"api_index: wrote {OUT_PATH} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
