"""MOOSE `.i` deck parameter extractor + the runtime TOML deck reader.

TWO HALVES. (1) The `.i` regex extractor described below. (2) The TOML half:
TOML_SECTIONS (this file) is the single source of truth for every deck key
set -- P, q and theta -- and load_params_toml is the reader every runtime
deck path goes through, materials.from_toml included.
GATES: g6a_converter_roundtrip and g16a-e reach the TOML half as
`from deck_parser import load_params_toml` (the cases/-side shim); the key
sets themselves are checked by g29b_deck_schema_dualpath through
materials.from_toml.

Per SPEC.md: "Per-case parameters MUST be extracted from the referenced deck
file at case-authoring time. Never copy constants across cases, never use
values from this spec's examples without checking the deck." Rather than
hand-transcribing numbers (error-prone), this module regex-parses the deck
text directly so the extracted values are traceable to the actual file.

This is intentionally NOT a general MOOSE input parser -- just enough
structure (GenericConstantMaterial prop blocks, GeneratedMeshGenerator
geometry, Variables block) to pull out what solver.py's cases need.
"""
import os
import re


# --- (a) MOOSE .i extractor --- (the docstring's half (1), ending at
# parse_deck. parse_deck is its only externally-called name: cases/t180.py:46
# and cases/r109.py:38; the other five are its own internal steps.)


def _read(path):
    with open(path) as f:
        return f.read()


def parse_materials(text):
    """Return {prop_name: float value} merged over all GenericConstantMaterial blocks."""
    params = {}
    # Each MOOSE sub-block looks like:  [./name] ... type = GenericConstantMaterial ... [../]
    for block in re.findall(r"\[\./\w+\](.*?)\[\.\./\]", text, re.S):
        if "GenericConstantMaterial" not in block:
            continue
        names = re.search(r"prop_names\s*=\s*'([^']*)'", block)
        values = re.search(r"prop_values\s*=\s*'([^']*)'", block)
        if names and values:
            ns = names.group(1).split()
            vs = [float(v) for v in values.group(1).split()]
            if len(ns) != len(vs):
                raise ValueError(f"prop_names/prop_values length mismatch in block: {block[:80]!r}")
            params.update(dict(zip(ns, vs)))
    return params


def parse_mesh(text):
    """Return dict with nx,ny,xmin,xmax,ymin,ymax from GeneratedMeshGenerator."""
    m = re.search(r"type\s*=\s*GeneratedMeshGenerator(.*?)\[", text, re.S)
    block = m.group(1) if m else text
    out = {}
    for key in ("nx", "ny", "nz", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
        mm = re.search(rf"\b{key}\s*=\s*(-?[\d.eE+-]+)", block)
        if mm:
            out[key] = float(mm.group(1)) if "." in mm.group(1) or "e" in mm.group(1).lower() else int(mm.group(1))
    return out


def polar_components_present(text):
    """Which of polar_x/polar_y/polar_z are declared as [Variables] blocks."""
    var_section = re.search(r"\[Variables\](.*?)\n\[\]", text, re.S)
    block = var_section.group(1) if var_section else text
    return {c: bool(re.search(rf"\[\./{c}\]", block)) for c in ("polar_x", "polar_y", "polar_z")}


def parse_functions(text):
    """Return {function_name: expression_string} for [Functions] ParsedFunction blocks."""
    out = {}
    for name, body in re.findall(r"\[\./(\w+)\](.*?)\[\.\./\]", text, re.S):
        if "ParsedFunction" not in body:
            continue
        m = re.search(r"expression\s*=\s*'([^']*)'", body)
        if m:
            out[name] = m.group(1)
    return out


def parse_polar_ic(text):
    """For each polar_x/y/z [Variables] block, return ('constant', value) or
    ('function', func_name) from its [./InitialCondition] sub-block."""
    var_section = re.search(r"\[Variables\](.*?)\n\[\]", text, re.S)
    block = var_section.group(1) if var_section else text
    out = {}
    for comp in ("polar_x", "polar_y", "polar_z"):
        m = re.search(rf"\[\./{comp}\](.*?)\n  \[\.\./\]", block, re.S)
        if not m:
            continue
        sub = m.group(1)
        cm = re.search(r"type\s*=\s*ConstantIC.*?value\s*=\s*(-?[\d.eE+-]+)", sub, re.S)
        fm = re.search(r"type\s*=\s*FunctionIC.*?function\s*=\s*(\w+)", sub, re.S)
        if cm:
            out[comp] = ("constant", float(cm.group(1)))
        elif fm:
            out[comp] = ("function", fm.group(1))
    return out


def parse_deck(path):
    """MOOSE .i deck -> dict(path, materials, mesh, polar_present, functions, polar_ic); half (a)'s only external entry point."""
    text = _read(path)
    return dict(
        path=path,
        materials=parse_materials(text),
        mesh=parse_mesh(text),
        polar_present=polar_components_present(text),
        functions=parse_functions(text),
        polar_ic=parse_polar_ic(text),
    )


# --- (b) TOML deck I/O --- (the docstring's half (2), from here to the end.
# The halves meet at exactly ONE call: write_params_toml -- the offline
# generator -- uses half (a)'s _read + parse_materials to build a .toml. The
# RUNTIME path never crosses: load_params_toml, what the other 17 callers and
# g6a/g16a-e reach, reads the .toml and touches nothing above this line.)
# --------------------------------------------------------------------------
# SPEC §6.1: TOML params/ generator mode. stdlib has no TOML writer (tomllib
# is read-only), so this hand-formats the flat sections SPEC's schema needs --
# straightforward since every value here is a plain float.
# --------------------------------------------------------------------------
TOML_SECTIONS = {
    "landau": ("alpha1", "alpha11", "alpha12", "alpha111", "alpha112", "alpha123",
               "alpha1111", "alpha1112", "alpha1122", "alpha1123"),
    "gradient": ("G110", "G11_G110", "G12_G110", "G44_G110", "G44P_G110"),
    "electrostatics": ("permittivity",),
    "elastic": ("C11", "C12", "C44"),
    "electrostriction": ("Q11_phys", "Q12_phys", "Q44_phys"),
    # SPEC S23 (WP17/S1, AFE order parameter q) additions -- new, disjoint
    # key names (aq1... not alpha1...) so flattening into one `materials`
    # dict below can never silently overwrite the P-sector's own "landau"
    # section keys.
    "landau_q": ("aq1", "aq11", "aq12", "aq111", "aq112", "aq123"),
    "coupling_pq": ("t1", "t2", "u1", "u2"),
    "gradient_q": ("Gq",),
    "eigenstrain_q": ("Lam11", "Lam12", "Lam44"),
    # SPEC S28 (WP18/S2, octahedral tilt theta) additions -- new, disjoint key
    # names again (b1... not alpha1..., t11/t12/t44 not t1/t2/u1/u2, "_theta"
    # suffix on the eigenstrain names since q's own eigenstrain_q section
    # already claims bare Lam11/Lam12/Lam44 -- disjoint even though no current
    # single deck combines q and theta, per SPEC S28's own "future lane" note).
    "landau_theta": ("b1", "b11", "b12", "b111", "b112", "b123"),
    "coupling_Ptheta": ("t11", "t12", "t44"),
    "gradient_theta": ("Gtheta",),
    "eigenstrain_theta": ("Lam11_theta", "Lam12_theta", "Lam44_theta"),
}

# Which TOML_SECTIONS a deck may leave out entirely. SPEC §6.1 fixes the five
# P sections (landau/gradient/electrostatics/elastic/electrostriction) as the
# schema every deck has. The q sectors (SPEC S23) and theta sectors (SPEC S28)
# were added later for the AFE and tilt decks only: bto_298K/bto_118K and
# every literature_params set have none of them. A writer skips an optional
# section that has NONE of its keys (no header, no placeholder lines). If ANY
# key of it is present the section is written and treated like a P section.
# The reader (load_params_toml) already treats every section as optional.
_OPTIONAL_TOML_SECTIONS = ("landau_q", "coupling_pq", "gradient_q", "eigenstrain_q",
                          "landau_theta", "coupling_Ptheta", "gradient_theta", "eigenstrain_theta")


def _section_is_absent(section, params):
    """True if `section` is optional (_OPTIONAL_TOML_SECTIONS) and `params`
    holds none of its keys (a key whose value is None counts as absent), so
    a TOML writer should skip the whole section."""
    return (section in _OPTIONAL_TOML_SECTIONS
            and all(params.get(k) is None for k in TOML_SECTIONS[section]))


def materials_to_toml(materials, source_deck_rel, extracted_date):
    """Format a materials dict (as returned by parse_materials) as TOML text
    per SPEC §6.1's schema ([meta] + landau/gradient/electrostatics/elastic/
    electrostriction sections). The optional q/theta sections
    (_OPTIONAL_TOML_SECTIONS) are written only if the deck has any of their
    keys. Every key of a written section is required."""
    lines = ["[meta]", f'source_deck = "{source_deck_rel}"', f'extracted = "{extracted_date}"', ""]
    missing = []
    for section, keys in TOML_SECTIONS.items():
        if _section_is_absent(section, materials):
            continue
        lines.append(f"[{section}]")
        for k in keys:
            if k in materials:
                lines.append(f"{k} = {materials[k]!r}")
            else:
                missing.append(k)
        lines.append("")
    if missing:
        raise ValueError(f"deck is missing expected material keys: {missing}")
    return "\n".join(lines).rstrip() + "\n"


def write_params_toml(deck_path, out_path, ferret_root, extracted_date):
    """Extract materials from deck_path and write a SPEC §6.1 TOML file."""
    text = _read(deck_path)
    materials = parse_materials(text)
    rel = os.path.relpath(deck_path, ferret_root)
    toml_text = materials_to_toml(materials, rel, extracted_date)
    with open(out_path, "w") as f:
        f.write(toml_text)
    return toml_text


def load_params_toml(path):
    """Read a SPEC §6.1 TOML params file back into {'materials': {...flat...}, 'meta': {...}}."""
    import tomllib
    with open(path, "rb") as f:
        data = tomllib.load(f)
    materials = {}
    for section in TOML_SECTIONS:
        materials.update(data.get(section, {}))
    return dict(materials=materials, meta=data.get("meta", {}))


def generate_default_params(ferret_root, out_dir, extracted_date):
    """Generator mode entry point (SPEC §6.1): write params/bto_298K.toml and
    params/bto_118K.toml from the named source decks."""
    specs = [
        ("bto_298K.toml", os.path.join(ferret_root, "runs", "rhombo_walls", "inputs", "k_t180.i")),
        ("bto_118K.toml", os.path.join(ferret_root, "runs", "rhombo_walls", "inputs", "k_r109_L64.i")),
    ]
    for out_name, deck_path in specs:
        out_path = os.path.join(out_dir, out_name)
        write_params_toml(deck_path, out_path, ferret_root, extracted_date)
        print(f"wrote {out_path} (from {deck_path})")


if __name__ == "__main__":
    import datetime
    import sys
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from fftpf import ferret_paths as paths
    generate_default_params(
        ferret_root=paths.FERRET_ROOT,
        out_dir=os.path.join(_ROOT, "params"),
        extracted_date=datetime.date.today().isoformat(),
    )
