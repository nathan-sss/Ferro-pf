"""Setup-time input validation for the engine entry points.

WHY. Parameters travel through this engine as plain dicts and arrays, so the
commonest bugs are silent rather than loud: a misspelled Landau key such as
`a1121` or a deck-style `alpha1` is simply never read (bulk_force reads the
six required keys and defaults the four 8th-order ones to 0.0), a TOML
`[landau]` typo is dropped by the reader, and a field with one axis too few
broadcasts against the wrong axes. This module turns those cases into a
ValidationError (a ValueError) with a message that names the entry point, the
offending input and the rule.

CONTRACT.
  * Checks run only at SETUP entry points (grid/operator/elastic/deck builders,
    casekit's builders and the start of casekit.anneal), never inside a
    per-step function. Each entry point calls one function below on the
    first line of its body (materials.get/from_toml also pass the deck they
    return through validate.deck), so the checking logic lives here and
    nowhere else.
  * Checks only read. Nothing is converted, copied back, or mutated, so a
    valid input produces bit-for-bit the same output as before.
  * Cost: scalar checks plus at most one np.isfinite pass over each array
    argument, once per setup call.
  * Leaf module: imports only the standard library and NumPy (deck_parser
    lazily, for TOML decks), so every engine module can import it without a
    cycle.

What is checked where is tabulated in docs/VALIDATION.md, together with how to
add a check.
"""
from collections.abc import Mapping
import math
import os
import re

import numpy as np

# The Landau keys solver.bulk_force / bulk_energy_density read
# (solver.ALPHA_KEYS + solver.ALPHA_KEYS_8TH). Duplicated rather than imported
# so this module stays a leaf; tests/test_validate.py pins the two together.
LANDAU_REQUIRED = ("a1", "a11", "a12", "a111", "a112", "a123")
LANDAU_OPTIONAL = ("a1111", "a1112", "a1122", "a1123")
LANDAU_KEYS = LANDAU_REQUIRED + LANDAU_OPTIONAL

# A key that "looks like" a Landau coefficient: P-sector (a1, alpha1, a_11),
# AFE q-sector (aq1) or tilt sector (b1, beta1) names, any case. Such a key
# must be one of LANDAU_KEYS, because the engine would ignore it silently.
# Keys starting with "_" are private payload (polycrystal's "_cos"/"_sin")
# and are never checked; other non-matching keys are left alone too.
_LANDAU_LIKE = re.compile(r"^(?:a|alpha|aq|b|beta)_?\d+$", re.IGNORECASE)

# materials.get / from_toml deck schema (materials.py R1). A sector the source
# does not provide is ABSENT; a present sector must be complete.
DECK_SECTORS = {
    "gradient": ("G11", "G12", "G44", "G44p"),
    "elastic": ("C11", "C12", "C44"),
    "electrostriction": ("Q11_phys", "Q12_phys", "Q44_phys"),
}
DECK_KEYS = ("landau", "gradient", "elastic", "electrostriction", "kappa_b", "meta")

# Relative tolerance for L == N*h. make_grid computes h = L/N, so its own grids
# agree to an ulp; the slack only admits grids rebuilt from saved scalars.
_GRID_RTOL = 1e-9

_REAL_SCALARS = (int, float, np.integer, np.floating)    # bool is excluded before use


class ValidationError(ValueError):
    """An entry point received an input the engine would mis-handle."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _fmt(value):
    """Short description of a value for an error message."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return repr(arr.item())
    if arr.dtype.kind in "iuf" and arr.size and np.all(np.isfinite(arr)):
        return f"array{arr.shape} in [{arr.min():.6g}, {arr.max():.6g}]"
    return f"array{arr.shape} dtype={arr.dtype}"


def _real(value, name, where, *, scalar=False, positive=False):
    """np.asarray(value) after checking it is real (int/float) and finite.

    The result is only read: np.asarray returns an ndarray argument itself,
    and nothing here writes to it, so the caller's object is untouched."""
    if value is None or isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValidationError(f"{where}: {name} must be a real number"
                              f"{'' if scalar else ' or array'}, got {value!r}")
    if isinstance(value, _REAL_SCALARS):            # fast path: plain numbers
        if not math.isfinite(value):
            raise ValidationError(f"{where}: {name} is not finite, got {value!r}")
        if positive and not value > 0:
            raise ValidationError(f"{where}: {name} must be > 0, got {value!r}")
        return np.float64(value)
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{where}: {name} is not numeric ({exc})") from None
    if arr.dtype.kind not in "iuf":
        raise ValidationError(f"{where}: {name} must be real (int/float), got dtype "
                              f"{arr.dtype} from {type(value).__name__}")
    if scalar and arr.ndim != 0:
        raise ValidationError(f"{where}: {name} must be a scalar, got shape {arr.shape}")
    finite = np.isfinite(arr)
    if not np.all(finite):
        bad = int(arr.size - np.count_nonzero(finite))
        raise ValidationError(f"{where}: {name} is not finite "
                              f"({bad} of {arr.size} value(s) are nan/inf)")
    if positive and not np.all(arr > 0):
        raise ValidationError(f"{where}: {name} must be > 0, got {_fmt(value)}")
    return arr


def _count(value, name, where, *, minimum=1, allow_integral_float=False):
    """An integer size >= minimum (not bool); optionally an integral float."""
    arr = None if isinstance(value, (bool, np.bool_)) else np.asarray(value)
    ok = arr is not None and arr.ndim == 0 and arr.dtype.kind in "iu"
    if not ok and allow_integral_float and arr is not None and arr.ndim == 0:
        ok = arr.dtype.kind == "f" and bool(np.isfinite(arr)) and float(arr).is_integer()
    if not ok:
        raise ValidationError(f"{where}: {name} must be an integer, got {value!r}")
    if int(arr) < minimum:
        raise ValidationError(f"{where}: {name} must be >= {minimum}, got {value!r}")
    return int(arr)


def _fits(shape, field_shape, *, leading=False):
    """True when an array of `shape` broadcasts onto a field of `field_shape`
    without enlarging it. leading=True also accepts extra leading axes (an
    ensemble axis in front of the spatial ones)."""
    shape, field_shape = tuple(shape), tuple(field_shape)
    if leading and len(shape) > len(field_shape):
        shape = shape[len(shape) - len(field_shape):]
    try:
        return np.broadcast_shapes(shape, field_shape) == field_shape
    except ValueError:
        return False


def _field(value, name, where, field_shape, *, leading=False, positive=False):
    """Finite real scalar or array that broadcasts onto field_shape."""
    arr = _real(value, name, where, positive=positive)
    if arr.ndim and not _fits(arr.shape, field_shape, leading=leading):
        raise ValidationError(f"{where}: {name} has shape {arr.shape}, which does not "
                              f"broadcast onto the grid {tuple(field_shape)} (axes must equal "
                              f"the grid's or be 1)")
    return arr


def _triple(value, name, where):
    """Three components from a 3-sequence (not a string or dict)."""
    if isinstance(value, (str, bytes, Mapping)):
        raise ValidationError(f"{where}: {name} must be a 3-sequence, got {value!r}")
    try:
        items = tuple(value)
    except TypeError:
        raise ValidationError(f"{where}: {name} must be a 3-sequence, got {value!r}") from None
    if len(items) != 3:
        raise ValidationError(f"{where}: {name} must have 3 entries, got {len(items)}")
    return items


def _eps_star_static(value, field_shape, where):
    """(3,3,*field_shape), optionally with an ensemble axis after (3,3)."""
    if value is None:
        return
    arr = _real(value, "eps_star_static", where)
    n = len(field_shape)
    if (arr.ndim < n + 2 or arr.shape[:2] != (3, 3)
            or arr.shape[-n:] != tuple(field_shape) or arr.ndim > n + 3):
        raise ValidationError(f"{where}: eps_star_static must have shape "
                              f"{(3, 3) + tuple(field_shape)} (or (3, 3, S, ...) for an "
                              f"ensemble), got {arr.shape}")


# --------------------------------------------------------------------------
# Building blocks (usable on their own)
# --------------------------------------------------------------------------
def landau_coefficients(alpha, *, field_shape=None, require=True, where="alpha"):
    """Check a Landau coefficient dict in the engine's a1..a1123 naming.

    Rejects any key that looks like a Landau coefficient but is not one the
    engine reads (deck-style `alpha1`, `A11`, `a1121`, ...); with
    require=True, the six 6th-order keys must be present. Values must be
    finite real scalars or arrays; with field_shape given, arrays must
    broadcast onto it without enlarging it, otherwise they must broadcast
    against each other. Keys beginning with "_" are private and skipped."""
    if not isinstance(alpha, Mapping):
        raise ValidationError(f"{where}: expected a dict of Landau coefficients, "
                              f"got {type(alpha).__name__}")
    unknown = [k for k in alpha if isinstance(k, str) and not k.startswith("_")
               and k not in LANDAU_KEYS and _LANDAU_LIKE.match(k)]
    if unknown:
        notes = []
        for key in unknown:
            digits = re.sub(r"\D", "", key)
            plain = key.lower().replace("_", "")
            if "a" + digits in LANDAU_KEYS and plain in ("a" + digits, "alpha" + digits):
                kind = "deck-style name" if plain.startswith("alpha") else "misspelled"
                notes.append(f"{key!r} ({kind}; the engine key is 'a{digits}' -- "
                             f"materials.get()/from_toml() decks are already engine-native)")
            else:
                notes.append(f"{key!r} (not a term of the engine's cubic 6th/8th-order "
                             f"polynomial)")
        raise ValidationError(
            f"{where}: unrecognized Landau key(s) {', '.join(notes)}. The engine reads "
            f"{', '.join(LANDAU_REQUIRED)} (required) and {', '.join(LANDAU_OPTIONAL)} "
            f"(optional, default 0.0); any other coefficient would be silently ignored. "
            f"Private keys starting with '_' are allowed.")
    if require:
        missing = [k for k in LANDAU_REQUIRED if k not in alpha]
        if missing:
            raise ValidationError(f"{where}: missing required Landau key(s) {missing} "
                                  f"(the engine reads all of {list(LANDAU_REQUIRED)})")
    shapes = []
    for key in LANDAU_KEYS:
        if key not in alpha:
            continue
        name = f"alpha[{key!r}]"
        if field_shape is None:
            shapes.append(np.shape(_real(alpha[key], name, where)))
        else:
            _field(alpha[key], name, where, field_shape)
    if field_shape is None and shapes:
        try:
            np.broadcast_shapes(*shapes)
        except ValueError:
            raise ValidationError(f"{where}: Landau coefficient arrays have incompatible "
                                  f"shapes {sorted(set(shapes))}") from None
    return alpha


def grid_geometry(grid, ndim, *, where="grid"):
    """Check a solver.make_grid (ndim=2) or bulk3d.make_grid_3d (ndim=3) dict:
    positive integer sizes, positive finite lengths and spacings, L == N*h on
    every axis, and k-arrays of matching dimensionality. Returns the spatial
    field shape, (Ny,Nx) or (Nz,Ny,Nx)."""
    if not isinstance(grid, Mapping):
        raise ValidationError(f"{where}: grid must be a dict, got {type(grid).__name__}")
    axes = ("x", "y", "z")[:ndim]
    needed = ([f"N{a}" for a in axes] + [f"L{a}" for a in axes] + [f"h{a}" for a in axes]
              + [f"K{a.upper()}r" for a in axes])
    missing = [k for k in needed if k not in grid]
    if missing:
        maker = "solver.make_grid" if ndim == 2 else "bulk3d.make_grid_3d"
        raise ValidationError(f"{where}: grid lacks {missing}; build it with {maker}")
    sizes = {}
    for a in axes:
        n = _count(grid[f"N{a}"], f"grid['N{a}']", where)
        length = float(_real(grid[f"L{a}"], f"grid['L{a}']", where, scalar=True, positive=True))
        h = float(_real(grid[f"h{a}"], f"grid['h{a}']", where, scalar=True, positive=True))
        if abs(n * h - length) > _GRID_RTOL * length:
            raise ValidationError(f"{where}: grid['h{a}']={h!r} is inconsistent with "
                                  f"L{a}/N{a} = {length!r}/{n} (L = N*h must hold)")
        sizes[a] = n
    k_shapes = {np.shape(grid[f"K{a.upper()}r"]) for a in axes}
    if len(k_shapes) != 1 or len(next(iter(k_shapes))) != ndim:
        raise ValidationError(f"{where}: the grid's k-arrays have shapes {sorted(k_shapes)}; "
                              f"a {ndim}D engine needs {ndim}-D arrays of one shape (a grid "
                              f"from the other engine?)")
    return tuple(sizes[a] for a in reversed(axes))


def cubic_stiffness(C11, C12, C44, *, where="C", names=("C11", "C12", "C44")):
    """Finite cubic stiffness that is positive-definite: C11 > |C12|,
    C11 + 2*C12 > 0 and C44 > 0 (elementwise for arrays). `names` labels the
    three values in messages (bulk3d's tensor triple is c1111/c1122/c1212)."""
    c11, c12, c44 = (_real(v, n, where) for v, n in zip((C11, C12, C44), names))
    failed = [rule for ok, rule in ((c11 > np.abs(c12), "C11 > |C12|"),
                                    (c11 + 2 * c12 > 0, "C11 + 2*C12 > 0"),
                                    (c44 > 0, "C44 > 0")) if not np.all(ok)]
    if failed:
        values = ", ".join(f"{n}={_fmt(v)}" for n, v in zip(names, (C11, C12, C44)))
        raise ValidationError(
            f"{where}: cubic stiffness {values} is not positive-definite (violates "
            f"{', '.join(failed)}). A cubic C is stable only if C11 > |C12|, "
            f"C11 + 2*C12 > 0 and C44 > 0; check the units (deck = GPa) and the order "
            f"of the three values.")


def applied_field(E_ext, P_shape, *, where="E_ext"):
    """E_ext must be finite and either a 3-vector (3,) or a field with P's
    number of axes, leading axis 3, every other axis equal to P's or 1. A
    field with fewer axes than P (e.g. (3,Ny,Nx) against an ensemble
    (3,S,Ny,Nx)) would be broadcast against the wrong axes."""
    arr = _real(E_ext, "E_ext", where)
    P_shape = tuple(P_shape)
    if arr.shape == (3,):
        return
    if (arr.ndim != len(P_shape) or arr.shape[0] != 3
            or not all(e in (1, p) for e, p in zip(arr.shape[1:], P_shape[1:]))):
        raise ValidationError(f"{where}: E_ext has shape {arr.shape}; expected a 3-vector (3,) "
                              f"or a field shaped like P {P_shape} (axes of length 1 broadcast)")


def polarization(P, grid_shape, *, where="P"):
    """P must be finite real, shaped (3, *grid_shape) or (3, S, *grid_shape)."""
    arr = _real(P, "P", where)
    grid_shape = tuple(grid_shape)
    n = len(grid_shape)
    if arr.ndim not in (n + 1, n + 2) or arr.shape[0] != 3 or arr.shape[-n:] != grid_shape:
        raise ValidationError(f"{where}: P has shape {arr.shape}; expected {(3,) + grid_shape} "
                              f"or (3, S) + {grid_shape} with an ensemble axis S")


def deck(d, *, where="deck"):
    """Check a materials.get()/from_toml() deck (materials.py R1 schema) and
    return it unchanged: a "meta" dict always; every other sector optional
    but complete when present, with recognized engine-native Landau keys,
    finite values, a positive-definite cubic C and a positive kappa_b."""
    if not isinstance(d, Mapping):
        raise ValidationError(f"{where}: a deck is a dict, got {type(d).__name__}")
    unknown = [k for k in d if k not in DECK_KEYS]
    if unknown:
        raise ValidationError(f"{where}: unknown deck sector(s) {unknown}; the schema has "
                              f"{list(DECK_KEYS)} (extend validate.DECK_KEYS with the schema)")
    if not isinstance(d.get("meta"), Mapping):
        raise ValidationError(f"{where}: deck lacks its required 'meta' provenance dict")
    if "landau" in d:
        if not d["landau"]:
            raise ValidationError(f"{where}: 'landau' is present but empty; omit an "
                                  f"unsourced sector instead")
        landau_coefficients(d["landau"], require=False, where=f"{where} landau")
    deck_sectors(d, [s for s in (*DECK_SECTORS, "kappa_b") if s in d], where=where)
    return d


def deck_sectors(d, sectors, *, where="deck"):
    """The deck has each of `sectors`, complete: exactly the schema's keys
    with finite values, a positive-definite cubic C, a positive kappa_b. For
    consumers that need particular sectors (casekit's builders)."""
    if not isinstance(d, Mapping):
        raise ValidationError(f"{where}: a deck is a dict, got {type(d).__name__}")
    missing = [s for s in sectors if s not in d]
    if missing:
        raise ValidationError(
            f"{where}: deck lacks sector(s) {missing}. A deck omits what its source does "
            f"not provide (materials.py R1); use a deck that has them, or build the "
            f"operators by hand.")
    for sector in sectors:
        keys, body = DECK_SECTORS.get(sector), d[sector]
        if keys is None:
            continue
        if not isinstance(body, Mapping) or set(body) != set(keys):
            got = sorted(body) if isinstance(body, Mapping) else type(body).__name__
            raise ValidationError(f"{where}: sector '{sector}' must have exactly "
                                  f"{list(keys)}, got {got}")
        for key in keys:
            _real(body[key], f"{sector}[{key!r}]", where)
    if "elastic" in sectors:
        e = d["elastic"]
        cubic_stiffness(e["C11"], e["C12"], e["C44"], where=f"{where} elastic")
    if "kappa_b" in sectors:
        _real(d["kappa_b"], "kappa_b", where, positive=True)


def toml_deck(path):
    """Check a params/*.toml deck BEFORE deck_parser reads it, for the cases
    the reader would drop silently: a Landau-like key in [landau] that is not
    one of deck_parser.TOML_SECTIONS["landau"] (e.g. engine-native `a1` or a
    typo `alpha1121`), a section name differing from a known one only in
    case, a partial elastic/electrostriction sector, or G110 without all four
    gradient ratios; and non-finite or non-numeric values of known keys."""
    import tomllib
    from fftpf import deck_reader as deck_parser
    where = f"materials.from_toml({os.fspath(path)!r})"
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    known = deck_parser.TOML_SECTIONS
    for section in data:
        if section not in known and section != "meta" and section.lower() in known:
            raise ValidationError(f"{where}: section [{section}] is never read (section "
                                  f"names are case-sensitive; did you mean "
                                  f"[{section.lower()}]?)")
    landau = data.get("landau", {})
    bad = [k for k in landau if _LANDAU_LIKE.match(k) and k not in known["landau"]]
    if bad:
        raise ValidationError(f"{where}: [landau] key(s) {bad} are not read; TOML decks use "
                              f"the deck-style names {list(known['landau'])} (from_toml "
                              f"converts them to engine-native a1..)")
    for section, keys in known.items():
        body = data.get(section, {})
        for key in keys:
            if key in body:
                _real(body[key], f"[{section}] {key}", where, scalar=True)
    for section, keys in (("elastic", ("C11", "C12", "C44")),
                          ("electrostriction", ("Q11_phys", "Q12_phys", "Q44_phys"))):
        present = [k for k in keys if k in data.get(section, {})]
        if 0 < len(present) < len(keys):
            raise ValidationError(f"{where}: [{section}] has only {present}; a partial "
                                  f"sector is dropped silently -- give all of {list(keys)} "
                                  f"or none")
    gradient = data.get("gradient", {})
    ratios = known["gradient"][1:]
    if "G110" in gradient and not all(k in gradient for k in ratios):
        raise ValidationError(f"{where}: [gradient] has G110 but not all of {list(ratios)}; "
                              f"the gradient sector would be dropped silently")


# --------------------------------------------------------------------------
# Entry-point checks: one per engine setup function, called on the first line
# of that function's body with that function's own arguments.
# --------------------------------------------------------------------------
def _gradient_and_kappa(G11, G12, G44, G44p, kappa, field_shape, where):
    for name, value in (("G11", G11), ("G12", G12), ("G44", G44), ("G44p", G44p)):
        _real(value, name, where, scalar=True)          # sign is free (modulated phases)
    kap = _real(kappa, "kappa", where, positive=True)
    if kap.ndim and kap.shape != tuple(field_shape):
        raise ValidationError(f"{where}: kappa array shape {kap.shape} != grid "
                              f"{tuple(field_shape)}")


def build_operators(grid, G11, G12, G44, G44p, kappa):
    """Setup check for solver.build_operators (2D)."""
    where = "solver.build_operators"
    shape = grid_geometry(grid, 2, where=where)
    _gradient_and_kappa(G11, G12, G44, G44p, kappa, shape, where)


def build_operators_3d(grid, G11, G12, G44, G44p, kappa, kappa0_override=None):
    """Setup check for bulk3d.build_operators_3d."""
    where = "bulk3d.build_operators_3d"
    shape = grid_geometry(grid, 3, where=where)
    _gradient_and_kappa(G11, G12, G44, G44p, kappa, shape, where)
    if kappa0_override is not None:
        _real(kappa0_override, "kappa0_override", where, scalar=True, positive=True)


def setup_elastic(grid, C11, C12, C44, Q11, Q12, Q44, eps_star_static=None, eps_m=None):
    """Setup check for elastic.setup_elastic (2D, homogeneous C)."""
    where = "elastic.setup_elastic"
    shape = grid_geometry(grid, 2, where=where)
    for name, value in (("C11", C11), ("C12", C12), ("C44", C44)):
        _real(value, name, where, scalar=True)
    cubic_stiffness(C11, C12, C44, where=where)
    for name, value in (("Q11", Q11), ("Q12", Q12), ("Q44", Q44)):
        _field(value, name, where, shape, leading=True)
    _eps_star_static(eps_star_static, shape, where)
    if eps_m is not None:
        _real(eps_m, "eps_m", where, scalar=True)


def setup_elastic_3d(grid, C, Q, labels=None, eps_star_static=None):
    """Setup check for bulk3d.setup_elastic_3d (C and Q are tensor triples)."""
    where = "bulk3d.setup_elastic_3d"
    shape = grid_geometry(grid, 3, where=where)
    c = _triple(C, "C=(c1111, c1122, c1212)", where)
    for name, value in zip(("c1111", "c1122", "c1212"), c):
        _real(value, name, where, scalar=True)
    cubic_stiffness(*c, where=where, names=("c1111", "c1122", "c1212"))
    names = ("Q1111", "Q1122", "Q1212")
    if isinstance(Q, Mapping):
        for label, triple in Q.items():
            for name, value in zip(names, _triple(triple, f"Q[{label!r}]", where)):
                _real(value, f"Q[{label!r}] {name}", where, scalar=True)
    else:
        for name, value in zip(names, _triple(Q, "Q=(Q1111, Q1122, Q1212)", where)):
            _field(value, name, where, shape, leading=True)
    _eps_star_static(eps_star_static, shape, where)


def setup_film_elastic(stack_grid, C11, C12, C44, Q11, Q12, Q44, e0,
                       eps_star_static=None, C_substrate=None):
    """Setup check for film_elastic.setup_film_elastic (eps_star_static's
    shape and the substrate C are already checked there)."""
    where = "film_elastic.setup_film_elastic"
    if not isinstance(stack_grid, Mapping) or "film_lo" not in stack_grid:
        raise ValidationError(f"{where}: needs the film+substrate stack grid from "
                              f"film_elastic.make_stack_grid (no 'film_lo' key)")
    Nz = _count(stack_grid["Nz"], "stack_grid['Nz']", where, minimum=2)
    film_lo = _count(stack_grid["film_lo"], "stack_grid['film_lo']", where, minimum=0)
    Nz_film = _count(stack_grid.get("Nz_film", Nz - film_lo), "stack_grid['Nz_film']", where)
    if film_lo + Nz_film != Nz:
        raise ValidationError(f"{where}: stack grid film_lo={film_lo} + Nz_film={Nz_film} "
                              f"!= Nz={Nz}")
    if C_substrate is not None and film_lo < 1:
        raise ValidationError(f"{where}: C_substrate needs substrate layers: at least one substrate "
                              f"node (make_stack_grid(hs_dx>=1)); film_lo={film_lo}")
    shape = grid_geometry(stack_grid, 2, where=where)
    _real(stack_grid["hz"], "stack_grid['hz']", where, scalar=True, positive=True)
    for name, value in (("C11", C11), ("C12", C12), ("C44", C44)):
        _real(value, name, where, scalar=True)
    cubic_stiffness(C11, C12, C44, where=where)
    for name, value in (("Q11", Q11), ("Q12", Q12), ("Q44", Q44)):
        _field(value, name, where, (Nz_film,) + shape, leading=True)
    _real(e0, "e0", where, scalar=True)


def film_make_grid_3d(Nx, Ny, Lx, Ly, Nz, hf):
    """Setup check for film3d.make_grid_3d (Nz nodes span [0, hf] inclusive)."""
    where = "film3d.make_grid_3d"
    _count(Nx, "Nx", where)
    _count(Ny, "Ny", where)
    _count(Nz, "Nz", where, minimum=2, allow_integral_float=True)
    for name, value in (("Lx", Lx), ("Ly", Ly), ("hf", hf)):
        _real(value, name, where, scalar=True, positive=True)


def poisson_factor(grid, kappa_b):
    """Setup check for film3d.poisson_factor: kappa_b scalar or (Nz,) profile."""
    where = "film3d.poisson_factor"
    grid_geometry(grid, 2, where=where)
    Nz = _count(grid.get("Nz"), "grid['Nz']", where, minimum=2)
    _real(grid.get("hz"), "grid['hz']", where, scalar=True, positive=True)
    kap = _real(kappa_b, "kappa_b", where, positive=True)
    if kap.ndim and kap.shape != (Nz,):
        raise ValidationError(f"{where}: kappa_b must be a scalar or a (Nz,)=({Nz},) node "
                              f"profile, got shape {kap.shape}")


def _casekit_deck(d, sizes, h, elastic, where):
    for i, n in enumerate(sizes):
        _count(n, f"size[{i}]", where)
    _real(h, "cell size", where, scalar=True, positive=True)
    needed = ("gradient", "kappa_b") + (("elastic", "electrostriction") if elastic else ())
    deck_sectors(d, needed, where=where)


def build_2d(d, Nx, Ny, h):
    """Setup check for casekit.build_2d."""
    _casekit_deck(d, (Nx, Ny), h, False, "casekit.build_2d")


def build_3d(d, N, d_cell, elastic=False):
    """Setup check for casekit.build_3d."""
    _casekit_deck(d, (N,), d_cell, elastic, "casekit.build_3d")


def anneal(engine, P, alpha, grid, E_ext):
    """Setup check for casekit.anneal: P finite and shaped for the engine's
    grid, alpha a complete engine-native Landau dict that fits P's field
    shape, E_ext a 3-vector or a P-shaped field. An object that is not a
    solver-shaped or bulk3d-shaped engine is left to anneal's own error."""
    where = "casekit.anneal"
    ndim = 3 if hasattr(engine, "step_3d") else 2
    if not hasattr(engine, "step_langevin" + ("_3d" if ndim == 3 else "")):
        return
    if not isinstance(grid, Mapping):
        raise ValidationError(f"{where}: grid must be a dict, got {type(grid).__name__}")
    keys = ("Nz", "Ny", "Nx")[3 - ndim:]
    missing = [k for k in keys if k not in grid]
    if missing:
        raise ValidationError(f"{where}: grid lacks {missing} for a {ndim}D engine")
    shape = tuple(_count(grid[k], f"grid[{k!r}]", where) for k in keys)
    polarization(P, shape, where=where)
    P_shape = np.shape(P)
    landau_coefficients(alpha, field_shape=P_shape[1:], require=True, where=f"{where} alpha")
    applied_field(E_ext, P_shape, where=where)
