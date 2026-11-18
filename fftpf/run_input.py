"""RUN-ENTRY package E1 (SPEC § 16.3-16.5): parse and validate a run input file.

``load(path)`` reads a TOML file and returns ONE fully resolved plain nested dict
(every default written out); ``resolve(raw)`` does the same for an already parsed
dict; ``dumps(cfg)`` writes a resolved dict back as TOML ("resolved.toml").

This module imports no engine and builds no array.  It reads the TOML and checks
that referenced files exist.  Units stay in the key names (``_K``, ``_nm``,
``_kV_cm``, ``_pct``); nothing is converted here -- the builder does that.

Layout: SCHEMA is one declarative description (``Table`` / ``V`` objects); one
generic walker (``_walk``) checks unknown keys, types, ranges and fills defaults;
RULES is a flat list of the cross-key refusals of § 16.4 that can be decided from
the file alone; the last step refuses stage kinds and measures that the schema
knows but this version does not build (BUILT_STAGES, BUILT_MEASURES).

Every error is a ``ConfigError`` whose message is ``"<dotted.key.path>: <reason>"``.
A key whose default is None is absent from the resolved dict when not given.
"""
import copy
import difflib
import json
import math
import os
import tomllib

from fftpf.deck_reader import TOML_SECTIONS

BUILT_STAGES = ("relax", "anneal", "cool", "hold", "pe_loop", "ac", "fdt")
BUILT_MEASURES = ("mean_P", "energy_terms", "loop", "phase_fractions", "domain_map", "spectrum")
BUILT_WAVEFORMS = ("triangle", "unipolar")

_STAGE_KINDS = ("relax", "anneal", "cool", "hold", "pe_loop", "ac", "fdt")
_REQ = object()          # "no default: the key is required"


class ConfigError(ValueError):
    """Invalid run input; message is always '<dotted.key.path>: <reason>'."""


def _err(path, reason):
    raise ConfigError(f"{path}: {reason}")


def _join(path, key):
    return f"{path}.{key}" if path else key


# --------------------------------------------------------------------------- schema objects

class V:
    """A leaf value.  kind: str | int | num | bool | path | list | any.

    default: _REQ (required), None (absent when not given), a value, or a callable
    ``f(root, index)`` evaluated on the partially resolved config.  lo/hi bound
    numbers inclusively unless gt0 (strictly positive).  For kind 'list': item
    (str | int | num), n (exact length) or n_in (allowed lengths), choices (items).
    conv: optional ``f(value, path)`` for kind 'any' that validates and returns the value.
    """

    def __init__(self, kind, default=_REQ, choices=None, lo=None, hi=None, gt0=False,
                 item=None, n=None, n_in=None, conv=None):
        self.kind, self.default, self.choices = kind, default, choices
        self.lo, self.hi, self.gt0 = lo, hi, gt0
        self.item, self.n, self.n_in, self.conv = item, n, n_in, conv


class Table:
    """A table of named fields.

    variants: {discriminator value: extra fields}, discriminator key ``disc`` (must be
    a field of ``fields``).  optional: absent -> omitted from the result (else walked
    as an empty table so defaults fill).  required: absent -> error.  default: raw
    value (or callable ``f(root, index)``) substituted when absent.  pre: ``f(raw,
    path)`` rewriting shorthand before the walk.  check: ``f(out, path)`` for rules
    inside one table.
    """

    def __init__(self, fields, variants=None, disc=None, optional=False, required=False,
                 default=None, pre=None, check=None):
        self.fields, self.variants, self.disc = fields, variants, disc
        self.optional, self.required = optional, required
        self.default, self.pre, self.check = default, pre, check


class _Ctx:
    def __init__(self, base_dir, root):
        self.base_dir, self.root = base_dir, root


# --------------------------------------------------------------------------- leaf checking

def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_scalar(kind, v, path, ctx, gt0=False, lo=None, hi=None, choices=None):
    if kind == "str":
        if not isinstance(v, str):
            _err(path, f"must be a string, got {type(v).__name__}")
    elif kind == "path":
        if not isinstance(v, str):
            _err(path, f"must be a file path string, got {type(v).__name__}")
        v = os.path.abspath(os.path.join(ctx.base_dir, v))
        if not os.path.isfile(v):
            _err(path, f"file does not exist: {v}")
        return v
    elif kind == "bool":
        if not isinstance(v, bool):
            _err(path, f"must be true or false, got {type(v).__name__}")
    elif kind == "int":
        if isinstance(v, bool) or not isinstance(v, int):
            _err(path, f"must be an integer, got {type(v).__name__}")
    elif kind == "num":
        if not _is_num(v):
            _err(path, f"must be a number, got {type(v).__name__}")
        v = float(v)
        if not math.isfinite(v):
            _err(path, "must be finite")
    if kind in ("int", "num"):
        if gt0 and not v > 0:
            _err(path, f"must be > 0, got {v}")
        if lo is not None and v < lo:
            _err(path, f"must be >= {lo}, got {v}")
        if hi is not None and v > hi:
            _err(path, f"must be <= {hi}, got {v}")
    if choices is not None and v not in choices:
        hint = ""
        if isinstance(v, str):
            close = difflib.get_close_matches(v, [c for c in choices if isinstance(c, str)], n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
        _err(path, f"must be one of {list(choices)}, got {v!r}{hint}")
    return v


def _check_leaf(spec, v, path, ctx):
    if spec.kind == "any":
        return spec.conv(v, path)
    if spec.kind == "list":
        if not isinstance(v, list):
            _err(path, f"must be a list, got {type(v).__name__}")
        if spec.n is not None and len(v) != spec.n:
            _err(path, f"must have exactly {spec.n} entries, got {len(v)}")
        if spec.n_in is not None and len(v) not in spec.n_in:
            _err(path, f"must have {' or '.join(map(str, spec.n_in))} entries, got {len(v)}")
        if spec.n is None and spec.n_in is None and not v:
            _err(path, "must not be empty")
        return [_check_scalar(spec.item, x, f"{path}[{i}]", ctx, spec.gt0, spec.lo, spec.hi, spec.choices)
                for i, x in enumerate(v)]
    return _check_scalar(spec.kind, v, path, ctx, spec.gt0, spec.lo, spec.hi, spec.choices)


# --------------------------------------------------------------------------- the generic walker

def _unknown(key, valid, path):
    close = difflib.get_close_matches(key, list(valid), n=1)
    hint = f"; did you mean {close[0]!r}?" if close else f"; valid keys: {sorted(valid)}"
    _err(_join(path, key), f"unknown key{hint}")


def _walk(spec, raw, path, ctx, index=None, present=True):
    """Validate ``raw`` against ``spec``; return the resolved value (or _REQ-free None if absent-optional)."""
    if isinstance(spec, V):
        if not present:
            if spec.default is _REQ:
                _err(path, "required key is missing")
            d = spec.default(ctx.root, index) if callable(spec.default) else spec.default
            return d
        return _check_leaf(spec, raw, path, ctx)
    # Table
    if not present:
        if spec.required:
            _err(path, "required table is missing")
        if spec.default is not None:
            raw = spec.default(ctx.root, index) if callable(spec.default) else spec.default
        elif spec.optional:
            return None
        else:
            raw = {}
    if spec.pre is not None:
        raw = spec.pre(raw, path)
    if not isinstance(raw, dict):
        _err(path, f"must be a table, got {type(raw).__name__}")
    fields = dict(spec.fields)
    if spec.variants is not None:
        # the discriminator is checked first so a bad kind is reported before its keys are called unknown
        kind = _walk(fields[spec.disc], raw.get(spec.disc), _join(path, spec.disc), ctx, index,
                     present=spec.disc in raw)
        fields.update(spec.variants[kind])
    for key in raw:
        if key not in fields:
            _unknown(key, fields, path)
    out = {}
    for key, sub in fields.items():
        val = _walk(sub, raw.get(key), _join(path, key), ctx, index, present=key in raw)
        if val is not None:
            out[key] = val
    if spec.check is not None:
        spec.check(out, path)
    return out


# --------------------------------------------------------------------------- schema conversions / checks

def _conv_cell(v, path):
    if _is_num(v):
        return _check_scalar("num", v, path, None, gt0=True)
    return _check_leaf(V("list", item="num", gt0=True, n_in=(2, 3)), v, path, None)


def _conv_dt(v, path):
    if v == "verified":
        return v
    if _is_num(v):
        return _check_scalar("num", v, path, None, gt0=True)
    _err(path, f"must be a positive number or \"verified\", got {v!r}")


def _conv_depol(v, path):
    if _is_num(v):
        return _check_scalar("num", v, path, None, lo=0.0, hi=1.0)
    if (isinstance(v, list) and len(v) == 3
            and all(isinstance(r, list) and len(r) == 3 and all(_is_num(x) for x in r) for r in v)):
        m = [[float(x) for x in r] for r in v]
        if any(m[i][j] != m[j][i] for i in range(3) for j in range(3)):
            _err(path, "3x3 tensor must be symmetric")
        return m
    _err(path, "must be a number in [0, 1] or a symmetric 3x3 list of lists")


def _conv_at(v, path):
    if v == "all":
        return v
    if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
        return list(v)
    _err(path, f"must be \"all\" or a non-empty list of \"<stage kind or index>:<event>\" strings, got {v!r}")


RTO_MAX_ANGLE = 17.5          # deg: half of the 35.26 deg between <110> and <111> is 17.63, so the three cones stay disjoint


def _check_phase_fractions(out, path):
    scheme = out.get("scheme", "rt")
    if scheme == "rto":
        for key in ("P_ref", "R_factor"):
            if key in out:
                _err(_join(path, key), "is not used by scheme 'rto' (the classes are set by angle_deg and P_min alone); remove it")
        out.setdefault("angle_deg", 15.0)
        if out["angle_deg"] > RTO_MAX_ANGLE:
            _err(_join(path, "angle_deg"), f"must be <= {RTO_MAX_ANGLE:g} in scheme 'rto' so that the <100>, <110> and <111> cones "
                 f"stay disjoint, got {out['angle_deg']:g}")
        return
    del out["scheme"]                     # "rt" is the default and is not written to the resolved file
    if "P_ref" not in out:
        _err(_join(path, "P_ref"), "required key is missing")
    out.setdefault("angle_deg", 15.0)
    out.setdefault("R_factor", 0.7)
    if out["R_factor"] * out["P_ref"] < out["P_min"]:
        _err(_join(path, "R_factor"), f"R_factor * P_ref = {out['R_factor'] * out['P_ref']:g} is below P_min = {out['P_min']:g}: "
             "the R and low-|P| classes would overlap")


def _pre_start(raw, path):
    if isinstance(raw, str):
        return {"kind": raw}
    return raw


def _pre_stage(raw, path):
    """A unipolar pe_loop is deterministic: langevin defaults to false there."""
    if (isinstance(raw, dict) and raw.get("kind") == "pe_loop" and raw.get("waveform") == "unipolar"
            and "langevin" not in raw):
        raw = dict(raw, langevin=False)
    return raw


def _check_stage(out, path):
    kind = out["kind"]
    if kind == "anneal" and "block_stationary" in out["stop"] and out["sample_every"] == 0:
        out["sample_every"] = min(100, out["stop"]["n_max"])         # the stop test reads the samples
    if kind == "hold" and out["ramp_steps"] > out["n_steps"]:
        _err(_join(path, "ramp_steps"), f"must be <= n_steps ({out['n_steps']})")
    if kind == "ac":
        if out["discard_periods"] >= out["periods"]:
            _err(_join(path, "discard_periods"), f"must be < periods ({out['periods']}): at least one period is kept")
        tags = [f"{w:.6g}" for w in out["omega"]]
        if len(set(tags)) != len(tags):
            _err(_join(path, "omega"), "the tones must differ in their first 6 significant digits (they name the series files)")
    if kind == "fdt":
        n, seg = out["n_steps"], out["nperseg"]
        if seg > n:
            _err(_join(path, "nperseg"), f"must be <= n_steps ({n})")
        out.setdefault("max_lag", min(seg, n // 8))
        if out["max_lag"] < 2:
            _err(_join(path, "n_steps"), f"too short: the correlation needs max_lag >= 2 and max_lag <= n_steps // 8")
        if out["max_lag"] > n // 8:
            _err(_join(path, "max_lag"), f"must be <= n_steps // 8 = {n // 8} (the resolved range is read from 4 sub-series)")
    if kind != "pe_loop":
        return
    wf = out["waveform"]
    if wf == "triangle":
        for key in ("n_points", "n_dwell"):
            if key in out:
                _err(_join(path, key), "only for waveform 'unipolar'")
        if "steps_per_cycle" not in out:
            _err(_join(path, "steps_per_cycle"), "required for waveform 'triangle'")
        spc = out["steps_per_cycle"]
        if spc % 4:
            _err(_join(path, "steps_per_cycle"), f"must be a multiple of 4 (four sweeps per cycle), got {spc}")
        cycles = out.get("cycles", 1.0)
        if abs(cycles * 4 - round(cycles * 4)) > 1e-9:
            _err(_join(path, "cycles"), f"must be a multiple of 0.25, got {cycles}")
        if "record_every" in out and (spc // 4) % out["record_every"]:
            _err(_join(path, "record_every"), f"must divide the quarter cycle ({spc // 4} steps)")
    elif wf == "unipolar":
        for key in ("steps_per_cycle", "cycles", "record_every"):
            if key in out:
                _err(_join(path, key), "not used by waveform 'unipolar' (one record per field point)")
        for key in ("n_points", "n_dwell"):
            if key not in out:
                _err(_join(path, key), "required for waveform 'unipolar'")


def _check_anneal_stop(out, path):
    has_n, has_b = "n_steps" in out, "block_stationary" in out
    if has_n == has_b:
        _err(path, "give exactly one of n_steps or block_stationary (with n_max)")
    if has_b and "n_max" not in out:
        _err(_join(path, "n_max"), "required with block_stationary")
    if has_n and "n_max" in out:
        _err(_join(path, "n_max"), "only used with block_stationary")
    if has_n and "n_blocks" in out:
        _err(_join(path, "n_blocks"), "only used with block_stationary")


def _check_material(out, path):
    inline = [k for k in ("landau", "landau_si", "gradient", "gradient_si", "elastic", "electrostriction") if k in out]
    if "deck" in out and inline:
        _err(_join(path, "deck"), f"deck and inline tables are both given ({inline[0]}); give one")
    if "deck" not in out:
        if not inline:
            _err(path, "give either deck or the inline tables landau (or landau_si), gradient, elastic, electrostriction")
        if "landau" in out and "landau_si" in out:
            _err(_join(path, "landau_si"), "landau and landau_si are both given; give one")
        if "gradient" in out and "gradient_si" in out:
            _err(_join(path, "gradient_si"), "gradient and gradient_si are both given; give one")
        if "gradient" not in out and "gradient_si" not in out:
            _err(_join(path, "gradient"), "required with an inline material (gradient or gradient_si)")
        for need in ("elastic", "electrostriction"):
            if need not in out:
                _err(_join(path, need), "required with an inline material")
        lk = "landau" if "landau" in out else "landau_si"
        if lk not in out:
            _err(_join(path, "landau"), "required with an inline material (landau or landau_si)")
        for coef in ("alpha1", "alpha11", "alpha12", "alpha111", "alpha112", "alpha123"):    # the sixth-order set; mat_registry
            if coef not in out[lk]:                         # zero-fills only the eighth-order ones (input_validation.LANDAU_OPTIONAL)
                _err(_join(_join(path, lk), coef), "required")
        if lk == "landau" and isinstance(out[lk]["alpha1"], dict):
            _err(_join(path, "landau.alpha1"), "the slope / T0_K form is for landau_si only")


def _conv_deck(v, path):
    if not isinstance(v, str):
        _err(path, f"must be a string, got {type(v).__name__}")
    return v


def _registry_names():
    from fftpf import mat_registry          # lazy: a material table, not an engine
    return [n for n, _ in mat_registry.list()], tuple(mat_registry._ARCHIVED)


def _registry_problem(name):
    """None if ``name`` is a live registry deck, else the reason it is refused."""
    names, archived = _registry_names()
    if name in archived:
        return f"{name!r} is an archived deck (no computation uses it)"
    if name not in names:
        close = difflib.get_close_matches(name, names, n=1)
        hint = f"; did you mean {close[0]!r}?" if close else ""
        return f"unknown material {name!r}; available: {names}{hint}"
    return None


def _inline_table(keys):
    return Table({k: V("num", default=None) for k in keys}, optional=True)


def _conv_alpha1(v, path):
    """A number, or the temperature law { slope = <J/m^3/K per (C/m^2)^2>, T0_K = <K> }."""
    if _is_num(v):
        return _check_scalar("num", v, path, None)
    if isinstance(v, dict):
        for key in v:
            if key not in ("slope", "T0_K"):
                _unknown(key, ("slope", "T0_K"), path)
        for key in ("slope", "T0_K"):
            if key not in v:
                _err(_join(path, key), "required in the temperature law")
        return {"slope": _check_scalar("num", v["slope"], _join(path, "slope"), None),
                "T0_K": _check_scalar("num", v["T0_K"], _join(path, "T0_K"), None)}
    _err(path, "must be a number or { slope = ..., T0_K = ... }")


def _landau_si_table():
    fields = {k: V("num", default=None) for k in TOML_SECTIONS["landau"]}
    fields["alpha1"] = V("any", default=None, conv=_conv_alpha1)
    return Table(fields, optional=True)


def _conv_rungs(v, path):
    if not isinstance(v, list) or not v:
        _err(path, "must be a non-empty list of [T_K, n_steps] pairs")
    out = []
    for j, r in enumerate(v):
        p = f"{path}[{j}]"
        if not (isinstance(r, list) and len(r) == 2):
            _err(p, "must be a pair [T_K, n_steps]")
        out.append([_check_scalar("num", r[0], p + "[0]", None, gt0=True),
                    _check_scalar("int", r[1], p + "[1]", None, lo=1)])
    return out


def _field_variants():
    return {
        "gaussian": {"sigma_kV_cm": V("num", default=None, gt0=True), "sigma": V("num", default=None, gt0=True),
                     "sigma_K": V("num", default=None, gt0=True),
                     "correlation_nm": V("num", gt0=True), "seed": V("int", default=None),
                     "seeds": V("list", item="int", n=3, default=None)},
        "from_file": {"file": V("path"), "key": V("str")},
    }


def _check_field(out, path):
    if out["kind"] != "gaussian":
        return
    target = out["target"]
    want = {"E_ext": "sigma_kV_cm", "T0": "sigma_K"}.get(target, "sigma")
    if want not in out:
        _err(_join(path, want), f"required for target {target!r}" + (" (deck units)" if want == "sigma" else ""))
    for key in ("sigma_kV_cm", "sigma_K", "sigma"):
        if key != want and key in out:
            _err(_join(path, key), f"not for target {target!r}; use {want}")
    if "seeds" in out:
        if target != "E_ext":
            _err(_join(path, "seeds"), "only for target 'E_ext' (one seed per component)")
        if "seed" in out:
            _err(_join(path, "seeds"), "seed and seeds are both given; give one")
        if len(set(out["seeds"])) != 3:
            _err(_join(path, "seeds"), "the three seeds must be distinct")
    elif "seed" not in out:
        _err(_join(path, "seed"), "required (or seeds = [s1, s2, s3] for target 'E_ext')")


def _conv_target(v, path):
    if not isinstance(v, str):
        _err(path, f"must be a string, got {type(v).__name__}")
    if v in ("E_ext", "T0", "eps_star"):
        return v
    if v.startswith("landau."):
        coef = v.split(".", 1)[1]
        if coef not in TOML_SECTIONS["landau"]:
            close = difflib.get_close_matches(coef, TOML_SECTIONS["landau"], n=1)
            _err(path, f"unknown Landau coefficient {coef!r}" + (f"; did you mean 'landau.{close[0]}'?" if close else ""))
        return v
    _err(path, f"must be 'E_ext', 'T0', 'eps_star' or 'landau.<coef>', got {v!r}")


_AXIS = V("list", item="num", n=3)
_T_K = V("num", default=None, gt0=True)                    # absent: material.temperature_K
_SEED = V("int", default=None)                             # absent: the run's one generator carries on
_RECORD = V("int", default=None, lo=1)
_LANGEVIN = V("bool", default=lambda root, i: root["material"].get("temperature_K") is not None)

_STAGE_VARIANTS = {
    "relax": {"T_K": _T_K,
              "stop": Table({"tol": V("num", default=1e-6, gt0=True), "n_max": V("int", default=50000, lo=1)})},
    "anneal": {
        "T_K": _T_K, "langevin": _LANGEVIN,
        "stop": Table({"n_steps": V("int", default=None, lo=1), "block_stationary": V("num", default=None, gt0=True),
                       "n_max": V("int", default=None, lo=1), "n_blocks": V("int", default=None, lo=2)}, required=True, check=_check_anneal_stop),
        "sample_every": V("int", default=0, lo=0),
        "seed": _SEED,
    },
    "cool": {"rungs": V("any", conv=_conv_rungs), "langevin": _LANGEVIN, "seed": _SEED, "record_every": _RECORD},
    "hold": {"T_K": _T_K, "n_steps": V("int", lo=1), "E_kV_cm": V("num", default=0.0), "axis": V("list", item="num", n=3, default=None),
             "ramp_steps": V("int", default=0, lo=0), "langevin": _LANGEVIN, "seed": _SEED, "record_every": _RECORD},
    "pe_loop": {
        "T_K": _T_K, "axis": _AXIS, "E_max_kV_cm": V("num", gt0=True),
        "waveform": V("str", default="triangle", choices=("triangle", "sine", "staircase", "unipolar")),
        "cycles": V("num", default=None, gt0=True), "steps_per_cycle": V("int", default=None, lo=1),
        "n_points": V("int", default=None, lo=1), "n_dwell": V("int", default=None, lo=1),
        "langevin": _LANGEVIN, "seed": _SEED, "record_every": _RECORD,
    },
    "ac": {                       # one tone after another from the entry state; omega in rad per deck time
        "T_K": _T_K, "langevin": _LANGEVIN, "seed": _SEED,
        "axis": _AXIS, "E0_kV_cm": V("num", gt0=True), "omega": V("list", item="num", gt0=True),
        "periods": V("int", lo=1), "discard_periods": V("int", default=0, lo=0),
        "dc_bias_kV_cm": V("num", default=0.0),
    },
    "fdt": {                      # zero-field fluctuation route; Langevin is required
        "T_K": _T_K, "langevin": _LANGEVIN, "seed": _SEED,
        "n_steps": V("int", lo=1), "nperseg": V("int", lo=2),
        "burn_steps": V("int", default=0, lo=0), "max_lag": V("int", default=None, lo=2),
        "axis": V("list", item="num", n=3, default=None),     # absent: the axis of the file's ac stage
    },
}

SCHEMA = {
    "run": Table({"name": V("str"), "seed": V("int", default=0), "replicas": V("int", default=1, lo=1)}, required=True),
    "material": Table({
        "deck": V("any", default=None, conv=_conv_deck),
        "temperature_K": V("num", default=None, gt0=True),
        "landau": _inline_table(TOML_SECTIONS["landau"]),
        "landau_si": _landau_si_table(),
        "gradient": _inline_table(TOML_SECTIONS["gradient"]),
        "gradient_si": Table({"isotropic": V("num", default=None, gt0=True)}, optional=True),
        "elastic": _inline_table(TOML_SECTIONS["elastic"]),
        "electrostriction": _inline_table(TOML_SECTIONS["electrostriction"]),
    }, required=True, check=_check_material),
    "geometry": Table({
        "engine": V("str", choices=("pf2d", "pf3d", "film")),
        "N": V("list", item="int", lo=1, n_in=(2, 3)),
        "cell_nm": V("any", default=1.0, conv=_conv_cell),
        "slab_thickness_nm": V("num", default=None, gt0=True),
        "film_thickness_nm": V("num", default=None, gt0=True),
    }, required=True),
    "physics": Table({
        "electrostatics": Table({"eps_b": V("num", default=None, gt0=True),     # absent: deck kappa_b, else 10
                                 "depol_N": V("any", default=None, conv=_conv_depol)}),
        "elastic": Table({
            "k0": V("str", default="stress_free", choices=("stress_free", "clamped", "off")),   # off: no elasticity
            "z_mode": V("str", default="generalized_plane_strain",
                        choices=("generalized_plane_strain", "plane_stress")),
            "misfit_strain_pct": V("num", default=None),
            "substrate": V("str", default=None),
            "substrate_layers": V("int", default=None, lo=1),
        }),
        "noise": Table({"cutoff_nm": V("num", default=2.0, lo=0.0)}),   # 0 = no cutoff
    }),
    "numerics": Table({
        "scheme": V("str", default="sbdf1", choices=("sbdf1", "sbdf2")),
        "dt": V("any", default="verified", conv=_conv_dt),
        "dt_verify_time": V("num", default=100.0, gt0=True),
        "L": V("num", default=1.0, gt0=True),
        "backend": V("str", default="numpy", choices=("numpy", "numba", "fused")),
        "threads": V("int", default=1, lo=1),
    }),
    "measure": Table({
        "mean_P": V("bool", default=True), "energy_terms": V("bool", default=False),
        "loop": V("list", item="str", default=None,
                  choices=("Pr", "Ec", "Pmax", "W_store", "W_rec", "W_loss", "eta")),
        "phase_fractions": Table({
            "scheme": V("str", default="rt", choices=("rt", "rto")),      # rt: R / T (P_ref required); rto: T / O / R cones
            "P_ref": V("num", gt0=True, default=None),            # C/m^2, the reference rhombohedral |P| (rt only; required there)
            "angle_deg": V("num", default=None, gt0=True, hi=25.0),   # default 15; rt: <= 25 keeps T15 inside T; rto: <= 17.5
            "P_min": V("num", default=0.05, gt0=True),            # C/m^2
            "R_factor": V("num", default=None, gt0=True),         # rt only, default 0.7
        }, optional=True, check=_check_phase_fractions),
        "domain_map": Table({
            "at": V("any", default="all", conv=_conv_at),
            "kind": V("list", item="str", choices=("phase", "direction"),
                      default=lambda root, i: ["direction"]),
        }, optional=True),
        "spectrum": Table({
            "quantities": V("list", item="str", choices=("eps_real", "eps_imag", "tan_delta")),
            "routes": V("list", item="str", choices=("lockin", "fdt")),
        }, optional=True),
        "seconds_per_deck_time": V("num", default=None, gt0=True),
    }),
    "output": Table({"dir": V("str", default="auto"), "checkpoint_every": V("int", default=0, lo=0)}),
}

_STAGE = Table({
    "kind": V("str", choices=_STAGE_KINDS),
    "start": Table({"kind": V("str", choices=("random", "white", "zero", "previous", "from_file"))},
                   variants={"zero": {},
                             "random": {"amplitude": V("num", default=0.005, gt0=True),
                                        "kmax_frac": V("num", default=0.3, gt0=True, hi=1.0),
                                        "seed": V("int", default=lambda root, i: root["run"]["seed"])},
                             "white": {"std": V("num", gt0=True),
                                       "seed": V("int", default=lambda root, i: root["run"]["seed"])},
                             "previous": {},
                             "from_file": {"file": V("path"), "key": V("str")}},
                   disc="kind", pre=_pre_start,
                   default=lambda root, i: "random" if i == 0 else "previous"),
}, variants=_STAGE_VARIANTS, disc="kind", pre=_pre_stage, check=_check_stage)

_FIELD = Table({
    "target": V("any", conv=_conv_target),
    "kind": V("str", choices=("gaussian", "from_file")),
}, variants=_field_variants(), disc="kind", check=_check_field)

_TOP_ORDER = ("run", "material", "geometry", "physics", "numerics", "stage", "field", "measure", "output")


# --------------------------------------------------------------------------- cross-key refusals (§ 16.4)

def _rules_geometry(cfg):
    g, ph = cfg["geometry"], cfg["physics"]
    eng, N = g["engine"], g["N"]
    want = 2 if eng == "pf2d" else 3
    if len(N) != want:
        yield "geometry.N", f"engine {eng!r} needs {want} numbers, got {len(N)}"
    cell = g["cell_nm"]
    n_cell = 2 if eng == "film" else want        # film: in-plane cell only; the z spacing comes from film_thickness_nm / Nz
    if isinstance(cell, list) and len(cell) != n_cell:
        yield "geometry.cell_nm", f"engine {eng!r} needs a scalar or {n_cell} numbers, got {len(cell)}"
    if eng != "pf2d" and "slab_thickness_nm" in g and not (eng == "pf3d" and len(N) == 3 and N[2] == 1):
        yield "geometry.slab_thickness_nm", (f"only for engine 'pf2d' (Dz) or engine 'pf3d' with Nz = 1 (Lz); "
                                             f"engine {eng!r} has its own z spacing")
    if eng == "film" and "film_thickness_nm" not in g:
        yield "geometry.film_thickness_nm", "required for engine 'film'"
    if eng != "film" and "film_thickness_nm" in g:
        yield "geometry.film_thickness_nm", "only for engine 'film'"
    e = ph["elastic"]
    if eng != "pf3d" and "depol_N" in ph["electrostatics"]:
        yield "physics.electrostatics.depol_N", f"only for engine 'pf3d', not {eng!r}"
    if e["z_mode"] == "plane_stress" and not (eng == "pf3d" and len(N) == 3 and N[2] == 1):
        yield "physics.elastic.z_mode", "plane_stress needs engine 'pf3d' with Nz = 1"
    if e["k0"] == "clamped" and cfg["run"]["replicas"] > 1:
        yield "physics.elastic.k0", "clamped is not supported with run.replicas > 1"
    for key in ("misfit_strain_pct", "substrate", "substrate_layers"):
        if eng != "film" and key in e:
            yield f"physics.elastic.{key}", f"only for engine 'film', not {eng!r}"
    if eng == "film" and "misfit_strain_pct" not in e:
        yield "physics.elastic.misfit_strain_pct", "required for engine 'film'"
    if eng == "film" and cfg["run"]["replicas"] > 1:
        yield "run.replicas", "replicas > 1 is not supported for engine 'film'"
    if "substrate" in e:
        if _registry_problem(e["substrate"]):
            yield "physics.elastic.substrate", _registry_problem(e["substrate"])
        else:
            d, why = _deck_of(e["substrate"], cfg["material"].get("temperature_K"))
            if d is None:
                yield "physics.elastic.substrate", why
            elif "elastic" not in d:
                yield "physics.elastic.substrate", f"{e['substrate']!r} has no elastic sector; a substrate deck must give C11, C12, C44"


P_SECTORS = ("landau", "gradient", "elastic", "electrostriction")      # what a material deck must have to run the P engines


def _deck_of(deck, T):
    """(the deck dict, None) of a registry name or an absolute .toml path, or (None, the reason it cannot be read)."""
    from fftpf import mat_registry
    try:
        if deck.endswith(".toml") or os.sep in deck:
            return mat_registry.from_toml(deck), None
        return mat_registry.get(deck, **({} if T is None else {"T": T})), None
    except (KeyError, ValueError, OSError, TypeError) as exc:
        return None, str(exc).strip("'\"")


def deck_problem(name, d):
    """None when the deck dict ``d`` has what a run reads (the Landau sector with its six sixth-order coefficients, gradient,
    elastic, electrostriction), else the reason: mat_registry zero-fills only the eighth-order Landau coefficients, so a
    deck without a sixth-order one would fail at the first step."""
    lacking = [s for s in P_SECTORS if s not in d]
    if lacking:
        only = (" (a substrate deck: it holds the elastic constants only; use it as physics.elastic.substrate)"
                if {s for s in d if s in P_SECTORS} == {"elastic"} else "")
        return (f"{name!r} has no {', '.join(lacking)} sector(s), which a run needs{only}; "
                "give a deck that has the Landau, gradient, elastic and electrostriction sectors")
    from fftpf.input_validation import LANDAU_REQUIRED
    missing = [k for k in LANDAU_REQUIRED if k not in d["landau"]]
    if missing:
        return f"{name!r} has no sixth-order Landau coefficient(s) {missing}; the engine reads all of {list(LANDAU_REQUIRED)}"
    return None


def _rules_material(cfg):
    deck = cfg["material"].get("deck")
    if deck is not None:
        if deck.endswith(".toml") or os.sep in deck:
            return                      # a file: read by the builder (run_builder.material), whose dry build is ``check``
        if _registry_problem(deck):
            yield "material.deck", _registry_problem(deck)
            return
        d, why = _deck_of(deck, cfg["material"].get("temperature_K"))
        problem = why if d is None else deck_problem(deck, d)
        if problem:
            yield "material.deck", problem


def _law(cfg):
    """How alpha1 depends on T: 'slope' (inline landau_si.alpha1 = {slope, T0_K}), 'registry' (a registry deck whose
    generator takes T) or None (a fixed-temperature deck: toml file, inline numbers, or a registry deck without T)."""
    m = cfg["material"]
    if isinstance(m.get("landau_si", {}).get("alpha1"), dict):
        return "slope"
    deck = m.get("deck")
    if deck is not None and not (deck.endswith(".toml") or os.sep in deck) and not _registry_problem(deck):
        import inspect
        from fftpf import mat_registry
        if "T" in inspect.signature(mat_registry._REGISTRY[deck][0]).parameters:
            return "registry"
    return None


def _temperatures(cfg):
    """[(key path, T_K)] of every temperature a stage states."""
    out = []
    for i, st in enumerate(cfg["stage"]):
        if st["kind"] == "cool":
            out += [(f"stage[{i}].rungs[{j}]", r[0]) for j, r in enumerate(st["rungs"])]
        elif "T_K" in st:
            out.append((f"stage[{i}].T_K", st["T_K"]))
    return out


def _rules_temperature(cfg):
    m, temps = cfg["material"], _temperatures(cfg)
    T0, law = m.get("temperature_K"), _law(cfg)
    ref = T0 if T0 is not None else (temps[0][1] if temps else None)
    if law is None:
        for path, T in temps:
            if T != ref:
                what = f"deck {m['deck']!r}" if "deck" in m else "the inline material"
                yield path, (f"{what} is a fixed-temperature deck (valid at {ref} K); give a registry deck whose generator takes T "
                             "or material.landau_si.alpha1 = { slope = ..., T0_K = ... } to run stages at other temperatures")
                break
    if law == "slope" and ref is None:
        yield "material.temperature_K", "needed with the temperature law alpha1 = { slope, T0_K } (or a T_K on a stage)"
    for i, st in enumerate(cfg["stage"]):
        if st.get("langevin") and st["kind"] != "cool" and "T_K" not in st and T0 is None:
            yield f"stage[{i}].langevin", "needs stage T_K or material.temperature_K"
    for i, fd in enumerate(cfg["field"]):
        if fd["target"] == "T0" and law != "slope":
            yield f"field[{i}].target", "'T0' needs material.landau_si.alpha1 = { slope = ..., T0_K = ... }"
        if fd["target"] == "landau.alpha1" and law == "slope":
            yield f"field[{i}].target", "with the temperature law alpha1 = { slope, T0_K } the alpha1 disorder is a 'T0' field"


def _rules_stages(cfg):
    eng, T = cfg["geometry"]["engine"], cfg["material"].get("temperature_K")
    scheme, slab = cfg["numerics"]["scheme"], cfg["geometry"].get("slab_thickness_nm")
    kinds = [s["kind"] for s in cfg["stage"]]
    if cfg["stage"] and cfg["stage"][0]["start"]["kind"] == "previous":
        yield "stage[0].start", "'previous' has nothing to continue from in the first stage"
    axis = None
    for i, st in enumerate(cfg["stage"]):
        p = f"stage[{i}]"
        if st["kind"] in ("ac", "fdt") and eng == "film":
            yield f"{p}.kind", f"{st['kind']!r} is not available for engine 'film' (first version: no anchor case exists)"
        if st["kind"] == "fdt" and not st["langevin"]:
            yield f"{p}.langevin", "the fdt stage reads the equilibrium fluctuations and needs Langevin noise (langevin = true, with a temperature)"
        if st.get("langevin"):
            if T is None and "T_K" not in st and st["kind"] != "cool":
                yield f"{p}.langevin", "needs material.temperature_K"
            if scheme == "sbdf2":
                yield f"{p}.langevin", "Langevin noise is not available with numerics.scheme = 'sbdf2'"
            if eng == "pf2d" and slab is None:
                yield f"{p}.langevin", "Langevin on engine 'pf2d' needs geometry.slab_thickness_nm"
        if "axis" in st:
            if not any(st["axis"]):
                yield f"{p}.axis", "must not be the zero vector"
            if st["kind"] in ("hold", "pe_loop"):          # the axis ctx["n"] carries; ac / fdt keep their own
                axis = st["axis"]
        if st["kind"] == "hold" and (st["E_kV_cm"] != 0.0 or st["ramp_steps"] > 0) and axis is None:
            yield f"{p}.axis", "needed for a hold with a field (or a ramp): give it here or in an earlier hold / pe_loop"
        if st["kind"] == "pe_loop" and st["waveform"] == "unipolar":
            if st["langevin"]:
                yield f"{p}.langevin", "waveform 'unipolar' is deterministic (the reference path run_unipolar); set langevin = false"
            if scheme != "sbdf1":
                yield "numerics.scheme", "waveform 'unipolar' steps with sbdf1 (the reference path run_unipolar)"
    m = cfg["measure"]
    if "loop" in m:
        loops = [s for s in cfg["stage"] if s["kind"] == "pe_loop"]
        if not loops:
            yield "measure.loop", "needs a pe_loop stage in the file"
        for st in loops:
            wf = st["waveform"]
            if wf not in BUILT_WAVEFORMS:
                continue
            allowed = {"triangle": ("Pr", "Ec", "Pmax", "W_store", "W_rec", "W_loss", "eta"),
                       "unipolar": ("Pr", "Pmax", "W_store", "W_rec", "W_loss", "eta")}[wf]
            for name in m["loop"]:
                if name not in allowed:
                    yield "measure.loop", f"{name!r} cannot be read from waveform {wf!r} (it gives {list(allowed)})"
            if wf == "triangle" and st.get("cycles", 1.0) < 1:
                yield "measure.loop", "a triangle loop needs at least one full cycle (cycles >= 1) to read a loop measure"
    if "domain_map" in m:
        if "phase" in m["domain_map"]["kind"] and "phase_fractions" not in m:
            yield "measure.domain_map.kind", "'phase' needs the measure.phase_fractions table (it gives the class thresholds)"
        from fftpf import run_measure
        run_measure.domain_map_targets(cfg)            # raises ConfigError for a selector that matches no saved state
    if "spectrum" in m:
        need = {"lockin": "ac", "fdt": "fdt"}
        for r in m["spectrum"]["routes"]:
            if need[r] not in kinds:
                yield "measure.spectrum.routes", f"route {r!r} needs a {need[r]!r} stage in the file"
            elif kinds.count(need[r]) > 1:
                yield "measure.spectrum.routes", f"route {r!r} reads exactly one {need[r]!r} stage; the file has {kinds.count(need[r])}"
    if cfg["run"]["replicas"] > 1 and not {"ac", "fdt"} & set(kinds):
        yield "run.replicas", "replicas pool a spectrum; the file needs an 'ac' or 'fdt' stage"
    if cfg["run"]["replicas"] > 1 and "spectrum" not in m:
        yield "run.replicas", "needs measure.spectrum (replicas exist to pool the spectrum)"


_RULES = (_rules_geometry, _rules_material, _rules_temperature, _rules_stages)


def _refuse_unbuilt(cfg):
    for i, st in enumerate(cfg["stage"]):
        if st["kind"] not in BUILT_STAGES:
            _err(f"stage[{i}].kind", f"{st['kind']!r} is specified in SPEC § 16 but not built in this version")
        if st["kind"] == "pe_loop" and st["waveform"] not in BUILT_WAVEFORMS:
            _err(f"stage[{i}].waveform", f"{st['waveform']!r} is specified in SPEC § 16 but not built in this version "
                 f"(built: {list(BUILT_WAVEFORMS)})")
    for key, val in cfg["measure"].items():
        if key == "seconds_per_deck_time" or val is None or val is False:
            continue
        if key not in BUILT_MEASURES:
            _err(f"measure.{key}", f"{key!r} is specified in SPEC § 16 but not built in this version")


# --------------------------------------------------------------------------- public API

def _finish(cfg, base_dir):
    """Fill the defaults that need several keys at once (cell_nm, film), and resolve deck paths."""
    g = cfg["geometry"]
    n = 2 if g["engine"] in ("pf2d", "film") else 3
    if _is_num(g["cell_nm"]):
        g["cell_nm"] = [float(g["cell_nm"])] * n
    if g["engine"] == "film":
        cfg["physics"]["elastic"].setdefault("substrate_layers", 12)
    ac_axes = [st["axis"] for st in cfg["stage"] if st["kind"] == "ac"]
    for i, st in enumerate(cfg["stage"]):
        if st["kind"] == "fdt" and "axis" not in st:
            if not ac_axes:
                _err(f"stage[{i}].axis", "required: the file has no 'ac' stage whose axis it could follow")
            st["axis"] = list(ac_axes[0])
    deck = cfg["material"].get("deck")
    if deck is not None and (deck.endswith(".toml") or os.sep in deck):
        path = os.path.abspath(os.path.join(base_dir, deck))
        if not os.path.isfile(path):
            _err("material.deck", f"file does not exist: {path}")
        cfg["material"]["deck"] = path


def resolve(raw, base_dir="."):
    """Validate ``raw`` (a parsed TOML dict) and return a new fully resolved dict.

    ``raw`` is not modified.  Relative file paths are resolved against ``base_dir``.
    """
    if not isinstance(raw, dict):
        _err("<root>", "input must be a table")
    raw = copy.deepcopy(raw)
    for key in raw:
        if key not in _TOP_ORDER:
            _unknown(key, _TOP_ORDER, "")
    cfg = {}
    ctx = _Ctx(os.path.abspath(base_dir), cfg)
    for key in _TOP_ORDER:
        if key == "stage":
            cfg[key] = _walk_list(_STAGE, raw.get(key), "stage", ctx, required=True)
        elif key == "field":
            cfg[key] = _walk_list(_FIELD, raw.get(key), "field", ctx, required=False)
        else:
            cfg[key] = _walk(SCHEMA[key], raw.get(key), key, ctx, present=key in raw)
    _finish(cfg, ctx.base_dir)
    for rule in _RULES:
        for path, reason in rule(cfg):
            _err(path, reason)
    _refuse_unbuilt(cfg)
    return cfg


def _walk_list(spec, raw, path, ctx, required):
    if raw is None:
        if required:
            _err(path, "at least one [[stage]] is required")
        return []
    if not isinstance(raw, list) or not all(isinstance(x, dict) for x in raw):
        _err(path, "must be an array of tables ([[%s]])" % path)
    if required and not raw:
        _err(path, "at least one [[stage]] is required")
    return [_walk(spec, x, f"{path}[{i}]", ctx, index=i) for i, x in enumerate(raw)]


def load(path):
    """Read the TOML file ``path`` and return the resolved config dict."""
    path = os.path.abspath(path)
    with open(path, "rb") as fh:
        try:
            raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: not valid TOML ({exc})") from None
    return resolve(raw, base_dir=os.path.dirname(path))


# --------------------------------------------------------------------------- TOML writer

def _val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_val(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} as TOML")


def _emit(table, header, lines):
    scalars = {k: v for k, v in table.items() if not isinstance(v, dict) and not _is_tables(v)}
    if header:
        lines.append(f"[{header}]" if not header.endswith("]]") else header)
    for k, v in scalars.items():
        lines.append(f"{k} = {_val(v)}")
    lines.append("")
    for k, v in table.items():
        if isinstance(v, dict):
            _emit(v, f"{_hdr(header)}{k}", lines)
        elif _is_tables(v):
            for item in v:
                _emit(item, f"[[{_hdr(header)}{k}]]", lines)


def _hdr(header):
    return header.strip("[]") + "." if header else ""


def _is_tables(v):
    return isinstance(v, list) and bool(v) and all(isinstance(x, dict) for x in v)


def dumps(cfg):
    """Return TOML text of a resolved config; ``resolve(tomllib.loads(dumps(cfg)))`` equals ``cfg``."""
    lines = []
    _emit(cfg, "", lines)
    return "\n".join(lines).rstrip() + "\n"
