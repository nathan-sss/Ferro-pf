"""RUN-ENTRY package E2 (SPEC § 16.2, 16.4): turn a resolved run config into built objects.

``build(cfg)`` returns a plain dict with the deck, grid, ``alpha`` (array entries where a
``[[field]]`` targets ``landau.<coef>``), ``E_static`` (the zero 3-vector or the summed
``E_ext`` field), the ``Adapter``, ``dt``, ``kT`` and the thread/backend line.  It makes
exactly the calls the existing code makes for the same numbers (casekit.build_2d,
casekit.build_3d, cases/bto_film_biaxial.setup) -- ``fftpf`` never imports ``casekit`` or
``cases`` (§ 1 rule 7), tests prove the equality.  ``explain(cfg)`` describes the choices
without building arrays.

Every refusal that needs a built object (or an array file) raises ``ConfigError`` with
"<key path>: <reason>" before any step is taken.  ``precheck(cfg)`` holds the cheapest ones;
``check(cfg)`` is ``build(cfg, dry=True)``: the same path as a run up to the arrays, the backend and the first step, so
the CLI ``check`` and ``explain`` raise every refusal ``run`` would raise before its first step.

Conventions fixed here:
  * ``physics.electrostatics.eps_b`` absent: the deck's own ``kappa_b`` (registry decks have
    it); inline decks have none and use eps_b = 10.  Given: ``units.relative_to_deck_permittivity``.
  * ``physics.noise.cutoff_nm = 0`` means no cutoff (``noise_cutoff_nm=None`` to the engine).
  * ``physics.elastic.k0 = "off"``: no elasticity (pf2d / pf3d only; the film always has it).
  * ``[[field]] kind="gaussian"``: target ``E_ext`` -> one field per Cartesian component, seeds
    ``seed``, ``seed + 1``, ``seed + 2`` (the convention of random_field_mechanisms.charge_
    disorder_field), rms ``sigma_kV_cm`` converted by ``units.si_to_deck_field``; target
    ``landau.<coef>`` -> the deck value plus a zero-mean field of rms ``sigma`` (deck units).
    ``kind="from_file"``: an npz array; target ``E_ext`` -> shape (3, *space), taken as ALREADY
    IN DECK UNITS (1 deck field unit = 1 V/nm); target ``landau.<coef>`` -> shape *space, the
    coefficient itself in deck units.  Targets ``T0`` and ``eps_star`` are not built.
  * ``space`` = (Ny, Nx) for pf2d, (Nz, Ny, Nx) for pf3d and film.
  * E3b additions.  Temperature is a run-time variable: ``set_T(b, T)`` rebuilds alpha (registry deck at T, or the
    inline law a1 = si_to_deck_landau(slope * (T - T0(r)))) and, for a Langevin run, kT and the OU operators
    (driver B ``Model.set_T``).  ``[[field]] target "T0"`` (needs the slope law): a gaussian field adds
    ``sigma_K * unit field`` to T0_K, as many fields as listed, in order; from_file adds the (K) array.
    ``E_ext`` gaussian with ``seeds = [s1, s2, s3]``: component c is
    ``si_to_deck_field((sigma_kV_cm * unit_field(seed_c)) * 1e5)`` -- driver B's order of operations (the
    ``seed`` form keeps ``rms = si_to_deck_field(sigma * 1e5)`` handed to the generator; the two differ in the
    last bits).  ``field_arrays`` of the built dict hold every field's array by index: run_output saves them in
    the run directory and a resume passes them back as ``saved`` (nothing is regenerated).
  * ``geometry.slab_thickness_nm`` on pf3d at Nz = 1 is Lz (driver B's Dz_nm).
  * ``material.gradient_si = { isotropic = beta_SI }``: G11 = g, G12 = 0, G44 = G44' = g / 2, g =
    units.si_to_deck_gradient(beta_SI).
"""
import hashlib
import os

import numpy as np

from fftpf import backend_config, input_validation, mat_registry, run_input, units
from fftpf.run_adapter import Adapter
from fftpf.run_input import ConfigError


def _err(path, reason):
    raise ConfigError(f"{path}: {reason}")


def file_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_array(path, key, where):
    """Array ``key`` of the npz file ``path`` (a ConfigError at ``where`` if it is absent)."""
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            _err(f"{where}.key", f"{key!r} is not an array of {path}; it has {sorted(data.files)}")
        return np.array(data[key])


# --------------------------------------------------------------------------- cheap refusals

def precheck(cfg):
    """Refusals that need no built array: stage keys the schema allows but this version does not build."""
    for i, fd in enumerate(cfg["field"]):
        if fd["target"] == "eps_star":
            _err(f"field[{i}].target", f"{fd['target']!r} is not built in this version")


def base_T(cfg):
    """The temperature the engine objects are first built at: material.temperature_K, else the first stage's T."""
    T = cfg["material"].get("temperature_K")
    if T is not None:
        return T
    st = cfg["stage"][0]
    return st["rungs"][0][0] if st["kind"] == "cool" else st.get("T_K")


def stage_temperatures(cfg):
    """Every temperature the stages visit (build-time checks run over all of them)."""
    out = []
    for st in cfg["stage"]:
        out += [r[0] for r in st["rungs"]] if st["kind"] == "cool" else ([st["T_K"]] if "T_K" in st else [])
    return sorted(set(out))


# --------------------------------------------------------------------------- material

def material(cfg, T=None):
    """Deck dict (registry layout) at temperature T (default base_T): registry name, params/*.toml path, or inline tables."""
    m = cfg["material"]
    T = base_T(cfg) if T is None else T
    try:
        if "deck" in m:
            d = m["deck"]
            got = (mat_registry.from_toml(d) if d.endswith(".toml") or os.sep in d
                   else mat_registry.get(d, **({} if T is None else {"T": T})))
            problem = run_input.deck_problem(d, got)
            if problem:
                _err("material.deck", problem)
            return got
        flat = {}
        if "landau" in m:
            flat.update(m["landau"])
        else:
            flat.update({k: units.si_to_deck_landau(v) for k, v in m["landau_si"].items() if not isinstance(v, dict)})
            law = m["landau_si"]["alpha1"]
            if isinstance(law, dict):
                flat["alpha1"] = units.si_to_deck_landau(law["slope"] * (T - law["T0_K"]))
        if "gradient_si" in m:
            flat.update(G110=units.si_to_deck_gradient(m["gradient_si"]["isotropic"]),
                        G11_G110=1.0, G12_G110=0.0, G44_G110=0.5, G44P_G110=0.5)
            secs = (("elastic", ("C11", "C12", "C44")), ("electrostriction", ("Q11_phys", "Q12_phys", "Q44_phys")))
        else:
            secs = (("gradient", ("G110", "G11_G110", "G12_G110", "G44_G110", "G44P_G110")),
                    ("elastic", ("C11", "C12", "C44")), ("electrostriction", ("Q11_phys", "Q12_phys", "Q44_phys")))
        for sec, need in secs:
            for k in need:
                if k not in m[sec]:
                    _err(f"material.{sec}.{k}", "required with an inline material")
            flat.update(m[sec])
        meta = dict(source="inline material of the run input file", T=T)
        return input_validation.deck(mat_registry._flat_to_deck(flat, meta), where="material (inline)")
    except ConfigError:
        raise
    except (KeyError, ValueError, OSError) as exc:        # ValidationError is a ValueError
        _err("material.deck" if "deck" in m else "material", str(exc).strip("'\""))


def make_alpha(cfg, parts, deck, T):
    """The alpha dict at temperature T: the deck's Landau coefficients (registry decks regenerated at T), a1 from the
    inline law with T0(r), then the landau.<coef> field arrays."""
    m = cfg["material"]
    law = m.get("landau_si", {}).get("alpha1")
    if "deck" in m and not m["deck"].endswith(".toml") and os.sep not in m["deck"]:
        deckT = material(cfg, T)
        for sec in ("gradient", "elastic", "electrostriction"):
            if deckT[sec] != deck[sec]:
                _err("material.deck", f"the {sec} sector of {m['deck']!r} changes with T; the operators are built once "
                     "(only the Landau coefficients may depend on temperature)")
        alpha = dict(deckT["landau"])
    else:
        alpha = dict(deck["landau"])
    if isinstance(law, dict):
        T0 = law["T0_K"] if parts["T0"] is None else parts["T0"]
        alpha["a1"] = units.si_to_deck_landau(law["slope"] * (T - T0))
    for key, (mode, arr) in parts["landau"].items():
        alpha[key] = alpha.get(key, 0.0) + arr if mode == "add" else arr
    return alpha


def set_T(b, T):
    """Move the built objects to temperature T (driver B ``Model.set_T``): alpha again; for a Langevin run kT and the OU
    operators again (Adapter.build).  A no-op at the current T."""
    if T == b["T"]:
        return
    cfg, ad = b["cfg"], b["adapter"]
    if b["law"]:
        ad.alpha = make_alpha(cfg, b["parts"], b["deck"], T)
    if b["kT"] is not None:
        b["kT"] = units.kT_langevin(T, b["grid"]["hx"], b["grid"]["hy"], b["hz"])
        ad.build(b["dt"], scheme=cfg["numerics"]["scheme"], kT=b["kT"], noise_cutoff_nm=b["noise_cutoff_nm"])
    b["T"] = T


# --------------------------------------------------------------------------- geometry / operators / elastic

def _kappa(cfg, deck):
    eps_b = cfg["physics"]["electrostatics"].get("eps_b")
    if eps_b is not None:
        return units.relative_to_deck_permittivity(eps_b)
    return deck["kappa_b"] if "kappa_b" in deck else units.relative_to_deck_permittivity(10.0)


def _space(cfg):
    n = cfg["geometry"]["N"]
    return (n[1], n[0]) if cfg["geometry"]["engine"] == "pf2d" else (n[2], n[1], n[0])


def geometry(cfg, deck, dry=False):
    """(grid, ops, el, film_dict, kappa); ops / film as the Adapter wants them for the engine.

    The refusals come first (they need no array); ``dry`` stops there and returns (None, None, None, None, kappa)."""
    from fftpf import elastic_base, film_elastic, film_engine, pf2d_engine, pf3d_engine
    g, ph = cfg["geometry"], cfg["physics"]
    eng, N, cell = g["engine"], g["N"], g["cell_nm"]
    e = ph["elastic"]
    kappa = _kappa(cfg, deck)
    C, Qp = deck["elastic"], deck["electrostriction"]
    k0 = e["k0"]
    sub = None
    if eng == "pf2d" and k0 == "clamped":
        _err("physics.elastic.k0", "clamped is not available for engine 'pf2d' (elastic_base has only the stress-free mode)")
    if eng == "pf3d" and k0 == "off" and e["z_mode"] == "plane_stress":
        _err("physics.elastic.z_mode", "plane_stress needs elasticity; physics.elastic.k0 is 'off'")
    if eng == "film":
        if k0 != "stress_free":
            _err("physics.elastic.k0", f"{k0!r} is not available for engine 'film' (misfit film on a substrate stack)")
        if "substrate" in e:
            T = cfg["material"].get("temperature_K")
            try:
                sub = mat_registry.get(e["substrate"], **({} if T is None else {"T": T}))["elastic"]
            except (KeyError, ValueError) as exc:
                _err("physics.elastic.substrate", str(exc).strip("'\""))
    if dry:
        return None, None, None, None, kappa
    if eng == "pf2d":
        grid = pf2d_engine.make_grid(Nx=N[0], Ny=N[1], Lx=N[0] * cell[0], Ly=N[1] * cell[1])
        ops = pf2d_engine.build_operators(grid, **deck["gradient"], kappa=kappa)
        el = None if k0 == "off" else elastic_base.setup_elastic(
            grid, C["C11"], C["C12"], C["C44"], Qp["Q11_phys"], Qp["Q12_phys"], Qp["Q44_phys"])
        return grid, ops, el, None, kappa
    if eng == "pf3d":
        Lz = g["slab_thickness_nm"] if N[2] == 1 and "slab_thickness_nm" in g else N[2] * cell[2]   # Nz = 1: Dz
        grid = pf3d_engine.make_grid_3d(Nx=N[0], Ny=N[1], Nz=N[2], Lx=N[0] * cell[0],
                                        Ly=N[1] * cell[1], Lz=Lz)
        depol = {} if "depol_N" not in ph["electrostatics"] else {"depol_N": ph["electrostatics"]["depol_N"]}
        ops = pf3d_engine.build_operators_3d(grid, **deck["gradient"], kappa=kappa, **depol)
        if k0 == "off":
            return grid, ops, None, None, kappa
        zm = {} if e["z_mode"] == "generalized_plane_strain" else {"z_mode": e["z_mode"]}
        el = pf3d_engine.setup_elastic_3d(
            grid, C=(C["C11"], C["C12"], C["C44"]),
            Q=(Qp["Q11_phys"], Qp["Q12_phys"], Qp["Q44_phys"] / 4.0), k0=k0, **zm)    # Q44_phys/4: tensor trap, casekit.build_3d
        return grid, ops, el, None, kappa
    # film (cases/bto_film_biaxial.setup)
    grid = film_engine.make_grid_3d(Nx=N[0], Ny=N[1], Lx=N[0] * cell[0], Ly=N[1] * cell[1],
                                    Nz=N[2], hf=g["film_thickness_nm"])
    stack = film_elastic.make_stack_grid(grid, hs_dx=e["substrate_layers"])
    el = film_elastic.setup_film_elastic(
        stack, C["C11"], C["C12"], C["C44"], Qp["Q11_phys"], Qp["Q12_phys"], Qp["Q44_phys"],
        e0=e["misfit_strain_pct"] / 100.0, C_substrate=sub)
    film = dict(G=deck["gradient"]["G11"], kappa_b=kappa)
    return grid, None, el, film, kappa


# --------------------------------------------------------------------------- fields

def _field_array(cfg, fd, p, grid, dry=False):
    """The array of one [[field]] (generated or read), before it is combined with the others.  ``dry``: a gaussian field is
    not generated (None); a from_file array is read and its shape checked."""
    eng, space = cfg["geometry"]["engine"], _space(cfg)
    target = fd["target"]
    if fd["kind"] == "gaussian":
        from fftpf import disorder_fields
        if eng == "film":
            _err(f"{p}.kind", "gaussian fields are not built for engine 'film'; use kind = \"from_file\"")
        if dry:
            return None
        gen = disorder_fields.random_field_gaussian if eng == "pf2d" else disorder_fields.random_field_gaussian_3d
        xi = fd["correlation_nm"]
        if target == "E_ext" and "seeds" in fd:          # driver B: (sigma * unit field) * 1e5, then x 1e-9
            return np.stack([units.si_to_deck_field(fd["sigma_kV_cm"] * gen(grid, 1.0, xi, s) * 1e5) for s in fd["seeds"]])
        if target == "E_ext":
            rms = units.si_to_deck_field(fd["sigma_kV_cm"] * 1e5)      # 1 kV/cm = 1e5 V/m
            return np.stack([gen(grid, rms, xi, fd["seed"] + c) for c in range(3)])
        if target == "T0":
            return fd["sigma_K"] * gen(grid, 1.0, xi, fd["seed"])
        return gen(grid, fd["sigma"], xi, fd["seed"])
    arr = load_array(fd["file"], fd["key"], p)
    want = (3,) + space if target == "E_ext" else space
    if arr.shape != want:
        unit = " (deck units, 1 deck field unit = 1 V/nm)" if target == "E_ext" else ""
        _err(f"{p}.file", f"array {fd['key']!r} has shape {arr.shape}, the grid needs {want}{unit}")
    return arr


def fields(cfg, grid, saved=None, dry=False):
    """parts = dict(landau={key: ("add"|"set", array)}, T0=array or None, E_static=zero tuple or the summed E_ext field,
    arrays={index: array}).  ``saved`` (a resume): the per-field arrays of an earlier build, used instead of generating.
    ``dry`` (``check``): gaussian fields are not generated and count as zero; every refusal is still raised."""
    space = _space(cfg)
    landau, E_sum, T0, taken, arrays = {}, None, None, {}, {}
    law = cfg["material"].get("landau_si", {}).get("alpha1")
    for i, fd in enumerate(cfg["field"]):
        p = f"field[{i}]"
        target = fd["target"]
        arr = _field_array(cfg, fd, p, grid, dry) if saved is None else saved[f"field_{i}"]
        arrays[i] = arr
        if target == "E_ext":
            if arr is not None:
                E_sum = arr if E_sum is None else E_sum + arr
        elif target == "T0":
            if arr is not None:
                T0 = (np.full(space, float(law["T0_K"])) if T0 is None else T0) + arr
        else:
            key = mat_registry._ALPHA_TO_ENGINE[target.split(".", 1)[1]]
            if target in taken:
                _err(f"{p}.target", f"{target!r} is already set by field[{taken[target]}]; give one field per Landau coefficient")
            taken[target] = i
            landau[key] = ("add", 0.0) if arr is None else ("add" if fd["kind"] == "gaussian" else "set", arr)
    return dict(landau=landau, T0=T0, E_static=(0.0, 0.0, 0.0) if E_sum is None else E_sum, arrays=arrays)


# --------------------------------------------------------------------------- start state

def make_start(adapter, start, where, space):
    """P0 for a ``start`` table: random (the engine's generator) or from_file (an npz array (3, *space))."""
    if start["kind"] == "random":
        return adapter.random_start(start["amplitude"], start["kmax_frac"], start["seed"])
    if start["kind"] == "zero":                  # P = 0 (cases/wp13_debye_case.py)
        return np.zeros((3,) + space)
    if start["kind"] == "white":                 # driver B start_state: N(0, std) per component, one generator
        return np.random.default_rng(start["seed"]).normal(0.0, start["std"], (3,) + space)
    P = load_array(start["file"], start["key"], where + ".start")
    if P.shape != (3,) + space:
        _err(f"{where}.start.file", f"array {start['key']!r} has shape {P.shape}, the state needs {(3,) + space}")
    return P


def thread_line(settings):
    env = " ".join(f"{k}={v}" for k, v in settings["environment"].items())
    return (f"backend={settings['backend']} fused={settings.get('fused', False)} workers={settings['workers']} "
            f"fft_workers={settings['fft_workers']} elastic_workers={settings['elastic_workers']} | {env}")


# --------------------------------------------------------------------------- build

def check(cfg):
    """The refusals of ``run`` that come before its first step, without arrays or steps: ``build(cfg, dry=True)``."""
    build(cfg, dry=True)


def build(cfg, saved=None, dry=False):
    """Build everything a run needs from the resolved config; no step is taken.  ``saved``: the per-field arrays of
    the run directory being resumed (``field_<i>`` keys), used instead of regenerating the quenched fields.

    ``dry`` (the CLI ``check`` and ``explain``): the same path up to the point where arrays, the backend, the adapter or
    a step would be needed -- grid, operators, elastic setup, gaussian fields, the backend, the random start and the
    ``dt = "verified"`` probe are skipped; every refusal that does not need them is raised, from_file arrays are read.
    Returns None."""
    from fftpf import timestep_verify
    precheck(cfg)
    g, nu, ph = cfg["geometry"], cfg["numerics"], cfg["physics"]
    eng, T = g["engine"], base_T(cfg)
    deck = material(cfg)
    langevin = any(st.get("langevin") for st in cfg["stage"])
    if langevin and eng == "film":
        gr = deck["gradient"]                   # the film's gradient is G11 alone: G12 = 0, G44 = G44p = G11 / 2
        iso = (abs(gr["G12"]) <= 1e-9 * abs(gr["G11"]) and abs(gr["G44"] - gr["G11"] / 2) <= 1e-9 * abs(gr["G11"])
               and abs(gr["G44p"] - gr["G11"] / 2) <= 1e-9 * abs(gr["G11"]))
        if gr["G11"] < 0 or not iso:
            _err("material.gradient", "film Langevin needs an isotropic gradient with G11 > 0 (G12 = 0, "
                 f"G44 = G44p = G11 / 2); this deck has G11={gr['G11']}, G12={gr['G12']}, "
                 f"G44={gr['G44']}, G44p={gr['G44p']}")
    grid, ops, el, film, kappa = geometry(cfg, deck, dry)
    parts = fields(cfg, grid, saved, dry)
    E_static = parts["E_static"]
    alpha = make_alpha(cfg, parts, deck, T)
    law = _has_law(cfg)
    if law:
        for Tk in stage_temperatures(cfg):       # a refusal belongs before the first step
            make_alpha(cfg, parts, deck, Tk)
    for i, st in enumerate(cfg["stage"]):               # every start file is checked before the first step
        if st["start"]["kind"] == "from_file":
            make_start(None, st["start"], f"stage[{i}]", _space(cfg))
    if dry:
        return None
    settings = backend_config.configure(
        nu["threads"], "numpy" if nu["backend"] == "numpy" else "numba", el=el if eng == "film" else None,
        fused=nu["backend"] == "fused")                 # el selects the FILM elastic backend; pf2d / pf3d elastic params are no film el
    adapter = Adapter(eng, grid, alpha, L=nu["L"], ops=ops, film=film, elastic=el)
    kT, hz = None, g["slab_thickness_nm"] if eng == "pf2d" and "slab_thickness_nm" in g else grid.get("hz")
    if langevin:
        kT = units.kT_langevin(T, grid["hx"], grid["hy"], hz)
    dt, info = nu["dt"], None
    if dt == "verified":
        P0 = make_start(adapter, cfg["stage"][0]["start"], "stage[0]", _space(cfg))
        engine = {"pf2d": "pf2d_engine", "pf3d": "pf3d_engine", "film": "film_engine"}[eng]
        module = __import__("fftpf." + engine, fromlist=[engine])
        vops = ops if eng != "film" else {"G11": film["G"], "kappa_b": kappa}
        dt, info = timestep_verify.verified_dt(
            module, P0, alpha, vops, grid, L=nu["L"], E_ext=E_static, T_total=nu["dt_verify_time"],
            elastic_params=el, scheme=nu["scheme"])
    cut = ph["noise"]["cutoff_nm"]
    adapter.build(dt, scheme=nu["scheme"], kT=kT, noise_cutoff_nm=(cut or None) if kT is not None else None)
    return dict(cfg=cfg, deck=deck, grid=grid, ops=ops, el=el, film=film, kappa=kappa, alpha=alpha,
                E_static=E_static, adapter=adapter, dt=dt, dt_info=info, kT=kT, T=T, hz=hz, law=law, parts=parts,
                field_arrays=parts["arrays"],
                noise_cutoff_nm=(cut or None), space=_space(cfg), settings=settings,
                thread_line=thread_line(settings))


def _has_law(cfg):
    """True when alpha depends on T (inline slope law or a registry deck); False for a fixed-temperature deck."""
    from fftpf import run_input
    return run_input._law(cfg) is not None


# --------------------------------------------------------------------------- explain

_ENGINE_FUNCS = {
    "pf2d": ("pf2d_engine", "make_grid, build_operators, build_Minv[2], step[_sbdf2], build_ou_operators, step_langevin, total_energy"),
    "pf3d": ("pf3d_engine", "make_grid_3d, build_operators_3d, build_Minv_3d, step_3d[_sbdf2_3d], build_ou_operators_3d, step_langevin_3d, total_energy_3d"),
    "film": ("film_engine + film_langevin + film_elastic", "make_grid_3d, poisson_factor, build_grad_factors, step[_sbdf2], build_ou_operators_film, step_langevin_film, total_energy_3d, make_stack_grid, setup_film_elastic"),
}


def explain(cfg):
    """Text: which engine functions the file selects, the dt / kT choices known without stepping, the stages."""
    g, ph, nu, m = cfg["geometry"], cfg["physics"], cfg["numerics"], cfg["material"]
    eng, T = g["engine"], base_T(cfg)
    mod, funcs = _ENGINE_FUNCS[eng]
    L = [f"run {cfg['run']['name']!r} seed {cfg['run']['seed']}", f"engine {eng}: fftpf.{mod}", f"  calls: {funcs}"]
    if "deck" in m:
        L.append(f"material: {m['deck']} -> mat_registry.{'from_toml' if m['deck'].endswith('.toml') else 'get'}"
                 f"(T={T})")
    else:
        L.append("material: inline tables -> mat_registry._flat_to_deck (landau_si via units.si_to_deck_landau"
                 + ("; a1 = slope * (T - T0(r)) re-evaluated at every temperature" if isinstance(m.get("landau_si", {}).get("alpha1"), dict) else "") + ")")
    eb = ph["electrostatics"].get("eps_b")
    L.append("electrostatics: kappa_b = " + ("the deck's kappa_b (inline: eps_b = 10)" if eb is None else
             f"units.relative_to_deck_permittivity({eb}) = {units.relative_to_deck_permittivity(eb):.10g}")
             + (f"; depol_N = {ph['electrostatics']['depol_N']} (build_operators_3d)" if "depol_N" in ph["electrostatics"] else ""))
    e = ph["elastic"]
    if eng == "film":
        L.append(f"elastic: film_elastic.setup_film_elastic, misfit {e['misfit_strain_pct']} %, substrate "
                 f"{e.get('substrate', 'same stiffness as film')}, {e['substrate_layers']} layers")
    elif e["k0"] == "off":
        L.append("elastic: off")
    else:
        L.append(f"elastic: {'elastic_base.setup_elastic' if eng == 'pf2d' else 'pf3d_engine.setup_elastic_3d'}"
                 f" k0={e['k0']}" + (f" z_mode={e['z_mode']}" if eng == "pf3d" else ""))
    cell = g["cell_nm"]
    langevin = any(st.get("langevin") for st in cfg["stage"])
    if langevin:
        if eng == "pf2d":
            hz = g["slab_thickness_nm"]
        elif eng == "pf3d":
            hz = g.get("slab_thickness_nm", cell[2]) if g["N"][2] == 1 else cell[2]
        else:
            hz = g["film_thickness_nm"] / (g["N"][2] - 1)
        kT = units.kT_langevin(T, cell[0], cell[1], hz)
        cut = ph["noise"]["cutoff_nm"]
        L.append(f"noise: kT = units.kT_langevin({T}, {cell[0]}, {cell[1]}, {hz:.6g}) = {kT:.6g}; "
                 f"cutoff {'none' if not cut else str(cut) + ' nm'}")
    else:
        L.append("noise: no Langevin stage")
    L.append("dt: " + (f"timestep_verify.verified_dt(T_total={nu['dt_verify_time']}, scheme={nu['scheme']}), "
                        "value known after the verification run" if nu["dt"] == "verified" else f"{nu['dt']} (fixed)"))
    L.append(f"numerics: scheme {nu['scheme']}, L {nu['L']}, backend_config.configure({nu['threads']}, {nu['backend']!r})")
    if cfg["run"]["replicas"] > 1:
        L.append(f"replicas: {cfg['run']['replicas']} complete runs, seeds {cfg['run']['seed']}..{cfg['run']['seed'] + cfg['run']['replicas'] - 1}, "
                 "in rep_<r>/ one after another; the top directory holds the pooled spectrum")
    if "phase_fractions" in cfg["measure"]:
        from fftpf import run_measure
        L.append(run_measure.describe(cfg["measure"]["phase_fractions"]))
    for i, fd in enumerate(cfg["field"]):
        L.append(f"field[{i}]: {fd['kind']} -> {fd['target']}")
    for i, st in enumerate(cfg["stage"]):
        k = st["kind"]
        if k in ("relax", "anneal"):
            L.append(f"stage[{i}] {k}: start {st['start']['kind']}, stop {st['stop']}"
                     + (f", langevin {st['langevin']}, sample_every {st['sample_every']}" if k == "anneal" else ""))
            continue
        what = {"cool": lambda: f"rungs {st['rungs']}",
                "hold": lambda: f"{st['n_steps']} steps, E {st['E_kV_cm']} kV/cm, ramp {st['ramp_steps']} steps",
                "pe_loop": lambda: f"{st['waveform']} to {st['E_max_kV_cm']} kV/cm along {st['axis']}",
                "ac": lambda: (f"{len(st['omega'])} tones omega {st['omega']} (rad per deck time), {st['periods']} periods "
                               f"({st['discard_periods']} discarded), E0 {st['E0_kV_cm']} kV/cm + bias {st['dc_bias_kV_cm']} "
                               f"along {st['axis']}: dielectric_spectrum.lockin_chi, chi_deck = amplitude of mean P / E0"),
                "fdt": lambda: (f"{st['burn_steps']} burn + {st['n_steps']} steps, nperseg {st['nperseg']}, max_lag {st['max_lag']}: "
                                "dielectric_spectrum.welch_chi_pp + sample_correlation + fl_chi with kT x N_cells")}.get(k, lambda: "")()
        L.append(f"stage[{i}] {k}: start {st['start']['kind']}, {what}"
                 + (f", T_K {st['T_K']}" if "T_K" in st else "") + (f", langevin {st['langevin']}" if "langevin" in st else "")
                 + (", own seed" if "seed" in st else ", carried generator" if st.get("langevin") else ""))
    return "\n".join(L)
