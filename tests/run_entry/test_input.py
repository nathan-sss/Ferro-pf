"""RUN-ENTRY E1 (SPEC § 16.5): the input layer parses, fills defaults and refuses before any step."""
import copy
import os
import sys
import tomllib

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fftpf import run_input as ri  # noqa: E402
from fftpf.run_input import ConfigError, dumps, load, resolve  # noqa: E402

EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")

BASE = {
    "pf2d": {"engine": "pf2d", "N": [16, 16], "slab_thickness_nm": 1.0},
    "pf3d": {"engine": "pf3d", "N": [8, 8, 8]},
    "film": {"engine": "film", "N": [8, 8, 4], "film_thickness_nm": 20.0},
}
FILM_ELASTIC = {"misfit_strain_pct": -0.5}


def cfg(engine="pf2d", stage=None, **tables):
    """A minimal valid raw config for `engine`, with whole tables replaceable via keyword."""
    raw = {
        "run": {"name": "t"},
        "material": {"deck": "bto_lcc2005_8th", "temperature_K": 300.0},
        "geometry": dict(BASE[engine]),
        "stage": stage or [{"kind": "relax"}],
    }
    if engine == "film":
        raw["physics"] = {"elastic": dict(FILM_ELASTIC)}
    raw.update(copy.deepcopy(tables))
    return raw


def merged(raw, path, value):
    """Return raw with the dotted `path` set (tables created as needed; [i] indexes stage)."""
    raw = copy.deepcopy(raw)
    node = raw
    keys = path.split(".")
    for k in keys[:-1]:
        if k.startswith("stage["):
            node = node["stage"][int(k[6:-1])]
        else:
            node = node.setdefault(k, {})
    node[keys[-1]] = value
    return raw


def refuses(raw, key_path, base_dir="."):
    with pytest.raises(ConfigError) as ei:
        resolve(raw, base_dir)
    assert str(ei.value).startswith(key_path + ":"), str(ei.value)
    return str(ei.value)


ANNEAL = {"kind": "anneal", "stop": {"n_steps": 100}}


# ------------------------------------------------------------------ (1) every refusal has its own test

def test_pf2d_needs_two_numbers():
    refuses(merged(cfg("pf2d"), "geometry.N", [8, 8, 8]), "geometry.N")


def test_pf3d_needs_three_numbers():
    refuses(merged(cfg("pf3d"), "geometry.N", [8, 8]), "geometry.N")


def test_film_needs_three_numbers():
    refuses(merged(cfg("film"), "geometry.N", [8, 8]), "geometry.N")


def test_depol_N_only_pf3d():
    for eng in ("pf2d", "film"):
        refuses(merged(cfg(eng), "physics.electrostatics.depol_N", 0.3), "physics.electrostatics.depol_N")
    assert resolve(merged(cfg("pf3d"), "physics.electrostatics.depol_N", 0.3))["physics"]["electrostatics"]["depol_N"] == 0.3


def test_plane_stress_needs_pf3d_nz1():
    for eng in ("pf2d", "film"):
        refuses(merged(cfg(eng), "physics.elastic.z_mode", "plane_stress"), "physics.elastic.z_mode")
    refuses(merged(cfg("pf3d"), "physics.elastic.z_mode", "plane_stress"), "physics.elastic.z_mode")   # Nz = 8
    ok = merged(merged(cfg("pf3d"), "physics.elastic.z_mode", "plane_stress"), "geometry.N", [8, 8, 1])
    assert resolve(ok)["physics"]["elastic"]["z_mode"] == "plane_stress"


def test_clamped_with_replicas():
    raw = merged(merged(cfg("pf3d"), "physics.elastic.k0", "clamped"), "run.replicas", 2)
    refuses(raw, "physics.elastic.k0")
    assert resolve(merged(cfg("pf3d"), "physics.elastic.k0", "clamped"))["physics"]["elastic"]["k0"] == "clamped"


def test_film_keys_only_on_film():
    for key, val in (("misfit_strain_pct", 0.1), ("substrate", "mgo_cubic_rt"), ("substrate_layers", 6)):
        for eng in ("pf2d", "pf3d"):
            refuses(merged(cfg(eng), f"physics.elastic.{key}", val), f"physics.elastic.{key}")
    refuses(merged(cfg("pf3d"), "geometry.film_thickness_nm", 10.0), "geometry.film_thickness_nm")


def test_film_requires_its_keys():
    raw = cfg("film")
    del raw["geometry"]["film_thickness_nm"]
    refuses(raw, "geometry.film_thickness_nm")
    raw = cfg("film")
    raw["physics"] = {}
    refuses(raw, "physics.elastic.misfit_strain_pct")


def test_replicas_with_film():
    refuses(merged(cfg("film"), "run.replicas", 4), "run.replicas")


def test_langevin_with_sbdf2():
    refuses(merged(cfg("pf3d", stage=[ANNEAL]), "numerics.scheme", "sbdf2"), "stage[0].langevin")
    # deterministic anneal is fine with sbdf2
    raw = merged(merged(cfg("pf3d", stage=[dict(ANNEAL, langevin=False)]), "numerics.scheme", "sbdf2"), "run.seed", 1)
    assert resolve(raw)["numerics"]["scheme"] == "sbdf2"


def test_langevin_pf2d_needs_slab_thickness():
    raw = cfg("pf2d", stage=[ANNEAL])
    del raw["geometry"]["slab_thickness_nm"]
    refuses(raw, "stage[0].langevin")


def test_langevin_needs_temperature():
    raw = cfg("pf3d", stage=[ANNEAL])
    del raw["material"]["temperature_K"]
    assert resolve(raw)["stage"][0]["langevin"] is False          # default follows temperature_K
    refuses(merged(raw, "stage[0].langevin", True), "stage[0].langevin")


def test_slab_thickness_only_pf2d():
    refuses(merged(cfg("pf3d"), "geometry.slab_thickness_nm", 1.0), "geometry.slab_thickness_nm")


def test_deck_and_inline_both():
    inline = {"landau": {"alpha1": -1.0, "alpha11": 1.0, "alpha12": 1.0}}
    refuses(merged(cfg("pf3d"), "material.landau", inline["landau"]), "material.deck")


def test_inline_material_needs_all_tables_and_one_landau():
    mat = {"landau": {"alpha1": -1.0, "alpha11": 1.0, "alpha12": 1.0, "alpha111": 0.0, "alpha112": 0.0, "alpha123": 0.0},
           "gradient": {"G110": 0.1}, "elastic": {"C11": 1.0, "C12": 0.5, "C44": 0.5},
           "electrostriction": {"Q11_phys": 0.1, "Q12_phys": -0.03, "Q44_phys": 0.05}}
    raw = cfg("pf3d", material=mat)
    assert "deck" not in resolve(raw)["material"]
    bad = copy.deepcopy(raw)
    del bad["material"]["gradient"]
    refuses(bad, "material.gradient")
    bad = merged(raw, "material.landau_si", dict(mat["landau"]))
    refuses(bad, "material.landau_si")
    bad = copy.deepcopy(raw)
    del bad["material"]["landau"]
    refuses(bad, "material.landau")
    bad = merged(raw, "material.landau", {"alpha1": -1.0})
    refuses(bad, "material.landau.alpha11")


def test_archived_and_unknown_deck():
    refuses(merged(cfg("pf3d"), "material.deck", "bto_pan2019"), "material.deck")
    msg = refuses(merged(cfg("pf3d"), "material.deck", "bto_lcc2005_8tn"), "material.deck")
    assert "bto_lcc2005_8th" in msg
    refuses(merged(cfg("pf3d"), "material.deck", "no_such.toml"), "material.deck")


def test_unbuilt_stage_kinds(monkeypatch):
    # E3b built cool, hold and pe_loop, E3c ac and fdt (tests/run_entry/test_spectrum.py): the guard is checked as it was before E3c
    monkeypatch.setattr(ri, "BUILT_STAGES", ("relax", "anneal", "cool", "hold", "pe_loop"))
    for st in ({"kind": "ac", "axis": [1, 0, 0], "E0_kV_cm": 1, "omega": [0.1], "periods": 5},
               {"kind": "fdt", "n_steps": 100, "nperseg": 32, "axis": [1, 0, 0]}):
        msg = refuses(cfg("pf3d", stage=[{"kind": "relax"}, st]), "stage[1].kind")
        assert "not built in this version" in msg and st["kind"] in msg


def test_unbuilt_measures():
    raw = cfg("pf3d", measure={"domain_map": {"at": ["relax:end"]}})
    assert resolve(raw)["measure"]["domain_map"]["at"] == ["relax:end"]      # built in E4 (no longer refused)
    # a measure that needs a stage the file lacks is refused as such (decided before 'not built')
    refuses(cfg("pf3d", measure={"loop": ["Pr"]}), "measure.loop")
    refuses(cfg("pf3d", measure={"spectrum": {"quantities": ["eps_real"], "routes": ["lockin"]}}),
            "measure.spectrum.routes")


def test_unbuilt_measure_alone(monkeypatch):
    monkeypatch.setattr(ri, "BUILT_MEASURES", ("mean_P", "energy_terms"))        # as before E3b built the loop measures
    raw = cfg("pf3d", stage=[{"kind": "relax"}, {"kind": "pe_loop", "axis": [1, 0, 0], "E_max_kV_cm": 1,
                                                 "steps_per_cycle": 8}], measure={"loop": ["Pr", "Ec"]})
    msg = refuses(raw, "measure.loop")
    assert "not built in this version" in msg


def test_previous_as_first_start():
    refuses(merged(cfg("pf3d"), "stage[0].start", "previous"), "stage[0].start")


def test_from_file_must_exist(tmp_path):
    start = {"kind": "from_file", "file": "nope.npz", "key": "P"}
    refuses(merged(cfg("pf3d"), "stage[0].start", start), "stage[0].start.file", base_dir=tmp_path)
    (tmp_path / "p.npz").write_bytes(b"x")
    out = resolve(merged(cfg("pf3d"), "stage[0].start", dict(start, file="p.npz")), tmp_path)
    assert out["stage"][0]["start"]["file"] == str(tmp_path / "p.npz")
    field = {"target": "E_ext", "kind": "from_file", "file": "nope.npz", "key": "E"}
    refuses(cfg("pf3d", field=[field]), "field[0].file", base_dir=tmp_path)


def test_deck_path_must_exist(tmp_path):
    refuses(merged(cfg("pf3d"), "material.deck", "params/x.toml"), "material.deck", base_dir=tmp_path)


def test_field_rules():
    g = {"target": "E_ext", "kind": "gaussian", "sigma_kV_cm": 30.0, "correlation_nm": 6.0, "seed": 11}
    out = resolve(cfg("pf3d", field=[g]))
    assert out["field"][0]["seed"] == 11
    refuses(cfg("pf3d", field=[dict(g, target="landau.alpha1")]), "field[0].sigma")
    refuses(cfg("pf3d", field=[dict(g, target="landau.alpha9")]), "field[0].target")
    bad = {k: v for k, v in g.items() if k != "seed"}
    refuses(cfg("pf3d", field=[bad]), "field[0].seed")


# ------------------------------------------------------------------ (2) misspelt keys

def test_misspelt_keys_suggest():
    msg = refuses(merged(cfg("pf3d"), "material.tempreature_K", 300.0), "material.tempreature_K")
    assert "temperature_K" in msg
    msg = refuses(merged(cfg("pf3d"), "numerics.bakend", "numpy"), "numerics.bakend")
    assert "backend" in msg
    msg = refuses(merged(cfg("pf3d"), "physics.noise.cutoff_mn", 2.0), "physics.noise.cutoff_mn")
    assert "cutoff_nm" in msg
    msg = refuses(merged(cfg("pf3d"), "stage[0].stpo", {}), "stage[0].stpo")
    assert "stop" in msg
    msg = refuses(merged(cfg("pf3d"), "geometry.engin", "pf3d"), "geometry.engin")
    assert "engine" in msg
    refuses(dict(cfg("pf3d"), measures={}), "measures")
    msg = refuses(merged(cfg("pf3d"), "stage[0].kind", "relaxx"), "stage[0].kind")
    assert "relax" in msg
    refuses(merged(cfg("pf3d"), "stage[0].tol", 1e-3), "stage[0].tol")      # key of another table level
    refuses(merged(cfg("pf3d"), "geometry.N", [8, 8.5, 8]), "geometry.N[1]")


def test_engine_is_required_and_never_inferred():
    raw = cfg("pf3d")
    del raw["geometry"]["engine"]
    refuses(raw, "geometry.engine")


# ------------------------------------------------------------------ (3) minimal valid files

@pytest.mark.parametrize("engine", ["pf2d", "pf3d", "film"])
def test_minimal_relax_and_anneal(engine):
    out = resolve(cfg(engine))
    g, ph, nu = out["geometry"], out["physics"], out["numerics"]
    assert g["engine"] == engine and g["cell_nm"] == [1.0] * (3 if engine == "pf3d" else 2)
    assert "eps_b" not in ph["electrostatics"]        # absent: the builder takes the deck's kappa_b, else eps_b = 10
    assert ph["noise"]["cutoff_nm"] == 2.0                       # l_c default lands here (§ 2)
    assert ph["elastic"]["k0"] == "stress_free" and ph["elastic"]["z_mode"] == "generalized_plane_strain"
    assert (nu["scheme"], nu["dt"], nu["L"], nu["backend"], nu["threads"]) == ("sbdf1", "verified", 1.0, "numpy", 1)
    assert out["run"] == {"name": "t", "seed": 0, "replicas": 1}
    assert out["stage"][0] == {"kind": "relax", "start": {"kind": "random", "amplitude": 0.005, "kmax_frac": 0.3, "seed": 0},
                               "stop": {"tol": 1e-6, "n_max": 50000}}
    assert out["output"] == {"dir": "auto", "checkpoint_every": 0}
    assert out["measure"] == {"mean_P": True, "energy_terms": False}
    assert out["field"] == []
    if engine == "film":
        assert ph["elastic"]["substrate_layers"] == 12
    an = resolve(cfg(engine, stage=[{"kind": "anneal", "stop": {"block_stationary": 1e-4, "n_max": 500}}, {"kind": "relax"}]))
    s0, s1 = an["stage"]
    assert s0["langevin"] is True and "seed" not in s0 and s0["sample_every"] == 100   # run generator carries on; block rule samples
    assert s0["stop"] == {"block_stationary": 1e-4, "n_max": 500}
    assert s1["start"] == {"kind": "previous"}


def test_anneal_stop_rules():
    refuses(cfg("pf3d", stage=[{"kind": "anneal", "stop": {}}]), "stage[0].stop")
    refuses(cfg("pf3d", stage=[{"kind": "anneal", "stop": {"n_steps": 5, "block_stationary": 1e-3, "n_max": 9}}]), "stage[0].stop")
    refuses(cfg("pf3d", stage=[{"kind": "anneal", "stop": {"block_stationary": 1e-3}}]), "stage[0].stop.n_max")
    refuses(cfg("pf3d", stage=[{"kind": "anneal"}]), "stage[0].stop")


def test_resolve_leaves_input_untouched():
    raw = cfg("pf2d")
    before = copy.deepcopy(raw)
    resolve(raw)
    assert raw == before


# ------------------------------------------------------------------ (4) the two § 16.3 examples

EVERYTHING = ("mean_P", "energy_terms", "loop", "domain_map", "spectrum")


@pytest.mark.parametrize("name,kind,measure", [("pe_loop", "pe_loop", "loop"),
                                               ("dielectric_spectrum", "ac", "spectrum")])
def test_spec_examples(name, kind, measure):
    path = os.path.join(EX, name + ".toml")
    with open(path, "rb") as fh:
        tomllib.load(fh)                                           # syntactically valid TOML
    out = load(path)                                               # both examples load verbatim since E3c (stages) and E4 (measures)
    if name == "pe_loop":
        assert out["stage"][0]["sample_every"] == 100              # block_stationary rule: default sampling
        assert out["stage"][1]["waveform"] == "triangle" and out["measure"]["loop"] == ["Pr", "Ec", "Pmax", "W_rec", "eta"]
    else:
        assert [st["kind"] for st in out["stage"]] == ["anneal", "ac", "fdt"] and out["run"]["replicas"] == 8
        assert out["stage"][1]["omega"] == [0.002, 0.005, 0.01, 0.02, 0.05, 0.1] and out["stage"][2]["max_lag"] == 8192
        assert out["measure"]["spectrum"]["routes"] == ["lockin", "fdt"]
    assert out["geometry"]["engine"] == "pf2d" and out["physics"]["noise"]["cutoff_nm"] == 2.0
    assert os.path.isabs(out["material"].get("deck", "/")) or out["material"]["deck"] == "bto_lcc2005_8th"
    # (5) round trip
    assert resolve(tomllib.loads(dumps(out)), EX) == out


# ------------------------------------------------------------------ (5) round trip

@pytest.mark.parametrize("engine", ["pf2d", "pf3d", "film"])
def test_round_trip_minimal(engine):
    out = resolve(cfg(engine, stage=[{"kind": "anneal", "stop": {"n_steps": 10}}, {"kind": "relax"}],
                      field=[{"target": "landau.alpha1", "kind": "gaussian", "sigma": 0.1, "correlation_nm": 4.0, "seed": 3}]))
    assert resolve(tomllib.loads(dumps(out))) == out


def test_round_trip_inline_and_depol(tmp_path):
    (tmp_path / "start.npz").write_bytes(b"x")
    mat = {"landau_si": {"alpha1": -1.0, "alpha11": 1.0, "alpha12": 1.0, "alpha111": 0.0, "alpha112": 0.0, "alpha123": 0.0},
           "gradient": {"G110": 0.1}, "elastic": {"C11": 1.0, "C12": 0.5, "C44": 0.5},
           "electrostriction": {"Q11_phys": 0.1, "Q12_phys": -0.03, "Q44_phys": 0.05}}
    raw = cfg("pf3d", material=mat, stage=[{"kind": "relax", "start": {"kind": "from_file", "file": "start.npz", "key": "P"}}])
    raw["physics"] = {"electrostatics": {"depol_N": [[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.4]]}}
    raw["numerics"] = {"dt": 0.05}
    out = resolve(raw, tmp_path)
    assert out["numerics"]["dt"] == 0.05 and out["physics"]["electrostatics"]["depol_N"][2][2] == 0.4
    assert resolve(tomllib.loads(dumps(out)), tmp_path) == out


def test_load_resolves_paths_against_file_dir(tmp_path):
    (tmp_path / "a.toml").write_text(
        '[run]\nname="x"\n[material]\ndeck="d.toml"\n[geometry]\nengine="pf3d"\nN=[4,4,4]\n[[stage]]\nkind="relax"\n')
    refuses_cwd = pytest.raises(ConfigError)
    with refuses_cwd as ei:
        load(tmp_path / "a.toml")
    assert str(ei.value).startswith("material.deck:")
    (tmp_path / "d.toml").write_text("")
    assert load(tmp_path / "a.toml")["material"]["deck"] == str(tmp_path / "d.toml")
