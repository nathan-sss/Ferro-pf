"""The 2026-11 materials archive: decks and registry entries no computation uses.

Pins four things: the archived TOMLs live in params/archive/ and still load
through materials.from_toml; materials.get on an archived name raises a
KeyError that points at solvers.materials_archive; materials_archive.get
returns exactly the pre-move values; and the snapshot writers still
reproduce the committed archived files.
"""
import ast
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from solvers import literature_params, materials, materials_archive, pmn_field  # noqa: E402

PARAMS = os.path.join(_ROOT, "params")
ARCHIVE = os.path.join(PARAMS, "archive")
ARCHIVED_TOMLS = ("bto_BC1984_6th", "bto_LCC2005_8th", "pto_Haun1987", "pzt_Haun_x0.6",
                  "pmn_clean_x0", "pmn_pt_landau_x0.30", "pmn_disorder_R3",
                  "pzt_r3c_g22e_model", "pzt_r3c_haun_x02")


@pytest.mark.parametrize("stem", ARCHIVED_TOMLS)
def test_archived_toml_lives_in_archive_and_loads(stem):
    assert not os.path.exists(os.path.join(PARAMS, stem + ".toml"))
    path = os.path.join(ARCHIVE, stem + ".toml")
    deck = materials.from_toml(path)
    assert deck["meta"]["source_toml"] == f"params/archive/{stem}.toml"


def test_archived_registry_names_point_to_materials_archive():
    names = set(materials_archive.ARCHIVED_REGISTRY)
    assert names == {"bto_bc1984_6th", "bfo_pan2019", "bto_pan2019", "sto_pan2019",
                     "pmn_clean_x0", "pmn_pt_placeholder"}
    assert set(materials._ARCHIVED) == names
    assert not names & set(materials._REGISTRY)
    for name in names:
        with pytest.raises(KeyError, match=r"archived.*solvers\.materials_archive\.get"):
            materials.get(name)
    with pytest.raises(KeyError, match="unknown material"):
        materials.get("no_such_material")
    with pytest.raises(KeyError, match="not an archived entry"):
        materials_archive.get("bto_lcc2005_8th")
    assert [n for n, _d in materials_archive.list()] == sorted(names)


def test_archive_get_reproduces_pre_move_values():
    # literal outputs of materials.get(<name>) before the move (T = 298 K)
    bc = materials_archive.get("bto_bc1984_6th", T=298.0)
    assert bc["landau"]["a111"] == pytest.approx(8.004, rel=1e-12)
    assert bc["landau"] == {"a1": -0.027722, "a11": -0.6475500000000001, "a12": 0.323,
                            "a111": 8.004000000000001, "a112": 4.470000000000001,
                            "a123": 4.9190000000000005,
                            "a1111": 0.0, "a1112": 0.0, "a1122": 0.0, "a1123": 0.0}
    assert bc["elastic"] == {"C11": 275.0, "C12": 179.0, "C44": 54.3}
    assert bc["kappa_b"] == 0.08854187
    assert materials_archive.get("bto_bc1984_6th", T=118.0)["landau"]["a111"] == 17.94
    assert materials_archive.bto_bc1984_6th_legacy(T=298.0)["alpha111"] == 8.004000000000001

    grad = {"G11": 0.08447999999999999, "G12": 0.0,
            "G44": 0.04223999999999999, "G44p": 0.04223999999999999}
    bto = materials_archive.get("bto_pan2019")
    assert bto["landau"] == {"a1": -0.027722, "a11": -0.6475500000000001, "a12": 0.323,
                             "a111": 5.5200000000000005, "a112": 4.470000000000001, "a123": 4.91}
    assert bto["gradient"] == grad and bto["kappa_b"] == 0.8854187
    assert materials_archive.get("bfo_pan2019")["landau"] == {
        "a1": -0.39445, "a11": 0.65, "a12": 0.1, "a111": 0.0, "a112": 0.0, "a123": 0.0}
    assert materials_archive.get("sto_pan2019")["landau"] == {
        "a1": 0.184619, "a11": 1.7000000000000002, "a12": 1.37, "a111": 0.0, "a112": 0.0,
        "a123": 0.0}

    pmn = materials_archive.get("pmn_clean_x0", T=298.0)
    assert pmn["landau"] == {"a1": -0.04705871114054246}
    assert pmn["landau"]["a1"] == pmn_field.pmn_clean_alpha1(298.0)
    assert set(materials_archive.get("pmn_pt_placeholder")) == {"meta"}


def test_archived_tomls_agree_with_their_generators():
    for stem, name in (("bto_BC1984_6th", "bto_bc1984_6th"),
                       ("pmn_clean_x0", "pmn_clean_x0"),
                       ("pmn_pt_landau_x0.30", "pmn_pt_placeholder")):
        toml = materials.from_toml(os.path.join(ARCHIVE, stem + ".toml"))
        gen = materials_archive.get(name)
        assert {k: v for k, v in toml.items() if k != "meta"} == \
               {k: v for k, v in gen.items() if k != "meta"}, stem
    for stem, name, kw in (("bto_LCC2005_8th", "bto_lcc2005_8th", {}),
                           ("pto_Haun1987", "pto_haun1987", {}),
                           ("pzt_Haun_x0.6", "pzt_haun_x", {"x": 0.6})):
        toml = materials.from_toml(os.path.join(ARCHIVE, stem + ".toml"))
        gen = materials.get(name, **kw)
        assert {k: v for k, v in toml.items() if k != "meta"} == \
               {k: v for k, v in gen.items() if k != "meta"}, stem


def test_snapshot_writers_reproduce_committed_files(tmp_path):
    literature_params.generate_all_toml(str(tmp_path))
    materials_archive.generate_archive_toml(str(tmp_path / "archive"))
    written = sorted(os.path.relpath(os.path.join(d, f), tmp_path)
                     for d, _s, fs in os.walk(tmp_path) for f in fs)
    assert written == ["archive/bto_BC1984_6th.toml", "archive/bto_LCC2005_8th.toml",
                       "archive/pto_Haun1987.toml", "archive/pzt_Haun_x0.6.toml",
                       "bto_LCC2005_paper.toml"]
    extra = {"archive/pmn_pt_landau_x0.30.toml": materials_archive.pmn_pt_landau_placeholder()}
    flat, meta = materials_archive._pmn_clean_x0_flat(T=298.0)
    extra["archive/pmn_clean_x0.toml"] = dict(flat, _meta=meta)
    for rel, params in extra.items():
        literature_params.write_toml(params, str(tmp_path / rel))
        written.append(rel)
    for rel in written:
        text = (tmp_path / rel).read_text()
        committed = open(os.path.join(PARAMS, rel)).read()
        assert text[text.index("[landau]"):] == committed[committed.index("[landau]"):], rel


def test_nothing_in_the_engine_imports_the_archive():
    roots = [os.path.join(_ROOT, d) for d in ("fftpf", "cases", "Runs", "tools")]
    offenders = []
    for root in roots:
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ("archive", "__pycache__")]
            for f in files:
                path = os.path.join(dirpath, f)
                if not f.endswith(".py") or path.endswith(os.path.join("fftpf", "mat_registry_archive.py")):
                    continue
                tree = ast.parse(open(path, encoding="utf-8").read())
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [a.name for a in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module or ""] + [a.name for a in node.names]
                    if any(n.split(".")[-1] == "materials_archive" for n in names):
                        offenders.append(os.path.relpath(path, _ROOT))
    assert offenders == []
