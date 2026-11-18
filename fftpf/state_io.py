"""WP3 §6.2(d): state IO. .npz save/load (P, eps_star_static, disorder alpha
arrays, params provenance, step, t); quick-look PNG (P components + |P|);
energy CSV is already handled by solver.relax()'s history list (callers write
it with plain csv/np.savetxt, no new helper needed here). VTK: minimal
hand-rolled legacy ASCII STRUCTURED_POINTS writer -- zero new dependencies
(SPEC: "VTK only if zero new deps").

NPZ FORMAT v2 (SPEC V2-02.1 E8/E9/E10, adopted 2026-09-16 from the codex
branch, ADAPTED). A v2 file is a v1 file plus three things:

  * `format_version = 2`, and the geometry keys v1 dropped (`Nz`, `Lz`,
    `hx`, `hy`, `hz`) whenever the grid carries them -- v1 stored only
    Nx/Ny/Lx/Ly, so a 3D snapshot could not state its own z geometry
    (claim C of the 2026-09-10 assessment, verified TRUE);
  * `metadata_json`, a JSON tree of grid/params/alpha/restart/fields whose
    array leaves are `{"array": "payload_<n>"}` references into ordinary
    sidecar arrays of the SAME npz -- no pickle, allow_pickle stays off;
  * `restart`, the continuation certificate (RNG bit-generator state, the
    SBDF2 previous field, step/t/dt/scheme), which is what lets
    casekit.resume_anneal continue a run bit-for-bit.

TOP-LEVEL `alpha_<key>` KEEPS MAIN'S RULE (V2-02.1 E8): only ARRAY-valued
(disordered) coefficients go there (np.ndim > 0); scalar alphas live in the
metadata tree, so an old reader's key set stays predictable.

NEVER RAISES ON METADATA (E10): a value that cannot be stored losslessly --
a set, a NaN/inf float, an object array, a custom object -- is stored as its
`str()`, the same fallback `json.dump(..., default=str)` already gives every
run's params.json. A finished run is never lost to one unserializable knob;
the restart tree itself is numeric, so nothing a restart needs degrades.

`load_state_npz` reads BOTH versions: a v1 file returns exactly what it
always returned.
"""
import numpy as np

_JSON_SCALARS = (str, int, float, bool)


def _pack(value, arrays):
    """Encode one metadata value as a JSON-able tree, moving arrays out into
    `arrays` (the npz kwargs) as `payload_<n>` sidecars. Lossless for
    numbers/strings/arrays and the dict/list/tuple nestings of them; anything
    else degrades to str(value) instead of aborting the save (E10)."""
    import os
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return str(value)                 # cannot live in an allow_pickle=False npz
        key = f"payload_{len(arrays)}"        # len() only grows -> the name is unique
        arrays[key] = value
        return {"array": key}
    if isinstance(value, np.generic):
        return _pack(value.item(), arrays)
    if isinstance(value, dict):
        return {"dict": [[k if k is None or isinstance(k, _JSON_SCALARS) else str(k),
                          _pack(v, arrays)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {"tuple" if isinstance(value, tuple) else "list":
                [_pack(v, arrays) for v in value]}
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)                     # allow_nan=False cannot hold nan/inf
    if value is None or isinstance(value, _JSON_SCALARS):
        return value
    return str(value)


def _unpack(value, arrays):
    """Inverse of _pack: rebuild the tree, resolving `payload_<n>` refs
    against the arrays actually present in the file."""
    if not isinstance(value, dict):
        return value
    if "array" in value:
        return arrays[value["array"]]
    if "dict" in value:
        return {k: _unpack(v, arrays) for k, v in value["dict"]}
    if "tuple" in value:
        return tuple(_unpack(v, arrays) for v in value["tuple"])
    return [_unpack(v, arrays) for v in value["list"]]


def save_state_npz(path, P, grid, step, t, prm=None, alpha=None, eps_star_static=None,
                   *, restart=None, fields=None):
    """Save full state to a single .npz (format v2, module docstring): P,
    the complete grid geometry, step, t, the disorder arrays, the params
    provenance (deck_path/params_path) and the metadata tree.

    restart: the continuation certificate casekit.anneal(return_state=True)
        returns (step/t/trace/observations/P_previous/rng_state/model).
        Pass it and casekit.resume_anneal(path) continues the SAME
        recurrence: without P_previous an SBDF2 run restarts through an
        SBDF1 bootstrap step, and without rng_state a Langevin run draws a
        different noise sequence -- a P-only snapshot is not a restart.
    fields: extra fields worth keeping (a quenched E_d, a q/theta pair).

    The write is ATOMIC -- a temp file in the same directory, then
    os.replace -- so an interrupted save cannot destroy the checkpoint it
    overwrites. numpy appends ".npz" only when handed a FILENAME, never a
    file object, so this re-appends it itself (V2-02.1 E9)."""
    import json
    import os
    import tempfile
    kwargs = dict(P=np.asarray(P), step=np.asarray(step), t=np.asarray(t),
                  format_version=np.asarray(2))
    for key in ("Nx", "Ny", "Nz", "Lx", "Ly", "Lz", "hx", "hy", "hz"):
        if key in grid:
            kwargs[key] = np.asarray(grid[key])
    if eps_star_static is not None:
        kwargs["eps_star_static"] = eps_star_static
    if alpha is not None:
        for k, v in alpha.items():
            if np.ndim(v) > 0:  # only the array (disordered) coefficients are worth saving
                kwargs[f"alpha_{k}"] = np.asarray(v)
    if prm is not None:
        for key in ("deck_path", "params_path"):
            kwargs[key] = np.asarray(str(prm.get(key, "")))
    payload = dict(grid=grid, params=prm, alpha=alpha, restart=restart, fields=fields)
    kwargs["metadata_json"] = np.asarray(json.dumps(_pack(payload, kwargs), allow_nan=False))
    target = os.fspath(path)
    if not target.endswith(".npz"):
        target += ".npz"
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(target)), suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **kwargs)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_state_npz(path):
    """Inverse of save_state_npz -- returns a plain dict of whatever arrays
    were saved. A v1 file (no metadata_json) returns exactly what it always
    returned; a v2 file additionally carries `checkpoint`, the decoded
    grid/params/alpha/restart/fields tree."""
    import json
    with np.load(path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    if "metadata_json" in data:
        version = int(data.get("format_version", 0))
        if version != 2:
            raise ValueError(f"unsupported checkpoint version {version}")
        data["checkpoint"] = _unpack(json.loads(str(data["metadata_json"])), data)
    return data


def provenance(input_paths=()):
    """Versions and SHA256 of inputs plus the current solver source (dirty
    edits included) -- the stamp casekit.save_run puts in params.json next
    to the git commit, so a run folder says what CODE produced it even when
    the tree was not clean."""
    import hashlib
    import platform
    from importlib.metadata import version
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    paths = set(root.joinpath("fftpf").glob("*.py")) | {Path(p).resolve() for p in input_paths}
    hashes = {}
    for path in sorted(paths):
        label = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(python=platform.python_version(), platform=platform.platform(),
                numpy=version("numpy"), scipy=version("scipy"),
                sympy=version("sympy"), input_sha256=hashes)


def save_quicklook_png(path, P, grid, title=""):
    """Quick-look PNG: P1, P2, P3, |P| side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Pmag = np.sqrt((P**2).sum(axis=0))
    panels = [("P1", P[0]), ("P2", P[1]), ("P3", P[2]), ("|P|", Pmag)]
    vmax = float(np.max(np.abs(P))) or 1.0
    extent = [0, grid["Lx"], 0, grid["Ly"]]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (label, field) in zip(axes, panels):
        cmap, vmin, vm = ("viridis", 0.0, Pmag.max() or 1.0) if label == "|P|" else ("RdBu_r", -vmax, vmax)
        im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vm)
        ax.set_title(label)
        ax.set_xlabel("x (nm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("y (nm)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_vtk(path, field_dict, grid):
    """Minimal legacy VTK ASCII STRUCTURED_POINTS writer (no new deps).
    field_dict: {name: (Ny,Nx) or (Ny,Nx,1) real array} -- SCALARS per name."""
    Nx, Ny = grid["Nx"], grid["Ny"]
    hx, hy = grid["hx"], grid["hy"]
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("FFT-spectral state\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {Nx} {Ny} 1\n")
        f.write("ORIGIN 0 0 0\n")
        f.write(f"SPACING {hx} {hy} 1\n")
        f.write(f"POINT_DATA {Nx * Ny}\n")
        for name, field in field_dict.items():
            arr = np.asarray(field).reshape(Ny, Nx)
            f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            # VTK structured-points ordering: x fastest, then y
            f.write("\n".join(" ".join(f"{v:.8e}" for v in row) for row in arr))
            f.write("\n")
