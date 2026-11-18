"""RUN-ENTRY packages E3a / E3b (SPEC § 16.2): the run directory, checkpoints and resume.

Files: resolved.toml (the input with every default; re-running it reproduces the run),
provenance.json, trace.csv, state_<i>_<kind>.npz per relax / anneal stage end and
state_<i>_<kind>_<event>.npz per event of cool / hold / pe_loop, measures.json, quicklook.png
(written last, the mid-z plane for 3D and film), quenched.npz (one array per [[field]], written once),
figures/<state file stem>_<phase|direction>.png (measure.domain_map, written right after the state file),
checkpoint.npz + checkpoint_prev.npz (when output.checkpoint_every > 0).

Spectrum stages (E3c): series_<i>_ac_w<omega:.6g>.npz / series_<i>_fdt.npz (the RAW series s = sum(P . n) with dt, omega,
E0, kT, N_cells, cell_volume_nm3, axis), spectrum_<i>_<kind>.npz (the per-volume estimator output of the stage), and with
measure.spectrum: spectrum.csv + figures/spectrum.png (``write_spectrum``).  run.replicas = R > 1 (run_protocol.run_replicas):
the top directory holds resolved.toml, replicas.json, the pooled spectrum.csv / figure / measures.json, and rep_<r>/, each a
complete run directory of its own.

``output.dir = "auto"`` is reports/runs/<YYYYMMDD-HHMM>_<run.name>/ with casekit.run_dir's
rules re-implemented here (casekit is not imported): FFT_RUN_ID replaces the folder name,
FFT_RUN_ROOT replaces reports/runs, a second folder of the same minute gets ``_2``, ``_3``.
Any other value is the directory itself (relative to the working directory); it must be new or empty,
or hold a checkpoint of the SAME resolved config (everything but ``output``, plus the sha256 of every
from_file input), which resumes the run.  A checkpoint of another config is a ConfigError.

trace.csv rows are appended as they are produced and flushed at every checkpoint and stage end; the checkpoint
records the file's byte count and a resume cuts the file back to it.  Floats are written with repr, so the
numbers read back exactly (a resumed stage rebuilds its sample list from the file).

checkpoint.npz (written atomically: temp file, fsync, rename; the previous one is kept as checkpoint_prev.npz):
arrays ``P`` and ``P_prev`` (a size-0 array when the stage has none: SBDF1 / first step) and ``meta``, a JSON
string with identity (the config hash), stage_index, stage_step (steps done in that stage; 0 = at a stage
boundary), step0 (steps of all finished stages), t, E / n / E_entry (the applied field state: kV/cm, unit axis, field
on stage entry), rng (bit-generator state), trace_bytes, summaries (one dict per finished stage) and fractions_at (the
phase fractions of the event states the running stage has saved so far, {event: {...}}).
"""
import csv
import hashlib
import json
import os
from fftpf._locations import default_run_root
import subprocess
import time

import numpy as np

from fftpf import ENGINE_NAME, ENGINE_VERSION, run_builder, run_input, run_measure, state_io
from fftpf.run_input import ConfigError

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NOT_TERMS = ("E_total", "E_kV_cm")
_FRACTION_COLUMNS = tuple(f"f_{k}" for k in dict.fromkeys(run_measure.FRACTION_KEYS + run_measure.FRACTION_KEYS_RTO))   # either scheme


class RunInterrupted(Exception):
    """Raised by the ``stop_after_checkpoints`` test hook right after a checkpoint has been written."""


def git_commit():
    """HEAD hash, with "+dirty" appended when the tree has changes, "unknown" without git."""
    # An installed wheel can live inside another project's virtual environment.
    if not os.path.exists(os.path.join(_ROOT, ".git")):
        return "unknown"
    try:
        out = subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", _ROOT, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"
    return (out + "+dirty") if (out and dirty) else (out or "unknown")


def identity(cfg):
    """sha256 of the resolved config without ``output``, plus the sha256 of every from_file input."""
    files = [fd["file"] for fd in cfg["field"] if fd["kind"] == "from_file"]
    files += [st["start"]["file"] for st in cfg["stage"] if st["start"]["kind"] == "from_file"]
    core = {k: v for k, v in cfg.items() if k != "output"}
    text = json.dumps([core, [(f, run_builder.file_sha256(f)) for f in files]], sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def existing_dir(cfg):
    """The output directory when it is already named and present (a resume candidate), else None."""
    name = cfg["output"]["dir"]
    if name == "auto":
        run_id = os.environ.get("FFT_RUN_ID")
        if not run_id:
            return None
        path = os.path.join(os.environ.get("FFT_RUN_ROOT", default_run_root()), run_id)
    else:
        path = os.path.abspath(name)
    return path if os.path.isdir(path) and os.listdir(path) else None


def save_npz(path, **arrays):
    """Atomic: a temporary file, fsync, then rename."""
    temp = path + ".tmp"
    with open(temp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp, path)


def find_checkpoint(cfg, ident):
    """None, or dict(dir, P, P_prev, meta) of the newest readable checkpoint of this config in output.dir."""
    d = existing_dir(cfg)
    if d is None:
        return None
    for name in ("checkpoint.npz", "checkpoint_prev.npz"):
        path = os.path.join(d, name)
        if not os.path.isfile(path):
            continue
        try:
            with np.load(path) as z:
                P, prev, meta = np.array(z["P"]), np.array(z["P_prev"]), json.loads(str(z["meta"]))
        except Exception as exc:                      # a kill during a save can truncate the newest file
            print(f"unreadable {path}: {exc!r}", flush=True)
            continue
        if meta["identity"] != ident:
            raise ConfigError(f"output.dir: {d} holds a checkpoint of a different configuration "
                              f"({meta['identity'][:12]} != {ident[:12]}); use a new directory")
        return dict(dir=d, P=P, P_prev=prev if prev.size else None, meta=meta)
    return None


def load_quenched(d):
    """The per-field arrays saved by an earlier build in ``d`` ({'field_<i>': array}), or None if no field was saved."""
    path = os.path.join(d, "quenched.npz")
    if not os.path.isfile(path):
        return None
    with np.load(path) as z:
        return {k: np.array(z[k]) for k in z.files}


def _target(cfg):
    """(path the run directory would have, what names it) -- auto: the folder of the minute, which may get a suffix."""
    name = cfg["output"]["dir"]
    if name != "auto":
        return os.path.abspath(name), "output.dir"
    root = os.environ.get("FFT_RUN_ROOT", default_run_root())
    return root, "FFT_RUN_ROOT" if "FFT_RUN_ROOT" in os.environ else "the default reports/runs"


def check_dir(cfg):
    """ConfigError (output.dir) when the run directory cannot be created or written; creates nothing.

    The nearest existing ancestor of the path must be a directory the process can write and enter (an existing run
    directory must itself be writable)."""
    path, what = _target(cfg)
    probe = path
    while not os.path.lexists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    if not os.path.isdir(probe):
        raise ConfigError(f"output.dir: cannot create {path} ({what}): {probe} is not a directory")
    if not os.access(probe, os.W_OK | os.X_OK):
        raise ConfigError(f"output.dir: cannot create {path} ({what}): {probe} is not writable")


def _makedirs(path, cfg, exist_ok=False):
    try:
        os.makedirs(path, exist_ok=exist_ok)
    except OSError as exc:
        _, what = _target(cfg)
        raise ConfigError(f"output.dir: cannot create {path} ({what}): {exc.strerror or exc}") from None


def make_dir(cfg):
    """Create and return the run directory named by cfg["output"]["dir"] (a ConfigError on output.dir when it cannot be made)."""
    name = cfg["output"]["dir"]
    if name == "auto":
        run_id = os.environ.get("FFT_RUN_ID")
        root = os.environ.get("FFT_RUN_ROOT", default_run_root())
        if run_id:
            path = os.path.join(root, run_id)
            _makedirs(path, cfg, exist_ok=True)
            return path
        base = os.path.join(root, f"{time.strftime('%Y%m%d-%H%M')}_{cfg['run']['name']}")
        path, k = base, 2
        while os.path.exists(path):
            path, k = f"{base}_{k}", k + 1
        _makedirs(path, cfg)
        return path
    path = os.path.abspath(name)
    if os.path.isdir(path) and os.listdir(path):
        raise ConfigError(f"output.dir: {path} exists and is not empty; a run never overwrites "
                          "(it resumes only when it holds a checkpoint of the same configuration)")
    _makedirs(path, cfg, exist_ok=True)
    return path


def sample_from_row(row):
    """A sample dict (the form run_protocol makes) from one trace.csv row (strings)."""
    s = dict(step=int(row["step"]), step_total=int(row["step_total"]), t=float(row["t"]), T_K=float(row["T_K"]),
             E_kV_cm=float(row["E_kV_cm"]), Pn=float(row["Pn"]), E_total=float(row["E_total"]), f_total=float(row["f_total"]))
    if "Px" in row:
        s["mean_P"] = [float(row[c]) for c in ("Px", "Py", "Pz")]
    s.update({k: float(v) for k, v in row.items() if k in _FRACTION_COLUMNS})
    terms = {k[2:]: float(v) for k, v in row.items() if k.startswith("E_") and k not in _NOT_TERMS}
    if terms:
        s["terms"] = terms
    return s


class RunDir:
    """The files of one run; created after the build, filled stage by stage.

    resume: the dict find_checkpoint returned (the directory is reused and trace.csv cut back);
    stop_after: the test hook, RunInterrupted after that many checkpoints written by this object."""

    def __init__(self, cfg, b, input_path, ident, resume=None, stop_after=None):
        self.cfg, self.b, self.ident, self.rows = cfg, b, ident, []
        self.stop_after, self.n_checkpoints, self.cols = stop_after, 0, None
        self.trace, self.event_fractions, self.no_matplotlib = None, {}, False
        self.maps = {t["stem"] for t in run_measure.domain_map_targets(cfg)} if cfg["measure"].get("domain_map") else set()
        if resume is not None:
            self.path = resume["dir"]
            self.trace = os.path.join(self.path, "trace.csv")
            self._cut(resume["meta"]["trace_bytes"])
            return
        self.path = make_dir(cfg)
        self.trace = os.path.join(self.path, "trace.csv")
        with open(os.path.join(self.path, "resolved.toml"), "w") as fh:
            fh.write(run_input.dumps(cfg))
        arrays = {}
        for i, fd in enumerate(cfg["field"]):
            if fd["kind"] == "from_file":
                arrays[f"field[{i}]"] = (fd["file"], run_builder.file_sha256(fd["file"]))
        for i, st in enumerate(cfg["stage"]):
            if st["start"]["kind"] == "from_file":
                arrays[f"stage[{i}].start"] = (st["start"]["file"], run_builder.file_sha256(st["start"]["file"]))
        inputs = ([input_path] if input_path else []) + [v[0] for v in arrays.values()]
        info = b["dt_info"]
        rec = dict(
            provenance=state_io.provenance(inputs), commit=git_commit(),
            engine=f"{ENGINE_NAME} {ENGINE_VERSION}", thread_line=b["thread_line"], config_sha256=ident,
            input_file=input_path, input_file_sha256=run_builder.file_sha256(input_path) if input_path else None,
            from_file_arrays={k: dict(file=v[0], sha256=v[1]) for k, v in arrays.items()},
            seed=cfg["run"]["seed"],
            fields=[{k: fd[k] for k in ("target", "kind", "seed", "seeds") if k in fd} for fd in cfg["field"]],
            dt=b["dt"], dt_verified=None if info is None else dict(agree_relerr=info["agree_relerr"], T_total=info["T_total"]),
            kT=b["kT"], noise_cutoff_nm=b["noise_cutoff_nm"])
        self._json("provenance.json", rec)
        if b["field_arrays"]:                        # quenched fields: saved once, reloaded by a resume
            save_npz(os.path.join(self.path, "quenched.npz"), **{f"field_{i}": a for i, a in b["field_arrays"].items()})

    def _json(self, name, obj):
        with open(os.path.join(self.path, name), "w") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=str)

    # -- trace.csv ----------------------------------------------------------
    def _cut(self, nbytes):
        """Cut trace.csv back to ``nbytes`` (the checkpoint's) and read its header."""
        if not os.path.isfile(self.trace):
            if nbytes:
                raise ConfigError(f"output.dir: {self.trace} is missing but the checkpoint expects {nbytes} bytes of it")
            return
        if os.path.getsize(self.trace) < nbytes:
            raise ConfigError(f"output.dir: {self.trace} is shorter than the checkpoint's {nbytes} bytes")
        with open(self.trace, "r+b") as fh:
            fh.truncate(nbytes)
        if nbytes:
            with open(self.trace, newline="") as fh:
                self.cols = next(csv.reader(fh))

    def add_row(self, i, kind, s):
        """Queue the trace row of sample ``s`` of stage ``i`` (written at the next flush)."""
        row = dict(stage=i, kind=kind, step=s["step"], step_total=s["step_total"], t=s["t"], T_K=s["T_K"],
                   E_kV_cm=s["E_kV_cm"], Pn=s["Pn"])
        if "mean_P" in s:
            row.update(zip(("Px", "Py", "Pz"), s["mean_P"]))
        row.update(E_total=s["E_total"], f_total=s["f_total"])
        row.update({f"E_{k}": v for k, v in s.get("terms", {}).items()})
        row.update({k: v for k, v in s.items() if k in _FRACTION_COLUMNS})
        self.rows.append(row)

    def flush_trace(self):
        """Append the queued rows (header first in an empty file), fsync, and return the file size in bytes."""
        if self.rows:
            if self.cols is None:
                self.cols = list(self.rows[0])
            fresh = not os.path.isfile(self.trace) or os.path.getsize(self.trace) == 0
            with open(self.trace, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=self.cols)
                if fresh:
                    w.writeheader()
                w.writerows(self.rows)
                fh.flush()
                os.fsync(fh.fileno())
            self.rows = []
        return os.path.getsize(self.trace) if os.path.isfile(self.trace) else 0

    def stage_samples(self, i):
        """The samples of stage ``i`` already in trace.csv (a resumed stage rebuilds its list from them)."""
        if not os.path.isfile(self.trace) or os.path.getsize(self.trace) == 0:
            return []
        with open(self.trace, newline="") as fh:
            return [sample_from_row(r) for r in csv.DictReader(fh) if int(r["stage"]) == i]

    # -- states -------------------------------------------------------------
    def _save_state(self, name, P, step, t, prm, restart):
        b = self.b
        state_io.save_state_npz(os.path.join(self.path, name), P, b["grid"], step=step, t=t, prm=prm,
                                alpha=b["adapter"].alpha, restart=restart)

    def _model(self):
        b = self.b
        return dict(engine=b["adapter"].engine, dt=b["dt"], L=b["adapter"].L, scheme=b["adapter"].scheme,
                    kT=b["kT"], noise_cutoff_nm=b["noise_cutoff_nm"])

    def write_stage(self, i, res):
        """relax / anneal: queue the stage's readings, flush trace.csv, save the end state.
        cool / hold / pe_loop (res["streamed"]): their rows and event states were written as they happened."""
        kind = res["kind"]
        for name, meta in (res.get("series") or {}).items():
            save_npz(os.path.join(self.path, f"series_{i}_{name}.npz"), **meta)
        for route, arrays in (res.get("spectrum") or {}).items():
            save_npz(os.path.join(self.path, f"spectrum_{i}_{kind}.npz"), **arrays)
        if not res.get("streamed"):
            for s in res["samples"]:
                self.add_row(i, kind, s)
        self.flush_trace()
        if res.get("streamed"):
            return
        restart = dict(step=res["step"], t=res["t"], trace=res["trace"], P_previous=res["P_previous"],
                       rng_state=res["rng_state"], stop_reason=res["stop_reason"], model=self._model())
        self._save_state(f"state_{i}_{kind}.npz", res["P"], res["step"], res["t"],
                         dict(run=self.cfg["run"]["name"], stage=i, kind=kind), restart)
        self._maps(f"state_{i}_{kind}", res["P"], i, kind, "end", self.b["T"], 0.0)

    def write_event(self, i, kind, name, P, previous, ctx, s, T, E, step, t):
        """A saved state at a named event (cool rung end, hold end, loop extremes)."""
        restart = dict(step=step, t=t, P_previous=previous, rng_state=ctx["rng"].bit_generator.state,
                       T_K=T, E_kV_cm=E, n=ctx["n"], stage_step=s, model=self._model())
        self._save_state(f"state_{i}_{kind}_{name}.npz", P, step, t,
                         dict(run=self.cfg["run"]["name"], stage=i, kind=kind, event=name), restart)
        prm = self.cfg["measure"].get("phase_fractions")
        if prm:
            self.event_fractions.setdefault(i, {})[name] = run_measure.phase_fractions(self.b["adapter"], P, prm)
        self._maps(f"state_{i}_{kind}_{name}", P, i, kind, name, T, E)

    def _maps(self, stem, P, i, kind, event, T, E):
        """The domain maps of the state just saved (measure.domain_map), in figures/<stem>_<kind>.png."""
        if stem not in self.maps:
            return
        from fftpf import run_figures
        os.makedirs(os.path.join(self.path, "figures"), exist_ok=True)
        prm = self.cfg["measure"].get("phase_fractions")
        fr = run_measure.phase_fractions(self.b["adapter"], P, prm) if prm else None
        title = (f"{self.cfg['run']['name']}: stage {i} {kind}, {event}\n"
                 f"T {'-' if T is None else format(T, 'g')} K, E {E:g} kV/cm")
        for what in self.cfg["measure"]["domain_map"]["kind"]:
            try:
                run_figures.domain_map(os.path.join(self.path, "figures", f"{stem}_{what}.png"), what, P, self.b["grid"],
                                       self.b["adapter"].engine, title, prm, fr)
            except ImportError:                      # matplotlib is optional; every number is already written
                if not self.no_matplotlib:
                    print("warning: matplotlib is not installed; domain maps are skipped", flush=True)
                self.no_matplotlib = True
                return

    # -- checkpoint ---------------------------------------------------------
    def checkpoint(self, P, previous, ctx, stage_index, stage_step, summaries):
        """Flush trace.csv, then write checkpoint.npz (the old one becomes checkpoint_prev.npz)."""
        nbytes = self.flush_trace()
        path, prev = os.path.join(self.path, "checkpoint.npz"), os.path.join(self.path, "checkpoint_prev.npz")
        meta = dict(identity=self.ident, stage_index=stage_index, stage_step=stage_step, step0=ctx["step0"],
                    t=(ctx["step0"] + stage_step) * self.b["dt"], E=ctx["E"], n=ctx["n"], E_entry=ctx["E_entry"],
                    rng=ctx["rng"].bit_generator.state, trace_bytes=nbytes, summaries=summaries,
                    fractions_at=self.event_fractions.get(stage_index, {}))
        if os.path.isfile(path):
            os.replace(path, prev)
        if self.b["adapter"].scheme != "sbdf2":          # only the SBDF2 recurrence reads the previous field
            previous = None
        save_npz(path, P=np.asarray(P), P_prev=np.zeros(0) if previous is None else np.asarray(previous),
                 meta=json.dumps(meta))
        self.n_checkpoints += 1
        if self.stop_after is not None and self.n_checkpoints >= self.stop_after:
            raise RunInterrupted(f"stopped after checkpoint {self.n_checkpoints} (stage {stage_index}, step {stage_step})")

    def finish(self, summaries, P):
        """measures.json, then quicklook.png last."""
        b = self.b
        out = dict(dt=b["dt"], kT=b["kT"], stages=summaries)
        if self.cfg["measure"].get("spectrum"):
            out["spectrum"] = write_spectrum(self.path, self.cfg, [self.path])
        self._json("measures.json", out)
        P = np.asarray(P)
        zmid = P.shape[1] // 2 if P.ndim == 4 else None
        try:
            state_io.save_quicklook_png(os.path.join(self.path, "quicklook.png"), P if zmid is None else P[:, zmid],
                                        b["grid"], title=os.path.basename(self.path) + ("" if zmid is None else f"  (z={zmid})"))
        except ImportError:
            pass                                    # matplotlib is optional; every number is already written


# --------------------------------------------------------------------------- spectrum.csv and figure (E3c)

_CSV_COLUMNS = ("omega", "f_Hz", "chi_real", "chi_imag", "eps_real", "eps_imag", "tan_delta")


def load_spectrum(d, cfg):
    """{route: dict of arrays} of the spectrum stages saved in run directory ``d`` (one stage per route; ConfigError if missing)."""
    rec = {}
    for route, kind in (("lockin", "ac"), ("fdt", "fdt")):
        if route not in cfg["measure"]["spectrum"]["routes"]:
            continue
        i = [j for j, st in enumerate(cfg["stage"]) if st["kind"] == kind][0]
        path = os.path.join(d, f"spectrum_{i}_{kind}.npz")
        if not os.path.isfile(path):
            raise ConfigError(f"measure.spectrum: {path} is missing (stage {i} {kind!r} has not finished or did not give a spectrum)")
        with np.load(path) as z:
            rec[route] = {k: np.array(z[k]) for k in z.files}
    return rec


def write_spectrum(path, cfg, dirs):
    """spectrum.csv and figures/spectrum.png in ``path`` from the spectrum stages of the run directories ``dirs`` (one per
    replica); returns the JSON-able info of run_measure.spectrum_tables plus the replica count.

    Rows: route, omega (rad per deck time), f_Hz (only with measure.seconds_per_deck_time), chi_real, chi_imag (deck units,
    per volume), eps_real, eps_imag, tan_delta, and with more than one replica the standard errors <name>_se.  Floats are
    written with repr.  A missing matplotlib skips the figure with one warning."""
    m = cfg["measure"]
    spec = m["spectrum"]
    tables, info = run_measure.spectrum_tables([load_spectrum(d, cfg) for d in dirs], spec["routes"], m.get("seconds_per_deck_time"))
    cols = [c for c in _CSV_COLUMNS if c in next(iter(tables.values()))]
    cols += [c + "_se" for c in cols if c != "omega" and c != "f_Hz" and c + "_se" in next(iter(tables.values()))]
    with open(os.path.join(path, "spectrum.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["route"] + cols)
        for route in spec["routes"]:
            t = tables[route]
            for j in range(len(t["omega"])):
                w.writerow([route] + [repr(float(t[c][j])) for c in cols])
    try:
        from fftpf import run_figures
        os.makedirs(os.path.join(path, "figures"), exist_ok=True)
        run_figures.spectrum(os.path.join(path, "figures", "spectrum.png"), tables, spec["quantities"],
                             f"{cfg['run']['name']}: dielectric spectrum" + (f", {len(dirs)} replicas" if len(dirs) > 1 else ""),
                             m.get("seconds_per_deck_time") is not None)
    except ImportError:                              # matplotlib is optional; every number is already written
        print("warning: matplotlib is not installed; the spectrum figure is skipped", flush=True)
    return dict(replicas=len(dirs), units="chi per volume in deck units; eps and tan_delta dimensionless (eps = eps_b + chi' / eps0_deck)",
                routes=info)
