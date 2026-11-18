#!/usr/bin/env /opt/miniconda3/bin/python3
"""SPEC S41.C / S41.1 A3(ii) (WP29) -- submit / status / tail / kill for one
long computation, over a plain JSON registry. No daemon, stdlib only.

    python3 tools/job.py submit cases/x.py [--name N] [--nice 10] [--threads 1] [-- ARGS...]
    python3 tools/job.py status [ID|all]
    python3 tools/job.py tail   ID [-n 40] [-f]
    python3 tools/job.py kill   ID

Replaces the hand-typed `nohup nice -n 10 env OMP_NUM_THREADS=1 ... python3
case.py > reports/x.log 2>&1 & disown` of ENGINE_GUIDE S1.3, and fixes two
measured faults of it (REVIEW_wp29_inventory.md A3(ii)):

  * 0 of 24 existing reports/*.log say which script, commit or thread
    settings produced them. Every job here opens with a 9-field header
    (8 until SPEC S V2-01.1(b) added the `engine` field: the name+version
    of the engine that produced the log, read from solvers.ENGINE_NAME /
    solvers.ENGINE_VERSION, never retyped here).
  * `pkill -f case.py` misses an orphaned scipy.fft worker pool (the WP19
    incident recorded in cases/archive/.../wp19 prose: PPID=1, ~14 cores,
    outliving its parent). Here the case runs in its OWN SESSION
    (start_new_session=True) and `kill` signals the whole PROCESS GROUP, so
    the pool goes with it.

MECHANISM, no daemon anywhere: `submit` writes reports/jobs/<id>.json, then
spawns a detached wrapper (`job.py _run <id>`) with the log as its stdout.
The wrapper records its own pid/pgid, prints the header, starts the case as
its child with the built environment (os.environ + execution.environment limits +
PYTHONPATH=<repo root> + FFT_RUN_ID=<id>) and `nice` applied in the child
via preexec_fn, waits, and writes ended/exit/state back into the JSON.
`status` needs no daemon either: for a job whose JSON still says running it
probes the whole process group, os.killpg(pgid, 0) -- not just the wrapper,
which dies at the first SIGTERM while a case that ignores SIGTERM lives on
in the group. Records are written atomically (temp file + os.replace), so a
concurrent status/kill never reads a half-written JSON.

RUN FOLDER: reports/runs/<id>/ by default, or --run-root <parent>/<id>,
with the log at run.log -- the same
convention solvers/casekit.run_dir() uses. They agree because FFT_RUN_ID is
exported into the child alongside FFT_RUN_ROOT: a case that calls casekit.run_dir() under a job
gets THAT id's folder and writes its params.json/npz/png next to the log.
Run outputs are gitignored by default (SPEC S41.1 A3(iii)); a deliverable
figure is copied out of the run folder explicitly by its author.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from solvers import ENGINE_NAME, ENGINE_VERSION   # plain constants: no numpy, no engine module
from solvers.execution import environment

JOBS_DIR = os.path.join(_ROOT, "reports", "jobs")
RUNS_DIR = os.path.join(_ROOT, "reports", "runs")
# The fourth is Apple Accelerate's VECLIB_MAXIMUM_THREADS, not *_NUM_THREADS:
# Accelerate ignores the latter, so a job that set it left the pool unlimited
# (WP29 C6). Same four names gates/wp27_profiling exports.
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
STAMP = "%Y-%m-%dT%H:%M:%S"


def _now():
    return time.strftime(STAMP)


def _job_path(job_id):
    return os.path.join(JOBS_DIR, job_id + ".json")


def _read(job_id):
    path = _job_path(job_id)
    if not os.path.exists(path):
        sys.exit(f"job.py: no such job: {job_id}")
    with open(path) as fh:
        return json.load(fh)


def _write(rec):
    """Dump to a temp file in JOBS_DIR, then os.replace it over the record:
    a reader sees the old JSON or the new one, never a truncated one. The
    temp name does not end in .json, so `status all` never lists it."""
    os.makedirs(JOBS_DIR, exist_ok=True)
    path = _job_path(rec["id"])
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _git(*args):
    """git output, or "" when git or the .git dir is unavailable."""
    try:
        return subprocess.run(["git", "-C", _ROOT, *args],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def _alive(rec):
    """True while ANY process of the job's process group exists: the wrapper,
    the case, or a worker pool the case forked. The wrapper alone is not
    evidence -- it dies at the first SIGTERM, while a case that ignores
    SIGTERM survives it, reparented (PPID 1), in the same group. pgid/pid
    None = just submitted, the wrapper has not recorded them yet -- not
    evidence of death; PermissionError = the group exists but is not ours
    to signal -- alive."""
    pgid = rec["pgid"] or rec["pid"]
    if not pgid:
        return True
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _refresh(job_id):
    """Re-read the JSON and, if it still claims `running` while the whole
    process group is gone, close the record out. That happens when the
    wrapper was itself signalled (a process-group kill from outside) and
    never got to write its own ending. A record whose wrapper is gone but
    whose case still runs in the group stays `running`, so `kill` can
    still reach it."""
    rec = _read(job_id)
    if rec["state"] == "running" and not _alive(rec):
        rec.update(state="killed", ended=rec["ended"] or _now())
        _write(rec)
    return rec


def _elapsed(rec):
    start = time.mktime(time.strptime(rec["started"], STAMP))
    end = time.mktime(time.strptime(rec["ended"], STAMP)) if rec["ended"] else time.time()
    secs = int(end - start)
    return f"{secs // 3600}h{secs // 60 % 60:02d}m" if secs >= 3600 else f"{secs // 60}m{secs % 60:02d}s"


def _last_line(rec, width=48):
    path = os.path.join(_ROOT, rec["log"])
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    return lines[-1][:width] if lines else ""


def submit(args, passthrough):
    script = os.path.relpath(os.path.abspath(args.script), _ROOT)
    if not os.path.exists(os.path.join(_ROOT, script)):
        sys.exit(f"job.py: no such script: {args.script}")
    name = args.name or os.path.splitext(os.path.basename(script))[0]
    job_id = f"{time.strftime('%Y%m%d-%H%M')}_{name}"
    dup = 1
    while os.path.exists(_job_path(job_id)):   # two submits in one minute
        dup += 1
        job_id = f"{time.strftime('%Y%m%d-%H%M')}_{name}-{dup}"
    run_root = os.path.abspath(args.run_root) if args.run_root else RUNS_DIR
    run_dir = os.path.join(run_root, job_id)
    os.makedirs(run_dir, exist_ok=True)
    rec = dict(id=job_id, script=script, argv=[script] + passthrough,
               pid=None, pgid=None, started=_now(), ended=None, exit=None,
               state="running", log=os.path.relpath(os.path.join(run_dir, "run.log"), _ROOT),
               run_dir=os.path.relpath(run_dir, _ROOT) + "/",
               commit=_git("rev-parse", "--short", "HEAD") or "unknown",
               dirty=bool(_git("status", "--porcelain")), nice=args.nice,
               env=environment(args.threads))
    _write(rec)
    with open(os.path.join(_ROOT, rec["log"]), "a") as log:
        subprocess.Popen([sys.executable, os.path.abspath(__file__), "_run", job_id],
                         cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    print(f"submitted {job_id}\n  log      {rec['log']}\n"
          f"  status   python3 tools/job.py status {job_id}")


def run_wrapper(job_id):
    """The detached wrapper (`job.py _run ID`). Its stdout IS the log."""
    rec = _read(job_id)
    rec["pid"], rec["pgid"] = os.getpid(), os.getpgid(0)
    _write(rec)
    threads = " ".join(f"{v.split('_')[0]}={rec['env'][v]}" for v in THREAD_ENV)
    # 9 fields (SPEC S V2-01.1(b)): the commit row carries a third column with
    # the engine name+version, so a log says WHICH engine produced it.
    for row in (("job     " + rec["id"], f"commit  {rec['commit']} (dirty: {'yes' if rec['dirty'] else 'no'})",
                 f"engine  {ENGINE_NAME} {ENGINE_VERSION}"),
                ("script  " + rec["script"], f"started {rec['started']}"),
                ("argv    " + " ".join(rec["argv"][1:]) if rec["argv"][1:] else "argv    -",
                 f"threads {threads} | nice {rec['nice']}"),
                ("python  " + sys.executable, f"run_dir {rec['run_dir']}")):
        line = f"# {row[0]:<39} # {row[1]:<37}" if len(row) > 2 else f"# {row[0]:<39} # {row[1]}"
        print(line + (f" # {row[2]}" if len(row) > 2 else ""))
    sys.stdout.flush()

    def _renice():
        os.nice(rec["nice"])

    env = dict(os.environ, PYTHONPATH=_ROOT, FFT_RUN_ID=job_id,
               FFT_RUN_ROOT=os.path.dirname(os.path.join(_ROOT, rec["run_dir"].rstrip("/"))),
               **rec["env"])
    child = subprocess.Popen([sys.executable, rec["script"], *rec["argv"][1:]],
                             cwd=_ROOT, env=env, preexec_fn=_renice)
    code = child.wait()
    rec = _read(job_id)          # re-read: `kill` may have written to it meanwhile
    rec["ended"], rec["exit"] = _now(), code
    rec["state"] = "done" if code == 0 else ("killed" if code < 0 else "failed")
    _write(rec)
    print(f"# ended   {rec['ended']}  exit {code}  state {rec['state']}  elapsed {_elapsed(rec)}")


def status(args):
    if args.job in (None, "all"):
        ids = (sorted(f[:-5] for f in os.listdir(JOBS_DIR) if f.endswith(".json"))
               if os.path.isdir(JOBS_DIR) else [])
    else:
        ids = [args.job]
    if not ids:
        print("no jobs in reports/jobs/")
        return
    print(f"{'ID':<30} {'STATE':<8} {'ELAPSED':>8}  {'RUN_DIR':<38} LAST LOG LINE")
    for job_id in ids:
        rec = _refresh(job_id)
        print(f"{rec['id']:<30} {rec['state']:<8} {_elapsed(rec):>8}  "
              f"{rec['run_dir']:<38} {_last_line(rec)}")


def tail(args):
    rec = _read(args.job)
    with open(os.path.join(_ROOT, rec["log"]), errors="replace") as fh:
        lines = fh.readlines()                                 # leaves fh at EOF
        sys.stdout.write("".join(lines[-args.n:] if args.n > 0 else []))   # lines[-0:] is ALL
        while args.follow:
            chunk = fh.read()
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            elif _refresh(args.job)["state"] != "running":
                return
            else:
                time.sleep(1.0)


def kill(args):
    rec = _refresh(args.job)
    if rec["state"] != "running":
        sys.exit(f"job.py: {rec['id']} is already {rec['state']}")
    pgid = rec["pgid"] or rec["pid"]
    os.killpg(pgid, signal.SIGTERM)      # the GROUP: wrapper, case, and any
    for _ in range(20):                  # FFT worker pool the case forked
        time.sleep(0.5)
        if not _alive(rec):              # the group, not the wrapper pid
            break
    else:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:       # the group went in the last half second
            pass
        time.sleep(0.5)
    rec = _read(args.job)
    if rec["state"] == "running":
        rec.update(state="killed", ended=_now())
        _write(rec)
    print(f"{rec['id']}: {rec['state']} (process group {pgid})")


def main():
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:                     # everything after `--` goes to the case,
        cut = argv.index("--")           # split off before argparse sees it
        argv, passthrough = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(prog="tools/job.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("submit")
    p.add_argument("script")
    p.add_argument("--name", help="job name; default = the script's basename")
    p.add_argument("--nice", type=int, default=10)
    p.add_argument("--threads", type=int, default=1, help="OMP/Numba worker ceiling; BLAS remains single-threaded")
    p.add_argument("--run-root", help="parent directory for results and run.log; default reports/runs")
    sub.add_parser("status").add_argument("job", nargs="?", help="job id, or 'all' (default)")
    p = sub.add_parser("tail")
    p.add_argument("job")
    p.add_argument("-n", type=int, default=40)
    p.add_argument("-f", "--follow", action="store_true")
    sub.add_parser("kill").add_argument("job")
    sub.add_parser("_run").add_argument("job")
    args = ap.parse_args(argv)
    if args.cmd == "submit":
        submit(args, passthrough)
    elif args.cmd == "status":
        status(args)
    elif args.cmd == "tail":
        tail(args)
    elif args.cmd == "kill":
        kill(args)
    else:
        run_wrapper(args.job)


if __name__ == "__main__":
    main()
