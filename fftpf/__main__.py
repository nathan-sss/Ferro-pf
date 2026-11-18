"""``python -m fftpf run|check|explain <case.toml>`` (SPEC § 16, RUN-ENTRY).

run      execute the stages, write the run directory, print its path; an output.dir that holds a checkpoint
         of the same config resumes (exit 3 after --stop-after-checkpoints k, a test hook)
check    validate the file and run the dry build (run_builder.check: every refusal that run raises
         before its first step; no step, no array, no directory); exit 0, or print the ConfigError and
         exit 2
explain  the same checks first (a file check refuses is refused with the same message and exit 2),
         then print which engine functions the file selects, the dt / kT choices known without
         stepping, and the stage list

The thread variables are set from numerics.threads BEFORE numpy is imported (the raw TOML is
read with tomllib first), with the repo's own rule (backend_config.environment): OMP and NUMBA
get the thread count, every BLAS pool is pinned to 1.
"""
import os
import sys
import tomllib


def _threads(path):
    """numerics.threads from the raw file, 1 when the file or key is missing or malformed."""
    try:
        with open(path, "rb") as fh:
            n = tomllib.load(fh).get("numerics", {}).get("threads", 1)
        return n if isinstance(n, int) and not isinstance(n, bool) and n >= 1 else 1
    except (OSError, tomllib.TOMLDecodeError):
        return 1


def main(argv):
    argv = list(argv)
    stop = None
    if "--stop-after-checkpoints" in argv:           # test hook: stop right after the k-th checkpoint of this launch
        j = argv.index("--stop-after-checkpoints")
        try:
            stop = int(argv[j + 1])
            del argv[j:j + 2]
        except (IndexError, ValueError):
            print("error: --stop-after-checkpoints needs an integer", file=sys.stderr)
            return 2
    if len(argv) != 3 or argv[1] not in ("run", "check", "explain"):
        print("usage: python -m fftpf run|check|explain <case.toml> [--stop-after-checkpoints k]", file=sys.stderr)
        return 2
    command, path = argv[1], argv[2]
    if not os.path.isfile(path):
        print(f"error: {path}: file does not exist", file=sys.stderr)
        return 2
    from fftpf import backend_config                 # imports no numpy
    os.environ.update(backend_config.environment(_threads(path)))
    from fftpf import run_builder, run_input, run_output, run_protocol
    try:
        if command == "run":
            print(run_protocol.run(path, stop_after_checkpoints=stop)["run_dir"])
            return 0
        cfg = run_input.load(path)
        run_protocol.check(cfg)                      # explain refuses what check refuses
        if command == "explain":
            print(run_builder.explain(cfg))
        elif command == "check":
            print(f"{path}: ok")
    except run_output.RunInterrupted as exc:
        print(f"interrupted: {exc}", file=sys.stderr)
        return 3
    except run_input.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
