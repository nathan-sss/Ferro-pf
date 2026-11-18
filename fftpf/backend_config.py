"""CPU execution settings shared by all geometries.

Call configure before importing NumPy in a new script. Settings are process-wide;
independent simulations should use separate processes, each with its own state.
"""
import importlib
import os
import sys

BLAS_ENV = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
FFT_MODULES = ("pf2d_engine", "elastic_base", "pf3d_engine", "film_engine",
               "antiferroelectric_order", "octahedral_tilt_order", "disorder_fields")


def environment(workers):
    """Thread limits for a fresh process; FFT and Numba run in separate phases."""
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    return dict.fromkeys(BLAS_ENV, "1") | {
        "OMP_NUM_THREADS": str(workers), "NUMBA_NUM_THREADS": str(workers)}


def set_fft_workers(workers):
    environment(workers)  # Validate before changing any module.
    for name in FFT_MODULES:
        importlib.import_module("fftpf." + name).FFT_WORKERS = workers


def close(el):
    """Detach and release an optional film backend; reference mode needs none."""
    previous = el.pop("_force_solver", None)
    if previous is not None:
        previous.close()


def configure(workers=1, backend="numpy", el=None, *, elastic_workers=None, fused=False):
    """Select reference NumPy or fused Numba kernels and an explicit CPU budget.

    A film el additionally selects its elastic backend. elastic_workers is only
    needed by legacy film scripts: numpy + positive count selects the slab pool;
    zero selects reference elasticity. Numba defaults to workers for both kernels.
    The shared configuration must not change while a step is running.

    fused=True (opt-in, backend="numba" only) additionally routes the 2D,
    bulk3d and film3d steppers through fftpf.kernels_fused (one real-space RHS
    pass, one transform, per-wavevector Green/implicit kernels, reused
    scratch) and gives a film el the fused direct elastic force. It agrees
    with the reference to round-off, not bit for bit; moving an existing
    checkpoint onto it needs the migration certificate + dt/2 audit.
    """
    limits = environment(workers)
    if backend not in ("numpy", "numba"):
        raise ValueError("backend must be numpy or numba")
    if not isinstance(fused, bool):
        raise ValueError("fused must be True or False")
    if fused and backend != "numba":
        raise ValueError("fused=True requires backend='numba'")
    count = (workers if backend == "numba" else 0) if elastic_workers is None else elastic_workers
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("elastic_workers must be a nonnegative integer")
    if backend == "numba" and count == 0:
        raise ValueError("Numba film backend requires positive elastic_workers")
    # Changing these after NumPy import does not reconfigure an existing BLAS pool.
    if "numpy" in sys.modules and any(os.environ.get(k) != "1" for k in BLAS_ENV[:3]):
        raise RuntimeError("Call execution.configure before importing NumPy, or launch with execution.environment limits")
    # The pool ceiling is fixed at Numba import; active threads are set below.
    ceiling = str(max(os.cpu_count() or 1, workers, count))
    for key, value in limits.items():
        os.environ.setdefault(key, ceiling if key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS") else value)
    for key in BLAS_ENV:
        os.environ[key] = "1"
    if fused:
        # Numba's OpenMP pool otherwise spins after every compiled region and
        # steals the cores scipy.fft's own threads run on next (measured 2x
        # slower transforms). Read once, when the pool starts: set it, as the
        # other limits, before the first Numba parallel call of the process.
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    implementation = None
    kernels = None
    versions = {}
    if backend == "numba":
        import numba
        import llvmlite
        from fftpf.kernels_per_term import landau_force
        from fftpf import kernels_per_term as kernels
        if max(workers, count) > numba.config.NUMBA_NUM_THREADS:
            raise ValueError("Requested workers exceed NUMBA_NUM_THREADS; start a fresh process with a larger limit")
        implementation = landau_force
        versions = {"numba": numba.__version__, "llvmlite": llvmlite.__version__}

    candidate = None
    if el is not None and count and fused:
        from fftpf import kernels_fused as cpu_fused
        candidate = cpu_fused.film_force_solver(el, count)
    elif el is not None and count:
        name = "film_elastic_force_numba" if backend == "numba" else "film_elastic_force_numpy"
        candidate = importlib.import_module("fftpf." + name).ForceSolver(el, count)
    if el is not None:
        close(el)
        if candidate is not None:
            el["_force_solver"] = candidate
    set_fft_workers(workers)
    from fftpf import pf2d_engine as solver, elastic_base as elastic
    solver._BULK_FORCE_IMPL = implementation
    elastic._CPU_KERNELS = kernels
    if fused:
        from fftpf import kernels_fused as cpu_fused
        cpu_fused.release_workspaces()
        solver._FUSED = cpu_fused
    else:
        if solver._FUSED is not None:
            solver._FUSED.release_workspaces()
        solver._FUSED = None
    if backend == "numba":
        numba.set_num_threads(workers)
    settings = {"backend": backend, "workers": max(workers, count),
                "fft_workers": workers, "elastic_workers": count,
                "numba_threads": numba.get_num_threads() if backend == "numba" else None,
                "elastic_backend": backend, "backend_versions": versions,
                "dtype": "float64", "fastmath": False,
                "environment": {k: os.environ.get(k) for k in limits}}
    if fused:  # The key appears only when opted in; existing settings are unchanged.
        settings["fused"] = True
    return settings
