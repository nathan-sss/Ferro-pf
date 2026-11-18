"""BTO film under a specified equibiaxial in-plane misfit (one run per command).

Uses existing film3d/film_elastic kernels without changing their physics.
Geometry arguments are explicit: periodic x/y cells, inclusive z nodes.
Temperature affects the Landau potential; Langevin noise is OFF.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from solvers import casekit, elastic, film3d as f3, film_elastic as fe
from solvers import io_utils, materials, solver, timestep, units


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, required=True)
    p.add_argument("--ny", type=int, required=True)
    p.add_argument("--nz", type=int, required=True)
    p.add_argument("--lx-nm", type=float, required=True)
    p.add_argument("--ly-nm", type=float, required=True)
    p.add_argument("--thickness-nm", type=float, required=True)
    p.add_argument("--temperature-c", type=float, required=True)
    p.add_argument("--strain-pct", type=float, required=True)
    p.add_argument("--eps-b", type=float, default=10.0)
    p.add_argument("--substrate-layers", type=int, default=12)
    p.add_argument("--substrate-material", default=None, help="Registry name; omitted retains legacy same-stiffness substrate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--end-time", type=float, default=400.0,
                   help="Total elapsed deck time, not seconds")
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--scheme", choices=("sbdf1", "sbdf2"), default="sbdf2")
    p.add_argument("--dt0", type=float, default=0.02)
    p.add_argument("--verify-time", type=float, default=100.0)
    p.add_argument("--fixed-dt-ladder", action="store_true", help="Start dt-vs-dt/2 audit at dt0 without a growth probe; adopts finer passing step")
    p.add_argument("--verify-tol", type=float, default=0.01)
    p.add_argument("--check-every", type=int, default=51)
    p.add_argument("--residual-tol", type=float, default=1e-6)
    p.add_argument("--plateau-windows", type=int, default=5)
    p.add_argument("--fft-workers", type=int, default=1)
    p.add_argument("--elastic-backend", choices=("numpy", "numba"), default="numpy")
    p.add_argument("--elastic-workers", type=int, default=0, help="0: reference solver; positive: optimized force with this many wavevector workers")
    p.add_argument("--output", type=Path)
    return p.parse_args()


def setup(a):
    # dx_nm fixes the MATERIAL gradient coefficient, not the mesh resolution.
    deck = materials.get("bto_lcc2005_paper", T=a.temperature_c + 273.15,
                         dx_nm=0.4)
    grid = f3.make_grid_3d(Nx=a.nx, Ny=a.ny, Nz=a.nz, Lx=a.lx_nm,
                           Ly=a.ly_nm, hf=a.thickness_nm)
    stack = fe.make_stack_grid(grid, hs_dx=a.substrate_layers)
    c, q = deck["elastic"], deck["electrostriction"]
    substrate_name = getattr(a, "substrate_material", None)
    substrate = materials.get(substrate_name, T=a.temperature_c+273.15) if substrate_name else None
    el = fe.setup_film_elastic(stack, c["C11"], c["C12"], c["C44"],
                               q["Q11_phys"], q["Q12_phys"], q["Q44_phys"],
                               e0=a.strain_pct / 100.0,
                               C_substrate=substrate["elastic"] if substrate else None)
    kappa = units.relative_to_deck_permittivity(a.eps_b)
    return deck, grid, el, kappa


def configure_execution(fft_workers, elastic_workers, el=None, elastic_backend="numpy"):
    """Compatibility entry point; all execution policy lives in one module."""
    from solvers.execution import configure
    return configure(fft_workers, elastic_backend, el, elastic_workers=elastic_workers)


def initial_state(grid, seed):
    # Independent band-limited x/y noise on every z layer, following the film case.
    return np.stack([solver.random_smooth_field(
        grid, 3, amplitude=0.005, kmax_frac=0.5, seed=seed * 100000 + z)
        for z in range(grid["Nz"])], axis=1)


def volume_mean(field, grid):
    return np.sum(field.mean(axis=(-2, -1)) * grid["qw"], axis=-1) / grid["hf"]


def diagnostics(P, alpha, grid, G, kappa, pf, el):
    density, _ = f3.total_energy_3d(P, alpha, grid, kappa, G,
                                    poisson_factors=pf, el=el)
    force = f3.grad_force_3d(P, grid, G) + f3.explicit_N(P, alpha, grid, pf, el=el)
    means = volume_mean(P, grid)
    rms = np.sqrt(volume_mean(P * P, grid))
    magnitude = np.sqrt(np.sum(P * P, axis=0))
    c_like = (magnitude > 0.05) & (P[2] ** 2 > 0.9 * magnitude ** 2)
    return dict(**{f"f_{k}_deck": float(v) for k, v in density.items()},
                **{f"mean_P{i+1}_C_m2": float(v) for i, v in enumerate(means)},
                **{f"rms_P{i+1}_C_m2": float(v) for i, v in enumerate(rms)},
                mean_Pmag_C_m2=float(volume_mean(magnitude, grid)),
                c_like_fraction=float(volume_mean(c_like, grid)),
                max_abs_P_C_m2=float(np.max(np.abs(P))),
                force_max_deck=float(np.max(np.abs(force))))


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    a = arguments()
    threads = configure_execution(a.fft_workers, a.elastic_workers, elastic_backend=a.elastic_backend)
    run = a.output or Path(casekit.run_dir("bto_film_biaxial"))
    run.mkdir(parents=True, exist_ok=True)
    if (run / "params.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {run}")
    start = time.perf_counter()
    deck, grid, el, kappa = setup(a)
    configure_execution(a.fft_workers, a.elastic_workers, el, a.elastic_backend)
    alpha, G = deck["landau"], deck["gradient"]["G11"]
    P = initial_state(grid, a.seed)
    params = dict(arguments=vars(a), deck=deck, kappa_deck=kappa,
                  e0_fraction=a.strain_pct / 100.0, temperature_K=a.temperature_c + 273.15,
                  substrate_nm=a.substrate_layers * grid["hz"],
                  spacings_nm=[grid["hx"], grid["hy"], grid["hz"]],
                  scheme=a.scheme, mobility=1.0, E_ext=[0, 0, 0], langevin=False,
                  strain="epsilon_xx = epsilon_yy = e0; mean epsilon_xy = 0",
                  boundaries="periodic x/y; natural P at z surfaces; short-circuit electrodes; "
                             "traction-free top; clamped substrate bottom",
                  substrate=(materials.get(a.substrate_material, T=a.temperature_c+273.15)
                             if a.substrate_material else "same homogeneous stiffness as film; zero eigenstrain"),
                  IC="independent layerwise band-limited random fields; amplitude 0.005 C/m2",
                  thread_settings=threads,
                  provenance=io_utils.provenance([__file__, ROOT / "params/bto_LCC2005_paper.toml"] +
                      ([ROOT / materials.get(a.substrate_material)["meta"]["source_toml"]] if a.substrate_material else [])),
                  commit=casekit._git_commit())
    write_json(run / "params.json", params)
    print(f"run={run}\n{threads}\nshape={P.shape} e0={a.strain_pct}% eps_b={a.eps_b}", flush=True)
    # The deterministic check compares full fields at the SAME elapsed deck time.
    dt, certificate = timestep.verified_dt(
        f3, P, alpha, {"G11": G, "kappa_b": kappa}, grid, L=1.0,
        E_ext=(0., 0., 0.), elastic_params=el, T_total=a.verify_time,
        dt0=a.dt0, n_probe=50, tol=a.verify_tol, observable="full_field_L2",
        scheme=a.scheme, skip_probe=a.fixed_dt_ladder)
    requested_steps = int(np.ceil(a.end_time / dt))
    dt = a.end_time / requested_steps  # at most the verified dt; land exactly on end-time
    params.update(dt=dt, dt_certificate=certificate, requested_steps=requested_steps)
    write_json(run / "params.json", params)
    print(f"verified dt={dt:.8g}; full-field error={certificate['agree_relerr']:.3g}", flush=True)
    pf = f3.poisson_factor(grid, kappa)
    gf = f3.build_grad_factors(grid, G, dt, 1.0)
    gf2 = f3.build_grad_factors_sbdf2(grid, G, dt, 1.0) if a.scheme == "sbdf2" else None
    P_previous = None
    first = diagnostics(P, alpha, grid, G, kappa, pf, el)
    records = [dict(step=0, time_deck=0.0, max_delta_P_C_m2=0.0, **first)]
    previous_P, previous_f = P.copy(), first["f_total_deck"]
    consecutive = 0
    n_steps = min(requested_steps, a.max_steps)
    status = "time_limit" if n_steps == requested_steps else "step_limit"
    with (run / "energy_trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerow(records[0])
        for step in range(1, n_steps + 1):
            new = (f3.step_sbdf2(P, P_previous, alpha, grid, gf2, pf, dt, 1.0, el=el)
                   if gf2 is not None and P_previous is not None else
                   f3.step(P, alpha, grid, gf, pf, dt, 1.0, el=el))
            P_previous, P = P, new
            if not np.all(np.isfinite(P)):
                raise FloatingPointError(f"Nonfinite field at step {step}; run failed")
            if step % a.check_every and step != n_steps:
                continue
            values = diagnostics(P, alpha, grid, G, kappa, pf, el)
            delta_P = float(np.max(np.abs(P - previous_P)))
            delta_t = (step - records[-1]["step"]) * dt
            f_rate = abs(values["f_total_deck"] - previous_f) / delta_t
            good = (values["force_max_deck"] <= a.residual_tol
                    and delta_P < 1e-5 and f_rate < 1e-8)
            consecutive = consecutive + 1 if good else 0
            record = dict(step=step, time_deck=step * dt,
                          max_delta_P_C_m2=delta_P, **values)
            records.append(record)
            writer.writerow(record)
            stream.flush()
            previous_P, previous_f = P.copy(), values["f_total_deck"]
            elapsed = time.perf_counter() - start
            write_json(run / "progress.json", dict(**record, seconds=elapsed,
                                                    stationary_windows=consecutive))
            if len(records) % 10 == 0 or consecutive >= a.plateau_windows or step == n_steps:
                io_utils.save_state_npz(run / "state.npz", P, grid, step, step * dt,
                                        prm=params, alpha=alpha, fields={"P_previous": P_previous})
                print(f"step={step} t={step*dt:.3f} f={previous_f:.8g} "
                      f"force={values['force_max_deck']:.3g} dP={delta_P:.3g} "
                      f"windows={consecutive} seconds={elapsed:.1f}", flush=True)
            if consecutive >= a.plateau_windows:
                status = "converged"
                break
    summary = dict(**records[-1], status=status, strain_pct=a.strain_pct, seed=a.seed,
                   dt=dt, seconds=time.perf_counter() - start,
                   stop_rule=f"{a.plateau_windows} consecutive checks: force <= residual_tol, delta_P < 1e-5, "
                             "abs(delta_f)/delta_t < 1e-8",
                   geometry_nm=[a.lx_nm, a.ly_nm, a.thickness_nm])
    write_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
