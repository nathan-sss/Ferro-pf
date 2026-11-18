"""WP29 §41.C C5(a) 的端到端演示:BTO 298K、2D 周期、一个替位电荷失配推出的淬火 E 场,
从小噪声弛豫到能量平稳,判停用 block_stationary。全部用 `solvers/casekit.py` 写。

    /opt/miniconda3/bin/python3 tools/job.py submit cases/demo_casekit_2d.py --name demo2d

结果全在 reports/runs/<id>/(run.log/params.json/state.npz/energy_trace.csv/quicklook.png/
extra.npz)。前台直接跑也行,那时 run_dir 按当前时间自己命名。"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")               # 不可省:必须早于 import numpy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers import casekit, materials, quenched_fields as qf, solver, timestep


def main():
    deck = materials.get("bto_lcc2005_8th", T=298.0)
    threads = casekit.single_thread(solver)
    grid, ops = casekit.build_2d(deck, Nx=64, Ny=64, h=0.4)
    E_d = qf.charge_disorder_field(grid, conc=0.5, delta_q=3.0, lambda_s_nm=2.0,
                                   eps_r=1000.0, a_lattice_nm=0.4049, seed=1)
    P0 = solver.random_smooth_field(grid, 3, amplitude=0.005, seed=0)
    dt, info = timestep.verified_dt(solver, P0, deck["landau"], ops, grid,
                                    L=1.0, E_ext=E_d, T_total=400.0)
    P, trace, why, state = casekit.anneal(solver, P0, deck["landau"], ops, grid, dt=dt,
                                          E_ext=E_d, n_max=6000, sample_every=50,
                                          return_state=True)
    run = casekit.run_dir("demo_casekit_2d")
    # state= 让 n_steps/t/stop_reason 取自这次运行本身,state.npz 也可续算
    casekit.save_run(run, params=dict(deck="bto_lcc2005_8th", T=298.0, N=64, h=0.4, dt=dt,
                                      dt_agree_relerr=info["agree_relerr"], meta=deck["meta"]),
                     P=P, grid=grid, trace=trace, thread_line=threads, extra=dict(E_d=E_d),
                     state=state)
    print(threads)
    print(f"dt={dt:.4g}  步数={state['step']}  t={state['t']:.4g}  判停={why}  末态总能={trace[-1]:.6f}")
    print(f"max|P|={abs(P).max():.4f}  ->  {os.path.relpath(run, os.getcwd())}")


if __name__ == "__main__":
    main()
