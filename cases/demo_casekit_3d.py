"""WP29 §41.C C5(b) 的端到端演示:bulk3d 16³ 全周期、**弹性开**、298K Langevin 退火到
能量平稳。全部用 `solvers/casekit.py` 写;Q 的 Voigt→张量 换算(Q1212 = Q44_phys/4,
SPEC §37.5 具名陷阱)由 build_3d 代做。

    /opt/miniconda3/bin/python3 tools/job.py submit cases/demo_casekit_3d.py --name demo3d

结果全在 reports/runs/<id>/;quicklook.png 取 z 中间那一层(3D 态 npz 里存的是全场)。

ERRATUM (SPEC V2-03.C.1 / V2-04.0):这次已存档的演示跑用的是 V2-04.0 之前的朗之万
归一化,噪声低了 N_cells/V_cell = 4096 倍;标称 T = 298 K 实际是
T_equivalent = T*V_cell/N_cells = 0.0728 K(16³ / 1.0 nm 网格)。代码已改用在案
约定(units.kT_langevin),重跑**不会**复现已记录的数字。定性:INTERPRETATION ——
这是演示/成本卡,不含有限温结论,那次退火按淬火(quench)理解;见 SPEC V2-03.C.1(a)
与 reports/REVIEW_v2_03c_kt.md 的 impact table。
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")               # 不可省:必须早于 import numpy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers import bulk3d, casekit, materials, timestep, units


def main():
    deck = materials.get("bto_lcc2005_8th", T=298.0)
    threads = casekit.single_thread(bulk3d)
    grid, ops, el = casekit.build_3d(deck, N=16, d=1.0, elastic=True, k0="stress_free")
    P0 = bulk3d.random_smooth_field_3d(grid, 3, amplitude=0.005, kmax_frac=0.5, seed=0)
    dt, info = timestep.verified_dt(bulk3d, P0, deck["landau"], ops, grid, L=1.0,
                                    T_total=400.0, elastic_params=el)
    # 朗之万 FDT 系数只有一个约定:k_B T·1e18/V_cell(SPEC V2-04.0);kT_deck_3d 是能量不是能量密度
    kT = units.kT_langevin(298.0, grid["hx"], grid["hy"], grid["hz"])
    P, trace, why, state = casekit.anneal(bulk3d, P0, deck["landau"], ops, grid, dt=dt, kT=kT,
                                          elastic_params=el, seed=0, n_max=2000, sample_every=25,
                                          return_state=True)
    run = casekit.run_dir("demo_casekit_3d")
    # state= 让 n_steps/t/stop_reason 取自这次运行本身,state.npz 也可续算
    casekit.save_run(run, params=dict(deck="bto_lcc2005_8th", T=298.0, N=16, d=1.0, dt=dt,
                                      dt_agree_relerr=info["agree_relerr"], kT=kT,
                                      elastic="on (k0=stress_free)", meta=deck["meta"]),
                     P=P, grid=grid, trace=trace, thread_line=threads, state=state)
    print(threads)
    print(f"dt={dt:.4g}  kT={kT:.4g}  步数={state['step']}  t={state['t']:.4g}  判停={why}")
    print(f"末态总能={trace[-1]:.6f}   max|P|={abs(P).max():.4f}  ->  {os.path.relpath(run, os.getcwd())}")


if __name__ == "__main__":
    main()
