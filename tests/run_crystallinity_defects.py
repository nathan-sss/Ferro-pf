"""Portable feature checks; does not replace the repository's local CORE gates.

Run from any directory: python /path/to/repository/tests/run_crystallinity_defects.py
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent/'crystallinity_defects'))
from solvers import solver,bulk3d,film3d
for module in (solver,bulk3d,film3d):module.FFT_WORKERS=1
from check_crystallinity import energy_force,limits,stationary
from check_defects_static import static_fd,static_identity,static_anchors,static_wall_control
from check_defects_aging import aging_local_fd,aging_limits,aging_exponential,aging_square_wave
from check_defects_drive import coupled_convergence,disabled,restart,spatial_restart
CHECKS=dict(crystallinity_energy_force=energy_force,crystallinity_limits=limits,
    crystallinity_stationary=stationary,defects_static_fd=static_fd,
    defects_static_identity=static_identity,defects_static_anchors=static_anchors,
    defects_static_wall_control=static_wall_control,defects_aging_local_fd=aging_local_fd,
    defects_aging_limits=aging_limits,defects_aging_exponential=aging_exponential,
    defects_aging_square_wave=aging_square_wave,defects_drive_convergence=coupled_convergence,
    defects_drive_disabled=disabled,defects_drive_restart=restart,
    defects_spatial_restart=spatial_restart)


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',help='comma-separated check names')
    parser.add_argument('--json',type=Path,help='optional result file')
    args=parser.parse_args()
    names=args.only.split(',') if args.only else list(CHECKS)
    unknown=set(names)-set(CHECKS)
    if unknown:parser.error('unknown checks: '+','.join(sorted(unknown)))
    results={}
    for name in names:
        start=time.monotonic()
        result=CHECKS[name]()
        assert result['ok'],(name,result)
        results[name]=dict(result,seconds=time.monotonic()-start)
        print('PASS',name)
    payload=dict(suite='portable crystallinity/defects feature checks, not full CORE',
                 n_pass=len(results),n_checks=len(names),checks=results)
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(payload,indent=2,default=_json_scalar)+'\n')
    print(f'{len(results)}/{len(names)} passed')


if __name__=='__main__':main()
