"""Portable tiny checks for film energy/force adjoints; not the local CORE suite."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from solvers import solver,film3d as f3,film3d_spm as spm
f3.FFT_WORKERS=1


def check():
    rng=np.random.default_rng(924)
    g=f3.make_grid_3d(6,6,6.,6.,5,4.)
    shape=(3,5,6,6);P=rng.normal(0,.15,shape);Q=rng.normal(0,.1,shape)
    alpha=dict(a1=.6,a11=.2,a12=.1,a111=.03,a112=.02,a123=.01)
    weights=g['qw'][None,:,None,None]*g['hx']*g['hy']
    integrate=lambda a: float(np.sum(a*weights))
    numbers={}
    kappas=[('scalar',2.),('profile',np.linspace(1.9,2.1,5)),('spm',2.+.05*rng.normal(size=shape[1:]))]
    gradients=[('scalar',.1),('tensor',dict(G11=.1,G12=.02,G44=.04,G44p=.03))]
    for kname,kappa in kappas:
        pf=f3.poisson_factor(g,kappa) if kname!='spm' else spm.poisson_factor_spm(g,kappa)
        pf['spm_tol']=1e-13
        def cold():
            pf.pop('_warm_E',None)
            return pf
        ep=f3.solve_poisson_factored(P,g,cold())[0]
        eq=f3.solve_poisson_factored(Q,g,cold())[0]
        lhs,rhs=integrate(P*eq),integrate(Q*ep)
        numbers[kname+'_reciprocity']=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)
        for gname,G in gradients:
            force=f3.explicit_N(P,alpha,g,cold())+f3.grad_force_3d(P,g,G)
            energy=lambda p:f3.total_energy_3d(p,alpha,g,kappa,G,poisson_factors=cold())[1]['total']
            for mode in ('all','surface'):
                d=Q.copy()
                if mode=='surface':d[:,1:-1]=0
                h=1e-5;fd=(energy(P+h*d)-energy(P-h*d))/(2*h);an=integrate(force*d)
                numbers[kname+'_'+gname+'_'+mode]=abs(fd-an)/max(abs(fd),abs(an),1e-12)
    # Uniform divergence has a quadratic potential exactly on this z grid.
    P=np.zeros(shape);P[2]=g['z'][:,None,None]*.1
    physical=f3.physical_electric_field_3d(P,g,f3.poisson_factor(g,2.))[0]
    expected=.1*(g['hf']-2*g['z'])/4
    numbers['linear_Pz_physical_field']=float(np.max(np.abs(physical[2]-expected[:,None,None])))
    # Constant polarization under short circuit has no depolarizing field.
    P[:]=.1
    numbers['uniform_zero_field']=float(np.max(np.abs(f3.solve_poisson_factored(P,g,f3.poisson_factor(g,2.))[0])))
    assert max(numbers.values())<1e-6,numbers
    return dict(ok=True,numbers=numbers)


if __name__=='__main__':
    print(json.dumps(check(),indent=2))
