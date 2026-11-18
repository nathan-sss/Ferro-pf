"""Deterministic coupled-aging references, geometry parity and restart gates."""
import tempfile
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from solvers import solver, bulk3d, film3d
from solvers.drive import AgingDrive


def fixture(engine, dt=0.01, tau=0.7, steps=(20,), fields=((0.,0.,0.),)):
    if engine == '2d':
        grid = solver.make_grid(3,3,3.,3.)
        shape = (3,3,3)
    elif engine == 'bulk3d':
        grid = bulk3d.make_grid_3d(Nx=3,Ny=3,Nz=3,Lx=3.,Ly=3.,Lz=3.)
        shape = (3,3,3,3)
    else:
        grid = film3d.make_grid_3d(Nx=3,Ny=3,Nz=3,Lx=3.,Ly=3.,hf=2.)
        shape = (3,3,3,3)
    P = np.zeros(shape)
    P[0] = .2
    b = np.zeros(shape)
    b[0] = .03
    args = dict(engine=engine, grid=grid,
        alpha=dict(a1=.8,a11=0.,a12=0.,a111=0.,a112=0.,a123=0.),
        gradient=dict(G11=.2,G12=0.,G44=.1,G44p=.1), kappa=2.,
        dt=dt,g=.4,tau=tau,b=b,fields=fields,steps=steps)
    return P,args


def coupled_convergence():
    exact = expm(np.array([[-1.6,1.],[.4/.7,-1/.7]])*.2) @ [.2,.03]
    numbers = {}
    for engine in ('2d','bulk3d','film3d'):
        errors = []
        for dt in (.02,.01,.005):
            P,args = fixture(engine,dt=dt,steps=(round(.2/dt),))
            run = AgingDrive(P,**args)
            initial = run.energy()['total']
            rows = run.advance(round(.2/dt))
            assert np.max(np.diff([initial]+[r['energy'] for r in rows])) <= 1e-12
            # Full fields: every component and site enters both errors.
            refP,refb = np.zeros_like(P),np.zeros_like(P)
            refP[0],refb[0] = exact
            errP = np.linalg.norm(run.P-refP)/np.linalg.norm(refP)
            errb = np.linalg.norm(run.b-refb)/np.linalg.norm(refb)
            errors.append((errP,errb))
        errors = np.asarray(errors)
        ratios = errors[:-1]/errors[1:]
        assert np.all((ratios > 1.8) & (ratios < 2.2))
        assert errors[-1].max() < .01
        for n,(ep,eb) in enumerate(errors):
            numbers[f'{engine}_rung{n}_P_relerr'] = float(ep)
            numbers[f'{engine}_rung{n}_b_relerr'] = float(eb)
    return dict(ok=True,numbers=numbers)


def disabled():
    count = 0
    for engine in ('2d','bulk3d','film3d'):
        P,args = fixture(engine,steps=(2,))
        args.update(g=0.,b=None)
        run = AgingDrive(P,**args)
        m = run.model
        if engine == '2d':
            direct = solver.step(P,m['alpha'],run.ops,run.inverse,m['dt'],1.,(0.,0.,0.))
        elif engine == 'bulk3d':
            direct = bulk3d.step_3d(P,m['alpha'],run.ops,run.inverse,m['dt'],1.,(0.,0.,0.))
        else:
            direct = film3d.step(P,m['alpha'],m['grid'],run.inverse,run.ops,m['dt'],1.)
        run.advance()
        assert np.array_equal(run.P,direct) and not np.any(run.b)
        count += 1
    return dict(ok=True,numbers={'exact_disabled_geometries':count})


def restart():
    count = 0
    for engine in ('2d','bulk3d','film3d'):
        for tau in (.7,np.inf):
            P,args = fixture(engine,tau=tau,steps=(4,4),fields=((.01,0.,0.),(-.01,0.,0.)))
            uninterrupted = AgingDrive(P,**args)
            initial = uninterrupted.energy()['total']
            full = uninterrupted.advance(8)
            changes = np.diff([initial]+[r['energy'] for r in full])
            work = np.array([r['field_control_work_increment'] for r in full])
            assert np.max(changes-work) <= 1e-12
            part = AgingDrive(P,**args)
            prefix = part.advance(5)  # after field reversal, not just before
            with tempfile.TemporaryDirectory(prefix='aging-restart-') as temp:
                path = Path(temp)/'checkpoint.npz'
                part.save(path)
                resumed = AgingDrive(P,**args).restore(path)
                tail = resumed.advance(3)
                assert full == prefix+tail
                assert np.array_equal(uninterrupted.P,resumed.P)
                assert np.array_equal(uninterrupted.b,resumed.b)
                for key,value in [('dt',.02),('g',.5),('tau',.9),('fields',((.02,0.,0.),(-.01,0.,0.)))]:
                    changed = dict(args)
                    changed[key] = value
                    try:
                        AgingDrive(P,**changed).restore(path)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('accepted incompatible '+key)
            # Driving work is nonzero at the switch; both stored and resumed.
            assert full[4]['field_control_work_increment'] != 0
            count += 1
    return dict(ok=True,numbers={'exact_restarts':count,'incompatible_rejections':4*count})


def spatial_restart():
    """Mixed inactive, relaxing and frozen sites at protocol boundaries."""
    count=0
    for engine in ('2d','bulk3d','film3d'):
        P,args=fixture(engine,steps=(4,4),fields=((.01,0.,0.),(-.01,0.,0.)))
        g=np.full(P.shape[1:],.4);tau=np.full(P.shape[1:],.7)
        g.flat[0]=0.;tau.flat[1]=np.inf
        b=args['b'].copy();b[:,g==0]=0.
        args.update(g=g,tau=tau,b=b)
        full=AgingDrive(P,**args);rows=full.advance(8)
        for split in (0,4,5,8):
            part=AgingDrive(P,**args);prefix=part.advance(split)
            with tempfile.TemporaryDirectory(prefix='aging-spatial-') as temp:
                path=Path(temp)/'state.npz';part.save(path)
                resumed=AgingDrive(P,**args).restore(path);tail=resumed.advance(8-split)
            assert rows==prefix+tail
            assert np.array_equal(full.P,resumed.P) and np.array_equal(full.b,resumed.b)
            assert np.all(resumed.b[:,g==0]==0)
            assert np.array_equal(resumed.b[:,np.isinf(tau)],b[:,np.isinf(tau)])
            count+=1
    return dict(ok=True,numbers={'spatial_boundary_restarts':count})
