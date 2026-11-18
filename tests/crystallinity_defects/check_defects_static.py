"""Tiny prescribed-defect acceptance checks; core hosts set thread limits."""
import numpy as np
from solvers import solver, bulk3d, film3d, film3d_spm, defects


def _cold(factors):
    # Iterative warm starts change roundoff; compare identical cold solves.
    factors.pop('_warm_E', None)
    return factors


def _fixtures():
    alpha = dict(a1=.6, a11=.2, a12=.1, a111=.03, a112=.02, a123=.01)
    g2 = solver.make_grid(5, 5, 5., 5.)
    o2 = solver.build_operators(g2, .1, 0., .05, .05, 2.)
    yield ('2d', (3, 5, 5), lambda p, e: solver.total_energy(p, alpha, o2, e, g2)[1]['total'],
           lambda p, e: solver.total_force(p, alpha, o2, e)[-1], np.ones((5, 5)))
    g3 = bulk3d.make_grid_3d(5, 5, 5, 5., 5., 5.)
    o3 = bulk3d.build_operators_3d(g3, .1, 0., .05, .05, 2.)
    yield ('3d', (3, 5, 5, 5), lambda p, e: bulk3d.total_energy_3d(p, alpha, o3, e, g3)[1]['total'],
           lambda p, e: bulk3d.total_force_3d(p, alpha, o3, e)[-1], np.ones((5, 5, 5)))
    gf = film3d.make_grid_3d(6, 5, 6., 5., 5, 4.)
    kappa = 2. + .05 * np.sin(np.arange(150).reshape(5, 5, 6))
    for name, fac in [('film', film3d.poisson_factor(gf, 2.)),
                      ('film_spm', film3d_spm.poisson_factor_spm(gf, kappa))]:
        fac['spm_tol'] = 1e-12
        yield (name, (3, 5, 5, 6),
               lambda p, e, fac=fac: film3d.total_energy_3d(p, alpha, gf, 2., .1, poisson_factors=_cold(fac), E_ext=e)[1]['total'],
               lambda p, e, fac=fac: film3d.explicit_N(p, alpha, gf, _cold(fac), e) + film3d.grad_force_3d(p, gf, .1),
               gf['qw'][:, None, None] * np.ones((5, 5, 6)))


def static_fd():
    rng = np.random.default_rng(161)
    nums = {}
    for name, shape, energy, force, w in _fixtures():
        p = rng.normal(0, .15, shape)
        b = rng.normal(0, .02, shape)
        e = defects.compose_field((.01, -.02, .03), quenched=(.002, 0., 0.), bias=b, spatial_shape=shape[1:])
        for mode in ('all', 'boundary') if name.startswith('film') else ('all',):
            d = rng.normal(0, .1, shape)
            if mode == 'boundary':
                d[:, 1:-1] = 0
            h = 1e-5
            fd = (energy(p+h*d, e)-energy(p-h*d, e))/(2*h)
            ana = float(np.sum(force(p, e)*d*w))
            nums[name+'_'+mode] = abs(fd-ana)/max(abs(fd), abs(ana), 1e-12)
        observed = energy(p, e)-energy(p, e-b)
        expected = float(np.sum(defects.bias_energy_density(p,b)*w))
        nums[name+'_energy_count'] = abs(observed-expected)/max(abs(expected), 1e-12)
    return dict(ok=max(nums.values()) <= 1e-6, numbers=nums)


def static_identity():
    nums = {}
    e = np.array([.01, -.02, -.0])
    for name, shape, energy, force, w in _fixtures():
        p = np.random.default_rng(162).normal(0, .1, shape)
        off = defects.compose_field(e, bias=np.zeros(shape), spatial_shape=shape[1:])
        nums[name+'_off_bytes'] = force(p,e).tobytes() == force(p,off).tobytes()
        nums[name+'_energy_exact'] = energy(p,e) == energy(p,off)
        spatial = np.broadcast_to(e.reshape((3,)+(1,)*(len(shape)-1)), shape)
        nums[name+'_uniform_field'] = np.array_equal(force(p,e), force(p,spatial))
    nums['signed_zero_bytes'] = defects.compose_field(e,bias=(0.,0.,0.)).tobytes() == e.tobytes()
    invalid = [lambda: defects.compose_field((1.,2.)),
               lambda: defects.compose_field((0.,0.,0.),bias=np.ones((3,2,2))),
               lambda: defects.compose_field((0.,0.,0.),bias=np.ones((2,2,3)),spatial_shape=(2,2)),
               lambda: defects.compose_field((0.,0.,0.),bias=(0.,np.nan,0.))]
    caught = 0
    for call in invalid:
        try:
            call()
        except ValueError:
            caught += 1
    nums['invalid_rejected'] = caught == len(invalid)
    return dict(ok=all(nums.values()),numbers=nums)


def static_anchors():
    # Analytic linear equilibrium, independently evaluated engine derivative.
    a1, b = .6, .07
    alpha = dict(a1=a1, a11=0., a12=0., a111=0., a112=0., a123=0.)
    g = solver.make_grid(4,4,4.,4.)
    ops = solver.build_operators(g,.1,0.,.05,.05,2.)
    residuals=[]
    for applied in (-.15, 0., .15):
        p=np.zeros((3,4,4)); p[2]=(applied+b)/(2*a1)
        e=defects.compose_field((0.,0.,applied),bias=(0.,0.,b))
        residuals.append(float(np.max(np.abs(solver.total_force(p,alpha,ops,e)[-1]))))
    # A quartic double well has two spinodals; additive b shifts their center -b.
    a1q,a11=-.3,.6
    ps=np.sqrt(-a1q/(6*a11))
    centers=[]
    spinodal_residuals=[]
    quartic=dict(alpha,a1=a1q,a11=a11)
    for bias in (b,-b):
        switches=[2*a1q*p+4*a11*p**3-bias for p in (-ps,ps)]
        centers.append(sum(switches)/2)
        for pol, applied in zip((-ps,ps), switches):
            field=np.zeros((3,4,4));field[2]=pol
            total=defects.compose_field((0.,0.,applied),bias=(0.,0.,bias))
            spinodal_residuals.append(float(np.max(np.abs(solver.total_force(field,quartic,ops,total)[-1]))))
    error=max(abs(centers[0]+b),abs(centers[1]-b))
    return dict(ok=max(residuals+spinodal_residuals)<1e-12 and error<1e-14,
                numbers=dict(equilibrium_force=max(residuals),spinodal_force=max(spinodal_residuals),center_positive_bias=centers[0],center_negative_bias=centers[1],shift_error=error))


def wall_data():
    """Frozen periodic wall translation: local energy well, not a dynamic claim."""
    x=2*np.pi*np.arange(128)/128
    profile=lambda q: .3*np.tanh(np.sin(x-q)/.22)
    b=.03*profile(0)/.3
    shifts=np.array([-.2,-.1,0.,.1,.2])
    energy=np.array([-np.mean(b*profile(q)) for q in shifts])
    uniform=np.array([-np.mean(.03*profile(q)) for q in shifts])
    h=1e-5
    restoring=np.array([(np.mean(b*profile(q+h))-np.mean(b*profile(q-h)))/(2*h) for q in shifts])
    return dict(shifts=shifts,energy=energy,uniform_energy=uniform,restoring_force=restoring)


def static_wall_control():
    d=wall_data(); f=d['restoring_force']; en=d['energy']
    well=float(min(en[0],en[-1])-en[2])
    flat=float(np.ptp(d['uniform_energy']))
    ok=well>1e-5 and flat<1e-14 and f[1]>0 and f[3]<0 and abs(f[2])<1e-10
    return dict(ok=bool(ok),numbers=dict(well_depth=well,uniform_span=flat,force_left=float(f[1]),force_right=float(f[3]),force_center=float(f[2])))
