"""Local deterministic defect-aging gates; host owns threads and registration."""
import numpy as np
from solvers import defects


def aging_local_fd():
    rng=np.random.default_rng(290)
    p=rng.normal(0,.2,(3,3,5));b=rng.normal(0,.04,p.shape)
    g=np.linspace(.2,.8,15).reshape(3,5)
    dp=rng.normal(0,.1,p.shape);db=rng.normal(0,.05,b.shape)
    h=1e-5
    energy=lambda p,b: float(np.sum(defects.aging_energy_density(p,b,g)))
    fp=float(np.sum(defects.bias_force(p,b)*dp))
    fb=float(np.sum(defects.aging_force_bias(p,b,g)*db))
    fdp=(energy(p+h*dp,b)-energy(p-h*dp,b))/(2*h)
    fdb=(energy(p,b+h*db)-energy(p,b-h*db))/(2*h)
    errors=[abs(fdp-fp)/max(abs(fp),1e-12),abs(fdb-fb)/max(abs(fb),1e-12)]
    return dict(ok=max(errors)<1e-6,numbers=dict(P_fd=errors[0],b_fd=errors[1]))


def aging_limits():
    p=np.arange(18,dtype=float).reshape(3,2,3)/20
    g=np.array([[0.,.4,.4],[.2,0.,.6]])
    tau=np.array([[1.,np.inf,2.],[3.,np.inf,np.inf]])
    b=np.full_like(p,.03);b[:,g==0]=0.
    b[:,1,2]=-.0
    result=defects.aging_update(p,b,g,tau,.2)
    frozen=np.isinf(tau)
    exact_frozen=result[:,frozen].tobytes()==b[:,frozen].tobytes()
    exact_zero=bool(np.all(result[:,g==0]==0))
    zero_dt=defects.aging_update(p,b,g,tau,0.).tobytes()==b.tobytes()
    self_energy=defects.aging_self_energy_density(b,g)
    zero_energy=bool(np.all(self_energy[g==0]==0))
    no_alias=not np.shares_memory(defects.aging_state(p,g,tau,b),b)
    bad=[lambda: defects.aging_state(p,-.1,1.),
         lambda: defects.aging_state(p,np.inf,1.),
         lambda: defects.aging_state(p,np.nan,1.),
         lambda: defects.aging_state(p,.1,0.),
         lambda: defects.aging_state(p,.1,-np.inf),
         lambda: defects.aging_state(p,.1,np.nan),
         lambda: defects.aging_state(p,0.,1.,np.ones_like(p)),
         lambda: defects.aging_state(p,np.ones(3),1.),
         lambda: defects.aging_update(p,b,g,tau,-1.),
         lambda: defects.aging_update(p,b,g,tau,np.nan),
         lambda: defects.aging_update(p,b,g,tau,np.inf)]
    rejected=0
    for call in bad:
        try: call()
        except ValueError: rejected+=1
    nums=dict(frozen_bytes=exact_frozen,inactive_zero=exact_zero,zero_dt_bytes=zero_dt,
              inactive_self_zero=zero_energy,independent_state=no_alias,invalid_rejected=rejected)
    return dict(ok=all((exact_frozen,exact_zero,zero_dt,zero_energy,no_alias,rejected==len(bad))),numbers=nums)


def aging_exponential():
    p=np.array([.2,-.1,.3])[:,None,None]*np.ones((1,2,3))
    b=np.full_like(p,.02)
    g=np.array([[.1,.3,.5],[.2,.4,.6]])
    tau=np.array([[.5,1.,2.],[3.,4.,5.]])
    elapsed=.7
    actual=defects.aging_update(p,b,g,tau,elapsed)
    expected=g*p+(b-g*p)*np.exp(-elapsed/tau)
    composed=b.copy()
    for _ in range(7): composed=defects.aging_update(p,composed,g,tau,.1)
    error=float(np.max(np.abs(actual-expected)))
    semigroup=float(np.max(np.abs(composed-expected)))
    return dict(ok=max(error,semigroup)<1e-14,numbers=dict(closed_form_error=error,substep_error=semigroup))


def aging_square_wave():
    # Independent periodic steady-state solution of alternating +P/-P dwells:
    # peak b = gP (1-exp(-h/tau))/(1+exp(-h/tau)).
    p=np.zeros((3,2,3));p[2]=.3
    g=.4;tau=.7;hold=.2
    peak=g*p*np.tanh(hold/(2*tau))
    b=-peak
    positive=defects.aging_update(p,b,g,tau,hold)
    negative=defects.aging_update(-p,positive,g,tau,hold)
    error=max(float(np.max(np.abs(positive-peak))),float(np.max(np.abs(negative+peak))))
    self_initial=float(np.sum(defects.aging_energy_density(p,b,g)))
    self_final=float(np.sum(defects.aging_energy_density(p,positive,g)))
    return dict(ok=error<1e-14 and self_final<=self_initial,numbers=dict(periodic_error=error,held_P_energy_drop=self_initial-self_final,peak=float(peak[2,0,0])))
