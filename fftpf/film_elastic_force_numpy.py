"""Optional force-only direct film elasticity; reference energy/solve stays intact.

Uses the same assembled, factored stack operator. All substrate displacements
participate in mechanical equilibrium; only film-face stresses are recovered
because only their eigenstrain depends on P. No substrate energy is dropped:
energy diagnostics continue through film_elastic.elastic_solve_film.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from fftpf import elastic_base as elastic, film_engine as f3, film_elastic as fe

_ORDER = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _validate_el(el):
    if el.get('spm') or 'C_field' in el:
        raise ValueError('Fast force requires the direct homogeneous-layer stack')
    static = el.get('eps_star_static')
    if static is not None:
        if static.shape != (3, 3, el['Nz'], el['Ny'], el['Nx']):
            raise ValueError('Static eigenstrain shape does not match the stack')
        if not np.array_equal(static, static.swapaxes(0, 1)):
            raise ValueError('Fast force requires a symmetric static eigenstrain')
        if np.any(static[:, :, :el['film_lo']] != 0.0):
            raise ValueError('Static eigenstrain must be exactly zero in the substrate')


def _current_el(original, el, P):
    """Sources may change; geometry, stiffness and factors stay fixed."""
    el = original if el is None else el
    if el['_factors'] is not original['_factors']:
        raise ValueError('Create a new force solver when the elastic stack changes')
    _validate_el(el)
    if P.shape != (3, el['n_elem']-el['film_lo']+1, el['Ny'], el['Nx']):
        raise ValueError('Polarization shape does not match the elastic stack')
    return el


def _solve(factors, rhs):
    """Block Thomas, identical z recurrence; batched C matmul per wavevector."""
    sub, cp, inv = (factors[k] for k in ('sub', 'cp', 'piv_inv'))
    dp = np.empty_like(rhs)
    dp[0] = (inv[0] @ rhs[0, ..., None])[..., 0]
    for z in range(1, len(rhs)):
        corr = rhs[z] - (sub[z-1] @ dp[z-1, ..., None])[..., 0]
        dp[z] = (inv[z] @ corr[..., None])[..., 0]
    x = np.empty_like(dp)
    x[-1] = dp[-1]
    for z in range(len(rhs)-2, -1, -1):
        x[z] = dp[z] - (cp[z] @ x[z+1, ..., None])[..., 0]
    return x


class ForceSolver:
    """Reusable worker pool; disjoint ky slabs, no shared output writes."""
    def __init__(self, el, workers=1):
        _validate_el(el)
        if workers < 1 or workers > el['Ny']:
            raise ValueError('workers must be in [1, Ny]')
        self.el = el
        self.workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        edges = np.linspace(0, el['Ny'], workers+1, dtype=int)
        self.slabs = [slice(int(a), int(b)) for a,b in zip(edges[:-1], edges[1:])]
        self.factors = [{k:el['_factors'][k][:, s] for k in ('sub','cp','piv_inv')}
                        for s in self.slabs]

    def close(self):
        if self.pool:
            self.pool.shutdown()

    def __call__(self, P, el=None):
        el = _current_el(self.el, el, P)
        lo, n, ny, nx = el['film_lo'], el['n_elem'], el['Ny'], el['Nx']
        eps = elastic.eigenstrain(P, el['Q11'], el['Q12'], el['Q44'])
        if el.get('eps_star_static') is not None:
            eps = eps + el['eps_star_static'][:, :, lo:]
        src = np.stack([0.5*(eps[i,j,:-1]+eps[i,j,1:])-el['ebar'][i,j]
                        for i,j in _ORDER])
        # Substrate source is exactly zero; assemble only film-face contributions.
        es = f3._rfft2(src)
        local = np.einsum('yxij,jeyx->eyxi', el['M_el'], es)
        rhs = np.zeros((n,ny,es.shape[-1],3), dtype=complex)
        rhs[lo:] = local[...,3:]
        # Face lo+e's lower node is reduced row lo+e-1. Without a substrate
        # (lo=0) face 0's lower node is the clamped, dropped node 0.
        skip = 0 if lo else 1
        rhs[lo-1+skip:-1] += local[skip:,...,:3]
        if self.pool:
            futures = [self.pool.submit(_solve, f, rhs[:,s])
                       for f,s in zip(self.factors,self.slabs)]
            reduced = np.concatenate([f.result() for f in futures],axis=1)
        else:
            reduced = _solve(self.factors[0],rhs)
        uh = np.zeros((3,n+1,ny,es.shape[-1]), dtype=complex)
        uh[:,1:] = np.moveaxis(reduced,-1,0)
        # Retain the reference real-field projection, including Nyquist behavior.
        uh = f3._rfft2(f3._irfft2(uh,(ny,nx)))
        strains = np.stack(fe._face_strain(uh[:,lo:-1],uh[:,lo+1:],el['K1'],el['K2'],el['hz']))
        e = f3._irfft2(strains,(ny,nx)) - src
        c11,c12,c44 = el['C11'],el['C12'],el['C44']
        stress = np.empty_like(e)
        stress[0] = c11*e[0]+c12*(e[1]+e[2])
        stress[1] = c11*e[1]+c12*(e[0]+e[2])
        stress[2] = c11*e[2]+c12*(e[0]+e[1])
        stress[3:] = 2*c44*e[3:]
        # Exact source chain rule: each adjacent FILM face contributes 1/2,
        # including interface/top P nodes with only one P-dependent face.
        node = np.zeros((6,P.shape[1],ny,nx))
        node[:,:-1] += 0.5*stress
        node[:,1:] += 0.5*stress
        sigma = np.empty((3,3)+P.shape[1:])
        for s,(i,j) in zip(node,_ORDER):
            sigma[i,j] = sigma[j,i] = s
        return elastic.elastic_force_explicit(P,sigma,el['Q11'],el['Q12'],el['Q44'])
