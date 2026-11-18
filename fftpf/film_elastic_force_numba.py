"""Optional double-precision direct-stack force kernels; no FFT/energy changes.

One owner calls each ForceSolver sequentially. Scratch buffers are instance
local; compiled workers own disjoint modes or nodes. Numba is imported only
when this backend is explicitly selected. fastmath is deliberately disabled.
"""
import numpy as np
try:
    import numba
    from numba import njit, prange
except ImportError as exc:
    numba = None
    _IMPORT_ERROR = str(exc)
    # Keep package discovery/import hygiene usable without the optional extra.
    # Construction below fails explicitly; these functions are never a fallback.
    def njit(**kwargs):
        return lambda function: function
    prange = range
from fftpf import film_engine as f3, film_elastic as fe
from fftpf.film_elastic_force_numpy import _ORDER, _current_el, _validate_el


@njit(parallel=True, cache=True, fastmath=False)
def _solve(sub, cp, inv, rhs, dp, out):
    for k in prange(rhs.shape[0]):
        for z in range(rhs.shape[1]):
            v0, v1, v2 = rhs[k,z,0], rhs[k,z,1], rhs[k,z,2]
            if z:
                v0 -= sub[k,z-1,0,0]*dp[k,z-1,0] + sub[k,z-1,0,1]*dp[k,z-1,1] + sub[k,z-1,0,2]*dp[k,z-1,2]
                v1 -= sub[k,z-1,1,0]*dp[k,z-1,0] + sub[k,z-1,1,1]*dp[k,z-1,1] + sub[k,z-1,1,2]*dp[k,z-1,2]
                v2 -= sub[k,z-1,2,0]*dp[k,z-1,0] + sub[k,z-1,2,1]*dp[k,z-1,1] + sub[k,z-1,2,2]*dp[k,z-1,2]
            for i in range(3):
                dp[k,z,i] = inv[k,z,i,0]*v0 + inv[k,z,i,1]*v1 + inv[k,z,i,2]*v2
        out[k,-1] = dp[k,-1]
        for z in range(rhs.shape[1]-2,-1,-1):
            for i in range(3):
                out[k,z,i] = dp[k,z,i] - (cp[k,z,i,0]*out[k,z+1,0] + cp[k,z,i,1]*out[k,z+1,1] + cp[k,z,i,2]*out[k,z+1,2])


@njit(parallel=True, cache=True, fastmath=False)
def _source(P, q11, q12, q44, ebar, static, has_static, out):
    nz, nxy = P.shape[1:]
    for index in prange((nz-1)*nxy):
        z, j = index//nxy, index%nxy
        for side in range(2):
            x,y,w = P[0,z+side,j],P[1,z+side,j],P[2,z+side,j]
            xx,yy,ww = x*x,y*y,w*w
            values = (q11*xx+q12*(yy+ww),q11*yy+q12*(xx+ww),q11*ww+q12*(xx+yy),.5*q44*x*y,.5*q44*x*w,.5*q44*y*w)
            for i in range(6):
                value = values[i]
                if has_static:
                    value += static[i,z+side,j]
                if side == 0:
                    out[i,z,j] = value
                else:
                    out[i,z,j] = .5*(out[i,z,j]+value)-ebar[i]


@njit(parallel=True, cache=True, fastmath=False)
def _force(P, e, c11, c12, c44, q11, q12, q44, out):
    nz,nxy = P.shape[1:]
    for index in prange(nz*nxy):
        z,j = index//nxy,index%nxy
        s0=s1=s2=s3=s4=s5=0.
        for face in range(max(0,z-1),min(nz-1,z+1)):
            a,b,c = e[0,face,j],e[1,face,j],e[2,face,j]
            s0 += .5*(c11*a+c12*(b+c))
            s1 += .5*(c11*b+c12*(a+c))
            s2 += .5*(c11*c+c12*(a+b))
            s3 += .5*(2*c44*e[3,face,j])
            s4 += .5*(2*c44*e[4,face,j])
            s5 += .5*(2*c44*e[5,face,j])
        x,y,w = P[0,z,j],P[1,z,j],P[2,z,j]
        out[0,z,j] = -(2*x*(s0*q11+(s1+s2)*q12)+q44*(s3*y+s4*w))
        out[1,z,j] = -(2*y*(s1*q11+(s0+s2)*q12)+q44*(s3*x+s5*w))
        out[2,z,j] = -(2*w*(s2*q11+(s0+s1)*q12)+q44*(s4*x+s5*y))


class ForceSolver:
    def __init__(self, el, workers=1):
        if numba is None:
            raise ImportError('The numba elastic backend requires numba and llvmlite: '+_IMPORT_ERROR)
        _validate_el(el)
        if not 1 <= workers <= numba.config.NUMBA_NUM_THREADS:
            raise ValueError('Invalid numba worker count')
        self.el,self.workers = el,workers
        ny,nxr,n,lo = el['Ny'],el['Nx']//2+1,el['n_elem'],el['film_lo']
        self.factors = [np.ascontiguousarray(el['_factors'][k].transpose(1,2,0,3,4)).reshape(ny*nxr,-1,3,3) for k in ('sub','cp','piv_inv')]
        self.rhs = np.empty((ny*nxr,n,3),np.complex128)
        self.dp,self.solution = np.empty_like(self.rhs),np.empty_like(self.rhs)
        self.src = np.empty((6,n-lo,ny,el['Nx']),np.float64)

    def close(self):
        pass

    def __call__(self, P, el=None):
        if P.dtype != np.float64:
            raise ValueError('Numba force requires float64 polarization')
        previous = numba.get_num_threads()
        numba.set_num_threads(self.workers)
        try:
            return self._evaluate(P, el)
        finally:
            numba.set_num_threads(previous)

    def _evaluate(self, P, el):
        el = _current_el(self.el, el, P)
        lo,n,ny,nx = el['film_lo'],el['n_elem'],el['Ny'],el['Nx']
        p = np.ascontiguousarray(P).reshape(3,P.shape[1],-1)
        q = el['Q11'],el['Q12'],el['Q44']
        # Coupled q/theta calls supply a new source on the same factored stack.
        static = el.get('eps_star_static')
        packed = (np.stack([static[i,j,lo:] for i,j in _ORDER]).reshape(6,n-lo+1,-1)
                  if static is not None else np.empty((6,0,0)))
        ebar = np.array([el['ebar'][i,j] for i,j in _ORDER])
        _source(p,*q,ebar,packed,static is not None,self.src.reshape(6,n-lo,-1))
        es = f3._rfft2(self.src)
        local = np.einsum('yxij,jeyx->eyxi',el['M_el'],es)
        rhs = np.zeros((n,ny,es.shape[-1],3),dtype=complex)
        rhs[lo:] = local[...,3:]
        # Face lo+e's lower node is reduced row lo+e-1. Without a substrate
        # (lo=0) face 0's lower node is the clamped, dropped node 0.
        skip = 0 if lo else 1
        rhs[lo-1+skip:-1] += local[skip:,...,:3]
        self.rhs[:] = rhs.transpose(1,2,0,3).reshape(self.rhs.shape)
        _solve(*self.factors,self.rhs,self.dp,self.solution)
        uh = np.zeros((3,n+1,ny,es.shape[-1]),dtype=complex)
        uh[:,1:] = self.solution.reshape(ny,es.shape[-1],n,3).transpose(3,2,0,1)
        uh = f3._rfft2(f3._irfft2(uh,(ny,nx)))
        strains = np.stack(fe._face_strain(uh[:,lo:-1],uh[:,lo+1:],el['K1'],el['K2'],el['hz']))
        e = f3._irfft2(strains,(ny,nx))-self.src
        # The flattened kernel must write a view of the returned array, even
        # when P is strided/Fortran ordered (e.g. after Fourier interpolation).
        # Keep ownership too: callers may retain previous force evaluations.
        out = np.empty(P.shape, dtype=np.float64, order="C")
        _force(p,e.reshape(6,n-lo,-1),el['C11'],el['C12'],el['C44'],*q,out.reshape(3,P.shape[1],-1))
        return out
