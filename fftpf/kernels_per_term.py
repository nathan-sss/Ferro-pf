"""Optional fused CPU kernels; acceleration requires Numba."""
import numpy as np
try:
    import numba
    from numba import njit, prange
except ImportError as exc:
    numba = None
    _IMPORT_ERROR = str(exc)
    # Allow module discovery without the optional dependency, never execution.
    def njit(**kwargs):
        return lambda function: function
    prange = range

from fftpf.pf2d_engine import ALPHA_KEYS, ALPHA_KEYS_8TH


@njit(inline="always", fastmath=False)
def _component_force(Pi, qi, qj, qk, a, has8):
    a1, a11, a12, a111, a112, a123, a1111, a1112, a1122, a1123 = a
    rj, rk = qj * qj, qk * qk
    s, rs, p = qj + qk, rj + rk, qj * qk
    c0 = a1 + a12 * s + a112 * rs + a123 * p
    c1 = 2 * a11 + 2 * a112 * s
    c2 = 3 * a111
    if has8:
        c0 = c0 + a1112 * (rj * qj + rk * qk) + a1123 * (p * s)
        c1 = c1 + 2 * a1122 * rs + 2 * a1123 * p
        c2 = c2 + 3 * a1112 * s
        g = ((4 * a1111 * qi + c2) * qi + c1) * qi + c0
    else:
        g = (c2 * qi + c1) * qi + c0
    return 2 * Pi * g


@njit(inline="always", fastmath=False)
def _coefficient(a, n):
    return a[0] if a.size == 1 else a[n]


@njit(parallel=True, cache=True, fastmath=False)
def _landau_force(P, coefficients, has8, out):
    # Each iteration owns one site; there are no reductions or shared writes.
    for n in prange(P.shape[1]):
        a = (_coefficient(coefficients[0], n),
             _coefficient(coefficients[1], n),
             _coefficient(coefficients[2], n),
             _coefficient(coefficients[3], n),
             _coefficient(coefficients[4], n),
             _coefficient(coefficients[5], n),
             _coefficient(coefficients[6], n),
             _coefficient(coefficients[7], n),
             _coefficient(coefficients[8], n),
             _coefficient(coefficients[9], n))
        p1, p2, p3 = P[0, n], P[1, n], P[2, n]
        q1, q2, q3 = p1 * p1, p2 * p2, p3 * p3
        out[0, n] = _component_force(p1, q1, q2, q3, a, has8)
        out[1, n] = _component_force(p2, q2, q1, q3, a, has8)
        out[2, n] = _component_force(p3, q3, q1, q2, a, has8)


def landau_force(P, alpha):
    """Evaluate the existing Landau derivative using independent CPU sites.

    P is float64, shaped (3, ...). Coefficients are real scalars or arrays
    broadcasting to P.shape[1:]. Non-C inputs and broadcast coefficient fields
    may require copies. The returned C array is independent of all inputs and
    later calls. Coefficients are read every call, so temperature updates apply.
    """
    if numba is None:
        raise ImportError("The numba Landau backend requires numba and llvmlite: " + _IMPORT_ERROR)
    P = np.asarray(P)
    if P.dtype != np.float64 or P.ndim < 2 or P.shape[0] != 3:
        raise ValueError("Numba Landau force requires float64 P with shape (3, ...)")
    values = [alpha[k] for k in ALPHA_KEYS]
    values += [alpha.get(k, 0.0) for k in ALPHA_KEYS_8TH]
    has8 = any(np.any(np.asarray(a) != 0.0) for a in values[6:])
    coefficients = _coefficients(values, P.shape[1:])
    field = np.ascontiguousarray(P).reshape(3, -1)
    out = np.empty(P.shape, dtype=np.float64)
    _landau_force(field, coefficients, has8, out.reshape(3, -1))
    return out


@njit(parallel=True, cache=True, fastmath=False)
def _green_spectrum(t, K, Om, zero, out):
    for n in prange(t.shape[1]):
        k = n % K.shape[1]  # Reuse the same operator for each ensemble member.
        k0, k1, k2 = K[0, k], K[1, k], K[2, k]
        v0 = k0*t[0,n] + k1*t[3,n] + k2*t[4,n]
        v1 = k0*t[3,n] + k1*t[1,n] + k2*t[5,n]
        v2 = k0*t[4,n] + k1*t[5,n] + k2*t[2,n]
        keep = not zero[k]
        u0 = -1j*(Om[0,0,k]*v0 + Om[0,1,k]*v1 + Om[0,2,k]*v2)*keep
        u1 = -1j*(Om[1,0,k]*v0 + Om[1,1,k]*v1 + Om[1,2,k]*v2)*keep
        u2 = -1j*(Om[2,0,k]*v0 + Om[2,1,k]*v1 + Om[2,2,k]*v2)*keep
        out[0,n] = (0.5j*(k0*u0 + k0*u0))*keep
        out[1,n] = (0.5j*(k1*u1 + k1*u1))*keep
        out[2,n] = (0.5j*(k2*u2 + k2*u2))*keep
        out[3,n] = (0.5j*(k0*u1 + k1*u0))*keep
        out[4,n] = (0.5j*(k0*u2 + k2*u0))*keep
        out[5,n] = (0.5j*(k1*u2 + k2*u1))*keep


def green_delta_eps(tau, K, Om, zero, shape, fwd, inv):
    """Fuse Green arithmetic between the existing six forward/inverse FFTs."""
    if numba is None:
        raise ImportError("The numba Green kernel requires numba and llvmlite: " + _IMPORT_ERROR)
    slots = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    spectrum = fwd(np.stack([tau[i, j] for i, j in slots]))
    transformed = np.empty(spectrum.shape, dtype=np.complex128)
    _green_spectrum(np.ascontiguousarray(spectrum).reshape(6, -1),
                    K.reshape(3, -1), Om.reshape(3, 3, -1), zero.ravel(),
                    transformed.reshape(6, -1))
    d6 = inv(transformed, shape)
    out = np.empty((3, 3) + d6.shape[1:], dtype=d6.dtype)
    for n, (i, j) in enumerate(slots):
        out[i, j] = d6[n]
        out[j, i] = d6[n]
    return out


def _real_field(field):
    if numba is None:
        raise ImportError("CPU kernels require numba and llvmlite: " + _IMPORT_ERROR)
    field = np.asarray(field)
    if field.dtype != np.float64:
        raise ValueError("CPU kernels require float64 fields")
    return np.ascontiguousarray(field)


def _coefficients(values, shape):
    arrays = []
    for value in values:
        a = np.asarray(value, dtype=np.float64)
        arrays.append(a.reshape(1) if a.ndim == 0 else np.broadcast_to(a, shape).ravel())
    return tuple(arrays)


@njit(parallel=True, cache=True, fastmath=False)
def _eigenstrain(P, Q, out):
    for n in prange(P.shape[1]):
        q11, q12, q44 = _coefficient(Q[0], n), _coefficient(Q[1], n), _coefficient(Q[2], n)
        x, y, z = P[0,n], P[1,n], P[2,n]
        xx, yy, zz = x*x, y*y, z*z
        out[0,0,n] = q11*xx + q12*(yy + zz)
        out[1,1,n] = q11*yy + q12*(xx + zz)
        out[2,2,n] = q11*zz + q12*(xx + yy)
        out[0,1,n] = out[1,0,n] = .5*q44*x*y
        out[0,2,n] = out[2,0,n] = .5*q44*x*z
        out[1,2,n] = out[2,1,n] = .5*q44*y*z


def eigenstrain(P, Q11, Q12, Q44):
    """Voigt-sourced Q44 convention: eps12 = 0.5 * Q44 * P1 * P2."""
    P = _real_field(P)
    if P.ndim < 1 or P.shape[0] != 3:
        raise ValueError("Polarization must have shape (3, ...)")
    out = np.empty((3, 3) + P.shape[1:], dtype=np.float64)
    _eigenstrain(P.reshape(3, -1), _coefficients((Q11, Q12, Q44), P.shape[1:]),
                 out.reshape(3, 3, -1))
    return out


@njit(parallel=True, cache=True, fastmath=False)
def _contract_cubic(e, C, out):
    for n in prange(e.shape[2]):
        c11, c12, c44 = _coefficient(C[0], n), _coefficient(C[1], n), _coefficient(C[2], n)
        x, y, z = e[0,0,n], e[1,1,n], e[2,2,n]
        out[0,0,n] = c11*x + c12*(y + z)
        out[1,1,n] = c11*y + c12*(x + z)
        out[2,2,n] = c11*z + c12*(x + y)
        out[0,1,n] = out[1,0,n] = 2*c44*e[0,1,n]
        out[0,2,n] = out[2,0,n] = 2*c44*e[0,2,n]
        out[1,2,n] = out[2,1,n] = 2*c44*e[1,2,n]


def contract_cubic(C11, C12, C44, e):
    """Cubic stiffness contraction for symmetric tensor fields."""
    e = _real_field(e)
    if e.shape[:2] != (3, 3):
        raise ValueError("Strain must have shape (3, 3, ...)")
    out = np.empty(e.shape, dtype=np.float64)
    _contract_cubic(e.reshape(3, 3, -1), _coefficients((C11, C12, C44), e.shape[2:]),
                    out.reshape(3, 3, -1))
    return out


@njit(parallel=True, cache=True, fastmath=False)
def _elastic_force(P, s, Q, out):
    for n in prange(P.shape[1]):
        q11, q12, q44 = _coefficient(Q[0], n), _coefficient(Q[1], n), _coefficient(Q[2], n)
        x, y, z = P[0,n], P[1,n], P[2,n]
        s11, s22, s33 = s[0,0,n], s[1,1,n], s[2,2,n]
        s12, s13, s23 = s[0,1,n], s[0,2,n], s[1,2,n]
        out[0,n] = -(2*x*(s11*q11 + (s22+s33)*q12) + q44*(s12*y + s13*z))
        out[1,n] = -(2*y*(s22*q11 + (s11+s33)*q12) + q44*(s12*x + s23*z))
        out[2,n] = -(2*z*(s33*q11 + (s11+s22)*q12) + q44*(s13*x + s23*y))


def elastic_force(P, sigma, Q11, Q12, Q44):
    """Contract symmetric stress with the derivative of eigenstrain."""
    P, sigma = _real_field(P), _real_field(sigma)
    if P.ndim < 1 or P.shape[0] != 3 or sigma.shape[:2] != (3, 3):
        raise ValueError("Expected polarization (3, ...) and stress (3, 3, ...)")
    if sigma.shape[2:] != P.shape[1:]:
        padding = (1,) * (P.ndim + 1 - sigma.ndim)
        sigma = sigma.reshape((3, 3) + padding + sigma.shape[2:])
        sigma = np.ascontiguousarray(np.broadcast_to(sigma, (3, 3) + P.shape[1:]))
    out = np.empty(P.shape, dtype=np.float64)
    _elastic_force(P.reshape(3, -1), sigma.reshape(3, 3, -1),
                   _coefficients((Q11, Q12, Q44), P.shape[1:]), out.reshape(3, -1))
    return out
