"""Opt-in fused step kernels for the 2D, bulk3d and film3d engines.

Selected only by ``execution.configure(workers, backend="numba", fused=True)``.
The NumPy and plain Numba backends never reach this module, so their results
are unchanged bit for bit. A fused step evaluates the SAME discrete scheme
(Landau + E_ext + Khachaturyan elasticity explicit, the linear gradient/
depolarisation family implicit or exact-OU), but it

* builds the explicit SBDF1/SBDF2/Langevin right-hand side in REAL space in
  one compiled pass (extrapolant, Landau force, E_ext, elastic force
  contraction, the 4P^n - P^(n-1) combination and the optional ``stab``
  term) and transforms it once, instead of transforming P^n, P^(n-1) and
  N(P*) separately;
* builds the six eigenstress components tau = C:eps*(P*) in one pass;
* contracts the Green operator per wavevector straight to the stress
  spectrum sigma_hat = C:delta_eps_hat - tau_hat (with the stress-free or
  clamped mean strain applied at k=0), so the inverse transform returns the
  stress itself;
* applies the per-mode 3x3 implicit inverse (or the OU decay + noise pair)
  in one per-wavevector pass, in place.

The film stepper (film_step) does the same for its RHS, adds the
short-circuit Poisson field per in-plane mode in k-space and solves the
tridiagonal gradient operator per mode; film_force_solver() fuses the direct
layered elastic force (RHS assembly + block Thomas on the rows the film
reads, closed-form real-field projection, face strains, stress contraction).

Because the transforms are linear these are the same equations, but the
floating-point association differs: a fused trajectory agrees with the
reference to round-off (measured <= 1e-15 relative per step, a few 1e-14
after 1000 steps), not bit for bit. Switching an existing production
checkpoint to it therefore needs the usual migration certificate and dt/2
audit, as for any backend change.

Coverage (anything else returns None and the caller continues on the
selected backend's ordinary path, unchanged): float64 P, scalar or (Nz,)
background kappa (no electrostatic SPM), E_ext a 3-vector or a full
component-first field, and either no elasticity or homogeneous cubic periodic
elasticity in the stress-free (2D, bulk3d) or clamped (bulk3d) mean-strain
mode, with or without a static eigenstrain, or the direct film stack.
Film-misfit 2D modes, the C(r) SPM paths and the 2D Langevin stepper on a
batched P (its (3, Ny, Nx) draw is not ensemble-aware) are not fused; film
Langevin keeps its own stepper and gets only the fused elastic force.

bulk3d options. depol_N (SPEC §11) needs no kernel of its own: the k = 0 entry
of D(k) reaches a step only through Minv/Minv2 or the OU pair. z_mode=
"plane_stress" (SPEC §12, Nz == 1) has its own two kernels: at KZ = 0 with the
i3 eigenstrain dropped only the 11, 22 and 12 stresses are nonzero, so three
components are transformed instead of six; sigma_13 = sigma_23 = sigma_33 = 0
are written exactly, where the reference has sigma_33 = 0 to round-off. An el
with any other z_mode, or a plane-stress el carrying eps_star_static or Nz != 1
(a hand-edited dict), is not fused; the ordinary path decides what to do with it.

Scratch arrays live in a per-thread cache keyed by shape (the film force
solver owns its own, per el). Every buffer is written before it is read
within one call, and the returned field is a fresh array that aliases no
scratch, so trajectories and ensemble members that share a thread cannot
see each other's data and a step carries no hidden state (checkpoint/restart
behaviour is unchanged).
"""
import threading

import numpy as np
import scipy.fft as spfft

from fftpf.kernels_per_term import _coefficient, _coefficients, _component_force, numba, njit, prange
from fftpf.pf2d_engine import ALPHA_KEYS, ALPHA_KEYS_8TH

if numba is None:  # pragma: no cover - exercised by the optional-dependency test
    from fftpf.kernels_per_term import _IMPORT_ERROR

_LOCAL = threading.local()
_MAX_BUFFERS = 16


def _require():
    if numba is None:
        raise ImportError("Fused kernels require numba and llvmlite: " + _IMPORT_ERROR)


def workspace(name, shape, dtype=np.float64):
    """A per-thread scratch array, reused across steps of the same shape.

    Contents are undefined on return; callers overwrite them fully."""
    cache = getattr(_LOCAL, "buffers", None)
    if cache is None:
        cache = _LOCAL.buffers = {}
    key = (name, tuple(shape), np.dtype(dtype).str)
    buf = cache.get(key)
    if buf is None:
        if len(cache) >= _MAX_BUFFERS:
            cache.clear()
        buf = cache[key] = np.empty(shape, dtype=dtype)
    return buf


def release_workspaces():
    """Drop this thread's scratch arrays (configure() calls it on switches)."""
    _LOCAL.buffers = {}


# --------------------------------------------------------------------------
# Compiled kernels. Every prange iteration owns one grid point or one
# wavevector of one ensemble member; nothing is reduced across iterations.
# --------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=False)
def _eigenstress(Pn, Pm, two_level, Q, static, has_static, c11, c12, c44, out):
    """out[6, n] = C : (eps*(P*) + static), slots 00,11,22,01,02,12."""
    ms = static.shape[2]
    for n in prange(Pn.shape[1]):
        if two_level:
            x = 2 * Pn[0, n] - Pm[0, n]
            y = 2 * Pn[1, n] - Pm[1, n]
            z = 2 * Pn[2, n] - Pm[2, n]
        else:
            x, y, z = Pn[0, n], Pn[1, n], Pn[2, n]
        q11, q12, q44 = _coefficient(Q[0], n), _coefficient(Q[1], n), _coefficient(Q[2], n)
        xx, yy, zz = x * x, y * y, z * z
        e00 = q11 * xx + q12 * (yy + zz)
        e11 = q11 * yy + q12 * (xx + zz)
        e22 = q11 * zz + q12 * (xx + yy)
        e01 = .5 * q44 * x * y
        e02 = .5 * q44 * x * z
        e12 = .5 * q44 * y * z
        if has_static:
            m = n % ms
            e00 += static[0, 0, m]
            e11 += static[1, 1, m]
            e22 += static[2, 2, m]
            e01 += static[0, 1, m]
            e02 += static[0, 2, m]
            e12 += static[1, 2, m]
        out[0, n] = c11 * e00 + c12 * (e11 + e22)
        out[1, n] = c11 * e11 + c12 * (e00 + e22)
        out[2, n] = c11 * e22 + c12 * (e00 + e11)
        out[3, n] = 2 * c44 * e01
        out[4, n] = 2 * c44 * e02
        out[5, n] = 2 * c44 * e12


@njit(inline="always", fastmath=False)
def _rc(a, z):
    """Real coefficient times complex value, without the 0*imag cross terms."""
    return complex(a * z.real, a * z.imag)


@njit(parallel=True, cache=True, fastmath=False)
def _green_stress(t, K, Om, zero, c11, c12, c44, stress_free):
    """In place: tau_hat -> sigma_hat = C:delta_eps_hat - tau_hat (+ mean term).

    delta_eps_hat_ij = 0.5*(k_i w_j + k_j w_i), w = Omega (k . tau_hat), which
    is the reference sym(i k (x) u_hat) with u_hat = -i Omega k . tau_hat. At
    every zero-K mode delta_eps_hat = 0; at k=0 the stress-free mean strain
    ebar = <eps*> cancels the mean eigenstress exactly, the clamped one
    (ebar = 0) leaves -tau_hat."""
    M = K.shape[1]
    for n in prange(t.shape[1]):
        k = n % M
        t0, t1, t2, t3, t4, t5 = t[0, n], t[1, n], t[2, n], t[3, n], t[4, n], t[5, n]
        if zero[k]:
            if k == 0 and stress_free:
                for c in range(6):
                    t[c, n] = 0j
            else:
                for c in range(6):
                    t[c, n] = -t[c, n]
            continue
        k0, k1, k2 = K[0, k], K[1, k], K[2, k]
        v0 = _rc(k0, t0) + _rc(k1, t3) + _rc(k2, t4)
        v1 = _rc(k0, t3) + _rc(k1, t1) + _rc(k2, t5)
        v2 = _rc(k0, t4) + _rc(k1, t5) + _rc(k2, t2)
        w0 = _rc(Om[k, 0, 0], v0) + _rc(Om[k, 0, 1], v1) + _rc(Om[k, 0, 2], v2)
        w1 = _rc(Om[k, 1, 0], v0) + _rc(Om[k, 1, 1], v1) + _rc(Om[k, 1, 2], v2)
        w2 = _rc(Om[k, 2, 0], v0) + _rc(Om[k, 2, 1], v1) + _rc(Om[k, 2, 2], v2)
        d00 = _rc(k0, w0)
        d11 = _rc(k1, w1)
        d22 = _rc(k2, w2)
        d01 = _rc(.5, _rc(k0, w1) + _rc(k1, w0))
        d02 = _rc(.5, _rc(k0, w2) + _rc(k2, w0))
        d12 = _rc(.5, _rc(k1, w2) + _rc(k2, w1))
        t[0, n] = _rc(c11, d00) + _rc(c12, d11 + d22) - t0
        t[1, n] = _rc(c11, d11) + _rc(c12, d00 + d22) - t1
        t[2, n] = _rc(c11, d22) + _rc(c12, d00 + d11) - t2
        t[3, n] = _rc(2 * c44, d01) - t3
        t[4, n] = _rc(2 * c44, d02) - t4
        t[5, n] = _rc(2 * c44, d12) - t5


@njit(parallel=True, cache=True, fastmath=False)
def _eigenstress_ps(Pn, Pm, two_level, Q, c11, c12, c44, out):
    """SPEC §12: out[3, n] = C_ps : eps*_in(P*), slots 00,11,01. c11, c12 are the
    reduced C11_ps, C12_ps; eps*_33, eps*_13, eps*_23 do not enter."""
    for n in prange(Pn.shape[1]):
        if two_level:
            x = 2 * Pn[0, n] - Pm[0, n]
            y = 2 * Pn[1, n] - Pm[1, n]
            z = 2 * Pn[2, n] - Pm[2, n]
        else:
            x, y, z = Pn[0, n], Pn[1, n], Pn[2, n]
        q11, q12, q44 = _coefficient(Q[0], n), _coefficient(Q[1], n), _coefficient(Q[2], n)
        xx, yy, zz = x * x, y * y, z * z
        e00 = q11 * xx + q12 * (yy + zz)
        e11 = q11 * yy + q12 * (xx + zz)
        out[0, n] = c11 * e00 + c12 * e11
        out[1, n] = c11 * e11 + c12 * e00
        out[2, n] = c44 * q44 * x * y          # 2 C44 eps*_12, eps*_12 = q44 x y / 2


@njit(parallel=True, cache=True, fastmath=False)
def _green_stress_ps(t, K, Om, zero, c11, c12, c44, stress_free):
    """SPEC §12, in place on slots 00,11,01: tau_hat -> sigma_hat =
    C_ps:delta_eps_hat - tau_hat (+ mean term), _green_stress restricted to the
    in-plane block (k3 = 0, Omega from C_ps, whose 3-row decouples). With e_33 =
    -(C12/C11)(e_11 + e_22) the full C gives sigma_ab = C_ps:e_ab, so these are
    the reference's in-plane stresses and sigma_i3 = 0."""
    M = K.shape[1]
    for n in prange(t.shape[1]):
        k = n % M
        t0, t1, t3 = t[0, n], t[1, n], t[2, n]
        if zero[k]:
            if k == 0 and stress_free:
                t[0, n], t[1, n], t[2, n] = 0j, 0j, 0j
            else:
                t[0, n], t[1, n], t[2, n] = -t0, -t1, -t3
            continue
        k0, k1 = K[0, k], K[1, k]
        v0 = _rc(k0, t0) + _rc(k1, t3)
        v1 = _rc(k0, t3) + _rc(k1, t1)
        w0 = _rc(Om[k, 0, 0], v0) + _rc(Om[k, 0, 1], v1)
        w1 = _rc(Om[k, 1, 0], v0) + _rc(Om[k, 1, 1], v1)
        d00 = _rc(k0, w0)
        d11 = _rc(k1, w1)
        d01 = _rc(.5, _rc(k0, w1) + _rc(k1, w0))
        t[0, n] = _rc(c11, d00) + _rc(c12, d11) - t0
        t[1, n] = _rc(c11, d11) + _rc(c12, d00) - t1
        t[2, n] = _rc(2 * c44, d01) - t3


@njit(parallel=True, cache=True, fastmath=False)
def _rhs(Pn, Pm, two_level, cA, h, hs, a, has8, E, sig, has_el, Q, out):
    """Real-space explicit RHS, one pass per grid point.

    N = dF_bulk/dP(P*) - E_ext [+ dF_el/dP(P*, sigma)]
    two_level (SBDF2): out = 4 P^n - P^(n-1) - h N + hs P*, P* = 2P^n - P^(n-1)
    otherwise (SBDF1/Langevin): out = cA P^n - h N, P* = P^n."""
    me = E.shape[1]
    for n in prange(Pn.shape[1]):
        if two_level:
            x = 2 * Pn[0, n] - Pm[0, n]
            y = 2 * Pn[1, n] - Pm[1, n]
            z = 2 * Pn[2, n] - Pm[2, n]
        else:
            x, y, z = Pn[0, n], Pn[1, n], Pn[2, n]
        c = (_coefficient(a[0], n), _coefficient(a[1], n), _coefficient(a[2], n),
             _coefficient(a[3], n), _coefficient(a[4], n), _coefficient(a[5], n),
             _coefficient(a[6], n), _coefficient(a[7], n), _coefficient(a[8], n),
             _coefficient(a[9], n))
        xx, yy, zz = x * x, y * y, z * z
        m = n % me
        f0 = _component_force(x, xx, yy, zz, c, has8) - E[0, m]
        f1 = _component_force(y, yy, xx, zz, c, has8) - E[1, m]
        f2 = _component_force(z, zz, xx, yy, c, has8) - E[2, m]
        if has_el:
            q11, q12, q44 = _coefficient(Q[0], n), _coefficient(Q[1], n), _coefficient(Q[2], n)
            s00, s11, s22 = sig[0][n], sig[1][n], sig[2][n]
            s01, s02, s12 = sig[3][n], sig[4][n], sig[5][n]
            f0 = f0 - (2 * x * (s00 * q11 + (s11 + s22) * q12) + q44 * (s01 * y + s02 * z))
            f1 = f1 - (2 * y * (s11 * q11 + (s00 + s22) * q12) + q44 * (s01 * x + s12 * z))
            f2 = f2 - (2 * z * (s22 * q11 + (s00 + s11) * q12) + q44 * (s02 * x + s12 * y))
        if two_level:
            out[0, n] = 4 * Pn[0, n] - Pm[0, n] - h * f0 + hs * x
            out[1, n] = 4 * Pn[1, n] - Pm[1, n] - h * f1 + hs * y
            out[2, n] = 4 * Pn[2, n] - Pm[2, n] - h * f2 + hs * z
        else:
            out[0, n] = cA * Pn[0, n] - h * f0
            out[1, n] = cA * Pn[1, n] - h * f1
            out[2, n] = cA * Pn[2, n] - h * f2


@njit(parallel=True, cache=True, fastmath=False)
def _apply_inverse(X, Mi):
    """In place X <- Mi(k) X per wavevector (Mi real (M,3,3))."""
    M = Mi.shape[0]
    for n in prange(X.shape[1]):
        k = n % M
        r0, r1, r2 = X[0, n], X[1, n], X[2, n]
        X[0, n] = _rc(Mi[k, 0, 0], r0) + _rc(Mi[k, 0, 1], r1) + _rc(Mi[k, 0, 2], r2)
        X[1, n] = _rc(Mi[k, 1, 0], r0) + _rc(Mi[k, 1, 1], r1) + _rc(Mi[k, 1, 2], r2)
        X[2, n] = _rc(Mi[k, 2, 0], r0) + _rc(Mi[k, 2, 1], r1) + _rc(Mi[k, 2, 2], r2)


@njit(parallel=True, cache=True, fastmath=False)
def _apply_ou(X, D, A, eta):
    """In place X <- D(k) X + A(k) eta(k) per wavevector (exact OU update)."""
    M = D.shape[0]
    me = eta.shape[1]
    for n in prange(X.shape[1]):
        k = n % M
        e = n % me
        r0, r1, r2 = X[0, n], X[1, n], X[2, n]
        g0, g1, g2 = eta[0, e], eta[1, e], eta[2, e]
        d0 = _rc(D[k, 0, 0], r0) + _rc(D[k, 0, 1], r1) + _rc(D[k, 0, 2], r2)
        d1 = _rc(D[k, 1, 0], r0) + _rc(D[k, 1, 1], r1) + _rc(D[k, 1, 2], r2)
        d2 = _rc(D[k, 2, 0], r0) + _rc(D[k, 2, 1], r1) + _rc(D[k, 2, 2], r2)
        n0 = _rc(A[k, 0, 0], g0) + _rc(A[k, 0, 1], g1) + _rc(A[k, 0, 2], g2)
        n1 = _rc(A[k, 1, 0], g0) + _rc(A[k, 1, 1], g1) + _rc(A[k, 1, 2], g2)
        n2 = _rc(A[k, 2, 0], g0) + _rc(A[k, 2, 1], g1) + _rc(A[k, 2, 2], g2)
        X[0, n] = d0 + n0
        X[1, n] = d1 + n1
        X[2, n] = d2 + n2


# --------------------------------------------------------------------------
# Python drivers
# --------------------------------------------------------------------------
def _field(value, P, space, what):
    """Component-first field as (lead, m) with m in {1, |space|, |P.shape[1:]|},
    read at index n % m; None when the shape is not one the kernel covers."""
    v = np.asarray(value, dtype=np.float64)
    lead = v.shape[:len(what)] if v.ndim >= len(what) else None
    if lead != what:
        return None
    rest = v.shape[len(what):]
    if rest == () and what == (3,):
        return np.ascontiguousarray(v).reshape(3, 1)
    if rest not in (space, P.shape[1:]):
        return None
    return np.ascontiguousarray(v).reshape(what + (-1,))


def _operator(op, kshape):
    op = np.asarray(op)
    if op.dtype != np.float64 or op.shape != kshape + (3, 3):
        return None
    return np.ascontiguousarray(op).reshape(-1, 3, 3)


def _elastic(el, P, space):
    """Resolve the elastic inputs the fused path covers, else None."""
    if el.get("spm"):
        return None
    plane_stress = False
    if "k0" in el:                                     # bulk3d.setup_elastic_3d
        if el["k0"] not in ("stress_free", "clamped"):
            return None
        stress_free = el["k0"] == "stress_free"
        Q = (el["Q1111"], el["Q1122"], 4 * np.asarray(el["Q1212"]))
        z_mode = el.get("z_mode", "generalized_plane_strain")
        if z_mode == "plane_stress":                   # SPEC §12
            if space[0] != 1 or el.get("eps_star_static") is not None:
                return None
            plane_stress = True
        elif z_mode != "generalized_plane_strain":
            return None
    else:                                              # elastic.setup_elastic
        if el.get("eps_m") is not None or el.get("film_full"):
            return None
        stress_free = True
        Q = (el["Q11"], el["Q12"], el["Q44"])
    C = [np.asarray(el[k]) for k in (("C11_ps", "C12_ps", "C44") if plane_stress
                                     else ("C11", "C12", "C44"))]
    if any(c.ndim for c in C):
        return None
    static = el.get("eps_star_static")
    if static is None:
        static = np.zeros((3, 3, 1))
        has_static = False
    else:
        static = _field(static, P, space, (3, 3))
        if static is None:
            return None
        has_static = True
    zero = np.ascontiguousarray(el["zero_k"]).reshape(-1)
    K = np.ascontiguousarray(el["K"], dtype=np.float64).reshape(3, -1)
    Om = np.ascontiguousarray(el["Omega"], dtype=np.float64).reshape(-1, 3, 3)
    return dict(Q=_coefficients(Q, P.shape[1:]), static=static, has_static=has_static,
                C=tuple(float(c) for c in C), stress_free=stress_free, K=K, Om=Om, zero=zero,
                plane_stress=plane_stress)


# With ONE worker, two in-plane axes are inverse-transformed one component at
# a time: scipy's batched multi-axis c2r on a (c, Ny, Nxr) stack measured up to
# 1.5x slower per step than c separate calls (2D 256^2; even at 512^2). With
# several workers the batched call won by ~20% (A/B in docs/CPU_PARALLELISM.md),
# as it does for three axes. A switch only so the choice can be timed.
PER_COMPONENT_2D = True


def _per_component(axes, workers):
    return PER_COMPONENT_2D and len(axes) == 2 and workers == 1


def _inverse(X, space, axes, workers):
    """Inverse real transform of each leading component, as a tuple of flat arrays."""
    if _per_component(axes, workers):
        return tuple(spfft.irfftn(X[i], s=space, axes=axes, workers=workers,
                                  overwrite_x=True).reshape(-1) for i in range(X.shape[0]))
    full = spfft.irfftn(X, s=space, axes=axes, workers=workers, overwrite_x=True)
    return tuple(full[i].reshape(-1) for i in range(X.shape[0]))


def periodic_step(kind, P, P_prev, alpha, ops, op, dt, L, E_ext, el, stab, workers, ndim,
                  rng=None, noise_shape=None):
    """One fused periodic step, or None when the inputs are outside coverage.

    kind: "sbdf1" (op = Minv), "sbdf2" (op = Minv2, P_prev = P^(n-1)) or
    "langevin" (op = OU dict, rng the caller's Generator, noise_shape the
    reference draw shape). ndim: 2 (solver) or 3 (bulk3d) spatial axes."""
    _require()
    if ops.get("dkappa") is not None or P.dtype != np.float64 or P.shape[0] != 3:
        return None
    space = P.shape[-ndim:]
    kshape = space[:-1] + (space[-1] // 2 + 1,)
    two_level = kind == "sbdf2"
    if two_level and (P_prev.shape != P.shape or P_prev.dtype != np.float64):
        return None
    E = _field(E_ext, P, space, (3,))
    if E is None:
        return None
    if kind == "langevin":
        if tuple(noise_shape) != P.shape:
            # solver.step_langevin draws (3, Ny, Nx) even for a batched P; how
            # that draw meets the batch is the reference's business, not ours.
            return None
        D, A = _operator(op["decay"], kshape), _operator(op["noise_amp"], kshape)
        if D is None or A is None:
            return None
    else:
        Mi = _operator(op, kshape)
        if Mi is None:
            return None
    mech = None
    if el is not None:
        mech = _elastic(el, P, space)
        if mech is None:
            return None

    axes = tuple(range(-ndim, 0))
    if ndim == 3 and space[0] == 1:
        # Nz == 1: the z transform is the identity (length 1, unit factor), so it
        # is skipped, which also selects the per-component 2D inverse at one worker.
        axes, space = axes[1:], space[1:]
    Pn = np.ascontiguousarray(P).reshape(3, -1)
    Pm = np.ascontiguousarray(P_prev).reshape(3, -1) if two_level else Pn
    values = [alpha[k] for k in ALPHA_KEYS] + [alpha.get(k, 0.0) for k in ALPHA_KEYS_8TH]
    has8 = any(np.any(np.asarray(a) != 0.0) for a in values[6:])
    coeffs = _coefficients(values, P.shape[1:])

    if mech is None:
        sig, Q = (np.empty(1),) * 6, _coefficients((0., 0., 0.), P.shape[1:])
    elif mech["plane_stress"]:
        tau = workspace("tau_ps", (3,) + P.shape[1:])
        _eigenstress_ps(Pn, Pm, two_level, mech["Q"], *mech["C"], tau.reshape(3, -1))
        spectrum = spfft.rfftn(tau, axes=axes, workers=workers)
        _green_stress_ps(spectrum.reshape(3, -1), mech["K"], mech["Om"], mech["zero"],
                         *mech["C"], mech["stress_free"])
        s00, s11, s01 = _inverse(spectrum, space, axes, workers)
        none = workspace("sigma_i3", s00.shape)
        none.fill(0.0)
        sig = (s00, s11, none, s01, none, none)
        Q = mech["Q"]
    else:
        tau = workspace("tau", (6,) + P.shape[1:])
        _eigenstress(Pn, Pm, two_level, mech["Q"], mech["static"], mech["has_static"],
                     *mech["C"], tau.reshape(6, -1))
        spectrum = spfft.rfftn(tau, axes=axes, workers=workers)
        _green_stress(spectrum.reshape(6, -1), mech["K"], mech["Om"], mech["zero"],
                      *mech["C"], mech["stress_free"])
        sig = _inverse(spectrum, space, axes, workers)
        Q = mech["Q"]

    if kind == "sbdf2":
        h, hs, cA = 2 * dt * L, (2 * dt * L * stab if stab else 0.0), 0.0
    elif kind == "sbdf1":
        h, hs, cA = dt * L, 0.0, (1.0 + dt * L * stab if stab else 1.0)
    else:
        h, hs, cA = dt * L, 0.0, 1.0
    R = workspace("rhs", P.shape)
    _rhs(Pn, Pm, two_level, cA, h, hs, coeffs, has8, E, sig, mech is not None, Q,
         R.reshape(3, -1))
    X = spfft.rfftn(R, axes=axes, workers=workers)
    if kind == "langevin":
        eta = workspace("eta", noise_shape)
        rng.standard_normal(out=eta)
        eta_k = spfft.rfftn(eta, axes=axes, workers=workers)
        _apply_ou(X.reshape(3, -1), D, A, eta_k.reshape(3, -1))
    else:
        _apply_inverse(X.reshape(3, -1), Mi)
    if not _per_component(axes, workers):
        return spfft.irfftn(X, s=space, axes=axes, workers=workers, overwrite_x=True)
    out = np.empty(P.shape)
    for i in range(3):
        out[i] = spfft.irfftn(X[i], s=space, axes=axes, workers=workers, overwrite_x=True)
    return out


# --------------------------------------------------------------------------
# Film (direct layered stack): fused force-only elastic solve. Same assembled,
# factored operator and block-Thomas recurrence as film_elastic_numba; the
# per-mode RHS assembly (M_el contraction), the reduced-solution scatter, the
# real-field (Hermitian) projection of the displacement spectrum and the face
# strains are one compiled per-wavevector pass, so the two full-stack
# displacement transforms of the reference projection are replaced by its
# exact closed form: on the kx = 0 (and even-Nx Nyquist) column,
# u(ky) <- (u(ky) + conj(u(-ky))) / 2, every other mode unchanged.
# --------------------------------------------------------------------------
@njit(inline="always", fastmath=False)
def _film_row(es, M, lo, k, m, c):
    """Reduced-system RHS row m, component c: upper-node part of face m-lo
    plus lower-node part of face m-lo+1 (film_elastic_numba's assembly)."""
    F = es.shape[1]
    a = 0j
    e = m - lo
    if 0 <= e < F:
        for j in range(6):
            a += M[k, 3 + c, j] * es[j, e, k]
    e = m - lo + 1
    if 0 <= e < F:
        for j in range(6):
            a += M[k, c, j] * es[j, e, k]
    return a


@njit(parallel=True, cache=True, fastmath=False)
def _film_solve(es, M, sub, cp, inv, lo, out):
    """Assemble each mode's RHS on the fly and run the block-Thomas recurrence
    of film_elastic_numba._solve on the rows the film strains read.

    Rows below lo-1 have an exactly zero RHS (zero substrate eigenstrain, as
    _validate_el requires), so their forward-sweep values are exact zeros and
    the sweep can start at lo-1 with the same result; the back substitution
    stops at lo-1 because no film face reads a lower node. Rows < lo-1 of
    out are left unwritten."""
    n = out.shape[1]
    start = lo - 1 if lo > 0 else 0
    for k in prange(out.shape[0]):
        for z in range(start, n):
            v0 = _film_row(es, M, lo, k, z, 0)
            v1 = _film_row(es, M, lo, k, z, 1)
            v2 = _film_row(es, M, lo, k, z, 2)
            if z > start:
                d0, d1, d2 = out[k, z - 1, 0], out[k, z - 1, 1], out[k, z - 1, 2]
                v0 -= sub[k, z - 1, 0, 0] * d0 + sub[k, z - 1, 0, 1] * d1 + sub[k, z - 1, 0, 2] * d2
                v1 -= sub[k, z - 1, 1, 0] * d0 + sub[k, z - 1, 1, 1] * d1 + sub[k, z - 1, 1, 2] * d2
                v2 -= sub[k, z - 1, 2, 0] * d0 + sub[k, z - 1, 2, 1] * d1 + sub[k, z - 1, 2, 2] * d2
            for i in range(3):
                out[k, z, i] = inv[k, z, i, 0] * v0 + inv[k, z, i, 1] * v1 + inv[k, z, i, 2] * v2
        for z in range(n - 2, start - 1, -1):
            x0, x1, x2 = out[k, z + 1, 0], out[k, z + 1, 1], out[k, z + 1, 2]
            for i in range(3):
                out[k, z, i] = out[k, z, i] - (cp[k, z, i, 0] * x0 + cp[k, z, i, 1] * x1
                                               + cp[k, z, i, 2] * x2)


@njit(inline="always", fastmath=False)
def _film_node(u, k, kp, sym, g, c):
    if g == 0:                     # clamped substrate bottom, dropped from the system
        return 0j
    v = u[k, g - 1, c]
    if sym:
        w = u[kp, g - 1, c]
        v = .5 * (v + complex(w.real, -w.imag))
    return v


@njit(parallel=True, cache=True, fastmath=False)
def _film_strain(u, K1, K2, hz, lo, ny, nxr, nyquist, out):
    """Face strains (00,11,22,01,02,12) of the film faces from the reduced solution."""
    F = out.shape[1]
    for k in prange(u.shape[0]):
        y, x = k // nxr, k % nxr
        sym = x == 0 or (nyquist and x == nxr - 1)
        kp = ((ny - y) % ny) * nxr + x
        a, b = complex(0., K1[k]), complex(0., K2[k])
        for f in range(F):
            g = lo + f
            u1n = _film_node(u, k, kp, sym, g, 0)
            u2n = _film_node(u, k, kp, sym, g, 1)
            u3n = _film_node(u, k, kp, sym, g, 2)
            u1p = _film_node(u, k, kp, sym, g + 1, 0)
            u2p = _film_node(u, k, kp, sym, g + 1, 1)
            u3p = _film_node(u, k, kp, sym, g + 1, 2)
            out[0, f, k] = a * (u1n + u1p) / 2
            out[1, f, k] = b * (u2n + u2p) / 2
            out[2, f, k] = (u3p - u3n) / hz
            out[3, f, k] = (b * (u1n + u1p) / 2 + a * (u2n + u2p) / 2) / 2
            out[4, f, k] = ((u1p - u1n) / hz + a * (u3n + u3p) / 2) / 2
            out[5, f, k] = ((u2p - u2n) / hz + b * (u3n + u3p) / 2) / 2


@njit(parallel=True, cache=True, fastmath=False)
def _film_force(P, er, src, c11, c12, c44, q11, q12, q44, out):
    """film_elastic_numba._force with the elastic strain e = er - src formed inline."""
    nz, nxy = P.shape[1:]
    for index in prange(nz * nxy):
        z, j = index // nxy, index % nxy
        s0 = s1 = s2 = s3 = s4 = s5 = 0.
        for face in range(max(0, z - 1), min(nz - 1, z + 1)):
            a = er[0, face, j] - src[0, face, j]
            b = er[1, face, j] - src[1, face, j]
            c = er[2, face, j] - src[2, face, j]
            s0 += .5 * (c11 * a + c12 * (b + c))
            s1 += .5 * (c11 * b + c12 * (a + c))
            s2 += .5 * (c11 * c + c12 * (a + b))
            s3 += .5 * (2 * c44 * (er[3, face, j] - src[3, face, j]))
            s4 += .5 * (2 * c44 * (er[4, face, j] - src[4, face, j]))
            s5 += .5 * (2 * c44 * (er[5, face, j] - src[5, face, j]))
        x, y, w = P[0, z, j], P[1, z, j], P[2, z, j]
        out[0, z, j] = -(2 * x * (s0 * q11 + (s1 + s2) * q12) + q44 * (s3 * y + s4 * w))
        out[1, z, j] = -(2 * y * (s1 * q11 + (s0 + s2) * q12) + q44 * (s3 * x + s5 * w))
        out[2, z, j] = -(2 * w * (s2 * q11 + (s0 + s1) * q12) + q44 * (s4 * x + s5 * y))


def film_force_solver(el, workers):
    """A film_elastic_numba.ForceSolver whose evaluation uses the fused passes."""
    _require()
    from fftpf import film_elastic_force_numba as fen

    class FusedFilmForceSolver(fen.ForceSolver):
        def __init__(self, el, workers):
            super().__init__(el, workers)
            ny, nxr, n, lo = el['Ny'], el['Nx'] // 2 + 1, el['n_elem'], el['film_lo']
            self.M = np.ascontiguousarray(el['M_el']).reshape(ny * nxr, 6, 6)
            self.K1 = np.ascontiguousarray(el['K1'], dtype=np.float64).reshape(-1)
            self.K2 = np.ascontiguousarray(el['K2'], dtype=np.float64).reshape(-1)
            self.strain = np.empty((6, n - lo, ny, nxr), np.complex128)
            self.rhs = self.dp = None      # assembled on the fly in _film_solve

        def _evaluate(self, P, el):
            from fftpf import film_engine as f3
            el = fen._current_el(self.el, el, P)
            lo, n, ny, nx = el['film_lo'], el['n_elem'], el['Ny'], el['Nx']
            nxr = nx // 2 + 1
            p = np.ascontiguousarray(P).reshape(3, P.shape[1], -1)
            q = el['Q11'], el['Q12'], el['Q44']
            static = el.get('eps_star_static')
            packed = (np.stack([static[i, j, lo:] for i, j in fen._ORDER]).reshape(6, n - lo + 1, -1)
                      if static is not None else np.empty((6, 0, 0)))
            ebar = np.array([el['ebar'][i, j] for i, j in fen._ORDER])
            src = self.src.reshape(6, n - lo, -1)
            fen._source(p, *q, ebar, packed, static is not None, src)
            es = spfft.rfft2(self.src, axes=(-2, -1), workers=f3.FFT_WORKERS)
            _film_solve(es.reshape(6, n - lo, -1), self.M, *self.factors, lo, self.solution)
            _film_strain(self.solution, self.K1, self.K2, float(el['hz']), lo, ny, nxr,
                         nx % 2 == 0, self.strain.reshape(6, n - lo, -1))
            er = spfft.irfft2(self.strain, s=(ny, nx), axes=(-2, -1), workers=f3.FFT_WORKERS,
                              overwrite_x=True)
            out = np.empty(P.shape, dtype=np.float64, order="C")
            _film_force(p, er.reshape(6, n - lo, -1), src, el['C11'], el['C12'], el['C44'], *q,
                        out.reshape(3, P.shape[1], -1))
            return out

    return FusedFilmForceSolver(el, workers)


# --------------------------------------------------------------------------
# Film stepper: real-space explicit RHS (Landau - E_ext + qw-rescaled elastic
# force, SBDF combination, stab) transformed once; the short-circuit Poisson
# field is solved per in-plane mode from rfft2(P*) and added to the RHS
# spectrum directly (its real-space round trip is skipped); the scalar
# tridiagonal implicit gradient is solved per mode in place.
# --------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=False)
def _film_pstar(Pn, Pm, out):
    for n in prange(Pn.shape[1]):
        out[0, n] = 2 * Pn[0, n] - Pm[0, n]
        out[1, n] = 2 * Pn[1, n] - Pm[1, n]
        out[2, n] = 2 * Pn[2, n] - Pm[2, n]


@njit(parallel=True, cache=True, fastmath=False)
def _film_step_rhs(Pn, Pm, two_level, cA, h, hs, a, has8, E, Fel, has_el, qwr, nxy, out):
    me = E.shape[1]
    for n in prange(Pn.shape[1]):
        if two_level:
            x = 2 * Pn[0, n] - Pm[0, n]
            y = 2 * Pn[1, n] - Pm[1, n]
            z = 2 * Pn[2, n] - Pm[2, n]
        else:
            x, y, z = Pn[0, n], Pn[1, n], Pn[2, n]
        c = (_coefficient(a[0], n), _coefficient(a[1], n), _coefficient(a[2], n),
             _coefficient(a[3], n), _coefficient(a[4], n), _coefficient(a[5], n),
             _coefficient(a[6], n), _coefficient(a[7], n), _coefficient(a[8], n),
             _coefficient(a[9], n))
        xx, yy, zz = x * x, y * y, z * z
        m = n % me
        f0 = _component_force(x, xx, yy, zz, c, has8) - E[0, m]
        f1 = _component_force(y, yy, xx, zz, c, has8) - E[1, m]
        f2 = _component_force(z, zz, xx, yy, c, has8) - E[2, m]
        if has_el:
            w = qwr[n // nxy]
            f0 = f0 + Fel[0, n] * w
            f1 = f1 + Fel[1, n] * w
            f2 = f2 + Fel[2, n] * w
        if two_level:
            out[0, n] = 4 * Pn[0, n] - Pm[0, n] - h * f0 + hs * x
            out[1, n] = 4 * Pn[1, n] - Pm[1, n] - h * f1 + hs * y
            out[2, n] = 4 * Pn[2, n] - Pm[2, n] - h * f2 + hs * z
        else:
            out[0, n] = cA * Pn[0, n] - h * f0
            out[1, n] = cA * Pn[1, n] - h * f1
            out[2, n] = cA * Pn[2, n] - h * f2


@njit(inline="always", fastmath=False)
def _cdiv(z, p):
    return complex(z.real / p, z.imag / p)


@njit(parallel=True, cache=True, fastmath=False)
def _film_poisson_add(Pk, X, ka, kb, hz, sub, cp, piv, h, phi):
    """X += h * E_k(P*_k): film3d.solve_poisson_factored's per-mode algebra
    (Dirichlet interior Thomas, energy-conjugate end rows), mode by mode."""
    nz, K = Pk.shape[1], Pk.shape[2]
    ni = nz - 2
    for k in prange(K):
        a, b = complex(0., ka[k]), complex(0., kb[k])
        for z in range(nz):
            phi[k, z] = 0j
        for i in range(ni):                       # forward elimination, rhs = -source
            z = i + 1
            d3 = (Pk[2, z + 1, k] - Pk[2, z - 1, k]) / (2 * hz)
            r = -(a * Pk[0, z, k] + b * Pk[1, z, k] + d3)
            if i:
                r = r - sub[i - 1] * phi[k, z - 1]
            phi[k, z] = _cdiv(r, piv[i, k])
        for i in range(ni - 2, -1, -1):           # back substitution
            phi[k, i + 1] = phi[k, i + 1] - cp[i, k] * phi[k, i + 2]
        for z in range(nz):
            p = phi[k, z]
            X[0, z, k] += h * (-(a * p))
            X[1, z, k] += h * (-(b * p))
            if z == 0:
                e3 = -phi[k, 1] / hz
            elif z == nz - 1:
                e3 = phi[k, nz - 2] / hz
            else:
                e3 = -((phi[k, z + 1] - phi[k, z - 1]) / (2 * hz))
            X[2, z, k] += h * e3


@njit(parallel=True, cache=True, fastmath=False)
def _film_tri_solve(X, sub, cp, piv):
    """In place, per component and mode: film3d.solve_tridiag_factored."""
    nz, K = X.shape[1], X.shape[2]
    for idx in prange(3 * K):
        c, k = idx // K, idx % K
        X[c, 0, k] = _cdiv(X[c, 0, k], piv[0, k])
        for i in range(1, nz):
            X[c, i, k] = _cdiv(X[c, i, k] - sub[i - 1] * X[c, i - 1, k], piv[i, k])
        for i in range(nz - 2, -1, -1):
            X[c, i, k] = X[c, i, k] - cp[i, k] * X[c, i + 1, k]


def film_step(kind, P, P_prev, alpha, grid, factors, poisson_factors, dt, L, E_ext, el, stab):
    """One fused film3d step ("sbdf1" or "sbdf2"), or None outside coverage
    (film SPM Poisson, Nz < 3, non-float64 P, an E_ext shape other than (3,)
    or the full field)."""
    _require()
    from fftpf import film_engine as f3
    if (poisson_factors.get("spm") or P.dtype != np.float64 or P.ndim != 4 or P.shape[0] != 3
            or P.shape[1] < 3):
        return None
    two_level = kind == "sbdf2"
    if two_level and (P_prev.shape != P.shape or P_prev.dtype != np.float64):
        return None
    E = _field(E_ext, P, P.shape[1:], (3,))
    if E is None:
        return None
    nz, ny, nx = P.shape[1:]
    w = f3.FFT_WORKERS
    Pn = np.ascontiguousarray(P).reshape(3, -1)
    Pm = np.ascontiguousarray(P_prev).reshape(3, -1) if two_level else Pn
    if two_level:
        Ps = workspace("film_pstar", P.shape)
        _film_pstar(Pn, Pm, Ps.reshape(3, -1))
    else:
        Ps = Pn.reshape(P.shape)
    values = [alpha[k] for k in ALPHA_KEYS] + [alpha.get(k, 0.0) for k in ALPHA_KEYS_8TH]
    has8 = any(np.any(np.asarray(a) != 0.0) for a in values[6:])
    coeffs = _coefficients(values, P.shape[1:])
    if el is None:
        Fel, qwr = np.empty((3, 1)), np.empty(1)
    else:
        if "_force_solver" in el:
            Fel = el["_force_solver"](Ps, el=el)
        else:
            from fftpf import film_elastic as fel
            Fel, _ = fel.elastic_force_film(Ps, el)
        Fel = np.ascontiguousarray(Fel).reshape(3, -1)
        qwr = np.ascontiguousarray(grid["hz"] / grid["qw"], dtype=np.float64)
    if kind == "sbdf2":
        h, hs, cA = 2 * dt * L, (2 * dt * L * stab if stab else 0.0), 0.0
    else:
        h, hs, cA = dt * L, 0.0, (1.0 + dt * L * stab if stab else 1.0)
    Pk = spfft.rfft2(Ps, axes=(-2, -1), workers=w)
    R = workspace("film_rhs", P.shape)
    _film_step_rhs(Pn, Pm, two_level, cA, h, hs, coeffs, has8, E, Fel, el is not None, qwr,
                   ny * nx, R.reshape(3, -1))
    X = spfft.rfft2(R, axes=(-2, -1), workers=w)
    K = X.shape[2] * X.shape[3]
    aK, bK = f3._odd_k(grid)
    pf = poisson_factors
    _film_poisson_add(Pk.reshape(3, nz, K), X.reshape(3, nz, K),
                      np.ascontiguousarray(aK.imag).reshape(-1), np.ascontiguousarray(bK.imag).reshape(-1),
                      float(grid["hz"]), np.ascontiguousarray(pf["sub"], dtype=np.float64),
                      np.ascontiguousarray(pf["cp"]).reshape(max(nz - 3, 0), K),
                      np.ascontiguousarray(pf["piv"]).reshape(nz - 2, K), h,
                      workspace("film_phi", (K, nz), np.complex128))
    if factors.get("kind", "tri") == "tri":
        _film_tri_solve(X.reshape(3, nz, K), np.ascontiguousarray(factors["sub"], dtype=np.float64),
                        np.ascontiguousarray(factors["cp"]).reshape(nz - 1, K),
                        np.ascontiguousarray(factors["piv"]).reshape(nz, K))
    else:
        X = f3._solve_grad_factored(factors, X)
    return spfft.irfft2(X, s=(ny, nx), axes=(-2, -1), workers=w, overwrite_x=True)
