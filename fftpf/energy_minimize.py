"""Direct free-energy minimization (preconditioned L-BFGS / FIRE) for the
three deterministic engines: the 2D periodic solver (solver.py, with the H11
sector of grad4.py), the 3D periodic bulk3d.py (with its H11 sector) and the
film (film3d.py + film_elastic.py).

WHAT IT IS FOR. A production run that only wants the RELAXED state
integrates the gradient flow dP/dt = -L dF/dP for ~1e4 fixed-dt steps; this
module drives the SAME discrete F_total straight to a stationary point with
far fewer force evaluations. It computes no dynamics and no kinetics: the
path it takes is not a trajectory, the iteration count is not a time, and it
has no Langevin mode. Starting from the same P0, a minimizer and a gradient
flow may end in DIFFERENT local minima (domain configurations), because the
two paths cross the energy landscape differently -- see docs/MINIMIZE.md.

OPT-IN AND ADDITIVE. Nothing in the engines is modified or monkey-patched;
every function here is a composition of the engines' own public (and a few
private, read-only) functions. No existing code path changes.

THE THREE PIECES, per engine ("problem" objects below):
  energy(P)      F_total, the engine's own number: it CALLS total_energy /
                 total_energy_3d / grad4.total_energy_hog[_3d].
  gradient(P)    the exact nodal derivative dF_total/dP = W * force, W the
                 quadrature weight of a node: the cell volume Volume/N for
                 the periodic engines, (Lx*Ly/(Nx*Ny))*qw(z) for the film
                 (trapezoidal qw: hz inside, hz/2 at both surfaces); the
                 film's elastic force is first brought to the film's (1/qw)
                 convention exactly as film3d.explicit_N does.
  evaluate(P)    (energy, gradient) in ONE pass that shares the expensive
                 solves (depolarization field, elastic equilibrium). The
                 energy is assembled with the engine's own expressions in
                 the engine's own order; with exact_energy=True it is the
                 SAME float as energy(P), by default only the Landau density
                 is the pow-free bulk_energy_density_fast (round-off equal,
                 ~2x cheaper). tests/test_minimize.py checks both.
  precondition(v) H0 v = tau * (I + tau*A)^-1 W^-1 v, A an engine linear
                 operator. Periodic engines: their own IMPLICIT operator
                 A(k) + D(k)/kappa per k (solver._lin_matrix / build_Minv
                 arithmetic), so a first L-BFGS step of unit length is
                 exactly ONE SBDF1 step of dt=tau. Film, precondition=
                 "gradient": the stepper's banded gradient (+H11) factors
                 (film3d.build_grad_factors) -- again one SBDF1 step; film,
                 "depolarization" (default): per in-plane mode the dense
                 3Nz x 3Nz symbol of gradient + H11 + short-circuit
                 depolarization, probed from film3d's own operators, so the
                 film's explicit 1/kappa_b stiffness is preconditioned too.
                 Either way (I + tau W^-1 K)^-1 W^-1 = (W + tau K)^-1 with K
                 the symmetric nodal stiffness: symmetric positive definite
                 in the Euclidean metric of gradient(), i.e. consistent with
                 the film's weighted (qw) metric, as L-BFGS requires.

SUPPORTED: Landau 6th/8th order (scalar or spatial alpha), isotropic and
anisotropic G, H11 (grad4 ops on 2D/3D; the film's H11 argument), the
homogeneous depolarization (scalar kappa; the film's kappa(z) node profile),
homogeneous elasticity (2D stress-free / misfit eps_m / film_full,
eps_star_static; bulk3d stress_free / clamped, per-label Q; film + substrate
+ misfit e0 + C_substrate), uniform or field E_ext.
REFUSED with a ValueError: SPM kappa(r) (2D/3D ops["dkappa"], film3d_spm
factors), SPM elasticity C(r) (any el with "spm"), ensemble-batched P, and
an indefinite preconditioner on the film. Langevin is refused by the casekit
front doors (relax / relax_film take kT only to refuse it).

ALGORITHM. Limited-memory BFGS (two-loop recursion, memory m, initial scaling
gamma = s.y / y.H0 y in the preconditioned metric), strong-Wolfe line search
(Nocedal & Wright Alg. 3.5/3.6, cubic interpolation), monotone energy: a
point is accepted only with sufficient decrease. A failed line search resets
the memory and retries along the preconditioned steepest-descent direction
once before reporting "roundoff_floor" / "line_search_failed" (the state
returned is the last accepted, lowest-energy one). method="fire" is a preconditioned FIRE
(Bitzek et al. PRL 97, 170201 (2006), with the energy-monotone backtracking
of Guenole et al. Comp. Mat. Sci. 175, 109584 (2020)).

STOPPING (both must hold): max|force density| = max|W^-1 dF/dP| < gtol, the
engine's own force units -- the same number cases/bto_film_biaxial.py calls
force_max_deck and casekit.anneal's block_v2 calls the residual -- so it is
independent of grid size and cell volume; AND the last accepted relative
energy change |dF|/max(|F|, energy_scale) < ftol. Or max_iter.
"""
import time

import numpy as np

from fftpf import pf3d_engine as bulk3d, elastic_base as elastic, film_engine as film3d, gradient4 as grad4, pf2d_engine as solver

_TINY = 1e-300


def _dot(a, b):
    """Euclidean inner product of two real fields WITHOUT BLAS: np.vdot is a
    BLAS ddot, which an unpinned OpenBLAS pool turns from ~3 us into ~20 ms
    per call on a loaded machine (measured); einsum's own loop is as fast
    single-threaded and never spawns threads."""
    return float(np.einsum("i,i->", a.reshape(-1), b.reshape(-1)))


# ==========================================================================
# Pow-free Landau ENERGY. solver.bulk_energy_density is sympy-lambdified and
# spends most of its time in libm pow (P**6, P**8): measured ~45% of one
# fused film evaluation at 40x40x21. The minimizer evaluates the energy as
# often as the force, so its fused evaluate() uses this Horner-style form in
# q_i = P_i^2 instead -- the same polynomial, checked SYMBOLICALLY against
# solver._F_BULK at import (the engine's own "sympy verifies, numpy
# evaluates" discipline, cf. solver._verify_bulk_force_horner), and agreeing
# to round-off (~1e-15 relative per point). energy() and exact_energy=True
# keep the engine's lambdified number bit for bit.
# ==========================================================================
def bulk_energy_density_fast(P, alpha):
    """f_bulk(P), pow-free; alpha as in solver.bulk_energy_density."""
    a1, a11, a12, a111, a112, a123 = (alpha[k] for k in solver.ALPHA_KEYS)
    a1111, a1112, a1122, a1123 = (alpha.get(k, 0.0) for k in solver.ALPHA_KEYS_8TH)
    q1, q2, q3 = P[0] * P[0], P[1] * P[1], P[2] * P[2]
    s1 = q1 + q2 + q3                              # sum q_i
    s2 = q1 * q2 + q2 * q3 + q1 * q3               # sum_{i<j} q_i q_j
    s3 = q1 * q2 * q3
    r1, r2, r3 = q1 * q1, q2 * q2, q3 * q3
    p2 = r1 + r2 + r3                              # sum q_i^2
    p3 = r1 * q1 + r2 * q2 + r3 * q3               # sum q_i^3
    m21 = r1 * (q2 + q3) + r2 * (q1 + q3) + r3 * (q1 + q2)   # sum_{i!=j} q_i^2 q_j
    f = a1 * s1 + a11 * p2 + a12 * s2 + a111 * p3 + a112 * m21 + a123 * s3
    if any(np.any(np.asarray(a) != 0.0) for a in (a1111, a1112, a1122, a1123)):
        p4 = r1 * r1 + r2 * r2 + r3 * r3
        m31 = r1 * q1 * (q2 + q3) + r2 * q2 * (q1 + q3) + r3 * q3 * (q1 + q2)
        m22 = r1 * r2 + r2 * r3 + r1 * r3
        f = f + a1111 * p4 + a1112 * m31 + a1122 * m22 + a1123 * (s3 * s1)
    return f


def _verify_bulk_energy_fast():
    """bulk_energy_density_fast == solver._F_BULK: the function itself is run
    on sympy symbols (a symbolic 8th-order coefficient compares != 0, so the
    8th-order branch is included) and the difference must expand to 0; then
    a numeric spot check against the lambdified engine density."""
    import sympy as sp
    names = solver.ALPHA_KEYS + solver.ALPHA_KEYS_8TH
    sym_alpha = dict(zip(names, (solver._a1, solver._a11, solver._a12, solver._a111,
                                 solver._a112, solver._a123, solver._a1111, solver._a1112,
                                 solver._a1122, solver._a1123)))
    expr = bulk_energy_density_fast([solver._P1, solver._P2, solver._P3], sym_alpha)
    if sp.expand(expr - solver._F_BULK) != 0:
        raise AssertionError("minimize: bulk_energy_density_fast != solver._F_BULK")
    rng = np.random.default_rng(0)
    P = rng.uniform(-0.6, 0.6, (3, 64))
    a = {k: float(v) for k, v in zip(names, rng.uniform(-2, 2, len(names)))}
    ref = solver.bulk_energy_density(P, a)
    if not np.allclose(bulk_energy_density_fast(P, a), ref, rtol=1e-12,
                       atol=1e-14 * np.max(np.abs(ref))):
        raise AssertionError("minimize: bulk_energy_density_fast numeric mismatch")


_verify_bulk_energy_fast()


def _bulk_density(exact):
    return solver.bulk_energy_density if exact else bulk_energy_density_fast


# ==========================================================================
# Problems: one per engine.
# ==========================================================================
class PeriodicProblem:
    """solver.py (2D) or bulk3d.py (3D), optionally with grad4's H11 ops."""

    def __init__(self, engine, alpha, ops, grid, E_ext=(0., 0., 0.), elastic_params=None,
                 tau=None, precondition=True, exact_energy=False):
        if engine is solver:
            self.dim = 2
        elif engine is bulk3d:
            self.dim = 3
        else:
            raise ValueError("minimize: engine must be solvers.solver or solvers.bulk3d "
                             "(the film has its own FilmProblem / casekit.relax_film)")
        if ops.get("dkappa") is not None:
            raise ValueError("minimize: SPM kappa(r) (ops['dkappa']) is not supported -- the "
                             "iterated depolarization field is only tol-accurate, so energy "
                             "and force are not an exact pair; use casekit.anneal")
        if elastic_params is not None and elastic_params.get("spm"):
            raise ValueError("minimize: SPM elasticity C(r) (el['spm']) is not supported; "
                             "use casekit.anneal")
        self.engine, self.alpha, self.ops, self.grid = engine, alpha, ops, grid
        self.E_ext, self.el = E_ext, elastic_params
        self.hog = "H11" in ops
        if self.dim == 2:
            self.shape = (3, grid["Ny"], grid["Nx"])
            self._fwd, self._inv = solver._rfft2, solver._irfft2
        else:
            self.shape = (3, grid["Nz"], grid["Ny"], grid["Nx"])
            self._fwd, self._inv = bulk3d._rfftn, bulk3d._irfftn
        self.n_cells = int(np.prod(self.shape[1:]))
        # total = mean(f) * Volume = sum(f) * Volume / N  =>  dF/dP_node = (Volume/N) * force
        self.weight = grid["Volume"] / self.n_cells
        self.tau = default_tau(alpha) if tau is None else float(tau)
        self.exact_energy = bool(exact_energy)
        self.shift = 0.0
        self.Minv = self._build_Minv() if precondition else None

    # -- energy / gradient ---------------------------------------------------
    def energy(self, P):
        """The engine's own F_total (totals dict, key 'total')."""
        self._check(P)
        if self.dim == 2:
            fn = grad4.total_energy_hog if self.hog else solver.total_energy
        else:
            fn = grad4.total_energy_hog_3d if self.hog else bulk3d.total_energy_3d
        return fn(P, self.alpha, self.ops, self.E_ext, self.grid,
                  elastic_params=self.el)[1]["total"]

    def force(self, P):
        """The engine's own total force (dF/dP density), last element of total_force."""
        self._check(P)
        if self.dim == 2:
            fn = grad4.total_force_hog if self.hog else solver.total_force
        else:
            fn = grad4.total_force_hog_3d if self.hog else bulk3d.total_force_3d
        return fn(P, self.alpha, self.ops, self.E_ext, elastic_params=self.el)[-1]

    def gradient(self, P):
        """Exact nodal dF_total/dP."""
        return self.weight * self.force(P)

    def evaluate(self, P):
        """(F_total, dF_total/dP) sharing the depolarization and elastic solves. The
        energy repeats total_energy[_3d]/total_energy_hog[_3d] expression for
        expression, in the same order: with exact_energy=True it is the same
        float as energy(P); by default only the Landau density is the pow-free
        bulk_energy_density_fast (equal to round-off)."""
        self._check(P)
        alpha, ops, grid, el = self.alpha, self.ops, self.grid, self.el
        fb = _bulk_density(self.exact_energy)(P, alpha)
        if self.dim == 2:
            fg = (grad4.grad_energy_density_hog(P, grid, ops) if self.hog
                  else solver.grad_energy_density(P, grid, ops))
            Ed = solver._elec_Ed_core(P, ops)
        else:
            fg = (grad4.grad_energy_density_hog_3d(P, grid, ops) if self.hog
                  else bulk3d.grad_energy_density_3d(P, grid, ops))
            Ed = bulk3d._elec_Ed_core_3d(P, ops)
        Eext = solver._reshape_vec3(self.E_ext, P.ndim)
        fe = (-0.5 * (P[0] * Ed[0] + P[1] * Ed[1] + P[2] * Ed[2])
              - (P[0] * Eext[0] + P[1] * Eext[1] + P[2] * Eext[2]))
        dens = dict(bulk=float(fb.mean()), grad=float(fg.mean()), elec=float(fe.mean()))
        Fel = None
        if el is not None:
            if self.dim == 2:
                sol = elastic.elastic_solve(P, el)
                dens["elastic"] = float(elastic.elastic_energy_density(sol).mean())
                Fel = elastic.elastic_force_explicit(P, sol["sigma"], el["Q11"], el["Q12"],
                                                     el["Q44"])
            else:
                sol = bulk3d.elastic_solve_3d(P, el)
                dens["elastic"] = float(bulk3d.elastic_energy_density_3d(sol).mean())
                Fel = bulk3d.elastic_force_explicit_3d(P, sol["sigma"], el["Q1111"],
                                                       el["Q1122"], el["Q1212"])
        dens["total"] = sum(dens.values())
        F = dens["total"] * grid["Volume"]
        Fg = solver.grad_force(P, ops) if self.dim == 2 else bulk3d.grad_force_3d(P, ops)
        force = solver.bulk_force(P, alpha) + Fg + (-Ed - Eext)
        if Fel is not None:
            force = force + Fel
        return F, self.weight * force

    def force_density(self, g):
        """W^-1 g: the engine's force units (what the stopping rule thresholds)."""
        return g / self.weight

    # -- preconditioner ------------------------------------------------------
    def _build_Minv(self):
        """(I + tau*Lin(k))^-1 per k-point, Lin = A + D/kappa (build_Minv's own
        arithmetic). An INDEFINITE Lin (grad4 ops with G11 < 0 < H11, the AFE
        stripe instability) is shifted so the operator's smallest eigenvalue
        is 1/2; recorded as self.shift."""
        Lin = solver._lin_matrix(self.ops)
        lam_min = float(np.linalg.eigvalsh(Lin).min())
        if 1.0 + self.tau * lam_min >= 0.5:
            build = solver.build_Minv if self.dim == 2 else bulk3d.build_Minv_3d
            return build(self.grid, self.ops, self.tau, 1.0)
        self.shift = 0.5 - (1.0 + self.tau * lam_min)
        return np.linalg.inv((1.0 + self.shift) * np.eye(3) + self.tau * Lin)

    def precondition(self, v):
        """H0 v = tau * (I + tau*A)^-1 W^-1 v (identity metric if precondition=False)."""
        if self.Minv is None:
            return self.tau * v / self.weight
        Vk = self._fwd(v)
        M = self.Minv
        out = np.empty_like(Vk)
        out[0] = M[..., 0, 0] * Vk[0] + M[..., 0, 1] * Vk[1] + M[..., 0, 2] * Vk[2]
        out[1] = M[..., 1, 0] * Vk[0] + M[..., 1, 1] * Vk[1] + M[..., 1, 2] * Vk[2]
        out[2] = M[..., 2, 0] * Vk[0] + M[..., 2, 1] * Vk[1] + M[..., 2, 2] * Vk[2]
        return (self.tau / self.weight) * self._inv(out, self.shape[1:])

    def _check(self, P):
        if P.shape != self.shape:
            raise ValueError(f"minimize: P has shape {P.shape}, expected {self.shape} "
                             "(ensemble-batched P is not supported)")


class FilmProblem:
    """film3d.py (+ film_elastic.py when el is given)."""

    def __init__(self, alpha, grid, G, kappa_b, E_ext=(0., 0., 0.), el=None, H11=None,
                 poisson_factors=None, tau=None, precondition="depolarization",
                 exact_energy=False):
        pf = film3d.poisson_factor(grid, kappa_b) if poisson_factors is None else poisson_factors
        if pf.get("spm"):
            raise ValueError("minimize: film3d_spm kappa(x,y,z) factors are not supported -- "
                             "the SPM field is only tol-accurate; use the fixed-dt stepper")
        if el is not None and el.get("spm"):
            raise ValueError("minimize: film elastic SPM C(r) (el['spm']) is not supported")
        self.alpha, self.grid, self.G, self.kappa_b = alpha, grid, G, kappa_b
        self.E_ext, self.el, self.H11, self.pf = E_ext, el, H11, pf
        self.shape = (3, grid["Nz"], grid["Ny"], grid["Nx"])
        self.area = grid["Lx"] * grid["Ly"]
        # total = area * sum_z qw * mean_xy(f)  =>  dF/dP_node = (area/(Nx*Ny)) * qw(z) * force
        self.W = ((self.area / (grid["Nx"] * grid["Ny"])) * grid["qw"])[None, :, None, None]
        self.qw_ratio = (grid["hz"] / grid["qw"])[None, :, None, None]
        mode = {True: "depolarization", False: None, None: None}.get(precondition, precondition)
        if mode not in (None, "gradient", "depolarization"):
            raise ValueError(f"minimize: film precondition must be 'depolarization', 'gradient' "
                             f"or False, got {precondition!r}")
        self.mode = mode
        if tau is None:
            tau = default_tau(alpha, kappa_b=None if mode == "depolarization" else kappa_b)
        self.tau = float(tau)
        self.exact_energy = bool(exact_energy)
        self.factors = self.Hinv = None
        if mode == "gradient":
            self.factors = film3d.build_grad_factors(grid, G, self.tau, 1.0, H11=H11)
            if not _factors_positive(self.factors):
                raise ValueError("minimize: the film's implicit gradient operator I + tau*A is "
                                 "not positive definite (negative gradient coefficients?); "
                                 "pass precondition=False")
        elif mode == "depolarization":
            self.Hinv = self._build_dense()

    def _build_dense(self):
        """Per in-plane mode k, the exact 3Nz x 3Nz symbol A(k) of the film's
        LINEAR forces -- gradient (+H11) and short-circuit depolarization --
        read off by PROBING: both operators are translation invariant in
        x/y, so the response of the engine's own grad_force_3d and
        solve_poisson_factored to a nodal delta at (x,y)=(0,0) in component
        c, layer z has rfft2 = column (c,z) of A(k) for every k at once (the
        delta's rfft2 is 1 everywhere). With w = qw(z) the nodal weights,
        K = w*A is Hermitian (energy-force adjointness, symmetrized here to
        round-off), and the stored operator is (diag(w) + tau*K)^-1, which
        must be positive definite (checked). precondition() then applies
        H0 = tau*(I + tau*A)^-1 W^-1 = (tau*N/area) * (diag(w) + tau*K)^-1."""
        grid = self.grid
        Nz, Ny, Nx = grid["Nz"], grid["Ny"], grid["Nx"]
        n = 3 * Nz
        A = np.empty((Ny, Nx // 2 + 1, n, n), dtype=complex)
        probe = np.zeros(self.shape)
        for c in range(3):
            for z in range(Nz):
                probe[c, z, 0, 0] = 1.0
                F = film3d.grad_force_3d(probe, grid, self.G, H11=self.H11)
                E, _ = film3d.solve_poisson_factored(probe, grid, self.pf)
                probe[c, z, 0, 0] = 0.0
                A[..., c * Nz + z] = np.moveaxis(film3d._rfft2(F - E).reshape(n, Ny, -1), 0, -1)
        w = np.tile(grid["qw"], 3)
        K = w[:, None] * A
        M = np.diag(w) + self.tau * 0.5 * (K + np.conj(np.swapaxes(K, -1, -2)))
        if np.linalg.eigvalsh(M).min() <= 0.0:
            raise ValueError("minimize: the film's linear operator W + tau*K is not positive "
                             "definite (negative gradient coefficients?); pass "
                             "precondition=False")
        return np.linalg.inv(M)

    def energy(self, P):
        """film3d.total_energy_3d's 'total'."""
        self._check(P)
        return film3d.total_energy_3d(P, self.alpha, self.grid, self.kappa_b, self.G,
                                      poisson_factors=self.pf, E_ext=self.E_ext, el=self.el,
                                      H11=self.H11)[1]["total"]

    def force(self, P):
        """grad_force_3d + explicit_N: the film's (1/qw)-convention dF/dP density,
        the quantity cases/bto_film_biaxial.py reports as force_max_deck."""
        self._check(P)
        return (film3d.grad_force_3d(P, self.grid, self.G, H11=self.H11)
                + film3d.explicit_N(P, self.alpha, self.grid, self.pf, self.E_ext, el=self.el))

    def gradient(self, P):
        """Exact nodal dF_total/dP, trapezoidal z weights included."""
        return self.W * self.force(P)

    def evaluate(self, P):
        """(F_total, dF_total/dP) sharing the Poisson and the full-stack elastic
        solve; total_energy_3d's expressions, in its order (the same float as
        energy(P) with exact_energy=True; see PeriodicProblem.evaluate)."""
        self._check(P)
        alpha, grid, el, qw, area = self.alpha, self.grid, self.el, self.grid["qw"], self.area
        fb = _bulk_density(self.exact_energy)(P, alpha)
        F_bulk = area * float(np.sum(qw * fb.mean(axis=(-2, -1))))
        F_grad = film3d.grad_energy_3d(P, grid, self.G)
        E, Eext = film3d.solve_poisson_factored(P, grid, self.pf, self.E_ext)
        fe = (-0.5 * (P[0] * E[0] + P[1] * E[1] + P[2] * E[2])
              - (P[0] * Eext[0] + P[1] * Eext[1] + P[2] * Eext[2]))
        F_elec = area * float(np.sum(qw * fe.mean(axis=(-2, -1))))
        tot = dict(bulk=F_bulk, grad=F_grad, elec=F_elec)
        if self.H11 is not None:
            tot["hog"] = film3d.hog_energy_film(P, grid, self.H11)
        N = solver.bulk_force(P, alpha) + (-E - Eext)
        if el is not None:
            from fftpf import film_elastic as fel
            sol = fel.elastic_solve_film(P, el)
            dens_el = 0.5 * np.einsum("ije...,ije...->e...", sol["sigma_face"], sol["e_face"])
            tot["elastic"] = area * el["hz"] * float(np.sum(dens_el.mean(axis=(-2, -1))))
            Fel = elastic.elastic_force_explicit(P, sol["sigma_node"], el["Q11"], el["Q12"],
                                                 el["Q44"])
            N = N + Fel * self.qw_ratio
        F = sum(tot.values())
        force = film3d.grad_force_3d(P, grid, self.G, H11=self.H11) + N
        return F, self.W * force

    def force_density(self, g):
        return g / self.W

    def precondition(self, v):
        """H0 v = tau * (I + tau*A)^-1 W^-1 v = tau * (W + tau*K)^-1 v, A the
        banded gradient operator ("gradient") or the dense per-mode gradient +
        depolarization symbol ("depolarization"); tau * W^-1 v without one."""
        Ny, Nx = self.grid["Ny"], self.grid["Nx"]
        if self.Hinv is not None:
            n = self.shape[0] * self.shape[1]
            Xk = film3d._rfft2(v)
            X = np.moveaxis(Xk.reshape(n, Ny, -1), 0, -1)[..., None]
            Yk = np.moveaxis(np.matmul(self.Hinv, X)[..., 0], -1, 0).reshape(Xk.shape)
            scale = self.tau * Nx * Ny / self.area
            return scale * film3d._irfft2(Yk, (Ny, Nx))
        if self.factors is None:
            return self.tau * v / self.W
        Xk = film3d._rfft2(v / self.W)
        Yk = film3d._solve_grad_factored(self.factors, Xk)
        return self.tau * film3d._irfft2(Yk, (Ny, Nx))

    def _check(self, P):
        if P.shape != self.shape:
            raise ValueError(f"minimize: P has shape {P.shape}, expected {self.shape}")


def _factors_positive(factors):
    """Positive definiteness of a factored symmetric/Hermitian banded operator:
    every pivot (scalar LDL^T) or pivot block (block LDL^H) positive."""
    kind = factors.get("kind", "tri")
    if kind in ("tri", "penta"):
        return bool(np.all(factors["piv"] > 0.0))
    blocks = np.linalg.inv(factors["piv_inv"])
    blocks = 0.5 * (blocks + np.conj(np.swapaxes(blocks, -1, -2)))
    return bool(np.all(np.linalg.eigvalsh(blocks) > 0.0))


def default_tau(alpha, kappa_b=None):
    """Default preconditioner scale ~ the engine's typical stable dt, i.e. the
    inverse stiffness of what the preconditioner does NOT contain:

    - periodic engines, and the film with precondition="depolarization": the
      depolarization is inside the preconditioner, the binding stiffness left
      is the Landau curvature, so tau = 1/c with c the largest eigenvalue of
      the bulk Hessian at the uniform minimum of f_bulk along [001], [110]
      and [111] (2|a1| in the paraelectric phase); capped at 1.
    - film with precondition="gradient" or False (kappa_b given): the
      depolarization is explicit there (route A), its stiffness is
      1/kappa_b, so tau = kappa_b (0.0885 at eps_b = 10, where production
      uses dt = 0.08). A kappa(z) profile uses its minimum.

    A heuristic, not a law: pass your production dt as tau when you have one."""
    if kappa_b is not None:
        return float(np.min(kappa_b))
    a = {k: float(np.mean(v)) for k, v in alpha.items()}
    c = abs(2.0 * a["a1"])
    mags = np.linspace(0.0, 1.0, 2001)[1:]
    for direction in ((0., 0., 1.), (1., 1., 0.), (1., 1., 1.)):
        u = np.asarray(direction) / np.linalg.norm(direction)
        Pline = u[:, None] * mags[None, :]
        f = solver.bulk_energy_density(Pline, a)
        j = int(np.argmin(f))
        if f[j] < 0.0:
            c = max(c, _bulk_hessian_max(Pline[:, j], a))
    return float(min(1.0, 1.0 / max(c, 1e-12)))


def _bulk_hessian_max(P, alpha, h=1e-6):
    H = np.empty((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        H[:, j] = (solver.bulk_force(P + e, alpha) - solver.bulk_force(P - e, alpha)) / (2 * h)
    return float(np.linalg.eigvalsh(0.5 * (H + H.T)).max())


# ==========================================================================
# Driver
# ==========================================================================
def minimize(problem, P0, *, method="lbfgs", gtol=1e-6, ftol=1e-10, max_iter=10000,
             memory=10, sample_every=1, energy_scale=None, max_step=None, callback=None):
    """Minimize problem's F_total from P0. Returns (P, trace, stop_reason, info).

    trace: F_total at iteration 0 and every `sample_every` iterations, plus the
    final iterate (info["trace_iterations"] holds the iteration numbers).
    stop_reason: "converged" (both tolerances met), "max_iter",
    "roundoff_floor" (no further decrease is resolvable: the decrease a step
    promises is below the round-off of F -- gtol was set below what the
    energy can certify; info["max_force"] says where it stopped) or
    "line_search_failed" (no decrease along the L-BFGS nor the
    preconditioned steepest-descent direction for another reason).
    info: n_iter, n_energy_evals, n_grad_evals, max_force (final max|force
    density|), energy, rel_energy_change, wall_time, tau, method, n_resets.

    gtol: on max|W^-1 dF/dP|, the engine's force units. ftol: on the last
    accepted |dF|/max(|F|, energy_scale). max_step: optional cap on max|dP|
    per iteration (deck P units); None = no cap. callback(it, P, F, max_force)
    is called after every accepted iteration."""
    if method not in ("lbfgs", "fire"):
        raise ValueError(f"minimize: unknown method {method!r} (lbfgs | fire)")
    if gtol <= 0 or ftol < 0 or max_iter < 0 or sample_every < 1 or memory < 1:
        raise ValueError("minimize: invalid tolerance, iteration count, memory or sampling")
    t0 = time.perf_counter()
    P = np.array(P0, dtype=float, copy=True)
    if not np.all(np.isfinite(P)):
        raise ValueError("minimize: P0 is not finite")
    run = _lbfgs if method == "lbfgs" else _fire
    P, trace, its, reason, stats = run(problem, P, gtol=gtol, ftol=ftol, max_iter=max_iter,
                                       memory=memory, sample_every=sample_every,
                                       energy_scale=energy_scale, max_step=max_step,
                                       callback=callback)
    # info["energy"] is the ENGINE's own number for the returned state (one
    # extra energy call); "energy_minimizer" is what the iteration itself saw.
    info = dict(stats, energy_minimizer=stats["energy"], energy=float(problem.energy(P)),
                method=method, tau=problem.tau, wall_time=time.perf_counter() - t0,
                trace_iterations=its, gtol=gtol, ftol=ftol)
    return P, trace, reason, info


class _Counter:
    """Wraps problem.evaluate, counting fused (energy+gradient) evaluations."""

    def __init__(self, problem):
        self.problem, self.n = problem, 0

    def __call__(self, P):
        self.n += 1
        F, g = self.problem.evaluate(P)
        return float(F), g


def _converged(problem, g, dF, F, gtol, ftol, energy_scale):
    fmax = float(np.max(np.abs(problem.force_density(g))))
    scale = max(abs(F), energy_scale or 0.0, _TINY)
    rel = abs(dF) / scale
    return fmax, rel, (fmax < gtol and rel < ftol)


def _lbfgs(problem, P, *, gtol, ftol, max_iter, memory, sample_every, energy_scale, max_step,
           callback):
    ev = _Counter(problem)
    F, g = ev(P)
    trace, its = [F], [0]
    S, Y, rho = [], [], []
    gamma, n_resets, dF, it = 1.0, 0, 0.0, 0
    reason = "max_iter"
    fmax, rel, done = _converged(problem, g, dF, F, gtol, ftol, energy_scale)
    if done:
        reason = "converged"
    while not done and it < max_iter:
        d = -_two_loop(problem, g, S, Y, rho, gamma)
        gd = _dot(g, d)
        if not gd < 0.0:                               # not a descent direction: reset
            S, Y, rho, gamma, n_resets = [], [], [], 1.0, n_resets + 1
            d = -problem.precondition(g)
            gd = _dot(g, d)
        step = _line_search(ev, P, F, g, d, gd, max_step)
        if step is None and S:                         # retry once without the memory
            S, Y, rho, gamma, n_resets = [], [], [], 1.0, n_resets + 1
            d = -problem.precondition(g)
            gd = _dot(g, d)
            step = _line_search(ev, P, F, g, d, gd, max_step)
        if step is None:
            # |g.d| is the decrease a unit step promises: below the round-off
            # of F itself no energy-monotone search can make progress.
            floor = abs(gd) <= 1e4 * np.finfo(float).eps * max(abs(F), _TINY)
            reason = "roundoff_floor" if floor else "line_search_failed"
            break
        a, F_new, g_new = step
        s = a * d
        y = g_new - g
        sy = _dot(s, y)
        if sy > 1e-12 * np.sqrt(_dot(s, s) * _dot(y, y)):
            S.append(s)
            Y.append(y)
            rho.append(1.0 / sy)
            if len(S) > memory:
                S.pop(0), Y.pop(0), rho.pop(0)
            gamma = sy / _dot(y, problem.precondition(y))
        P = P + s
        dF, F, g = F - F_new, F_new, g_new
        it += 1
        fmax, rel, done = _converged(problem, g, dF, F, gtol, ftol, energy_scale)
        if done:
            reason = "converged"
        if it % sample_every == 0 or done or it == max_iter:
            trace.append(F)
            its.append(it)
        if callback is not None:
            callback(it, P, F, fmax)
    if its[-1] != it:
        trace.append(F)
        its.append(it)
    return P, trace, its, reason, dict(n_iter=it, n_energy_evals=ev.n, n_grad_evals=ev.n,
                                       max_force=fmax, energy=F, rel_energy_change=rel,
                                       n_resets=n_resets, memory=memory)


def _two_loop(problem, g, S, Y, rho, gamma):
    q = g.copy()
    a = [0.0] * len(S)
    for i in range(len(S) - 1, -1, -1):
        a[i] = rho[i] * _dot(S[i], q)
        q -= a[i] * Y[i]
    r = gamma * problem.precondition(q)
    for i in range(len(S)):
        b = rho[i] * _dot(Y[i], r)
        r += (a[i] - b) * S[i]
    return r


def _line_search(ev, P, F0, g0, d, gd0, max_step, c1=1e-4, c2=0.9, max_evals=30):
    """Strong-Wolfe line search (Nocedal & Wright 2006, Alg. 3.5 + 3.6) on
    phi(a) = F(P + a d). Returns (a, F, g) with F < F0, or None. A non-finite
    energy counts as +inf (the step is shrunk). If the Wolfe conditions cannot
    be met within max_evals, the best sufficient-decrease point is returned."""
    dmax = float(np.max(np.abs(d)))
    if dmax == 0.0:
        return None
    a_max = np.inf if max_step is None else max_step / dmax
    a1 = min(1.0, a_max)
    best = [None]
    n = [0]

    def phi(a):
        n[0] += 1
        F, g = ev(P + a * d)
        if not np.isfinite(F) or not np.all(np.isfinite(g)):
            return np.inf, np.nan, None
        dphi = _dot(g, d)
        if F <= F0 + c1 * a * gd0 and F < F0 and (best[0] is None or F < best[0][1]):
            best[0] = (a, F, g)
        return F, dphi, g

    def zoom(a_lo, f_lo, d_lo, a_hi, f_hi, d_hi):
        while n[0] < max_evals:
            a = _interpolate(a_lo, f_lo, d_lo, a_hi, f_hi, d_hi)
            f, dphi, g = phi(a)
            if not np.isfinite(f) or f > F0 + c1 * a * gd0 or f >= f_lo:
                a_hi, f_hi, d_hi = a, f, dphi
            else:
                if abs(dphi) <= -c2 * gd0:
                    return a, f, g
                if dphi * (a_hi - a_lo) >= 0:
                    a_hi, f_hi, d_hi = a_lo, f_lo, d_lo
                a_lo, f_lo, d_lo = a, f, dphi
            if abs(a_hi - a_lo) <= 1e-14 * max(abs(a_lo), abs(a_hi)):
                break
        return best[0]

    a_prev, f_prev, d_prev = 0.0, F0, gd0
    a = a1
    while n[0] < max_evals:
        f, dphi, g = phi(a)
        if not np.isfinite(f) or f > F0 + c1 * a * gd0 or (a_prev > 0 and f >= f_prev):
            return zoom(a_prev, f_prev, d_prev, a, f, dphi)
        if abs(dphi) <= -c2 * gd0:
            return a, f, g
        if dphi >= 0:
            return zoom(a, f, dphi, a_prev, f_prev, d_prev)
        if a >= a_max:
            return best[0]
        a_prev, f_prev, d_prev = a, f, dphi
        a = min(4.0 * a, a_max)
    return best[0]


def _interpolate(a_lo, f_lo, d_lo, a_hi, f_hi, d_hi):
    """Safeguarded cubic (Hermite) minimizer between a_lo and a_hi; bisection
    when the bracket end is non-finite or the cubic lands near an end."""
    lo, hi = min(a_lo, a_hi), max(a_lo, a_hi)
    mid = 0.5 * (a_lo + a_hi)
    if not (np.isfinite(f_hi) and np.isfinite(d_hi)):
        return a_lo + 0.25 * (a_hi - a_lo) if a_hi > a_lo else mid
    dd = a_hi - a_lo
    d1 = d_lo + d_hi - 3.0 * (f_lo - f_hi) / (a_lo - a_hi)
    rad = d1 * d1 - d_lo * d_hi
    if rad < 0:
        return mid
    d2 = np.sign(dd) * np.sqrt(rad)
    den = d_hi - d_lo + 2.0 * d2
    if den == 0:
        return mid
    a = a_hi - dd * (d_hi + d2 - d1) / den
    margin = 0.1 * (hi - lo)
    if not np.isfinite(a) or a < lo + margin or a > hi - margin:
        return mid
    return float(a)


def _fire(problem, P, *, gtol, ftol, max_iter, memory, sample_every, energy_scale, max_step,
          callback, dt0=1.0, dt_max=10.0, n_min=5, f_inc=1.1, f_dec=0.5, a0=0.1, f_a=0.99):
    """Preconditioned FIRE: the 'force' is -H0 g (one SBDF1 step of dt=tau per
    unit FIRE time), mixing and velocity live in that preconditioned space.
    Energy-monotone: an uphill step is undone and retaken with dt/2 and v=0
    (Guenole et al. 2020's backtracking variant)."""
    ev = _Counter(problem)
    F, g = ev(P)
    trace, its = [F], [0]
    v = np.zeros_like(P)
    dt, a, n_pos, it, dF = dt0, a0, 0, 0, 0.0
    reason = "max_iter"
    fmax, rel, done = _converged(problem, g, dF, F, gtol, ftol, energy_scale)
    if done:
        reason = "converged"
    n_fail = 0
    while not done and it < max_iter:
        f = -problem.precondition(g)
        power = _dot(f, v)
        if power > 0:
            fn, vn = np.sqrt(_dot(f, f)), np.sqrt(_dot(v, v))
            v = (1 - a) * v + a * vn * f / max(fn, _TINY)
            n_pos += 1
            if n_pos > n_min:
                dt, a = min(dt * f_inc, dt_max), a * f_a
        else:
            v[:] = 0.0
            dt, a, n_pos = dt * f_dec, a0, 0
        v = v + dt * f                                      # semi-implicit Euler
        dP = dt * v
        if max_step is not None:
            m = float(np.max(np.abs(dP)))
            if m > max_step:
                dP *= max_step / m
        F_new, g_new = ev(P + dP)
        if not np.isfinite(F_new) or F_new > F:              # uphill: undo, halve, stop
            v[:] = 0.0
            dt, a, n_pos = dt * f_dec, a0, 0
            n_fail += 1
            if n_fail > 60:
                reason = "line_search_failed"
                break
            continue
        n_fail = 0
        P = P + dP
        dF, F, g = F - F_new, F_new, g_new
        it += 1
        fmax, rel, done = _converged(problem, g, dF, F, gtol, ftol, energy_scale)
        if done:
            reason = "converged"
        if it % sample_every == 0 or done or it == max_iter:
            trace.append(F)
            its.append(it)
        if callback is not None:
            callback(it, P, F, fmax)
    if its[-1] != it:
        trace.append(F)
        its.append(it)
    return P, trace, its, reason, dict(n_iter=it, n_energy_evals=ev.n, n_grad_evals=ev.n,
                                       max_force=fmax, energy=F, rel_energy_change=rel,
                                       n_resets=0, dt_fire=dt)
