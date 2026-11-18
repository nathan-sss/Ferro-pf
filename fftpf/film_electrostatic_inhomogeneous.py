"""film3d_spm.py -- SPEC V2-03 B3: the electrostatic SPM (inhomogeneous
kappa(x,y,z)) for the FILM lane, as a separate module over film3d.py.

film3d.py is NOT modified by this package (SPEC V2-03 B3's own instruction,
and batch B's file split: film3d.py belongs to the gradient/H11 package this
round). Everything here is a thin, explicit layer on top of the A7 kappa(z)
machinery film3d already carries.

WHAT THE FILM LANE ALREADY HAD, AND WHAT THIS ADDS
  film3d._poisson_bands/poisson_factor (SPEC V2-03 A7) solve
  d_i[kappa(z) d_i phi] = d_i P_i per in-plane mode: spectral in (x1,x2),
  a finite-volume tridiagonal in z with the HARMONIC face average
  (film3d._kappa_faces), Dirichlet phi=0 at both film surfaces. That covers
  dead layers and graded interfacial permittivity -- anything that varies
  along the thickness ONLY, because a kappa that varies in-plane couples the
  in-plane modes and destroys the per-mode banded structure the whole film
  lane is built on.

  This module restores the banded structure by splitting the coefficient the
  way solver.py's 2D WP5 SPM does (Wang et al., Acta Mater 61 (2013) 7591
  SS2), around a z-PROFILE reference instead of a scalar:

      kappa(x,y,z) = kappa0(z) + dkappa(x,y,z)
      d_i[kappa0(z) d_i phi] = d_i P_i - d_i[dkappa d_i phi]
                             = d_i[P_i + dkappa*E_i]          (E = -grad phi)

  so every iterate is ONE call to film3d.solve_poisson_factored -- the
  unmodified A7 banded solve -- with the source field P replaced by
  P + dkappa*E^(n-1). Zero order (n=0) has no extra source, i.e. it IS the
  A7 solve, which is why a purely-z kappa (dkappa == 0) reproduces that path
  bit-for-bit in one iteration (K2 `film_spm_kappa_z_only`).

  kappa0(z) DEFAULT = the IN-PLANE MEAN profile (SPEC B3: "the A7 profile
  (or the mean)"); `kappa0_profile=` overrides it. The fixed point's
  contraction factor is bounded by max|dkappa|/kappa0 over the film, so the
  reference is not cosmetic: a two-region lateral split of equal area puts
  the in-plane mean exactly at (kappa_min+kappa_max)/2, which MINIMIZES that
  bound (0.82 at 10x contrast); a strongly unbalanced split does not, and a
  thin high-kappa inclusion can push the bound past 1, where the iteration
  diverges rather than converging slowly. K4 `film_spm_residual` reports the
  bound next to the iteration count for exactly this reason.

DISCRETIZATION OF THE NEW TERM: kappa0 uses the symmetric film Poisson
matrix and its quadrature-adjoint source/field pair. The dkappa correction
uses that same map. At a converged fixed point the map remains symmetric
in the nodal quadrature: (I + T0 D)^{-1} T0 = T0 (I + D T0)^{-1}, where T0
is the negative base field map and D is nodal dkappa multiplication.
K1 checks discrete derivatives at every mesh; K4 checks reciprocity.

ENERGY AND FORCE CONVENTION (which discrete energy this module uses):
EXACTLY film3d's, with the SPM-converged E in place of the banded one --
  f_elec = -1/2 P.E_d - P.E_ext   (nodal density, (Nz,Ny,Nx))
  delta F_elec/delta P_i = -E_d,i - E_ext,i   (no extra 1/2)
integrated over z with film3d's own trapezoidal node weights grid["qw"].
film3d.py's module docstring proves the reciprocity that licenses dropping
the 1/2 -- P -> E_d linear and self-adjoint, from two integrations by parts
whose boundary terms vanish (Dirichlet phi=0 on both z faces, periodic
in-plane faces cancel pairwise) -- and that proof uses kappa ONLY as the
coefficient inside d_i[kappa d_i .], so it goes through verbatim for a fully
inhomogeneous kappa(x,y,z): the converged SPM map is the exact inverse of
that same elliptic operator, self-adjoint for ANY strictly positive kappa(r).
Gated two ways: K1 `fd_film_kappa_xyz` (central FD of the energy vs this
force) and K4 `film_spm_residual`'s reported reciprocity residual
int P1.E[P2] vs int P2.E[P1].

STEPPING: no separate entry point here, deliberately. film3d is Route A --
the electrostatic force is fully EXPLICIT inside film3d.explicit_N -- so
there is no implicit-kappa0 term to correct and nothing to rebalance
(contrast bulk3d, whose M(k) carries D(k)/kappa0 implicitly and therefore
needs the hybrid dE term in _explicit_N_3d). The SPM IS wired into film3d
stepping: film3d.solve_poisson_factored dispatches on factors["spm"] (SPEC
V2-03 C2 carry-over), so passing poisson_factor_spm's output as
`poisson_factors` to film3d.explicit_N / step / step_sbdf2 /
total_energy_3d runs this module's fixed point. film3d.relax and
probe_stable_dt build their own banded factors from kappa_b and so do not.

OPT-IN CG: poisson_factor_spm(..., method="cg") solves the same discrete
equations by solver._spm_pcg (preconditioned conjugate gradients in the qw
inner product) instead of the fixed point -- no contraction bound, but it
needs the discrete operator to be positive definite, which a z-graded
reference next to a dead layer can break at high contrast (it then raises).
docs/CG_SPM.md. The default method="basic" is untouched.

NO FFT KNOB: this module never transforms anything itself (every FFT happens
inside film3d.solve_poisson_factored), so film3d.FFT_WORKERS governs, and
_kit.fft_workers_one(..., film3d) in a gate file already covers it.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from fftpf import film_engine as f3
from fftpf import pf2d_engine as solver


def kappa0_inplane_mean(kappa_field):
    """The default reference profile: the in-plane mean of kappa(x,y,z) at
    each z node, shape (Nz,).

    EXACTNESS GUARD (the `==` guard film3d._kappa_faces uses, for the same
    reason): a z node whose in-plane values are ALL equal returns that value
    itself, not a recomputed mean -- a mean over n equal floats is not
    bit-for-bit that float in general, and without this the degenerate cases
    would be "small" instead of exactly zero (K2 `film_spm_kappa_z_only`)."""
    kmin = kappa_field.min(axis=(1, 2))
    kmax = kappa_field.max(axis=(1, 2))
    return np.where(kmin == kmax, kmin, kappa_field.mean(axis=(1, 2)))


def poisson_factor_spm(grid, kappa_field, kappa0_profile=None, method="basic",
                       spm_cg_max_iter=None):
    """Factorize the SPM's reference operator once per (grid, kappa_field).

    kappa_field: (Nz,Ny,Nx), strictly positive. kappa0_profile: optional
    (Nz,) reference (default: kappa0_inplane_mean -- see the module docstring
    for when overriding it is not optional).

    method: "basic" (default -- the fixed point, dict unchanged) or "cg"
    (opt-in: solve_poisson_spm runs solver._spm_pcg, the preconditioned CG
    on the same discrete equations, in the qw-weighted nodal inner product
    in which the base map is self-adjoint; docs/CG_SPM.md). The dict then
    carries spm_method="cg" and spm_cg_max_iter (None ->
    solver.SPM_CG_DEFAULT_MAX_ITER). CG needs the discrete operator to be
    positive definite, which holds for a z-uniform reference but NOT in
    general next to a dead/graded layer at high contrast (nodal dkappa
    against the harmonic face average of kappa0(z)) -- there it raises a
    RuntimeError at once, where the basic iteration diverges as well.

    Returns a dict carrying
      kappa0          (Nz,) reference profile
      dkappa          (Nz,Ny,Nx) = kappa_field - kappa0[:,None,None]
      kappa0_factors  film3d.poisson_factor(grid, kappa0) -- the A7 banded
                      factorization, built by film3d itself
      dkappa_over_kappa0  max|dkappa|/min(kappa0), the contraction bound
      spm             True

    The A7 factors are kept in their OWN slot rather than merged into this
    dict: a dict carrying film3d's sub/cp/piv/n keys at top level would be
    accepted by film3d.solve_poisson_factored, which would then silently
    solve the kappa0-only problem and drop dkappa. Nesting makes that
    mistake impossible instead of merely unlikely."""
    if method not in solver.SPM_METHODS:
        raise ValueError(f"poisson_factor_spm: method must be one of {solver.SPM_METHODS}, "
                         f"got {method!r}")
    kap = np.asarray(kappa_field, dtype=float)
    shape = (grid["Nz"], grid["Ny"], grid["Nx"])
    if kap.shape != shape:
        raise ValueError(f"poisson_factor_spm: kappa_field shape {kap.shape} != grid {shape}")
    if not np.all(kap > 0.0):
        raise ValueError("poisson_factor_spm: kappa(x,y,z) must be strictly positive everywhere")
    if kappa0_profile is None:
        kappa0 = kappa0_inplane_mean(kap)
    else:
        kappa0 = np.asarray(kappa0_profile, dtype=float)
        if kappa0.shape != (grid["Nz"],):
            raise ValueError(f"poisson_factor_spm: kappa0_profile shape {kappa0.shape} "
                             f"!= (Nz,)=({grid['Nz']},)")
    dkappa = kap - kappa0[:, None, None]
    out = dict(kappa0=kappa0, dkappa=dkappa,
               kappa0_factors=f3.poisson_factor(grid, kappa0),
               dkappa_over_kappa0=float(np.max(np.abs(dkappa)) / np.min(kappa0)),
               spm=True)
    if method == "cg":                       # a basic factors dict gains no key
        out.update(spm_method="cg", spm_cg_max_iter=(solver.SPM_CG_DEFAULT_MAX_ITER
                                                     if spm_cg_max_iter is None
                                                     else int(spm_cg_max_iter)))
    return out


def _solve_poisson_spm_cg(P, grid, factors, E_ext, tol, max_iter, warn_iter, warm,
                          record_history):
    """solve_poisson_spm for a method="cg" factors dict: solver._spm_pcg with
    the film's banded kappa0(z) solve as the core and the quadrature inner
    product sum_c sum_z qw[z] sum_xy a b, in which that solve is self-adjoint
    (the module docstring's reciprocity). Same return contract."""
    f0 = factors["kappa0_factors"]
    qw = np.asarray(grid["qw"], dtype=float)
    held = {}

    def core(src):
        E, Eext = f3.solve_poisson_factored(src, grid, f0, E_ext)
        held.setdefault("Eext", Eext)
        return E

    def inner(a, b):
        return np.asarray(np.einsum("c...zyx,c...zyx,z->...", a, b, qw), dtype=float)

    res = solver._spm_pcg(P, factors["dkappa"], core, 3, tol=tol,
                          max_iter=max_iter if max_iter is not None else factors["spm_cg_max_iter"],
                          warn_iter=warn_iter, record_history=record_history, warm=warm,
                          where="film3d_spm.solve_poisson_spm", inner=inner)
    return (res[0], held["Eext"]) + tuple(res[2:])


def solve_poisson_spm(P, grid, factors, E_ext=(0.0, 0.0, 0.0), tol=1e-10,
                      max_iter=None, warn_iter=None, warm=None, record_history=False):
    """The fixed point of the module docstring: each iterate is ONE
    film3d.solve_poisson_factored on the kappa0(z) operator with the source
    field P + dkappa*E^(n-1).

    Returns (E, Eext, n_iter, resid[, history]) -- E is the (3,Nz,Ny,Nx)
    depolarization field, Eext film3d's own (3,1,1,1) broadcast of E_ext
    (kept separate and un-redistributed, film3d's convention), and E is ALSO
    the warm state: pass it back as `warm=` on the next solve from a nearby P
    to start inside the basin instead of at the kappa0-only field.

    E_ext is deliberately NOT folded into the dkappa source -- the same
    ruling solver._spm_iterate's docstring derives in 2D (gate G8c): folding
    it in makes the converged E_d AFFINE rather than linear in P and breaks
    the f_elec = -1/2 P.E_d / force = -E_d pair by a clean, tol-independent
    amount. Convergence is on max|E^n - E^(n-1)|/max|E^n| <= tol, capped at
    max_iter, warned past warn_iter (SPEC's 1e-10 / 200 / 80, the same three
    numbers as both bulk engines; None means those defaults).

    A factors dict built with method="cg" runs _solve_poisson_spm_cg instead
    (same contract; n = CG iterations, resid = CG relative residual,
    max_iter=None -> factors["spm_cg_max_iter"])."""
    if factors.get("spm_method") == "cg":
        return _solve_poisson_spm_cg(P, grid, factors, E_ext, tol, max_iter, warn_iter, warm,
                                     record_history)
    max_iter = 200 if max_iter is None else max_iter
    warn_iter = 80 if warn_iter is None else warn_iter
    dkappa = factors["dkappa"]
    f0 = factors["kappa0_factors"]
    E, Eext = f3.solve_poisson_factored(P, grid, f0, E_ext)
    E_prev = E if warm is None else warm
    history = [] if record_history else None
    resid = float("inf")
    n = 0
    for n in range(1, max_iter + 1):
        E, Eext = f3.solve_poisson_factored(P + dkappa * E_prev, grid, f0, E_ext)
        denom = float(np.max(np.abs(E)))
        num = float(np.max(np.abs(E - E_prev)))
        resid = num / denom if denom > 0 else 0.0
        if record_history:
            history.append(resid)
        E_prev = E
        if resid <= tol:
            break
    if n > warn_iter:
        print(f"[film3d_spm.solve_poisson_spm] WARNING: SPM ran {n} iterations "
              f"(> warn_iter={warn_iter}), final resid={resid:.3e} (tol={tol:.0e}).")
    if resid > tol:
        print(f"[film3d_spm.solve_poisson_spm] WARNING: SPM did NOT converge within "
              f"max_iter={max_iter} (final resid={resid:.3e} > tol={tol:.0e}).")
    if record_history:
        return E, Eext, n, resid, history
    return E, Eext, n, resid


def elec_force_film_spm(P, grid, factors, E_ext=(0.0, 0.0, 0.0), tol=1e-10, warm=None):
    """delta F_elec/delta P_i = -E_d,i - E_ext,i -- film3d.elec_force_3d's
    convention with the SPM-converged field (module docstring's ENERGY AND
    FORCE CONVENTION for why the 1/2 still disappears at inhomogeneous
    kappa)."""
    E, Eext, _n, _resid = solve_poisson_spm(P, grid, factors, E_ext, tol=tol, warm=warm)
    return -E - Eext


def elec_energy_film_spm(P, grid, factors, E_ext=(0.0, 0.0, 0.0), tol=1e-10, warm=None):
    """f_elec = -0.5*P.E_d - P.E_ext, nodal density (Nz,Ny,Nx) -- film3d.
    elec_energy_density_3d's own expression with the SPM-converged field."""
    E, Eext, _n, _resid = solve_poisson_spm(P, grid, factors, E_ext, tol=tol, warm=warm)
    return (-0.5 * (P[0] * E[0] + P[1] * E[1] + P[2] * E[2])
            - (P[0] * Eext[0] + P[1] * Eext[1] + P[2] * Eext[2]))


def total_energy_film_spm(P, alpha, grid, factors, G11, E_ext=(0.0, 0.0, 0.0), tol=1e-10):
    """(densities, totals) for bulk/grad/elec/total -- film3d.total_energy_3d's
    contract and quadrature with the elec term replaced by the SPM one.

    The bulk and gradient totals are taken FROM film3d.total_energy_3d
    (called on the kappa0(z) profile and its already-built factors) rather
    than recomputed here: those two terms have nothing to do with kappa, and
    film3d's versions carry the validated quadrature conventions (qw
    trapezoid in z for bulk and the in-plane gradient, the face-based
    (P_{j+1}-P_j)^2/2hz form for the z gradient). The elec total that call
    also produces is DISCARDED and replaced -- it is the kappa0-only energy.
    One extra banded solve, against ~tens in the SPM loop; the alternative
    is a copy of that bookkeeping that can drift from film3d's."""
    _dens0, tot0 = f3.total_energy_3d(P, alpha, grid, factors["kappa0"], G11,
                                      poisson_factors=factors["kappa0_factors"], E_ext=E_ext)
    area = grid["Lx"] * grid["Ly"]
    dens_elec = elec_energy_film_spm(P, grid, factors, E_ext, tol=tol)
    tot = dict(bulk=tot0["bulk"], grad=tot0["grad"],
               elec=area * float(np.sum(grid["qw"] * dens_elec.mean(axis=(-2, -1)))))
    tot["total"] = sum(tot.values())
    dens = {k: v / grid["Volume3D"] for k, v in tot.items()}
    return dens, tot
