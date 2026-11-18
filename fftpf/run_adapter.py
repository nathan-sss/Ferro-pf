"""One calling convention over the three geometry engines (SPEC § 16.2, package E0).

Adapter(engine, grid, alpha, ...) binds a deck to pf2d_engine / pf3d_engine /
film_engine and exposes build / step / mean_P / energy / random_start. It is a
pure call-through: every number comes from the engine function it names, with
the arguments a direct caller would pass, so results are bit-for-bit those of
the direct calls. The layer holds only the operators `build` made; it owns no
state of the field (P, P_prev and the RNG belong to the caller) and no numerics.

Engines: "pf2d" (P shape (3,[S,]Ny,Nx)), "pf3d" ((3,[S,]Nz,Ny,Nx)), "film"
((3,Nz,Ny,Nx), no ensemble axis). Unsupported requests raise ValueError naming
the option and the reason; the engines' own refusals propagate unchanged.
"""
import numpy as np

from fftpf import film_engine, film_langevin, pf2d_engine, pf3d_engine
from fftpf.timestep_adaptive import check_stab

ENGINES = ("pf2d", "pf3d", "film")
SCHEMES = ("sbdf1", "sbdf2")


class Adapter:
    """engine: "pf2d" | "pf3d" | "film". pf2d/pf3d take `ops` (the engine's ops
    dict) and `elastic` (elastic_params or None). film takes
    film=dict(G=, kappa_b=, H11=None, poisson_factors=None) and `elastic` (el)."""

    def __init__(self, engine, grid, alpha, *, L=1.0, ops=None, film=None, elastic=None,
                 stab=0.0):
        if engine not in ENGINES:
            raise ValueError(f"engine: {engine!r} is not one of {ENGINES}")
        if engine == "film":
            if film is None or ops is not None:
                raise ValueError("film engine: pass film=dict(G, kappa_b, ...) and no ops")
            for key in ("G", "kappa_b"):
                if key not in film:
                    raise ValueError(f"film engine: film dict lacks {key!r}")
        elif ops is None or film is not None:
            raise ValueError(f"{engine} engine: pass ops= (the engine's ops dict) and no film")
        self.engine, self.grid, self.alpha, self.L = engine, grid, alpha, L
        self.ops, self.el = ops, elastic
        self.stab = check_stab(stab)
        self._kw = {"stab": self.stab} if self.stab else {}
        if engine == "film":
            self.G, self.kappa_b, self.H11 = film["G"], film["kappa_b"], film.get("H11")
            self.pf = film.get("poisson_factors")
            if self.pf is None:
                self.pf = film_engine.poisson_factor(grid, self.kappa_b)
        self.scheme = self.dt = self.kT = None
        self._op = self._op2 = self._ou = None

    # -- operators ---------------------------------------------------------
    def _build(self, dt, second):
        kw = self._kw
        if self.engine == "pf2d":
            fn = pf2d_engine.build_Minv2 if second else pf2d_engine.build_Minv
            return fn(self.grid, self.ops, dt, self.L, **kw)
        if self.engine == "pf3d":
            fn = pf3d_engine.build_Minv2_3d if second else pf3d_engine.build_Minv_3d
            return fn(self.grid, self.ops, dt, self.L, **kw)
        fn = film_engine.build_grad_factors_sbdf2 if second else film_engine.build_grad_factors
        return fn(self.grid, self.G, dt, self.L, H11=self.H11, **kw)

    def build(self, dt, *, scheme="sbdf1", kT=None, noise_cutoff_nm=None):
        """Build and hold the implicit operators: Minv (sbdf1), plus Minv2
        (sbdf2), plus the OU operators when kT is not None (Langevin)."""
        if scheme not in SCHEMES:
            raise ValueError(f"scheme: {scheme!r} is not one of {SCHEMES}")
        if kT is not None and scheme != "sbdf1":
            raise ValueError(f"kT with scheme={scheme!r}: the Langevin stepper is first order "
                             "only (exact OU + explicit drift); use scheme='sbdf1'")
        if kT is not None and self.stab:
            raise ValueError("kT with stab > 0: the Langevin steppers have no linear "
                             "stabilization; use stab=0")
        if noise_cutoff_nm is not None and kT is None:
            raise ValueError("noise_cutoff_nm needs kT: it only acts on the Langevin noise")
        self._op = self._build(dt, False)
        self._op2 = self._build(dt, True) if scheme == "sbdf2" else None
        self._ou = None
        if kT is not None:
            if self.engine == "pf2d":
                self._ou = pf2d_engine.build_ou_operators(
                    self.ops, dt, self.L, kT, noise_cutoff_nm=noise_cutoff_nm, grid=self.grid)
            elif self.engine == "pf3d":
                self._ou = pf3d_engine.build_ou_operators_3d(
                    self.ops, dt, self.L, kT, noise_cutoff_nm=noise_cutoff_nm, grid=self.grid)
            else:
                self._ou = film_langevin.build_ou_operators_film(
                    self.grid, self.G, dt, self.L, kT, noise_cutoff_nm=noise_cutoff_nm)
        self.scheme, self.dt, self.kT = scheme, dt, kT

    # -- stepping ----------------------------------------------------------
    def step(self, P, E, *, rng=None, P_prev=None):
        """One step; returns the new P. rng given -> Langevin (build(kT=...)
        required). sbdf2 with P_prev None -> one sbdf1 bootstrap step."""
        if self._op is None:
            raise ValueError("step: call build(dt, ...) first")
        if self.engine == "film" and np.ndim(P) != 4:
            raise ValueError("film engine: P must be (3,Nz,Ny,Nx); no ensemble axis")
        a, g, dt, L, kw = self.alpha, self.grid, self.dt, self.L, self._kw
        if rng is not None:
            if self._ou is None:
                raise ValueError("step: rng given but build() had kT=None (no OU operators)")
            if self.engine == "pf2d":
                return pf2d_engine.step_langevin(P, a, self.ops, self._ou, dt, L, E, rng,
                                                 elastic_params=self.el)
            if self.engine == "pf3d":
                return pf3d_engine.step_langevin_3d(P, a, self.ops, self._ou, dt, L, E, rng,
                                                    elastic_params=self.el)
            return film_langevin.step_langevin_film(P, a, g, self._ou, self.pf, dt, L, E, rng,
                                                    el=self.el)
        two = self.scheme == "sbdf2" and P_prev is not None
        if self.engine == "pf2d":
            if two:
                return pf2d_engine.step_sbdf2(P, P_prev, a, self.ops, self._op2, dt, L, E,
                                              elastic_params=self.el, **kw)
            return pf2d_engine.step(P, a, self.ops, self._op, dt, L, E,
                                    elastic_params=self.el, **kw)
        if self.engine == "pf3d":
            if two:
                return pf3d_engine.step_sbdf2_3d(P, P_prev, a, self.ops, self._op2, dt, L, E,
                                                 elastic_params=self.el, **kw)
            return pf3d_engine.step_3d(P, a, self.ops, self._op, dt, L, E,
                                       elastic_params=self.el, **kw)
        if two:
            return film_engine.step_sbdf2(P, P_prev, a, g, self._op2, self.pf, dt, L, E,
                                          el=self.el, **kw)
        return film_engine.step(P, a, g, self._op, self.pf, dt, L, E, el=self.el, **kw)

    # -- diagnostics -------------------------------------------------------
    def mean_P(self, P):
        """Volume mean, shape (3,); (3,S) for an ensemble P. pf2d/pf3d: plain
        mean over space. film: z-trapezoid mean, weights grid["qw"] / sum(qw),
        of the in-plane mean."""
        P = np.asarray(P)
        if self.engine == "film":
            if P.ndim != 4:
                raise ValueError("film engine: P must be (3,Nz,Ny,Nx); no ensemble axis")
            qw = self.grid["qw"]
            return (P.mean(axis=(2, 3)) * qw).sum(axis=1) / qw.sum()
        nsp = 2 if self.engine == "pf2d" else 3
        if P.ndim not in (1 + nsp, 2 + nsp):
            raise ValueError(f"{self.engine}: P has {P.ndim} axes, expected {1 + nsp} "
                             f"or {2 + nsp} (with ensemble axis)")
        return P.mean(axis=tuple(range(P.ndim - nsp, P.ndim)))

    def energy(self, P, E):
        """The engine's own total_energy return, (densities, totals), unchanged."""
        if self.engine == "pf2d":
            return pf2d_engine.total_energy(P, self.alpha, self.ops, E, self.grid,
                                            elastic_params=self.el)
        if self.engine == "pf3d":
            return pf3d_engine.total_energy_3d(P, self.alpha, self.ops, E, self.grid,
                                               elastic_params=self.el)
        return film_engine.total_energy_3d(P, self.alpha, self.grid, self.kappa_b, self.G,
                                           poisson_factors=self.pf, E_ext=E, el=self.el,
                                           H11=self.H11)

    def random_start(self, amplitude, kmax_frac, seed):
        """Band-limited random P (3,...) from the engine's own generator. film:
        independent in-plane field per z layer, seed*100000 + z (the convention
        of cases/bto_film_biaxial.initial_state)."""
        if self.engine == "pf2d":
            return pf2d_engine.random_smooth_field(self.grid, 3, amplitude, kmax_frac, seed)
        if self.engine == "pf3d":
            return pf3d_engine.random_smooth_field_3d(self.grid, 3, amplitude, kmax_frac, seed)
        return np.stack([pf2d_engine.random_smooth_field(
            self.grid, 3, amplitude=amplitude, kmax_frac=kmax_frac, seed=seed * 100000 + z)
            for z in range(self.grid["Nz"])], axis=1)
