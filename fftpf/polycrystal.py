"""2D periodic polycrystal layer for the 2D engine (solver.py): grain
geometry (periodic Voronoi), a grain-boundary SHELL label of exact area
fraction (Cai et al., APL 119, 032901 (2021) core-shell ceramic; Su, Liu &
Weng, Acta Mater. 87, 293 (2015) GB-affected-zone / dead-layer model), and
PER-GRAIN CRYSTAL ORIENTATION for the Landau sector -- the one thing
solver.py could not express before this module: its cubic f_bulk is
written in ONE fixed crystal frame, so a polycrystal whose grains are
rotated in-plane (Su 2015 S3: 0/30/45/60 degrees; Cai 2021 inherits the
model) had no representation.

How the rotation enters (and only there): for a grain rotated by theta
about z, the Landau energy is f_bulk(R(theta)^T P) with the SAME alpha's;
by the chain rule the force in the global frame is R(theta) * bulk_force(
R(theta)^T P). The isotropic gradient energy (Su-Landis a0, engine ratios
(1,0,0.5,0.5)) and the electrostatics are frame-invariant, so `step` below
is solver.step with ONLY its explicit real-space N(P) term swapped for the
rotated Landau force -- the implicit k-space operator M(k), the FFT
plumbing and the SBDF1 update are solver.py's own, called unchanged.
P3 stays identically zero for an in-plane start (no force couples it in),
matching Su 2015's plane problem (P3=E3=D3=0).

Everything alpha-array-valued rides on the contract SPEC S1.1 /
composite.py already use ("alpha may be a spatial array"). The orientation
fields ride along INSIDE the alpha dict under the private keys "_cos" and
"_sin" (bulk_force reads only ALPHA_KEYS, so solver.py ignores them), which
keeps casekit.anneal / timestep.verified_dt's engine interface intact:
pass THIS module as `engine`.

Elasticity: NOT supported here (raises). A rotated polycrystal needs
per-grain rotated C and Q; solvers/elastic.py is homogeneous-C fixed-frame
(its WP10 SPM path takes cubic-form C11/C12/C44 FIELDS, not rotated
tensors). Recorded as an engine gap for the local full test
(reports/CLOUD_TEST_NOTE.md); the cloud benchmark runs elastic OFF on the
stress-free effective Landau set (solvers/su_landis.py).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from fftpf import pf2d_engine as solver

ALPHA_KEYS = solver.ALPHA_KEYS + solver.ALPHA_KEYS_8TH
FFT_WORKERS = 1


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def _poisson_seeds(n, Lx, Ly, rng, dmin_frac=0.62, max_tries=200000):
    """n seeds uniformly in the periodic box with a periodic minimum
    separation dmin = dmin_frac*sqrt(Lx*Ly/n) (dart throwing, seeded)."""
    dmin = dmin_frac * np.sqrt(Lx * Ly / n)
    pts = []
    for _ in range(max_tries):
        p = rng.uniform((0.0, 0.0), (Lx, Ly))
        ok = True
        for q in pts:
            d = np.abs(p - q)
            d = np.minimum(d, np.array([Lx, Ly]) - d)
            if np.hypot(d[0], d[1]) < dmin:
                ok = False
                break
        if ok:
            pts.append(p)
            if len(pts) == n:
                return np.array(pts)
    raise RuntimeError(f"_poisson_seeds: could not place {n} seeds with dmin={dmin:.1f}")


def periodic_voronoi(grid, n_side, seed, jitter=0.35, mode="poisson"):
    """Periodic Voronoi tessellation of the (Lx,Ly) box with n_side^2 seed
    points. mode='poisson' (default): seeds uniformly random with a
    periodic minimum separation (0.62 x the mean seed spacing) -- irregular
    polygonal grains like a ceramic micrograph (Su 2015 Fig. 1/8). mode=
    'lattice': an n_side x n_side lattice each displaced by +-jitter*cell --
    NOTE this keeps the lattice rows, so nearly straight full-width grain
    boundaries appear between seed rows (seen in the first cloud test).
    Returns (grain_id (Ny,Nx) int, seeds (n,2) in nm [x,y], dist_gb (Ny,Nx)
    nm = distance to the nearest grain boundary, computed EXACTLY as the
    smallest distance from the point to any bisector between its own seed
    and another seed over the 9 periodic images -- for a convex Voronoi
    cell that is the distance to its boundary)."""
    Lx, Ly, Nx, Ny = grid["Lx"], grid["Ly"], grid["Nx"], grid["Ny"]
    rng = np.random.default_rng(seed)
    if mode == "poisson":
        seeds = _poisson_seeds(n_side * n_side, Lx, Ly, rng)
    elif mode == "lattice":
        cx, cy = Lx / n_side, Ly / n_side
        sx = (np.arange(n_side) + 0.5) * cx
        sy = (np.arange(n_side) + 0.5) * cy
        SX, SY = np.meshgrid(sx, sy, indexing="xy")
        seeds = np.stack([SX.ravel(), SY.ravel()], axis=1)
        seeds += rng.uniform(-jitter, jitter, seeds.shape) * np.array([cx, cy])
        seeds %= np.array([Lx, Ly])
    else:
        raise ValueError(mode)
    x = (np.arange(Nx) + 0.5) * (Lx / Nx)
    y = (np.arange(Ny) + 0.5) * (Ly / Ny)
    X, Y = np.meshgrid(x, y, indexing="xy")                     # (Ny,Nx)
    # periodic images of every seed
    shifts = np.array([(i * Lx, j * Ly) for i in (-1, 0, 1) for j in (-1, 0, 1)])
    img = (seeds[:, None, :] + shifts[None, :, :]).reshape(-1, 2)   # (9n,2)
    owner = np.repeat(np.arange(len(seeds)), len(shifts))
    d2 = (X[..., None] - img[:, 0]) ** 2 + (Y[..., None] - img[:, 1]) ** 2   # (Ny,Nx,9n)
    j_near = np.argmin(d2, axis=-1)
    grain_id = owner[j_near]
    d2_own = np.take_along_axis(d2, j_near[..., None], axis=-1)[..., 0]
    # distance to the bisector between own nearest image s0 and image sj:
    #   (|x-sj|^2 - |x-s0|^2) / (2 |sj - s0|)
    s0 = img[j_near]                                            # (Ny,Nx,2)
    sep = np.sqrt(((img[None, None, :, :] - s0[:, :, None, :]) ** 2).sum(-1))  # (Ny,Nx,9n)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist_bis = (d2 - d2_own[..., None]) / (2.0 * sep)
    dist_bis = np.where(owner[None, None, :] == grain_id[..., None], np.inf, dist_bis)
    dist_gb = np.min(dist_bis, axis=-1)
    return grain_id, seeds, dist_gb


def shell_label(dist_gb, fraction):
    """Binary shell label L(r) in {0=core, 1=shell} of EXACT area fraction
    `fraction` (rank-and-cut on the GB distance, the composite.py
    construction: the fraction of cells closest to a grain boundary is the
    shell). fraction=0 -> no shell."""
    n = dist_gb.size
    k = int(round(fraction * n))
    L = np.zeros(dist_gb.shape, dtype=float)
    if k <= 0:
        return L
    order = np.argsort(dist_gb, axis=None, kind="stable")
    L.flat[order[:k]] = 1.0
    return L


def shell_thickness_nm(dist_gb, L):
    """Mean shell half-width proxy: the largest GB distance inside the
    shell (the cut radius), i.e. the shell extends this far from the GB on
    each side -> shell thickness ~ 2*value."""
    return float(dist_gb[L > 0.5].max()) if np.any(L > 0.5) else 0.0


def orientation_fields(grain_id, angles_deg):
    """(cos theta, sin theta) fields from per-grain angles (degrees, in-plane
    rotation of the cubic axes about z)."""
    th = np.deg2rad(np.asarray(angles_deg, dtype=float))[grain_id]
    return np.cos(th), np.sin(th)


def make_alpha(core_alpha, shell_alpha, L, cos_th, sin_th):
    """alpha_i(r) = core_i*(1-L) + shell_i*L for every 6th+8th-order key
    (composite.alpha_field's rule, extended to the 8th-order keys), plus the
    orientation fields under private keys. The shell is isotropic (a1
    only), so rotating it is a no-op -- one cos/sin field serves both.

    A coefficient may also be an ARRAY broadcasting against L, e.g. the
    a1 field crystallinity.effective_alpha returns for an Xc(r) field;
    bulk_force/bulk_energy_density already take array alphas (SPEC S1.1).
    Such a key is always blended, with no "equal, so keep the core value"
    shortcut. Scalar keys take the scalar path below unchanged."""
    out = {}
    for k in ALPHA_KEYS:
        c = core_alpha.get(k, 0.0)
        s = shell_alpha.get(k, 0.0)
        if np.ndim(c) or np.ndim(s):
            c = np.asarray(c, dtype=float)
            s = np.asarray(s, dtype=float)
            try:
                ok = np.broadcast_shapes(c.shape, s.shape, np.shape(L)) == np.shape(L)
            except ValueError:
                ok = False
            if not ok:
                raise ValueError(f"make_alpha: {k} core/shell shapes {c.shape}/{s.shape} "
                                 f"must broadcast to the label field's shape {np.shape(L)}")
            out[k] = c * (1.0 - L) + s * L
            continue
        c = float(c)
        s = float(s)
        out[k] = c * (1.0 - L) + s * L if (c != s) else c
    out["_cos"] = cos_th
    out["_sin"] = sin_th
    return out


# --------------------------------------------------------------------------
# Rotated Landau sector
# --------------------------------------------------------------------------
def _to_local(P, c, s):
    """P_local = R(theta)^T P  (in-plane), P3 untouched."""
    Q = np.empty_like(P)
    Q[0] = c * P[0] + s * P[1]
    Q[1] = -s * P[0] + c * P[1]
    Q[2] = P[2]
    return Q


def _to_global(F, c, s):
    """F_global = R(theta) F_local."""
    G = np.empty_like(F)
    G[0] = c * F[0] - s * F[1]
    G[1] = s * F[0] + c * F[1]
    G[2] = F[2]
    return G


def bulk_force(P, alpha):
    """delta F_bulk/delta P in the GLOBAL frame for per-grain rotated cubic
    Landau energy: R * solver.bulk_force(R^T P, alpha)."""
    c, s = alpha.get("_cos"), alpha.get("_sin")
    if c is None:
        return solver.bulk_force(P, alpha)
    return _to_global(solver.bulk_force(_to_local(P, c, s), alpha), c, s)


def bulk_energy_density(P, alpha):
    c, s = alpha.get("_cos"), alpha.get("_sin")
    if c is None:
        return solver.bulk_energy_density(P, alpha)
    return solver.bulk_energy_density(_to_local(P, c, s), alpha)


# --------------------------------------------------------------------------
# Engine interface (solver-shaped: build_Minv / step / total_energy /
# probe_stable_dt), see module docstring.
# --------------------------------------------------------------------------
build_Minv = solver.build_Minv
block_stationary = solver.block_stationary
random_smooth_field = solver.random_smooth_field
build_ou_operators = solver.build_ou_operators


def step_langevin(P, alpha, ops, ou, dt, L, E_ext, rng, elastic_params=None):
    """Present so casekit.anneal's engine check (`hasattr(engine,
    'step_langevin')`) accepts this module for its DETERMINISTIC path
    (kT=None); the thermal path is not wired for the rotated force."""
    raise NotImplementedError("polycrystal: Langevin stepping not wired (rotated Landau force); use kT=None")


def _explicit_N(P, alpha, ops, E_ext, elastic_params):
    if elastic_params is not None:
        raise NotImplementedError("polycrystal: elasticity with per-grain rotated C/Q is not "
                                  "available in the 2D engine (elastic.py is homogeneous-C, "
                                  "fixed-frame) -- run elastic OFF; gap recorded for the local full test")
    if ops.get("dkappa") is not None:
        raise NotImplementedError("polycrystal: kappa(x) SPM path not wired here (shell is a PE filler)")
    return bulk_force(P, alpha) - solver._reshape_vec3(E_ext, P.ndim, ensemble=P.ndim == 4)


def step(P, alpha, ops, Minv, dt, L, E_ext, elastic_params=None):
    """One SBDF1 semi-implicit step, solver.step with the rotated Landau force
    as its explicit term (same M(k), same FFT plumbing)."""
    N = _explicit_N(P, alpha, ops, E_ext, elastic_params)
    s = (ops["Ny"], ops["Nx"])
    Pk = solver._rfft2(P)
    Nk = solver._rfft2(N)
    RHS = Pk - dt * L * Nk
    Pk_new = np.empty_like(RHS)
    Pk_new[0] = Minv[..., 0, 0] * RHS[0] + Minv[..., 0, 1] * RHS[1] + Minv[..., 0, 2] * RHS[2]
    Pk_new[1] = Minv[..., 1, 0] * RHS[0] + Minv[..., 1, 1] * RHS[1] + Minv[..., 1, 2] * RHS[2]
    Pk_new[2] = Minv[..., 2, 0] * RHS[0] + Minv[..., 2, 1] * RHS[1] + Minv[..., 2, 2] * RHS[2]
    return solver._irfft2(Pk_new, s)


def total_force(P, alpha, ops, E_ext, elastic_params=None):
    """(Fb, Fg, Fe, sum) in the global frame -- the G0-style consistency
    instrument for this module (see cases/cloud_cai2021_vortex_relax.py's
    self-check)."""
    if elastic_params is not None:
        raise NotImplementedError("polycrystal: elastic OFF only")
    Fb = bulk_force(P, alpha)
    Fg = solver.grad_force(P, ops)
    Fe = solver.elec_force(P, ops, E_ext)
    return Fb, Fg, Fe, Fb + Fg + Fe


def total_energy(P, alpha, ops, E_ext, grid, elastic_params=None, spm_tol=1e-10):
    """Mirrors solver.total_energy's (dens, tot) contract with the rotated
    Landau density."""
    if elastic_params is not None:
        raise NotImplementedError("polycrystal: elastic OFF only")
    fb = bulk_energy_density(P, alpha)
    fg = solver.grad_energy_density(P, grid, ops)
    fe = solver.elec_energy_density(P, ops, E_ext, spm_tol=spm_tol)
    dens = dict(bulk=float(fb.mean()), grad=float(fg.mean()), elec=float(fe.mean()))
    dens["total"] = sum(dens.values())
    tot = {k: v * grid["Volume"] for k, v in dens.items()}
    return dens, tot


def probe_stable_dt(P0, alpha, ops, grid, L, E_ext=(0., 0., 0.), elastic_params=None,
                    dt0=0.02, n_probe=200, growth=2.0, p_cap=None, max_doublings=30, scheme="sbdf1"):
    """solver.probe_stable_dt's doubling protocol on THIS module's step."""
    if scheme != "sbdf1":
        raise ValueError("polycrystal: sbdf1 only")
    if p_cap is None:
        p_cap = max(10.0 * np.max(np.abs(P0)), 5.0)
    dt = dt0
    last_stable = None
    for _ in range(max_doublings):
        Minv = build_Minv(grid, ops, dt, L)
        P = P0.copy()
        ok = True
        for _ in range(n_probe):
            P = step(P, alpha, ops, Minv, dt, L, E_ext, elastic_params=elastic_params)
            if not np.all(np.isfinite(P)) or np.max(np.abs(P)) > p_cap:
                ok = False
                break
        if not ok:
            break
        last_stable = dt
        dt *= growth
    if last_stable is None:
        raise RuntimeError(f"polycrystal.probe_stable_dt: dt0={dt0} already unstable")
    return last_stable / 2.0


# --------------------------------------------------------------------------
# Diagnostics: vortex cores (winding number) and per-grain descriptors
# --------------------------------------------------------------------------
def winding_number(P, p_min=0.0):
    """Winding number of the in-plane P direction around every grid
    plaquette (periodic), shape (Ny,Nx), values in {-1,0,+1} (rarely +-2);
    a plaquette whose four corners are not all above |P|>p_min is set to 0
    (the shell, where P is a tiny linear response, would otherwise produce
    spurious windings)."""
    ang = np.arctan2(P[1], P[0])
    mag = np.hypot(P[0], P[1])
    a00 = ang
    a01 = np.roll(ang, -1, axis=1)          # x+1
    a11 = np.roll(np.roll(ang, -1, axis=1), -1, axis=0)   # x+1, y+1
    a10 = np.roll(ang, -1, axis=0)          # y+1

    def wrap(d):
        return (d + np.pi) % (2.0 * np.pi) - np.pi

    w = wrap(a01 - a00) + wrap(a11 - a01) + wrap(a10 - a11) + wrap(a00 - a10)
    w = np.rint(w / (2.0 * np.pi)).astype(int)
    if p_min > 0.0:
        ok = ((mag > p_min) & (np.roll(mag, -1, 1) > p_min)
              & (np.roll(np.roll(mag, -1, 1), -1, 0) > p_min) & (np.roll(mag, -1, 0) > p_min))
        w = np.where(ok, w, 0)
    return w


def grain_descriptors(P, grain_id, L, n_grains, p_min):
    """Per grain (core cells only): mean |P|, |<P>|/<|P|> (net-polarization
    fraction: ~1 single domain, ~0 closed flux), <P_y>/<|P|>, number of
    +-1 vortex cores (plaquettes fully inside the core)."""
    core = L < 0.5
    mag = np.hypot(P[0], P[1])
    w = winding_number(P, p_min=p_min)
    corners_core = (core & np.roll(core, -1, 1) & np.roll(np.roll(core, -1, 1), -1, 0) & np.roll(core, -1, 0))
    rows = []
    for g in range(n_grains):
        m = core & (grain_id == g)
        if m.sum() == 0:
            rows.append(dict(grain=g, n_cells=0))
            continue
        mp = float(mag[m].mean())
        px, py = float(P[0][m].mean()), float(P[1][m].mean())
        net = float(np.hypot(px, py) / mp) if mp > 0 else 0.0
        mw = corners_core & (grain_id == g)
        rows.append(dict(grain=g, n_cells=int(m.sum()), mean_absP=mp, net_frac=net,
                         Py_over_absP=(py / mp if mp > 0 else 0.0),
                         n_vortex=int(np.abs(w[mw]).sum())))
    return rows


if __name__ == "__main__":
    # smoke test: geometry + force consistency (central-FD of the rotated energy vs the rotated force)
    from fftpf import mat_bto_su_landis_landau as su_landis
    N, h = 96, 1.0
    grid = solver.make_grid(N, N, N * h, N * h)
    deck = su_landis.deck()
    ops = solver.build_operators(grid, **deck["gradient"], kappa=deck["kappa_b"])
    gid, seeds, dgb = periodic_voronoi(grid, 3, seed=0)
    Lsh = shell_label(dgb, 0.15)
    print("grains:", np.bincount(gid.ravel()), " shell frac:", Lsh.mean(), " cut radius nm:", shell_thickness_nm(dgb, Lsh))
    cth, sth = orientation_fields(gid, np.random.default_rng(0).uniform(0, 90, 9))
    alpha = make_alpha(deck["landau"], su_landis.shell_alpha_deck(20.0), Lsh, cth, sth)
    rng = np.random.default_rng(1)
    P = rng.normal(0, 0.15, (3, N, N)); P[2] = 0.0
    F = bulk_force(P, alpha)
    # FD check at a few random cells
    eps = 1e-6
    errs = []
    for _ in range(20):
        i, j, k = rng.integers(0, 2), rng.integers(0, N), rng.integers(0, N)
        Pp = P.copy(); Pp[i, j, k] += eps
        Pm = P.copy(); Pm[i, j, k] -= eps
        fd = (bulk_energy_density(Pp, alpha)[j, k] - bulk_energy_density(Pm, alpha)[j, k]) / (2 * eps)
        errs.append(abs(fd - F[i, j, k]) / max(abs(F[i, j, k]), 1e-12))
    print("rotated force vs FD, max relerr:", max(errs))
    # energy invariance: rotating a uniform state by the grain angle must give the same Landau energy as the unrotated
    Pu = np.zeros((3, N, N)); Pu[0] = 0.2
    e_local = solver.bulk_energy_density(Pu, {k: alpha[k] for k in ALPHA_KEYS})
    Pr = np.stack([cth * 0.2, sth * 0.2, np.zeros((N, N))])
    e_rot = bulk_energy_density(Pr, alpha)
    print("frame-invariance max abs diff:", np.abs(e_rot - e_local).max())
