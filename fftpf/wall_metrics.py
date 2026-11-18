"""SPEC S36.3 (WP25 Phase 2, stage 2C): domain-wall energy/position shared
metrics, merged from solvers/wall_utils.py + solvers/walls.py (WP24/SPEC
S35 moved them here from cases/, AS-IS, as two separate modules --
merging them into one was explicitly deferred to Phase 2, see SPEC S35.1
"DEFERRED TO PHASE 2"). Both old names (solvers/wall_utils.py,
solvers/walls.py) are now PERMANENT forwarding shims to this module
(SPEC S35 user ruling (3), extended to Phase 2 renames by SPEC S36.5); new
code should import solvers.wall_metrics directly.

The two halves below answer DIFFERENT questions and never call each
other (unchanged from the pre-merge module docstrings, kept verbatim so
this file stays the single reference for both):

  WALL ENERGY (T180/R109 family, quasi-1D scenarios) -- was
  solvers/wall_utils.py. sigma_w = [sum_x (f(x) - f_uniform) * hx] /
  n_walls * 1000 (mJ/m^2). Ferret's own formula is (Ftotal[aJ] -
  fu[aJ/nm^3]*V[nm^3])/A[nm^2] * 1000; for a y,z-independent field the
  Delta_y*Delta_z factors in Ftotal and A cancel exactly, leaving
  [sum_x (f-fu)*hx]/n_walls in native aJ/nm^2 = J/m^2 -- hence the same
  *1000 -> mJ/m^2 conversion Ferret applies (WP2_REPORT.md has the full
  derivation). 2*xi from curve_fit of A*tanh((x-x0)/xi) to the P profile
  near each wall (same functional form/default fit half-window, 2.0nm,
  as Ferret's own wall_metrics.wall()). 1/L extrapolation: same
  np.linalg.lstsq(A=[1, 1/L], sigma) as Ferret's wall_metrics.py __main__
  block.

  WALL POSITION/ROUGHNESS (WP4 S10(g), periodic-aware, general 2D
  patterns) -- was solvers/walls.py. Historic bug being fixed (per
  project memory / Ferret-side wall_roughness.py): when a wall's traced
  x(y) position crosses the periodic x boundary (wraps from near Lx back
  to near 0, or vice versa), a naive roughness = std(x(y)) sees a huge
  SPURIOUS jump (~Lx) and reports a large fake roughness even for a
  perfectly straight wall -- the bug only ever OVERESTIMATES (adds a
  jump), never underestimates. This half tracks each wall's crossings
  row-by-row with a PERIODIC nearest-neighbor match, then UNWRAPS the
  resulting x(y) sequence (removing +-Lx jumps) before computing any
  statistic on it. Method: (1) per row y, find all sub-pixel zero
  crossings of the (periodic-in-x) input field via linear interpolation
  between adjacent samples, INCLUDING the wrap-around pair (Nx-1, 0) --
  this is what lets a wall sitting exactly at the periodic seam be found
  at all; (2) assume a constant wall count across rows (true for the
  non-branching stripe/wall patterns this project's cases produce); track
  each wall as a sequence by periodic-nearest-neighbor matching from row
  to row; (3) unwrap each track (remove jumps > Lx/2 by adding/
  subtracting Lx), making x(y) continuous even through the periodic
  boundary; (4) roughness(track) = std(unwrapped x(y)) (per-wall); wall
  count = number of tracks; mean period = Lx / wall count.

GATES: g2/g3/g4/g7c anchor the wall-ENERGY half and g7g the wall-POSITION
half, all importing the pre-merge names bare (`import wall_utils as wu`,
`import walls`) off cases/, which forwards through solvers/wall_utils.py /
solvers/walls.py to here -- so grepping "wall_metrics" in gates/ finds
nothing.
"""
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from scipy.optimize import curve_fit

from fftpf import pf2d_engine as solver
from fftpf import elastic_base as elastic_mod
from fftpf import timestep_verify as timestep  # WP28 T4; timestep imports only numpy -- no cycle


# ==========================================================================
# Wall ENERGY (T180/R109 family, quasi-1D) -- was solvers/wall_utils.py.
# ==========================================================================
def parse_tanh_params(expr):
    """'A*tanh((H-abs(x))/W)' -> (A, H, W)."""
    m = re.match(r"\s*([\d.eE+-]+)\*tanh\(\(([\d.eE+-]+)-abs\(x\)\)/([\d.eE+-]+)\)\s*$", expr)
    if not m:
        raise ValueError(f"unrecognized tanh IC expression: {expr!r}")
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def periodic_tanh_wall_1d(grid, amplitude, width, wall_frac=0.25):
    """Two-wall periodic tanh profile matching k_t180.i/k_r109_*.i's IC convention
    (+amplitude in the middle, -amplitude outside; walls at x=wall_frac*Lx and
    x=(1-wall_frac)*Lx in this grid's [0,Lx) coordinate). Returns (Nx,) 1D array."""
    X = np.arange(grid["Nx"]) * grid["hx"]
    xs = X - grid["Lx"] / 2.0  # shift to the deck's symmetric [-Lx/2,Lx/2) convention
    half_width = wall_frac * grid["Lx"]
    return amplitude * np.tanh((half_width - np.abs(xs)) / width)


def total_density_field(P, alpha, ops, E_ext, grid, elastic_params=None):
    """Per-point total energy density (Ny,Nx), summing solver.py's per-term
    density arrays directly (total_energy() only returns scalar means)."""
    f = (solver.bulk_energy_density(P, alpha)
         + solver.grad_energy_density(P, grid, ops)
         + solver.elec_energy_density(P, ops, E_ext))
    if elastic_params is not None:
        sol = elastic_mod.elastic_solve(P, elastic_params)
        f = f + elastic_mod.elastic_energy_density(sol)
    return f


def wall_sigma(P, alpha, ops, E_ext, grid, f_uniform, n_walls=2, elastic_params=None):
    """sigma_w in mJ/m^2 (see module docstring for the unit derivation)."""
    f = total_density_field(P, alpha, ops, E_ext, grid, elastic_params)
    f_x = f.mean(axis=0)  # average over y (exact no-op when Ny=1)
    return float(np.sum(f_x - f_uniform) * grid["hx"] / n_walls * 1000.0)


def fit_wall_width(x, profile, x0, half=2.0):
    """curve_fit A*tanh((x-x0)/xi) near x0 (same form/window as wall_metrics.wall()).
    Returns (amplitude, x0_fit, 2*xi)."""
    mask = np.abs(x - x0) < half
    popt, _ = curve_fit(lambda t, A, t0, xi: A * np.tanh((t - t0) / xi),
                         x[mask], profile[mask], p0=[profile[mask][-1], x0, 0.3])
    return popt[0], popt[1], 2 * abs(popt[2])


def extrapolate_1_over_L(Ls, vals):
    """Linear least-squares fit vs 1/L; returns (intercept_at_Linf, slope)."""
    A = np.vstack([np.ones(len(Ls)), 1.0 / np.array(Ls, dtype=float)]).T
    c, *_ = np.linalg.lstsq(A, np.array(vals, dtype=float), rcond=None)
    return float(c[0]), float(c[1])


def setup_case_elastic(grid, prm):
    """elastic.setup_elastic from a case prm dict's C11/C12/C44 + Q11_phys/Q12_phys/Q44_phys (elastic.py's Voigt Q, no /4)."""
    return elastic_mod.setup_elastic(grid, prm["C11"], prm["C12"], prm["C44"],
                                      prm["Q11_phys"], prm["Q12_phys"], prm["Q44_phys"])


def make_tie(i, j):
    """post_step hook: force P[i]=P[j]=average(P[i],P[j]) every step -- mirrors
    gld1d.py's own `tie=(1,2)` mechanism for the R109 Ising/SPP branch (SPEC's
    branch is a saddle point of the free relaxation; see G4 report section for
    the empirical drift measurement that motivated this)."""
    def tie(P):
        P = P.copy()
        avg = 0.5 * (P[i] + P[j])
        P[i] = avg
        P[j] = avg
        return P
    return tie


def compute_f_uniform(prm, P0_guess, n_max=20000, tol_F=1e-13, n_plateau=30, scheme="sbdf1",
                       dt_tol=1e-8):
    """Relax a tiny homogeneous-IC grid (with elasticity active) to get the
    stress-free uniform-phase energy density used as the wall-energy baseline.
    scheme: passed through to solver.relax (WP4 §10c: "sbdf1" or "sbdf2").
    Returns (f_uniform, <P>, dt_used, n_steps) -- unchanged 4-tuple.

    WP28 T4 (coordinator ruling, SPEC §40.5). This REPLACED
    `solver.relax(..., dt=None)`, which takes its dt from
    solver.probe_stable_dt -- a BOUNDEDNESS probe (solver.py:614-636: only
    "finite and max|P| <= p_cap" over n_probe steps), never a convergence
    check. SPEC §36.3 / ENGINE_GUIDE §3.4 already say new code must use
    timestep.verified_dt for exactly this reason; this function predated that
    law and was still on the probe.
    WHAT WENT WRONG (the incident that motivated the fix, gates/g7c part 3):
    for scheme="sbdf2" on the R109 deck the probe's dt=2.56 rung is past
    SBDF2's convergence limit but its orbit is CHAOTIC-BUT-BOUNDED, so
    whether it trips p_cap=5.0 within 200 steps is decided by roundoff (a
    4.9e-15 change in bulk_force flipped it; the two orbits diverge by
    Lyapunov growth to O(1) within 60 steps). Once accepted, the probe
    returned 1.28 instead of 0.64, this relaxation converged to a DIFFERENT
    uniform state (<P> 0.2089 vs 0.2202) and f_uniform came out 1.78% low
    (-0.011068066 vs -0.011269011). wall_sigma below is
    sum(f - f_uniform)*hx/n_walls*1000 -- a catastrophic cancellation -- so
    that baseline error is amplified by L*500: sigma moved -3.2152 at L=32
    and -6.4302 at L=64, i.e. G4(sbdf2) read -1.8694 instead of 7.7759.
    NOW: timestep.verified_dt halves from the probe's dt until the OBSERVABLE
    THIS FUNCTION RETURNS -- the relaxed f_uniform itself -- is dt-independent.
    The standard adapter does not fit (it marches a fixed T_total and compares
    mean|P|, whereas this is a relax-to-plateau whose observable is the
    converged energy density), so the `run_fn` escape hatch is used: run_fn
    re-runs THIS relaxation at each trial dt and returns its f_uniform.
    dt_tol=1e-8 is relative agreement on f_uniform. Justification: sigma
    amplifies a baseline error by L*500, so 1e-8 relative (~1e-10 absolute
    here) moves sigma by <=4e-6 at L=64 -- four orders inside the 2% the
    sigma gates allow -- while staying well above the noise floor of the
    tol_F=1e-13 plateau itself (callers' own wall runs use tol_F=1e-10).
    CLASS: TOLERANCE. For every sbdf1 caller the converged fixed point was
    already dt-independent to ~1e-10, so G2/G3/G4's numbers move at most in
    their last digits (before/after measured in reports/WP28_2_REPORT.md);
    the sbdf2 path is where the value actually changes -- back to the sbdf1
    answer, which is the correct one.
    GATES: G2, G3, G4, G7c (part 3 is the one that caught this), G7g, g28a."""
    grid = solver.make_grid(Nx=8, Ny=1, Lx=1.0, Ly=prm["deck_h"])
    ops = solver.build_operators(grid, prm["G11"], prm["G12"], prm["G44"], prm["G44p"], prm["kappa"])
    el = setup_case_elastic(grid, prm)
    P0 = np.zeros((3, grid["Ny"], grid["Nx"]))
    for c in range(3):
        P0[c] = P0_guess[c]

    def _relax_at(dt_try):
        """One full relax-to-plateau at dt_try -> its f_uniform (None if it
        blew up, which makes verified_dt back off before comparing)."""
        Pf_, _dt, _n, _h = solver.relax(P0, prm["alpha"], ops, grid, prm["L"], elastic_params=el,
                                        dt=dt_try, tol_F=tol_F, n_max=n_max,
                                        n_plateau=n_plateau, scheme=scheme)
        if not np.all(np.isfinite(Pf_)):
            return None
        d = total_density_field(Pf_, prm["alpha"], ops, (0., 0., 0.), grid, elastic_params=el)
        return float(d.mean())

    dt, _info = timestep.verified_dt(solver, P0, prm["alpha"], ops, grid, L=prm["L"],
                                     T_total=float(n_max), elastic_params=el, scheme=scheme,
                                     tol=dt_tol, run_fn=_relax_at)
    Pf, dt, n, hist = solver.relax(P0, prm["alpha"], ops, grid, prm["L"], elastic_params=el,
                                    dt=dt, tol_F=tol_F, n_max=n_max, n_plateau=n_plateau,
                                    scheme=scheme)
    dens = total_density_field(Pf, prm["alpha"], ops, (0., 0., 0.), grid, elastic_params=el)
    return float(dens.mean()), Pf.mean(axis=(1, 2)), dt, n


def run_wall_series(prm, ic_builder, Ls, n_max=8000, tol_F=1e-10, n_plateau=25,
                     check_every=20, post_step=None, dt=None, scheme="sbdf1"):
    """Relax the periodic 2-wall IC at each box length in Ls (quasi-1D, Ny=1,
    h=prm['deck_h']). scheme: passed through to solver.relax (WP4 §10c).
    Returns a list of per-L result dicts."""
    out = []
    for Lx in Ls:
        Nx = int(round(Lx / prm["deck_h"]))
        grid = solver.make_grid(Nx=Nx, Ny=1, Lx=Lx, Ly=prm["deck_h"])
        P0 = ic_builder(grid, prm)
        ops = solver.build_operators(grid, prm["G11"], prm["G12"], prm["G44"], prm["G44p"], prm["kappa"])
        el = setup_case_elastic(grid, prm)
        Pf, dt_used, nsteps, hist = solver.relax(
            P0, prm["alpha"], ops, grid, prm["L"], elastic_params=el, dt=dt,
            tol_F=tol_F, n_max=n_max, check_every=check_every, n_plateau=n_plateau, post_step=post_step,
            scheme=scheme)
        out.append(dict(Lx=Lx, grid=grid, ops=ops, el=el, Pf=Pf, dt=dt_used, nsteps=nsteps, hist=hist))
    return out


# ==========================================================================
# Wall POSITION/ROUGHNESS (WP4 S10(g), periodic-aware) -- was solvers/walls.py.
# ==========================================================================
def _crossings_1d_periodic(row, hx, Lx):
    """Sub-pixel x positions (ascending, each in [0,Lx)) where the periodic
    1D array `row` crosses zero (linear interpolation between adjacent
    samples, INCLUDING the wrap-around pair (Nx-1,0))."""
    Nx = len(row)
    xs = []
    for i in range(Nx):
        j = (i + 1) % Nx
        a, b = row[i], row[j]
        if (a < 0) == (b < 0):
            continue  # no sign change (also skips a==b==0)
        t = a / (a - b)  # fraction of the way from i to j where it crosses 0
        x = ((i + t) * hx) % Lx
        xs.append(x)
    return sorted(xs)


def _periodic_delta(x1, x0, Lx):
    """Signed displacement x1-x0 taking the periodic image nearest x0
    (in (-Lx/2, Lx/2])."""
    d = (x1 - x0 + Lx / 2) % Lx - Lx / 2
    return d


def track_walls(field, grid):
    """field: (Ny,Nx) real array (e.g. Pz), periodic in x, sampled at
    y=0..Ly-hy. Returns list of dicts, one per tracked wall:
    {y: (Ny,), x: (Ny,) unwrapped sub-pixel x position, x_mean, roughness}.
    Raises if the per-row crossing count is not constant (this project's
    wall patterns don't branch; a varying count means the field isn't a
    clean stripe/wall pattern and the tracker's assumption doesn't hold)."""
    Ny, Nx = grid["Ny"], grid["Nx"]
    hx, Lx = grid["hx"], grid["Lx"]
    hy = grid["hy"]
    Y = np.arange(Ny) * hy

    rows_x = [_crossings_1d_periodic(field[y, :], hx, Lx) for y in range(Ny)]
    counts = {len(r) for r in rows_x}
    if len(counts) != 1:
        raise ValueError(f"track_walls: crossing count varies by row ({sorted(counts)}) -- "
                          "not a clean non-branching wall pattern")
    n_walls = counts.pop()
    if n_walls == 0:
        return []

    # Initialize tracks from row 0 (already sorted ascending in x).
    tracks_x = [[rows_x[0][k]] for k in range(n_walls)]
    prev = list(rows_x[0])
    for y in range(1, Ny):
        cur = rows_x[y]
        # periodic-nearest-neighbor match: for each track, pick the closest
        # unclaimed crossing in `cur` (small n_walls in every use case here,
        # so an O(n^2) greedy match is plenty and easy to verify correct).
        used = [False] * len(cur)
        new_prev = [None] * n_walls
        for k in range(n_walls):
            dists = [abs(_periodic_delta(cur[m], prev[k], Lx)) if not used[m] else np.inf
                     for m in range(len(cur))]
            m_best = int(np.argmin(dists))
            used[m_best] = True
            # unwrap: continue the track by the SIGNED periodic displacement
            # from its own last (already-unwrapped) position, not the raw
            # cur[m_best] value -- this is the fix for the historic bug.
            step = _periodic_delta(cur[m_best], prev[k], Lx)
            tracks_x[k].append(tracks_x[k][-1] + step)
            new_prev[k] = cur[m_best]
        prev = new_prev

    out = []
    for k in range(n_walls):
        x = np.array(tracks_x[k])
        out.append(dict(y=Y, x=x, x_mean=float(x.mean()), roughness=float(x.std())))
    out.sort(key=lambda d: d["x_mean"] % Lx)
    return out


def measure_walls(field, grid):
    """Top-level SPEC §10(g) entry point: wall count, mean period, roughness
    (mean over tracked walls; per-wall list also returned)."""
    tracks = track_walls(field, grid)
    n = len(tracks)
    period = grid["Lx"] / n if n > 0 else float("nan")
    roughness_per_wall = [t["roughness"] for t in tracks]
    roughness_mean = float(np.mean(roughness_per_wall)) if n > 0 else float("nan")
    return dict(count=n, period=period, roughness=roughness_mean,
                roughness_per_wall=roughness_per_wall, tracks=tracks)


def fit_sinusoidal_amplitude(track):
    """Fit x(y) = x0 + A*sin(2*pi*m*y/Ly + phi) to a tracked wall (m fixed
    at the dominant FFT peak of x-mean(x), everything else free). Returns
    fitted A (>=0)."""
    y, x = track["y"], track["x"]
    Ly = y[-1] + (y[1] - y[0]) if len(y) > 1 else 1.0
    xc = x - x.mean()
    spec = np.abs(np.fft.rfft(xc))
    m = int(np.argmax(spec[1:]) + 1) if len(spec) > 1 else 1  # exclude DC

    def model(yy, A, phi, x0):
        return x0 + A * np.sin(2 * np.pi * m * yy / Ly + phi)

    p0 = [xc.std() * np.sqrt(2), 0.0, x.mean()]
    popt, _ = curve_fit(model, y, x, p0=p0, maxfev=20000)
    return abs(popt[0])
