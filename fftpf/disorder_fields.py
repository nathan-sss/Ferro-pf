"""WP3 §6.2(a)(b): spatial Landau disorder helpers and the static eigenstrain
disorder generator.

Filter-exponent note (read before changing anything here): SPEC §6.2b says
"spectral Gaussian filter exp(-k^2 xi^2/4)". Read as a filter on the AMPLITUDE
spectrum, applying it to white noise gives a POWER spectrum exp(-k^2 xi^2/2),
whose real-space correlation is exp(-r^2/(2 xi^2)) -- NOT exp(-r^2/xi^2).
Read instead as describing the resulting POWER spectrum (the natural reading
of "spectral filter", and the exact phrase used by the reference generator's
own docstring, `make_disorder.py`: "功率谱 P(k) ∝ exp(-k^2 xi^2/4) 给出关联函数
C(r) ∝ exp(-r^2/xi^2)" -- power spectrum P(k) ∝ exp(-k^2 xi^2/4) gives
correlation function C(r) ∝ exp(-r^2/xi^2)), the amplitude filter is
exp(-k^2 xi^2/8) (since power = |amplitude|^2). This is what's implemented
below; verified empirically against G5a's own measurement (see WP3_REPORT.md):
the exp(-k^2 xi^2/4)-on-amplitude reading measured ~46% off target at xi=2nm,
while exp(-k^2 xi^2/8) measured ~0.5-2%.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from scipy.optimize import curve_fit


# --------------------------------------------------------------------------
# (a) Spatial Landau disorder helpers
# --------------------------------------------------------------------------
def piecewise_regions_1d(grid, axis, boundaries, values):
    """Piecewise-constant field along one axis (e.g. dielectric layers with
    different alpha1). `boundaries` are ascending coordinates in [0, L) on the
    given axis (0='x', 1='y'); `values[i]` fills [boundaries[i], boundaries[i+1])
    (periodic wrap for the last segment: [boundaries[-1], L) plus
    [0, boundaries[0]); a single boundary therefore fills the whole axis)."""
    assert len(values) == len(boundaries)
    coord = (np.arange(grid["Nx"]) * grid["hx"] if axis == 0 else np.arange(grid["Ny"]) * grid["hy"])
    field_1d = np.empty_like(coord)
    b = list(boundaries) + [boundaries[0]]
    for i in range(len(values)):
        lo, hi = b[i], b[i + 1]
        if i < len(values) - 1 and lo <= hi:
            mask = (coord >= lo) & (coord < hi)
        else:  # wraps past the periodic boundary
            mask = (coord >= lo) | (coord < hi)
        field_1d[mask] = values[i]
    if axis == 0:
        return np.tile(field_1d, (grid["Ny"], 1))
    return np.tile(field_1d[:, None], (1, grid["Nx"]))


def random_alpha1(grid, mean, amplitude, kmax_frac=0.3, seed=0):
    """Smooth random alpha1(x) field (thin wrapper on solver.random_smooth_field,
    WP1) -- e.g. mean<0 with amplitude>|mean| gives a mixed-sign field (G5b)."""
    from fftpf import pf2d_engine as solver
    return mean + solver.random_smooth_field(grid, 1, amplitude, kmax_frac=kmax_frac, seed=seed)[0]


# --------------------------------------------------------------------------
# (b) Static eigenstrain disorder: white noise -> spectral Gaussian filter ->
# zero mean -> rescale to exact target RMS (SPEC §6.2b)
# --------------------------------------------------------------------------
def random_field_gaussian(grid, rms, xi, seed):
    """Single zero-mean field with power spectrum ~exp(-k^2 xi^2/4) (real-space
    correlation ~exp(-r^2/xi^2)), rescaled to EXACTLY the target RMS.

    2D only: a bulk3d grid with Nz > 1 is REFUSED. Its (Nz,Ny,Nx) KX/KY would
    broadcast one (Ny,Nx) noise draw into a z-stack of near-copies, not a 3D
    field -- use random_field_gaussian_3d (or the caller's own *_3d twin).
    A film3d grid (Nz > 1 but in-plane (Ny,Nx) KX/KY) still gets its (Ny,Nx)
    in-plane field."""
    if grid.get("Nz", 1) > 1 and np.ndim(grid["KX"]) == 3:
        raise ValueError(
            f"disorder.random_field_gaussian is 2D-only, got a 3D grid with Nz={grid['Nz']}: "
            "use disorder.random_field_gaussian_3d (or the *_3d twin of the calling "
            "generator, e.g. quenched_fields.charge_disorder_field_3d, "
            "composition.composition_field_3d, composite.make_label_field_multi_3d)")
    KX, KY = grid["KX"], grid["KY"]
    k2 = KX**2 + KY**2
    amp_filter = np.exp(-k2 * xi**2 / 8.0)  # amplitude filter; see module docstring
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((grid["Ny"], grid["Nx"]))
    nk = np.fft.fft2(noise) * amp_filter
    nk[0, 0] = 0.0  # zero mean
    field = np.fft.ifft2(nk).real
    field -= field.mean()  # exact zero mean (guards residual fp noise)
    std = field.std()
    if std > 0:
        field *= rms / std  # exact target RMS
    return field


def random_field_gaussian_3d(grid, rms, xi, seed):
    """SPEC V2-03 A6. 3D twin of random_field_gaussian: a single zero-mean
    (Nz,Ny,Nx) field, EXACTLY the target RMS, on a bulk3d.make_grid_3d grid
    (keys Nz/Ny/Nx and the FULL-spectrum KX/KY/KZ -- duck-typed, this module
    imports no engine file).

    SAME FILTER, SAME EXPONENT, NO 3D RE-DERIVATION NEEDED. The module
    docstring's amplitude filter exp(-k^2 xi^2/8) -> power spectrum
    exp(-k^2 xi^2/4) -> correlation exp(-r^2/xi^2) chain is DIMENSION-
    INDEPENDENT: in d dimensions C(r) = int P(k) e^{ik.r} d^dk/(2pi)^d, and
    the Fourier transform of exp(-a k^2) is proportional to exp(-r^2/(4a))
    in EVERY d (only the prefactor, which the exact-RMS rescale below
    overwrites anyway, carries the d-dependence). With a = xi^2/4 that is
    exp(-r^2/xi^2) in 2D and in 3D alike -- so xi means the same thing here
    as in the 2D generator and nothing about the variance normalisation
    changes either: the field is rescaled to EXACTLY `rms` at the end,
    whatever the grid's dimension or resolution.

    Nz=1 DESIGN RULE (SPEC V2-03 A6, gated by K2.generators_nz1_vs_2d): the
    white noise is drawn as ONE (Nz,Ny,Nx) standard_normal call, so at Nz=1
    the draw is the SAME stream in the SAME C order as the 2D generator's
    (Ny,Nx) call; KZ is then [0.0] so KX^2+KY^2+KZ^2 adds an exact 0.0, and
    np.fft.fftn/ifftn on a length-1 leading axis reproduce fft2/ifft2
    bit-for-bit (the same identity bulk3d relies on for rfftn, gate G27a).
    Hence random_field_gaussian_3d(...)[0] == random_field_gaussian(...)
    byte for byte at Nz=1 -- verified, not assumed."""
    KX, KY, KZ = grid["KX"], grid["KY"], grid["KZ"]
    k2 = KX**2 + KY**2 + KZ**2
    amp_filter = np.exp(-k2 * xi**2 / 8.0)  # amplitude filter; see module docstring
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((grid["Nz"], grid["Ny"], grid["Nx"]))
    nk = np.fft.fftn(noise) * amp_filter
    nk[0, 0, 0] = 0.0  # zero mean
    field = np.fft.ifftn(nk).real
    field -= field.mean()  # exact zero mean (guards residual fp noise)
    std = field.std()
    if std > 0:
        field *= rms / std  # exact target RMS
    return field


def make_random_strain(grid, eps_rms, xi, seed, components=("xx", "yy", "xy")):
    """SPEC §6.2b generator. Returns {comp: (Ny,Nx) array}, one independently-
    seeded field per named component (seed+i for the i-th component in
    `components`, matching the reference generator's "independently seeded"
    but with a simpler, explicit offset)."""
    return {c: random_field_gaussian(grid, eps_rms, xi, seed + i) for i, c in enumerate(components)}


def assemble_eps_star_static(grid, comp_dict):
    """{comp in xx/yy/zz/xy/xz/yz: (Ny,Nx) or (S,Ny,Nx)} -> full symmetric
    (3,3,Ny,Nx) or (3,3,S,Ny,Nx) eps_star_static tensor for
    elastic.setup_elastic (missing components = 0). Shape-generic (WP4 §10e:
    works for a batched (S,Ny,Nx) comp_dict without any extra call)."""
    idx = dict(xx=(0, 0), yy=(1, 1), zz=(2, 2), xy=(0, 1), xz=(0, 2), yz=(1, 2))
    sample_shape = next(iter(comp_dict.values())).shape
    out = np.zeros((3, 3) + sample_shape)
    for comp, field in comp_dict.items():
        i, j = idx[comp]
        out[i, j] = field
        out[j, i] = field
    return out


# --------------------------------------------------------------------------
# WP4 §10(e): ensemble batching -- "make_random_strain vectorized over
# seeds". random_field_gaussian_batch generates S independent realizations
# (RNG draws are still per-seed, unavoidably serial -- each seed needs its
# own stream -- but cheap next to the FFT) via ONE rfft2/irfft2 pass over a
# stacked (S,Ny,Nx) array, instead of S separate calls each paying its own
# FFT round trip. Uses scipy.fft directly (not solver.py's _rfft2/_irfft2,
# which are private to that module), configured by solvers.execution.
# --------------------------------------------------------------------------
FFT_WORKERS = -1


def random_field_gaussian_batch(grid, rms, xi, seeds):
    """Batched random_field_gaussian: returns (S,Ny,Nx), slice s independently
    seeded with seeds[s] and rescaled to EXACTLY the target RMS per slice
    (same per-realization exact-RMS contract as the single-seed version)."""
    import scipy.fft as spfft
    KX, KY = grid["KXr"], grid["KYr"]
    amp_filter = np.exp(-(KX**2 + KY**2) * xi**2 / 8.0)  # see module docstring for the 1/8 derivation
    noise = np.stack([np.random.default_rng(s).standard_normal((grid["Ny"], grid["Nx"])) for s in seeds])
    nk = spfft.rfft2(noise, axes=(-2, -1), workers=FFT_WORKERS) * amp_filter
    nk[:, 0, 0] = 0.0  # zero mean, per realization
    field = spfft.irfft2(nk, s=(grid["Ny"], grid["Nx"]), axes=(-2, -1), workers=FFT_WORKERS)
    field = field - field.mean(axis=(-2, -1), keepdims=True)  # exact zero mean (roundoff guard)
    std = field.std(axis=(-2, -1), keepdims=True)
    scale = np.where(std > 0, rms / np.where(std > 0, std, 1.0), 1.0)
    return field * scale


def make_random_strain_batch(grid, eps_rms, xi, seeds, components=("xx", "yy", "xy")):
    """Batched make_random_strain (WP4 §10e): returns {comp: (S,Ny,Nx)}, with
    slice s of component i using seed seeds[s]+i -- EXACTLY reproducing what
    S separate make_random_strain(..., seed=seeds[s]) calls would give (see
    gate G7e)."""
    return {c: random_field_gaussian_batch(grid, eps_rms, xi, [s + i for s in seeds])
            for i, c in enumerate(components)}


# --------------------------------------------------------------------------
# Field statistics (G5a): radially-averaged autocorrelation, Gaussian fit
# --------------------------------------------------------------------------
def radial_autocorrelation(field, grid, nbins, rmax):
    """Radially-averaged (Wiener-Khinchin) autocorrelation, normalized to 1 at r=0."""
    Ny, Nx = grid["Ny"], grid["Nx"]
    F = np.fft.fft2(field)
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(F)**2).real)
    ac /= ac[Ny // 2, Nx // 2]
    hx, hy = grid["hx"], grid["hy"]
    yy, xx = np.indices(ac.shape)
    r = np.sqrt(((xx - Nx // 2) * hx)**2 + ((yy - Ny // 2) * hy)**2)
    rbins = np.linspace(0, rmax, nbins + 1)
    prof = np.full(nbins, np.nan)
    for i in range(nbins):
        sel = ac[(r >= rbins[i]) & (r < rbins[i + 1])]
        if sel.size > 0:  # a bin can be empty if its width undersamples the pixel grid
            prof[i] = sel.mean()
    rc = 0.5 * (rbins[1:] + rbins[:-1])
    return rc, prof


def fit_correlation_length(field, grid, xi_guess, fit_mult=0.75, nbins=30):
    """Gaussian fit C(r)=exp(-r^2/xi^2) to the radially-averaged autocorrelation,
    fit over r in [0, fit_mult*xi_guess] -- the "core" where the fit is least
    contaminated by the noisy far tail (see WP3_REPORT.md for the empirical
    tuning: fitting out to r~2*xi biases xi low by >20% at xi=8nm)."""
    rmax = fit_mult * xi_guess
    rc, prof = radial_autocorrelation(field, grid, nbins, rmax)
    m = np.isfinite(prof) & (rc > 0)
    popt, _ = curve_fit(lambda r, xi: np.exp(-r**2 / xi**2), rc[m], prof[m], p0=[xi_guess])
    return popt[0]


# --------------------------------------------------------------------------
# WP4 §10(f): shared-realization ("master-grid") mode. Generate ONE
# realization at N_max resolution; every target N resamples FROM that same
# master field (solver.spectral_resample_2d) instead of being independently
# seeded -- SPEC: "never regenerate per N: same seed at different N is a
# different realization". Used by item (h)'s 128^2/256^2/512^2 resolution
# study so all three see the identical physical disorder pattern.
# --------------------------------------------------------------------------
def make_random_strain_master(N_max, Lx, Ly, eps_rms, xi, seed, components=("xx", "yy", "xy")):
    """Generate the master realization at N_max x N_max (same physical box
    Lx,Ly every resample will share). Returns {comp: (N_max,N_max) array}."""
    from fftpf import pf2d_engine as solver
    grid_max = solver.make_grid(Nx=N_max, Ny=N_max, Lx=Lx, Ly=Ly)
    return make_random_strain(grid_max, eps_rms, xi, seed, components=components)


def resample_strain_dict(strain_master, N_new, eps_rms):
    """Spectrally resample each component of a master-grid strain dict
    (make_random_strain_master's output) to N_new x N_new, then rescale back
    to EXACTLY eps_rms at the new resolution -- consistent with
    random_field_gaussian's own exact-RMS-at-its-resolution contract (a
    deliberate, documented design choice: rescaling here does not hide a
    resampling defect, since gate G7f's round-trip check compares the FULL
    field, not just this scalar)."""
    from fftpf import pf2d_engine as solver
    out = {}
    for c, field in strain_master.items():
        resampled = solver.spectral_resample_2d(field, N_new, N_new)
        resampled = resampled - resampled.mean()
        std = resampled.std()
        if std > 0:
            resampled = resampled * (eps_rms / std)
        out[c] = resampled
    return out
