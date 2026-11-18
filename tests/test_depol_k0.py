#!/usr/bin/env /opt/miniconda3/bin/python3
"""Portable regression for SPEC § 11 (DEPOL-K0): the optional k = 0
depolarization factor of bulk3d.build_operators_3d(..., depol_N=N).

Before § 11 the six depolarization symbols had D_ij(k=0) = 0 hard-coded (tin
foil). depol_N writes N into the k = 0 bin only: E_d(k=0) = -N.<P>/kappa0, the
energy density gains (1/2)<P>.N.<P>/kappa0, and the uniform mode of the linear
operator becomes Lin(0) = N/kappa0. This file checks, in miniature:

  1. depol_N=None and depol_N=0 are byte-identical to a test-local copy of the
     pre-§ 11 operators (ops, energy, force, Minv, Minv2, OU dict);
  2. uniform P (and a mean plus one cosine mode) against the closed forms;
  3. central FD vs force with every term on, including against an energy
     built from the tin-foil ops plus V*(1/2)<P>.N.<P>/kappa0;
  4. one step_3d / step_langevin_3d step scales a uniform eigen-mode by
     1/(1 + dt*L*n/kappa0) / exp(-dt*L*n/kappa0), the pre-§ 11 ops miss both,
     and the k = 0 OU noise amplitude and stationary variance match closed form;
  5. the refusals (array kappa, non-symmetric N, eigenvalue outside [0, 1]);
  6. the ensemble axis: S = 1 and S = 2 batches equal the unbatched calls.

Run: /opt/miniconda3/bin/python3 -m pytest tests/test_depol_k0.py -q

Scope: a targeted regression, NOT a CORE gate claim -- the core items are
K1.fd_3d_depol_k0, K2.depol_k0_off_byte, K2.depol_k0_spm_refused,
K2.depol_k0_bad_N_refused, K2.depol_k0_ensemble_s1_byte,
K3.depol_k0_uniform_closed_form, K3.depol_k0_mode_rates in gates/core/ (local)."""
import os

# Thread caps MUST precede the numpy/scipy imports (ENGINE_GUIDE 1.2).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pytest

from solvers import bulk3d

G = (1.3, -0.2, 0.9, 0.55)
KAPPA = 0.885
ALPHA = dict(a1=-0.3, a11=0.6, a12=0.15, a111=0.05, a112=0.02, a123=0.01)
ALPHA_OFF = dict(a1=0.0, a11=0.0, a12=0.0, a111=0.0, a112=0.0, a123=0.0)
E_EXT = (0.02, -0.015, 0.01)
ZERO = (0.0, 0.0, 0.0)
N_ANISO = np.array([[0.40, 0.05, -0.03], [0.05, 0.30, 0.02], [-0.03, 0.02, 0.30]])
C, Q = (3.02, 1.62, 0.68), (0.1104, -0.0452, 0.01445)
DT, L, KT = 0.1, 1.0, 1e-3
GRIDS = (dict(Nx=6, Ny=4, Nz=4, Lx=3.0, Ly=2.4, Lz=2.8),     # even
         dict(Nx=5, Ny=7, Nz=3, Lx=10.0, Ly=14.0, Lz=7.5))   # odd


def _ops_pre(grid):
    """Copy of bulk3d.build_operators_3d at a9496680 (before SPEC § 11),
    scalar kappa: D_ij(k=0) = 0."""
    K1, K2, K3 = grid["KXr"], grid["KYr"], grid["KZr"]
    A11, A22, A33, A12, A13, A23 = bulk3d._grad_symbols_3d(grid, *G)
    k2 = K1**2 + K2**2 + K3**2
    k2safe = np.where(k2 == 0.0, 1.0, k2)
    D11 = np.where(k2 == 0.0, 0.0, K1 * K1 / k2safe)
    D22 = np.where(k2 == 0.0, 0.0, K2 * K2 / k2safe)
    D33 = np.where(k2 == 0.0, 0.0, K3 * K3 / k2safe)
    D12 = np.where(k2 == 0.0, 0.0, K1 * K2 / k2safe)
    D13 = np.where(k2 == 0.0, 0.0, K1 * K3 / k2safe)
    D23 = np.where(k2 == 0.0, 0.0, K2 * K3 / k2safe)
    if grid["Nx"] % 2 == 0:
        D12[:, :, -1] = 0.0
        D13[:, :, -1] = 0.0
    if grid["Ny"] % 2 == 0:
        D12[:, grid["Ny"] // 2, :] = 0.0
        D23[:, grid["Ny"] // 2, :] = 0.0
    if grid["Nz"] % 2 == 0:
        D13[grid["Nz"] // 2, :, :] = 0.0
        D23[grid["Nz"] // 2, :, :] = 0.0
    return dict(A11=A11, A22=A22, A33=A33, A12=A12, A13=A13, A23=A23,
                D11=D11, D22=D22, D33=D33, D12=D12, D13=D13, D23=D23,
                G11=G[0], G12=G[1], G44=G[2], G44p=G[3], kappa=KAPPA,
                kappa0=KAPPA, dkappa=None, spm=False, spm_warm=False, _spm_warm=None,
                Nx=grid["Nx"], Ny=grid["Ny"], Nz=grid["Nz"], Nxr=grid["Nxr"])


def _same(a, b):
    """Bit-for-bit (0.0 vs -0.0 differ), recursing into dicts and tuples."""
    if isinstance(a, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _field(grid, seed, mean=(0.05, -0.04, 0.03)):
    rng = np.random.default_rng(seed)
    P = rng.normal(0.0, 0.05, (3, grid["Nz"], grid["Ny"], grid["Nx"]))
    return P + np.asarray(mean)[:, None, None, None]


@pytest.fixture(autouse=True)
def _single_fft_worker():
    saved = bulk3d.FFT_WORKERS
    bulk3d.FFT_WORKERS = 1
    yield
    bulk3d.FFT_WORKERS = saved


@pytest.mark.parametrize("gspec", GRIDS)
def test_off_is_byte_identical(gspec):
    grid = bulk3d.make_grid_3d(**gspec)
    pre = _ops_pre(grid)
    P = _field(grid, 1)

    def numbers(ops):
        return (bulk3d.total_energy_3d(P, ALPHA, ops, E_EXT, grid),
                bulk3d.total_force_3d(P, ALPHA, ops, E_EXT),
                bulk3d.build_Minv_3d(grid, ops, DT, L), bulk3d.build_Minv2_3d(grid, ops, DT, L),
                bulk3d.build_ou_operators_3d(ops, DT, L, KT))

    ref = numbers(pre)
    for dN in (None, 0):
        ops = bulk3d.build_operators_3d(grid, *G, KAPPA, depol_N=dN)
        extra = set(ops) - set(pre)
        assert extra == (set() if dN is None else {"depol_N"})
        assert all(_same(ops[k], pre[k]) for k in pre if k not in ("dkappa", "_spm_warm"))
        assert ops["dkappa"] is None and ops["_spm_warm"] is None
        got = numbers(ops)
        assert all(_same(x, y) for x, y in zip(got, ref))


@pytest.mark.parametrize("gspec", GRIDS)
@pytest.mark.parametrize("dN", (0.3, N_ANISO))
def test_uniform_and_mode_closed_form(gspec, dN):
    grid = bulk3d.make_grid_3d(**gspec)
    N = dN * np.eye(3) if np.ndim(dN) == 0 else dN
    ops = bulk3d.build_operators_3d(grid, *G, KAPPA, depol_N=dN)
    shp = (grid["Nz"], grid["Ny"], grid["Nx"])
    Pu = np.array([0.12, -0.07, 0.05])
    P = np.broadcast_to(Pu[:, None, None, None], (3,) + shp).copy()
    Ed = bulk3d.elec_Ed_3d(P, ops)
    np.testing.assert_allclose(Ed, (-(N @ Pu) / KAPPA)[:, None, None, None] * np.ones(shp),
                               rtol=0, atol=1e-13 * np.max(np.abs(N @ Pu / KAPPA)))
    f = bulk3d.elec_energy_density_3d(P, ops, ZERO)
    np.testing.assert_allclose(f, 0.5 * Pu @ N @ Pu / KAPPA, rtol=1e-13, atol=0)
    # mean + one cosine mode: the two closed forms add
    z = np.arange(grid["Nz"])[:, None, None] * grid["hz"]
    y = np.arange(grid["Ny"])[None, :, None] * grid["hy"]
    x = np.arange(grid["Nx"])[None, None, :] * grid["hx"]
    kv = 2.0 * np.pi * np.array([1.0 / grid["Lx"], 1.0 / grid["Ly"], 1.0 / grid["Lz"]])
    khat = kv / np.linalg.norm(kv)
    cosr = np.cos(kv[0] * x + kv[1] * y + kv[2] * z)
    Pbar, amp = np.array([0.03, 0.06, -0.04]), np.array([0.09, -0.05, 0.07])
    Pm = Pbar[:, None, None, None] + amp[:, None, None, None] * cosr[None]
    want = (-(N @ Pbar) / KAPPA)[:, None, None, None] \
        - (khat * (khat @ amp) / KAPPA)[:, None, None, None] * cosr[None]
    np.testing.assert_allclose(bulk3d.elec_Ed_3d(Pm, ops), want, rtol=0,
                               atol=1e-13 * np.max(np.abs(want)))
    fm = float(bulk3d.elec_energy_density_3d(Pm, ops, ZERO).mean())
    assert fm == pytest.approx(0.5 * Pbar @ N @ Pbar / KAPPA + 0.25 * (khat @ amp) ** 2 / KAPPA,
                               rel=1e-13)


@pytest.mark.parametrize("gspec", GRIDS)
def test_fd_every_term(gspec):
    grid = bulk3d.make_grid_3d(**gspec)
    ops = bulk3d.build_operators_3d(grid, *G, KAPPA, depol_N=N_ANISO)
    ops_tf = bulk3d.build_operators_3d(grid, *G, KAPPA)
    el = bulk3d.setup_elastic_3d(grid, C=C, Q=Q, k0="stress_free")
    P, dP = _field(grid, 2), _field(grid, 3, mean=(0.03, 0.02, -0.04))
    dV = grid["hx"] * grid["hy"] * grid["hz"]

    def F(Pv, indep):
        m = Pv.mean(axis=(1, 2, 3))
        elec = (bulk3d.elec_energy_density_3d(Pv, ops_tf, E_EXT).sum() * dV
                + grid["Volume"] * 0.5 * m @ N_ANISO @ m / KAPPA) if indep else \
            bulk3d.elec_energy_density_3d(Pv, ops, E_EXT).sum() * dV
        rest = (bulk3d.bulk_energy_density(Pv, ALPHA) + bulk3d.grad_energy_density_3d(Pv, grid, ops)
                + bulk3d.elastic_energy_density_3d(bulk3d.elastic_solve_3d(Pv, el))).sum() * dV
        return float(elec + rest)

    force = bulk3d.total_force_3d(P, ALPHA, ops, E_EXT, elastic_params=el)[-1]
    analytic = float(np.sum(force * dP)) * dV
    eps = 1e-5
    for indep in (False, True):
        fd = (F(P + eps * dP, indep) - F(P - eps * dP, indep)) / (2 * eps)
        assert abs(fd - analytic) / max(abs(fd), abs(analytic)) < 1e-6


@pytest.mark.parametrize("gspec", GRIDS)
def test_mode_rates_and_ou_noise(gspec):
    grid = bulk3d.make_grid_3d(**gspec)
    shp = (grid["Nz"], grid["Ny"], grid["Nx"])
    ops = bulk3d.build_operators_3d(grid, *G, KAPPA, depol_N=N_ANISO)
    pre = _ops_pre(grid)
    lam, V = np.linalg.eigh(N_ANISO)
    for o, expect_hit in ((ops, True), (pre, False)):
        Minv = bulk3d.build_Minv_3d(grid, o, DT, L)
        ou0 = bulk3d.build_ou_operators_3d(o, DT, L, 0.0)
        for j in range(3):
            P = np.broadcast_to((0.1 * V[:, j])[:, None, None, None], (3,) + shp).copy()
            r_be = bulk3d.step_3d(P, ALPHA_OFF, o, Minv, DT, L, ZERO).mean(axis=(1, 2, 3)) \
                / (0.1 * V[:, j])
            r_ou = bulk3d.step_langevin_3d(P, ALPHA_OFF, o, ou0, DT, L, ZERO,
                                           np.random.default_rng(0)).mean(axis=(1, 2, 3)) \
                / (0.1 * V[:, j])
            f_be = 1.0 / (1.0 + DT * L * lam[j] / KAPPA)
            f_ou = np.exp(-DT * L * lam[j] / KAPPA)
            hit = np.allclose(r_be, f_be, rtol=1e-12, atol=0) and \
                np.allclose(r_ou, f_ou, rtol=1e-12, atol=0)
            assert hit == expect_hit
    ou = bulk3d.build_ou_operators_3d(ops, DT, L, KT)
    na, dc = ou["noise_amp"][0, 0, 0], ou["decay"][0, 0, 0]
    std = np.sqrt(KT * KAPPA / lam * (-np.expm1(-2.0 * DT * L * lam / KAPPA)))
    np.testing.assert_allclose(na, V @ np.diag(std) @ V.T, rtol=0, atol=1e-12 * std.max())
    for j in range(3):
        s, d = V[:, j] @ na @ V[:, j], V[:, j] @ dc @ V[:, j]
        assert s * s / (1.0 - d * d) == pytest.approx(KT * KAPPA / lam[j], rel=1e-12)


def test_refusals():
    grid = bulk3d.make_grid_3d(**GRIDS[1])
    kappa_r = np.full((grid["Nz"], grid["Ny"], grid["Nx"]), KAPPA)
    kappa_r[0] *= 2.0
    for dN in (0.3, 0, N_ANISO):
        with pytest.raises(ValueError, match="SPEC § 11"):
            bulk3d.build_operators_3d(grid, *G, kappa_r, depol_N=dN)
    assert bulk3d.build_operators_3d(grid, *G, kappa_r)["spm"]
    nonsym = N_ANISO.copy()
    nonsym[0, 1] += 1e-9
    for bad in (nonsym, -0.1, 1.2, np.diag([0.2, 0.3, 1.0 + 1e-9]),
                np.array([[0.3, 0.4, 0.0], [0.4, 0.3, 0.0], [0.0, 0.0, 0.3]]),
                np.array([0.1, 0.2, 0.3]), float("nan")):
        with pytest.raises(ValueError, match="SPEC § 11"):
            bulk3d.build_operators_3d(grid, *G, KAPPA, depol_N=bad)
    for good in (0.0, 1.0, N_ANISO):
        ops = bulk3d.build_operators_3d(grid, *G, KAPPA, depol_N=good)
        N = good * np.eye(3) if np.ndim(good) == 0 else good
        assert ops["D12"][0, 0, 0] == N[0, 1] and ops["D33"][0, 0, 0] == N[2, 2]


def test_ensemble_s1_s2_byte():
    grid = bulk3d.make_grid_3d(**GRIDS[0])
    ops = bulk3d.build_operators_3d(grid, *G, KAPPA, depol_N=N_ANISO)
    Minv = bulk3d.build_Minv_3d(grid, ops, DT, L)
    el = bulk3d.setup_elastic_3d(grid, C=C, Q=Q, k0="stress_free")
    ou, ou0 = (bulk3d.build_ou_operators_3d(ops, DT, L, kT) for kT in (KT, 0.0))

    def run(P, ou_=None, steps=5):
        rng = np.random.default_rng(7)
        for _ in range(steps):
            P = (bulk3d.step_3d(P, ALPHA, ops, Minv, DT, L, E_EXT, elastic_params=el)
                 if ou_ is None else bulk3d.step_langevin_3d(P, ALPHA, ops, ou_, DT, L, E_EXT, rng))
        return P

    P0 = _field(grid, 4)
    assert np.array_equal(run(P0), run(P0[:, None])[:, 0])
    assert np.array_equal(run(P0, ou), run(P0[:, None], ou)[:, 0])
    PB = np.stack([P0, _field(grid, 5)], axis=1)
    for ou_ in (None, ou0):
        out = run(PB, ou_)
        assert all(np.array_equal(out[:, i], run(PB[:, i].copy(), ou_)) for i in range(2))
    eB = bulk3d.total_energy_3d(PB, ALPHA, ops, E_EXT, grid, elastic_params=el)[1]
    for i in range(2):
        ei = bulk3d.total_energy_3d(PB[:, i].copy(), ALPHA, ops, E_EXT, grid, elastic_params=el)[1]
        assert all(np.float64(ei[k]).tobytes() == np.float64(eB[k][i]).tobytes() for k in ei)
