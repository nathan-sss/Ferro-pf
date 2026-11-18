#!/usr/bin/env /opt/miniconda3/bin/python3
"""Portable regression for SPEC §10 (OU-NEG): the Langevin OU propagator on
an INDEFINITE linear operator Lin(k).

Before §10, solver._ou_from_lin (reused by bulk3d) and film3d_langevin._z_eigen
did `np.clip(lam, 0, None)`: every negative eigenvalue became a neutral mode,
so on the H11 lane (G < 0 < H11, params/nnbm_wang2026.toml) the k* instability
vanished from Langevin runs. §10 replaced the clip by a roundoff floor (only
|lam| <= 1e3*eps64*max|lam(k)| negatives go to 0) and made the film OU refuse
G11 < 0 (it carries no H11 regulariser). This file checks, in miniature:

  1. byte identity of the OU dict on a PSD deck (2D, 3D, film) against a
     test-local copy of the pre-§10 formula;
  2. the k* mode of nnbm_wang2026 grows by exactly
     (1 - 2*L*a1*dt)*exp(-L*lambda*dt) in one kT = 0 step_langevin step
     (lambda* < 0), and the pre-§10 formula misses that factor;
  3. build_ou_operators_film raises for G11 < 0.

Run: /opt/miniconda3/bin/python3 -m pytest tests/test_ou_indefinite.py -q

Scope: a targeted regression, NOT a CORE gate claim -- the core items are
K2.ou_psd_byte, K2.film_ou_negative_g_refused, K3.ou_negative_mode_{2d,3d},
K4.ou_negative_energy_monotone in gates/core/ (local)."""
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

from solvers import bulk3d, grad4, materials, solver
from solvers import film3d as f3
from solvers import film3d_langevin as fl

DT, L, KT = 0.1, 1.0, 1e-3


def _phi1_pre(x):
    """Copy of solver._phi1 as of 06c1dfb."""
    small = np.abs(x) < 1e-8
    xs = np.where(small, 1.0, x)
    reg = -np.expm1(-xs) / xs
    ser = 1.0 - x / 2.0 + x * x / 6.0
    return np.where(small, ser, reg)


def _ou_from_lin_pre(Lin, dt, L, kT):
    """Copy of solver._ou_from_lin BEFORE SPEC §10 (np.clip included)."""
    lam, V = np.linalg.eigh(Lin)
    lam = np.clip(lam, 0.0, None)
    Lambda = L * lam
    decay_eig = np.exp(-Lambda * dt)
    std_eig = np.sqrt(2.0 * kT * L * dt * _phi1_pre(2.0 * Lambda * dt))
    decay = np.einsum("...ij,...j,...kj->...ik", V, decay_eig, V)
    noise_amp = np.einsum("...ij,...j,...kj->...ik", V, std_eig, V)

    def cf(M):
        return np.ascontiguousarray(np.moveaxis(M, (-2, -1), (0, 1))).astype(np.complex128)
    return dict(decay=decay, noise_amp=noise_amp, lam=lam,
                decay_cf=cf(decay), noise_amp_cf=cf(noise_amp))


def _film_ou_pre(grid, G11, dt, L, kT):
    """Copy of film3d_langevin._z_eigen + build_ou_operators_film (no cutoff)
    BEFORE SPEC §10 (np.clip included, no G11 refusal)."""
    sub, diag, sup = f3._grad_z_bands(grid["Nz"], grid["hz"], G11)
    T0 = np.diag(diag) + np.diag(sup, 1) + np.diag(sub, -1)
    sqrt_w = np.sqrt(grid["qw"] / grid["hz"])
    B = T0 * sqrt_w[:, None] / sqrt_w[None, :]
    B = 0.5 * (B + B.T)
    lam, U = np.linalg.eigh(B)
    lam = np.clip(lam, 0.0, None)
    k2 = grid["KXr"] ** 2 + grid["KYr"] ** 2
    stiffness = lam[:, None, None] + G11 * k2[None, :, :]
    Lambda = L * stiffness
    var = 2.0 * kT * L * dt * _phi1_pre(2.0 * Lambda * dt)
    return dict(U=U, sqrt_w=sqrt_w, lam=lam, k2=k2, stiffness=stiffness, Lambda=Lambda,
                decay=np.exp(-Lambda * dt), noise_std=np.sqrt(var),
                kT=float(kT), dt=float(dt), L=float(L),
                n_inplane=int(grid["Nx"] * grid["Ny"]))


def _assert_same_bytes(new, old):
    assert set(new) == set(old)
    for k in old:
        a, b = np.asarray(new[k]), np.asarray(old[k])
        assert a.dtype == b.dtype and a.shape == b.shape, k
        assert a.tobytes() == b.tobytes(), k          # bit-for-bit: -0.0 != 0.0 here


@pytest.fixture(scope="module")
def bto():
    d = materials.from_toml(os.path.join(_ROOT, "params", "bto_298K.toml"))
    g = d["gradient"]
    return (g["G11"], g["G12"], g["G44"], g["G44p"], d["kappa_b"])


@pytest.fixture(scope="module")
def nnbm():
    return grad4.load_nnbm_deck()


@pytest.mark.parametrize("N", [16, 15])
def test_psd_byte_2d(bto, N):
    grid = solver.make_grid(N, N, N * 0.5, N * 0.5)
    ops = solver.build_operators(grid, *bto)
    _assert_same_bytes(solver.build_ou_operators(ops, DT, L, KT),
                       _ou_from_lin_pre(solver._lin_matrix(ops), DT, L, KT))


@pytest.mark.parametrize("N", [8, 7])
def test_psd_byte_3d(bto, N):
    grid = bulk3d.make_grid_3d(N, N, N, N * 0.5, N * 0.5, N * 0.5)
    ops = bulk3d.build_operators_3d(grid, *bto)
    _assert_same_bytes(bulk3d.build_ou_operators_3d(ops, DT, L, KT),
                       _ou_from_lin_pre(solver._lin_matrix(ops), DT, L, KT))


@pytest.mark.parametrize("Nz", [8, 7])
def test_psd_byte_film(bto, Nz):
    grid = f3.make_grid_3d(8, 8, 4.0, 4.0, Nz, (Nz - 1) * 0.5)
    _assert_same_bytes(fl.build_ou_operators_film(grid, bto[0], DT, L, KT),
                       _film_ou_pre(grid, bto[0], DT, L, KT))


def test_negative_mode_grows_2d(nnbm):
    """k* = 2*pi/51.2 nm is the n = 1 mode of a 51.2 nm box; a transverse P2
    cosine there has Lin diagonal with lambda* = (G44+G44p)k^2 + H11 k^4 < 0.
    Landau: a1 kept, higher alphas zeroed, so the drift is exactly linear."""
    g, H11 = nnbm["gradient"], nnbm["H11"]
    a1 = nnbm["alpha"]["a1"]
    alpha = dict(a1=a1, a11=0.0, a12=0.0, a111=0.0, a112=0.0, a123=0.0)
    grid = solver.make_grid(32, 8, 51.2, 12.8)
    ops = grad4.build_operators_hog(grid, g["G11"], g["G12"], g["G44"], g["G44p"],
                                    nnbm["kappa_b"], H11)
    k = grid["KXr"][0, 1]
    lam_cf = (g["G44"] + g["G44p"]) * k ** 2 + H11 * k ** 4
    dt = 2.0
    ou = solver.build_ou_operators(ops, dt, L, 0.0)
    assert lam_cf < 0.0
    assert ou["lam"][0, 1, 0] == pytest.approx(lam_cf, rel=1e-12)

    P = np.zeros((3, 8, 32))
    P[1] = 1e-3 * np.cos(k * np.arange(32) * grid["hx"])
    expected = (1.0 - 2.0 * L * a1 * dt) * np.exp(-L * lam_cf * dt)

    def growth(ou_used):
        P1 = solver.step_langevin(P, alpha, ops, ou_used, dt, L, (0., 0., 0.),
                                  np.random.default_rng(0))
        return (np.fft.rfft2(P1[1])[0, 1] / np.fft.rfft2(P[1])[0, 1]).real

    assert growth(ou) == pytest.approx(expected, rel=1e-12)
    pre = growth(_ou_from_lin_pre(solver._lin_matrix(ops), dt, L, 0.0))
    assert abs(pre / expected - 1.0) > 1e-6                 # the clip threw the growth away


def test_film_negative_g11_refused(nnbm):
    grid = f3.make_grid_3d(8, 8, 16.0, 16.0, 5, 8.0)
    with pytest.raises(ValueError, match="H11"):
        fl.build_ou_operators_film(grid, nnbm["gradient"]["G11"], DT, L, KT)
    fl.build_ou_operators_film(grid, 0.0, DT, L, KT)       # G11 = 0 stays allowed


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
