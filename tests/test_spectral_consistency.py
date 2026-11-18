#!/usr/bin/env /opt/miniconda3/bin/python3
"""Focused regression for the even-grid Nyquist spectral-consistency repair.

The defect: solver.grad_energy_density (and bulk3d.grad_energy_density_3d)
formed f_grad from first spectral derivatives of the real FFT fields, which
drop the alternating (Nyquist) mode on every even-length axis, while
grad_force / the implicit integrator kept the k^2 diagonal symbols. Energy
and force therefore disagreed (e.g. 8x8, P1=(-1)^x gives energy 0 but force
0.98696044 for G11=0.1). The repair: (i) the mixed off-diagonal symbols
(A12/D12 in 2D; A12/A13/A23/D12/D13/D23 in 3D) are zeroed on the Nyquist
lines of the axes they are odd in -- the exact real-field-consistent
half-spectrum symbol; (ii) f_grad is the COLLOCATION local-gradient density
(the pointwise |grad P|^2 form) evaluated with real skew-adjoint first
derivatives -- each derivative axis zeroed on its own even-axis Nyquist line
or plane -- PLUS a positive even-axis Nyquist correction that adds back the
alternating mode those derivatives drop: 0.5*a_ij*(pi/h_j)^2*(NyquistProject_j
P_i)^2, a_ii = G11, a_ij = G44+G44p (i != j). Its grid mean is the energy
grad_force is the variation of; on odd grids the correction is exactly zero
and the original pointwise density is unchanged bit-for-bit. (An earlier
draft used the operator-quadratic density 0.5*P.(A_G P); it was superseded
because it changed the local energy map everywhere, odd grids included.)
The same convention covers the q (kittel_afe) and theta (octa_tilt) gradient
energies, 2D and 3D (section 10).

Run: /opt/miniconda3/bin/python3 -m pytest tests/test_spectral_consistency.py -q
or:   /opt/miniconda3/bin/python3 tests/test_spectral_consistency.py

Scope: solver (2D) + bulk3d (3D) + the kittel_afe / octa_tilt gradient
sectors. This is a targeted regression, NOT a CORE gate claim -- the core
gate items are K1.even_grid_* in gates/core/k1_consistency.py (local)."""
import os

# Thread caps MUST precede the numpy/scipy imports (ENGINE_GUIDE 1.2).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np

from solvers import solver, bulk3d, grad4, kittel_afe, octa_tilt

solver.FFT_WORKERS = 1
bulk3d.FFT_WORKERS = 1


ALPHA = dict(a1=-1.0, a11=1.0, a12=0.3, a111=0.0, a112=0.0, a123=0.0)
G = dict(G11=0.8, G12=0.4, G44=0.3, G44p=0.2)   # A_off = 0.5 (nonzero mixed)
KAPPA = 8.0
EPS = 1e-5
TOL_FD = 1e-6      # matches the gate lineage; the measured FD is ~1e-11


def _rng(seed):
    return np.random.default_rng(seed)


def _alternating_2d(Nx, Ny, mode, amp=0.7):
    """Pure alternating (Nyquist) 2D vector field. mode in {'x','y','xy'}."""
    x = np.arange(Nx)[None, :]
    y = np.arange(Ny)[:, None]
    P = np.zeros((3, Ny, Nx))
    if mode == "x":
        P[0] = amp * (-1.0) ** x
    elif mode == "y":
        P[1] = amp * (-1.0) ** y
    elif mode == "xy":
        P[0] = amp * (-1.0) ** (x + y)
        P[2] = amp * (-1.0) ** x
    else:
        raise ValueError(mode)
    return P


def _alternating_3d(Nx, Ny, Nz, mode, amp=0.7):
    x = np.arange(Nx)[None, None, :]
    y = np.arange(Ny)[None, :, None]
    z = np.arange(Nz)[:, None, None]
    P = np.zeros((3, Nz, Ny, Nx))
    if mode == "x":
        P[0] = amp * (-1.0) ** x
    elif mode == "y":
        P[1] = amp * (-1.0) ** y
    elif mode == "z":
        P[2] = amp * (-1.0) ** z
    elif mode == "xyz":
        P[0] = amp * (-1.0) ** (x + y)
        P[2] = amp * (-1.0) ** z
    else:
        raise ValueError(mode)
    return P


# --------------------------------------------------------------------------
# Independent full-spectrum (fftfreq + fft2/fftn) operator references.
# These do NOT use the engine's rfft2/rfftn half-spectrum path at all, but
# they apply the SAME declared Nyquist convention (mixed odd symbols zeroed
# on each even-axis Nyquist line/plane), so the parity checks below verify
# convention consistency between the two independent code paths, not the
# raw KX*KY symbol on the full grid.
# --------------------------------------------------------------------------
def _full_symbols_2d(grid, G11, G12, G44, G44p, kappa):
    kxf = 2 * np.pi * np.fft.fftfreq(grid["Nx"], d=grid["hx"])
    kyf = 2 * np.pi * np.fft.fftfreq(grid["Ny"], d=grid["hy"])
    KX, KY = np.meshgrid(kxf, kyf)
    A11 = G11 * KX**2 + (G44 + G44p) * KY**2
    A22 = G11 * KY**2 + (G44 + G44p) * KX**2
    A12 = (G12 + G44 - G44p) * KX * KY
    A33 = (G44 + G44p) * (KX**2 + KY**2)
    k2 = KX**2 + KY**2
    k2s = np.where(k2 == 0.0, 1.0, k2)
    D11 = np.where(k2 == 0.0, 0.0, KX * KX / k2s)
    D22 = np.where(k2 == 0.0, 0.0, KY * KY / k2s)
    D12 = np.where(k2 == 0.0, 0.0, KX * KY / k2s)
    if grid["Nx"] % 2 == 0:
        A12[:, grid["Nx"] // 2] = 0.0
        D12[:, grid["Nx"] // 2] = 0.0
    if grid["Ny"] % 2 == 0:
        A12[grid["Ny"] // 2, :] = 0.0
        D12[grid["Ny"] // 2, :] = 0.0
    return A11, A22, A12, A33, D11, D22, D12, kappa


def _full_grad_force_2d(P, syms):
    A11, A22, A12, A33, *_ = syms
    Pk = np.fft.fft2(P, axes=(-2, -1))
    F1 = A11 * Pk[0] + A12 * Pk[1]
    F2 = A12 * Pk[0] + A22 * Pk[1]
    F3 = A33 * Pk[2]
    return np.fft.ifft2(np.stack([F1, F2, F3]), axes=(-2, -1)).real


def _full_elec_force_2d(P, syms):
    _, _, _, _, D11, D22, D12, kappa = syms
    Pk = np.fft.fft2(P, axes=(-2, -1))
    F1 = (D11 * Pk[0] + D12 * Pk[1]) / kappa
    F2 = (D12 * Pk[0] + D22 * Pk[1]) / kappa
    F3 = np.zeros_like(F1)
    return np.fft.ifft2(np.stack([F1, F2, F3]), axes=(-2, -1)).real


def _full_symbols_3d(grid, G11, G12, G44, G44p, kappa):
    kxf = 2 * np.pi * np.fft.fftfreq(grid["Nx"], d=grid["hx"])
    kyf = 2 * np.pi * np.fft.fftfreq(grid["Ny"], d=grid["hy"])
    kzf = 2 * np.pi * np.fft.fftfreq(grid["Nz"], d=grid["hz"])
    KZ, KY, KX = np.meshgrid(kzf, kyf, kxf, indexing="ij")  # (Nz,Ny,Nx)
    A11 = G11 * KX**2 + (G44 + G44p) * (KY**2 + KZ**2)
    A22 = G11 * KY**2 + (G44 + G44p) * (KX**2 + KZ**2)
    A33 = G11 * KZ**2 + (G44 + G44p) * (KX**2 + KY**2)
    A_off = G12 + G44 - G44p
    A12 = A_off * KX * KY
    A13 = A_off * KX * KZ
    A23 = A_off * KY * KZ
    k2 = KX**2 + KY**2 + KZ**2
    k2s = np.where(k2 == 0.0, 1.0, k2)
    D11 = np.where(k2 == 0.0, 0.0, KX * KX / k2s)
    D22 = np.where(k2 == 0.0, 0.0, KY * KY / k2s)
    D33 = np.where(k2 == 0.0, 0.0, KZ * KZ / k2s)
    D12 = np.where(k2 == 0.0, 0.0, KX * KY / k2s)
    D13 = np.where(k2 == 0.0, 0.0, KX * KZ / k2s)
    D23 = np.where(k2 == 0.0, 0.0, KY * KZ / k2s)
    if grid["Nx"] % 2 == 0:
        A12[:, :, grid["Nx"] // 2] = 0.0
        A13[:, :, grid["Nx"] // 2] = 0.0
        D12[:, :, grid["Nx"] // 2] = 0.0
        D13[:, :, grid["Nx"] // 2] = 0.0
    if grid["Ny"] % 2 == 0:
        A12[:, grid["Ny"] // 2, :] = 0.0
        A23[:, grid["Ny"] // 2, :] = 0.0
        D12[:, grid["Ny"] // 2, :] = 0.0
        D23[:, grid["Ny"] // 2, :] = 0.0
    if grid["Nz"] % 2 == 0:
        A13[grid["Nz"] // 2, :, :] = 0.0
        A23[grid["Nz"] // 2, :, :] = 0.0
        D13[grid["Nz"] // 2, :, :] = 0.0
        D23[grid["Nz"] // 2, :, :] = 0.0
    return (A11, A22, A33, A12, A13, A23,
            D11, D22, D33, D12, D13, D23, kappa)


def _full_grad_force_3d(P, syms):
    A11, A22, A33, A12, A13, A23, *_ = syms
    Pk = np.fft.fftn(P, axes=(-3, -2, -1))
    F1 = A11 * Pk[0] + A12 * Pk[1] + A13 * Pk[2]
    F2 = A12 * Pk[0] + A22 * Pk[1] + A23 * Pk[2]
    F3 = A13 * Pk[0] + A23 * Pk[1] + A33 * Pk[2]
    return np.fft.ifftn(np.stack([F1, F2, F3]), axes=(-3, -2, -1)).real


def _full_elec_force_3d(P, syms):
    _, _, _, _, _, _, D11, D22, D33, D12, D13, D23, kappa = syms
    Pk = np.fft.fftn(P, axes=(-3, -2, -1))
    F1 = (D11 * Pk[0] + D12 * Pk[1] + D13 * Pk[2]) / kappa
    F2 = (D12 * Pk[0] + D22 * Pk[1] + D23 * Pk[2]) / kappa
    F3 = (D13 * Pk[0] + D23 * Pk[1] + D33 * Pk[2]) / kappa
    return np.fft.ifftn(np.stack([F1, F2, F3]), axes=(-3, -2, -1)).real


def _fd_rel(F_of, force, P, dP, dV):
    fd = (F_of(P + EPS * dP) - F_of(P - EPS * dP)) / (2 * EPS)
    an = float(np.sum(force * dP)) * dV
    denom = max(abs(an), abs(fd), 1e-300)
    return abs(fd - an) / denom


# --------------------------------------------------------------------------
# 1. Mixed gradient/electrostatic symbols: half-spectrum == full-spectrum.
# --------------------------------------------------------------------------
def test_mixed_symbols_2d_parities():
    for Nx, Ny in [(8, 8), (8, 7), (7, 8), (7, 7), (6, 9), (10, 4)]:
        grid = solver.make_grid(Nx, Ny, Lx=Nx, Ly=Ny)
        ops = solver.build_operators(grid, G["G11"], G["G12"], G["G44"],
                                     G["G44p"], KAPPA)
        P = _rng(Nx * 100 + Ny).standard_normal((3, Ny, Nx))
        syms = _full_symbols_2d(grid, G["G11"], G["G12"], G["G44"], G["G44p"],
                                KAPPA)
        np.testing.assert_allclose(solver.grad_force(P, ops),
                                   _full_grad_force_2d(P, syms),
                                   rtol=1e-11, atol=1e-11,
                                   err_msg=f"grad_force parity {Nx}x{Ny}")
        np.testing.assert_allclose(solver.elec_force(P, ops, (0., 0., 0.)),
                                   _full_elec_force_2d(P, syms),
                                   rtol=1e-11, atol=1e-11,
                                   err_msg=f"elec_force parity {Nx}x{Ny}")


def test_mixed_symbols_3d_parities():
    for Nx, Ny, Nz in [(4, 4, 4), (4, 3, 4), (3, 4, 4), (5, 4, 3), (4, 5, 6)]:
        grid = bulk3d.make_grid_3d(Nx, Ny, Nz, Lx=Nx, Ly=Ny, Lz=Nz)
        ops = bulk3d.build_operators_3d(grid, G["G11"], G["G12"], G["G44"],
                                        G["G44p"], KAPPA)
        P = _rng(700 + Nx * 100 + Ny * 10 + Nz).standard_normal((3, Nz, Ny, Nx))
        syms = _full_symbols_3d(grid, G["G11"], G["G12"], G["G44"], G["G44p"],
                                KAPPA)
        np.testing.assert_allclose(bulk3d.grad_force_3d(P, ops),
                                   _full_grad_force_3d(P, syms),
                                   rtol=1e-11, atol=1e-11,
                                   err_msg=f"grad_force_3d parity {Nx}x{Ny}x{Nz}")
        np.testing.assert_allclose(bulk3d.elec_force_3d(P, ops, (0., 0., 0.)),
                                   _full_elec_force_3d(P, syms),
                                   rtol=1e-11, atol=1e-11,
                                   err_msg=f"elec_force_3d parity {Nx}x{Ny}x{Nz}")


# --------------------------------------------------------------------------
# 2. Energy/force variational agreement (grad term), unfiltered random fields.
# --------------------------------------------------------------------------
def test_grad_energy_force_2d_random():
    for Nx, Ny in [(8, 8), (8, 7), (7, 8), (7, 7), (6, 9)]:
        grid = solver.make_grid(Nx, Ny, Lx=Nx, Ly=Ny)
        ops = solver.build_operators(grid, G["G11"], G["G12"], G["G44"],
                                     G["G44p"], KAPPA)
        P = _rng(41 + Nx * Ny).standard_normal((3, Ny, Nx))
        dP = _rng(42 + Nx * Ny).standard_normal((3, Ny, Nx))
        dV = grid["hx"] * grid["hy"]
        F = lambda Pv: float(np.sum(solver.grad_energy_density(Pv, grid, ops))) * dV
        rel = _fd_rel(F, solver.grad_force(P, ops), P, dP, dV)
        assert rel < TOL_FD, f"grad FD {Nx}x{Ny}: relerr={rel:.3e}"


def test_grad_energy_force_3d_random():
    for Nx, Ny, Nz in [(4, 4, 4), (4, 3, 4), (3, 4, 4), (5, 4, 3)]:
        grid = bulk3d.make_grid_3d(Nx, Ny, Nz, Lx=Nx, Ly=Ny, Lz=Nz)
        ops = bulk3d.build_operators_3d(grid, G["G11"], G["G12"], G["G44"],
                                        G["G44p"], KAPPA)
        P = _rng(51 + Nx * 100 + Ny * 10 + Nz).standard_normal((3, Nz, Ny, Nx))
        dP = _rng(52 + Nx * 100 + Ny * 10 + Nz).standard_normal((3, Nz, Ny, Nx))
        dV = grid["hx"] * grid["hy"] * grid["hz"]
        F = lambda Pv: float(np.sum(bulk3d.grad_energy_density_3d(Pv, grid, ops))) * dV
        rel = _fd_rel(F, bulk3d.grad_force_3d(P, ops), P, dP, dV)
        assert rel < TOL_FD, f"grad FD {Nx}x{Ny}x{Nz}: relerr={rel:.3e}"


# --------------------------------------------------------------------------
# 3. Pure alternating (Nyquist) modes: energy must agree with the force now.
# --------------------------------------------------------------------------
def test_grad_energy_force_2d_alternating():
    Nx, Ny = 8, 8
    grid = solver.make_grid(Nx, Ny, Lx=Nx, Ly=Ny)
    ops = solver.build_operators(grid, G["G11"], G["G12"], G["G44"], G["G44p"],
                                 KAPPA)
    dV = grid["hx"] * grid["hy"]
    for mode in ("x", "y", "xy"):
        P = _alternating_2d(Nx, Ny, mode)
        dP = _rng(60).standard_normal((3, Ny, Nx))
        F = lambda Pv: float(np.sum(solver.grad_energy_density(Pv, grid, ops))) * dV
        rel = _fd_rel(F, solver.grad_force(P, ops), P, dP, dV)
        assert rel < TOL_FD, f"alternating {mode}: relerr={rel:.3e}"
        # the defect was exactly this: energy must NOT be identically zero
        assert abs(float(solver.grad_energy_density(P, grid, ops).mean())) > 1e-8


def test_grad_energy_force_3d_alternating():
    Nx, Ny, Nz = 4, 4, 4
    grid = bulk3d.make_grid_3d(Nx, Ny, Nz, Lx=Nx, Ly=Ny, Lz=Nz)
    ops = bulk3d.build_operators_3d(grid, G["G11"], G["G12"], G["G44"],
                                    G["G44p"], KAPPA)
    dV = grid["hx"] * grid["hy"] * grid["hz"]
    for mode in ("x", "y", "z", "xyz"):
        P = _alternating_3d(Nx, Ny, Nz, mode)
        dP = _rng(61).standard_normal((3, Nz, Ny, Nx))
        F = lambda Pv: float(np.sum(bulk3d.grad_energy_density_3d(Pv, grid, ops))) * dV
        rel = _fd_rel(F, bulk3d.grad_force_3d(P, ops), P, dP, dV)
        assert rel < TOL_FD, f"alternating 3d {mode}: relerr={rel:.3e}"
        assert abs(float(bulk3d.grad_energy_density_3d(P, grid, ops).mean())) > 1e-8


# --------------------------------------------------------------------------
# 4. Confirmed-defect reproduction numbers (8x8, dx=1, G11=.1, G44=G44p=.05).
# --------------------------------------------------------------------------
def test_confirmed_defect_reproduction():
    Nx = Ny = 8
    grid = solver.make_grid(Nx, Ny, Lx=Nx, Ly=Ny)
    ops = solver.build_operators(grid, 0.1, 0.0, 0.05, 0.05, KAPPA)
    P = np.zeros((3, Ny, Nx))
    P[0] = (-1.0) ** np.arange(Nx)[None, :]
    force_max = float(np.max(np.abs(solver.grad_force(P, ops))))
    energy = float(solver.grad_energy_density(P, grid, ops).mean())
    assert abs(force_max - 0.9869604401089358) < 1e-8
    assert abs(energy - 0.4934802200544679) < 1e-8


# --------------------------------------------------------------------------
# 5. One-step semi-implicit vs an independent dense real-space solve.
# --------------------------------------------------------------------------
def _dense_step_2d(P, alpha, ops, grid, dt, L):
    syms = _full_symbols_2d(grid, ops["G11"], ops["G12"], ops["G44"],
                            ops["G44p"], ops["kappa"])
    A11, A22, A12, A33, D11, D22, D12, kappa = syms
    M11 = 1.0 + dt * L * (A11 + D11 / kappa)
    M22 = 1.0 + dt * L * (A22 + D22 / kappa)
    M33 = 1.0 + dt * L * A33
    off = dt * L * (A12 + D12 / kappa)
    Eext = np.zeros((3, 1, 1))
    N = solver.bulk_force(P, alpha) - Eext
    rhs = P - dt * L * N

    Ny, Nx = grid["Ny"], grid["Nx"]
    n = 3 * Nx * Ny

    def apply_M(vflat):
        v = vflat.reshape(3, Ny, Nx)
        vk = np.fft.fft2(v, axes=(-2, -1))
        w1 = M11 * vk[0] + off * vk[1]
        w2 = off * vk[0] + M22 * vk[1]
        w3 = M33 * vk[2]
        return np.fft.ifft2(np.stack([w1, w2, w3]), axes=(-2, -1)).real.ravel()

    Mreal = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        Mreal[:, j] = apply_M(e)
    return np.linalg.solve(Mreal, rhs.ravel()).reshape(3, Ny, Nx)


def _dense_step_3d(P, alpha, ops, grid, dt, L):
    syms = _full_symbols_3d(grid, ops["G11"], ops["G12"], ops["G44"],
                            ops["G44p"], ops["kappa"])
    A11, A22, A33, A12, A13, A23, D11, D22, D33, D12, D13, D23, kappa = syms
    M11 = 1.0 + dt * L * (A11 + D11 / kappa)
    M22 = 1.0 + dt * L * (A22 + D22 / kappa)
    M33 = 1.0 + dt * L * (A33 + D33 / kappa)
    off12 = dt * L * (A12 + D12 / kappa)
    off13 = dt * L * (A13 + D13 / kappa)
    off23 = dt * L * (A23 + D23 / kappa)
    Eext = np.zeros((3, 1, 1, 1))
    N = bulk3d.bulk_force(P, alpha) - Eext
    rhs = P - dt * L * N

    Nz, Ny, Nx = grid["Nz"], grid["Ny"], grid["Nx"]
    n = 3 * Nx * Ny * Nz

    def apply_M(vflat):
        v = vflat.reshape(3, Nz, Ny, Nx)
        vk = np.fft.fftn(v, axes=(-3, -2, -1))
        w1 = M11 * vk[0] + off12 * vk[1] + off13 * vk[2]
        w2 = off12 * vk[0] + M22 * vk[1] + off23 * vk[2]
        w3 = off13 * vk[0] + off23 * vk[1] + M33 * vk[2]
        return np.fft.ifftn(np.stack([w1, w2, w3]), axes=(-3, -2, -1)).real.ravel()

    Mreal = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        Mreal[:, j] = apply_M(e)
    return np.linalg.solve(Mreal, rhs.ravel()).reshape(3, Nz, Ny, Nx)


def test_step_vs_dense_2d():
    Nx, Ny, dt, L = 4, 4, 0.01, 1.0
    grid = solver.make_grid(Nx, Ny, Lx=Nx, Ly=Ny)
    ops = solver.build_operators(grid, G["G11"], G["G12"], G["G44"],
                                 G["G44p"], KAPPA)
    P = _rng(71).standard_normal((3, Ny, Nx))
    Minv = solver.build_Minv(grid, ops, dt, L)
    stepped = solver.step(P, ALPHA, ops, Minv, dt, L, (0., 0., 0.))
    dense = _dense_step_2d(P, ALPHA, ops, grid, dt, L)
    np.testing.assert_allclose(stepped, dense, rtol=1e-10, atol=1e-10)


def test_step_vs_dense_3d():
    Nx, Ny, Nz, dt, L = 4, 4, 4, 0.01, 1.0
    grid = bulk3d.make_grid_3d(Nx, Ny, Nz, Lx=Nx, Ly=Ny, Lz=Nz)
    ops = bulk3d.build_operators_3d(grid, G["G11"], G["G12"], G["G44"],
                                    G["G44p"], KAPPA)
    P = _rng(72).standard_normal((3, Nz, Ny, Nx))
    Minv = bulk3d.build_Minv_3d(grid, ops, dt, L)
    stepped = bulk3d.step_3d(P, ALPHA, ops, Minv, dt, L, (0., 0., 0.))
    dense = _dense_step_3d(P, ALPHA, ops, grid, dt, L)
    np.testing.assert_allclose(stepped, dense, rtol=1e-10, atol=1e-10)


# --------------------------------------------------------------------------
# 6. Odd-grid and smooth-mode regressions (the pre-repair passing cases).
# --------------------------------------------------------------------------
def test_smooth_mode_2d():
    grid = solver.make_grid(32, 32, Lx=32.0, Ly=32.0)
    ops = solver.build_operators(grid, G["G11"], G["G12"], G["G44"],
                                 G["G44p"], KAPPA)
    P = solver.random_smooth_field(grid, 3, amplitude=0.15, kmax_frac=0.25,
                                   seed=42)
    dP = solver.random_smooth_field(grid, 3, amplitude=0.15, kmax_frac=0.25,
                                    seed=1337)
    dV = grid["hx"] * grid["hy"]
    F = lambda Pv: float(np.sum(solver.grad_energy_density(Pv, grid, ops))) * dV
    rel = _fd_rel(F, solver.grad_force(P, ops), P, dP, dV)
    assert rel < TOL_FD, f"smooth 2D relerr={rel:.3e}"


def test_odd_grid_3d():
    grid = bulk3d.make_grid_3d(7, 7, 7, Lx=7.0, Ly=7.0, Lz=7.0)
    ops = bulk3d.build_operators_3d(grid, G["G11"], G["G12"], G["G44"],
                                    G["G44p"], KAPPA)
    P = _rng(81).standard_normal((3, 7, 7, 7))
    dP = _rng(82).standard_normal((3, 7, 7, 7))
    dV = grid["hx"] * grid["hy"] * grid["hz"]
    F = lambda Pv: float(np.sum(bulk3d.grad_energy_density_3d(Pv, grid, ops))) * dV
    rel = _fd_rel(F, bulk3d.grad_force_3d(P, ops), P, dP, dV)
    assert rel < TOL_FD, f"odd 3D relerr={rel:.3e}"


# --------------------------------------------------------------------------
# 7. 2D vs Nz=1 3D consistency.
# --------------------------------------------------------------------------
def test_2d_vs_3d_nz1():
    Nx, Ny = 6, 4
    grid2 = solver.make_grid(Nx, Ny, Lx=Nx, Ly=Ny)
    ops2 = solver.build_operators(grid2, G["G11"], G["G12"], G["G44"],
                                  G["G44p"], KAPPA)
    grid3 = bulk3d.make_grid_3d(Nx, Ny, 1, Lx=Nx, Ly=Ny, Lz=1.0)
    ops3 = bulk3d.build_operators_3d(grid3, G["G11"], G["G12"], G["G44"],
                                     G["G44p"], KAPPA)
    P2 = _rng(91).standard_normal((3, Ny, Nx))
    P3 = P2[:, None, :, :]
    np.testing.assert_allclose(bulk3d.grad_force_3d(P3, ops3)[:, 0, :, :],
                               solver.grad_force(P2, ops2), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        bulk3d.grad_energy_density_3d(P3, grid3, ops3)[0, :, :],
        solver.grad_energy_density(P2, grid2, ops2), rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# 8. H11 (higher-order gradient): no double counting, force/energy agreement.
# --------------------------------------------------------------------------
def test_h11_no_double_count_2d():
    grid = solver.make_grid(8, 8, Lx=8.0, Ly=8.0)
    H11 = 0.05
    ops_hog = grad4.build_operators_hog(grid, G["G11"], G["G12"], G["G44"],
                                        G["G44p"], KAPPA, H11)
    ops_base = solver.build_operators(grid, G["G11"], G["G12"], G["G44"],
                                      G["G44p"], KAPPA)
    P = _rng(101).standard_normal((3, 8, 8))
    # the G term must not change when ops carries the H11-augmented A arrays
    np.testing.assert_allclose(solver.grad_energy_density(P, grid, ops_hog),
                               solver.grad_energy_density(P, grid, ops_base),
                               rtol=1e-12, atol=1e-12)
    # total hog energy = G + H11; its FD derivative is grad_force(ops_hog)
    dP = _rng(102).standard_normal((3, 8, 8))
    dV = grid["hx"] * grid["hy"]
    F = lambda Pv: float(np.sum(grad4.grad_energy_density_hog(Pv, grid, ops_hog))) * dV
    rel = _fd_rel(F, solver.grad_force(P, ops_hog), P, dP, dV)
    assert rel < TOL_FD, f"hog 2D relerr={rel:.3e}"


def test_h11_no_double_count_3d():
    grid = bulk3d.make_grid_3d(4, 4, 4, Lx=4.0, Ly=4.0, Lz=4.0)
    H11 = 0.05
    ops_hog = grad4.build_operators_hog_3d(grid, G["G11"], G["G12"], G["G44"],
                                           G["G44p"], KAPPA, H11)
    ops_base = bulk3d.build_operators_3d(grid, G["G11"], G["G12"], G["G44"],
                                         G["G44p"], KAPPA)
    P = _rng(111).standard_normal((3, 4, 4, 4))
    np.testing.assert_allclose(bulk3d.grad_energy_density_3d(P, grid, ops_hog),
                               bulk3d.grad_energy_density_3d(P, grid, ops_base),
                               rtol=1e-12, atol=1e-12)
    dP = _rng(112).standard_normal((3, 4, 4, 4))
    dV = grid["hx"] * grid["hy"] * grid["hz"]
    F = lambda Pv: float(np.sum(grad4.grad_energy_density_hog_3d(Pv, grid, ops_hog))) * dV
    rel = _fd_rel(F, bulk3d.grad_force_3d(P, ops_hog), P, dP, dV)
    assert rel < TOL_FD, f"hog 3D relerr={rel:.3e}"


# --------------------------------------------------------------------------
# 9. Local-density compatibility: pointwise odd-grid equivalence, integrated
#    even correction, pure-alternating analytic energy, ensemble broadcasting.
# --------------------------------------------------------------------------
def _old_pointwise_2d(P, grid, ops):
    """Independent reconstruction of the ORIGINAL pointwise |grad P|^2 local
    gradient density using the raw (unmodified) grid wavevectors -- what
    grad_energy_density computed before the even-grid repair."""
    G11, G12, G44, G44p = ops["G11"], ops["G12"], ops["G44"], ops["G44p"]
    P1x, P1y = solver._spectral_grad_xy(P[0], grid)
    P2x, P2y = solver._spectral_grad_xy(P[1], grid)
    P3x, P3y = solver._spectral_grad_xy(P[2], grid)
    return (0.5 * G11 * (P1x**2 + P2y**2) + G12 * P1x * P2y
            + 0.5 * (G44 + G44p) * (P3x**2 + P3y**2)
            + 0.5 * G44 * (P1y + P2x)**2 + 0.5 * G44p * (P1y - P2x)**2)


def _old_pointwise_3d(P, grid, ops):
    G11, G12, G44, G44p = ops["G11"], ops["G12"], ops["G44"], ops["G44p"]
    P1x, P1y, P1z = bulk3d._spectral_grad_xyz(P[0], grid)
    P2x, P2y, P2z = bulk3d._spectral_grad_xyz(P[1], grid)
    P3x, P3y, P3z = bulk3d._spectral_grad_xyz(P[2], grid)
    diag = 0.5 * G11 * (P1x**2 + P2y**2 + P3z**2)
    cross = G12 * (P1x * P2y + P2y * P3z + P1x * P3z)
    sym = 0.5 * G44 * ((P1y + P2x)**2 + (P2z + P3y)**2 + (P1z + P3x)**2)
    antisym = 0.5 * G44p * ((P1y - P2x)**2 + (P2z - P3y)**2 + (P1z - P3x)**2)
    return diag + cross + sym + antisym


def test_pointwise_odd_grid_old_density_2d():
    for Nx, Ny in [(7, 7), (7, 5), (5, 9), (9, 7)]:
        grid = solver.make_grid(Nx, Ny, Lx=Nx, Ly=Ny)
        ops = solver.build_operators(grid, G["G11"], G["G12"], G["G44"],
                                     G["G44p"], KAPPA)
        P = _rng(120 + Nx * Ny).standard_normal((3, Ny, Nx))
        np.testing.assert_allclose(solver.grad_energy_density(P, grid, ops),
                                   _old_pointwise_2d(P, grid, ops),
                                   rtol=1e-12, atol=1e-12,
                                   err_msg=f"odd 2D {Nx}x{Ny} pointwise")


def test_pointwise_odd_grid_old_density_3d():
    for Nx, Ny, Nz in [(7, 7, 7), (5, 7, 9), (7, 5, 3)]:
        grid = bulk3d.make_grid_3d(Nx, Ny, Nz, Lx=Nx, Ly=Ny, Lz=Nz)
        ops = bulk3d.build_operators_3d(grid, G["G11"], G["G12"], G["G44"],
                                        G["G44p"], KAPPA)
        P = _rng(130 + Nx * 100 + Ny * 10 + Nz).standard_normal((3, Nz, Ny, Nx))
        np.testing.assert_allclose(bulk3d.grad_energy_density_3d(P, grid, ops),
                                   _old_pointwise_3d(P, grid, ops),
                                   rtol=1e-12, atol=1e-12,
                                   err_msg=f"odd 3D {Nx}x{Ny}x{Nz} pointwise")


def test_even_correction_integrated_2d():
    for Nx, Ny in [(8, 8), (8, 7), (7, 8), (6, 9)]:
        grid = solver.make_grid(Nx, Ny, Lx=Nx, Ly=Ny)
        ops = solver.build_operators(grid, G["G11"], G["G12"], G["G44"],
                                     G["G44p"], KAPPA)
        P = _rng(140 + Nx * Ny).standard_normal((3, Ny, Nx))
        dV = grid["hx"] * grid["hy"]
        new = solver.grad_energy_density(P, grid, ops)
        ref = 0.5 * float(np.sum(P * solver.grad_force(P, ops))) * dV
        got = float(np.sum(new)) * dV
        assert abs(got - ref) <= 1e-10 * max(abs(ref), 1e-300), \
            f"2D integrated even correction {Nx}x{Ny}: {got} vs {ref}"


def test_even_correction_integrated_3d():
    for Nx, Ny, Nz in [(4, 4, 4), (4, 3, 4), (3, 4, 4), (4, 5, 6)]:
        grid = bulk3d.make_grid_3d(Nx, Ny, Nz, Lx=Nx, Ly=Ny, Lz=Nz)
        ops = bulk3d.build_operators_3d(grid, G["G11"], G["G12"], G["G44"],
                                        G["G44p"], KAPPA)
        P = _rng(150 + Nx * 100 + Ny * 10 + Nz).standard_normal((3, Nz, Ny, Nx))
        dV = grid["hx"] * grid["hy"] * grid["hz"]
        new = bulk3d.grad_energy_density_3d(P, grid, ops)
        ref = 0.5 * float(np.sum(P * bulk3d.grad_force_3d(P, ops))) * dV
        got = float(np.sum(new)) * dV
        assert abs(got - ref) <= 1e-10 * max(abs(ref), 1e-300), \
            f"3D integrated even correction {Nx}x{Ny}x{Nz}: {got} vs {ref}"


def test_alternating_analytic_energy_2d():
    Nx, Ny, amp = 8, 6, 0.7
    grid = solver.make_grid(Nx, Ny, Lx=Nx, Ly=Ny)
    ops = solver.build_operators(grid, G["G11"], G["G12"], G["G44"],
                                 G["G44p"], KAPPA)
    G11, G44, G44p = G["G11"], G["G44"], G["G44p"]
    # P1 = amp*(-1)^x: only the x axis is even-active, a_11 = G11.
    P = np.zeros((3, Ny, Nx))
    P[0] = amp * (-1.0) ** np.arange(Nx)[None, :]
    e = float(solver.grad_energy_density(P, grid, ops).mean())
    np.testing.assert_allclose(e, 0.5 * G11 * (np.pi / grid["hx"])**2 * amp**2,
                               rtol=1e-12, atol=1e-12)
    # P1 = amp*(-1)^(x+y): both axes active, a_11=G11 (x), a_12=G44+G44p (y).
    x = np.arange(Nx)[None, :]
    y = np.arange(Ny)[:, None]
    P = np.zeros((3, Ny, Nx))
    P[0] = amp * (-1.0) ** (x + y)
    e = float(solver.grad_energy_density(P, grid, ops).mean())
    expected = 0.5 * (G11 * (np.pi / grid["hx"])**2
                      + (G44 + G44p) * (np.pi / grid["hy"])**2) * amp**2
    np.testing.assert_allclose(e, expected, rtol=1e-12, atol=1e-12)


def test_alternating_analytic_energy_3d():
    Nx, Ny, Nz, amp = 4, 6, 4, 0.7
    grid = bulk3d.make_grid_3d(Nx, Ny, Nz, Lx=Nx, Ly=Ny, Lz=Nz)
    ops = bulk3d.build_operators_3d(grid, G["G11"], G["G12"], G["G44"],
                                    G["G44p"], KAPPA)
    G11, G44, G44p = G["G11"], G["G44"], G["G44p"]
    # P1 = amp*(-1)^x
    P = np.zeros((3, Nz, Ny, Nx))
    P[0] = amp * (-1.0) ** np.arange(Nx)[None, None, :]
    e = float(bulk3d.grad_energy_density_3d(P, grid, ops).mean())
    np.testing.assert_allclose(e, 0.5 * G11 * (np.pi / grid["hx"])**2 * amp**2,
                               rtol=1e-12, atol=1e-12)
    # P1 = amp*(-1)^(x+y)
    x = np.arange(Nx)[None, None, :]
    y = np.arange(Ny)[None, :, None]
    P = np.zeros((3, Nz, Ny, Nx))
    P[0] = amp * (-1.0) ** (x + y)
    e = float(bulk3d.grad_energy_density_3d(P, grid, ops).mean())
    expected = 0.5 * (G11 * (np.pi / grid["hx"])**2
                      + (G44 + G44p) * (np.pi / grid["hy"])**2) * amp**2
    np.testing.assert_allclose(e, expected, rtol=1e-12, atol=1e-12)


def test_ensemble_broadcast_2d():
    Nx, Ny, S = 6, 8, 5
    grid = solver.make_grid(Nx, Ny, Lx=Nx, Ly=Ny)
    ops = solver.build_operators(grid, G["G11"], G["G12"], G["G44"],
                                 G["G44p"], KAPPA)
    P = _rng(160).standard_normal((3, S, Ny, Nx))
    fg = solver.grad_energy_density(P, grid, ops)
    assert fg.shape == (S, Ny, Nx)
    for s in range(S):
        np.testing.assert_allclose(fg[s],
                                   solver.grad_energy_density(P[:, s], grid, ops),
                                   rtol=1e-12, atol=1e-12)


def test_ensemble_broadcast_3d():
    Nx, Ny, Nz, S = 4, 4, 4, 3
    grid = bulk3d.make_grid_3d(Nx, Ny, Nz, Lx=Nx, Ly=Ny, Lz=Nz)
    ops = bulk3d.build_operators_3d(grid, G["G11"], G["G12"], G["G44"],
                                    G["G44p"], KAPPA)
    P = _rng(170).standard_normal((3, S, Nz, Ny, Nx))
    fg = bulk3d.grad_energy_density_3d(P, grid, ops)
    assert fg.shape == (S, Nz, Ny, Nx)
    for s in range(S):
        np.testing.assert_allclose(fg[s],
                                   bulk3d.grad_energy_density_3d(P[:, s], grid, ops),
                                   rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# 10. The q (kittel_afe) and theta (octa_tilt) gradient sectors: same
#     convention, each sector with its own coefficient (isotropic, a_ij = G).
# --------------------------------------------------------------------------
GQ, GT = 0.37, 0.61      # not 1.0: a correction that dropped its G would pass at 1


def _sectors_2d(grid):
    return (("q", lambda f: kittel_afe.grad_energy_density_q(f, grid, GQ),
             kittel_afe.grad_force_q, kittel_afe.build_ops_q(grid, GQ), GQ),
            ("theta", lambda f: octa_tilt.grad_energy_density_theta(f, grid, GT),
             octa_tilt.grad_force_theta, octa_tilt.build_ops_theta(grid, GT), GT))


def _sectors_3d(grid):
    return (("q", lambda f: kittel_afe.grad_energy_density_q_3d(f, grid, GQ),
             kittel_afe.grad_force_q_3d, kittel_afe.build_ops_q_3d(grid, GQ), GQ),
            ("theta", lambda f: octa_tilt.grad_energy_density_theta_3d(f, grid, GT),
             octa_tilt.grad_force_theta_3d, octa_tilt.build_ops_theta_3d(grid, GT), GT))


def _raw_isotropic_2d(f, grid, Gc):
    """The pre-repair isotropic density, raw grid wavevectors, same expression order."""
    d = [solver._spectral_grad_xy(f[i], grid) for i in range(3)]
    return 0.5 * Gc * (d[0][0]**2 + d[0][1]**2 + d[1][0]**2 + d[1][1]**2
                       + d[2][0]**2 + d[2][1]**2)


def _raw_isotropic_3d(f, grid, Gc):
    d = [bulk3d._spectral_grad_xyz(f[i], grid) for i in range(3)]
    return 0.5 * Gc * (d[0][0]**2 + d[0][1]**2 + d[0][2]**2
                       + d[1][0]**2 + d[1][1]**2 + d[1][2]**2
                       + d[2][0]**2 + d[2][1]**2 + d[2][2]**2)


def test_q_theta_odd_grid_bit_identical():
    for Nx, Ny in [(7, 7), (7, 5), (9, 1)]:
        grid = solver.make_grid(Nx, Ny, Lx=Nx, Ly=Ny)
        f = _rng(210 + Nx * Ny).standard_normal((3, Ny, Nx))
        for name, dens, _force, _ops, Gc in _sectors_2d(grid):
            assert np.array_equal(dens(f), _raw_isotropic_2d(f, grid, Gc)), f"{name} 2D {Nx}x{Ny}"
    for Nx, Ny, Nz in [(5, 7, 9), (7, 5, 3), (7, 9, 1)]:
        grid = bulk3d.make_grid_3d(Nx, Ny, Nz, Lx=Nx, Ly=Ny, Lz=Nz)
        f = _rng(220 + Nx * Ny * Nz).standard_normal((3, Nz, Ny, Nx))
        for name, dens, _force, _ops, Gc in _sectors_3d(grid):
            assert np.array_equal(dens(f), _raw_isotropic_3d(f, grid, Gc)), \
                f"{name} 3D {Nx}x{Ny}x{Nz}"


def test_q_theta_energy_force_random():
    for Nx, Ny in [(8, 8), (8, 7), (7, 8), (6, 9)]:
        grid = solver.make_grid(Nx, Ny, Lx=0.5 * Nx, Ly=0.6 * Ny)
        f = _rng(230 + Nx * Ny).standard_normal((3, Ny, Nx))
        df = _rng(231 + Nx * Ny).standard_normal((3, Ny, Nx))
        dV = grid["hx"] * grid["hy"]
        for name, dens, force, ops, _Gc in _sectors_2d(grid):
            F = lambda v: float(np.sum(dens(v))) * dV
            rel = _fd_rel(F, force(f, ops), f, df, dV)
            assert rel < TOL_FD, f"{name} 2D FD {Nx}x{Ny}: relerr={rel:.3e}"
            got, ref = F(f), 0.5 * float(np.sum(f * force(f, ops))) * dV
            assert abs(got - ref) <= 1e-12 * abs(ref), f"{name} 2D quadratic {Nx}x{Ny}"
    for Nx, Ny, Nz in [(4, 6, 4), (4, 3, 4), (5, 4, 3), (8, 6, 1)]:
        grid = bulk3d.make_grid_3d(Nx, Ny, Nz, Lx=0.5 * Nx, Ly=0.6 * Ny, Lz=0.7 * Nz)
        f = _rng(240 + Nx * Ny * Nz).standard_normal((3, Nz, Ny, Nx))
        df = _rng(241 + Nx * Ny * Nz).standard_normal((3, Nz, Ny, Nx))
        dV = grid["hx"] * grid["hy"] * grid["hz"]
        for name, dens, force, ops, _Gc in _sectors_3d(grid):
            F = lambda v: float(np.sum(dens(v))) * dV
            rel = _fd_rel(F, force(f, ops), f, df, dV)
            assert rel < TOL_FD, f"{name} 3D FD {Nx}x{Ny}x{Nz}: relerr={rel:.3e}"
            got, ref = F(f), 0.5 * float(np.sum(f * force(f, ops))) * dV
            assert abs(got - ref) <= 1e-12 * abs(ref), f"{name} 3D quadratic {Nx}x{Ny}x{Nz}"


def test_q_theta_confirmed_defect_and_analytic():
    # The recorded defect: 8x8, dx=1, coefficient 0.1, component 1 = (-1)^x:
    # energy was 0.0 while max|force| = 0.1*pi^2 = 0.98696044.
    n = 8
    grid = solver.make_grid(n, n, Lx=n, Ly=n)
    g3 = bulk3d.make_grid_3d(n, n, n, Lx=n, Ly=n, Lz=n)
    f2 = np.zeros((3, n, n))
    f2[0] = (-1.0) ** np.arange(n)[None, :]
    f3 = np.zeros((3, n, n, n))
    f3[0] = (-1.0) ** np.arange(n)[None, None, :]
    for e, fmax in ((kittel_afe.grad_energy_density_q(f2, grid, 0.1),
                     kittel_afe.grad_force_q(f2, kittel_afe.build_ops_q(grid, 0.1))),
                    (octa_tilt.grad_energy_density_theta(f2, grid, 0.1),
                     octa_tilt.grad_force_theta(f2, octa_tilt.build_ops_theta(grid, 0.1))),
                    (kittel_afe.grad_energy_density_q_3d(f3, g3, 0.1),
                     kittel_afe.grad_force_q_3d(f3, kittel_afe.build_ops_q_3d(g3, 0.1))),
                    (octa_tilt.grad_energy_density_theta_3d(f3, g3, 0.1),
                     octa_tilt.grad_force_theta_3d(f3, octa_tilt.build_ops_theta_3d(g3, 0.1)))):
        assert abs(float(np.max(np.abs(fmax))) - 0.9869604401089358) < 1e-8
        assert abs(float(e.mean()) - 0.4934802200544679) < 1e-8
    # Two active axes, non-square cells: 0.5*G*[(pi/hx)^2 + (pi/hy)^2]*amp^2.
    Nx, Ny, amp = 8, 6, 0.7
    grid = solver.make_grid(Nx, Ny, Lx=0.5 * Nx, Ly=0.6 * Ny)
    f = np.zeros((3, Ny, Nx))
    f[2] = amp * (-1.0) ** (np.arange(Nx)[None, :] + np.arange(Ny)[:, None])
    for name, dens, _force, _ops, Gc in _sectors_2d(grid):
        expected = 0.5 * Gc * ((np.pi / grid["hx"])**2 + (np.pi / grid["hy"])**2) * amp**2
        np.testing.assert_allclose(float(dens(f).mean()), expected, rtol=1e-12, atol=0,
                                   err_msg=f"{name} two-axis alternating")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    raise SystemExit(1 if failed else 0)
