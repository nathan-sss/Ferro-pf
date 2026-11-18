#!/usr/bin/env /opt/miniconda3/bin/python3
"""Portable regression for SPEC § 12 (PLANE-STRESS-2D): bulk3d.setup_elastic_3d(
..., z_mode="plane_stress") at Nz = 1 -- per-cell sigma_33 = sigma_13 =
sigma_23 = 0, in-plane problem on C11_ps = C11 - C12^2/C11, C12_ps = C12 -
C12^2/C11, C44. This file checks, in miniature:

  1. the default solve equals a test-local copy of the pre-§ 12 solve, bit for bit;
  2. sigma_i3 = 0 (exact for 13/23, round-off for 33);
  3. uniform P costs no elastic energy (stress_free);
  4. a sharp R | T laminate against the closed forms of both modes;
  5. central FD vs force with depol_N and a quenched E_ext field on;
  6. the refusals (Nz != 1, unknown z_mode, eps_star_static).

Run: /opt/miniconda3/bin/python3 -m pytest tests/test_plane_stress_2d.py -q

Scope: a targeted regression, NOT a CORE gate claim -- the core items are
K1.fd_3d_plane_stress, K2.plane_stress_off_byte, K2.plane_stress_refused,
K2.plane_stress_ensemble_byte, K3.plane_stress_sigma_i3_zero,
K3.plane_stress_uniform_zero, K3.plane_stress_laminate_closed_form in
gates/core/ (local)."""
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

from solvers import bulk3d, elastic

C = (178.0, 96.4, 122.0)          # P005 deck C (GPa-scale)
Q = (0.1, -0.034, 0.00725)        # P005 deck Q, tensor form
ALPHA = dict(a1=-0.3, a11=0.6, a12=0.15, a111=0.05, a112=0.02, a123=0.01)


def _grid(Nx=8, Ny=6, Nz=1):
    return bulk3d.make_grid_3d(Nx=Nx, Ny=Ny, Nz=Nz, Lx=0.5 * Nx, Ly=0.5 * Ny, Lz=2.8)


def _P(shape, seed=1):
    return np.random.default_rng(seed).normal(0.0, 0.2, (3,) + shape) + 0.1


def _solve_pre(P, el):
    """Test-local copy of bulk3d.elastic_solve_3d before § 12 (no spm, no static)."""
    eps_star = bulk3d.eigenstrain_3d(P, el["Q1111"], el["Q1122"], el["Q1212"])
    shape = (el["Nz"], el["Ny"], el["Nx"])
    tau = elastic.contract_C_cubic(el["C11"], el["C12"], el["C44"], eps_star)
    delta_eps = elastic._green_delta_eps(tau, el["K"], el["Omega"], el["zero_k"], shape,
                                         fwd=bulk3d._rfftn, inv=bulk3d._irfftn, pre=el["green_pre"])
    ebar = np.zeros((3, 3)) if el["k0"] == "clamped" else eps_star.mean(axis=(-3, -2, -1))
    e = ebar[..., None, None, None] + delta_eps - eps_star
    sigma = elastic.contract_C_cubic(el["C11"], el["C12"], el["C44"], e)
    return dict(eps_star=eps_star, ebar=ebar, delta_eps=delta_eps, e=e, sigma=sigma)


@pytest.mark.parametrize("Nz", [1, 4])
@pytest.mark.parametrize("k0", ["stress_free", "clamped"])
def test_default_byte_identical(Nz, k0):
    grid = _grid(Nz=Nz)
    P = _P((Nz, 6, 8))
    for kw in ({}, dict(z_mode="generalized_plane_strain")):
        el = bulk3d.setup_elastic_3d(grid, C, Q, k0=k0, **kw)
        assert "z_mode" not in el
        Om, _K, _z = bulk3d.build_Omega_3d(grid, elastic.build_C_tensor(*C))
        assert Om.tobytes() == el["Omega"].tobytes()
        new, old = bulk3d.elastic_solve_3d(P, el), _solve_pre(P, el)
        for key in old:
            assert new[key].tobytes() == old[key].tobytes(), key


@pytest.mark.parametrize("k0", ["stress_free", "clamped"])
def test_sigma_i3_zero(k0):
    grid = _grid(Nx=7, Ny=9)
    sol = bulk3d.elastic_solve_3d(_P((1, 9, 7)), bulk3d.setup_elastic_3d(
        grid, C, Q, k0=k0, z_mode="plane_stress"))
    s = sol["sigma"]
    assert np.all(s[0, 2] == 0.0) and np.all(s[1, 2] == 0.0)
    assert np.max(np.abs(s[2, 2])) <= 1e-13 * np.max(np.abs(s[:2, :2]))


def test_uniform_zero():
    el = bulk3d.setup_elastic_3d(_grid(), C, Q, z_mode="plane_stress")
    for d in ((1, 1, 1), (0, 0, 1), (0.3, -0.7, 0.2)):
        P = np.broadcast_to((0.31 * np.array(d) / np.linalg.norm(d))[:, None, None, None],
                            (3, 1, 6, 8)).copy()
        sol = bulk3d.elastic_solve_3d(P, el)
        assert np.max(bulk3d.elastic_energy_density_3d(sol)) <= 1e-30
        assert np.max(np.abs(sol["sigma"])) <= 1e-13 * C[0] * np.max(np.abs(sol["eps_star"]))


def test_laminate_closed_form():
    """Sharp 50/50 R[111] | T[100] stripes, normal x, 64 cells (no Nyquist content)."""
    p, n = 0.31, 64
    R, T = p / np.sqrt(3.0) * np.ones(3), np.array([p, 0.0, 0.0])
    P = np.where(np.arange(n)[None, :] < n // 2, R[:, None], T[:, None])[:, None, None, :] \
        * np.ones((3, 1, 3, n))
    C11, C12, C44 = C
    c11, c12 = C11 - C12**2 / C11, C12 - C12**2 / C11
    es = bulk3d.eigenstrain_3d(P, *Q)
    D = es - es.mean(axis=(-3, -2, -1))[..., None, None, None]
    gps = np.mean(0.5 * (c11 * (D[1, 1]**2 + D[2, 2]**2) + 2 * c12 * D[1, 1] * D[2, 2])
                  + 2 * C44 * D[1, 2]**2)
    ps = np.mean(0.5 * (c11 - c12**2 / c11) * D[1, 1]**2)
    grid = _grid(Nx=n, Ny=3)
    for ref, kw in ((gps, {}), (ps, dict(z_mode="plane_stress"))):
        el = bulk3d.setup_elastic_3d(grid, C, Q, **kw)
        f = bulk3d.elastic_energy_density_3d(bulk3d.elastic_solve_3d(P, el)).mean()
        assert abs(f - ref) <= 1e-10 * abs(ref)


@pytest.mark.parametrize("k0", ["stress_free", "clamped"])
def test_fd_plane_stress(k0):
    grid = _grid(Nx=7, Ny=9)
    ops = bulk3d.build_operators_3d(grid, 1.3, -0.2, 0.9, 0.55, 0.885, depol_N=0.3)
    el = bulk3d.setup_elastic_3d(grid, C, Q, k0=k0, z_mode="plane_stress")
    P, dP = _P((1, 9, 7), 2), 0.25 * _P((1, 9, 7), 3)
    E = np.random.default_rng(4).normal(0.0, 0.02, P.shape)
    dV = grid["hx"] * grid["hy"] * grid["hz"]
    F = lambda X: bulk3d.total_energy_3d(X, ALPHA, ops, E, grid, elastic_params=el)[1]["total"]
    eps = 1e-5
    fd = (F(P + eps * dP) - F(P - eps * dP)) / (2 * eps)
    an = float(np.sum(bulk3d.total_force_3d(P, ALPHA, ops, E, elastic_params=el)[-1] * dP)) * dV
    assert abs(fd - an) <= 1e-6 * max(abs(fd), abs(an))


def test_refusals():
    with pytest.raises(ValueError):
        bulk3d.setup_elastic_3d(_grid(Nz=2), C, Q, z_mode="plane_stress")
    with pytest.raises(ValueError):
        bulk3d.setup_elastic_3d(_grid(), C, Q, z_mode="plane_strain")
    static = np.zeros((3, 3, 1, 6, 8))
    with pytest.raises(ValueError):
        bulk3d.setup_elastic_3d(_grid(), C, Q, z_mode="plane_stress", eps_star_static=static)
    el = dict(bulk3d.setup_elastic_3d(_grid(), C, Q, z_mode="plane_stress"), eps_star_static=static)
    with pytest.raises(ValueError):
        bulk3d.elastic_solve_3d(_P((1, 6, 8)), el)
