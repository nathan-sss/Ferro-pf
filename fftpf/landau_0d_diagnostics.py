"""Homogeneous Landau diagnostics in deck units (P in C/m^2).

Independent polynomial expansion supplies the energy, analytic gradient and
Hessian for comparison with the production Horner force. T/O/R labels refer
to polarization directions only, not to octahedral-tilt space groups.
"""
from itertools import permutations
import numpy as np

POWERS = dict(a1=(2,0,0), a11=(4,0,0), a12=(2,2,0), a111=(6,0,0),
              a112=(4,2,0), a123=(2,2,2), a1111=(8,0,0), a1112=(6,2,0),
              a1122=(4,4,0), a1123=(4,2,2))
DIRECTIONS = dict(T=np.array([1.,0.,0.]), O=np.array([1.,1.,0.])/np.sqrt(2),
                  R=np.ones(3)/np.sqrt(3))


def evaluate(P, alpha, E=(0.,0.,0.)):
    """Return energy density, dF/dP, Hessian for one homogeneous P vector."""
    P, E = np.asarray(P, dtype=float), np.asarray(E, dtype=float)
    f, grad, hessian = -float(E@P), -E.copy(), np.zeros((3,3))
    for key, coefficient in alpha.items():
        for powers in sorted(set(permutations(POWERS[key]))):
            ex = np.array(powers)
            f += coefficient*np.prod(P**ex)
            for i in range(3):
                if ex[i] == 0:
                    continue
                ei = ex.copy(); ei[i] -= 1
                grad[i] += coefficient*ex[i]*np.prod(P**ei)
                for j in range(3):
                    if ei[j] == 0:
                        continue
                    eij = ei.copy(); eij[j] -= 1
                    hessian[i,j] += coefficient*ex[i]*ei[j]*np.prod(P**eij)
    return float(f), grad, hessian


def radial_coefficients(alpha, direction):
    """Coefficients b_n for F(r*direction)=sum b_n*(r^2)^n."""
    n = np.asarray(direction, dtype=float)
    n = n/np.linalg.norm(n)
    b = np.zeros(5)
    for key, value in alpha.items():
        powers = POWERS[key]
        b[sum(powers)//2] += value*sum(np.prod(n**np.array(ex))
                                     for ex in set(permutations(powers)))
    return b


def symmetry_stationary_points(alpha):
    """All real nonnegative radial extrema on C/T/O/R branches, with full Hessians.

    Includes unstable branches. This is not a global-minimum proof for arbitrary
    eighth-order potentials, which can have lower-symmetry minima.
    """
    points = [("C", np.zeros(3))]
    for label, direction in DIRECTIONS.items():
        b = radial_coefficients(alpha, direction)
        for q in np.polynomial.polynomial.polyroots(np.arange(1,5)*b[1:]):
            if abs(q.imag) < 1e-10 and q.real > 0:
                points.append((label, np.sqrt(q.real)*direction))
    rows = []
    for label, P in points:
        f, force, H = evaluate(P, alpha)
        eig = np.linalg.eigvalsh(H)
        stable = bool(eig.min() > 1e-10)
        # inverse curvature is dP/dE in deck units; multiply 1e-9 for SI.
        response = np.linalg.inv(H)*1e-9 if stable else None
        rows.append(dict(branch=label, P=P.tolist(), magnitude=float(np.linalg.norm(P)),
            energy_deck=f, energy_J_m3=f*1e9, gradient_max=float(max(abs(force))),
            hessian_eigenvalues=eig.tolist(), stable=stable,
            susceptibility_SI=None if response is None else response.tolist()))
    return sorted(rows, key=lambda row: row['energy_deck'])


def local_response(alpha, P0, E, *, tolerance=1e-12, max_iterations=40):
    """Newton solve in a stable homogeneous basin for a small applied deck field.

    No switching/loop claim: fail if an iterate exits a positive-curvature basin.
    """
    P = np.asarray(P0,dtype=float).copy()
    for _ in range(max_iterations):
        _, gradient, H = evaluate(P,alpha,E)
        if np.linalg.eigvalsh(H).min() <= 0:
            raise ValueError('local response left the stable basin')
        if np.max(abs(gradient)) < tolerance:
            return P
        P -= np.linalg.solve(H,gradient)
    raise RuntimeError('local response did not converge')
