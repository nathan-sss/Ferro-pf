"""Prescribed crystallinity: f_X = A*(1-Xc)*|P|²/2.

Xc is a fixed structural-quality descriptor, not an evolving phase fraction.
A is a nonnegative scalar in the same deck units as a1. Its value needs
calibration; no material-independent default coupling is implied. The helper
folds this energy into a1, so energy_density is diagnostic, NOT an additional
term to add to the solver total. Component axis is 0; remaining axes can
include ensemble axes. Pass shape=P.shape[1:] to validate coefficient fields.
"""
import numpy as np


def _strength(Xc, A, shape=None):
    if Xc is None:
        return 0.0
    amplitude = np.asarray(A, dtype=float)
    if amplitude.ndim or not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("A must be a finite nonnegative scalar")
    fraction = np.asarray(Xc, dtype=float)
    if not np.all(np.isfinite(fraction)) or np.any((fraction < 0) | (fraction > 1)):
        raise ValueError("Xc must be finite and lie in [0, 1]")
    if shape is not None:
        np.broadcast_to(fraction, tuple(shape))  # rejects shape expansion
    return float(amplitude) * (1.0 - fraction)


def effective_alpha(alpha, Xc=None, A=0.0, *, shape=None):
    """Return a shallow copy with only a1 changed; never mutate input arrays.

    Xc=None disables the feature. Xc=1 and A=0 retain the original a1 object
    and perform no floating-point addition, preserving exact legacy results.
    Other coefficient values (including optional eighth-order keys) are kept.
    With shape omitted, alpha['a1'] and Xc must broadcast together; supplying
    the target noncomponent shape also prevents accidental shape expansion.
    """
    strength = _strength(Xc, A, shape)
    result = dict(alpha)
    if shape is not None:
        np.broadcast_to(np.asarray(alpha['a1']), tuple(shape))
    if np.any(strength):
        result['a1'] = np.asarray(alpha['a1']) + 0.5 * strength
    return result


def energy_density(P, Xc=None, A=0.0):
    """Diagnostic f_X on each site, with the component axis reduced."""
    P = np.asarray(P)
    if P.ndim < 2 or P.shape[0] != 3 or not np.all(np.isfinite(P)):
        raise ValueError("P must be finite with shape (3, ...)")
    return 0.5 * _strength(Xc, A, P.shape[1:]) * np.sum(P * P, axis=0)


def force(P, Xc=None, A=0.0):
    """Positive energy derivative d f_X/dP; TDGL applies its negative."""
    P = np.asarray(P)
    if P.ndim < 2 or P.shape[0] != 3 or not np.all(np.isfinite(P)):
        raise ValueError("P must be finite with shape (3, ...)")
    return _strength(Xc, A, P.shape[1:]) * P
