"""Prescribed defect bias in deck field units (1e9 V/m).

The bias contributes -b.P to energy through the engine's E_ext argument.
The energy helper is a diagnostic: do not add it again to engine totals.
These fields are phenomenological prescribed inputs, not mobile charges.
"""
import numpy as np


def _field(value, spatial_shape=None):
    value = np.asarray(value, dtype=float)
    if value.ndim < 1 or value.shape[0] != 3 or not np.all(np.isfinite(value)):
        raise ValueError("field must be finite and component-first with three components")
    if spatial_shape is not None:
        target = (3,) + tuple(spatial_shape)
        if value.shape == (3,):
            value = value.reshape((3,) + (1,) * len(spatial_shape))
        if value.ndim != len(target):
            raise ValueError("spatial field must have one axis per spatial dimension")
        try:
            np.broadcast_to(value, target)
        except ValueError as exc:
            raise ValueError("field cannot broadcast to the specified spatial shape") from exc
    return value


def compose_field(applied=(0., 0., 0.), *, quenched=None, bias=None,
                  spatial_shape=None):
    """Compose applied, quenched and defect fields for E_ext without double counting.

    Each input is a (3,) vector or component-first spatial field. Spatial
    inputs require spatial_shape explicitly, preventing component-axis
    ambiguity. A missing or exactly zero additive field performs no sum,
    preserving the applied field's numerical values and signed zeros.
    """
    values = [applied] + [x for x in (quenched, bias) if x is not None]
    if spatial_shape is None and any(np.asarray(x).shape != (3,) for x in values):
        raise ValueError("spatial_shape is required for spatial fields")
    checked = [_field(x, spatial_shape) for x in values]
    result = np.asarray(applied, dtype=float)
    for add in checked[1:]:
        if np.any(add != 0):
            result = _field(result, spatial_shape) + add
    return result


def bias_energy_density(P, bias):
    """Return -b.P in deck energy-density units (1e9 J/m3); diagnostic only."""
    P = _field(P)
    b = _field(bias, P.shape[1:])
    return -np.sum(P * b, axis=0)


def bias_force(P, bias):
    """Return energy derivative -b, broadcast to P's component-first shape."""
    P = _field(P)
    return -np.broadcast_to(_field(bias, P.shape[1:]), P.shape)


def _aging_parameter(value, shape, name, allow_infinite=False):
    value = np.asarray(value, dtype=float)
    if value.ndim not in (0, len(shape)):
        raise ValueError(name + " must be scalar or a spatial field")
    try:
        value = np.broadcast_to(value, shape)
    except ValueError as exc:
        raise ValueError(name + " cannot broadcast to the spatial shape") from exc
    if allow_infinite:
        valid = (value > 0) & ~np.isnan(value)
    else:
        valid = np.isfinite(value) & (value >= 0)
    if not np.all(valid):
        raise ValueError(name + " is outside its physical domain")
    return value


def _aging_inputs(P, bias, g, tau):
    P = _field(P)
    if P.ndim < 2:
        raise ValueError("P must include spatial axes")
    b = np.broadcast_to(_field(bias, P.shape[1:]), P.shape)
    coupling = _aging_parameter(g, P.shape[1:], "g")
    relaxation = _aging_parameter(tau, P.shape[1:], "tau", allow_infinite=True)
    if np.any(b[:, coupling == 0] != 0):
        raise ValueError("bias must be zero wherever g is zero")
    return P, b, coupling, relaxation


def aging_state(P, g, tau, bias=None):
    """Initialize an independent bias array, validating g>=0 and tau>0.

    g has field/polarization units; tau is reduced engine time, not seconds.
    Positive infinity freezes an active site's existing bias. g=0 disables
    that site's aging channel and requires exactly zero bias there.
    """
    if bias is None:
        bias = np.zeros_like(np.asarray(P, dtype=float))
    return _aging_inputs(P, bias, g, tau)[1].copy()


def aging_self_energy_density(bias, g):
    """Return |b|^2/(2g), zero on inactive sites, in deck energy density.

    Add only this self energy to engine totals when b already enters E_ext.
    The engine already counts -b.P. g=0 requires b=0, not a division limit.
    """
    b = _field(bias)
    if b.ndim < 2:
        raise ValueError("bias must include spatial axes")
    coupling = _aging_parameter(g, b.shape[1:], "g")
    if np.any(b[:, coupling == 0] != 0):
        raise ValueError("bias must be zero wherever g is zero")
    out = np.zeros(b.shape[1:], dtype=float)
    np.divide(np.sum(b*b, axis=0), 2*coupling, out=out, where=coupling > 0)
    return out


def aging_energy_density(P, bias, g, tau=np.inf):
    """Full local aging energy for diagnostics: self energy minus b.P."""
    P, b, coupling, _ = _aging_inputs(P, bias, g, tau)
    return aging_self_energy_density(b, coupling) - np.sum(b*P, axis=0)


def aging_force_bias(P, bias, g, tau=np.inf):
    """Derivative with respect to b on g>0: b/g-P.

    g=0 sites have constrained b=0, so return zero on their inactive channel.
    Derivative with respect to P is bias_force(P,bias), namely -b.
    """
    P, b, coupling, _ = _aging_inputs(P, bias, g, tau)
    out = np.zeros_like(P)
    np.divide(b, coupling, out=out, where=coupling > 0)
    return np.where(coupling > 0, out-P, 0.)


def aging_update(P, bias, g, tau, dt):
    """Exact db/dt=(gP-b)/tau update for polarization held fixed over dt.

    A coupled polarization/aging integrator using this update after a P step
    is first order. No defect noise or conserved defect transport is modeled.
    """
    P, b, coupling, relaxation = _aging_inputs(P, bias, g, tau)
    dt = float(dt)
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative")
    if dt == 0:
        return b.copy()
    weight = -np.expm1(-dt/relaxation)
    result = b.copy()
    active = (coupling > 0) & np.isfinite(relaxation)
    # No arithmetic on frozen/inactive entries preserves their exact bytes.
    result[:, active] += weight[active]*(coupling[active]*P[:, active]-b[:, active])
    return result
