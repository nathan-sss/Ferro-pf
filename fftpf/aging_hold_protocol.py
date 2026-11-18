"""Deterministic first-order polarization/defect-aging hold protocol.

One step advances P with old b, then exactly relaxes b with P held at its new
value. This coupled scheme is FIRST order. Separate geometry solvers retain
all spatial operators. No noise, SBDF2, elasticity or charge migration here.
Checkpoint v2 stores the complete fixed model and mutable P,b,step/work state.
"""
import hashlib
import json
import numpy as np
from fftpf import pf2d_engine as solver, pf3d_engine as bulk3d, film_engine as film3d, defect_bias_field as defects, state_io as io_utils


def _signature(value):
    """Deterministic signature of inputs, never mutable solver/SPM caches."""
    def pack(v):
        if isinstance(v, dict):
            return {key: pack(v[key]) for key in sorted(v)}
        if isinstance(v, (tuple, list)):
            return [pack(x) for x in v]
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            return dict(shape=a.shape, dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest())
        if isinstance(v, np.generic):
            return pack(v.item())
        return v
    return hashlib.sha256(json.dumps(pack(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


class AgingDrive:
    """Finite sequence of uniform applied-field holds, with optional static bias.

    fields has shape (number_of_holds,3); steps gives positive integer hold
    lengths. dt and tau use the same reduced engine time unit. alpha may already
    include fixed crystallinity. kappa is scalar for this initial protocol.
    Ensemble and elastic protocols are intentionally not supported yet.
    The local defects helpers can be composed with existing elastic steppers.
    Treat model as immutable after construction; create a new runner for edits.
    """
    def __init__(self, P, *, engine, grid, alpha, gradient, kappa, dt,
                 g=0., tau=np.inf, b=None, static_bias=None,
                 fields=((0., 0., 0.),), steps=(1,), L=1.):
        import copy
        if engine not in ('2d', 'bulk3d', 'film3d'):
            raise ValueError('engine must be 2d, bulk3d or film3d')
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(L) or L <= 0:
            raise ValueError('dt and L must be finite and positive')
        if np.ndim(kappa) or not np.isfinite(kappa) or kappa <= 0:
            raise ValueError('this protocol requires scalar positive kappa')
        fields = np.asarray(fields, dtype=float)
        raw_steps = np.asarray(steps)
        if (fields.ndim != 2 or fields.shape[1] != 3 or not np.all(np.isfinite(fields))
                or raw_steps.shape != (len(fields),) or len(fields) == 0
                or not np.all(np.isfinite(raw_steps)) or np.any(raw_steps < 1)
                or np.any(raw_steps != np.floor(raw_steps))):
            raise ValueError('fields and positive integer steps must describe matching holds')
        P = np.asarray(P, dtype=float)
        spatial = (grid['Ny'], grid['Nx']) if engine == '2d' else (grid['Nz'], grid['Ny'], grid['Nx'])
        if P.shape != (3,)+spatial or not np.all(np.isfinite(P)):
            raise ValueError('P must match the unbatched geometry')
        self.P = P.copy()
        self.b = defects.aging_state(self.P, g, tau, bias=b)
        if static_bias is not None:
            defects.compose_field(bias=static_bias, spatial_shape=spatial)
        # tau is always an ndarray: NPZ v2 preserves scalar +inf numerically.
        self.model = copy.deepcopy(dict(engine=engine, grid=grid, alpha=alpha,
            gradient=gradient, kappa=float(kappa), dt=float(dt), L=float(L),
            g=np.asarray(g, dtype=float), tau=np.asarray(tau, dtype=float),
            static_bias=static_bias, fields=fields, steps=raw_steps.astype(int)))
        self.signature = _signature(self.model)
        self.step_index = 0
        self.field_control_work = 0.
        self.applied = fields[0].copy()
        self._ends = np.cumsum(raw_steps.astype(int))
        grid, gradient = self.model['grid'], self.model['gradient']
        if engine == '2d':
            self.ops = solver.build_operators(grid, **gradient, kappa=kappa)
            self.inverse = solver.build_Minv(grid, self.ops, dt, L)
        elif engine == 'bulk3d':
            self.ops = bulk3d.build_operators_3d(grid, **gradient, kappa=kappa)
            self.inverse = bulk3d.build_Minv_3d(grid, self.ops, dt, L)
        else:
            self.ops = film3d.poisson_factor(grid, kappa)
            self.inverse = film3d.build_grad_factors(grid, gradient, dt, L)

    def _integral(self, density):
        grid = self.model['grid']
        if self.model['engine'] == 'film3d':
            return grid['Lx']*grid['Ly']*float(np.sum(grid['qw']*density.mean(axis=(-2,-1))))
        return grid['Volume']*float(np.mean(density))

    def _field(self):
        return defects.compose_field(self.applied, bias=self.b,
            quenched=self.model['static_bias'], spatial_shape=self.P.shape[1:])

    def energy(self):
        """Totals; engine already counts -b.P, only self-energy is added."""
        m = self.model
        field = self._field()
        if m['engine'] == '2d':
            _, total = solver.total_energy(self.P, m['alpha'], self.ops, field, m['grid'])
        elif m['engine'] == 'bulk3d':
            _, total = bulk3d.total_energy_3d(self.P, m['alpha'], self.ops, field, m['grid'])
        else:
            _, total = film3d.total_energy_3d(self.P, m['alpha'], m['grid'], m['kappa'],
                m['gradient'], poisson_factors=self.ops, E_ext=field)
        total = dict(total)
        total['aging_self'] = self._integral(defects.aging_self_energy_density(self.b, m['g']))
        total['total'] += total['aging_self']
        return total

    def advance(self, count=1):
        """Advance at most the remaining holds; return stepwise energy/work rows.

        At a hold boundary, field_control_work=-int P.dE is electric-enthalpy
        control work, NOT terminal work int E.dP or recoverable energy Wrec.
        Compare delta F against control work rather than demanding decreasing F.
        """
        if int(count) != count or count < 0 or self.step_index+count > self._ends[-1]:
            raise ValueError('count must be nonnegative and stay inside protocol')
        rows = []
        m = self.model
        for _ in range(int(count)):
            applied = m['fields'][np.searchsorted(self._ends, self.step_index, side='right')]
            delta = (applied-self.applied).reshape((3,)+(1,)*(self.P.ndim-1))
            work = -self._integral(np.sum(self.P*delta, axis=0))
            self.field_control_work += work
            self.applied = applied.copy()
            field = self._field()
            if m['engine'] == '2d':
                self.P = solver.step(self.P, m['alpha'], self.ops, self.inverse, m['dt'], m['L'], field)
            elif m['engine'] == 'bulk3d':
                self.P = bulk3d.step_3d(self.P, m['alpha'], self.ops, self.inverse, m['dt'], m['L'], field)
            else:
                self.P = film3d.step(self.P, m['alpha'], m['grid'], self.inverse,
                    self.ops, m['dt'], m['L'], E_ext=field)
            self.b = defects.aging_update(self.P, self.b, m['g'], m['tau'], m['dt'])
            self.step_index += 1
            energy = self.energy()
            rows.append(dict(step=self.step_index, time=self.step_index*m['dt'],
                mean_Px=self._integral(self.P[0])/self._integral(np.ones(self.P.shape[1:])),
                mean_bx=self._integral(self.b[0])/self._integral(np.ones(self.P.shape[1:])),
                energy=energy['total'], aging_self=energy['aging_self'],
                field_control_work_increment=work, field_control_work=self.field_control_work))
        return rows

    def save(self, path):
        m = self.model
        io_utils.save_state_npz(path, self.P, m['grid'], self.step_index,
            self.step_index*m['dt'], alpha=m['alpha'], fields={'aging_bias': self.b},
            restart=dict(kind='aging_drive_v1', scheme='sbdf1_staggered', model=m,
                signature=self.signature, step=self.step_index,
                applied=self.applied, field_control_work=self.field_control_work))

    def restore(self, path):
        """Restore only into an identically configured runner; reject drift."""
        saved = io_utils.load_state_npz(path)
        cp = saved.get('checkpoint', {})
        r = cp.get('restart') or {}
        if (r.get('kind') != 'aging_drive_v1' or r.get('scheme') != 'sbdf1_staggered'
                or r.get('signature') != self.signature
                or _signature(r['model']) != self.signature):
            raise ValueError('incompatible aging checkpoint model or scheme')
        step = r['step']
        if int(step) != step or not 0 <= step <= self._ends[-1]:
            raise ValueError('invalid checkpoint protocol position')
        P = np.asarray(saved['P'])
        if P.shape != self.P.shape or not np.all(np.isfinite(P)):
            raise ValueError('invalid checkpoint P')
        b = defects.aging_state(P, self.model['g'], self.model['tau'], bias=cp['fields']['aging_bias'])
        expected = self.model['fields'][np.searchsorted(self._ends, max(step-1,0), side='right')]
        if not np.array_equal(r['applied'], expected) or not np.isfinite(r['field_control_work']):
            raise ValueError('invalid checkpoint applied-field history')
        self.P, self.b = P.copy(), b
        self.step_index, self.applied = int(step), np.array(r['applied'], copy=True)
        self.field_control_work = float(r['field_control_work'])
        return self
