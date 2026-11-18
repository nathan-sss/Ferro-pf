"""Small deterministic SBDF2 problems for the multicore benchmark."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solvers import bulk3d, casekit, elastic, film3d, film_elastic
from solvers import execution, io_utils, materials, solver, units


class Fixture:
    def __init__(self, engine, n, dt=0.005):
        self.engine, self.n, self.dt = engine, n, dt
        self.seed = 20261017
        name = 'bto_lcc2005_paper' if engine == 'film3d' else 'bto_lcc2005_8th'
        kwargs = {'T': 298.15}
        if engine == 'film3d':
            kwargs['dx_nm'] = 0.4
        self.deck = materials.get(name, **kwargs)
        self.alpha = self.deck['landau']
        c, q = self.deck['elastic'], self.deck['electrostriction']
        if engine == '2d':
            self.grid, self.ops = casekit.build_2d(self.deck, n, n, 0.4)
            self.el = elastic.setup_elastic(self.grid, c['C11'], c['C12'], c['C44'],
                q['Q11_phys'], q['Q12_phys'], q['Q44_phys'])
            self.P = solver.random_smooth_field(self.grid, 3, .005, .5, self.seed)
            self.first = solver.build_Minv(self.grid, self.ops, dt, 1.)
            self.second = solver.build_Minv2(self.grid, self.ops, dt, 1.)
        elif engine == 'bulk3d':
            self.grid, self.ops, self.el = casekit.build_3d(self.deck, n, .4,
                elastic=True, k0='stress_free')
            self.P = bulk3d.random_smooth_field_3d(self.grid, 3, .005, .5, self.seed)
            self.first = bulk3d.build_Minv_3d(self.grid, self.ops, dt, 1.)
            self.second = bulk3d.build_Minv2_3d(self.grid, self.ops, dt, 1.)
        else:
            self.grid = film3d.make_grid_3d(n, n, n*.1, n*.1, 21, 8.)
            self.kappa = units.relative_to_deck_permittivity(10.)
            stack = film_elastic.make_stack_grid(self.grid, hs_dx=12)
            self.el = film_elastic.setup_film_elastic(stack, c['C11'], c['C12'], c['C44'],
                q['Q11_phys'], q['Q12_phys'], q['Q44_phys'], e0=-.01,
                C_substrate=materials.get('ss316_rt', T=298.15)['elastic'])
            self.P = np.stack([solver.random_smooth_field(self.grid, 3, .005, .5,
                self.seed+z) for z in range(21)], axis=1)
            G = self.deck['gradient']['G11']
            self.first = film3d.build_grad_factors(self.grid, G, dt, 1.)
            self.second = film3d.build_grad_factors_sbdf2(self.grid, G, dt, 1.)
            self.poisson = film3d.poisson_factor(self.grid, self.kappa)
        self.previous = None

    def advance(self, steps):
        for _ in range(steps):
            p, old, dt = self.P, self.previous, self.dt
            if self.engine == '2d':
                if old is None:
                    new = solver.step(p, self.alpha, self.ops, self.first, dt, 1.,
                        (0., 0., 0.), elastic_params=self.el)
                else:
                    new = solver.step_sbdf2(p, old, self.alpha, self.ops, self.second,
                        dt, 1., (0., 0., 0.), elastic_params=self.el)
            elif self.engine == 'bulk3d':
                if old is None:
                    new = bulk3d.step_3d(p, self.alpha, self.ops, self.first, dt, 1.,
                        (0., 0., 0.), elastic_params=self.el)
                else:
                    new = bulk3d.step_sbdf2_3d(p, old, self.alpha, self.ops, self.second,
                        dt, 1., (0., 0., 0.), elastic_params=self.el)
            elif old is None:
                new = film3d.step(p, self.alpha, self.grid, self.first, self.poisson,
                    dt, 1., el=self.el)
            else:
                new = film3d.step_sbdf2(p, old, self.alpha, self.grid, self.second,
                    self.poisson, dt, 1., el=self.el)
            self.previous, self.P = p, new

    def snapshot(self):
        return self.P.copy(), self.previous.copy()

    def restore(self, state):
        self.P, self.previous = (a.copy() for a in state)

    def energy(self):
        if self.engine == 'film3d':
            density, _ = film3d.total_energy_3d(self.P, self.alpha, self.grid,
                self.kappa, self.deck['gradient']['G11'],
                poisson_factors=self.poisson, el=self.el)
        else:
            fn = solver.total_energy if self.engine == '2d' else bulk3d.total_energy_3d
            density, _ = fn(self.P, self.alpha, self.ops, (0., 0., 0.), self.grid,
                elastic_params=self.el)
        return {key: float(value) for key, value in density.items()}

    def save(self, path, step):
        grid = {key: value for key, value in self.grid.items() if np.isscalar(value)}
        io_utils.save_state_npz(path, self.P, grid, step, step*self.dt,
            restart={'scheme': 'sbdf2', 'dt': self.dt},
            fields={'P_previous': self.previous})

    def load(self, path):
        state = io_utils.load_state_npz(path)
        self.P = state['P']
        self.previous = state['checkpoint']['fields']['P_previous']

    def close(self):
        execution.configure(workers=1, backend='numpy',
            el=self.el if self.engine == 'film3d' else None)

    def description(self):
        return dict(material=self.deck, temperature_K=298.15, scheme='sbdf2',
            dt=self.dt, mobility=1., field=[0., 0., 0.], seed=self.seed,
            grid_shape=list(self.P.shape[1:]),
            spacing=[self.grid['hx'], self.grid['hy'], self.grid.get('hz')],
            units={'length': 'nm', 'polarization': 'C/m2', 'energy_density': '1e9 J/m3',
                   'time': 'uncalibrated deck time'},
            initial='Band-limited random P, amplitude 0.005 C/m2; four reference warm steps.',
            boundaries=('Periodic x/y; natural polarization z surfaces; short circuit; '
                'SS316 substrate 4.8 nm, clamped bottom/free top, biaxial strain -1%'
                if self.engine == 'film3d' else 'Periodic; stress-free elastic mean mode'),
            limitations='Deterministic homogeneous BTO benchmark, not material calibration or convergence evidence.')
