"""Optional CPU elastic arithmetic, compared with the unchanged FFT operator."""
import unittest
import os

os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import numpy as np

from solvers import bulk3d, cpu_kernels, elastic, solver


@unittest.skipUnless(cpu_kernels.numba is not None, 'Optional Numba dependency is unavailable')
class ElasticKernelTests(unittest.TestCase):
    def setUp(self):
        self.previous = elastic._CPU_KERNELS
        self.fft_workers = elastic.FFT_WORKERS, bulk3d.FFT_WORKERS
        elastic.FFT_WORKERS = bulk3d.FFT_WORKERS = 1
        self.threads = cpu_kernels.numba.get_num_threads()
        cpu_kernels.numba.set_num_threads(1)
        elastic._CPU_KERNELS = None
        self.rng = np.random.default_rng(802)

    def tearDown(self):
        elastic._CPU_KERNELS = self.previous
        elastic.FFT_WORKERS, bulk3d.FFT_WORKERS = self.fft_workers
        cpu_kernels.numba.set_num_threads(self.threads)

    def compare(self, function, *args):
        elastic._CPU_KERNELS = None
        expected = function(*args)
        elastic._CPU_KERNELS = cpu_kernels
        actual = function(*args)
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
        self.assertTrue(actual.flags.c_contiguous and actual.flags.owndata)
        saved = actual.copy()
        function(*args)
        np.testing.assert_array_equal(actual, saved)
        elastic._CPU_KERNELS = None
        return actual

    def test_pointwise_coefficients_layouts_and_q_conventions(self):
        for shape in ((3, 5, 8), (3, 2, 5, 8), (3, 2, 3, 5, 8)):
            P = self.rng.normal(size=shape)
            stress = self.rng.normal(size=(3, 3) + shape[1:])
            stress = .5*(stress + stress.swapaxes(0, 1))
            for field in (P, np.asfortranarray(P), P[..., ::-1]):
                before = field.copy()
                for Q in ((.11, -.04, .08), (np.arange(8)*.01, np.ones((5, 1))*.03, .08)):
                    e = self.compare(elastic.eigenstrain, field, *Q)
                    self.compare(elastic.contract_C_cubic, 200., 70., 60., e)
                    self.compare(elastic.contract_C_cubic, np.arange(8)+190., 70., 60., e)
                    f = self.compare(elastic.elastic_force_explicit, field, stress, *Q)
                    q3 = (Q[0], Q[1], Q[2]/4)
                    np.testing.assert_array_equal(self.compare(bulk3d.eigenstrain_3d, field, *q3), e)
                    np.testing.assert_array_equal(self.compare(bulk3d.elastic_force_explicit_3d,
                                                             field, stress, *q3), f)
                np.testing.assert_array_equal(field, before)

    def test_green_operator_even_odd_and_ensemble(self):
        for shape in ((5, 7), (6, 8), (3, 5, 7), (4, 6, 8)):
            if len(shape) == 2:
                ny, nx = shape
                grid = solver.make_grid(nx, ny, float(nx), float(ny))
                el = elastic.setup_elastic(grid, 200., 70., 60., .11, -.04, .08)
                fwd, inv = elastic._rfft2, elastic._irfft2
            else:
                nz, ny, nx = shape
                grid = bulk3d.make_grid_3d(nx, ny, nz, float(nx), float(ny), float(nz))
                el = bulk3d.setup_elastic_3d(grid, (200., 70., 60.), (.11, -.04, .02))
                fwd, inv = bulk3d._rfftn, bulk3d._irfftn
            for ensemble in ((), (2,)):
                stress = self.rng.normal(size=(3, 3) + ensemble + shape)
                stress = .5*(stress + stress.swapaxes(0, 1))
                original = stress.copy()
                with self.subTest(shape=shape, ensemble=ensemble):
                    result = self.compare(elastic._green_delta_eps, stress, el['K'], el['Omega'],
                                          el['zero_k'], shape, fwd, inv, el['green_pre'])
                    np.testing.assert_array_equal(stress, original)
                    spectrum = fwd(result)
                    np.testing.assert_allclose(spectrum[..., el['zero_k']], 0., atol=3e-13)

    def test_homogeneous_and_broadcast_stress(self):
        P = self.rng.normal(size=(3, 5, 8))
        for shape in ((3, 3), (3, 3, 1, 8)):
            stress = self.rng.normal(size=shape)
            stress = .5*(stress + stress.swapaxes(0, 1))
            self.compare(elastic.elastic_force_explicit, P, stress, .11, -.04, .08)

    def test_invalid_component_shapes_fail_before_compiled_loop(self):
        bad = np.zeros((4, 3, 8))
        with self.assertRaises(ValueError):
            cpu_kernels.eigenstrain(bad, .11, -.04, .08)
        with self.assertRaises(ValueError):
            cpu_kernels.elastic_force(bad, np.zeros((3, 3, 3, 8)), .11, -.04, .08)
        with self.assertRaises(ValueError):
            cpu_kernels.contract_cubic(200., 70., 60., np.zeros((4, 3, 8)))
        with self.assertRaises(ValueError):
            cpu_kernels.elastic_force(np.zeros((3, 3, 8)), np.zeros((4, 3, 3, 8)), .11, -.04, .08)


if __name__ == '__main__':
    unittest.main()
