"""Check the optional fused Landau force against independent derivatives.

Run with a Numba-enabled Python: python -m unittest tests.test_cpu_kernels
"""
import unittest
import subprocess
import sys
import os

os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import numpy as np
try:
    from numba import get_num_threads, set_num_threads
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from solvers import bulk3d, solver
from solvers.cpu_kernels import landau_force


@unittest.skipUnless(HAS_NUMBA, 'Optional Numba dependency is unavailable')
class LandauKernelTests(unittest.TestCase):
    def setUp(self):
        self.previous_impl = solver._BULK_FORCE_IMPL
        self.previous_threads = get_num_threads()
        solver._BULK_FORCE_IMPL = None
        set_num_threads(1)
        self.rng = np.random.default_rng(731)

    def tearDown(self):
        solver._BULK_FORCE_IMPL = self.previous_impl
        set_num_threads(self.previous_threads)

    def alpha(self, eighth):
        keys = solver.ALPHA_KEYS + (solver.ALPHA_KEYS_8TH if eighth else ())
        return {key: float(self.rng.uniform(-1, 1)) for key in keys}

    def check_force(self, P, alpha):
        before = P.copy()
        reference = solver.bulk_force(P, alpha)
        args = ([P[0], P[1], P[2]] + [alpha[k] for k in solver.ALPHA_KEYS]
                + [alpha.get(k, 0.) for k in solver.ALPHA_KEYS_8TH])
        symbolic = np.stack([fn(*args) for fn in solver._df_bulk_fn_sympy])
        got = landau_force(P, alpha)
        np.testing.assert_allclose(got, reference, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(got, symbolic, rtol=2e-13, atol=2e-13)
        np.testing.assert_array_equal(P, before)
        self.assertEqual(got.dtype, np.float64)
        self.assertTrue(got.flags.c_contiguous and got.flags.owndata)
        self.assertFalse(np.shares_memory(got, P))
        saved = got.copy()
        landau_force(P * .5, alpha)
        np.testing.assert_array_equal(got, saved)

    def test_scalar_coefficients_and_dimensions(self):
        for shape in ((3, 7, 8), (3, 5, 7, 8), (3, 2, 5, 7, 8)):
            for eighth in (False, True):
                with self.subTest(shape=shape, eighth=eighth):
                    self.check_force(self.rng.uniform(-1, 1, shape), self.alpha(eighth))

    def test_spatial_broadcast_coefficients(self):
        for eighth in (False, True):
            P = self.rng.uniform(-1, 1, (3, 5, 7, 8))
            alpha = self.alpha(eighth)
            shapes = ((), (8,), (7, 1), (5, 1, 1), (5, 7, 8), (1,))
            for i, key in enumerate(alpha):
                alpha[key] = self.rng.uniform(-1, 1, shapes[i % len(shapes)])
            self.check_force(P, alpha)
            alpha['a1'] = np.broadcast_to(np.array(.4), (5, 7, 8))
            self.check_force(P, alpha)

    def test_layouts(self):
        P = self.rng.uniform(-1, 1, (3, 5, 7, 8))
        for field in (P, np.asfortranarray(P), P[:, :, :, ::-1], P.transpose(0, 3, 1, 2)):
            with self.subTest(strides=field.strides):
                self.check_force(field, self.alpha(True))

    def test_zero_eighth_order_arrays(self):
        P = self.rng.uniform(-1, 1, (3, 5, 8))
        alpha = self.alpha(False)
        alpha.update({key: np.zeros((1, 8)) for key in solver.ALPHA_KEYS_8TH})
        self.check_force(P, alpha)

    def test_dynamic_coefficients_and_alias_dispatch(self):
        P = self.rng.uniform(-1, 1, (3, 5, 8))
        alpha = self.alpha(True)
        alpha['a1'] = np.ones((5, 8))
        before = landau_force(P, alpha)
        alpha['a1'][:] = 2.
        reference = solver.bulk_force(P, alpha)
        self.assertFalse(np.array_equal(before, reference))
        alias = bulk3d.bulk_force
        solver._BULK_FORCE_IMPL = landau_force
        self.assertIs(alias, solver.bulk_force)
        np.testing.assert_allclose(alias(P, alpha), reference, rtol=2e-13, atol=2e-13)
        solver._BULK_FORCE_IMPL = None
        np.testing.assert_array_equal(alias(P, alpha), reference)

    def test_unsupported_dtype_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'float64'):
            landau_force(np.zeros((3, 5, 8), dtype=np.float32), self.alpha(False))


class OptionalDependencyTests(unittest.TestCase):
    def test_import_without_numba_and_explicit_call_failure(self):
        code = '''
import sys
sys.modules['numba'] = None
from solvers import cpu_kernels
assert cpu_kernels.numba is None
try:
    cpu_kernels.landau_force(None, {})
except ImportError as exc:
    assert 'requires numba and llvmlite' in str(exc)
else:
    raise AssertionError('Missing Numba must not silently run Python kernels')
'''
        subprocess.run([sys.executable, '-c', code], check=True,
                       capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
