"""Execution settings and film backend lifecycle, without performance timing."""
import os
import unittest
from unittest.mock import Mock

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

from solvers import execution, solver


class ExecutionTests(unittest.TestCase):
    def tearDown(self):
        execution.configure(1)

    def test_reference_removes_old_backend(self):
        old = Mock()
        el = {"_force_solver": old}
        settings = execution.configure(1, "numpy", el)
        old.close.assert_called_once()
        self.assertNotIn("_force_solver", el)
        self.assertEqual(settings["elastic_workers"], 0)
        self.assertIsNone(solver._BULK_FORCE_IMPL)

    def test_all_fft_modules_share_budget(self):
        execution.configure(8)
        for name in execution.FFT_MODULES:
            module = execution.importlib.import_module("fftpf." + name)
            self.assertEqual(module.FFT_WORKERS, 8)

    def test_invalid_config_leaves_backend_alone(self):
        old = Mock()
        for args in ((0, "numpy"), (1, "bad"), (True, "numpy")):
            with self.assertRaises(ValueError):
                execution.configure(*args, el={"_force_solver": old})
        old.close.assert_not_called()

    def test_environment_limits_nested_threads(self):
        env = execution.environment(8)
        self.assertEqual(env["NUMBA_NUM_THREADS"], "8")
        self.assertTrue(all(env[key] == "1" for key in execution.BLAS_ENV))

    def test_numpy_switch_clears_landau_hook(self):
        solver._BULK_FORCE_IMPL = Mock()
        execution.configure(1, "numpy")
        self.assertIsNone(solver._BULK_FORCE_IMPL)


if __name__ == "__main__":
    unittest.main()
