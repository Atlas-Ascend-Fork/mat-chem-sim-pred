# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import importlib.util
import unittest
from pathlib import Path
import numpy as np


def _import_mamba2_ssd_ref():
    module_file = Path(__file__).resolve().parents[1] / "reference" / "reference.py"
    module_spec = importlib.util.spec_from_file_location(
        "mamba2_ssd_ref_mod", module_file
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Unable to load Mamba2 reference module from {module_file}")
    loaded = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(loaded)
    return loaded.reference


SSD_REFERENCE_FN = _import_mamba2_ssd_ref()


class Mamba2SsdChunkScanReferenceTest(unittest.TestCase):
    def test_zero_input(self):
        """Zero input tensor u=0 yields exact zero output."""
        dims = (2, 32, 2, 8, 8)
        u_arr = np.zeros(dims[:4], dtype=np.float32)
        dt_arr = np.full(dims[:3], 0.1, dtype=np.float32)
        a_arr = -np.ones((dims[2],), dtype=np.float32)
        b_arr = np.random.randn(dims[0], dims[1], dims[2], dims[4]).astype(np.float32)
        c_arr = np.random.randn(dims[0], dims[1], dims[2], dims[4]).astype(np.float32)
        d_arr = np.ones((dims[2], dims[3]), dtype=np.float32)

        actual = SSD_REFERENCE_FN(u_arr, dt_arr, a_arr, b_arr, c_arr, d_arr)
        np.testing.assert_allclose(actual, np.zeros_like(u_arr), atol=1e-6)

    def test_basic_computation(self):
        """Valid shape and dtype propagation on non-zero state space scan."""
        dims = (1, 16, 2, 4, 4)
        u_arr = np.random.randn(*dims[:4]).astype(np.float32)
        dt_arr = np.full(dims[:3], 0.05, dtype=np.float32)
        a_arr = -np.ones((dims[2],), dtype=np.float32)
        b_arr = np.random.randn(dims[0], dims[1], dims[2], dims[4]).astype(np.float32)
        c_arr = np.random.randn(dims[0], dims[1], dims[2], dims[4]).astype(np.float32)

        actual = SSD_REFERENCE_FN(u_arr, dt_arr, a_arr, b_arr, c_arr)
        self.assertEqual(actual.shape, dims[:4])
        self.assertEqual(actual.dtype, np.float32)

    def test_invalid_shape_detection(self):
        """Mismatched sequence length raises ValueError."""
        u_arr = np.ones((2, 16, 2, 4), dtype=np.float32)
        dt_arr = np.ones((2, 8, 2), dtype=np.float32)
        a_arr = -np.ones((2,), dtype=np.float32)
        b_arr = np.ones((2, 16, 2, 4), dtype=np.float32)
        c_arr = np.ones((2, 16, 2, 4), dtype=np.float32)

        with self.assertRaises(ValueError):
            SSD_REFERENCE_FN(u_arr, dt_arr, a_arr, b_arr, c_arr)


def test_mamba2_ssd_reference_smoke():
    """Universal pytest runner discovery smoke test."""
    dims = (1, 4, 1, 4, 4)
    u_arr = np.zeros(dims[:4], dtype=np.float32)
    dt_arr = np.full(dims[:3], 0.1, dtype=np.float32)
    a_arr = -np.ones((dims[2],), dtype=np.float32)
    b_arr = np.zeros((dims[0], dims[1], dims[2], dims[4]), dtype=np.float32)
    c_arr = np.zeros((dims[0], dims[1], dims[2], dims[4]), dtype=np.float32)
    out = SSD_REFERENCE_FN(u_arr, dt_arr, a_arr, b_arr, c_arr)
    assert out.shape == dims[:4]


if __name__ == "__main__":
    unittest.main()
