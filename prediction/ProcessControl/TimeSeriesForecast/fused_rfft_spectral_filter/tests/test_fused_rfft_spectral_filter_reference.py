# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import unittest
from pathlib import Path
import numpy as np


def _load_reference():
    path = Path(__file__).resolve().parents[1] / "reference" / "reference.py"
    spec = importlib.util.spec_from_file_location("fused_rfft_spectral_filter_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference


reference = _load_reference()


class FusedRfftSpectralFilterReferenceTest(unittest.TestCase):
    @staticmethod
    def _create_tensors(batch_size, seq_len, d_model):
        x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        weights = np.zeros((seq_len // 2 + 1, d_model, 2), dtype=np.float32)
        return x, weights

    def test_identity_filtering(self):
        """Identity test: weight with real=1, imag=0 should reconstruct original x"""
        x, weights = self._create_tensors(2, 16, 4)
        weights[..., 0] = 1.0
        out = reference(x, weights)
        np.testing.assert_allclose(out, x, rtol=1e-5, atol=1e-5)

    def test_zero_filtering(self):
        """Zero test: all zero weights should output all zeros"""
        x, weights = self._create_tensors(2, 32, 8)
        out = reference(x, weights)
        np.testing.assert_allclose(out, np.zeros_like(x), atol=1e-6)

    def test_odd_sequence_length(self):
        """Test with odd sequence length L=15"""
        x = np.ones((1, 15, 4), dtype=np.float32)
        weights = np.ones((15 // 2 + 1, 4, 2), dtype=np.float32)
        out = reference(x, weights)
        self.assertEqual(out.shape, (1, 15, 4))

    def test_invalid_shapes(self):
        """Test validation error handling on invalid shapes"""
        x = np.ones((2, 16, 4), dtype=np.float32)
        bad_weights = np.ones((5, 4, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            reference(x, bad_weights)


if __name__ == "__main__":
    unittest.main()
