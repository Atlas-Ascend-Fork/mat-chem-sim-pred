# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reference")))
from reference import mlstm_chunkwise_matrix_memory_reference


class MlstmChunkwiseMatrixMemoryReferenceTest(unittest.TestCase):
    def test_zero_input(self):
        in0 = np.zeros((2, 16, 8), dtype=np.float32)
        in1 = np.zeros((2, 16, 8), dtype=np.float32)
        in2 = np.zeros((2, 16, 8), dtype=np.float32)
        in3 = np.zeros((2, 16, 8), dtype=np.float32)
        in4 = np.zeros((2, 16, 8), dtype=np.float32)
        out = mlstm_chunkwise_matrix_memory_reference(in0, in1, in2, in3, in4)
        self.assertEqual(out.shape, (2, 16, 8))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_basic_computation(self):
        in0 = np.ones((2, 16, 8), dtype=np.float32) * 0.15
        in1 = np.ones((2, 16, 8), dtype=np.float32) * 0.30
        in2 = np.ones((2, 16, 8), dtype=np.float32) * 0.45
        in3 = np.ones((2, 16, 8), dtype=np.float32) * 0.60
        in4 = np.ones((2, 16, 8), dtype=np.float32) * 0.75
        out = mlstm_chunkwise_matrix_memory_reference(
            in0, in1, in2, in3, in4, chunk_size=64)
        self.assertEqual(out.shape, (2, 16, 8))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_invalid_shape_detection(self):
        bad0 = np.ones((4,), dtype=np.float32)
        bad1 = np.ones((4,), dtype=np.float32)
        bad2 = np.ones((4,), dtype=np.float32)
        bad3 = np.ones((4,), dtype=np.float32)
        bad4 = np.ones((4,), dtype=np.float32)
        with self.assertRaises(ValueError):
            mlstm_chunkwise_matrix_memory_reference(bad0, bad1, bad2, bad3, bad4)


if __name__ == "__main__":
    unittest.main()
