# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np


def fused_kan_spline_basis_linear_reference(x, grid, spline_weight, grid_size=8):
    x_a = np.asarray(x, dtype=np.float32)
    g_a = np.asarray(grid, dtype=np.float32)
    w_a = np.asarray(spline_weight, dtype=np.float32)
    if x_a.ndim < 2 or x_a.shape != g_a.shape or x_a.shape != w_a.shape:
        raise ValueError("Invalid input shapes for fused_kan_spline_basis_linear")
    scale = 1.0 / float(max(int(grid_size), 1))
    u = (x_a - g_a) * scale
    basis = u * u + u
    return (basis * w_a + x_a).astype(np.float32)
