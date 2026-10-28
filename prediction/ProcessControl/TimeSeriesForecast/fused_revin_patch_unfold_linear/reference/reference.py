# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def fused_revin_patch_unfold_linear_reference(x, gamma, beta, patch_weight, patch_len=64):
    x_a = np.asarray(x, dtype=np.float32)
    gm_a = np.asarray(gamma, dtype=np.float32)
    bt_a = np.asarray(beta, dtype=np.float32)
    pw_a = np.asarray(patch_weight, dtype=np.float32)
    if x_a.ndim < 2 or x_a.shape != gm_a.shape:
        raise ValueError("Invalid input shapes for fused_revin_patch_unfold_linear")
    scale = 1.0 / float(max(int(patch_len), 1))
    norm_x = (x_a - bt_a) * gm_a
    return (norm_x + (norm_x * pw_a) * scale).astype(np.float32)
