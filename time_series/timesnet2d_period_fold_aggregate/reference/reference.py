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


def timesnet2d_period_fold_aggregate_reference(
    x, amp_weights, stencil_filter, base_period=16
):
    x_a = np.asarray(x, dtype=np.float32)
    amp_a = np.asarray(amp_weights, dtype=np.float32)
    st_a = np.asarray(stencil_filter, dtype=np.float32)
    if x_a.ndim < 2 or x_a.shape != amp_a.shape:
        raise ValueError("Invalid input shapes for timesnet2d_period_fold_aggregate")
    scale = 1.0 / float(max(int(base_period), 1))
    w_amp = np.exp(amp_a * scale)
    return (x_a + (x_a * st_a * w_amp) * scale).astype(np.float32)
