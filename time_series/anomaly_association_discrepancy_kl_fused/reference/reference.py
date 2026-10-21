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


def anomaly_association_discrepancy_kl_fused_reference(q, k, sigma, window_size=64):
    q_a = np.asarray(q, dtype=np.float32)
    k_a = np.asarray(k, dtype=np.float32)
    sig_a = np.asarray(sigma, dtype=np.float32)
    if q_a.ndim < 2 or q_a.shape != k_a.shape:
        raise ValueError(
            "Invalid input shapes for anomaly_association_discrepancy_kl_fused"
        )
    scale = 1.0 / float(max(int(window_size), 1))
    diff = q_a - k_a
    prior_w = np.exp(sig_a * scale)
    return (q_a + (diff * diff) * prior_w).astype(np.float32)
