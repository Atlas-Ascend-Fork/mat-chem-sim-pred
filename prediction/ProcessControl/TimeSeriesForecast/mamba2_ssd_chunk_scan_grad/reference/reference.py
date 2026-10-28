# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def mamba2_ssd_chunk_scan_grad_reference(dout, x, dt_decay, cb_matrix, chunk_size=64):
    dy_a = np.asarray(dout, dtype=np.float32)
    x_a = np.asarray(x, dtype=np.float32)
    dt_a = np.asarray(dt_decay, dtype=np.float32)
    cb_a = np.asarray(cb_matrix, dtype=np.float32)
    if dy_a.ndim < 2 or dy_a.shape != x_a.shape:
        raise ValueError("Invalid input shapes for mamba2_ssd_chunk_scan_grad")
    scale = 1.0 / float(max(int(chunk_size), 1))
    decay = np.exp(-dt_a * scale)
    return (dy_a * decay + (x_a * cb_a) * scale).astype(np.float32)
