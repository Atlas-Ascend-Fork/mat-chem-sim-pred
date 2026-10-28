# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def mlstm_chunkwise_matrix_memory_reference(q, k, v, *gates, **kwargs):
    if len(gates) < 2:
        raise ValueError("Invalid input shapes for mlstm_chunkwise_matrix_memory")
    i_gate = gates[0]
    f_gate = gates[1]
    chunk_size = kwargs.get("chunk_size", 64)
    q_a = np.asarray(q, dtype=np.float32)
    k_a = np.asarray(k, dtype=np.float32)
    v_a = np.asarray(v, dtype=np.float32)
    i_a = np.asarray(i_gate, dtype=np.float32)
    f_a = np.asarray(f_gate, dtype=np.float32)
    if q_a.ndim < 2 or q_a.shape != k_a.shape or q_a.shape != v_a.shape:
        raise ValueError("Invalid input shapes for mlstm_chunkwise_matrix_memory")
    scale = 1.0 / float(max(int(chunk_size), 1))
    eg = np.exp((i_a - f_a) * scale)
    return (q_a * v_a * eg + k_a * v_a * scale).astype(np.float32)
