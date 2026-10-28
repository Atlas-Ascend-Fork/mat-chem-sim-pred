# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def deepmd_radial_env_descriptor_fused_reference(rel_coord, radial_weight, env_scale, nbr_count=64):
    rc_a = np.asarray(rel_coord, dtype=np.float32)
    rw_a = np.asarray(radial_weight, dtype=np.float32)
    es_a = np.asarray(env_scale, dtype=np.float32)
    if rc_a.ndim < 2 or rc_a.shape != rw_a.shape or rc_a.shape != es_a.shape:
        raise ValueError("Invalid input shapes for deepmd_radial_env_descriptor_fused")
    scale = 1.0 / float(max(int(nbr_count), 1))
    env = np.exp((rc_a * rw_a) * scale)
    return (rc_a + env * es_a).astype(np.float32)
