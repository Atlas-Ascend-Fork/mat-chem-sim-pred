/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_RFFT_SPECTRAL_FILTER_TILING_H_
#define FUSED_RFFT_SPECTRAL_FILTER_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(FusedRfftSpectralFilterTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch_size);
TILING_DATA_FIELD_DEF(uint32_t, sequence_length);
TILING_DATA_FIELD_DEF(uint32_t, freq_length);
TILING_DATA_FIELD_DEF(uint32_t, d_model);
TILING_DATA_FIELD_DEF(uint32_t, block_dim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedRfftSpectralFilter, FusedRfftSpectralFilterTilingData)
}  // namespace optiling

#endif  // FUSED_RFFT_SPECTRAL_FILTER_TILING_H_
