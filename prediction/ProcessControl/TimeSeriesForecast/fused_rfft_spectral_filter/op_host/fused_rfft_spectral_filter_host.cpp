/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#if __has_include("../../common/op_host_utils.h")
#if __has_include("../../common/op_host_utils.h")
#include "../../common/op_host_utils.h"
#else
#include "op_host_utils.h"
#endif
#else
#include "op_host_utils.h"
#endif
#include "fused_rfft_spectral_filter_tiling.h"

namespace
{
constexpr uint32_t kMaxDim32 = std::numeric_limits<uint32_t>::max();

bool ValidateInputShapes(const gert::Shape& xShape, const gert::Shape& wShape)
{
    if (xShape.GetDimNum() != 3U || wShape.GetDimNum() != 3U) {
        return false;
    }
    const int64_t seqLen = xShape.GetDim(1);
    const int64_t dModel = xShape.GetDim(2);
    const int64_t freqLen = seqLen / 2 + 1;
    if (seqLen <= 0 || dModel <= 0 || seqLen > kMaxDim32 || dModel > kMaxDim32) {
        return false;
    }
    return wShape.GetDim(0) == freqLen && wShape.GetDim(1) == dModel && wShape.GetDim(2) == 2;
}

bool CheckTilingContext(gert::TilingContext* context)
{
    if (context == nullptr) {
        return false;
    }
    const auto* inX = context->GetInputShape(0);
    const auto* inW = context->GetInputShape(1);
    return inX != nullptr && inW != nullptr;
}
}  // namespace

namespace optiling
{
ge::graphStatus FusedRfftSpectralFilterTiling(gert::TilingContext* context)
{
    if (!CheckTilingContext(context)) {
        return ge::GRAPH_FAILED;
    }
    const auto& xShape = context->GetInputShape(0)->GetStorageShape();
    const auto& wShape = context->GetInputShape(1)->GetStorageShape();
    if (!ValidateInputShapes(xShape, wShape)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t batch = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t seqLen = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t dModel = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t freqLen = seqLen / 2U + 1U;

    FusedRfftSpectralFilterTilingData tiling;
    tiling.set_batch_size(batch);
    tiling.set_sequence_length(seqLen);
    tiling.set_freq_length(freqLen);
    tiling.set_d_model(dModel);

    const uint32_t aivCores = timeseries_host::GetVectorCoreCount(context);
    const uint32_t blockDim = timeseries_host::SelectBlockDim(batch, aivCores);
    tiling.set_block_dim(blockDim);

    context->SetBlockDim(blockDim);
    timeseries_host::StoreTilingData(context, tiling);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge
{
graphStatus InferFusedRfftSpectralFilterShape(gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetOutputShape(0) == nullptr) {
        return GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}

graphStatus InferFusedRfftSpectralFilterDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge
