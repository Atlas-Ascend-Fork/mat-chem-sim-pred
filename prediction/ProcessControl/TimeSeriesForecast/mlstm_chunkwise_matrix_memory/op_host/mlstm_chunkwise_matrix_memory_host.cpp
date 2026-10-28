/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mlstm_chunkwise_matrix_memory_host.h"
#include "mlstm_chunkwise_matrix_memory_tiling.h"
#include "op_host_utils.h"

namespace optiling {
static bool ValidateShapes(const gert::TilingContext* context, uint32_t& totalLength) {
    if (context == nullptr) {
        return false;
    }
    for (size_t idx = 0; idx < 5; ++idx) {
        if (context->GetInputShape(idx) == nullptr) {
            return false;
        }
    }
    const auto& s0 = context->GetInputShape(0)->GetStorageShape();
    size_t dimNum = s0.GetDimNum();
    for (size_t idx = 1; idx < 5; ++idx) {
        const auto& s = context->GetInputShape(idx)->GetStorageShape();
        if (s.GetDimNum() != dimNum) {
            return false;
        }
        for (size_t i = 0; i < dimNum; ++i) {
            if (s.GetDim(i) != s0.GetDim(i)) {
                return false;
            }
        }
    }
    uint64_t total = 1;
    for (size_t i = 0; i < dimNum; ++i) {
        int64_t d = s0.GetDim(i);
        if (d <= 0) {
            return false;
        }
        total *= static_cast<uint64_t>(d);
        if (total > static_cast<uint64_t>(UINT32_MAX)) {
            return false;
        }
    }
    totalLength = static_cast<uint32_t>(total);
    return true;
}

ge::graphStatus TilingForMlstmChunkwiseMatrixMemory(gert::TilingContext* context) {
    uint32_t totalLength = 0;
    if (!ValidateShapes(context, totalLength)) {
        return ge::GRAPH_FAILED;
    }
    MlstmChunkwiseMatrixMemoryTilingData tiling;
    int64_t attr_val = 64;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetInt(0) != nullptr) {
        attr_val = *(attrs->GetInt(0));
    }
    uint32_t tileNum = (totalLength + 1023U) / 1024U;
    if (tileNum == 0U) {
        tileNum = 1U;
    }
    tiling.set_totalLength(totalLength);
    tiling.set_attrParam(static_cast<uint32_t>(attr_val > 0 ? attr_val : 64));
    tiling.set_tileNum(tileNum);

    context->SetBlockDim(40);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
ge::graphStatus InferShapeForMlstmChunkwiseMatrixMemory(gert::InferShapeContext* context) {
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForMlstmChunkwiseMatrixMemory(gert::InferDataTypeContext* context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
} // namespace ops
