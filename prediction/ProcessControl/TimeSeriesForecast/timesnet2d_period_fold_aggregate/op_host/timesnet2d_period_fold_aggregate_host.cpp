// ----------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#include "timesnet2d_period_fold_aggregate_host.h"
#include "timesnet2d_period_fold_aggregate_tiling.h"
#include "op_host_utils.h"

namespace optiling {
ge::graphStatus TilingForTimesnet2dPeriodFoldAggregate(gert::TilingContext* context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    Timesnet2dPeriodFoldAggregateTilingData tiling;
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    if (x_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalLength = 1;
    const gert::Shape& shape = x_shape->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalLength *= static_cast<uint32_t>(shape.GetDim(i));
    }
    int64_t attr_val = 16;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetInt(0) != nullptr) {
        attr_val = *(attrs->GetInt(0));
    }
    uint32_t tileNum = (totalLength + 1023U) / 1024U;
    if (tileNum == 0U) {
        tileNum = 1U;
    }
    tiling.set_totalLength(totalLength);
    tiling.set_attrParam(static_cast<uint32_t>(attr_val > 0 ? attr_val : 16));
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
ge::graphStatus InferShapeForTimesnet2dPeriodFoldAggregate(gert::InferShapeContext* context) {
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForTimesnet2dPeriodFoldAggregate(gert::InferDataTypeContext* context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
} // namespace ops
