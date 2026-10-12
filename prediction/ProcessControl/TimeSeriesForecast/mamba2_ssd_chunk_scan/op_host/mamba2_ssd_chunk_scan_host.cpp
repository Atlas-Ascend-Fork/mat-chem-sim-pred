/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mamba2_ssd_chunk_scan_host.h"
#include "mamba2_ssd_chunk_scan_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace
{
constexpr uint32_t kDefaultChunkSize = 64U;
constexpr size_t kExtraWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;

inline uint32_t SelectBlockDim(uint32_t workItems, uint32_t coreLimit)
{
    if (workItems == 0U || coreLimit == 0U) {
        return 1U;
    }
    return (workItems < coreLimit) ? workItems : coreLimit;
}

inline bool CheckInputPresenceAndDims(const gert::TilingContext* context)
{
    for (size_t idx = 0U; idx < 6U; ++idx) {
        if (context->GetInputShape(idx) == nullptr) {
            return false;
        }
    }
    const size_t expectedDims[6] = {4, 3, 1, 4, 4, 2};
    for (size_t idx = 0U; idx < 6U; ++idx) {
        if (context->GetInputShape(idx)->GetStorageShape().GetDimNum() != expectedDims[idx]) {
            return false;
        }
    }
    return true;
}

inline bool CheckMatchingDims(const gert::TilingContext* context)
{
    const auto& u = context->GetInputShape(0)->GetStorageShape();
    const auto& dt = context->GetInputShape(1)->GetStorageShape();
    const auto& a = context->GetInputShape(2)->GetStorageShape();
    const auto& b = context->GetInputShape(3)->GetStorageShape();
    const auto& c = context->GetInputShape(4)->GetStorageShape();
    const auto& d = context->GetInputShape(5)->GetStorageShape();

    const bool bOk = (dt.GetDim(0) == u.GetDim(0)) & (b.GetDim(0) == u.GetDim(0)) & (c.GetDim(0) == u.GetDim(0));
    const bool sOk = (dt.GetDim(1) == u.GetDim(1)) & (b.GetDim(1) == u.GetDim(1)) & (c.GetDim(1) == u.GetDim(1));
    const bool hOk = (dt.GetDim(2) == u.GetDim(2)) & (a.GetDim(0) == u.GetDim(2)) &
                     (b.GetDim(2) == u.GetDim(2)) & (c.GetDim(2) == u.GetDim(2)) & (d.GetDim(0) == u.GetDim(2));
    const bool pOk = (d.GetDim(1) == u.GetDim(3)) & (c.GetDim(3) == b.GetDim(3));
    return bOk && sOk && hOk && pOk;
}

inline bool ValidateMamba2Shapes(const gert::TilingContext* context)
{
    if (context == nullptr) {
        return false;
    }
    if (!CheckInputPresenceAndDims(context)) {
        return false;
    }
    return CheckMatchingDims(context);
}

inline uint32_t GetChunkSizeAttr(gert::TilingContext* context)
{
    if (context == nullptr) {
        return kDefaultChunkSize;
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return kDefaultChunkSize;
    }
    const uint32_t* cs = attrs->GetAttrPointer<uint32_t>(0);
    if (cs != nullptr && *cs > 0U) {
        return *cs;
    }
    return kDefaultChunkSize;
}

void FillMamba2Tiling(gert::TilingContext* context, optiling::Mamba2SsdChunkScanTilingData& tiling)
{
    const auto& u = context->GetInputShape(0)->GetStorageShape();
    const auto& b = context->GetInputShape(3)->GetStorageShape();
    const uint32_t batch = static_cast<uint32_t>(u.GetDim(0));
    const uint32_t seqLen = static_cast<uint32_t>(u.GetDim(1));
    const uint32_t nheads = static_cast<uint32_t>(u.GetDim(2));
    const uint32_t headdim = static_cast<uint32_t>(u.GetDim(3));
    const uint32_t dstate = static_cast<uint32_t>(b.GetDim(3));
    uint32_t chunkSize = GetChunkSizeAttr(context);
    if (chunkSize == 0U) {
        chunkSize = kDefaultChunkSize;
    }
    const uint32_t nchunks = (chunkSize > 0U) ? ((seqLen + chunkSize - 1U) / chunkSize) : 1U;

    tiling.set_batch_size(batch);
    tiling.set_seq_len(seqLen);
    tiling.set_nheads(nheads);
    tiling.set_headdim(headdim);
    tiling.set_dstate(dstate);
    tiling.set_chunk_size(chunkSize);
    tiling.set_nchunks(nchunks);

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aicCores = platform.GetCoreNumAic();
    const uint32_t coreLimit = (aicCores > 0U) ? aicCores : platform.GetCoreNumAiv();
    const uint32_t neededBlocks = batch * nheads;
    const uint32_t blockDim = SelectBlockDim(neededBlocks, coreLimit);
    tiling.set_block_dim(blockDim);
    context->SetBlockDim(blockDim);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) + kExtraWorkspaceBytes;
    }
}
}  // namespace

namespace optiling
{
ge::graphStatus Mamba2SsdChunkScanTiling(gert::TilingContext* context)
{
    if (context == nullptr || !ValidateMamba2Shapes(context)) {
        return ge::GRAPH_FAILED;
    }

    Mamba2SsdChunkScanTilingData tiling;
    FillMamba2Tiling(context, tiling);
    gert::TilingData* rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge
{
graphStatus InferMamba2SsdChunkScanShape(gert::InferShapeContext* context)
{
    const gert::Shape* uShape = (context != nullptr) ? context->GetInputShape(0) : nullptr;
    gert::Shape* yShape = (context != nullptr) ? context->GetOutputShape(0) : nullptr;
    if (uShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }
    *yShape = *uShape;
    return GRAPH_SUCCESS;
}

graphStatus InferMamba2SsdChunkScanDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const ge::DataType uDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, uDtype);
    return GRAPH_SUCCESS;
}
}  // namespace ge
