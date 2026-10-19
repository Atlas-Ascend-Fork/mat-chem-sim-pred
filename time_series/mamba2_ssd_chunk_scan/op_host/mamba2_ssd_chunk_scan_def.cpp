/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace optiling
{
ge::graphStatus Mamba2SsdChunkScanTiling(gert::TilingContext* context);
}  // namespace optiling

namespace ge
{
graphStatus InferMamba2SsdChunkScanShape(gert::InferShapeContext* context);
graphStatus InferMamba2SsdChunkScanDataType(gert::InferDataTypeContext* context);
}  // namespace ge

namespace ops
{
class Mamba2SsdChunkScan : public OpDef
{
public:
    explicit Mamba2SsdChunkScan(const char* name) : OpDef(name)
    {
        const char* const inputNames[] = {"u", "dt", "a", "b", "c", "d"};
        for (const auto* inName : inputNames) {
            this->Input(inName).ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        }
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("chunk_size").AttrType(OPTIONAL).Int(64);

        this->SetInferShape(ge::InferMamba2SsdChunkScanShape)
            .SetInferDataType(ge::InferMamba2SsdChunkScanDataType);
        this->AICore().SetTiling(optiling::Mamba2SsdChunkScanTiling);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend950");
    }
};

OP_ADD(Mamba2SsdChunkScan);
}  // namespace ops
