/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_rfft_spectral_filter_host.h"
#include "register/op_def_registry.h"
#include "op_def_utils.h"

namespace optiling {
ge::graphStatus FusedRfftSpectralFilterTiling(gert::TilingContext* context);
}

namespace ge {
graphStatus InferFusedRfftSpectralFilterShape(gert::InferShapeContext* context);
graphStatus InferFusedRfftSpectralFilterDataType(gert::InferDataTypeContext* context);
}  // namespace ge

namespace ops {
class FusedRfftSpectralFilter : public OpDef {
public:
    explicit FusedRfftSpectralFilter(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weights").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferFusedRfftSpectralFilterShape)
            .SetInferDataType(ge::InferFusedRfftSpectralFilterDataType);
        this->AICore().SetTiling(optiling::FusedRfftSpectralFilterTiling);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(FusedRfftSpectralFilter);
}  // namespace ops
