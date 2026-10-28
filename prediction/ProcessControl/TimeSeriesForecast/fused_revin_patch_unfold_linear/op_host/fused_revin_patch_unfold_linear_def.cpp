/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_revin_patch_unfold_linear_host.h"
#include "register/op_def_registry.h"
#include "op_def_utils.h"

namespace ops {
class FusedRevinPatchUnfoldLinear : public OpDef {
public:
    explicit FusedRevinPatchUnfoldLinear(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("patch_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("patch_len").AttrType(OPTIONAL).Int(64);

        this->SetInferShape(ops::InferShapeForFusedRevinPatchUnfoldLinear)
            .SetInferDataType(ops::InferDataTypeForFusedRevinPatchUnfoldLinear);
        this->AICore().SetTiling(optiling::TilingForFusedRevinPatchUnfoldLinear);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(FusedRevinPatchUnfoldLinear);
} // namespace ops
