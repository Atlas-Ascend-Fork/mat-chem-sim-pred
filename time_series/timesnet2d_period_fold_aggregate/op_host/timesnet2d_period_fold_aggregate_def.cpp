// ----------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#include "op_def_utils.h"
#include "register/op_def_registry.h"
#include "timesnet2d_period_fold_aggregate_host.h"

namespace ops
{
class Timesnet2dPeriodFoldAggregate : public OpDef
{
   public:
    explicit Timesnet2dPeriodFoldAggregate(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("amp_weights").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("stencil_filter").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("base_period").AttrType(OPTIONAL).Int(16);

        this->SetInferShape(ops::InferShapeForTimesnet2dPeriodFoldAggregate)
            .SetInferDataType(ops::InferDataTypeForTimesnet2dPeriodFoldAggregate);
        this->AICore().SetTiling(optiling::TilingForTimesnet2dPeriodFoldAggregate);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Timesnet2dPeriodFoldAggregate);
}  // namespace ops
