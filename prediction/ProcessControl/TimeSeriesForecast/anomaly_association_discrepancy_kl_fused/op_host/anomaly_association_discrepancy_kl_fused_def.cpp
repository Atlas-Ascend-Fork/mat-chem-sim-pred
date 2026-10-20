// ----------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#include "anomaly_association_discrepancy_kl_fused_host.h"
#include "register/op_def_registry.h"
#include "op_def_utils.h"

namespace ops {
class AnomalyAssociationDiscrepancyKlFused : public OpDef {
public:
    explicit AnomalyAssociationDiscrepancyKlFused(const char* name) : OpDef(name) {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("sigma").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("window_size").AttrType(OPTIONAL).Int(64);

        this->SetInferShape(ops::InferShapeForAnomalyAssociationDiscrepancyKlFused)
            .SetInferDataType(ops::InferDataTypeForAnomalyAssociationDiscrepancyKlFused);
        this->AICore().SetTiling(optiling::TilingForAnomalyAssociationDiscrepancyKlFused);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AnomalyAssociationDiscrepancyKlFused);
} // namespace ops
