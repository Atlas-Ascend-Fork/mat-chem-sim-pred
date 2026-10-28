/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mamba2_ssd_chunk_scan_grad_host.h"
#include "register/op_def_registry.h"
#if __has_include("../../common/op_def_utils.h")
#include "../../common/op_def_utils.h"
#elif __has_include("../common/op_def_utils.h")
#include "../common/op_def_utils.h"
#else
#include "op_def_utils.h"
#endif

namespace ops {
class Mamba2SsdChunkScanGrad : public OpDef {
public:
    explicit Mamba2SsdChunkScanGrad(const char* name) : OpDef(name) {
        this->Input("dout").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("dt_decay").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("cb_matrix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("chunk_size").AttrType(OPTIONAL).Int(64);

        this->SetInferShape(ops::InferShapeForMamba2SsdChunkScanGrad)
            .SetInferDataType(ops::InferDataTypeForMamba2SsdChunkScanGrad);
        this->AICore().SetTiling(optiling::TilingForMamba2SsdChunkScanGrad);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Mamba2SsdChunkScanGrad);
} // namespace ops
