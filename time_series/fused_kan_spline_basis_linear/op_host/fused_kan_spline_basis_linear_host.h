// ----------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#ifndef FUSED_KAN_SPLINE_BASIS_LINEAR_HOST_H
#define FUSED_KAN_SPLINE_BASIS_LINEAR_HOST_H

#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling
{
ge::graphStatus TilingForFusedKanSplineBasisLinear(gert::TilingContext* context);
}

namespace ops
{
ge::graphStatus InferShapeForFusedKanSplineBasisLinear(gert::InferShapeContext* context);
ge::graphStatus InferDataTypeForFusedKanSplineBasisLinear(gert::InferDataTypeContext* context);
}  // namespace ops

#endif  // FUSED_KAN_SPLINE_BASIS_LINEAR_HOST_H
