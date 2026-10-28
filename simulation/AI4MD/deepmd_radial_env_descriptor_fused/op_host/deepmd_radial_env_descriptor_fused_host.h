/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEEPMD_RADIAL_ENV_DESCRIPTOR_FUSED_HOST_H
#define DEEPMD_RADIAL_ENV_DESCRIPTOR_FUSED_HOST_H

#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"

namespace optiling {
ge::graphStatus TilingForDeepmdRadialEnvDescriptorFused(gert::TilingContext* context);
}

namespace ops {
ge::graphStatus InferShapeForDeepmdRadialEnvDescriptorFused(gert::InferShapeContext* context);
ge::graphStatus InferDataTypeForDeepmdRadialEnvDescriptorFused(gert::InferDataTypeContext* context);
}

#endif // DEEPMD_RADIAL_ENV_DESCRIPTOR_FUSED_HOST_H
