/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAMBA2_SSD_CHUNK_SCAN_GRAD_HOST_H
#define MAMBA2_SSD_CHUNK_SCAN_GRAD_HOST_H

#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"

namespace optiling {
ge::graphStatus TilingForMamba2SsdChunkScanGrad(gert::TilingContext* context);
}

namespace ops {
ge::graphStatus InferShapeForMamba2SsdChunkScanGrad(gert::InferShapeContext* context);
ge::graphStatus InferDataTypeForMamba2SsdChunkScanGrad(gert::InferDataTypeContext* context);
}

#endif // MAMBA2_SSD_CHUNK_SCAN_GRAD_HOST_H
