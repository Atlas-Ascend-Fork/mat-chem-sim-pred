/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAMBA2_SSD_CHUNK_SCAN_HOST_H_
#define MAMBA2_SSD_CHUNK_SCAN_HOST_H_

#include <cstdint>
#include "acl/acl.h"

struct aclTensor;
struct aclOpExecutor;

#ifdef __cplusplus
extern "C" {
#endif

aclError aclnnMamba2SsdChunkScanGetWorkspaceSize(
    const aclTensor* u,
    const aclTensor* dt,
    const aclTensor* a,
    const aclTensor* b,
    const aclTensor* c,
    const aclTensor* d,
    int64_t chunkSize,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclError aclnnMamba2SsdChunkScan(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // MAMBA2_SSD_CHUNK_SCAN_HOST_H_
