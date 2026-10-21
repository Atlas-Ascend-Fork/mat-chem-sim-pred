// ----------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#include <cmath>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn_timesnet2d_period_fold_aggregate.h"

static aclTensor* MakeTensorOnDevice(const std::vector<int64_t>& shape, const float* hostData, void** devPtr)
{
    size_t count = 1;
    for (auto d : shape)
    {
        count *= static_cast<size_t>(d);
    }
    size_t bytes = count * sizeof(float);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i)
    {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    (void)aclrtMalloc(devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (hostData != nullptr)
    {
        (void)aclrtMemcpy(*devPtr, bytes, hostData, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                           shape.size(), *devPtr);
}

static bool CheckOutputAccuracy(const std::vector<float>& h_out, float in0, float in1, float in2, int64_t attrVal)
{
    float scale = 1.0f / static_cast<float>(attrVal > 0 ? attrVal : 16);
    float w_amp = std::exp(in1 * scale);
    float ref = in0 + (in0 * in2 * w_amp) * scale;
    float err = std::fabs(h_out[0] - ref);
    if (err > 1e-3f)
    {
        std::cerr << "Mismatch: got " << h_out[0] << " expected " << ref << std::endl;
        return false;
    }
    std::cout << "[SUCCESS] aclnnTimesnet2dPeriodFoldAggregate check passed (max_abs_err=" << err << ")." << std::endl;
    return true;
}

int main(int argc, char** argv)
{
    int32_t deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(deviceId) != ACL_SUCCESS)
    {
        return -1;
    }
    aclrtStream stream = nullptr;
    (void)aclrtCreateStream(&stream);

    std::vector<int64_t> shape = {4, 64, 16};
    size_t total = 4 * 64 * 16;
    size_t bytes = total * sizeof(float);
    std::vector<float> h0(total, 0.15f), h1(total, 0.30f), h2(total, 0.45f);
    std::vector<float> h_out(total, 0.0f);

    void *d0 = nullptr, *d1 = nullptr, *d2 = nullptr, *d_out = nullptr;
    aclTensor* t0 = MakeTensorOnDevice(shape, h0.data(), &d0);
    aclTensor* t1 = MakeTensorOnDevice(shape, h1.data(), &d1);
    aclTensor* t2 = MakeTensorOnDevice(shape, h2.data(), &d2);
    aclTensor* tout = MakeTensorOnDevice(shape, nullptr, &d_out);

    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    auto ret = aclnnTimesnet2dPeriodFoldAggregateGetWorkspaceSize(t0, t1, t2, 16, tout, &ws, &ex);
    if (ret == ACL_SUCCESS)
    {
        void* wptr = nullptr;
        if (ws > 0) (void)aclrtMalloc(&wptr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        (void)aclnnTimesnet2dPeriodFoldAggregate(wptr, ws, ex, stream);
        (void)aclrtSynchronizeStream(stream);
        (void)aclrtMemcpy(h_out.data(), bytes, d_out, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (wptr != nullptr) (void)aclrtFree(wptr);
    }

    bool ok = CheckOutputAccuracy(h_out, h0[0], h1[0], h2[0], 16);
    (void)aclDestroyTensor(t0);
    (void)aclDestroyTensor(t1);
    (void)aclDestroyTensor(t2);
    (void)aclDestroyTensor(tout);
    (void)aclrtFree(d0);
    (void)aclrtFree(d1);
    (void)aclrtFree(d2);
    (void)aclrtFree(d_out);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
    return ok ? 0 : -1;
}
