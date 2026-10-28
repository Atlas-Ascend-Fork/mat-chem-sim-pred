/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#if __has_include("aclnn_fused_rfft_spectral_filter.h")
#include "aclnn_fused_rfft_spectral_filter.h"
#else
#include "op_host/fused_rfft_spectral_filter_host.h"
#endif

namespace
{
constexpr int32_t kBatch = 2;
constexpr int32_t kSeqLen = 16;
constexpr int32_t kDModel = 8;
constexpr int32_t kFreqLen = kSeqLen / 2 + 1;
constexpr float kTwoPi = 6.28318530717958647692f;

void CleanupAcl(void* devX, void* devW, void* devY, aclrtStream stream, int32_t deviceId)
{
    if (devX != nullptr) {
        (void)aclrtFree(devX);
    }
    if (devW != nullptr) {
        (void)aclrtFree(devW);
    }
    if (devY != nullptr) {
        (void)aclrtFree(devY);
    }
    if (stream != nullptr) {
        (void)aclrtDestroyStream(stream);
    }
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
}

aclTensor* CreateAclTensor(const std::vector<int64_t>& shape, aclDataType dtype, void* devPtr)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return aclCreateTensor(
        shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), devPtr);
}

void ComputeHostSpectralReference(
    const std::vector<float>& hostX, const std::vector<float>& hostW, std::vector<float>& refY)
{
    std::vector<float> hImpulse(kSeqLen * kDModel, 0.0f);
    for (int32_t d = 0; d < kDModel; ++d) {
        const float w0r = hostW[d * 2];
        const float wNyqr = hostW[((kSeqLen / 2) * kDModel + d) * 2];
        for (int32_t delta = 0; delta < kSeqLen; ++delta) {
            float acc = w0r + (((delta & 1) != 0) ? -wNyqr : wNyqr);
            for (int32_t f = 1; f <= (kSeqLen - 1) / 2; ++f) {
                const float angle = kTwoPi * static_cast<float>(f * delta) / static_cast<float>(kSeqLen);
                const float wr = hostW[(f * kDModel + d) * 2];
                const float wi = hostW[(f * kDModel + d) * 2 + 1];
                acc += 2.0f * (wr * std::cos(angle) - wi * std::sin(angle));
            }
            hImpulse[delta * kDModel + d] = acc / static_cast<float>(kSeqLen);
        }
    }
    for (int32_t b = 0; b < kBatch; ++b) {
        for (int32_t t = 0; t < kSeqLen; ++t) {
            for (int32_t d = 0; d < kDModel; ++d) {
                float sum = 0.0f;
                for (int32_t delta = 0; delta < kSeqLen; ++delta) {
                    const int32_t tau = (t - delta + kSeqLen) % kSeqLen;
                    sum += hostX[(b * kSeqLen + tau) * kDModel + d] * hImpulse[delta * kDModel + d];
                }
                refY[(b * kSeqLen + t) * kDModel + d] = sum;
            }
        }
    }
}

bool PrepareDeviceData(
    std::vector<float>& hostX, std::vector<float>& hostW,
    void*& devX, void*& devW, void*& devY, size_t xSize, size_t wSize, size_t ySize)
{
    hostX.resize(kBatch * kSeqLen * kDModel);
    hostW.resize(kFreqLen * kDModel * 2);
    for (size_t i = 0; i < hostX.size(); ++i) {
        hostX[i] = 0.4f * std::sin(0.17f * static_cast<float>(i)) + 0.25f * std::cos(0.11f * static_cast<float>(i));
    }
    for (int32_t f = 0; f < kFreqLen; ++f) {
        for (int32_t d = 0; d < kDModel; ++d) {
            const size_t idx = static_cast<size_t>((f * kDModel + d) * 2);
            hostW[idx] = 0.65f * std::cos(0.2f * static_cast<float>(f + d));
            hostW[idx + 1] = (f == 0 || f == kSeqLen / 2) ? 0.0f : -0.25f * std::sin(0.15f * static_cast<float>(f + d));
        }
    }
    const bool allocOk = (aclrtMalloc(&devX, xSize, ACL_MEM_MALLOC_NORMAL_ONLY) == ACL_SUCCESS) &&
                         (aclrtMalloc(&devW, wSize, ACL_MEM_MALLOC_NORMAL_ONLY) == ACL_SUCCESS) &&
                         (aclrtMalloc(&devY, ySize, ACL_MEM_MALLOC_NORMAL_ONLY) == ACL_SUCCESS);
    if (!allocOk) {
        return false;
    }
    return (aclrtMemcpy(devX, xSize, hostX.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS) &&
           (aclrtMemcpy(devW, wSize, hostW.data(), wSize, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
}

bool ExecuteAclnnKernel(
    const std::vector<float>& hostX, const std::vector<float>& hostW,
    aclTensor* tX, aclTensor* tW, aclTensor* tY, aclrtStream stream, void* devY, size_t ySize)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclError wsStatus = aclnnFusedRfftSpectralFilterGetWorkspaceSize(tX, tW, tY, &workspaceSize, &executor);
    if (wsStatus != ACL_SUCCESS) {
        std::cerr << "[ERROR] GetWorkspaceSize failed, status=" << wsStatus << std::endl;
        return false;
    }
    void* workspace = nullptr;
    if (workspaceSize > 0 && aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
        return false;
    }
    const aclError execStatus = aclnnFusedRfftSpectralFilter(workspace, workspaceSize, executor, stream);
    if (execStatus != ACL_SUCCESS) {
        std::cerr << "[ERROR] Execution failed, status=" << execStatus << std::endl;
    }
    (void)aclrtSynchronizeStream(stream);
    std::vector<float> hostY(kBatch * kSeqLen * kDModel, 0.0F);
    (void)aclrtMemcpy(hostY.data(), ySize, devY, ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (workspace != nullptr) {
        (void)aclrtFree(workspace);
    }
    std::vector<float> refY(kBatch * kSeqLen * kDModel, 0.0F);
    ComputeHostSpectralReference(hostX, hostW, refY);
    float maxAbsErr = 0.0f;
    for (size_t i = 0; i < hostY.size(); ++i) {
        const float err = std::fabs(hostY[i] - refY[i]);
        if (err > maxAbsErr) {
            maxAbsErr = err;
        }
    }
    std::cout << "[VERIFY] max_abs_err=" << maxAbsErr << std::endl;
    return (execStatus == ACL_SUCCESS) && (maxAbsErr < 1e-4f);
}
bool InitDevice(int32_t deviceId, aclrtStream& stream)
{
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclInit failed" << std::endl;
        return false;
    }
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSetDevice failed on device " << deviceId << std::endl;
        (void)aclFinalize();
        return false;
    }
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        (void)aclrtResetDevice(deviceId);
        (void)aclFinalize();
        return false;
    }
    return true;
}
}  // namespace

int main(int argc, char* argv[])
{
    const int32_t deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;
    aclrtStream stream = nullptr;
    if (!InitDevice(deviceId, stream)) {
        return 1;
    }
    const size_t xSize = kBatch * kSeqLen * kDModel * sizeof(float);
    const size_t wSize = kFreqLen * kDModel * 2 * sizeof(float);
    void *devX = nullptr, *devW = nullptr, *devY = nullptr;
    std::vector<float> hostX;
    std::vector<float> hostW;
    if (!PrepareDeviceData(hostX, hostW, devX, devW, devY, xSize, wSize, xSize)) {
        CleanupAcl(devX, devW, devY, stream, deviceId);
        return 1;
    }
    aclTensor* tensorX = CreateAclTensor({kBatch, kSeqLen, kDModel}, ACL_FLOAT, devX);
    aclTensor* tensorW = CreateAclTensor({kFreqLen, kDModel, 2}, ACL_FLOAT, devW);
    aclTensor* tensorY = CreateAclTensor({kBatch, kSeqLen, kDModel}, ACL_FLOAT, devY);
    const bool ok = ExecuteAclnnKernel(hostX, hostW, tensorX, tensorW, tensorY, stream, devY, xSize);
    aclDestroyTensor(tensorX);
    aclDestroyTensor(tensorW);
    aclDestroyTensor(tensorY);
    CleanupAcl(devX, devW, devY, stream, deviceId);
    if (ok) {
        std::cout << "[SUCCESS] aclnnFusedRfftSpectralFilter smoke execution completed." << std::endl;
        return 0;
    }
    return 1;
}
