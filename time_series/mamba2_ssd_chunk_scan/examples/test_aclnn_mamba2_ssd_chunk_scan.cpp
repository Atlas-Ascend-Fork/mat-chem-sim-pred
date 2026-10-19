/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#if __has_include("aclnn_mamba2_ssd_chunk_scan.h")
#include "aclnn_mamba2_ssd_chunk_scan.h"
#else
#include "op_host/mamba2_ssd_chunk_scan_host.h"
#endif

namespace
{
constexpr int32_t kB = 2;
constexpr int32_t kL = 64;
constexpr int32_t kH = 2;
constexpr int32_t kP = 16;
constexpr int32_t kN = 16;
constexpr int64_t kQ = 32;

struct HostBuffers {
    std::vector<float> u;
    std::vector<float> dt;
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    std::vector<float> d;
    std::vector<float> out;
    std::vector<float> ref;
};

struct DeviceBuffers {
    void* u{nullptr};
    void* dt{nullptr};
    void* a{nullptr};
    void* b{nullptr};
    void* c{nullptr};
    void* d{nullptr};
    void* out{nullptr};
};

void InitHostData(HostBuffers& h)
{
    h.u.resize(kB * kL * kH * kP);
    h.dt.resize(kB * kL * kH);
    h.a.resize(kH);
    h.b.resize(kB * kL * kH * kN);
    h.c.resize(kB * kL * kH * kN);
    h.d.resize(kH * kP);
    h.out.assign(kB * kL * kH * kP, 0.0F);
    h.ref.assign(kB * kL * kH * kP, 0.0F);

    for (size_t i = 0; i < h.u.size(); ++i) {
        h.u[i] = 0.25F * std::sin(0.13F * static_cast<float>(i + 1U)) + 0.1F;
    }
    for (size_t i = 0; i < h.dt.size(); ++i) {
        h.dt[i] = 0.02F + 0.03F * static_cast<float>((i * 7U + 3U) % 11U) / 11.0F;
    }
    for (int32_t head = 0; head < kH; ++head) {
        h.a[head] = -0.35F * static_cast<float>(head + 1);
    }
    for (size_t i = 0; i < h.b.size(); ++i) {
        h.b[i] = 0.2F * std::cos(0.17F * static_cast<float>(i + 2U));
        h.c[i] = 0.2F * std::sin(0.19F * static_cast<float>(i + 5U));
    }
    for (size_t i = 0; i < h.d.size(); ++i) {
        h.d[i] = 0.5F + 0.05F * static_cast<float>(i % 7U);
    }
}

void UpdateSingleStepCpu(
    int32_t stepIdx, int32_t head, const HostBuffers& h, std::vector<float>& state, std::vector<float>& refOut)
{
    const float dtVal = h.dt[stepIdx];
    const float alpha = std::exp(dtVal * h.a[head]);
    for (int32_t n = 0; n < kN; ++n) {
        const float bVal = h.b[stepIdx * kN + n];
        for (int32_t p = 0; p < kP; ++p) {
            state[n * kP + p] = alpha * state[n * kP + p] + dtVal * bVal * h.u[stepIdx * kP + p];
        }
    }
    for (int32_t p = 0; p < kP; ++p) {
        float acc = h.d[head * kP + p] * h.u[stepIdx * kP + p];
        for (int32_t n = 0; n < kN; ++n) {
            acc += h.c[stepIdx * kN + n] * state[n * kP + p];
        }
        refOut[stepIdx * kP + p] = acc;
    }
}

void ComputeCpuReference(const HostBuffers& h, std::vector<float>& refOut)
{
    std::vector<float> state(kN * kP, 0.0F);
    for (int32_t b = 0; b < kB; ++b) {
        for (int32_t head = 0; head < kH; ++head) {
            std::fill(state.begin(), state.end(), 0.0F);
            for (int32_t t = 0; t < kL; ++t) {
                const int32_t stepIdx = (b * kL + t) * kH + head;
                UpdateSingleStepCpu(stepIdx, head, h, state, refOut);
            }
        }
    }
}

aclTensor* MakeAclTensor(const std::vector<int64_t>& shape, aclDataType dtype, void* devPtr)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return aclCreateTensor(
        shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), devPtr);
}

void AllocAndCopyBuffers(const HostBuffers& h, DeviceBuffers& buf)
{
    auto allocCopy = [](void** dPtr, const std::vector<float>& src) {
        const size_t bytes = src.size() * sizeof(float);
        (void)aclrtMalloc(dPtr, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
        (void)aclrtMemcpy(*dPtr, bytes, src.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    };
    allocCopy(&buf.u, h.u);
    allocCopy(&buf.dt, h.dt);
    allocCopy(&buf.a, h.a);
    allocCopy(&buf.b, h.b);
    allocCopy(&buf.c, h.c);
    allocCopy(&buf.d, h.d);
    allocCopy(&buf.out, h.out);
}

void FreeBuffers(const std::vector<void*>& ptrs)
{
    for (void* p : ptrs) {
        if (p != nullptr) {
            (void)aclrtFree(p);
        }
    }
}

bool RunAclnnOp(const DeviceBuffers& buf, aclrtStream stream)
{
    aclTensor* tU = MakeAclTensor({kB, kL, kH, kP}, ACL_FLOAT, buf.u);
    aclTensor* tDt = MakeAclTensor({kB, kL, kH}, ACL_FLOAT, buf.dt);
    aclTensor* tA = MakeAclTensor({kH}, ACL_FLOAT, buf.a);
    aclTensor* tB = MakeAclTensor({kB, kL, kH, kN}, ACL_FLOAT, buf.b);
    aclTensor* tC = MakeAclTensor({kB, kL, kH, kN}, ACL_FLOAT, buf.c);
    aclTensor* tD = MakeAclTensor({kH, kP}, ACL_FLOAT, buf.d);
    aclTensor* tOut = MakeAclTensor({kB, kL, kH, kP}, ACL_FLOAT, buf.out);

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclError wsRet = aclnnMamba2SsdChunkScanGetWorkspaceSize(
        tU, tDt, tA, tB, tC, tD, kQ, tOut, &wsSize, &executor);
    void* ws = nullptr;
    if (wsRet == ACL_SUCCESS && wsSize > 0) {
        (void)aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    }
    if (wsRet == ACL_SUCCESS) {
        (void)aclnnMamba2SsdChunkScan(ws, wsSize, executor, stream);
        (void)aclrtSynchronizeStream(stream);
    }
    if (ws != nullptr) {
        (void)aclrtFree(ws);
    }
    for (aclTensor* t : {tU, tDt, tA, tB, tC, tD, tOut}) {
        aclDestroyTensor(t);
    }
    return (wsRet == ACL_SUCCESS);
}
bool VerifyResults(const HostBuffers& h, bool ok)
{
    float maxAbsErr = 0.0F;
    float maxRelErr = 0.0F;
    for (size_t i = 0; i < h.out.size(); ++i) {
        const float diff = std::fabs(h.out[i] - h.ref[i]);
        const float rel = diff / (std::fabs(h.ref[i]) + 1e-6F);
        maxAbsErr = std::max(maxAbsErr, diff);
        maxRelErr = std::max(maxRelErr, rel);
    }
    std::cout << "[VERIFY] max_abs_error=" << maxAbsErr << " max_rel_error=" << maxRelErr << std::endl;
    if (!ok || maxAbsErr > 1e-4F || std::isnan(maxAbsErr)) {
        std::cerr << "[FAIL] Numerical mismatch against Mamba-2 SSD reference!" << std::endl;
        return false;
    }
    std::cout << "[SUCCESS] aclnnMamba2SsdChunkScan numerical & smoke execution completed." << std::endl;
    return true;
}

bool ExecuteSmoke(int32_t deviceId, HostBuffers& h)
{
    aclrtStream stream = nullptr;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(deviceId) != ACL_SUCCESS ||
        aclrtCreateStream(&stream) != ACL_SUCCESS) {
        return false;
    }
    DeviceBuffers buf;
    AllocAndCopyBuffers(h, buf);
    const bool ok = RunAclnnOp(buf, stream);
    const size_t outBytes = h.out.size() * sizeof(float);
    (void)aclrtMemcpy(h.out.data(), outBytes, buf.out, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    FreeBuffers({buf.u, buf.dt, buf.a, buf.b, buf.c, buf.d, buf.out});
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
    return ok;
}
}  // namespace

int main(int argc, char* argv[])
{
    const int32_t deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;
    HostBuffers h;
    InitHostData(h);
    ComputeCpuReference(h, h.ref);
    const bool ok = ExecuteSmoke(deviceId, h);
    return VerifyResults(h, ok) ? 0 : 1;
}
