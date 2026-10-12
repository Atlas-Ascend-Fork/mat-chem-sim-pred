/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>
#include <aclnn/opdev/common_types.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../op_host/mamba2_ssd_chunk_scan_host.h"

namespace
{
struct BenchCase {
    const char* name;
    int64_t B;
    int64_t L;
    int64_t H;
    int64_t P;
    int64_t N;
    int64_t chunkSize;
};

struct HostBenchBuffers {
    std::vector<float> hU;
    std::vector<float> hDt;
    std::vector<float> hA;
    std::vector<float> hB;
    std::vector<float> hC;
    std::vector<float> hD;
    std::vector<float> hOut;
};

struct AclBenchHandles {
    aclTensor* tU{nullptr};
    aclTensor* tDt{nullptr};
    aclTensor* tA{nullptr};
    aclTensor* tB{nullptr};
    aclTensor* tC{nullptr};
    aclTensor* tD{nullptr};
    aclTensor* tOut{nullptr};
    void* dU{nullptr};
    void* dDt{nullptr};
    void* dA{nullptr};
    void* dB{nullptr};
    void* dC{nullptr};
    void* dD{nullptr};
    void* dOut{nullptr};
    int64_t chunkSize{64};
    aclrtStream stream{nullptr};
};

aclTensor* CreateTensor1D(const std::vector<int64_t>& shape, const float* hostData, void** devPtr)
{
    size_t numel = 1;
    for (auto d : shape) {
        numel *= static_cast<size_t>(d);
    }
    const size_t bytes = numel * sizeof(float);
    aclrtMalloc(devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(*devPtr, bytes, hostData, bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
                           strides.data(), 0, ACL_FORMAT_ND,
                           shape.data(), shape.size(), *devPtr);
}

void RunSingleStep(AclBenchHandles& h)
{
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus st = aclnnMamba2SsdChunkScanGetWorkspaceSize(
        h.tU, h.tDt, h.tA, h.tB, h.tC, h.tD, h.chunkSize, h.tOut, &wsSize, &executor);
    if (st != ACL_SUCCESS) {
        std::printf("[ERROR] GetWorkspaceSize failed: %d\n", static_cast<int>(st));
        std::exit(1);
    }
    void* wsAddr = nullptr;
    if (wsSize > 0) {
        aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    aclnnMamba2SsdChunkScan(wsAddr, wsSize, executor, h.stream);
    aclrtSynchronizeStream(h.stream);
    if (wsAddr != nullptr) {
        aclrtFree(wsAddr);
    }
}

float VerifyFirstBatchHead(const BenchCase& c, const HostBenchBuffers& buf)
{
    float maxDiff = 0.0f;
    std::vector<float> state(static_cast<size_t>(c.N * c.P), 0.0f);
    for (int64_t t = 0; t < c.L; ++t) {
        const float dtVal = buf.hDt[static_cast<size_t>(t * c.H)];
        const float alpha = std::exp(dtVal * buf.hA[0]);
        for (int64_t n = 0; n < c.N; ++n) {
            const float bVal = buf.hB[static_cast<size_t>((t * c.H) * c.N + n)];
            for (int64_t p = 0; p < c.P; ++p) {
                const float uVal = buf.hU[static_cast<size_t>((t * c.H) * c.P + p)];
                state[static_cast<size_t>(n * c.P + p)] =
                    alpha * state[static_cast<size_t>(n * c.P + p)] + dtVal * bVal * uVal;
            }
        }
        for (int64_t p = 0; p < c.P; ++p) {
            float y = buf.hD[static_cast<size_t>(p)] * buf.hU[static_cast<size_t>((t * c.H) * c.P + p)];
            for (int64_t n = 0; n < c.N; ++n) {
                const float cVal = buf.hC[static_cast<size_t>((t * c.H) * c.N + n)];
                y += cVal * state[static_cast<size_t>(n * c.P + p)];
            }
            const float got = buf.hOut[static_cast<size_t>((t * c.H) * c.P + p)];
            const float diff = std::fabs(got - y);
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }
    }
    return maxDiff;
}

void InitHostBenchVectors(const BenchCase& c, HostBenchBuffers& buf)
{
    const size_t nU = static_cast<size_t>(c.B * c.L * c.H * c.P);
    const size_t nDt = static_cast<size_t>(c.B * c.L * c.H);
    const size_t nA = static_cast<size_t>(c.H);
    const size_t nBC = static_cast<size_t>(c.B * c.L * c.H * c.N);
    const size_t nD = static_cast<size_t>(c.H * c.P);

    buf.hU.resize(nU);
    buf.hDt.resize(nDt);
    buf.hA.resize(nA);
    buf.hB.resize(nBC);
    buf.hC.resize(nBC);
    buf.hD.resize(nD);
    buf.hOut.assign(nU, 0.0f);

    for (size_t i = 0; i < nU; ++i) {
        buf.hU[i] = std::sin(static_cast<float>(i % 97) * 0.07f) * 0.5f;
    }
    for (size_t i = 0; i < nDt; ++i) {
        buf.hDt[i] = 0.01f + 0.04f * std::cos(static_cast<float>(i % 53) * 0.1f);
    }
    for (size_t i = 0; i < nA; ++i) {
        buf.hA[i] = -0.5f - 0.1f * static_cast<float>(i % 8);
    }
    for (size_t i = 0; i < nBC; ++i) {
        buf.hB[i] = std::cos(static_cast<float>(i % 89) * 0.05f) * 0.25f;
        buf.hC[i] = std::sin(static_cast<float>(i % 83) * 0.06f) * 0.25f;
    }
    for (size_t i = 0; i < nD; ++i) {
        buf.hD[i] = 0.1f;
    }
}

void DestroyBenchHandles(AclBenchHandles& h)
{
    aclDestroyTensor(h.tU);
    aclDestroyTensor(h.tDt);
    aclDestroyTensor(h.tA);
    aclDestroyTensor(h.tB);
    aclDestroyTensor(h.tC);
    aclDestroyTensor(h.tD);
    aclDestroyTensor(h.tOut);
    aclrtFree(h.dU);
    aclrtFree(h.dDt);
    aclrtFree(h.dA);
    aclrtFree(h.dB);
    aclrtFree(h.dC);
    aclrtFree(h.dD);
    aclrtFree(h.dOut);
}

void InitAclBenchHandles(const BenchCase& c, const HostBenchBuffers& buf, aclrtStream stream, AclBenchHandles& h)
{
    h.chunkSize = c.chunkSize;
    h.stream = stream;
    h.tU = CreateTensor1D({c.B, c.L, c.H, c.P}, buf.hU.data(), &h.dU);
    h.tDt = CreateTensor1D({c.B, c.L, c.H}, buf.hDt.data(), &h.dDt);
    h.tA = CreateTensor1D({c.H}, buf.hA.data(), &h.dA);
    h.tB = CreateTensor1D({c.B, c.L, c.H, c.N}, buf.hB.data(), &h.dB);
    h.tC = CreateTensor1D({c.B, c.L, c.H, c.N}, buf.hC.data(), &h.dC);
    h.tD = CreateTensor1D({c.H, c.P}, buf.hD.data(), &h.dD);
    h.tOut = CreateTensor1D({c.B, c.L, c.H, c.P}, buf.hOut.data(), &h.dOut);
}

double MeasureLatencyUs(AclBenchHandles& h, int iters)
{
    if (iters <= 0) {
        return 0.0;
    }
    for (int i = 0; i < 5; ++i) {
        RunSingleStep(h);
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        RunSingleStep(h);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsedUs = std::chrono::duration<double, std::micro>(t1 - t0).count();
    return elapsedUs / static_cast<double>(iters);
}

void RunBenchCase(const BenchCase& c, aclrtStream stream)
{
    HostBenchBuffers buf;
    InitHostBenchVectors(c, buf);

    AclBenchHandles h;
    InitAclBenchHandles(c, buf, stream, h);
    const double avgUs = MeasureLatencyUs(h, 20);

    const size_t outBytes = buf.hOut.size() * sizeof(float);
    aclrtMemcpy(buf.hOut.data(), outBytes, h.dOut, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    const float maxDiff = VerifyFirstBatchHead(c, buf);

    std::printf("[BENCH] %-24s | B=%lld L=%4lld H=%2lld P=%3lld N=%3lld | AvgLatency=%8.2f us | MaxDiff(b0,h0)=%.3e\n",
                c.name, static_cast<long long>(c.B), static_cast<long long>(c.L),
                static_cast<long long>(c.H), static_cast<long long>(c.P),
                static_cast<long long>(c.N), avgUs, maxDiff);

    DestroyBenchHandles(h);
}
}  // namespace


int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    const std::vector<BenchCase> cases = {
        {"1_TailGeneric_P16_N16",   2,  96,  4,  16, 16, 64},
        {"2_CubeMix64_B2_L256_H8",  2, 256,  8,  64, 64, 64},
        {"3_CubeMix64_B4_L512_H16", 4, 512, 16,  64, 64, 64},
        {"4_CubeMix128_B2_L256_H8", 2, 256,  8, 128, 64, 64},
    };

    for (const auto& c : cases) {
        RunBenchCase(c, stream);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
