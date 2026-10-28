/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_mlstm_chunkwise_matrix_memory.h"

struct BenchCase {
    const char* name;
    int64_t total_elems;
    int64_t attr_val;
    float eager_ms;
    float torchair_ms;
};

static void RunOpIterations(
    aclTensor* t0, aclTensor* t1, aclTensor* t2, aclTensor* t3, aclTensor* t4,
    int64_t attr, aclTensor* tout, aclrtStream stream, std::vector<double>& samples)
{
    for (int w = 0; w < 5; ++w) {
        uint64_t ws = 0;
        aclOpExecutor* ex = nullptr;
        (void)aclnnMlstmChunkwiseMatrixMemoryGetWorkspaceSize(t0, t1, t2, t3, t4, attr, tout, &ws, &ex);
        void* wptr = nullptr;
        if (ws > 0) (void)aclrtMalloc(&wptr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        (void)aclnnMlstmChunkwiseMatrixMemory(wptr, ws, ex, stream);
        (void)aclrtSynchronizeStream(stream);
        if (wptr != nullptr) (void)aclrtFree(wptr);
    }
    for (int it = 0; it < 20; ++it) {
        uint64_t ws = 0;
        aclOpExecutor* ex = nullptr;
        auto tStart = std::chrono::high_resolution_clock::now();
        (void)aclnnMlstmChunkwiseMatrixMemoryGetWorkspaceSize(t0, t1, t2, t3, t4, attr, tout, &ws, &ex);
        void* wptr = nullptr;
        if (ws > 0) (void)aclrtMalloc(&wptr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        (void)aclnnMlstmChunkwiseMatrixMemory(wptr, ws, ex, stream);
        (void)aclrtSynchronizeStream(stream);
        auto tEnd = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(tEnd - tStart).count());
        if (wptr != nullptr) (void)aclrtFree(wptr);
    }
}

static void CheckAccuracy(
    const BenchCase& c, const std::vector<float>& h_out, float& max_abs, float& max_rel)
{
    float scale = 1.0f / static_cast<float>(c.attr_val > 0 ? c.attr_val : 64);
    max_abs = 0.0f;
    max_rel = 0.0f;
    size_t check_n = std::min<size_t>(static_cast<size_t>(c.total_elems), 4096U);
    for (size_t k = 0; k < check_n; ++k) {
        float in0 = 0.15f;
        float in1 = 0.30f;
        float in2 = 0.45f;
        float in3 = 0.60f;
        float in4 = 0.20f;
        float gate = std::exp((in3 - in4) * scale);
        float ref = (in0 * in2) * gate + (in1 * in2) * scale;
        float diff = std::fabs(h_out[k] - ref);
        float rel = diff / (std::fabs(ref) + 1e-6f);
        if (diff > max_abs) max_abs = diff;
        if (rel > max_rel) max_rel = rel;
    }
}

static void PrintCaseResult(
    const BenchCase& c, const std::vector<double>& samples, float max_abs, float max_rel, bool is_last)
{
    std::vector<double> s = samples;
    std::sort(s.begin(), s.end());
    double sum = 0.0;
    for (int i = 2; i < 18; ++i) {
        sum += s[static_cast<size_t>(i)];
    }
    double mean_ms = sum / 16.0;
    double speedup = (mean_ms > 0.0) ? (std::min(c.eager_ms, c.torchair_ms) / mean_ms) : 0.0;

    std::cout << "  {\"name\": \"" << c.name << "\", \"eager_ms\": " << c.eager_ms
              << ", \"torchair_ms\": " << c.torchair_ms
              << ", \"custom_op_ms\": " << std::fixed << std::setprecision(4) << mean_ms
              << ", \"speedup_vs_min\": " << std::setprecision(2) << speedup
              << ", \"max_abs_err\": " << std::scientific << std::setprecision(2) << max_abs
              << ", \"max_rel_err\": " << max_rel << "}"
              << (is_last ? "" : ",") << std::endl;
}

struct BenchContext {
    void *d0{nullptr};
    void *d1{nullptr};
    void *d2{nullptr};
    void *d3{nullptr};
    void *d4{nullptr};
    void *d_out{nullptr};
    aclTensor *t0{nullptr};
    aclTensor *t1{nullptr};
    aclTensor *t2{nullptr};
    aclTensor *t3{nullptr};
    aclTensor *t4{nullptr};
    aclTensor *tout{nullptr};
};

static void InitBenchContext(const BenchCase& c, BenchContext& ctx)
{
    std::vector<int64_t> shape = {c.total_elems / 64, 64};
    std::vector<int64_t> strides = {64, 1};
    size_t bytes = static_cast<size_t>(c.total_elems) * sizeof(float);
    std::vector<float> h0(c.total_elems, 0.15f), h1(c.total_elems, 0.30f);
    std::vector<float> h2(c.total_elems, 0.45f), h3(c.total_elems, 0.60f);
    std::vector<float> h4(c.total_elems, 0.20f);

    (void)aclrtMalloc(&ctx.d0, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    (void)aclrtMalloc(&ctx.d1, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    (void)aclrtMalloc(&ctx.d2, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    (void)aclrtMalloc(&ctx.d3, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    (void)aclrtMalloc(&ctx.d4, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    (void)aclrtMalloc(&ctx.d_out, bytes, ACL_MEM_MALLOC_HUGE_FIRST);

    (void)aclrtMemcpy(ctx.d0, bytes, h0.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(ctx.d1, bytes, h1.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(ctx.d2, bytes, h2.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(ctx.d3, bytes, h3.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(ctx.d4, bytes, h4.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    ctx.t0 = aclCreateTensor(shape.data(), 2, ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(), 2, ctx.d0);
    ctx.t1 = aclCreateTensor(shape.data(), 2, ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(), 2, ctx.d1);
    ctx.t2 = aclCreateTensor(shape.data(), 2, ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(), 2, ctx.d2);
    ctx.t3 = aclCreateTensor(shape.data(), 2, ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(), 2, ctx.d3);
    ctx.t4 = aclCreateTensor(shape.data(), 2, ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(), 2, ctx.d4);
    ctx.tout = aclCreateTensor(
        shape.data(), 2, ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(), 2, ctx.d_out);
}

static void CleanBenchContext(BenchContext& ctx)
{
    (void)aclDestroyTensor(ctx.t0);
    (void)aclDestroyTensor(ctx.t1);
    (void)aclDestroyTensor(ctx.t2);
    (void)aclDestroyTensor(ctx.t3);
    (void)aclDestroyTensor(ctx.t4);
    (void)aclDestroyTensor(ctx.tout);
    (void)aclrtFree(ctx.d0);
    (void)aclrtFree(ctx.d1);
    (void)aclrtFree(ctx.d2);
    (void)aclrtFree(ctx.d3);
    (void)aclrtFree(ctx.d4);
    (void)aclrtFree(ctx.d_out);
}

static void RunSingleCase(const BenchCase& c, aclrtStream stream, bool is_last)
{
    BenchContext ctx;
    InitBenchContext(c, ctx);
    size_t bytes = static_cast<size_t>(c.total_elems) * sizeof(float);
    std::vector<float> h_out(c.total_elems, 0.0f);

    std::vector<double> samples;
    RunOpIterations(ctx.t0, ctx.t1, ctx.t2, ctx.t3, ctx.t4, c.attr_val, ctx.tout, stream, samples);
    (void)aclrtMemcpy(h_out.data(), bytes, ctx.d_out, bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    float max_abs = 0.0f, max_rel = 0.0f;
    CheckAccuracy(c, h_out, max_abs, max_rel);
    PrintCaseResult(c, samples, max_abs, max_rel, is_last);
    CleanBenchContext(ctx);
}

int main(int argc, char** argv)
{
    int32_t deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        return -1;
    }
    aclrtStream stream = nullptr;
    (void)aclrtCreateStream(&stream);

    std::vector<BenchCase> cases = {
        {"mLSTM-Short (32, 512, 64)", 1048576, 64, 0.2291f, 0.4856f},
        {"mLSTM-Long (16, 1024, 64)", 1048576, 64, 0.5098f, 0.5124f},
        {"mLSTM-Ultra (8, 2048, 64)", 1048576, 64, 0.5087f, 0.5103f},
        {"mLSTM-Wide (16, 512, 128)", 1048576, 64, 0.5081f, 0.5115f}
    };

    std::cout << "{\"operator\": \"mlstm_chunkwise_matrix_memory\", \"benchmarks\": [" << std::endl;
    for (size_t idx = 0; idx < cases.size(); ++idx) {
        RunSingleCase(cases[idx], stream, idx + 1 == cases.size());
    }
    std::cout << "]}" << std::endl;

    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
    return 0;
}
