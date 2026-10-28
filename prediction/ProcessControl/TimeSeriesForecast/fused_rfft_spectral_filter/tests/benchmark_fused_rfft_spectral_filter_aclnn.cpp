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
#include <iomanip>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_fused_rfft_spectral_filter.h"

namespace {
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr int64_t kMaxActiveModes = 8;
}  // namespace

struct RfftCase {
    const char* name;
    int64_t b;
    int64_t l;
    int64_t c;
    float eager_ms;
    float torchair_ms;
};

static void InitBenchmarkInputs(
    const RfftCase& cs, std::vector<float>& h_x, std::vector<float>& h_w, std::vector<float>& h_true)
{
    size_t x_elems = static_cast<size_t>(cs.b * cs.l * cs.c);
    for (size_t i = 0; i < x_elems; ++i) {
        h_x[i] = 0.35f * std::sin(0.13f * static_cast<float>(i % 257)) +
                 0.20f * std::cos(0.09f * static_cast<float>(i % 193));
    }
    int64_t active_modes = std::min<int64_t>((cs.l - 1) / 2, kMaxActiveModes);
    for (int64_t f = 0; f <= active_modes; ++f) {
        for (int64_t d = 0; d < cs.c; ++d) {
            size_t w_idx = static_cast<size_t>((f * cs.c + d) * 2);
            h_w[w_idx] = 0.65f * std::cos(0.21f * static_cast<float>(f + d));
            h_w[w_idx + 1] = (f == 0) ? 0.0f : -0.25f * std::sin(0.17f * static_cast<float>(f + d));
        }
    }
    for (int64_t delta = 0; delta < cs.l; ++delta) {
        for (int64_t d = 0; d < cs.c; ++d) {
            float acc = h_w[static_cast<size_t>(d * 2)];
            for (int64_t f = 1; f <= active_modes; ++f) {
                float ang = kTwoPi * static_cast<float>(f * delta) / static_cast<float>(cs.l);
                size_t w_idx = static_cast<size_t>((f * cs.c + d) * 2);
                acc += 2.0f * (h_w[w_idx] * std::cos(ang) - h_w[w_idx + 1] * std::sin(ang));
            }
            h_true[static_cast<size_t>(delta * cs.c + d)] = acc / static_cast<float>(cs.l);
        }
    }
}

static bool RunAclnnOp(aclTensor* t_x, aclTensor* t_w, aclTensor* t_out, aclrtStream stream)
{
    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    auto ret = aclnnFusedRfftSpectralFilterGetWorkspaceSize(t_x, t_w, t_out, &ws, &ex);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] GetWorkspaceSize failed: " << ret << std::endl;
        return false;
    }
    void* wptr = nullptr;
    if (ws > 0) {
        (void)aclrtMalloc(&wptr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnFusedRfftSpectralFilter(wptr, ws, ex, stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] execute failed: " << ret << std::endl;
        if (wptr != nullptr) {
            (void)aclrtFree(wptr);
        }
        return false;
    }
    (void)aclrtSynchronizeStream(stream);
    if (wptr != nullptr) {
        (void)aclrtFree(wptr);
    }
    return true;
}

static void BenchmarkIterations(
    aclTensor* t_x, aclTensor* t_w, aclTensor* t_out, aclrtStream stream, std::vector<double>& samples)
{
    for (int w = 0; w < 5; ++w) {
        if (!RunAclnnOp(t_x, t_w, t_out, stream)) {
            return;
        }
    }
    for (int it = 0; it < 20; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!RunAclnnOp(t_x, t_w, t_out, stream)) {
            return;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
}

static void CheckAccuracy(
    const RfftCase& cs, const std::vector<float>& h_x, const std::vector<float>& h_out,
    const std::vector<float>& h_true, float& max_abs, float& max_rel)
{
    max_abs = 0.0f;
    max_rel = 0.0f;
    for (int64_t t = 0; t < cs.l; ++t) {
        for (int64_t d = 0; d < cs.c; ++d) {
            float ref_val = 0.0f;
            for (int64_t delta = 0; delta < cs.l; ++delta) {
                int64_t tau = (t - delta + cs.l) % cs.l;
                ref_val += h_x[static_cast<size_t>(tau * cs.c + d)] * h_true[static_cast<size_t>(delta * cs.c + d)];
            }
            float got_val = h_out[static_cast<size_t>(t * cs.c + d)];
            float diff = std::fabs(got_val - ref_val);
            float rel = diff / (std::fabs(ref_val) + 1e-5f);
            if (diff > max_abs) max_abs = diff;
            if (rel > max_rel) max_rel = rel;
        }
    }
}

static void PrintCaseResult(
    const RfftCase& cs, const std::vector<double>& samples, float max_abs, float max_rel, bool is_last)
{
    std::vector<double> s = samples;
    std::sort(s.begin(), s.end());
    double sum = 0.0;
    size_t start = s.size() >= 4 ? 2 : 0;
    size_t end = s.size() >= 18 ? 18 : s.size();
    size_t count = end > start ? (end - start) : 1;
    for (size_t i = start; i < end; ++i) {
        sum += s[i];
    }
    double mean_ms = sum / static_cast<double>(count);
    double speedup = std::min(cs.eager_ms, cs.torchair_ms) / mean_ms;

    std::cout << "  {\"name\": \"" << cs.name << "\", \"eager_ms\": " << cs.eager_ms
              << ", \"torchair_ms\": " << cs.torchair_ms
              << ", \"custom_op_ms\": " << std::fixed << std::setprecision(4) << mean_ms
              << ", \"speedup_vs_min\": " << std::setprecision(2) << speedup
              << ", \"max_abs_err\": " << std::scientific << std::setprecision(2) << max_abs
              << ", \"max_rel_err\": " << std::scientific << std::setprecision(2) << max_rel << "}"
              << (is_last ? "" : ",") << std::endl;
}

static void RunSingleBenchmark(const RfftCase& cs, aclrtStream stream, bool is_last)
{
    int64_t freq_bins = cs.l / 2 + 1;
    std::vector<int64_t> x_shape = {cs.b, cs.l, cs.c};
    std::vector<int64_t> x_strides = {cs.l * cs.c, cs.c, 1};
    std::vector<int64_t> w_shape = {freq_bins, cs.c, 2};
    std::vector<int64_t> w_strides = {cs.c * 2, 2, 1};

    size_t x_elems = static_cast<size_t>(cs.b * cs.l * cs.c);
    size_t w_elems = static_cast<size_t>(freq_bins * cs.c * 2);
    std::vector<float> h_x(x_elems, 0.0f), h_w(w_elems, 0.0f), h_out(x_elems, 0.0f);
    std::vector<float> h_true(static_cast<size_t>(cs.l * cs.c), 0.0f);
    InitBenchmarkInputs(cs, h_x, h_w, h_true);

    void *d_x = nullptr, *d_w = nullptr, *d_out = nullptr;
    (void)aclrtMalloc(&d_x, x_elems * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    (void)aclrtMalloc(&d_w, w_elems * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    (void)aclrtMalloc(&d_out, x_elems * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    (void)aclrtMemcpy(
        d_x, x_elems * sizeof(float), h_x.data(), x_elems * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(
        d_w, w_elems * sizeof(float), h_w.data(), w_elems * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    aclTensor* t_x = aclCreateTensor(
        x_shape.data(), 3, ACL_FLOAT, x_strides.data(), 0, ACL_FORMAT_ND, x_shape.data(), 3, d_x);
    aclTensor* t_w = aclCreateTensor(
        w_shape.data(), 3, ACL_FLOAT, w_strides.data(), 0, ACL_FORMAT_ND, w_shape.data(), 3, d_w);
    aclTensor* t_out = aclCreateTensor(
        x_shape.data(), 3, ACL_FLOAT, x_strides.data(), 0, ACL_FORMAT_ND, x_shape.data(), 3, d_out);

    std::vector<double> samples;
    BenchmarkIterations(t_x, t_w, t_out, stream, samples);
    (void)aclrtMemcpy(
        h_out.data(), x_elems * sizeof(float), d_out, x_elems * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    float max_abs = 0.0f, max_rel = 0.0f;
    CheckAccuracy(cs, h_x, h_out, h_true, max_abs, max_rel);
    PrintCaseResult(cs, samples, max_abs, max_rel, is_last);

    (void)aclDestroyTensor(t_x);
    (void)aclDestroyTensor(t_w);
    (void)aclDestroyTensor(t_out);
    (void)aclrtFree(d_x);
    (void)aclrtFree(d_w);
    (void)aclrtFree(d_out);
}

int main(int argc, char** argv)
{
    int32_t deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        return -1;
    }
    aclrtStream stream = nullptr;
    (void)aclrtCreateStream(&stream);

    std::vector<RfftCase> cases = {
        {"TimesNet-Short (32, 96, 32)", 32, 96, 32, 0.4120f, 0.3850f},
        {"FreTS-Medium (32, 192, 64)", 32, 192, 64, 0.8450f, 0.7620f},
        {"FITS-Long (16, 336, 64)", 16, 336, 64, 1.3240f, 1.1890f},
        {"FNO-Spectral (16, 512, 64)", 16, 512, 64, 2.1460f, 1.9420f}
    };

    std::cout << "{\"operator\": \"fused_rfft_spectral_filter\", \"benchmarks\": [" << std::endl;
    for (size_t idx = 0; idx < cases.size(); ++idx) {
        RunSingleBenchmark(cases[idx], stream, idx + 1 == cases.size());
    }
    std::cout << "]}" << std::endl;

    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
    return 0;
}
