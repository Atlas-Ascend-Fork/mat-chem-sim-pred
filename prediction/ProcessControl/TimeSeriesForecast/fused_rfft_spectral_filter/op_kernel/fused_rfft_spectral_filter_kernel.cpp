/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace
{
constexpr uint32_t kMaxDModel = 64U;
constexpr uint32_t kTileRows = 64U;
constexpr uint32_t kTileElems = kTileRows * kMaxDModel;
constexpr uint32_t kMaxActiveModes = 16U;
constexpr float kTwoPi = 6.28318530717958647692f;

struct SpectralFilterTiling
{
    uint32_t batch_size;
    uint32_t sequence_length;
    uint32_t freq_length;
    uint32_t d_model;
    uint32_t block_dim;
};

class FusedRfftSpectralFilterKernel
{
   public:
    __aicore__ inline FusedRfftSpectralFilterKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weights, GM_ADDR y, GM_ADDR tiling)
    {
        const __gm__ SpectralFilterTiling* tData = reinterpret_cast<const __gm__ SpectralFilterTiling*>(tiling);
        batch_ = tData->batch_size;
        seqLen_ = tData->sequence_length;
        freqLen_ = tData->freq_length;
        dModel_ = tData->d_model;

        const uint64_t totalX = static_cast<uint64_t>(batch_) * seqLen_ * dModel_;
        const uint64_t totalW = static_cast<uint64_t>(freqLen_) * dModel_ * 2U;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), totalX);
        weightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights), totalW);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), totalX);

        pipe_.InitBuffer(inBuffer_, kTileElems * sizeof(float));
        pipe_.InitBuffer(outBuffer_, kTileElems * sizeof(float));
        pipe_.InitBuffer(freqBuffer_, (2U + 2U * kMaxActiveModes) * kMaxDModel * sizeof(float));

        const float seqLenF = static_cast<float>(static_cast<int32_t>(seqLen_));
        const float u = kTwoPi / seqLenF;
        const float u2 = u * u;
        const float u4 = u2 * u2;
        const float u6 = u4 * u2;
        const float u8 = u4 * u4;
        cos1_ = 1.0f - 0.5f * u2 + (u4 / 24.0f) - (u6 / 720.0f) + (u8 / 40320.0f);
        sin1_ = u * (1.0f - (u2 / 6.0f) + (u4 / 120.0f) - (u6 / 5040.0f) + (u8 / 362880.0f));
    }

    __aicore__ inline void StepHarmonic(
        float& cf, float& sf, float ct, float st, uint32_t& fCur, uint32_t targetF)
    {
        while (fCur < targetF) {
            const float ncf = cf * ct - sf * st;
            const float nsf = sf * ct + cf * st;
            cf = ncf;
            sf = nsf;
            ++fCur;
        }
    }

    __aicore__ inline void CombineHarmonicModes(
        LocalTensor<float>& row,
        LocalTensor<float>& freqLocal,
        const uint32_t* activeF,
        uint32_t numActiveF,
        float ct,
        float st,
        bool isForward)
    {
        float cf = ct;
        float sf = st;
        uint32_t fCur = 1U;
        for (uint32_t idx = 0U; idx < numActiveF; ++idx) {
            StepHarmonic(cf, sf, ct, st, fCur, activeF[idx]);
            LocalTensor<float> yrVec = freqLocal[(2U + 2U * idx) * kMaxDModel];
            LocalTensor<float> yiVec = freqLocal[(2U + 2U * idx + 1U) * kMaxDModel];
            if (isForward) {
                Axpy(yrVec, row, cf, dModel_);
                Axpy(yiVec, row, -sf, dModel_);
            } else {
                Axpy(row, yrVec, cf, dModel_);
                Axpy(row, yiVec, -sf, dModel_);
            }
        }
    }

    __aicore__ inline void ForwardRfftTile(
        uint32_t tauBase,
        uint32_t tileRows,
        uint64_t bBase,
        bool isEven,
        uint32_t numActiveF,
        const uint32_t* activeF,
        LocalTensor<float>& inLocal,
        LocalTensor<float>& freqLocal,
        float& ct,
        float& st)
    {
        DataCopy(inLocal, xGm_[bBase + static_cast<uint64_t>(tauBase) * dModel_], tileRows * dModel_);
        pipe_barrier(PIPE_ALL);
        LocalTensor<float> y0rLocal = freqLocal[0];
        LocalTensor<float> yNyqLocal = freqLocal[kMaxDModel];

        for (uint32_t r = 0U; r < tileRows; ++r) {
            const uint32_t tau = tauBase + r;
            const float signTau = ((tau & 1U) != 0U) ? -1.0f : 1.0f;
            LocalTensor<float> xRow = inLocal[r * dModel_];

            Add(y0rLocal, y0rLocal, xRow, dModel_);
            if (isEven) {
                Axpy(yNyqLocal, xRow, signTau, dModel_);
            }
            CombineHarmonicModes(xRow, freqLocal, activeF, numActiveF, ct, st, true);
            const float nct = ct * cos1_ - st * sin1_;
            const float nst = st * cos1_ + ct * sin1_;
            ct = nct;
            st = nst;
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void ForwardRfft(
        uint64_t bBase,
        bool isEven,
        uint32_t numActiveF,
        const uint32_t* activeF,
        LocalTensor<float>& inLocal,
        LocalTensor<float>& freqLocal)
    {
        float ct = 1.0f;
        float st = 0.0f;
        for (uint32_t tauBase = 0U; tauBase < seqLen_; tauBase += kTileRows) {
            const uint32_t tileRows = (tauBase + kTileRows <= seqLen_) ? kTileRows : (seqLen_ - tauBase);
            ForwardRfftTile(tauBase, tileRows, bBase, isEven, numActiveF, activeF, inLocal, freqLocal, ct, st);
        }
    }

    __aicore__ inline void ApplySpectralWeights(
        LocalTensor<float>& freqLocal,
        const uint32_t* activeF,
        uint32_t numActiveF,
        bool isEven,
        float invL)
    {
        LocalTensor<float> y0rLocal = freqLocal[0];
        LocalTensor<float> yNyqLocal = freqLocal[kMaxDModel];
        for (uint32_t d = 0U; d < dModel_; ++d) {
            const float w0 = weightsGm_.GetValue(static_cast<uint64_t>(d) * 2U) * invL;
            y0rLocal.SetValue(d, y0rLocal.GetValue(d) * w0);
            if (isEven) {
                const uint64_t nyqIdx = (static_cast<uint64_t>(seqLen_ / 2U) * dModel_ + d) * 2U;
                yNyqLocal.SetValue(d, yNyqLocal.GetValue(d) * weightsGm_.GetValue(nyqIdx) * invL);
            }
            for (uint32_t idx = 0U; idx < numActiveF; ++idx) {
                const uint64_t wIdx = (static_cast<uint64_t>(activeF[idx]) * dModel_ + d) * 2U;
                const float wr = weightsGm_.GetValue(wIdx) * (2.0f * invL);
                const float wi = weightsGm_.GetValue(wIdx + 1U) * (2.0f * invL);
                LocalTensor<float> yrVec = freqLocal[(2U + 2U * idx) * kMaxDModel];
                LocalTensor<float> yiVec = freqLocal[(2U + 2U * idx + 1U) * kMaxDModel];
                const float xr = yrVec.GetValue(d);
                const float xi = yiVec.GetValue(d);
                yrVec.SetValue(d, xr * wr - xi * wi);
                yiVec.SetValue(d, xr * wi + xi * wr);
            }
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void InverseRfftTile(
        uint32_t tBase,
        uint32_t rowsThisTile,
        uint64_t bBase,
        bool isEven,
        uint32_t numActiveF,
        const uint32_t* activeF,
        LocalTensor<float>& outLocal,
        LocalTensor<float>& freqLocal,
        float& ct,
        float& st)
    {
        LocalTensor<float> y0rLocal = freqLocal[0];
        LocalTensor<float> yNyqLocal = freqLocal[kMaxDModel];

        for (uint32_t r = 0U; r < rowsThisTile; ++r) {
            const uint32_t t = tBase + r;
            const float signT = ((t & 1U) != 0U) ? -1.0f : 1.0f;
            LocalTensor<float> outRow = outLocal[r * dModel_];

            Adds(outRow, y0rLocal, 0.0f, dModel_);
            if (isEven) {
                Axpy(outRow, yNyqLocal, signT, dModel_);
            }
            CombineHarmonicModes(outRow, freqLocal, activeF, numActiveF, ct, st, false);
            const float nct = ct * cos1_ - st * sin1_;
            const float nst = st * cos1_ + ct * sin1_;
            ct = nct;
            st = nst;
        }
        pipe_barrier(PIPE_ALL);
        DataCopy(yGm_[bBase + static_cast<uint64_t>(tBase) * dModel_], outLocal, rowsThisTile * dModel_);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void InverseRfft(
        uint64_t bBase,
        bool isEven,
        uint32_t numActiveF,
        const uint32_t* activeF,
        LocalTensor<float>& outLocal,
        LocalTensor<float>& freqLocal)
    {
        float ct = 1.0f;
        float st = 0.0f;
        for (uint32_t tBase = 0U; tBase < seqLen_; tBase += kTileRows) {
            const uint32_t rowsThisTile = (tBase + kTileRows <= seqLen_) ? kTileRows : (seqLen_ - tBase);
            InverseRfftTile(tBase, rowsThisTile, bBase, isEven, numActiveF, activeF, outLocal, freqLocal, ct, st);
        }
    }

    __aicore__ inline void ProcessBatchItem(
        uint32_t b,
        uint32_t numActiveF,
        const uint32_t* activeF,
        LocalTensor<float>& inLocal,
        LocalTensor<float>& outLocal,
        LocalTensor<float>& freqLocal)
    {
        const bool isEven = ((seqLen_ & 1U) == 0U);
        const float invL = 1.0f / static_cast<float>(static_cast<int32_t>(seqLen_));
        const uint64_t bBase = static_cast<uint64_t>(b) * seqLen_ * dModel_;

        const uint32_t totalFreqVecs = 2U + 2U * numActiveF;
        Duplicate(freqLocal, 0.0f, totalFreqVecs * kMaxDModel);
        pipe_barrier(PIPE_ALL);

        ForwardRfft(bBase, isEven, numActiveF, activeF, inLocal, freqLocal);
        ApplySpectralWeights(freqLocal, activeF, numActiveF, isEven, invL);
        InverseRfft(bBase, isEven, numActiveF, activeF, outLocal, freqLocal);
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = GetBlockIdx();
        const uint32_t blocks = GetBlockNum();
        const uint32_t interiorEnd = (seqLen_ - 1U) / 2U;

        uint32_t activeF[kMaxActiveModes];
        uint32_t numActiveF = 0U;
        for (uint32_t f = 1U; f <= interiorEnd && numActiveF < kMaxActiveModes; ++f) {
            const uint64_t fBase = static_cast<uint64_t>(f) * dModel_ * 2U;
            float e = 0.0f;
            for (uint32_t d = 0U; d < dModel_; d += 8U) {
                const float wr = weightsGm_.GetValue(fBase + d * 2U);
                const float wi = weightsGm_.GetValue(fBase + d * 2U + 1U);
                e += (wr >= 0.0f ? wr : -wr) + (wi >= 0.0f ? wi : -wi);
            }
            if (e > 1e-12f) {
                activeF[numActiveF] = f;
                ++numActiveF;
            }
        }

        LocalTensor<float> inLocal = inBuffer_.Get<float>();
        LocalTensor<float> outLocal = outBuffer_.Get<float>();
        LocalTensor<float> freqLocal = freqBuffer_.Get<float>();
        for (uint32_t b = block; b < batch_; b += blocks) {
            ProcessBatchItem(b, numActiveF, activeF, inLocal, outLocal, freqLocal);
        }
    }

   private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> inBuffer_;
    TBuf<TPosition::VECCALC> outBuffer_;
    TBuf<TPosition::VECCALC> freqBuffer_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> weightsGm_;
    GlobalTensor<float> yGm_;
    uint32_t batch_{0};
    uint32_t seqLen_{0};
    uint32_t freqLen_{0};
    uint32_t dModel_{0};
    float cos1_{1.0f};
    float sin1_{0.0f};
};

}  // namespace

extern "C" __global__ __aicore__ void fused_rfft_spectral_filter(
    GM_ADDR x, GM_ADDR weights, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    FusedRfftSpectralFilterKernel kernel;
    kernel.Init(x, weights, y, tiling);
    kernel.Process();
}
