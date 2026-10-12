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
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace
{
constexpr uint32_t kChunkSize = 64U;
constexpr uint32_t kChunkStepsFallback = 16U;
constexpr uint32_t kMaxP = 64U;
constexpr uint32_t kMaxN = 128U;
constexpr uint32_t kAccRows = 8U;
constexpr uint32_t kMaxDtRawElems = 576U;
constexpr uint32_t kExpBufElems = 64U;
constexpr uint32_t kMatElems = 4096U;
constexpr uint32_t kPerCoreWsFloats = 57344U;

struct Mamba2SsdTiling
{
    uint32_t batch_size;
    uint32_t seq_len;
    uint32_t nheads;
    uint32_t headdim;
    uint32_t dstate;
    uint32_t chunk_size;
    uint32_t nchunks;
    uint32_t block_dim;
};

using MatmulTypeA = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, true>;
using MatmulTypeB = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, true>;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
using MatmulTypeC = matmul::MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, float>;
#else
using MatmulTypeC = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
#endif
using MatmulTypeBias = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
using CubeMatmulObj = matmul::Matmul<MatmulTypeA, MatmulTypeB, MatmulTypeC, MatmulTypeBias>;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
__aicore__ inline void Transpose64x64UB(const LocalTensor<float>& dst, const LocalTensor<float>& src)
{
    TransDataTo5HDParams transParams;
    transParams.dstHighHalf = false;
    transParams.srcHighHalf = false;
    transParams.repeatTimes = 8U;
    transParams.srcRepStride = 1U;
    transParams.dstRepStride = 64U;

    uint64_t dstLocalList[16];
    uint64_t srcLocalList[16];
    for (uint32_t s = 0; s < 4U; ++s) {
        const uint32_t rowBase = s * 16U;
        for (uint32_t i = 0; i < 16U; ++i) {
            srcLocalList[i] = reinterpret_cast<uint64_t>(src[(rowBase + i) * 64U].GetPhyAddr());
            const uint32_t dstOffset = rowBase + (i >> 1U) * 64U + (i & 1U) * 8U;
            dstLocalList[i] = reinterpret_cast<uint64_t>(dst[dstOffset].GetPhyAddr());
        }
        TransDataTo5HD<float>(dstLocalList, srcLocalList, transParams);
    }
}

__simd_vf__ inline void PrefixSumAlpha64VF(
    __ubuf__ float* alphaPtr, float aVal)
{
    Reg::RegTensor<float> rDt, rCum, rShift, rNeg;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    Reg::LoadAlign(rDt, alphaPtr + 128U);
    Reg::Muls(rCum, rDt, aVal, fullMask);
    Reg::StoreAlign(alphaPtr + 256U, rCum, fullMask);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

    for (uint32_t shift = 1U; shift <= 16U; shift <<= 1U) {
        Reg::Load(rShift, alphaPtr + 256U - shift);
        Reg::Add(rCum, rCum, rShift, fullMask);
        Reg::StoreAlign(alphaPtr + 256U, rCum, fullMask);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }

    Reg::Load(rShift, alphaPtr + 256U - 32U);
    Reg::Add(rCum, rCum, rShift, fullMask);
    Reg::StoreAlign(alphaPtr, rCum, fullMask);

    Reg::Muls(rNeg, rCum, -1.0f, fullMask);
    Reg::StoreAlign(alphaPtr + 64U, rNeg, fullMask);
}

__simd_vf__ inline void FuseMaskAndInterStrideVF(
    __ubuf__ float* mPtr, __ubuf__ float* outPtr, uint32_t outRowStride,
    __ubuf__ float* yInterRawPtr, __ubuf__ float* alphaPtr)
{
    Reg::RegTensor<float> rNegDt, rExpCumI, rL, rM, rYInter;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::LoadAlign(rNegDt, alphaPtr + 64U);
    Reg::Sub(rL, rNegDt, rNegDt, fullMask);
    for (uint16_t i = 0; i < 64U; ++i) {
        uint32_t validCols = static_cast<uint32_t>(i) + 1U;
        Reg::MaskReg rowMask = Reg::UpdateMask<float>(validCols);
        const uint32_t rowOffset = static_cast<uint32_t>(i) * 64U;

        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rExpCumI, alphaPtr + i);
        Reg::Mul(rL, rNegDt, rExpCumI, rowMask);
        Reg::LoadAlign(rM, mPtr + rowOffset);
        Reg::Mul(rL, rM, rL, rowMask);
        Reg::StoreAlign(mPtr + rowOffset, rL, fullMask);

        Reg::LoadAlign(rYInter, yInterRawPtr + rowOffset);
        Reg::Mul(rYInter, rYInter, rExpCumI, fullMask);
        Reg::StoreAlign(outPtr + static_cast<uint32_t>(i) * outRowStride, rYInter, fullMask);
    }
}

__aicore__ inline void FuseMaskAndInterNoAddVF(
    __ubuf__ float* mPtr, __ubuf__ float* outPtr,
    __ubuf__ float* yInterRawPtr, __ubuf__ float* alphaPtr)
{
    FuseMaskAndInterStrideVF(mPtr, outPtr, 64U, yInterRawPtr, alphaPtr);
}

__simd_vf__ inline void UpdateStateChunk64VF(
    __ubuf__ float* statePtr, __ubuf__ float* lPtr, __ubuf__ float* alphaPtr)
{
    Reg::RegTensor<float> rState, rL, rExpLast;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rExpLast, alphaPtr + 63U);
    for (uint16_t i = 0; i < 64U; ++i) {
        const uint32_t rowOffset = static_cast<uint32_t>(i) * 64U;
        Reg::LoadAlign(rState, statePtr + rowOffset);
        Reg::LoadAlign(rL, lPtr + rowOffset);
        Reg::Mul(rState, rState, rExpLast, fullMask);
        Reg::Add(rState, rState, rL, fullMask);
        Reg::StoreAlign(statePtr + rowOffset, rState, fullMask);
    }
}

__simd_vf__ inline void AccumOutAndUDStrideVF(
    __ubuf__ float* outPtr, uint32_t outRowStride, __ubuf__ float* workPtr,
    __ubuf__ float* uPtr, __ubuf__ float* dPtr)
{
    Reg::RegTensor<float> rOut, rWork, rU, rD;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::LoadAlign(rD, dPtr);
    for (uint16_t i = 0; i < 64U; ++i) {
        const uint32_t rowOffset = static_cast<uint32_t>(i) * 64U;
        Reg::LoadAlign(rOut, outPtr + static_cast<uint32_t>(i) * outRowStride);
        Reg::LoadAlign(rWork, workPtr + rowOffset);
        Reg::LoadAlign(rU, uPtr + rowOffset);
        Reg::Mul(rU, rU, rD, fullMask);
        Reg::Add(rOut, rOut, rWork, fullMask);
        Reg::Add(rOut, rOut, rU, fullMask);
        Reg::StoreAlign(outPtr + static_cast<uint32_t>(i) * outRowStride, rOut, fullMask);
    }
}

__simd_vf__ inline void AddWorkToOut64VF(
    __ubuf__ float* outPtr, __ubuf__ float* workPtr)
{
    Reg::RegTensor<float> rOut, rWork;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint16_t i = 0; i < 64U; ++i) {
        const uint32_t rowOffset = static_cast<uint32_t>(i) * 64U;
        Reg::LoadAlign(rOut, outPtr + rowOffset);
        Reg::LoadAlign(rWork, workPtr + rowOffset);
        Reg::Add(rOut, rOut, rWork, fullMask);
        Reg::StoreAlign(outPtr + rowOffset, rOut, fullMask);
    }
}

__aicore__ inline void UpdateStateAndOutVF(
    __ubuf__ float* statePtr, __ubuf__ float* lPtr,
    __ubuf__ float* outPtr, __ubuf__ float* workPtr,
    __ubuf__ float* alphaPtr)
{
    UpdateStateChunk64VF(statePtr, lPtr, alphaPtr);
    AddWorkToOut64VF(outPtr, workPtr);
}

__simd_vf__ inline void ScaleBChunk64OnlyVF(
    __ubuf__ float* bPtr, __ubuf__ float* alphaPtr)
{
    Reg::RegTensor<float> rC, rB, rScaleBI;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    Reg::LoadAlign(rC, alphaPtr + 64U);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rB, alphaPtr + 63U);
    Reg::Mul(rC, rC, rB, fullMask);
    Reg::StoreAlign(alphaPtr + 128U, rC, fullMask);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < 64U; ++i) {
        const uint32_t rowOffset = static_cast<uint32_t>(i) * 64U;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rScaleBI, alphaPtr + 128U + i);
        Reg::LoadAlign(rB, bPtr + rowOffset);
        Reg::Mul(rB, rB, rScaleBI, fullMask);
        Reg::StoreAlign(bPtr + rowOffset, rB, fullMask);
    }
}

__simd_vf__ inline void InitOutWithUD64VF(
    __ubuf__ float* uPtr, __ubuf__ float* dPtr, __ubuf__ float* outPtr)
{
    Reg::RegTensor<float> rU, rD;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::LoadAlign(rD, dPtr);
    for (uint16_t i = 0; i < 64U; ++i) {
        const uint32_t rowOffset = static_cast<uint32_t>(i) * 64U;
        Reg::LoadAlign(rU, uPtr + rowOffset);
        Reg::Mul(rU, rU, rD, fullMask);
        Reg::StoreAlign(outPtr + rowOffset, rU, fullMask);
    }
}

__aicore__ inline void ScaleBOutChunk64VF(
    __ubuf__ float* bPtr, __ubuf__ float* uPtr,
    __ubuf__ float* dPtr, __ubuf__ float* outPtr,
    __ubuf__ float* alphaPtr)
{
    ScaleBChunk64OnlyVF(bPtr, alphaPtr);
    InitOutWithUD64VF(uPtr, dPtr, outPtr);
}

__simd_vf__ inline void AddScaledYInterToOutVF(
    __ubuf__ float* outPtr, __ubuf__ float* yInterRawPtr, __ubuf__ float* alphaPtr)
{
    Reg::RegTensor<float> rExpCumI, rOut, rYInter;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint16_t i = 0; i < 64U; ++i) {
        const uint32_t rowOffset = static_cast<uint32_t>(i) * 64U;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rExpCumI, alphaPtr + i);
        Reg::LoadAlign(rYInter, yInterRawPtr + rowOffset);
        Reg::Mul(rYInter, rYInter, rExpCumI, fullMask);
        Reg::LoadAlign(rOut, outPtr + rowOffset);
        Reg::Add(rOut, rOut, rYInter, fullMask);
        Reg::StoreAlign(outPtr + rowOffset, rOut, fullMask);
    }
}

__aicore__ inline void FuseMaskAndInterVF(
    __ubuf__ float* mPtr, __ubuf__ float* outPtr,
    __ubuf__ float* yInterRawPtr, __ubuf__ float* alphaPtr)
{
    FuseMaskAndInterStrideVF(mPtr, yInterRawPtr, 64U, yInterRawPtr, alphaPtr);
    AddScaledYInterToOutVF(outPtr, yInterRawPtr, alphaPtr);
}
#endif

__aicore__ inline void SetDefaultCubeTilingPipeline(TCubeTiling& t, int32_t mVal, int32_t l1Size, int32_t l0cSize)
{
    int32_t* p = reinterpret_cast<int32_t*>(&t);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(int32_t); ++i) {
        p[i] = 0;
    }
    t.usedCoreNum = 1;
    t.M = mVal;
    t.N = 64;
    t.Ka = 64;
    t.Kb = 64;
    t.singleCoreM = mVal;
    t.singleCoreN = 64;
    t.singleCoreK = 64;
    t.baseM = mVal;
    t.baseN = 64;
    t.baseK = 64;
    t.depthA1 = 1;
    t.depthB1 = 1;
    t.stepM = 1;
    t.stepN = 1;
    t.stepKa = 1;
    t.stepKb = 1;
    t.dbL0A = 2;
    t.dbL0B = 2;
    t.dbL0C = 1;
    t.shareL1Size = l1Size;
    t.shareL0CSize = l0cSize;
    t.batchM = 1;
    t.batchN = 1;
    t.singleBatchM = 1;
    t.singleBatchN = 1;
}

__aicore__ inline void FillLocalCubeTiling(TCubeTiling& t)
{
    SetDefaultCubeTilingPipeline(t, 64, 32768, 16384);
}

__aicore__ inline void FillLocalCubeTiling128x64(TCubeTiling& t)
{
    SetDefaultCubeTilingPipeline(t, 128, 65536, 32768);
}

struct KernelGmAddrs {
    GM_ADDR u;
    GM_ADDR dt;
    GM_ADDR a;
    GM_ADDR b;
    GM_ADDR c;
    GM_ADDR d;
    GM_ADDR out;
    GM_ADDR userWs;
    GM_ADDR tiling;
};

struct SeqHeadContext {
    uint32_t b{0};
    uint32_t h{0};
    float aVal{0.0f};
    uint32_t dtRem{0};
    LocalTensor<float> stateLocal;
    LocalTensor<float> uLocal;
    LocalTensor<float> bLocal;
    LocalTensor<float> cLocal;
    LocalTensor<float> outLocal;
    LocalTensor<float> workLocal;
    LocalTensor<float> lLocal;
    LocalTensor<float> dLocal;
    LocalTensor<float> dtRawLocal;
    LocalTensor<float> alphaLocal;
};

struct InterleavedContext {
    uint32_t bArr[3]{0};
    uint32_t hArr[3]{0};
    float aVal[3]{0.0f};
    uint32_t K{0};
    bool hasPair01{false};
    bool outWritePending{false};
    LocalTensor<float> stateAll;
    LocalTensor<float> outAll;
    LocalTensor<float> outPairLocal;
    LocalTensor<float> workAll;
    LocalTensor<float> uLocal;
    LocalTensor<float> bLocal;
    LocalTensor<float> cLocal;
    LocalTensor<float> alphaAll;
    LocalTensor<float> dAll;
    LocalTensor<float> dtRawLocal;
    GlobalTensor<float> gmWsA0[3];
    GlobalTensor<float> gmWsA1[3];
    GlobalTensor<float> gmWsB0[3];
    GlobalTensor<float> gmWsB1[3];
};

struct AicDirectContext {
    uint32_t aiv0{0};
    uint32_t aiv1{0};
    uint32_t totalAivCores{0};
    int32_t strideBC{0};
    int32_t strideU{0};
    bool isCoalesced128{false};
    bool inited0{false};
    bool inited1{false};
    GlobalTensor<float> gmWsA0_0[3];
    GlobalTensor<float> gmWsB0_0[3];
    GlobalTensor<float> gmWsA0_1[3];
    GlobalTensor<float> gmWsB0_1[3];
};

struct AicSubTask {
    uint32_t subId{0};
    uint32_t off{0};
    event_t ev{EVENT_ID3};
    bool transB{false};
    uint64_t dstUbAddr{0};
};

struct StepSlotInfo {
    uint32_t b{0};
    uint32_t h{0};
    uint32_t t0{0};
    bool loadDt{true};
};

struct AicMatmulCall {
    AicSubTask task;
    GlobalTensor<float> gmA;
    GlobalTensor<float> gmB;
    uint32_t loadBL1Width{0};
    int32_t strideB{0};
};

class KernelMamba2SsdChunkScan
{
public:
    __aicore__ inline KernelMamba2SsdChunkScan() = default;

    __aicore__ inline void InitGlobalBuffers(const KernelGmAddrs& addrs)
    {
        const __gm__ Mamba2SsdTiling* tData = reinterpret_cast<const __gm__ Mamba2SsdTiling*>(addrs.tiling);
        batch_ = tData->batch_size;
        seqLen_ = tData->seq_len;
        nheads_ = tData->nheads;
        headdim_ = tData->headdim;
        dstate_ = tData->dstate;

        const uint64_t totalU = static_cast<uint64_t>(batch_) * seqLen_ * nheads_ * headdim_;
        const uint64_t totalDt = static_cast<uint64_t>(batch_) * seqLen_ * nheads_;
        const uint64_t totalBC = static_cast<uint64_t>(batch_) * seqLen_ * nheads_ * dstate_;

        coreId_ = static_cast<uint32_t>(GetBlockIdx());
        coreNum_ = static_cast<uint32_t>(GetBlockNum() * GetTaskRation());

        gmU_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(addrs.u), totalU);
        gmDt_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(addrs.dt), totalDt);
        gmA_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(addrs.a), nheads_);
        gmB_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(addrs.b), totalBC);
        gmC_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(addrs.c), totalBC);
        gmD_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(addrs.d), static_cast<uint64_t>(nheads_) * headdim_);
        gmOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(addrs.out), totalU);

        InitCoreWorkspaceBuffers(addrs.userWs);
    }

    __aicore__ inline void InitCoreWorkspaceBuffers(GM_ADDR userWs)
    {
        __gm__ float* coreWs = reinterpret_cast<__gm__ float*>(userWs) + static_cast<uint64_t>(coreId_) * kPerCoreWsFloats;
        gmWsA0_.SetGlobalBuffer(coreWs, kMatElems);
        gmWsA1_.SetGlobalBuffer(coreWs + 1U * kMatElems, kMatElems);
        gmWsB0_.SetGlobalBuffer(coreWs + 2U * kMatElems, kMatElems);
        gmWsB1_.SetGlobalBuffer(coreWs + 3U * kMatElems, kMatElems);
        gmWsB2_.SetGlobalBuffer(coreWs + 4U * kMatElems, kMatElems);
        gmWsC0_.SetGlobalBuffer(coreWs + 5U * kMatElems, kMatElems);
        gmWsC1_.SetGlobalBuffer(coreWs + 6U * kMatElems, kMatElems);
        gmWsU_.SetGlobalBuffer(coreWs + 7U * kMatElems, kMatElems);
        gmWsTril_.SetGlobalBuffer(coreWs + 8U * kMatElems, kMatElems);
        gmWsCdec0_.SetGlobalBuffer(coreWs + 9U * kMatElems, kMatElems);
        gmWsCdec1_.SetGlobalBuffer(coreWs + 10U * kMatElems, kMatElems);
        gmWsBdec0_.SetGlobalBuffer(coreWs + 11U * kMatElems, kMatElems);
        gmWsBdec1_.SetGlobalBuffer(coreWs + 12U * kMatElems, kMatElems);
        gmWsState1_.SetGlobalBuffer(coreWs + 13U * kMatElems, kMatElems);
    }

    __aicore__ inline void InitPipeBuffers()
    {
        const uint32_t maxN = (dstate_ == 128U) ? 128U : 64U;
        const bool isInterleaved64 = (headdim_ == 64U) && (dstate_ == 64U) && ((seqLen_ % 64U) == 0U);
        if (isInterleaved64) {
            pipe_->InitBuffer(stateBuf_, 3U * kMatElems * sizeof(float));
            pipe_->InitBuffer(outBuf_, 3U * kMatElems * sizeof(float));
            pipe_->InitBuffer(workBuf_, 3U * 2U * kMatElems * sizeof(float));
            pipe_->InitBuffer(uBuf_, kMatElems * sizeof(float));
            pipe_->InitBuffer(bBuf_, kMatElems * sizeof(float));
            pipe_->InitBuffer(cBuf_, kMatElems * sizeof(float));
            pipe_->InitBuffer(alphaBuf_, 3U * 8U * kExpBufElems * sizeof(float));
            pipe_->InitBuffer(dBuf_, 3U * kMaxP * sizeof(float));
            pipe_->InitBuffer(dtRawBuf_, kMaxDtRawElems * sizeof(float));
        } else {
            pipe_->InitBuffer(stateBuf_, maxN * kMaxP * sizeof(float));
            pipe_->InitBuffer(uBuf_, kChunkSize * kMaxP * sizeof(float));
            pipe_->InitBuffer(bBuf_, kChunkSize * maxN * sizeof(float));
            pipe_->InitBuffer(cBuf_, kChunkSize * maxN * sizeof(float));
            pipe_->InitBuffer(outBuf_, kChunkSize * kMaxP * sizeof(float));
            pipe_->InitBuffer(workBuf_, kChunkSize * kChunkSize * sizeof(float));
            pipe_->InitBuffer(lBuf_, kChunkSize * kChunkSize * sizeof(float));
            pipe_->InitBuffer(dBuf_, kMaxP * sizeof(float));
            pipe_->InitBuffer(dtRawBuf_, kMaxDtRawElems * sizeof(float));
            pipe_->InitBuffer(alphaBuf_, 8U * kExpBufElems * sizeof(float));
        }
    }

    __aicore__ inline void InitTrilWorkspace()
    {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
        if (headdim_ == 64U && (dstate_ == 64U || dstate_ == 128U)) {
            LocalTensor<float> workLocal = workBuf_.Get<float>(kMatElems);
            Duplicate(workLocal, 0.0F, kMatElems);
            pipe_barrier(PIPE_ALL);
            __ubuf__ float* trilPtr = reinterpret_cast<__ubuf__ float*>(workLocal.GetPhyAddr());
            for (uint32_t i = 0; i < 64U; ++i) {
                for (uint32_t j = 0; j <= i; ++j) {
                    trilPtr[i * 64U + j] = 1.0F;
                }
            }
            pipe_barrier(PIPE_ALL);
            DataCopy(gmWsTril_, workLocal, kMatElems);
            pipe_barrier(PIPE_ALL);
        }
#endif
    }

    __aicore__ inline void Init(const KernelGmAddrs& addrs, TPipe* pipe)
    {
        pipe_ = pipe;
        InitGlobalBuffers(addrs);
        if ASCEND_IS_AIC { return; }
        InitPipeBuffers();
        InitTrilWorkspace();
    }

    __aicore__ inline uint32_t IssueLoadDtAndCopyB0_64(
        const StepSlotInfo& slot,
        LocalTensor<float>& bLocal,
        LocalTensor<float>& dtRawLocal,
        const GlobalTensor<float>& gmWsB0_dst)
    {
        const uint64_t stepBase = (static_cast<uint64_t>(slot.b) * seqLen_ + slot.t0) * nheads_ + slot.h;
        const uint64_t bcBase = stepBase * dstate_;
        const uint16_t bcBlocks = static_cast<uint16_t>(dstate_ / 8U);

        const uint64_t dtBase = (static_cast<uint64_t>(slot.b) * seqLen_ + slot.t0) * nheads_;
        const uint64_t dtAligned = dtBase & ~7ULL;
        const uint32_t dtRem = static_cast<uint32_t>(dtBase & 7ULL);
        const uint32_t dtElems = ((dtRem + 64U * nheads_ + 7U) / 8U) * 8U;
        if (slot.loadDt) {
            DataCopy(dtRawLocal, gmDt_[dtAligned], dtElems);
        }

        DataCopyParams bcCopy{64U, bcBlocks, static_cast<uint16_t>((nheads_ - 1U) * bcBlocks), 0U};
        DataCopy(bLocal, gmB_[bcBase], bcCopy);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(gmWsB0_dst, bLocal, kMatElems);
        return dtRem;
    }

    __aicore__ inline void LoadCurrentU64(
        uint32_t b, uint32_t h, uint32_t t0, LocalTensor<float>& uLocal)
    {
        const uint64_t stepBase = (static_cast<uint64_t>(b) * seqLen_ + t0) * nheads_ + h;
        const uint64_t uBase = stepBase * headdim_;
        const uint16_t uBlocks = static_cast<uint16_t>(headdim_ / 8U);
        DataCopyParams uCopy{64U, uBlocks, static_cast<uint16_t>((nheads_ - 1U) * uBlocks), 0U};
        DataCopy(uLocal, gmU_[uBase], uCopy);
    }

    __aicore__ inline uint32_t IssueLoadChunkInputs(
        SeqHeadContext& ctx, uint32_t t0, uint32_t curSteps)
    {
        const uint64_t stepBase = (static_cast<uint64_t>(ctx.b) * seqLen_ + t0) * nheads_ + ctx.h;
        const uint64_t uBase = stepBase * headdim_;
        const uint64_t bcBase = stepBase * dstate_;
        const uint16_t uBlocks = static_cast<uint16_t>(headdim_ / 8U);
        const uint16_t bcBlocks = static_cast<uint16_t>(dstate_ / 8U);

        DataCopyParams uCopy{static_cast<uint16_t>(curSteps), uBlocks,
                             static_cast<uint16_t>((nheads_ - 1U) * uBlocks), 0U};
        DataCopy(ctx.uLocal, gmU_[uBase], uCopy);

        IssueLoadBcInputs(ctx, bcBase, curSteps, bcBlocks);

        const uint64_t dtBase = (static_cast<uint64_t>(ctx.b) * seqLen_ + t0) * nheads_;
        const uint64_t dtAligned = dtBase & ~7ULL;
        const uint32_t dtRem = static_cast<uint32_t>(dtBase & 7ULL);
        const uint32_t dtElems = ((dtRem + curSteps * nheads_ + 7U) / 8U) * 8U;
        DataCopy(ctx.dtRawLocal, gmDt_[dtAligned], dtElems);
        return dtRem;
    }

    __aicore__ inline void LoadBc128Half(
        SeqHeadContext& ctx, uint64_t bcBase, uint16_t bcBlocks)
    {
        const uint16_t halfBlocks = 8U;
        const uint16_t strideBlocks = static_cast<uint16_t>(nheads_ * bcBlocks - halfBlocks);
        DataCopyParams bcHalfCopy{64U, halfBlocks, strideBlocks, 0U};
        DataCopy(ctx.bLocal[0], gmB_[bcBase], bcHalfCopy);
        DataCopy(ctx.bLocal[kMatElems], gmB_[bcBase + 64U], bcHalfCopy);
        DataCopy(ctx.cLocal[0], gmC_[bcBase], bcHalfCopy);
        DataCopy(ctx.cLocal[kMatElems], gmC_[bcBase + 64U], bcHalfCopy);
    }

    __aicore__ inline void LoadBcGeneric(
        SeqHeadContext& ctx, uint64_t bcBase, uint32_t curSteps, uint16_t bcBlocks)
    {
        DataCopyParams bcCopy{static_cast<uint16_t>(curSteps), bcBlocks,
                              static_cast<uint16_t>((nheads_ - 1U) * bcBlocks), 0U};
        DataCopy(ctx.bLocal, gmB_[bcBase], bcCopy);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        const bool skipC = (dstate_ == 64U) & (headdim_ == 64U) & (curSteps == 64U) & ((seqLen_ % 64U) == 0U);
        if (!skipC) {
            DataCopy(ctx.cLocal, gmC_[bcBase], bcCopy);
        }
#else
        DataCopy(ctx.cLocal, gmC_[bcBase], bcCopy);
#endif
    }

    __aicore__ inline void IssueLoadBcInputs(
        SeqHeadContext& ctx, uint64_t bcBase, uint32_t curSteps, uint16_t bcBlocks)
    {
        if (dstate_ == 128U && curSteps == 64U) {
            LoadBc128Half(ctx, bcBase, bcBlocks);
        } else {
            LoadBcGeneric(ctx, bcBase, curSteps, bcBlocks);
        }
    }

    __aicore__ inline uint32_t LoadChunkInputs(
        SeqHeadContext& ctx, uint32_t t0, uint32_t curSteps)
    {
        const uint32_t rem = IssueLoadChunkInputs(ctx, t0, curSteps);
        pipe_barrier(PIPE_ALL);
        return rem;
    }

    template <typename DstTensorT>
    __aicore__ inline void LaunchCubeAsync(
        CubeMatmulObj& matmulObj,
        const GlobalTensor<float>& gmA, const GlobalTensor<float>& gmB,
        bool transB, const DstTensorT& dstC)
    {
        matmulObj.SetTensorA(gmA, false);
        matmulObj.SetTensorB(gmB, transB);
        matmulObj.IterateAll<false>(dstC, 0, false, true);
    }

    __aicore__ inline void WaitCubeAsync(CubeMatmulObj& matmulObj)
    {
        matmulObj.WaitIterateAll();
        matmulObj.End();
    }

    __aicore__ inline void ComputeChunkAlpha64(
        uint32_t hOffset, float aVal,
        LocalTensor<float>& dtRawLocal, LocalTensor<float>& alphaLocal)
    {
        LocalTensor<uint32_t> idxLocal = alphaLocal[320U].ReinterpretCast<uint32_t>();
        const uint32_t srcBaseOffset = hOffset * static_cast<uint32_t>(sizeof(float));
        Gather(alphaLocal[128U], dtRawLocal, idxLocal, srcBaseOffset, 64U);
        PipeBarrier<PIPE_V>();
        __ubuf__ float* alphaPtr = reinterpret_cast<__ubuf__ float*>(alphaLocal.GetPhyAddr());
        PrefixSumAlpha64VF(alphaPtr, aVal);
        PipeBarrier<PIPE_V>();
        Exp(alphaLocal, alphaLocal, 128U);
        PipeBarrier<PIPE_V>();
        Mul(alphaLocal[64U], alphaLocal[64U], alphaLocal[128U], 64U);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrepMix64ScaleAndWsA1(uint32_t t0, SeqHeadContext& ctx)
    {
        __ubuf__ float* alphaPtr = reinterpret_cast<__ubuf__ float*>(ctx.alphaLocal.GetPhyAddr());
        if (t0 == 0U) {
            set_intra_block(PIPE_MTE3, 4U);
            set_intra_block(PIPE_V, 5U);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            ComputeChunkAlpha64(ctx.dtRem + ctx.h, ctx.aVal, ctx.dtRawLocal, ctx.alphaLocal);
        }
        if (t0 > 0U) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        ScaleBOutChunk64VF(
            reinterpret_cast<__ubuf__ float*>(ctx.bLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(ctx.uLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(ctx.dLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(ctx.outLocal.GetPhyAddr()),
            alphaPtr);
        PipeBarrier<PIPE_V>();
        Transpose64x64UB(ctx.cLocal, ctx.bLocal);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        DataCopy(gmWsA1_, ctx.cLocal, kMatElems);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void ProcessChunkMix64PipelinedStageA(
        uint32_t t0, SeqHeadContext& ctx, uint32_t& nextDtRem, uint32_t nextSteps)
    {
        PrepMix64ScaleAndWsA1(t0, ctx);
        __ubuf__ float* alphaPtr = reinterpret_cast<__ubuf__ float*>(ctx.alphaLocal.GetPhyAddr());
        const bool hasNextChunk = (t0 + kChunkSize < seqLen_);
        if (hasNextChunk) {
            nextDtRem = IssueLoadChunkInputs(ctx, t0 + kChunkSize, nextSteps);
        }

        wait_intra_block(PIPE_V, 6U);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        Transpose64x64UB(ctx.cLocal, ctx.workLocal);
        Transpose64x64UB(ctx.workLocal, ctx.lLocal);
        PipeBarrier<PIPE_V>();

        FuseMaskAndInterVF(
            reinterpret_cast<__ubuf__ float*>(ctx.cLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(ctx.outLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(ctx.workLocal.GetPhyAddr()),
            alphaPtr);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(gmWsA0_, ctx.cLocal, kMatElems);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void UpdateMix64StateAndWsB1(SeqHeadContext& ctx)
    {
        __ubuf__ float* alphaPtr = reinterpret_cast<__ubuf__ float*>(ctx.alphaLocal.GetPhyAddr());
        wait_intra_block(PIPE_V, 6U);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        UpdateStateAndOutVF(
            reinterpret_cast<__ubuf__ float*>(ctx.stateLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(ctx.lLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(ctx.outLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(ctx.workLocal.GetPhyAddr()),
            alphaPtr);
        PipeBarrier<PIPE_V>();
        Transpose64x64UB(ctx.cLocal, ctx.stateLocal);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        DataCopy(gmWsB1_, ctx.cLocal, kMatElems);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void ProcessChunkMix64PipelinedStageB(
        uint32_t t0, SeqHeadContext& ctx, uint32_t nextDtRem, uint32_t nextSteps)
    {
        const uint64_t stepBase = (static_cast<uint64_t>(ctx.b) * seqLen_ + t0) * nheads_ + ctx.h;
        const uint64_t uBase = stepBase * headdim_;
        const bool hasNextChunk = (t0 + kChunkSize < seqLen_);

        set_intra_block(PIPE_MTE3, 4U);
        set_intra_block(PIPE_V, 5U);

        if (hasNextChunk) {
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopy(gmWsB0_, ctx.bLocal, kMatElems);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if (nextSteps == 64U) {
                ComputeChunkAlpha64(nextDtRem + ctx.h, ctx.aVal, ctx.dtRawLocal, ctx.alphaLocal);
            }
        }

        UpdateMix64StateAndWsB1(ctx);

        if (hasNextChunk && nextSteps == 64U) {
            set_intra_block(PIPE_MTE3, 4U);
            set_intra_block(PIPE_V, 5U);
        }

        const uint16_t uBlocks = static_cast<uint16_t>(headdim_ / 8U);
        DataCopyParams outCopy{64U, uBlocks, 0U, static_cast<uint16_t>((nheads_ - 1U) * uBlocks)};
        DataCopy(gmOut_[uBase], ctx.outLocal, outCopy);
        if (hasNextChunk) {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            ctx.dtRem = nextDtRem;
        } else {
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void ProcessChunkMix64Pipelined(
        CubeMatmulObj& matmulObj, uint32_t t0, SeqHeadContext& ctx)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        (void)matmulObj;
        const bool hasNextChunk = (t0 + kChunkSize < seqLen_);
        const uint32_t nextSteps = hasNextChunk ? ((t0 + 2U * kChunkSize <= seqLen_) ? kChunkSize : (seqLen_ - (t0 + kChunkSize))) : 0U;
        uint32_t nextDtRem = 0U;
        ProcessChunkMix64PipelinedStageA(t0, ctx, nextDtRem, nextSteps);
        ProcessChunkMix64PipelinedStageB(t0, ctx, nextDtRem, nextSteps);
#endif
    }

    __aicore__ inline void ComputeMix128Alpha(SeqHeadContext& ctx, float& exp_last)
    {
        __ubuf__ const float* dtHeadPtr =
            reinterpret_cast<__ubuf__ const float*>(ctx.dtRawLocal.GetPhyAddr()) + ctx.dtRem + ctx.h;
        __ubuf__ float* alphaPtr = reinterpret_cast<__ubuf__ float*>(ctx.alphaLocal.GetPhyAddr());
        const uint32_t dtStride = nheads_;
        float run_cum = 0.0f;
        for (uint32_t s = 0; s < 64U; ++s) {
            const float dtVal = dtHeadPtr[s * dtStride];
            run_cum += dtVal * ctx.aVal;
            alphaPtr[s] = run_cum;
            alphaPtr[64U + s] = -run_cum;
            alphaPtr[128U + s] = dtVal;
        }
        pipe_barrier(PIPE_ALL);
        Exp(ctx.alphaLocal, ctx.alphaLocal, 128U);
        PipeBarrier<PIPE_V>();
        Mul(ctx.alphaLocal[64U], ctx.alphaLocal[64U], ctx.alphaLocal[128U], 64U);
        exp_last = alphaPtr[63U];
    }

    __aicore__ inline void ScaleMix128LocalAndLaunchMatmul2(
        CubeMatmulObj& matmulObj, SeqHeadContext& ctx, float exp_last, __ubuf__ float* alphaPtr)
    {
        for (uint32_t i = 0; i < 64U; ++i) {
            Muls(ctx.lLocal[i * 64U], ctx.alphaLocal[64U], alphaPtr[i], 64U);
            Muls(ctx.cLocal[i * 64U], ctx.cLocal[i * 64U], alphaPtr[i], 64U);
            Muls(ctx.bLocal[i * 64U], ctx.bLocal[i * 64U], exp_last * alphaPtr[64U + i], 64U);
        }
        PipeBarrier<PIPE_V>();
        Mul(ctx.lLocal, ctx.lLocal, ctx.workLocal, kMatElems);
        pipe_barrier(PIPE_ALL);

        DataCopy(gmWsCdec0_, ctx.cLocal[0], kMatElems);
        DataCopy(gmWsBdec0_, ctx.bLocal[0], kMatElems);
        pipe_barrier(PIPE_ALL);
        WaitCubeAsync(matmulObj);

        LaunchCubeAsync(matmulObj, gmWsA1_, gmWsB1_, true, ctx.lLocal);
        for (uint32_t i = 0; i < 64U; ++i) {
            Muls(ctx.cLocal[kMatElems + i * 64U], ctx.cLocal[kMatElems + i * 64U], alphaPtr[i], 64U);
            Muls(ctx.bLocal[kMatElems + i * 64U], ctx.bLocal[kMatElems + i * 64U], exp_last * alphaPtr[64U + i], 64U);
        }
        pipe_barrier(PIPE_ALL);
        DataCopy(gmWsCdec1_, ctx.cLocal[kMatElems], kMatElems);
        DataCopy(gmWsBdec1_, ctx.bLocal[kMatElems], kMatElems);
        pipe_barrier(PIPE_ALL);
        WaitCubeAsync(matmulObj);
    }

    __aicore__ inline void ProcessChunkMix128StageA(
        CubeMatmulObj& matmulObj, SeqHeadContext& ctx, float& exp_last)
    {
        DataCopy(gmWsA0_, ctx.cLocal[0], kMatElems);
        DataCopy(gmWsB0_, ctx.bLocal[0], kMatElems);
        pipe_barrier(PIPE_ALL);
        LaunchCubeAsync(matmulObj, gmWsA0_, gmWsB0_, true, ctx.workLocal);

        DataCopy(gmWsA1_, ctx.cLocal[kMatElems], kMatElems);
        DataCopy(gmWsB1_, ctx.bLocal[kMatElems], kMatElems);
        DataCopy(gmWsU_, ctx.uLocal, kMatElems);
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
        DataCopy(ctx.workLocal, gmWsTril_, kMatElems);
#endif
        DataCopy(gmWsState1_, ctx.stateLocal[kMatElems], kMatElems);

        ComputeMix128Alpha(ctx, exp_last);
        __ubuf__ float* alphaPtr = reinterpret_cast<__ubuf__ float*>(ctx.alphaLocal.GetPhyAddr());

        Mul(ctx.outLocal, ctx.uLocal, ctx.dLocal, 64U, 64U, BinaryRepeatParams{1, 1, 1, 8, 8, 0});
        pipe_barrier(PIPE_ALL);

        ScaleMix128LocalAndLaunchMatmul2(matmulObj, ctx, exp_last, alphaPtr);
    }

    __aicore__ inline void AccumulateMix128Output(
        CubeMatmulObj& matmulObj, SeqHeadContext& ctx, float exp_last)
    {
        LaunchCubeAsync(matmulObj, gmWsCdec0_, gmWsB2_, false, ctx.cLocal[0]);
        Add(ctx.workLocal, ctx.workLocal, ctx.lLocal, kMatElems);
        PipeBarrier<PIPE_V>();
        Mul(ctx.workLocal, ctx.workLocal, ctx.lLocal, kMatElems);
        Muls(ctx.stateLocal[0], ctx.stateLocal[0], exp_last, kMatElems);
        pipe_barrier(PIPE_ALL);
        DataCopy(gmWsA0_, ctx.workLocal, kMatElems);
        pipe_barrier(PIPE_ALL);
        WaitCubeAsync(matmulObj);

        LaunchCubeAsync(matmulObj, gmWsCdec1_, gmWsState1_, false, ctx.lLocal);
        Muls(ctx.stateLocal[kMatElems], ctx.stateLocal[kMatElems], exp_last, kMatElems);
        Add(ctx.outLocal, ctx.outLocal, ctx.cLocal[0], kMatElems);
        pipe_barrier(PIPE_ALL);
        WaitCubeAsync(matmulObj);

        LaunchCubeAsync(matmulObj, gmWsA0_, gmWsU_, false, ctx.workLocal);
        Add(ctx.outLocal, ctx.outLocal, ctx.lLocal, kMatElems);
        pipe_barrier(PIPE_ALL);
        WaitCubeAsync(matmulObj);

        LaunchCubeAsync(matmulObj, gmWsBdec0_, gmWsU_, false, ctx.lLocal);
        Add(ctx.outLocal, ctx.outLocal, ctx.workLocal, kMatElems);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void UpdateMix128StateAndNext(
        CubeMatmulObj& matmulObj, uint32_t t0, SeqHeadContext& ctx)
    {
        const uint64_t outBase = ((static_cast<uint64_t>(ctx.b) * seqLen_ + t0) * nheads_ + ctx.h) * headdim_;
        DataCopyParams outCopy{64U, 8U, 0U, static_cast<uint16_t>((nheads_ - 1U) * 8U)};
        DataCopy(gmOut_[outBase], ctx.outLocal, outCopy);

        if (t0 + kChunkSize < seqLen_) {
            const uint32_t nextSteps = (t0 + 2U * kChunkSize <= seqLen_) ? kChunkSize : (seqLen_ - (t0 + kChunkSize));
            ctx.dtRem = IssueLoadChunkInputs(ctx, t0 + kChunkSize, nextSteps);
        }

        WaitCubeAsync(matmulObj);
        LaunchCubeAsync(matmulObj, gmWsBdec1_, gmWsU_, false, ctx.workLocal);
        Add(ctx.stateLocal[0], ctx.stateLocal[0], ctx.lLocal, kMatElems);
        pipe_barrier(PIPE_ALL);
        DataCopy(gmWsB2_, ctx.stateLocal[0], kMatElems);
        pipe_barrier(PIPE_ALL);

        WaitCubeAsync(matmulObj);
        Add(ctx.stateLocal[kMatElems], ctx.stateLocal[kMatElems], ctx.workLocal, kMatElems);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void ProcessChunkMix128StageB(
        CubeMatmulObj& matmulObj, uint32_t t0, SeqHeadContext& ctx, float exp_last)
    {
        AccumulateMix128Output(matmulObj, ctx, exp_last);
        UpdateMix128StateAndNext(matmulObj, t0, ctx);
    }

    __aicore__ inline void ProcessChunkMix128Pipelined(
        CubeMatmulObj& matmulObj, uint32_t t0, SeqHeadContext& ctx)
    {
        float exp_last = 0.0f;
        ProcessChunkMix128StageA(matmulObj, ctx, exp_last);
        ProcessChunkMix128StageB(matmulObj, t0, ctx, exp_last);
    }

    __aicore__ inline void ProcessSingleStepGeneric(
        uint32_t s, float dtVal, float alphaVal, SeqHeadContext& ctx)
    {
        __ubuf__ const float* bPtr = reinterpret_cast<__ubuf__ const float*>(ctx.bLocal.GetPhyAddr());
        __ubuf__ const float* cPtr = reinterpret_cast<__ubuf__ const float*>(ctx.cLocal.GetPhyAddr());
        LocalTensor<float> uStep = ctx.uLocal[s * headdim_];
        Muls(ctx.stateLocal, ctx.stateLocal, alphaVal, dstate_ * headdim_);
        PipeBarrier<PIPE_V>();

        Mul(ctx.workLocal, ctx.dLocal, uStep, headdim_);
        Duplicate(ctx.workLocal[headdim_], 0.0F, (kAccRows - 1U) * headdim_);
        for (uint32_t n = 0; n < dstate_; ++n) {
            const float dtB = dtVal * bPtr[s * dstate_ + n];
            Axpy(ctx.stateLocal[n * headdim_], uStep, dtB, headdim_);
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t n = 0; n < dstate_; ++n) {
            const float cVal = cPtr[s * dstate_ + n];
            Axpy(ctx.workLocal[(n & 7U) * headdim_], ctx.stateLocal[n * headdim_], cVal, headdim_);
        }
        PipeBarrier<PIPE_V>();

        Add(ctx.workLocal, ctx.workLocal, ctx.workLocal[4U * headdim_], 4U * headdim_);
        PipeBarrier<PIPE_V>();
        Add(ctx.workLocal, ctx.workLocal, ctx.workLocal[2U * headdim_], 2U * headdim_);
        PipeBarrier<PIPE_V>();
        Add(ctx.outLocal[s * headdim_], ctx.workLocal, ctx.workLocal[headdim_], headdim_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessTailChunkGeneric(
        uint32_t t0, uint32_t curSteps, SeqHeadContext& ctx)
    {
        __ubuf__ const float* dtRawPtr = reinterpret_cast<__ubuf__ const float*>(ctx.dtRawLocal.GetPhyAddr());
        __ubuf__ float* alphaPtr = reinterpret_cast<__ubuf__ float*>(ctx.alphaLocal.GetPhyAddr());
        for (uint32_t s = 0; s < curSteps; ++s) {
            const float dtVal = dtRawPtr[ctx.dtRem + s * nheads_ + ctx.h];
            alphaPtr[s] = dtVal * ctx.aVal;
        }
        pipe_barrier(PIPE_ALL);
        Exp(ctx.alphaLocal, ctx.alphaLocal, kExpBufElems);
        pipe_barrier(PIPE_ALL);
        for (uint32_t s = 0; s < curSteps; ++s) {
            const float dtVal = dtRawPtr[ctx.dtRem + s * nheads_ + ctx.h];
            ProcessSingleStepGeneric(s, dtVal, alphaPtr[s], ctx);
        }
        pipe_barrier(PIPE_ALL);
        const uint64_t outBase = ((static_cast<uint64_t>(ctx.b) * seqLen_ + t0) * nheads_ + ctx.h) * headdim_;
        const uint16_t uBlocks = static_cast<uint16_t>(headdim_ / 8U);
        DataCopyParams outCopy{static_cast<uint16_t>(curSteps), uBlocks,
                               0U, static_cast<uint16_t>((nheads_ - 1U) * uBlocks)};
        DataCopy(gmOut_[outBase], ctx.outLocal, outCopy);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void RunHeadSeqMix64(CubeMatmulObj& matmulObj, SeqHeadContext& ctx)
    {
        Duplicate(ctx.alphaLocal[192U], 0.0F, 64U);
        __ubuf__ uint32_t* idxPtr = reinterpret_cast<__ubuf__ uint32_t*>(ctx.alphaLocal[320U].GetPhyAddr());
        for (uint32_t s = 0; s < 64U; ++s) {
            idxPtr[s] = s * nheads_ * static_cast<uint32_t>(sizeof(float));
        }
        pipe_barrier(PIPE_ALL);
        DataCopy(gmWsB1_, ctx.stateLocal, kMatElems);
        const uint32_t firstSteps = (kChunkSize <= seqLen_) ? kChunkSize : seqLen_;
        ctx.dtRem = IssueLoadChunkInputs(ctx, 0U, firstSteps);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(gmWsB0_, ctx.bLocal, kMatElems);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);

        for (uint32_t t0 = 0; t0 < seqLen_; t0 += kChunkSize) {
            const uint32_t curSteps = (t0 + kChunkSize <= seqLen_) ? kChunkSize : (seqLen_ - t0);
            if (curSteps == 64U) {
                ProcessChunkMix64Pipelined(matmulObj, t0, ctx);
            } else {
                ProcessTailChunkGeneric(t0, curSteps, ctx);
            }
        }
    }

    __aicore__ inline void RunHeadSeqMix128(CubeMatmulObj& matmulObj, SeqHeadContext& ctx)
    {
        DataCopy(gmWsB2_, ctx.stateLocal[0], kMatElems);
        DataCopy(gmWsState1_, ctx.stateLocal[kMatElems], kMatElems);
        const uint32_t firstSteps = (kChunkSize <= seqLen_) ? kChunkSize : seqLen_;
        ctx.dtRem = LoadChunkInputs(ctx, 0U, firstSteps);

        for (uint32_t t0 = 0; t0 < seqLen_; t0 += kChunkSize) {
            const uint32_t curSteps = (t0 + kChunkSize <= seqLen_) ? kChunkSize : (seqLen_ - t0);
            if (curSteps == 64U) {
                ProcessChunkMix128Pipelined(matmulObj, t0, ctx);
            } else {
                ProcessTailChunkGeneric(t0, curSteps, ctx);
            }
        }
    }

    __aicore__ inline void ProcessHeadSequence(CubeMatmulObj& matmulObj, uint32_t b, uint32_t h)
    {
        if ASCEND_IS_AIC { return; }
        SeqHeadContext ctx;
        InitSeqHeadContext(b, h, ctx);

        if ((headdim_ == 64U) && (dstate_ == 64U)) {
            RunHeadSeqMix64(matmulObj, ctx);
        } else if ((headdim_ == 64U) && (dstate_ == 128U)) {
            RunHeadSeqMix128(matmulObj, ctx);
        } else {
            RunHeadSeqFallback(ctx);
        }
    }

    __aicore__ inline void InitSeqHeadContext(uint32_t b, uint32_t h, SeqHeadContext& ctx)
    {
        const uint32_t maxN = (dstate_ == 128U) ? 128U : 64U;
        ctx.b = b;
        ctx.h = h;
        ctx.aVal = gmA_.GetValue(h);
        ctx.stateLocal = stateBuf_.Get<float>(maxN * kMaxP);
        ctx.uLocal = uBuf_.Get<float>(kMatElems);
        ctx.bLocal = bBuf_.Get<float>(kChunkSize * maxN);
        ctx.cLocal = cBuf_.Get<float>(kChunkSize * maxN);
        ctx.outLocal = outBuf_.Get<float>(kMatElems);
        ctx.workLocal = workBuf_.Get<float>(kMatElems);
        ctx.lLocal = lBuf_.Get<float>(kMatElems);
        ctx.dLocal = dBuf_.Get<float>(kMaxP);
        ctx.dtRawLocal = dtRawBuf_.Get<float>(kMaxDtRawElems);
        ctx.alphaLocal = alphaBuf_.Get<float>(8U * kExpBufElems);

        DataCopy(ctx.dLocal, gmD_[static_cast<uint64_t>(h) * headdim_], headdim_);
        Duplicate(ctx.stateLocal, 0.0F, dstate_ * headdim_);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void RunHeadSeqFallback(SeqHeadContext& ctx)
    {
        for (uint32_t t0 = 0; t0 < seqLen_; t0 += kChunkStepsFallback) {
            const uint32_t curSteps = (t0 + kChunkStepsFallback <= seqLen_) ? kChunkStepsFallback : (seqLen_ - t0);
            ctx.dtRem = LoadChunkInputs(ctx, t0, curSteps);
            ProcessTailChunkGeneric(t0, curSteps, ctx);
        }
    }

    __aicore__ inline uint64_t MapSlotCoalesced128(uint32_t aivId, uint32_t k)
    {
        const uint32_t aicId = aivId >> 1U;
        const uint32_t pair = aivId & 1U;
        if (k == 0U) {
            return (static_cast<uint64_t>(aicId) << 32) | static_cast<uint64_t>(2U * pair + 0U);
        }
        if (k == 1U) {
            return (static_cast<uint64_t>(aicId) << 32) | static_cast<uint64_t>(2U * pair + 1U);
        }
        if (aicId < 8U) {
            const uint32_t bOut = 28U + (aicId >> 1U);
            const uint32_t hOut = 2U * (aicId & 1U) + pair;
            return (static_cast<uint64_t>(bOut) << 32) | static_cast<uint64_t>(hOut);
        }
        return UINT64_MAX;
    }

    __aicore__ inline uint64_t MapSlotToBatchHead(
        uint32_t groupBase, uint32_t aivId, uint32_t totalAivCores, uint32_t k)
    {
        if (nheads_ == 0U) {
            return UINT64_MAX;
        }
        const uint32_t totalHeads = batch_ * nheads_;
        const bool isCoalesced = (groupBase == 0U) & (nheads_ == 4U) & (totalHeads == 128U) & (totalAivCores == 56U);
        if (isCoalesced) {
            return MapSlotCoalesced128(aivId, k);
        }
        const uint32_t tIdx = groupBase + k * totalAivCores + aivId;
        if (tIdx >= totalHeads) return UINT64_MAX;
        const uint32_t bOut = tIdx / nheads_;
        const uint32_t hOut = tIdx % nheads_;
        return (static_cast<uint64_t>(bOut) << 32) | static_cast<uint64_t>(hOut);
    }

#ifdef SPLIT_CORE_CUBE
    __aicore__ inline void LoadCubeL0Inputs(
        const AicSubTask& task, const LocalTensor<float>& aL1, const LocalTensor<float>& bL1, bool initedSub)
    {
        const uint32_t subOff16 = task.subId * 16384U;
        const uint32_t subOff32 = task.subId * 32768U;
        LocalTensor<float> aL0{TPosition::A2, 0U + subOff32, 2U * kMatElems};
        LocalTensor<float> bL0{TPosition::B2, 0U + subOff16, kMatElems};

        SetFlag<HardEvent::MTE2_MTE1>(task.ev);
        WaitFlag<HardEvent::MTE2_MTE1>(task.ev);
        if (initedSub) {
            WaitFlag<HardEvent::M_MTE1>(task.ev);
        }
        LoadData2DParamsV2 loadA;
        loadA.mStartPosition = 0U;
        loadA.kStartPosition = 0U;
        loadA.mStep = 8U;
        loadA.kStep = 8U;
        loadA.srcStride = 8U;
        loadA.dstStride = 8U;
        loadA.ifTranspose = false;
        LoadData(aL0, aL1, loadA);

        LoadData2DParamsV2 loadB;
        loadB.mStartPosition = 0U;
        loadB.kStartPosition = 0U;
        loadB.mStep = 4U;
        loadB.kStep = 8U;
        loadB.srcStride = 4U;
        loadB.dstStride = 4U;
        loadB.ifTranspose = !task.transB;
        LoadData(bL0, bL1, loadB);
    }

    __aicore__ inline void ExecuteCubeMmadAndFixpipe(const AicSubTask& task, bool& initedSub)
    {
        const uint32_t subOff16 = task.subId * 16384U;
        const uint32_t subOff32 = task.subId * 32768U;
        LocalTensor<float> aL0{TPosition::A2, 0U + subOff32, 2U * kMatElems};
        LocalTensor<float> bL0{TPosition::B2, 0U + subOff16, kMatElems};
        LocalTensor<float> cL0{TPosition::CO1, 0U + subOff32, 2U * kMatElems};
        LocalTensor<float> cUb{TPosition::VECCALC, static_cast<uint32_t>(task.dstUbAddr), 2U * kMatElems};

        SetFlag<HardEvent::MTE1_MTE2>(task.ev);
        SetFlag<HardEvent::MTE1_M>(task.ev);
        WaitFlag<HardEvent::MTE1_M>(task.ev);
        if (initedSub) {
            WaitFlag<HardEvent::FIX_M>(task.ev);
        }
        MmadParams mmadParams;
        mmadParams.m = 128U;
        mmadParams.k = 64U;
        mmadParams.n = 64U;
        mmadParams.cmatrixSource = false;
        mmadParams.cmatrixInitVal = true;
        mmadParams.isBias = false;
        mmadParams.disableGemv = true;
        mmadParams.unitFlag = 3U;
        Mmad(cL0, aL0, bL0, mmadParams);
        SetFlag<HardEvent::M_MTE1>(task.ev);

        wait_intra_block(PIPE_FIX, 5U + task.off);
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixParams(64U, 128U, 128U, 64U);
        fixParams.params.ndNum = 1U;
        fixParams.subBlockId = (task.subId != 0U);
        fixParams.unitFlag = 3U;
        Fixpipe<float, float, AscendC::Impl::CFG_ROW_MAJOR_UB>(cUb, cL0, fixParams);
        set_intra_block(PIPE_FIX, 6U + task.off);
        SetFlag<HardEvent::FIX_M>(task.ev);
        initedSub = true;
    }

    __aicore__ inline void FinishCubeL0MmadAndFixpipe(
        const AicSubTask& task, const LocalTensor<float>& aL1, const LocalTensor<float>& bL1, bool& initedSub)
    {
        LoadCubeL0Inputs(task, aL1, bL1, initedSub);
        ExecuteCubeMmadAndFixpipe(task, initedSub);
    }
#endif

    __aicore__ inline void RunCubeCoalescedOnAIC(
        const AicMatmulCall& call, bool& initedSub, bool& initedOtherSub)
    {
#ifdef SPLIT_CORE_CUBE
        const uint32_t subOff16 = call.task.subId * 16384U;
        const uint32_t subOff32 = call.task.subId * 32768U;
        const event_t evOther = (call.task.subId == 0U) ? EVENT_ID4 : EVENT_ID3;

        LocalTensor<float> aL1{TPosition::A1, 0U + subOff32, 2U * kMatElems};
        LocalTensor<float> bL1All{TPosition::B1, 65536U, 4U * kMatElems};
        LocalTensor<float> bL1{TPosition::B1, 65536U + subOff16, kMatElems};

        if (initedSub) {
            WaitFlag<HardEvent::MTE1_MTE2>(call.task.ev);
        }
        wait_intra_block(PIPE_MTE2, 4U + call.task.off);

        Nd2NzParams nd2nzA{1U, 128U, 64U, 0U, 64U, 128U, 1U, 0U};
        DataCopy(aL1, call.gmA, nd2nzA);
        SetFlag<HardEvent::MTE2_S>(call.task.ev);
        WaitFlag<HardEvent::MTE2_S>(call.task.ev);

        if (call.loadBL1Width > 0U) {
            if ((call.loadBL1Width > 64U) && initedOtherSub) {
                WaitFlag<HardEvent::MTE1_MTE2>(evOther);
                SetFlag<HardEvent::MTE1_MTE2>(evOther);
            }
            Nd2NzParams nd2nzB{1U, 64U, static_cast<uint16_t>(call.loadBL1Width), 0U, static_cast<uint16_t>(call.strideB), 64U, 1U, 0U};
            DataCopy(bL1All, call.gmB, nd2nzB);
            SetFlag<HardEvent::MTE2_S>(call.task.ev);
            WaitFlag<HardEvent::MTE2_S>(call.task.ev);
        }
        FinishCubeL0MmadAndFixpipe(call.task, aL1, bL1, initedSub);
#endif
    }

    __aicore__ inline void RunCubeOnAIC(
        const AicMatmulCall& call, bool& initedSub)
    {
#ifdef SPLIT_CORE_CUBE
        const uint32_t subOff32 = call.task.subId * 32768U;
        const uint32_t subOff16 = call.task.subId * 16384U;

        LocalTensor<float> aL1{TPosition::A1, 0U + subOff32, 2U * kMatElems};
        LocalTensor<float> bL1{TPosition::B1, 65536U + subOff16, kMatElems};

        if (initedSub) {
            WaitFlag<HardEvent::MTE1_MTE2>(call.task.ev);
        }
        wait_intra_block(PIPE_MTE2, 4U + call.task.off);

        Nd2NzParams nd2nzA{1U, 128U, 64U, 0U, 64U, 128U, 1U, 0U};
        DataCopy(aL1, call.gmA, nd2nzA);
        SetFlag<HardEvent::MTE2_S>(call.task.ev);
        WaitFlag<HardEvent::MTE2_S>(call.task.ev);

        Nd2NzParams nd2nzB{1U, 64U, 64U, 0U, static_cast<uint16_t>(call.strideB), 64U, 1U, 0U};
        DataCopy(bL1, call.gmB, nd2nzB);
        SetFlag<HardEvent::MTE2_S>(call.task.ev);
        WaitFlag<HardEvent::MTE2_S>(call.task.ev);
        FinishCubeL0MmadAndFixpipe(call.task, aL1, bL1, initedSub);
#endif
    }

    __aicore__ inline bool RunDirectAICMatmul1Slot(
        AicDirectContext& actx, uint32_t groupBase, uint32_t t0, uint32_t k)
    {
        const uint64_t bh0 = MapSlotToBatchHead(groupBase, actx.aiv0, actx.totalAivCores, k);
        const uint64_t bh1 = MapSlotToBatchHead(groupBase, actx.aiv1, actx.totalAivCores, k);
        const bool has0 = (bh0 != UINT64_MAX);
        const bool has1 = (bh1 != UINT64_MAX);
        if (!has0 && !has1) return false;

        const uint32_t b0 = static_cast<uint32_t>(bh0 >> 32), h0 = static_cast<uint32_t>(bh0);
        const uint32_t b1 = static_cast<uint32_t>(bh1 >> 32), h1 = static_cast<uint32_t>(bh1);
        const uint64_t bcBase0 = ((static_cast<uint64_t>(b0) * seqLen_ + t0) * nheads_ + h0) * dstate_;
        const uint64_t bcBase1 = ((static_cast<uint64_t>(b1) * seqLen_ + t0) * nheads_ + h1) * dstate_;
        const uint64_t dstUbAddr = 98304ULL + static_cast<uint64_t>(k) * 32768ULL;

        AicSlotPair pair{
            has0, has1,
            {{0U, 3U * k, EVENT_ID3, true, dstUbAddr}, actx.gmWsB0_0[k], gmC_[bcBase0], 0U, actx.strideBC},
            {{1U, 16U + 3U * k, EVENT_ID4, true, dstUbAddr}, actx.gmWsB0_1[k], gmC_[bcBase1], 0U, actx.strideBC}
        };

        DispatchAICMatmulCalls(actx, groupBase, k, pair);
        return true;
    }

    struct AicSlotPair {
        bool has0;
        bool has1;
        AicMatmulCall c0;
        AicMatmulCall c1;
    };

    __aicore__ inline uint32_t GetCoalescedLoadBL1Width(uint32_t k)
    {
        if (k == 0U) return 256U;
        if (k == 2U) return 128U;
        return 0U;
    }

    __aicore__ inline void RunCoalescedSlotCalls(
        AicDirectContext& actx, uint32_t k, AicSlotPair& pair)
    {
        pair.c0.loadBL1Width = GetCoalescedLoadBL1Width(k);
        if (pair.has0) RunCubeCoalescedOnAIC(pair.c0, actx.inited0, actx.inited1);
        if (pair.has1) RunCubeCoalescedOnAIC(pair.c1, actx.inited1, actx.inited0);
    }

    __aicore__ inline void RunStandardSlotCalls(
        AicDirectContext& actx, AicSlotPair& pair)
    {
        if (pair.has0) RunCubeOnAIC(pair.c0, actx.inited0);
        if (pair.has1) RunCubeOnAIC(pair.c1, actx.inited1);
    }

    __aicore__ inline void DispatchAICMatmulCalls(
        AicDirectContext& actx, uint32_t groupBase, uint32_t k, AicSlotPair& pair)
    {
        if (actx.isCoalesced128 && (groupBase == 0U)) {
            RunCoalescedSlotCalls(actx, k, pair);
        } else {
            RunStandardSlotCalls(actx, pair);
        }
    }

    __aicore__ inline bool RunDirectAICMatmul2Slot(
        AicDirectContext& actx, uint32_t groupBase, uint32_t t0, uint32_t k)
    {
        const uint64_t bh0 = MapSlotToBatchHead(groupBase, actx.aiv0, actx.totalAivCores, k);
        const uint64_t bh1 = MapSlotToBatchHead(groupBase, actx.aiv1, actx.totalAivCores, k);
        const bool has0 = (bh0 != UINT64_MAX);
        const bool has1 = (bh1 != UINT64_MAX);
        if (!has0 && !has1) return false;

        const uint32_t b0 = static_cast<uint32_t>(bh0 >> 32), h0 = static_cast<uint32_t>(bh0);
        const uint32_t b1 = static_cast<uint32_t>(bh1 >> 32), h1 = static_cast<uint32_t>(bh1);
        const uint64_t uBase0 = ((static_cast<uint64_t>(b0) * seqLen_ + t0) * nheads_ + h0) * headdim_;
        const uint64_t uBase1 = ((static_cast<uint64_t>(b1) * seqLen_ + t0) * nheads_ + h1) * headdim_;
        const uint64_t dstUbAddr = 98304ULL + static_cast<uint64_t>(k) * 32768ULL;

        AicSlotPair pair{
            has0, has1,
            {{0U, 3U * k, EVENT_ID3, false, dstUbAddr}, actx.gmWsA0_0[k], gmU_[uBase0], 0U, actx.strideU},
            {{1U, 16U + 3U * k, EVENT_ID4, false, dstUbAddr}, actx.gmWsA0_1[k], gmU_[uBase1], 0U, actx.strideU}
        };

        DispatchAICMatmulCalls(actx, groupBase, k, pair);
        return true;
    }

    __aicore__ inline void InitDirectAicBuffers(AicDirectContext& actx, GM_ADDR userWs)
    {
        __gm__ float* ws0 = reinterpret_cast<__gm__ float*>(userWs) + static_cast<uint64_t>(actx.aiv0) * kPerCoreWsFloats;
        __gm__ float* ws1 = reinterpret_cast<__gm__ float*>(userWs) + static_cast<uint64_t>(actx.aiv1) * kPerCoreWsFloats;
        for (uint32_t k = 0; k < 3U; ++k) {
            actx.gmWsA0_0[k].SetGlobalBuffer(ws0 + (4U * k + 0U) * kMatElems, 2U * kMatElems);
            actx.gmWsB0_0[k].SetGlobalBuffer(ws0 + (4U * k + 2U) * kMatElems, 2U * kMatElems);
            actx.gmWsA0_1[k].SetGlobalBuffer(ws1 + (4U * k + 0U) * kMatElems, 2U * kMatElems);
            actx.gmWsB0_1[k].SetGlobalBuffer(ws1 + (4U * k + 2U) * kMatElems, 2U * kMatElems);
        }
    }

    __aicore__ inline void FinalizeDirectAicSync(const AicDirectContext& actx)
    {
        if (actx.inited0) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID3);
        }
        if (actx.inited1) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID4);
        }
    }

    __aicore__ inline void ProcessDirectAICTimeChunk(
        AicDirectContext& actx, uint32_t groupBase, uint32_t t0)
    {
        for (uint32_t k = 0; k < 3U; ++k) {
            if (!RunDirectAICMatmul1Slot(actx, groupBase, t0, k)) break;
        }
        for (uint32_t k = 0; k < 3U; ++k) {
            if (!RunDirectAICMatmul2Slot(actx, groupBase, t0, k)) break;
        }
    }

    __aicore__ inline void ProcessDirectAIC(CubeMatmulObj& matmulObj, GM_ADDR userWs)
    {
        (void)matmulObj;
#ifdef SPLIT_CORE_CUBE
        AicDirectContext actx;
        const uint32_t totalHeads = batch_ * nheads_;
        const uint32_t aicId = GetBlockIdx();
        actx.aiv0 = aicId * 2U + 0U;
        actx.aiv1 = aicId * 2U + 1U;
        actx.totalAivCores = static_cast<uint32_t>(GetBlockNum() * 2U);
        actx.strideBC = static_cast<int32_t>(nheads_ * dstate_);
        actx.strideU = static_cast<int32_t>(nheads_ * headdim_);
        actx.isCoalesced128 = (nheads_ == 4U) && (totalHeads == 128U) && (actx.totalAivCores == 56U);

        InitDirectAicBuffers(actx, userWs);

        for (uint32_t groupBase = 0; groupBase < totalHeads; groupBase += 3U * actx.totalAivCores) {
            for (uint32_t t0 = 0; t0 < seqLen_; t0 += kChunkSize) {
                ProcessDirectAICTimeChunk(actx, groupBase, t0);
            }
        }
        FinalizeDirectAicSync(actx);
#endif
    }

    __aicore__ inline uint32_t InitInterleavedGroupSlots(
        uint32_t groupBase, InterleavedContext& ictx)
    {
        uint32_t K = 0U;
        const uint32_t totalHeads = batch_ * nheads_;
        for (uint32_t k = 0; k < 3U; ++k) {
            const uint64_t bh = MapSlotToBatchHead(groupBase, coreId_, coreNum_, k);
            if (bh == UINT64_MAX) break;
            ictx.bArr[k] = static_cast<uint32_t>(bh >> 32);
            ictx.hArr[k] = static_cast<uint32_t>(bh);
            ictx.aVal[k] = gmA_.GetValue(ictx.hArr[k]);

            LocalTensor<float> dLocalK = ictx.dAll[k * kMaxP];
            LocalTensor<float> stateLocalK = ictx.stateAll[k * kMatElems];
            LocalTensor<float> alphaLocalK = ictx.alphaAll[k * 8U * kExpBufElems];
            DataCopy(dLocalK, gmD_[static_cast<uint64_t>(ictx.hArr[k]) * headdim_], headdim_);
            Duplicate(stateLocalK, 0.0F, kMatElems);
            Duplicate(alphaLocalK[192U], 0.0F, 64U);
            __ubuf__ uint32_t* idxPtr = reinterpret_cast<__ubuf__ uint32_t*>(alphaLocalK[320U].GetPhyAddr());
            for (uint32_t s = 0; s < 64U; ++s) {
                idxPtr[s] = s * nheads_ * static_cast<uint32_t>(sizeof(float));
            }
            ++K;
        }
        if (K == 0U) return 0U;
        pipe_barrier(PIPE_ALL);
        for (uint32_t k = 0; k < K; ++k) {
            LocalTensor<float> stateLocalK = ictx.stateAll[k * kMatElems];
            DataCopy(ictx.gmWsB1[k], stateLocalK, kMatElems);
        }
        pipe_barrier(PIPE_ALL);
        ictx.K = K;
        return K;
    }

    __aicore__ inline void RunInterleavedPrologueSlot(
        uint32_t k, InterleavedContext& ictx)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        const bool loadDt = (k == 0U) || (ictx.bArr[k] != ictx.bArr[k - 1U]);
        StepSlotInfo slotInfo{ictx.bArr[k], ictx.hArr[k], 0U, loadDt};
        const uint32_t dtRem = IssueLoadDtAndCopyB0_64(
            slotInfo, ictx.bLocal, ictx.dtRawLocal, ictx.gmWsB0[k]);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        set_intra_block(PIPE_MTE3, 4U + 3U * k);
        set_intra_block(PIPE_V, 5U + 3U * k);

        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        LocalTensor<float> alphaLocalK = ictx.alphaAll[k * 8U * kExpBufElems];
        ComputeChunkAlpha64(dtRem + ictx.hArr[k], ictx.aVal[k], ictx.dtRawLocal, alphaLocalK);
        ScaleBChunk64OnlyVF(
            reinterpret_cast<__ubuf__ float*>(ictx.bLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(alphaLocalK.GetPhyAddr()));
        PipeBarrier<PIPE_V>();
        Transpose64x64UB(ictx.cLocal, ictx.bLocal);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(ictx.gmWsA1[k], ictx.cLocal, kMatElems);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
#endif
    }

    __aicore__ inline void FuseMaskStepASlot(
        uint32_t k, LocalTensor<float>& workK, LocalTensor<float>& outK,
        __ubuf__ float* alphaPtrK, InterleavedContext& ictx)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if (ictx.hasPair01 && (k <= 1U)) {
            FuseMaskAndInterStrideVF(
                reinterpret_cast<__ubuf__ float*>(ictx.cLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(ictx.outPairLocal.GetPhyAddr()) + k * 64U,
                128U,
                reinterpret_cast<__ubuf__ float*>(workK.GetPhyAddr()),
                alphaPtrK);
        } else {
            FuseMaskAndInterNoAddVF(
                reinterpret_cast<__ubuf__ float*>(ictx.cLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(outK.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(workK.GetPhyAddr()),
                alphaPtrK);
        }
#endif
    }

    __aicore__ inline void RunInterleavedStepASlot(
        uint32_t k, InterleavedContext& ictx)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        wait_intra_block(PIPE_V, 6U + 3U * k);
        LocalTensor<float> workLocalK = ictx.workAll[(2U * k + 0U) * kMatElems];
        LocalTensor<float> lLocalK = ictx.workAll[(2U * k + 1U) * kMatElems];
        LocalTensor<float> outLocalK = ictx.outAll[k * kMatElems];
        LocalTensor<float> alphaLocalK = ictx.alphaAll[k * 8U * kExpBufElems];
        __ubuf__ float* alphaPtrK = reinterpret_cast<__ubuf__ float*>(alphaLocalK.GetPhyAddr());

        Transpose64x64UB(ictx.cLocal, workLocalK);
        Transpose64x64UB(workLocalK, lLocalK);
        PipeBarrier<PIPE_V>();
        if (ictx.outWritePending && (k == 0U)) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            ictx.outWritePending = false;
        }
        FuseMaskStepASlot(k, workLocalK, outLocalK, alphaPtrK, ictx);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(ictx.gmWsA0[k], ictx.cLocal, kMatElems);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        set_intra_block(PIPE_MTE3, 4U + 3U * k);
        set_intra_block(PIPE_V, 5U + 3U * k);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
#endif
    }

    __aicore__ inline void ComputeAndScaleNextB(
        uint32_t k, uint32_t nextDtRem, LocalTensor<float>& alphaLocalK,
        __ubuf__ float* alphaPtrK, InterleavedContext& ictx)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        ComputeChunkAlpha64(nextDtRem + ictx.hArr[k], ictx.aVal[k], ictx.dtRawLocal, alphaLocalK);
        ScaleBChunk64OnlyVF(reinterpret_cast<__ubuf__ float*>(ictx.bLocal.GetPhyAddr()), alphaPtrK);
        PipeBarrier<PIPE_V>();
        Transpose64x64UB(ictx.cLocal, ictx.bLocal);
        PipeBarrier<PIPE_V>();
#endif
    }

    __aicore__ inline void RunInterleavedStepBNextPrep(
        uint32_t k, uint32_t t0, bool hasNextChunk, uint32_t nextDtRem, InterleavedContext& ictx)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if (k + 1U < ictx.K) {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            LoadCurrentU64(ictx.bArr[k + 1U], ictx.hArr[k + 1U], t0, ictx.uLocal);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        LocalTensor<float> alphaLocalK = ictx.alphaAll[k * 8U * kExpBufElems];
        __ubuf__ float* alphaPtrK = reinterpret_cast<__ubuf__ float*>(alphaLocalK.GetPhyAddr());
        if (hasNextChunk) {
            ComputeAndScaleNextB(k, nextDtRem, alphaLocalK, alphaPtrK, ictx);
        }
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        if (hasNextChunk) {
            set_intra_block(PIPE_MTE3, 4U + 3U * k);
            set_intra_block(PIPE_V, 5U + 3U * k);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(ictx.gmWsA1[k], ictx.cLocal, kMatElems);
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
#endif
    }

    __aicore__ inline void ExecInterleavedVectorUpdate(
        uint32_t k, InterleavedContext& ictx)
    {
        LocalTensor<float> stateLocalK = ictx.stateAll[k * kMatElems];
        LocalTensor<float> workLocalK = ictx.workAll[(2U * k + 0U) * kMatElems];
        LocalTensor<float> lLocalK = ictx.workAll[(2U * k + 1U) * kMatElems];
        LocalTensor<float> outLocalK = ictx.outAll[k * kMatElems];
        LocalTensor<float> alphaLocalK = ictx.alphaAll[k * 8U * kExpBufElems];
        LocalTensor<float> dLocalK = ictx.dAll[k * kMaxP];
        __ubuf__ float* alphaPtrK = reinterpret_cast<__ubuf__ float*>(alphaLocalK.GetPhyAddr());

        UpdateStateChunk64VF(
            reinterpret_cast<__ubuf__ float*>(stateLocalK.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(lLocalK.GetPhyAddr()),
            alphaPtrK);
        if (ictx.hasPair01 && (k <= 1U)) {
            AccumOutAndUDStrideVF(
                reinterpret_cast<__ubuf__ float*>(ictx.outPairLocal.GetPhyAddr()) + k * 64U, 128U,
                reinterpret_cast<__ubuf__ float*>(workLocalK.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(ictx.uLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(dLocalK.GetPhyAddr()));
        } else {
            AccumOutAndUDStrideVF(
                reinterpret_cast<__ubuf__ float*>(outLocalK.GetPhyAddr()), 64U,
                reinterpret_cast<__ubuf__ float*>(workLocalK.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(ictx.uLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(dLocalK.GetPhyAddr()));
        }
    }

    __aicore__ inline void RunInterleavedStepBSlot(
        uint32_t k, uint32_t t0, bool hasNextChunk, InterleavedContext& ictx)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if (k == 0U) {
            LoadCurrentU64(ictx.bArr[0], ictx.hArr[0], t0, ictx.uLocal);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        uint32_t nextDtRem = 0U;
        if (hasNextChunk) {
            const bool loadDt = (k == 0U) || (ictx.bArr[k] != ictx.bArr[k - 1U]);
            StepSlotInfo slotInfo{ictx.bArr[k], ictx.hArr[k], t0 + kChunkSize, loadDt};
            nextDtRem = IssueLoadDtAndCopyB0_64(
                slotInfo, ictx.bLocal, ictx.dtRawLocal, ictx.gmWsB0[k]);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        wait_intra_block(PIPE_V, 6U + 3U * k);

        ExecInterleavedVectorUpdate(k, ictx);

        LocalTensor<float> stateLocalK = ictx.stateAll[k * kMatElems];
        PipeBarrier<PIPE_V>();
        Transpose64x64UB(ictx.uLocal, stateLocalK);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(ictx.gmWsB1[k], ictx.uLocal, kMatElems);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        RunInterleavedStepBNextPrep(k, t0, hasNextChunk, nextDtRem, ictx);
#endif
    }

    __aicore__ inline void FlushInterleavedChunkOut(
        uint32_t t0, InterleavedContext& ictx)
    {
        const uint16_t uBlocks = static_cast<uint16_t>(headdim_ / 8U);
        DataCopyParams outCopy{64U, uBlocks, 0U, static_cast<uint16_t>((nheads_ - 1U) * uBlocks)};
        if (ictx.hasPair01) {
            DataCopyParams outPairCopy{64U, static_cast<uint16_t>(2U * uBlocks), 0U, static_cast<uint16_t>((nheads_ - 2U) * uBlocks)};
            const uint64_t uBase0 = ((static_cast<uint64_t>(ictx.bArr[0]) * seqLen_ + t0) * nheads_ + ictx.hArr[0]) * headdim_;
            DataCopy(gmOut_[uBase0], ictx.outPairLocal, outPairCopy);
            if (ictx.K == 3U) {
                const uint64_t uBase2 = ((static_cast<uint64_t>(ictx.bArr[2]) * seqLen_ + t0) * nheads_ + ictx.hArr[2]) * headdim_;
                DataCopy(gmOut_[uBase2], ictx.outAll[2U * kMatElems], outCopy);
            }
        } else {
            for (uint32_t k = 0; k < ictx.K; ++k) {
                const uint64_t uBase = ((static_cast<uint64_t>(ictx.bArr[k]) * seqLen_ + t0) * nheads_ + ictx.hArr[k]) * headdim_;
                DataCopy(gmOut_[uBase], ictx.outAll[k * kMatElems], outCopy);
            }
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        ictx.outWritePending = true;
    }

    __aicore__ inline void InitInterleavedContext(GM_ADDR userWs, InterleavedContext& ictx)
    {
        ictx.stateAll = stateBuf_.Get<float>(3U * kMatElems);
        ictx.outAll = outBuf_.Get<float>(3U * kMatElems);
        ictx.workAll = workBuf_.Get<float>(6U * kMatElems);
        ictx.uLocal = uBuf_.Get<float>(kMatElems);
        ictx.bLocal = bBuf_.Get<float>(kMatElems);
        ictx.cLocal = cBuf_.Get<float>(kMatElems);
        ictx.alphaAll = alphaBuf_.Get<float>(24U * kExpBufElems);
        ictx.dAll = dBuf_.Get<float>(3U * kMaxP);
        ictx.dtRawLocal = dtRawBuf_.Get<float>(kMaxDtRawElems);

        __gm__ float* coreWs = reinterpret_cast<__gm__ float*>(userWs) + static_cast<uint64_t>(coreId_) * kPerCoreWsFloats;
        for (uint32_t k = 0; k < 3U; ++k) {
            ictx.gmWsA0[k].SetGlobalBuffer(coreWs + (4U * k + 0U) * kMatElems, kMatElems);
            ictx.gmWsA1[k].SetGlobalBuffer(coreWs + (4U * k + 1U) * kMatElems, kMatElems);
            ictx.gmWsB0[k].SetGlobalBuffer(coreWs + (4U * k + 2U) * kMatElems, kMatElems);
            ictx.gmWsB1[k].SetGlobalBuffer(coreWs + (4U * k + 3U) * kMatElems, kMatElems);
        }
    }

    __aicore__ inline void ProcessInterleavedTimeChunk(
        uint32_t t0, uint32_t K, InterleavedContext& ictx)
    {
        const bool hasNextChunk = (t0 + kChunkSize < seqLen_);
        for (uint32_t k = 0; k < K; ++k) {
            RunInterleavedStepASlot(k, ictx);
        }
        for (uint32_t k = 0; k < K; ++k) {
            RunInterleavedStepBSlot(k, t0, hasNextChunk, ictx);
        }
        FlushInterleavedChunkOut(t0, ictx);
    }

    __aicore__ inline bool ProcessInterleavedGroup(
        uint32_t groupBase, InterleavedContext& ictx)
    {
        const uint32_t K = InitInterleavedGroupSlots(groupBase, ictx);
        if (K == 0U) return false;
        ictx.hasPair01 = (K >= 2U) && (ictx.bArr[0] == ictx.bArr[1]) && (ictx.hArr[1] == ictx.hArr[0] + 1U);
        ictx.outWritePending = false;
        ictx.outPairLocal = LocalTensor<float>{TPosition::VECCALC, static_cast<uint32_t>(ictx.outAll.GetPhyAddr()), 2U * kMatElems};
        for (uint32_t k = 0; k < K; ++k) {
            RunInterleavedPrologueSlot(k, ictx);
        }
        for (uint32_t t0 = 0; t0 < seqLen_; t0 += kChunkSize) {
            ProcessInterleavedTimeChunk(t0, K, ictx);
        }
        if (ictx.outWritePending) { WaitFlag<HardEvent::MTE3_V>(EVENT_ID1); }
        return true;
    }

    __aicore__ inline void ProcessInterleavedMix64(GM_ADDR userWs)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if ASCEND_IS_AIC { return; }
        const uint32_t totalHeads = batch_ * nheads_;
        InterleavedContext ictx;
        InitInterleavedContext(userWs, ictx);

        for (uint32_t groupBase = 0; groupBase < totalHeads; groupBase += 3U * coreNum_) {
            if (!ProcessInterleavedGroup(groupBase, ictx)) break;
        }
        pipe_barrier(PIPE_ALL);
#endif
    }

    __aicore__ inline void Process(CubeMatmulObj& matmulObj, GM_ADDR userWs)
    {
        if ASCEND_IS_AIC {
            return;
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if ((headdim_ == 64U) && (dstate_ == 64U) && ((seqLen_ % 64U) == 0U)) {
            ProcessInterleavedMix64(userWs);
            return;
        }
#endif
        if (nheads_ == 0U) {
            return;
        }
        const uint32_t totalHeads = batch_ * nheads_;
        for (uint32_t taskIdx = coreId_; taskIdx < totalHeads; taskIdx += coreNum_) {
            const uint32_t b = taskIdx / nheads_;
            const uint32_t h = taskIdx % nheads_;
            ProcessHeadSequence(matmulObj, b, h);
        }
    }

private:
    TPipe* pipe_{nullptr};
    TBuf<TPosition::VECCALC> stateBuf_;
    TBuf<TPosition::VECCALC> uBuf_;
    TBuf<TPosition::VECCALC> bBuf_;
    TBuf<TPosition::VECCALC> cBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> lBuf_;
    TBuf<TPosition::VECCALC> dBuf_;
    TBuf<TPosition::VECCALC> dtRawBuf_;
    TBuf<TPosition::VECCALC> alphaBuf_;

    uint32_t batch_{0};
    uint32_t seqLen_{0};
    uint32_t nheads_{0};
    uint32_t headdim_{0};
    uint32_t dstate_{0};
    uint32_t coreId_{0};
    uint32_t coreNum_{1};

    GlobalTensor<float> gmU_;
    GlobalTensor<float> gmDt_;
    GlobalTensor<float> gmA_;
    GlobalTensor<float> gmB_;
    GlobalTensor<float> gmC_;
    GlobalTensor<float> gmD_;
    GlobalTensor<float> gmOut_;
    GlobalTensor<float> gmWsA0_;
    GlobalTensor<float> gmWsA1_;
    GlobalTensor<float> gmWsB0_;
    GlobalTensor<float> gmWsB1_;
    GlobalTensor<float> gmWsB2_;
    GlobalTensor<float> gmWsC0_;
    GlobalTensor<float> gmWsC1_;
    GlobalTensor<float> gmWsU_;
    GlobalTensor<float> gmWsTril_;
    GlobalTensor<float> gmWsCdec0_;
    GlobalTensor<float> gmWsCdec1_;
    GlobalTensor<float> gmWsBdec0_;
    GlobalTensor<float> gmWsBdec1_;
    GlobalTensor<float> gmWsState1_;
};
}  // namespace

#if !defined(HAVE_WORKSPACE) && !defined(GET_TILING_DATA)
__aicore__ inline GM_ADDR GetSysWorkSpacePtr()
{
    return nullptr;
}

__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
{
    return workspace;
}
#endif

extern "C" __global__ __aicore__ void mamba2_ssd_chunk_scan(
    GM_ADDR u, GM_ADDR dt, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    GM_ADDR sysWs = GetSysWorkSpacePtr();
    GM_ADDR userWs = GetUserWorkspace(workspace);
    TCubeTiling localTiling;
    const __gm__ Mamba2SsdTiling* tData = reinterpret_cast<const __gm__ Mamba2SsdTiling*>(tiling);
    const bool isDirectDualMaster = (tData->headdim == 64U) && (tData->dstate == 64U) && ((tData->seq_len % 64U) == 0U);
    CubeMatmulObj matmulObj;
    KernelGmAddrs addrs{u, dt, a, b, c, d, out, userWs, tiling};
    if (isDirectDualMaster) {
        FillLocalCubeTiling128x64(localTiling);
        KernelMamba2SsdChunkScan op;
        op.Init(addrs, &pipe);
#ifdef SPLIT_CORE_CUBE
        AscendC::TQue<AscendC::QuePosition::CO1, 1, &AscendC::gCO1Config> qCO1;
        AscendC::gCO1Que = &qCO1;
        AscendC::KfcCommServer<false> kfcSrv;
        kfcSrv.Init(sysWs, 0);
        matmulObj.cubeObj.cubeObj[0].InitKfc(&pipe, &localTiling, &kfcSrv, 0, sysWs);
        op.ProcessDirectAIC(matmulObj, userWs);
#elif defined(SPLIT_CORE_VEC)
        op.Process(matmulObj, userWs);
#endif
        return;
    }
    FillLocalCubeTiling(localTiling);
    REGIST_MATMUL_OBJ(&pipe, sysWs, matmulObj, &localTiling);

    KernelMamba2SsdChunkScan op;
    op.Init(addrs, &pipe);
    op.Process(matmulObj, userWs);
}
