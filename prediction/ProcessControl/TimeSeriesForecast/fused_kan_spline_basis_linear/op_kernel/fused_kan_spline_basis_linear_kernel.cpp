// ----------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#include "kernel_operator.h"

namespace {
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 1024;

class KernelFusedKanSplineBasisLinear {
public:
    __aicore__ inline KernelFusedKanSplineBasisLinear() = default;

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR grid, GM_ADDR spline_weight, GM_ADDR out,
        uint32_t totalLength, uint32_t attrParam) {
        uint32_t blockNum = AscendC::GetBlockNum();
        if (blockNum == 0U) {
            blockNum = 1U;
        }
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t avgLen = (totalLength + blockNum - 1) / blockNum;
        avgLen = ((avgLen + TILE_LENGTH - 1) / TILE_LENGTH) * TILE_LENGTH;
        uint32_t offset = blockIdx * avgLen;
        coreLen = (offset < totalLength) ? ((offset + avgLen <= totalLength) ? avgLen : (totalLength - offset)) : 0;
        if (coreLen == 0) {
            coreLen = TILE_LENGTH;
            offset = 0;
        }
        scaleFactor = 1.0f / static_cast<float>(static_cast<int32_t>(attrParam > 0 ? attrParam : 16));
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + offset, coreLen);
        gridGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(grid) + offset, coreLen);
        spline_weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(spline_weight) + offset, coreLen);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out) + offset, coreLen);

        pipe.InitBuffer(inQueueA, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueC, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process() {
        uint32_t loopCount = (coreLen + TILE_LENGTH - 1) / TILE_LENGTH;
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t curOffset = i * TILE_LENGTH;
            CopyIn(curOffset);
            Compute();
            CopyOut(curOffset);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset) {
        AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
        AscendC::DataCopy(aLocal, xGm[offset], TILE_LENGTH);
        inQueueA.EnQue(aLocal);
        AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
        AscendC::DataCopy(bLocal, gridGm[offset], TILE_LENGTH);
        inQueueB.EnQue(bLocal);
        AscendC::LocalTensor<float> cLocal = inQueueC.AllocTensor<float>();
        AscendC::DataCopy(cLocal, spline_weightGm[offset], TILE_LENGTH);
        inQueueC.EnQue(cLocal);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<float> aLocal = inQueueA.DeQue<float>();
        AscendC::LocalTensor<float> bLocal = inQueueB.DeQue<float>();
        AscendC::LocalTensor<float> cLocal = inQueueC.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        AscendC::Sub(tmpLocal, aLocal, bLocal, TILE_LENGTH);
        AscendC::Muls(tmpLocal, tmpLocal, scaleFactor, TILE_LENGTH);
        AscendC::Mul(yLocal, tmpLocal, tmpLocal, TILE_LENGTH);
        AscendC::Add(yLocal, yLocal, tmpLocal, TILE_LENGTH);
        AscendC::Mul(yLocal, yLocal, cLocal, TILE_LENGTH);
        AscendC::Add(yLocal, yLocal, aLocal, TILE_LENGTH);
        outQueue.EnQue<float>(yLocal);
        inQueueA.FreeTensor(aLocal);
        inQueueB.FreeTensor(bLocal);
        inQueueC.FreeTensor(cLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset) {
        AscendC::LocalTensor<float> yLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(outGm[offset], yLocal, TILE_LENGTH);
        outQueue.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueA;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueB;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueC;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> gridGm;
    AscendC::GlobalTensor<float> spline_weightGm;
    AscendC::GlobalTensor<float> outGm;
    uint32_t coreLen{1024};
    float scaleFactor{0.0625f};
};
} // namespace

extern "C" __global__ __aicore__ void fused_kan_spline_basis_linear(
    GM_ADDR x, GM_ADDR grid, GM_ADDR spline_weight, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelFusedKanSplineBasisLinear op;
    op.Init(x, grid, spline_weight, out, tilingData.totalLength, tilingData.attrParam);
    op.Process();
}
