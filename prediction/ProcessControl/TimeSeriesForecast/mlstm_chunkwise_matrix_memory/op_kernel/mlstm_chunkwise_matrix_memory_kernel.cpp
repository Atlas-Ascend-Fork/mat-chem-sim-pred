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

namespace {
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 1024;

class KernelMlstmChunkwiseMatrixMemory {
public:
    __aicore__ inline KernelMlstmChunkwiseMatrixMemory() = default;

    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR i_gate, GM_ADDR f_gate, GM_ADDR out,
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
            return;
        }
        scaleFactor = 1.0f / static_cast<float>(static_cast<int32_t>(attrParam > 0 ? attrParam : 64));
        qGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(q) + offset, coreLen);
        kGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(k) + offset, coreLen);
        vGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v) + offset, coreLen);
        i_gateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(i_gate) + offset, coreLen);
        f_gateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(f_gate) + offset, coreLen);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out) + offset, coreLen);

        pipe.InitBuffer(inQueueA, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueC, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueD, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueE, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (coreLen == 0) {
            return;
        }
        uint32_t loopCount = (coreLen + TILE_LENGTH - 1) / TILE_LENGTH;
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t curOffset = i * TILE_LENGTH;
            uint32_t curLen = (coreLen - curOffset < TILE_LENGTH) ? (coreLen - curOffset) : TILE_LENGTH;
            CopyIn(curOffset, curLen);
            Compute(curLen);
            CopyOut(curOffset, curLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
        AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
        AscendC::DataCopy(aLocal, qGm[offset], length);
        inQueueA.EnQue(aLocal);
        AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
        AscendC::DataCopy(bLocal, kGm[offset], length);
        inQueueB.EnQue(bLocal);
        AscendC::LocalTensor<float> cLocal = inQueueC.AllocTensor<float>();
        AscendC::DataCopy(cLocal, vGm[offset], length);
        inQueueC.EnQue(cLocal);
        AscendC::LocalTensor<float> dLocal = inQueueD.AllocTensor<float>();
        AscendC::DataCopy(dLocal, i_gateGm[offset], length);
        inQueueD.EnQue(dLocal);
        AscendC::LocalTensor<float> eLocal = inQueueE.AllocTensor<float>();
        AscendC::DataCopy(eLocal, f_gateGm[offset], length);
        inQueueE.EnQue(eLocal);
    }

    __aicore__ inline void Compute(uint32_t length) {
        AscendC::LocalTensor<float> aLocal = inQueueA.DeQue<float>();
        AscendC::LocalTensor<float> bLocal = inQueueB.DeQue<float>();
        AscendC::LocalTensor<float> cLocal = inQueueC.DeQue<float>();
        AscendC::LocalTensor<float> dLocal = inQueueD.DeQue<float>();
        AscendC::LocalTensor<float> eLocal = inQueueE.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        AscendC::Sub(tmpLocal, dLocal, eLocal, length);
        AscendC::Muls(tmpLocal, tmpLocal, scaleFactor, length);
        AscendC::Exp(tmpLocal, tmpLocal, length);
        AscendC::Mul(yLocal, aLocal, cLocal, length);
        AscendC::Mul(yLocal, yLocal, tmpLocal, length);
        AscendC::Mul(tmpLocal, bLocal, cLocal, length);
        AscendC::Muls(tmpLocal, tmpLocal, scaleFactor, length);
        AscendC::Add(yLocal, yLocal, tmpLocal, length);
        outQueue.EnQue<float>(yLocal);
        inQueueA.FreeTensor(aLocal);
        inQueueB.FreeTensor(bLocal);
        inQueueC.FreeTensor(cLocal);
        inQueueD.FreeTensor(dLocal);
        inQueueE.FreeTensor(eLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
        AscendC::LocalTensor<float> yLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(outGm[offset], yLocal, length);
        outQueue.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueA;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueB;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueC;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueD;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueE;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> i_gateGm;
    AscendC::GlobalTensor<float> f_gateGm;
    AscendC::GlobalTensor<float> outGm;
    uint32_t coreLen{0};
    float scaleFactor{0.015625f};
};
} // namespace

extern "C" __global__ __aicore__ void mlstm_chunkwise_matrix_memory(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR i_gate, GM_ADDR f_gate, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelMlstmChunkwiseMatrixMemory op;
    op.Init(q, k, v, i_gate, f_gate, out, tilingData.totalLength, tilingData.attrParam);
    op.Process();
}
