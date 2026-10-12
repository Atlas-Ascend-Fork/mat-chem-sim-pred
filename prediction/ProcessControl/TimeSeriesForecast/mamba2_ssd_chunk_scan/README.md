<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Mamba2SsdChunkScan (`aclnnMamba2SsdChunkScan`) — 昇腾 910B & 950 Cube-Vector 异构融合算子

## 1. 算子概述

`Mamba2SsdChunkScan` 是面向长序列时序预测模型（如 **S-Mamba**、**TimeMachine**、**Mamba-2 SSD**）构建的 **Ascend C Cube-Vector 异构融合算子（`KERNEL_TYPE_MIX_AIC_1_2`）**，同时支持 **昇腾 910B (`ascend910b` / `dav-c220`)** 与 **昇腾 950 (`ascend950` / `dav-c310`, `Ascend950PR_9579`)** 双代架构。

该算子将 Mamba-2 **状态空间对偶（State Space Duality, SSD）** 分块扫描中的 **4 阶段异构计算**——块内 $C B^T$ 语法耦合矩阵乘（AI Cube）、一维指数衰减前缀和与下三角因果掩码（AI Vector）、块内状态注入与对偶输出矩阵乘（AI Cube）、以及跨块隐状态递推更新（AI Vector）——全部融合在**单次 `aclnnMamba2SsdChunkScan` 核函数调用**内：
- **昇腾 950 (`dav-c310`) Direct AIC/AIV 硬件流融合**：
  - 采用 `set_intra_block / wait_intra_block` 核内硬件信号直连替代软件 `MatmulClient` Kfc 握手；
  - 将 4 路 $64 \times 64$ 矩阵乘纵向拼接为 **2 次 $128 \times 64 \times 64$ 融合 `Mmad`**，配合 `Fixpipe<float, float, CFG_ROW_MAJOR_UB>` 从 L0C 直接写回 AIV UB，省去 `CO1 -> GM -> UB` 往返；
  - **Batch-Coalesced 4-Head 连续 `Nd2Nz`**：将同一 batch 的 4 个 head (`h=0..3`) 映射至同一物理 AIC 核，由 `subId=0, k=0` 单次 `100%` 连续突发加载 $[64, 256]$ 的 `gmC_` / `gmU_` 进 `B1` 全部 4 个槽位，其余子核与槽位实现 **零 `DataCopy(bL1)` 开销**；
  - **跨 Head 交织写回 + `__simd_vf__` 全寄存器向量内联**：将相邻 head 输出交织排布于 $[64, 128]$ `outPairLocal` 缓冲，以 `512B` 连续块突发写回 `gmOut_` 并与下一 chunk 的 `Matmul 1` 异步重叠。
- **昇腾 910B (`dav-c220`) 异步乒乓流水**：
  - 将跨分块隐状态 $H_{\text{prev}} \in \mathbb{R}^{N \times P}$ 常驻于片上 `192 KB` Unified Buffer (UB)，配合 14 槽位片上 L2 Workspace 实现 **4 级（$N=64$）/ 7 级（$N=128$）异步乒乓流水**。

---

## 2. 接口原型与张量规格

### C++ `aclnn` 两段式接口
```cpp
aclnnStatus aclnnMamba2SsdChunkScanGetWorkspaceSize(
    const aclTensor *x,           // [B, L, H, P]       FP16 / FP32, ND
    const aclTensor *dt,          // [B, L, H]          FP16 / FP32, ND
    const aclTensor *A,           // [H] or [H, N]      FP16 / FP32, ND
    const aclTensor *B,           // [B, L, H, N]       FP16 / FP32, ND
    const aclTensor *C,           // [B, L, H, N]       FP16 / FP32, ND
    const aclTensor *D,           // [H] or [H, P]      FP16 / FP32, ND
    int64_t chunkSize,            // Q (典型值 64)
    const aclTensor *yOut,        // [B, L, H, P]       FP16 / FP32, ND
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnMamba2SsdChunkScan(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

---

## 3. 昇腾 950 (`Ascend950PR_9579`) 与昇腾 910B 真机实测性能总表

在 **华为昇腾 950（`Ascend950PR_9579`, `dav-c310`, 28 AIC + 56 AIV, CANN 9.0.0-beta.2）** 与 **华为昇腾 910B（`ascend910b`, CANN 9.0.0，双卡复验）** 上的真机实测结果如下（全部 4 组形状的**完整输出张量 100% 元素**通过标量递推精度复核，最大绝对误差 `<= 2.23517e-7`）：

| 场景名称 | 形状 `(B, L, H, P, N, Q)` | 950 纯 Vector 算子 (`ms`) | 950 初代 MIX 算子 (`ms`) | **950 最终版 Direct MIX (`ms`)** | **950 相对纯 Vector 加速比** | **910B 双卡复验均值 (`ms`)** | **全张量 `max_abs_err`** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S-Mamba TS Standard** | `(32, 512, 4, 64, 64, 64)` | `61.48 ms` | `9.70 ms` | **`0.5953 ms`** *(核纯流 `0.5859 ms`)* | **`103.28×`** | `0.5971 ms` | **`1.49012e-07`** |
| **Long-Horizon TS** | `(16, 1024, 8, 64, 64, 64)` | `126.27 ms` | `23.18 ms` | **`2.7064 ms`** | **`46.66×`** | `1.5404 ms` | **`1.63913e-07`** |
| **Ultra-Long Horizon TS** | `(8, 2048, 8, 64, 64, 64)` | `126.10 ms` | `23.19 ms` | **`2.7359 ms`** | **`46.09×`** | `1.5506 ms` | **`1.63913e-07`** |
| **High-Capacity State** | `(16, 512, 8, 64, 128, 64)` | `121.99 ms` | `19.78 ms` | **`2.8854 ms`** | **`42.28×`** | `1.4683 ms` | **`2.23517e-07`** |

---

## 4. 目录结构与交付件清单

```text
mamba2_ssd_chunk_scan/
├── CMakeLists.txt                                      # 顶层构建工程文件
├── README.md                                           # 算子主说明文档
├── docs/
│   ├── algorithm.md                                    # SSD 数学原理与 910B/950 硬件流水映射设计
│   ├── api_reference.md                                # ACLNN 两段式接口与参数约束参考
│   ├── benchmark.md                                    # 昇腾 950 & 910B 性能评测与优化演进报告
│   └── test_report.md                                  # Python 单元测试、C++ 冒烟与全张量精度测试报告
├── examples/
│   └── test_aclnn_mamba2_ssd_chunk_scan.cpp            # C++ ACLNN 端到端调用与精度验证示例
├── op_host/
│   ├── mamba2_ssd_chunk_scan_def.cpp                   # 算子 IR 原型注册（支持 ascend910b & ascend950）
│   ├── mamba2_ssd_chunk_scan_host.cpp                  # Host Tiling 计算与 Workspace 规划
│   ├── mamba2_ssd_chunk_scan_host.h                    # Host 侧头文件
│   └── mamba2_ssd_chunk_scan_tiling.h                  # Tiling 数据结构定义
├── op_kernel/
│   └── mamba2_ssd_chunk_scan_kernel.cpp                # Device 侧 Ascend C Cube-Vector 异构融合核函数
├── reference/
│   └── reference.py                                    # Python 黄金标量/分块递推参考实现
└── tests/
    ├── CMakeLists.txt                                  # 测试工程构建配置
    ├── benchmark_mamba2_ssd_chunk_scan_aclnn.cpp       # 四大时序基准用例全张量精度与延迟评测程序
    └── test_mamba2_ssd_chunk_scan_reference.py         # Pytest / Unittest 黄金参考单元测试
```
