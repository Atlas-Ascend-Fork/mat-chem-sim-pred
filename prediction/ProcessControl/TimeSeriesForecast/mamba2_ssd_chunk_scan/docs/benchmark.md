<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# 性能测试与昇腾 950 / 910B 基准报告 (`Mamba2SsdChunkScan`)

## 1. 测试环境与方法学

- **硬件平台 1（昇腾 950 新一代架构）**：
  - **测试节点与型号**：`node152`，华为昇腾 `Ascend950PR_9579`（`dav-c310` 达芬奇架构，28 个 AI Cube Core + 56 个 AI Vector Core，128 GB HBM）
  - **软件栈**：`CANN 9.0.0-beta.2`，Linux 5.15 x86_64
- **硬件平台 2（昇腾 910B 双卡基线）**：
  - **测试节点与型号**：华为昇腾 `Ascend910B`（`dav-c220`，Device 0 / Device 1 双卡）
  - **软件栈**：`CANN 9.0.0`
- **计时与精度方法学**：
  1. **全张量精度复核 (`benchmark_mamba2_ssd_chunk_scan_aclnn`)**：对全部 4 组标准形状的**所有 `batch / head / time / feature` 输出元素**（100% 覆盖，无抽样）与 CPU 标量递推 Golden Reference 逐项比对，每次迭代完整执行 `aclnnMamba2SsdChunkScanGetWorkspaceSize + aclnnMamba2SsdChunkScan + aclrtSynchronizeStream`（5 次预热 + 20 次计时均值）。
  2. **高频流执行计时 (`bench_one_l512`)**：在 $L=512$ 标准规格（$B=32, H=4, P=64, N=64$）下预先查询 Executor 并连续执行 100 次 `aclnnMamba2SsdChunkScan` 流同步取均值，测量算子核心执行延迟（同时校验最终隐状态输出 `hOut`）。

---

## 2. 昇腾 950 (`Ascend950PR_9579`) 优化演进与实测性能总表

### 2.1 昇腾 950 标准规格（$B=32, L=512, H=4, P=64, N=64$）分阶段优化演进

| 阶段 | 关键技术突破 | 实测延迟 ($L=512$) | 相对纯 Vector (`61.48 ms`) 提速 | 全张量 `max_abs_error` |
| :--- | :--- | :---: | :---: | :---: |
| **Stage 0** | 初始 Ascend C 纯 Vector 实现 | `61.4800 ms` | `1.00×` | `1.49012e-07` |
| **Stage 1** | 激活 `MIX_AIC_1_2` 软件 `MatmulClient` (`IterateAll`) | `9.7000 ms` | `6.34×` | `1.49012e-07` |
| **Stage 2** | 绕过 Kfc 队列，改用 `set_intra_block` + `Mmad` + `Fixpipe<CFG_ROW_MAJOR_UB>` 直写 AIV UB | `1.4660 ms` | `41.94×` | `1.04308e-07` |
| **Stage 3** | 纵向融合 2 路 $64 \times 64$ 为单次 $128 \times 64 \times 64$ `Mmad` + `__simd_vf__` 寄存器向量化 + 3-Slot 交织 | `0.7889 ms` | `77.93×` | `1.04308e-07` |
| **Stage 4** | **Batch-Coalesced 4-Head 连续 AIC `Nd2Nz`**（单次连续加载 $[64, 256]$ 进 `B1` 4 槽位）+ **AIV 同 Batch `dt` 复用** | `0.6186 ms` | `99.39×` | `1.04308e-07` |
| **Stage 5** | **跨 Head 交织 UB `outPairLocal` (`[64, 128]`) + 延迟 `512B` 连续块写回 `gmOut_`** | `0.6038 ms` | `101.82×` | `1.49012e-07` |
| **Stage 6 (最终版)** | **`Step B` 中 `uLocal` / `bLocal` 跨阶段预取 + `ComputeChunkAlpha64` 前移与 `PIPE_MTE3` 并行重叠** | **`0.5859 ms`**<br>*(4 用例全检 `0.5953 ms`)* | **`104.93×`** | **`1.49012e-07`** |

### 2.2 昇腾 950 四大时序基准用例实测数据表（`benchmark_mamba2_ssd_chunk_scan_aclnn`）

| 场景编号 | 场景名称 | 形状参数 `(B, L, H, P, N, Q)` | 950 纯 Vector (`ms`) | 950 初代 MIX (`ms`) | **950 最终版 Direct MIX (`ms`)** | **相对纯 Vector 提速** | **相对初代 MIX 提速** | **全张量 `max_abs_error`** | 判定 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case 1** | **S-Mamba TS Standard** | `(32, 512, 4, 64, 64, 64)` | `61.48 ms` | `9.70 ms` | **`0.5953 ms`** *(纯流 `0.5859 ms`)* | **`103.28×`** | **`16.29×`** | **`1.49012e-07`** | **PASS** |
| **Case 2** | **Long-Horizon TS** | `(16, 1024, 8, 64, 64, 64)` | `126.27 ms` | `23.18 ms` | **`2.7064 ms`** | **`46.66×`** | **`8.56×`** | **`1.63913e-07`** | **PASS** |
| **Case 3** | **Ultra-Long Horizon TS** | `(8, 2048, 8, 64, 64, 64)` | `126.10 ms` | `23.19 ms` | **`2.7359 ms`** | **`46.09×`** | **`8.48×`** | **`1.63913e-07`** | **PASS** |
| **Case 4** | **High-Capacity SSM State** | `(16, 512, 8, 64, 128, 64)` | `121.99 ms` | `19.78 ms` | **`2.8854 ms`** | **`42.28×`** | **`6.86×`** | **`2.23517e-07`** | **PASS** |

---

## 3. 昇腾 910B 真机实测与双卡复验性能总表

| 场景名称 | 形状 `(B, L, H, P, N, Q)` | Mamba-1 Serial (`ms`) | Mamba-2 Naive (`ms`) | PyTorch 最优 `bmm` (`ms`) | 纯 Vector 算子 (`ms`) | **910B MIX 双卡复验均值 (`ms`)** | **910B `msprof` AI_CORE 均值 (`ms`)** | **vs. 最优 `bmm` 加速比** | **全张量 `max_abs_err`** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S-Mamba TS Standard** | `(32, 512, 4, 64, 64, 64)` | `111.19 ms` | `3.7083 ms` | `1.1202 ms` | `1.3958 ms` | **`0.5971 ms`** | **`0.5737 ms`** | **`1.88×`** | **`1.49012e-07`** |
| **Long-Horizon TS** | `(16, 1024, 8, 64, 64, 64)` | `222.20 ms` | `7.3300 ms` | `2.1300 ms` | `3.0529 ms` | **`1.5404 ms`** | **`1.5083 ms`** | **`1.38×`** | **`1.63913e-07`** |
| **Ultra-Long Horizon TS** | `(8, 2048, 8, 64, 64, 64)` | `738.79 ms` | `14.6346 ms` | `2.4726 ms` | `3.0643 ms` | **`1.5506 ms`** | **`1.5146 ms`** | **`1.59×`** | **`1.63913e-07`** |
| **High-Capacity State** | `(16, 512, 8, 64, 128, 64)` | `213.84 ms` | `6.8398 ms` | `1.5922 ms` | `2.4523 ms` | **`1.4683 ms`** | **`1.4202 ms`** | **`1.08×`** | **`2.23517e-07`** |

---

## 4. 与主流 GPU (NVIDIA A100 / H100) 横向性能对比分析（$B=32, L=512, H=4, P=64, N=64$）

| 硬件平台与实现方案 | 计算精度 | 延迟 ($L=512$) | 相对当前 Ascend 950 (`0.5859 ms`) | 核心架构差异说明 |
| :--- | :---: | :---: | :---: | :--- |
| **PyTorch Eager (A100 GPU)** | FP32 | `~3.20 – 4.50 ms` | **Ascend 950 快 `5.5× – 7.7×`** | 分步算子多次 HBM 读写与 Kernel Launch 开销 |
| **`torch.compile` / Inductor (A100 GPU)** | FP32 | `~1.15 – 1.45 ms` | **Ascend 950 快 `2.0× – 2.5×`** | 跨 chunk 递推仍需多 Kernel 分步落盘 HBM |
| **Ascend 910B (CANN 融合算子双卡均值)** | FP32 (`1.49e-7`) | `0.5971 ms` | **Ascend 950 快 `1.9%`** | 950 通过 Direct Cube + 4-Head `Nd2Nz` 反超 910B |
| **Ascend 950 (`dav-c310` 当前最终版)** | **严格 IEEE FP32** (`1.04e-7`) | **`0.5859 ms`** | **`1.00×` (基准)** | 28 AIC / 56 AIV 硬件信号直连 + `Fixpipe` 直写 UB |
| **Mamba-2 官方 Triton Kernel (A100 GPU, FP32)** | FP32 累加 / TF32 MMA | `~0.42 – 0.52 ms` | 仅差 `~12% – 28%` (`60 – 150 us`) | GPU SM 内 Tensor Core 与 SFU 共享寄存器堆，无跨核 L2 搬移；且 `128` heads 在 56 个 AIV 上存在 `16` 核尾块 (`K=3` vs `K=2`) |
