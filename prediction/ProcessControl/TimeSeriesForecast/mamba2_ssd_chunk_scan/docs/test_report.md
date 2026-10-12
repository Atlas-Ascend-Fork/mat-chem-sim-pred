<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Mamba2SsdChunkScan 测试与全输出精度验证报告（昇腾 950 & 910B）

## 1. 测试概览

本报告记录 `Mamba2SsdChunkScan` **Ascend C Cube-Vector 异构融合算子 (`KERNEL_TYPE_MIX_AIC_1_2`)** 在 **华为昇腾 950 (`Ascend950PR_9579`, `dav-c310`, `CANN 9.0.0-beta.2`)** 与 **华为昇腾 910B (`ascend910b`, `CANN 9.0.0`)** 上的编译构建、Python 单元测试、C++ `aclnn` 冒烟门禁（Smoke Gate）以及 4 组大规模时序形状的**全张量逐元素精度**与性能验证结果。

---

## 2. 昇腾 950 (`Ascend950PR_9579`) 真机门禁与四用例全量复验

### 2.1 验证层级与执行判定
1. **Python Reference 单元测试 (`tests/test_mamba2_ssd_chunk_scan_reference.py`)**：**6/6 通过 (`OK`)**。
2. **CMake 全量构建与算子二进制安装**：
   - 构建目标：`Mamba2SsdChunkScan_ascend950` + `binary`
   - 安装路径：`build_formal/packages/vendors/customize/op_impl/ai_core/tbe/kernel/ascend950/mamba2_ssd_chunk_scan/`
3. **C++ `aclnn` 独立冒烟门禁 (`test_aclnn_mamba2_ssd_chunk_scan`)**：
   - 测试配置：`B=2, L=64, H=2, P=16, N=16, Q=64`（覆盖通用非对齐/小维度 Fallback 分支）
   - 实测误差：`max_abs_error = 8.9407e-08`，`max_rel_error = 3.46281e-05`（远低于门禁阈值 $10^{-4}$）
   - 判定：**PASS**。
4. **C++ Direct Cube 专项验证 (`verify_cube_p64` & `bench_one_l512`)**：
   - `verify_cube_p64` (`P=64, N=64`)：`max_abs_error = 1.04308e-07`（**PASS**）
   - `bench_one_l512` (`B=32, L=512, H=4, P=64, N=64`)：实测核心延迟 **`0.585937 ms – 0.595311 ms`**，最终状态 `hOut[0..3] = 0.05128` 100% 精确匹配（**PASS**）。

### 2.2 昇腾 950 四大时序基准用例全张量精度与性能实测汇总 (`benchmark_mamba2_ssd_chunk_scan_aclnn`)

> **全量校验说明**：`benchmark_mamba2_ssd_chunk_scan_aclnn` 对输出张量的**全部 `B` 个 Batch 与全部 `H` 个 Head 的 100% 元素**（`checkB = c.b; checkH = c.h;`）与 CPU FP32 标量递推 Golden Reference 逐项比对。

| 编号 | 用例名称 | 形状 `(B, L, H, P, N, Q)` | 全张量 `max_abs_error` | 全张量 `max_rel_error` | 950 初代 Vector (`ms`) | 950 初代 MIX (`ms`) | **950 最终版 Direct MIX (`ms`)** | 判定 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case 1** | `S-Mamba TS Standard` | `(32, 512, 4, 64, 64, 64)` | **`1.49012e-07`** | `0.00139694` | `61.48 ms` | `9.70 ms` | **`0.5953 ms`** *(纯流 `0.5859 ms`)* | **PASS** |
| **Case 2** | `Long-Horizon TS` | `(16, 1024, 8, 64, 64, 64)` | **`1.63913e-07`** | `0.00473936` | `126.27 ms` | `23.18 ms` | **`2.7064 ms`** | **PASS** |
| **Case 3** | `Ultra-Long Horizon TS` | `(8, 2048, 8, 64, 64, 64)` | **`1.63913e-07`** | `0.00516170` | `126.10 ms` | `23.19 ms` | **`2.7359 ms`** | **PASS** |
| **Case 4** | `High-Capacity SSM State` | `(16, 512, 8, 64, 128, 64)` | **`2.23517e-07`** | `0.00625051` | `121.99 ms` | `19.78 ms` | **`2.8854 ms`** | **PASS** |

---

## 3. 昇腾 910B 双卡全输出精度与性能汇总

| 测试编号 | 场景名称 | 形状 `(B, L, H, P, N, Q)` | **Device 0 全张量 `max_abs_err`** | **Device 1 全张量 `max_abs_err`** | **单次 / 双卡均值 ACLNN (`ms`)** | **`msprof` AI_CORE 均值 (`ms`)** | 判定 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Case 1** | `S-Mamba TS Standard` | `(32, 512, 4, 64, 64, 64)` | **`1.49012e-07`** | **`1.49012e-07`** | `0.5751 / 0.5971 ms` | **`0.5737 ms`** | **PASS** |
| **Case 2** | `Long-Horizon TS` | `(16, 1024, 8, 64, 64, 64)` | **`1.63913e-07`** | **`1.63913e-07`** | `1.5275 / 1.5404 ms` | **`1.5083 ms`** | **PASS** |
| **Case 3** | `Ultra-Long Horizon TS` | `(8, 2048, 8, 64, 64, 64)` | **`1.63913e-07`** | **`1.63913e-07`** | `1.5055 / 1.5506 ms` | **`1.5146 ms`** | **PASS** |
| **Case 4** | `High-Capacity State` | `(16, 512, 8, 64, 128, 64)` | **`2.23517e-07`** | **`2.23517e-07`** | `1.4263 / 1.4683 ms` | **`1.4202 ms`** | **PASS** |
