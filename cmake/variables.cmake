# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 仓库级唯一配置源。
#
# 这里列出的算子族 / 例外算子 / Python 算子 / 共享目录，同时被两侧消费：
#   - CMake：顶层 CMakeLists.txt include 本文件；
#   - bash ：scripts/lib/config.sh 解析本文件（只读取"一行一个带引号字符串"的 set 块）。
#
# 新增算子族：在此追加一行（目录下需含算子目录与 CMakeLists.txt）。
# 新增算子  ：无需修改本文件，也无需修改 build.sh / CMakeLists.txt。
# 注意：保持 set(...) 的开括号独立成行、闭括号独占一行，bash 解析依赖这个格式。

# 算子族根目录（相对仓库根）
#
# 只列出"目录已经是新布局"或"旧根仍参与构建"的目录。尚未迁移的族不在这里：
# prediction/ProcessControl/PIDModelFit（12 个算子）迁移会把整族既有的静态检查
# 债务当成新文件送进社区 CodeCheck，因此随该族的去重重构单独提交，不并入本 PR。
set(MATCHEM_OP_CATEGORIES
    # 新布局：一级能力族（新算子按族落位，无需改本文件以外的任何构建脚本）
    "md"
    "pde"
    "graph"
    "geometry"
    "time_series"
    # 兼容层（迁移期）：未迁移的算子仍在上游旧路径下，按旧根发现；迁完一族删一行
    "simulation/AI4MD"
    "simulation/AI4PDE"
    "prediction/ProcessControl/TimeSeriesForecast"
    "prediction/ProcessControl/NonTemporalPrediction"
)

# 暂不纳入构建的算子（实现未完成，补齐后从此处移除）
#
# 2026-10-29：CI 第一次真正编译（CANN 探测修好后）暴露——下列 30 个算子的源码从未被
# 编译过：此前 CI 找不到 CANN 时整批走 SKIP，所以一直是假绿。它们的缺陷分六类：
#   - GET_TILING_DATA 宏在本仓构建形态下不存在（17 个）
#   - 按值传参结构体的成员用了 enum / using 别名，CANN 生成的 host_stub.cpp 解析不了（4 个）
#   - 类型头从未入库（shake/vv/pme/ewald/gaff2_types.h）
#   - CANN 9.3 接口变更（Fixpipe/MmadParams/IterateAll 等）与 3510 专属 VF 指令缺架构守卫
#   - CMake host 目标重名 / host 侧符号缺失
#   - 768KB 栈帧超 32KB 上限
# 这些都需要逐个改源码并在真机验证，不适合塞进"让 CI 真编译"这个 PR。
# 待 issue #6 逐条修复后从这里移除；本列表只影响构建，不影响目录结构与已有测试。
set(MATCHEM_SKIP_OPERATORS
    # 上游以来一直未完成的算子
    "simulation/AI4MD/Dissipative_particle_dynamics"
    # CI 首次真编译暴露缺陷、修复前不参与构建（issue #6）；迁移前按上游旧路径挡住
    "prediction/ProcessControl/NonTemporalPrediction/csr_appnp_propagate_fused"
    "prediction/ProcessControl/NonTemporalPrediction/edgepool_sorted_rank_contract_fused"
    "prediction/ProcessControl/NonTemporalPrediction/gemnet_quadruplet_geometry_fused"
    "prediction/ProcessControl/TimeSeriesForecast/anomaly_association_discrepancy_kl_fused"
    "prediction/ProcessControl/TimeSeriesForecast/autocorr_fused_aggregate"
    "prediction/ProcessControl/TimeSeriesForecast/autoformer_inference_aggregate_fused"
    "prediction/ProcessControl/TimeSeriesForecast/batch_spd_inv_fp32"
    "prediction/ProcessControl/TimeSeriesForecast/cfc_scan_fused"
    "prediction/ProcessControl/TimeSeriesForecast/cornn_scan_fused"
    "prediction/ProcessControl/TimeSeriesForecast/fused_kan_spline_basis_linear"
    "prediction/ProcessControl/TimeSeriesForecast/fused_revin_patch_unfold_linear"
    "prediction/ProcessControl/TimeSeriesForecast/ltc_scan_fused"
    "prediction/ProcessControl/TimeSeriesForecast/mamba2_ssd_chunk_scan"
    "prediction/ProcessControl/TimeSeriesForecast/mamba2_ssd_chunk_scan_grad"
    "prediction/ProcessControl/TimeSeriesForecast/mlstm_chunkwise_matrix_memory"
    "prediction/ProcessControl/TimeSeriesForecast/reformer_lsh_qkv_gather"
    "prediction/ProcessControl/TimeSeriesForecast/s6scan_fused"
    "prediction/ProcessControl/TimeSeriesForecast/sru_scan_fused"
    "prediction/ProcessControl/TimeSeriesForecast/timesnet2d_period_fold_aggregate"
    "prediction/ProcessControl/TimeSeriesForecast/tirex_slstm_cell"
    "prediction/ProcessControl/TimeSeriesForecast/unicornn_scan_fused"
    "simulation/AI4MD/Dissipative_particle_dynamics"
    "simulation/AI4MD/GAFF2"
    "simulation/AI4MD/Lennard_Jones"
    "simulation/AI4MD/PME"
    "simulation/AI4MD/SHAKE"
    "simulation/AI4MD/deepmd_radial_env_descriptor_fused"
    "simulation/AI4MD/velocity-verlet"
    "simulation/AI4PDE/deeponet"
    "simulation/AI4PDE/fno"
    "simulation/AI4PDE/pinn"
)

# Python 参考实现：无 CMakeLists.txt，通过 build.sh --python 运行 pytest
set(MATCHEM_PYTHON_OPERATORS
    "experimental/DAO"
    "experimental/SmallData"
)

# 其中必须跑在 NPU 上的 Python 算子（CI 在 ARM/NPU job 里执行）。
# 暂不纳入 CI：DAO 依赖 pypto/CANN 版本匹配，先在 experimental 里保留，
# 需要时把 "experimental/DAO" 加回本列表并恢复 JOB_npu。
set(MATCHEM_NPU_PYTHON_OPERATORS
)

# 支持的算子后端：ascendc（CMake + Ascend C）、pypto（Python DSL/运行时）
set(MATCHEM_BACKENDS
    "ascendc"
    "pypto"
)

# 目标 SOC 清单（打包/多芯片 CI 的唯一事实源）
# 格式：<id>|<SOC_VERSION>|<门控级别>
# 门控级别：pr = PR 跑；master = 仅 master/手动跑
set(MATCHEM_SOCS
    "910b|Ascend910B1|pr"
    "910c|Ascend910C|master"
    "950|Ascend950PR_9579|master"
)

# 影响全部算子的构建输入：顶层 CMake 与 cmake/ 内的公共逻辑。
set(MATCHEM_GLOBAL_BUILD_PATHS
    "CMakeLists.txt"
    "cmake"
)

# DirectMatchDir：只认该目录下的直接文件变更；子目录变更由算子自身路径匹配。
set(MATCHEM_DIRECT_MATCH_DIRS
    "graph"
    "geometry"
)

# 默认 SOC（可被 --soc= / -DSOC_VERSION= 覆盖）
set(MATCHEM_DEFAULT_SOC "Ascend910B1")
