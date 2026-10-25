# ----------------------------------------------------------------------------
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
set(MATCHEM_OP_CATEGORIES
    "md"
    "pde"
    "pid"
    "time_series"
    "molecular"
)

# 暂不纳入构建的算子（实现未完成，补齐后从此处移除）
set(MATCHEM_SKIP_OPERATORS
    "md/dissipative_particle_dynamics"
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
    "molecular"
)

# 默认 SOC（可被 --soc= / -DSOC_VERSION= 覆盖）
set(MATCHEM_DEFAULT_SOC "Ascend910B1")
