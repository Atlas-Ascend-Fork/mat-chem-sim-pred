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

# DirectMatchDir：只认该目录下的直接文件变更；子目录变更由算子自身路径匹配。
set(MATCHEM_DIRECT_MATCH_DIRS
    "molecular"
)

# 默认 SOC（可被 --soc= / -DSOC_VERSION= 覆盖）
set(MATCHEM_DEFAULT_SOC "Ascend910B1")
