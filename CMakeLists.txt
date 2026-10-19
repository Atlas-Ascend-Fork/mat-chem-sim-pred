# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 顶层入口：仓库骨架 + 公共层 + 可选的算子聚合配置。
#
# 当前（骨架阶段）：
#   - 默认只配置公共层 common/，不触碰任何算子工程；
#   - 单算子构建仍走 `bash build.sh <op>`（每个算子的 CMakeLists.txt 仍是独立 project()）。
#
# 可选（实验性）：
#   cmake -S . -B build -DMATCHEM_BUILD_OPS=ON [-DMATCHEM_OP=<相对路径>]
# 该模式依赖 CANN 工具链，并在同一棵构建树里 add_subdirectory 各算子工程。
# 只有在算子 CMakeLists 完成"独立工程 → 父级片段"契约迁移（roadmap Step 3）后才会成为默认入口。

cmake_minimum_required(VERSION 3.16)

project(mat-chem-sim-pred LANGUAGES CXX)

include(cmake/variables.cmake)
include(cmake/ops.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MATCHEM_BUILD_OPS OFF CACHE BOOL
    "Configure operator projects in-tree (experimental; requires CANN toolchain)")
set(MATCHEM_OP "" CACHE STRING
    "Single operator path to configure when MATCHEM_BUILD_OPS=ON (empty = all discovered)")

matchem_discover_ops(MATCHEM_ALL_OPS)

# 公共层：仅头文件与跨算子共享实现的挂载点，不产生设备侧目标。
add_subdirectory(common)

if(MATCHEM_BUILD_OPS)
    if(NOT DEFINED ENV{ASCEND_CANN_PACKAGE_PATH} AND NOT DEFINED ENV{ASCEND_TOOLKIT_HOME})
        message(FATAL_ERROR
            "MATCHEM_BUILD_OPS=ON 需要 CANN 工具链：请先 source set_env.sh，"
            "或设置 ASCEND_CANN_PACKAGE_PATH / ASCEND_TOOLKIT_HOME。")
    endif()

    if(MATCHEM_OP)
        list(FIND MATCHEM_ALL_OPS "${MATCHEM_OP}" _op_index)
        if(_op_index EQUAL -1)
            message(FATAL_ERROR
                "未知算子路径: ${MATCHEM_OP}\n"
                "可选值见 cmake/variables.cmake 的 MATCHEM_OP_CATEGORIES，或运行 bash build.sh --list")
        endif()
        add_subdirectory("${MATCHEM_OP}")
    else()
        foreach(_op IN LISTS MATCHEM_ALL_OPS)
            add_subdirectory("${_op}")
        endforeach()
    endif()
endif()

message(STATUS "")
message(STATUS "mat-chem-sim-pred 配置完成")
message(STATUS "  算子目录: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "  单算子构建: bash build.sh <op>")
message(STATUS "  列举算子  : bash build.sh --list")
message(STATUS "  聚合配置  : cmake -S . -B build -DMATCHEM_BUILD_OPS=ON [-DMATCHEM_OP=<path>]")
