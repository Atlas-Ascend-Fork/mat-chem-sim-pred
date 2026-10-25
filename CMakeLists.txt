# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 顶层入口：公共层 + 算子聚合。
#
#   cmake -S . -B build                          # 默认编译全部 59 个算子（需 CANN）
#   cmake -S . -B build -DMATCHEM_OP=graph/<op>   # 只聚合一个算子
#   cmake -S . -B build -DMATCHEM_BUILD_OPS=OFF  # 只配置 common/，不需要 CANN
#
# 算子目录的 CMakeLists 是父级片段：CANN 探测与工程初始化在 cmake/op_prelude.cmake，
# 由本文件 include 一次后，各算子目录中的 include 直接返回。

cmake_minimum_required(VERSION 3.16)

project(mat-chem-sim-pred LANGUAGES C CXX)

include(cmake/variables.cmake)
include(cmake/ops.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MATCHEM_BUILD_OPS ON CACHE BOOL
    "Configure operator projects in-tree (requires CANN toolchain)")
set(MATCHEM_OP "" CACHE STRING
    "Single operator path to configure when MATCHEM_BUILD_OPS=ON (empty = all discovered)")

matchem_discover_ops(MATCHEM_ALL_OPS)

# 公共层：仅头文件与跨算子共享实现的挂载点，不产生设备侧目标。
add_subdirectory(common)

if(MATCHEM_BUILD_OPS)
    include(cmake/op_prelude.cmake)

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
message(STATUS "  聚合构建  : cmake -S . -B build [-DMATCHEM_OP=<path>]")
