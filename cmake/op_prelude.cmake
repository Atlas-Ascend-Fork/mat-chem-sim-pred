# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 算子工程前置（prelude）。
#
# 两种进入方式：
#   1. 顶层聚合：顶层 CMakeLists.txt 先 include 本文件，再 add_subdirectory(<op>)；
#      算子目录里的 include 会因 MATCHEM_TOOLCHAIN_READY 直接返回。
#   2. 单算子独立构建：cmake -S <op> -B <op>/build；本文件自举 cmake_minimum_required /
#      project / CANN 工具链探测。
#
# 目的：把原先复制在 61 个算子 CMakeLists 里的 CANN 探测样板收敛到一处，
# 新增算子只写 include + 目标定义。

if(DEFINED MATCHEM_TOOLCHAIN_READY AND MATCHEM_TOOLCHAIN_READY)
    return()
endif()

cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_PROJECT_NAME)
    get_filename_component(_matchem_op_name "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
    project(${_matchem_op_name} LANGUAGES C CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# CANN 安装路径：显式变量 > ASCEND_CANN_PACKAGE_PATH > ASCEND_TOOLKIT_HOME > 默认路径
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH OR NOT ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH} AND NOT "$ENV{ASCEND_CANN_PACKAGE_PATH}" STREQUAL "")
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
    elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME} AND NOT "$ENV{ASCEND_TOOLKIT_HOME}" STREQUAL "")
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_TOOLKIT_HOME}")
    else()
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
endif()

# 与各算子原先的缓存变量保持一致
set(SOC_VERSION "Ascend910B1" CACHE STRING "SOC version")
set(RUN_MODE "npu" CACHE STRING "Run mode: npu or sim")
set(BUILD_MODE "normal" CACHE STRING "Build mode")
set(DYNAMIC_MODE OFF CACHE BOOL "Enable dynamic mode")

# 定位 ascendc_kernel_cmake：覆盖不同 CANN 目录布局
set(ASCENDC_CMAKE_PATH "")
foreach(_candidate
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake")
    if(EXISTS "${_candidate}/ascendc.cmake")
        set(ASCENDC_CMAKE_PATH "${_candidate}")
        break()
    endif()
endforeach()

if(NOT ASCENDC_CMAKE_PATH)
    file(GLOB_RECURSE _ascendc_cmake_files
        "${ASCEND_CANN_PACKAGE_PATH}/../*/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
        "${ASCEND_CANN_PACKAGE_PATH}/../*/*-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
    if(_ascendc_cmake_files)
        list(GET _ascendc_cmake_files 0 _first_ascendc_cmake)
        get_filename_component(ASCENDC_CMAKE_PATH "${_first_ascendc_cmake}" DIRECTORY)
    endif()
endif()

if(NOT ASCENDC_CMAKE_PATH)
    message(FATAL_ERROR
        "Cannot find ascendc_kernel_cmake under '${ASCEND_CANN_PACKAGE_PATH}'.\n"
        "请先 source CANN 的 set_env.sh，或设置 ASCEND_CANN_PACKAGE_PATH / ASCEND_TOOLKIT_HOME。")
endif()

message(STATUS "[matchem] CANN: ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "[matchem] ascendc.cmake: ${ASCENDC_CMAKE_PATH}/ascendc.cmake")

include("${ASCENDC_CMAKE_PATH}/ascendc.cmake")
set(ASCENDC_CMAKE_LOADED TRUE)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(ASCEND_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")

set(MATCHEM_TOOLCHAIN_READY TRUE)
