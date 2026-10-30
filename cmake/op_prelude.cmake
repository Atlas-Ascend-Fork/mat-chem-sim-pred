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

# CANN 安装路径：显式变量 > ASCEND_CANN_PACKAGE_PATH > ASCEND_TOOLKIT_HOME > 常见安装根
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH OR NOT ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH} AND NOT "$ENV{ASCEND_CANN_PACKAGE_PATH}" STREQUAL "")
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
    elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME} AND NOT "$ENV{ASCEND_TOOLKIT_HOME}" STREQUAL "")
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_TOOLKIT_HOME}")
    endif()
endif()

# 与各算子原先的缓存变量保持一致
set(SOC_VERSION "Ascend910B1" CACHE STRING "SOC version")
set(RUN_MODE "npu" CACHE STRING "Run mode: npu or sim")
set(BUILD_MODE "normal" CACHE STRING "Build mode")
set(DYNAMIC_MODE OFF CACHE BOOL "Enable dynamic mode")

# 定位 ascendc_kernel_cmake：覆盖不同 CANN 目录布局。
#
# CANN CI 编译镜像（如 ci_cann/ubuntu24.04_arm:lv6）把 toolkit 装在
# /home/jenkins/Ascend/cann-<version>/ 下，不是 /usr/local/Ascend，也没有 source 过
# set_env.sh，所以这里按"常见安装根 + 递归查找"兜底，而不是只认一个默认路径。
set(ASCENDC_CMAKE_PATH "")

# 1) 显式路径 / 环境变量给出的根
if(ASCEND_CANN_PACKAGE_PATH)
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
endif()

# 2) 常见安装根兜底（本机 /usr/local/Ascend、CI 镜像 /home/<user>/Ascend、/opt/Ascend）
if(NOT ASCENDC_CMAKE_PATH)
    set(_matchem_cann_roots
        "/usr/local/Ascend/ascend-toolkit/latest"
        "/usr/local/Ascend/ascend-toolkit"
        "/opt/Ascend/ascend-toolkit/latest"
        "/opt/Ascend")
    file(GLOB _matchem_cann_extra
        "/home/*/Ascend/ascend-toolkit"
        "/home/*/Ascend/ascend-toolkit/latest"
        "/home/*/Ascend/cann-*"
        "/usr/local/Ascend/ascend-toolkit/*"
        "/opt/Ascend/*")
    list(APPEND _matchem_cann_roots ${_matchem_cann_extra})
    list(REMOVE_DUPLICATES _matchem_cann_roots)

    foreach(_root IN LISTS _matchem_cann_roots)
        if(NOT IS_DIRECTORY "${_root}")
            continue()
        endif()
        foreach(_candidate
            "${_root}/compiler/tikcpp/ascendc_kernel_cmake"
            "${_root}/tikcpp/ascendc_kernel_cmake"
            "${_root}/x86_64-linux/tikcpp/ascendc_kernel_cmake"
            "${_root}/aarch64-linux/tikcpp/ascendc_kernel_cmake")
            if(EXISTS "${_candidate}/ascendc.cmake")
                set(ASCENDC_CMAKE_PATH "${_candidate}")
                break()
            endif()
        endforeach()
        if(ASCENDC_CMAKE_PATH)
            break()
        endif()
    endforeach()
endif()

# 3) 最后兜底：在候选根下按 2 层递归找
if(NOT ASCENDC_CMAKE_PATH)
    foreach(_root IN LISTS _matchem_cann_roots)
        if(NOT IS_DIRECTORY "${_root}")
            continue()
        endif()
        file(GLOB_RECURSE _ascendc_hits "${_root}/*/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
        if(_ascendc_hits)
            list(GET _ascendc_hits 0 _first_ascendc_cmake)
            get_filename_component(ASCENDC_CMAKE_PATH "${_first_ascendc_cmake}" DIRECTORY)
            break()
        endif()
    endforeach()
endif()

# 从 ascendc.cmake 位置反推 toolkit 根（<cann>/<arch>-linux/tikcpp/ascendc_kernel_cmake），
# 供算子里的 include 路径使用。
if(ASCENDC_CMAKE_PATH AND (NOT DEFINED ASCEND_CANN_PACKAGE_PATH OR NOT ASCEND_CANN_PACKAGE_PATH))
    get_filename_component(_p1 "${ASCENDC_CMAKE_PATH}" DIRECTORY)   # tikcpp
    get_filename_component(_p2 "${_p1}" DIRECTORY)                 # <arch>-linux
    get_filename_component(_p3 "${_p2}" DIRECTORY)                 # <cann>
    set(ASCEND_CANN_PACKAGE_PATH "${_p3}")
endif()

if(NOT ASCENDC_CMAKE_PATH)
    # CI 的 x86 runner 上没有 CANN。此时只做配置、不产生算子目标，并把状态写进 cache
    # 供 build.sh 判断是否跳过；真机或需要"缺 CANN 即失败"时设 MATCHEM_REQUIRE_CANN=1。
    if(DEFINED ENV{MATCHEM_REQUIRE_CANN} AND "$ENV{MATCHEM_REQUIRE_CANN}" STREQUAL "1")
        message(FATAL_ERROR
            "Cannot find ascendc_kernel_cmake under '${ASCEND_CANN_PACKAGE_PATH}'.\n"
            "MATCHEM_REQUIRE_CANN=1 要求必须找到 CANN：请先 source CANN 的 set_env.sh，"
            "或设置 ASCEND_CANN_PACKAGE_PATH / ASCEND_TOOLKIT_HOME。")
    endif()
    message(WARNING
        "[matchem] 未找到 CANN（${ASCEND_CANN_PACKAGE_PATH}）：本次只做配置，不产生算子目标。\n"
        "真机编译请 source <CANN>/set_env.sh 或 export ASCEND_TOOLKIT_HOME=<CANN>；"
        "需要缺 CANN 即失败时设 MATCHEM_REQUIRE_CANN=1。")
    set(MATCHEM_CANN_AVAILABLE FALSE CACHE BOOL "CANN toolchain available" FORCE)
    return()
endif()

set(MATCHEM_CANN_AVAILABLE TRUE CACHE BOOL "CANN toolchain available" FORCE)

message(STATUS "[matchem] CANN: ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "[matchem] ascendc.cmake: ${ASCENDC_CMAKE_PATH}/ascendc.cmake")

include("${ASCENDC_CMAKE_PATH}/ascendc.cmake")
set(ASCENDC_CMAKE_LOADED TRUE)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(ASCEND_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")

set(MATCHEM_TOOLCHAIN_READY TRUE)
