# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
# --- 工具链前置：CANN 探测与工程初始化统一由 cmake/op_prelude.cmake 负责 ---
set(_matchem_root "${CMAKE_CURRENT_SOURCE_DIR}")
while(NOT EXISTS "${_matchem_root}/cmake/op_prelude.cmake")
    get_filename_component(_matchem_parent "${_matchem_root}" DIRECTORY)
    if(_matchem_parent STREQUAL _matchem_root)
        message(FATAL_ERROR "cmake/op_prelude.cmake not found above ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    set(_matchem_root "${_matchem_parent}")
endwhile()
include("${_matchem_root}/cmake/op_prelude.cmake")

ascendc_library(reformer_lsh_bucket_sort_kernel_lib SHARED op_kernel/reformer_lsh_bucket_sort_kernel.cpp)
ascendc_include_directories(reformer_lsh_bucket_sort_kernel_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
add_library(reformer_lsh_bucket_sort_op_def SHARED
    op_host/reformer_lsh_bucket_sort_host.cpp
    op_host/reformer_lsh_bucket_sort_def.cpp)
target_include_directories(reformer_lsh_bucket_sort_op_def PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
install(FILES msopgen/reformer_lsh_bucket_sort_msopgen.json DESTINATION msopgen)
