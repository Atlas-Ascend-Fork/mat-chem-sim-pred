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

set(KERNEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/selective_scan_1d_kernel.cpp
)

ascendc_library(selective_scan_1d_kernel_lib SHARED ${KERNEL_SOURCES})

ascendc_include_directories(selective_scan_1d_kernel_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
)

add_library(selective_scan_1d_host SHARED
    op_host/selective_scan_1d_host.cpp
)

target_include_directories(selective_scan_1d_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_link_directories(selective_scan_1d_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(selective_scan_1d_host
    ascendcl
    selective_scan_1d_kernel_lib
)

add_executable(test_aclnn_selective_scan_1d
    examples/test_aclnn_selective_scan_1d.cpp
)

target_include_directories(test_aclnn_selective_scan_1d PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_link_directories(test_aclnn_selective_scan_1d PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/lib
)

target_link_libraries(test_aclnn_selective_scan_1d
    ascendcl
    selective_scan_1d_host
)

add_executable(benchmark_selective_scan_1d
    tests/benchmark_selective_scan_1d_aclnn.cpp
)

target_include_directories(benchmark_selective_scan_1d PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_link_directories(benchmark_selective_scan_1d PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/lib
)

target_link_libraries(benchmark_selective_scan_1d
    ascendcl
    selective_scan_1d_host
)

install(TARGETS selective_scan_1d_host LIBRARY DESTINATION lib)
install(FILES op_host/selective_scan_1d_host.h DESTINATION include)
