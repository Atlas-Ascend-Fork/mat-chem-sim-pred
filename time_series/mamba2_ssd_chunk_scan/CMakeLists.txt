# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

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

ascendc_library(mamba2_ssd_chunk_scan_kernel_lib SHARED op_kernel/mamba2_ssd_chunk_scan_kernel.cpp)
ascendc_include_directories(mamba2_ssd_chunk_scan_kernel_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw
    ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw
    ${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/impl
    ${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/interface)

add_library(mamba2_ssd_chunk_scan_op_def SHARED
    op_host/mamba2_ssd_chunk_scan_host.cpp
    op_host/mamba2_ssd_chunk_scan_def.cpp)
target_include_directories(mamba2_ssd_chunk_scan_op_def PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/..)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    add_subdirectory(tests)
endif()
