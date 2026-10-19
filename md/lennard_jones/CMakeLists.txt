# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

# Build Kernel Library
set(KERNEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/lj_force.cpp
)

ascendc_library(lj_force SHARED
    ${KERNEL_SOURCES}
)

ascendc_include_directories(lj_force PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
)

# Build Host Library
set(HOST_SOURCES
    op_host/lj_force.cpp
)

add_library(lj_force_host SHARED ${HOST_SOURCES})

target_include_directories(lj_force_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

target_link_directories(lj_force_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(lj_force_host
    ascendcl
    lj_force
)

# Install rules
install(TARGETS lj_force_host LIBRARY DESTINATION lib)
install(FILES op_kernel/lj_force.h DESTINATION include)
