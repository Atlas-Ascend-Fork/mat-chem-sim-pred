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
ascendc_library(dimenet_triplet_enumerate_fused_kernel_lib SHARED
    op_kernel/dimenet_triplet_enumerate_fused_kernel.cpp)
add_library(dimenet_triplet_enumerate_fused_host SHARED
    op_host/dimenet_triplet_enumerate_fused_host.cpp)
target_include_directories(dimenet_triplet_enumerate_fused_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include op_host)
target_link_directories(dimenet_triplet_enumerate_fused_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(dimenet_triplet_enumerate_fused_host PRIVATE
    ascendcl dimenet_triplet_enumerate_fused_kernel_lib)
target_link_options(dimenet_triplet_enumerate_fused_host PRIVATE
    "-Wl,-rpath,$ORIGIN/lib")
add_library(dimenet_triplet_enumerate_fused_op_def SHARED
    op_host/dimenet_triplet_enumerate_fused_def.cpp)
target_include_directories(dimenet_triplet_enumerate_fused_op_def PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include)
enable_testing()
add_subdirectory(tests/acl_smoke)
install(TARGETS dimenet_triplet_enumerate_fused_host LIBRARY DESTINATION lib)
install(FILES op_host/dimenet_triplet_enumerate_fused_host.h DESTINATION include)
