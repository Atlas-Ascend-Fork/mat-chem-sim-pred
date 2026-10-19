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
ascendc_library(csr_point_transformer_attention_aggregate_fused_kernel_lib SHARED
    op_kernel/csr_point_transformer_attention_aggregate_fused_kernel.cpp)
add_library(csr_point_transformer_attention_aggregate_fused_host SHARED
    op_host/csr_point_transformer_attention_aggregate_fused_host.cpp)
target_include_directories(csr_point_transformer_attention_aggregate_fused_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include op_host)
target_link_directories(csr_point_transformer_attention_aggregate_fused_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(csr_point_transformer_attention_aggregate_fused_host PRIVATE
    ascendcl csr_point_transformer_attention_aggregate_fused_kernel_lib)
add_library(csr_point_transformer_attention_aggregate_fused_op_def SHARED
    op_host/csr_point_transformer_attention_aggregate_fused_def.cpp)
target_include_directories(csr_point_transformer_attention_aggregate_fused_op_def PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include)
enable_testing()
add_subdirectory(examples)
option(ENABLE_HOST_SANITIZERS "Enable ASAN and UBSAN for host and smoke code" OFF)
if(ENABLE_HOST_SANITIZERS)
    set(HOST_SANITIZER_LIBRARY_DIR "" CACHE PATH "Directory containing libasan.so and libubsan.so")
    if(NOT HOST_SANITIZER_LIBRARY_DIR)
        file(GLOB host_asan_candidates "/usr/lib/gcc/*/*/libasan.so")
        list(LENGTH host_asan_candidates host_asan_candidate_count)
        if(host_asan_candidate_count EQUAL 0)
            message(FATAL_ERROR "ENABLE_HOST_SANITIZERS needs libasan.so; set HOST_SANITIZER_LIBRARY_DIR")
        endif()
        list(GET host_asan_candidates 0 host_asan_library)
        get_filename_component(HOST_SANITIZER_LIBRARY_DIR "${host_asan_library}" DIRECTORY)
    endif()
    if(NOT EXISTS "${HOST_SANITIZER_LIBRARY_DIR}/libasan.so" OR NOT EXISTS "${HOST_SANITIZER_LIBRARY_DIR}/libubsan.so")
        message(FATAL_ERROR "ENABLE_HOST_SANITIZERS needs libasan.so and libubsan.so; set HOST_SANITIZER_LIBRARY_DIR")
    endif()
    foreach(target csr_point_transformer_attention_aggregate_fused_host
                   csr_point_transformer_attention_aggregate_fused_op_def
                   test_aclnn_csr_point_transformer_attention_aggregate_fused)
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_directories(${target} PRIVATE ${HOST_SANITIZER_LIBRARY_DIR})
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endforeach()
endif()
install(TARGETS csr_point_transformer_attention_aggregate_fused_host LIBRARY DESTINATION lib)
install(FILES op_host/csr_point_transformer_attention_aggregate_fused_host.h DESTINATION include)
