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

    ascendc_library(edgepool_sorted_rank_contract_fused_kernel SHARED
        op_kernel/edgepool_sorted_rank_contract_fused_kernel.cpp)
    add_library(edgepool_sorted_rank_contract_fused_host SHARED
        op_host/edgepool_sorted_rank_contract_fused_host.cpp)
    target_include_directories(edgepool_sorted_rank_contract_fused_host PUBLIC
        ${ASCEND_CANN_PACKAGE_PATH}/include ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
    target_link_directories(edgepool_sorted_rank_contract_fused_host PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64)
    target_link_libraries(edgepool_sorted_rank_contract_fused_host
        ascendcl edgepool_sorted_rank_contract_fused_kernel)
    add_library(edgepool_sorted_rank_contract_fused_op_def SHARED
        op_host/edgepool_sorted_rank_contract_fused_def.cpp)
    target_include_directories(edgepool_sorted_rank_contract_fused_op_def PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include ${CMAKE_CURRENT_SOURCE_DIR}/op_host)

    if(BUILD_TESTING)
        enable_testing()
        add_subdirectory(examples)
    endif()

    install(TARGETS edgepool_sorted_rank_contract_fused_host
        edgepool_sorted_rank_contract_fused_op_def LIBRARY DESTINATION lib)
    install(FILES op_host/edgepool_sorted_rank_contract_fused_host.h
        DESTINATION include)
