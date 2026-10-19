# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

cmake_minimum_required(VERSION 3.16)
project(test_mamba2_ssd_chunk_scan LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_library(CUST_OPAPI_LIB cust_opapi PATHS "${ASCEND_CANN_PACKAGE_PATH}/op_api/lib" NO_DEFAULT_PATH)
if(NOT TARGET mamba2_ssd_chunk_scan_op_def OR NOT CUST_OPAPI_LIB)
    return()
endif()

add_executable(test_aclnn_mamba2_ssd_chunk_scan
    ../examples/test_aclnn_mamba2_ssd_chunk_scan.cpp)
target_include_directories(test_aclnn_mamba2_ssd_chunk_scan PRIVATE
    .. ${ASCEND_CANN_PACKAGE_PATH}/include)
target_link_libraries(test_aclnn_mamba2_ssd_chunk_scan PRIVATE
    ascendcl nnopbase ${CUST_OPAPI_LIB})
