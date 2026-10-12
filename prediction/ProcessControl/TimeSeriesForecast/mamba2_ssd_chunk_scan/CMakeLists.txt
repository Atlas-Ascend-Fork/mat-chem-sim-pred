# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

cmake_minimum_required(VERSION 3.16)
project(mamba2_ssd_chunk_scan LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    if(DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_TOOLKIT_HOME})
    else()
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
else()
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_CANN_PACKAGE_PATH})
endif()
set(SOC_VERSION "Ascend910B1" CACHE STRING "SOC version")

foreach(candidate
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake")
    if(EXISTS "${candidate}")
        set(ASCENDC_CMAKE_PATH "${candidate}")
        break()
    endif()
endforeach()

enable_testing()
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mamba2_ssd_chunk_scan_reference.py")
    add_test(NAME mamba2_ssd_chunk_scan_reference_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mamba2_ssd_chunk_scan_reference.py)
endif()

if(NOT ASCENDC_CMAKE_PATH)
    message(STATUS "Cannot find ascendc_kernel_cmake; skip build on non-NPU host")
    return()
endif()
include(${ASCENDC_CMAKE_PATH}/ascendc.cmake)

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
