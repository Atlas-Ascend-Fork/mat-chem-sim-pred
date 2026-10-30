# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 算子构建宏。由顶层 CMakeLists.txt include 一次，算子目录只做声明。
#
# 形态对齐社区：
#   cann/ops-nn   activation/<op>/CMakeLists.txt → add_modules_sources(...)
#   cann/ops-math conversion/<op>/CMakeLists.txt → add_all_modules_sources(...)
# 共同点：算子 CMakeLists 不 include 仓外路径、不 project()、不探测 CANN，
# 只调用父级提供的宏。
# 这样算子目录自包含，也能过 CANN CodeCheck 的"算子目录不得引用外部路径"约束。
#
# 用法（算子目录 CMakeLists.txt 全文）：
#   add_matchem_operator(NAME <op>)
#
# 默认源码布局（与仓内 58 个算子一致）：
#   op_kernel/<op>_kernel.cpp
#   op_host/<op>_host.cpp  op_host/<op>_def.cpp
# 历史命名不一致的算子显式传 KERNEL / HOST：
#   add_matchem_operator(NAME lennard_jones KERNEL op_kernel/lj_force.cpp HOST op_host/lj_force.cpp)

macro(add_matchem_operator)
    cmake_parse_arguments(ARG "" "NAME" "KERNEL;HOST" ${ARGN})
    if(NOT ARG_NAME)
        message(FATAL_ERROR "add_matchem_operator() 需要 NAME <算子名>")
    endif()

    set(_matchem_op_name "${ARG_NAME}")
    set(_matchem_op_dir "${CMAKE_CURRENT_SOURCE_DIR}")

    if(NOT ARG_KERNEL)
        set(ARG_KERNEL "${_matchem_op_dir}/op_kernel/${_matchem_op_name}_kernel.cpp")
    endif()
    if(NOT ARG_HOST)
        set(ARG_HOST
            "${_matchem_op_dir}/op_host/${_matchem_op_name}_host.cpp"
            "${_matchem_op_dir}/op_host/${_matchem_op_name}_def.cpp")
    endif()

    # 设备侧 kernel 库
    ascendc_library(${_matchem_op_name}_kernel_lib SHARED ${ARG_KERNEL})
    ascendc_include_directories(${_matchem_op_name}_kernel_lib PRIVATE
        ${_matchem_op_dir}/op_kernel
        ${_matchem_op_dir}/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw
        ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw/impl
        ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw/interface
        ${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw
        ${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/impl
        ${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/interface)

    # Host 侧算子定义库
    add_library(${_matchem_op_name}_op_def SHARED ${ARG_HOST})
    target_include_directories(${_matchem_op_name}_op_def PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${_matchem_op_dir}/op_host
        ${_matchem_op_dir}/..)

    # 单元测试 / 示例：目录存在就交给它自己决定是否真正构建
    if(EXISTS "${_matchem_op_dir}/tests")
        add_subdirectory("${_matchem_op_dir}/tests")
    endif()
endmacro()

# ---------------------------------------------------------------------------
# 形态化构建片段
#
# 仓内存量算子有 24 种目标组合。下面三个函数只覆盖其中**归一后完全一致**的三组
# （11 个 graph、7 个时序扫描、5 个 pid），调用后与逐行手写等价；其余形态保持原样，
# 需要时按同一思路再加函数，不要把它们塞进参数几十个的万能宏。
#
# 三者都假定：工具链已由顶层 include 的 cmake/op_prelude.cmake 提供，
# 因此不在算子目录里探测 CANN、不 project()。
# ---------------------------------------------------------------------------

# 形态 1：kernel 库 + host 库 + op_def + examples 子目录 + 安装 host 头与库
#        （graph 族 11 个算子）
function(matchem_add_kernel_host_opdef op)
    ascendc_library(${op}_kernel_lib SHARED
        op_kernel/${op}_kernel.cpp)

    add_library(${op}_host SHARED
        op_host/${op}_host.cpp)
    target_include_directories(${op}_host PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include
        op_host)
    target_link_directories(${op}_host PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64)
    target_link_libraries(${op}_host PRIVATE
        ascendcl
        ${op}_kernel_lib)

    add_library(${op}_op_def SHARED
        op_host/${op}_def.cpp)
    target_include_directories(${op}_op_def PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include)

    enable_testing()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
        add_subdirectory(examples)
    endif()

    install(TARGETS ${op}_host LIBRARY DESTINATION lib)
    install(FILES op_host/${op}_host.h DESTINATION include)
endfunction()

# 形态 2：kernel 库 + op_def（host/def 合在一个库）+ 安装 msopgen 描述
#        （time_series 族 7 个扫描类算子）
function(matchem_add_kernel_opdef op)
    ascendc_library(${op}_kernel_lib SHARED
        op_kernel/${op}_kernel.cpp)
    ascendc_include_directories(${op}_kernel_lib PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host)

    add_library(${op}_op_def SHARED
        op_host/${op}_host.cpp
        op_host/${op}_def.cpp)
    target_include_directories(${op}_op_def PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host)

    install(FILES msopgen/${op}_msopgen.json DESTINATION msopgen)
endfunction()

# 形态 3：kernel 库（带 ../common 依赖）+ host 库 + aclnn 示例/benchmark 可执行文件 + 安装
#        （pid 族 5 个指标类算子）
function(matchem_add_kernel_host_opdef_cli op)
    ascendc_library(${op}_kernel_lib SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/${op}_kernel.cpp)
    ascendc_include_directories(${op}_kernel_lib PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/../common)

    add_library(${op}_host SHARED
        op_host/${op}_host.cpp)
    target_include_directories(${op}_host PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/../common)
    target_link_directories(${op}_host PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64)
    target_link_libraries(${op}_host
        ascendcl
        ${op}_kernel_lib)

    foreach(_exe test_aclnn_${op} benchmark_${op})
        if(_exe MATCHES "^test_aclnn_")
            set(_src examples/${_exe}.cpp)
            set(_lib ${op}_host)
        else()
            set(_src tests/${_exe}_aclnn.cpp)
            set(_lib ${op}_host)
        endif()
        add_executable(${_exe} ${_src})
        target_include_directories(${_exe} PRIVATE
            ${ASCEND_CANN_PACKAGE_PATH}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
        target_link_directories(${_exe} PRIVATE
            ${ASCEND_CANN_PACKAGE_PATH}/lib64
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/lib)
        target_link_libraries(${_exe}
            ascendcl
            ${_lib})
    endforeach()

    install(TARGETS ${op}_host LIBRARY DESTINATION lib)
    install(FILES op_host/${op}_host.h DESTINATION include)
endfunction()
