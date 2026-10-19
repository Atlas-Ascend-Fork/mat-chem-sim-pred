# velocity_verlet/CMakeLists.txt
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

option(ENABLE_ASCEND_KERNEL "Build Ascend C kernel operators" ON)
# Ascend C kernel build (ascendc_library loaded at root level)
if(ENABLE_ASCEND_KERNEL AND ASCENDC_CMAKE_LOADED)
    # Build 3 kernels

    ascendc_library(vv_integrate_kernel SHARED
        op_kernel/vv_integrate.cpp
    )
    target_include_directories(vv_integrate_kernel_interface PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    )
    get_target_property(_CUR_INCLUDES vv_integrate_kernel_interface INTERFACE_INCLUDE_DIRECTORIES)
    if(_CUR_INCLUDES)
        set_target_properties(vv_integrate_kernel_interface PROPERTIES INCLUDES "${_CUR_INCLUDES}")
    endif()

    ascendc_library(vv_finish_kernel SHARED
        op_kernel/vv_finish.cpp
    )
    target_include_directories(vv_finish_kernel_interface PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    )
    get_target_property(_CUR_INCLUDES vv_finish_kernel_interface INTERFACE_INCLUDE_DIRECTORIES)
    if(_CUR_INCLUDES)
        set_target_properties(vv_finish_kernel_interface PROPERTIES INCLUDES "${_CUR_INCLUDES}")
    endif()

    ascendc_library(thermo_scale_kernel SHARED
        op_kernel/thermo_scale.cpp
    )
    target_include_directories(thermo_scale_kernel_interface PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    )
    get_target_property(_CUR_INCLUDES thermo_scale_kernel_interface INTERFACE_INCLUDE_DIRECTORIES)
    if(_CUR_INCLUDES)
        set_target_properties(thermo_scale_kernel_interface PROPERTIES INCLUDES "${_CUR_INCLUDES}")
    endif()

    message(STATUS "VV integrator kernels configured with CANN ascendc_library")
endif()

# Host library — provides the host-side API for launching these kernels
add_library(vv_host STATIC
    op_host/vv_host.cpp
)

target_include_directories(vv_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_SOURCE_DIR}/operators/ensemble/op_host
    ${CANN_INC_DIR}
)

target_link_libraries(vv_host PUBLIC
    ${ASCENDCL_LIB}
    dl
)
