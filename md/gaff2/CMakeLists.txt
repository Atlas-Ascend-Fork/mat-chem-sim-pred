# gaff2/CMakeLists.txt
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
# Ascend C kernel build
# ascendc_library is loaded at root level — use it directly
if(ENABLE_ASCEND_KERNEL AND ASCENDC_CMAKE_LOADED)
    ascendc_library(gaff2_force_kernel SHARED
        op_kernel/gaff2_force.cpp
    )

    target_include_directories(gaff2_force_kernel_interface PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    )

    # CANN's ascendc_library reads INCLUDES from ${target_name}_interface
    get_target_property(_CUR_INCLUDES gaff2_force_kernel_interface INTERFACE_INCLUDE_DIRECTORIES)
    if(_CUR_INCLUDES)
        set_target_properties(gaff2_force_kernel_interface PROPERTIES INCLUDES "${_CUR_INCLUDES}")
    endif()

    message(STATUS "gaff2_force_kernel configured with CANN ascendc_library")
elseif(ENABLE_ASCEND_KERNEL)
    # Fallback: try local ascend.cmake
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ascend.cmake")
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ascend.cmake)
    endif()
endif()

# Host library — provides the host-side API for launching this kernel
add_library(gaff2_force_host STATIC
    op_host/gaff2_host.cpp
    op_host/gaff2_def.cpp
)

target_include_directories(gaff2_force_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CANN_INC_DIR}
)

target_link_libraries(gaff2_force_host PUBLIC
    ${ASCENDCL_LIB}
    dl
)
