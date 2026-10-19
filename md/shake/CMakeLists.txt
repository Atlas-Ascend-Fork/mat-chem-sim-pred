# shake/CMakeLists.txt
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
option(ENABLE_ASCEND_KERNEL "Build Ascend C kernel" ON)

# ============================================================
# Ascend C SHAKE iteration kernel
# ============================================================
if(ENABLE_ASCEND_KERNEL AND ASCENDC_CMAKE_LOADED)
    ascendc_library(shake_iteration_kernel SHARED
        op_kernel/shake.cpp
    )
    target_include_directories(shake_iteration_kernel_interface PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    )
    get_target_property(_CUR_INCLUDES shake_iteration_kernel_interface INTERFACE_INCLUDE_DIRECTORIES)
    if(_CUR_INCLUDES)
        set_target_properties(shake_iteration_kernel_interface PROPERTIES INCLUDES "${_CUR_INCLUDES}")
    endif()

    # Extract kernel binary from merge object and convert to .inc
    # kernel binary = the entire ELF device_aiv.o file
    set(SHAKE_MERGE_OBJ
        ${CMAKE_CURRENT_BINARY_DIR}/shake_iteration_kernel_merge_obj_dir/device_aiv.o)
    set(SHAKE_KERNEL_INC
        ${CMAKE_CURRENT_BINARY_DIR}/shake_iteration_device_aiv_bin.inc)

    add_custom_command(OUTPUT ${SHAKE_KERNEL_INC}
        COMMAND xxd -i ${SHAKE_MERGE_OBJ} > ${SHAKE_KERNEL_INC}
        DEPENDS ${SHAKE_MERGE_OBJ}
        COMMENT "Converting SHAKE kernel binary to .inc (xxd -i device_aiv.o)"
    )

    add_custom_target(shake_kernel_inc_target DEPENDS ${SHAKE_KERNEL_INC})

    message(STATUS "SHAKE iteration kernel configured with CANN ascendc_library")
endif()

# ============================================================
# SHAKE NPU host library
# Links against the shake_iteration_kernel binary via .inc file
# ============================================================
add_library(shake_host STATIC
    op_host/shake_host.cpp
)

target_include_directories(shake_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_BINARY_DIR}      # For .inc files
    ${CANN_INC_DIR}
)

target_link_libraries(shake_host PUBLIC
    ${ASCENDCL_LIB}
    dl
)

add_dependencies(shake_host shake_kernel_inc_target)

# ============================================================
# Header-only library — CPU reference SHAKE (for backwards compatibility)
# ============================================================
add_library(shake_constraint INTERFACE)
target_include_directories(shake_constraint INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)
