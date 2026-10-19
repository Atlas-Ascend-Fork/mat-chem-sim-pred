# pme/CMakeLists.txt
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

# Find CANN
if(NOT DEFINED CANN_INC_DIR)
    set(CANN_INC_DIR "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include")
endif()
if(NOT DEFINED ASCENDCL_LIB)
    set(ASCENDCL_LIB "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64/libascendcl.so")
endif()

# ============================================================
# PME Kernel — Ascend C kernel (build with ascendc.cmake)
# ============================================================
option(ENABLE_ASCEND_KERNEL "Build Ascend C kernel" ON)

if(ENABLE_ASCEND_KERNEL)

    # Define the Ascend C kernel
    ascendc_library(pme_kernel SHARED
        op_kernel/pme_kernel.cpp
    )

    set_target_properties(pme_kernel PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )

endif()

# ============================================================
# PME Host library — links the kernel .so
# ============================================================
add_library(pme_host2 STATIC
    op_host/pme_host2.cpp
)

target_include_directories(pme_host2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CANN_INC_DIR}
)

target_link_libraries(pme_host2 PUBLIC
    ${ASCENDCL_LIB}
)

# ============================================================
# Ewald Host library (original, kept for reference)
# ============================================================
add_library(ewald_host STATIC
    op_host/ewald_host.cpp
)

target_include_directories(ewald_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CANN_INC_DIR}
)

target_link_libraries(ewald_host PUBLIC
    ${ASCENDCL_LIB}
)

install(TARGETS ewald_host pme_host2 LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
