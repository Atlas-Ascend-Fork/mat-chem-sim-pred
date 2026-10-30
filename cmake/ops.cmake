# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 算子发现：与 build.sh 使用同一份 MATCHEM_OP_CATEGORIES / MATCHEM_SKIP_OPERATORS。
# 判定条件与 bash 侧一致：族目录的直接子目录，且含 CMakeLists.txt。

function(matchem_discover_ops out_var)
    set(_ops "")
    foreach(_category IN LISTS MATCHEM_OP_CATEGORIES)
        file(GLOB _entries RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${_category}/*")
        foreach(_entry IN LISTS _entries)
            if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_entry}"
               AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_entry}/CMakeLists.txt")
                list(FIND MATCHEM_SKIP_OPERATORS "${_entry}" _skip_index)
                if(_skip_index EQUAL -1)
                    list(APPEND _ops "${_entry}")
                endif()
            endif()
        endforeach()
    endforeach()

    list(LENGTH _ops _count)
    message(STATUS "mat-chem-sim-pred: discovered ${_count} operator(s)")
    foreach(_op IN LISTS _ops)
        message(STATUS "  - ${_op}")
    endforeach()

    set(${out_var} "${_ops}" PARENT_SCOPE)
endfunction()
