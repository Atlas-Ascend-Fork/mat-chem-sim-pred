# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 公共层当前只提供 include 挂载点（INTERFACE），不产出编译目标。
# 当确有跨三个以上算子复用的实现时，再在 src/ 下新增源文件并改为 STATIC 库。

add_library(matchem_common INTERFACE)

target_include_directories(matchem_common INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/inc/op_host"
    "${CMAKE_CURRENT_SOURCE_DIR}/inc/op_kernel"
    "${CMAKE_CURRENT_SOURCE_DIR}/inc/op_api")

target_compile_features(matchem_common INTERFACE cxx_std_17)
