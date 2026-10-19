# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 版本与 CANN 依赖声明。
# 当前算子仍是独立工程，本文件暂不参与编译；发布打包流程接入后由顶层 CMake 读取。

set_cann_package(mat_chem_sim_pred VERSION "0.1.0")

# 编译期依赖（版本下限随发布流程建立后补全，当前以 CI 镜像为准）
# set_cann_build_dependencies(runtime ">=8.0")
# set_cann_build_dependencies(opbase ">=8.0")
# set_cann_build_dependencies(metadef ">=8.0")

# 运行期依赖
# set_cann_run_dependencies(runtime ">=8.0")
# set_cann_run_dependencies(opbase ">=8.0")
