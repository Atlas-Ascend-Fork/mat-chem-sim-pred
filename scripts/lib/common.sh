#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 通用工具：日志、CANN 环境、GTest 探测。
# 由 build.sh source；不单独执行。

log_info() { echo "[INFO] $*"; }
log_warn() { echo "[WARN] $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

# 依次尝试仓库文档中约定的 CANN 环境脚本；缺失时静默跳过（无 NPU 环境不应让门禁"没有结论"）。
source_cann_env() {
    local env_script="" path
    for path in \
        "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
        "${ASCEND_HOME_PATH:+${ASCEND_HOME_PATH}/set_env.sh}" \
        "${ASCEND_TOOLKIT_HOME:+${ASCEND_TOOLKIT_HOME}/set_env.sh}"; do
        if [ -n "$path" ] && [ -f "$path" ]; then
            env_script="$path"
            break
        fi
    done
    if [ -n "$env_script" ]; then
        # shellcheck disable=SC1090
        source "$env_script" 2>/dev/null || true
    fi
}

# 探测 GTest：CI 镜像未必预装，缺失时跳过 UT 而不是报错。
gtest_available() {
    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists gtest 2>/dev/null; then
        return 0
    fi
    if [ -f /usr/include/gtest/gtest.h ] || [ -f /usr/local/include/gtest/gtest.h ]; then
        if ls /usr/lib/libgtest.* /usr/lib/*/libgtest.* /usr/local/lib/libgtest.* >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}
