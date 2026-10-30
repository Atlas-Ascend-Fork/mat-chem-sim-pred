#!/bin/bash
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

# 加载 CANN 环境脚本（set_env.sh），供编译/UT 使用。
#
# 候选顺序：显式变量 → 默认安装根 → CI 镜像布局。CI 编译镜像（如
# ci_cann/ubuntu24.04_arm）把 toolkit 装在 /home/<user>/Ascend/ 下且没有 source，
# 与 /usr/local/Ascend 不同，这里一并兜底。
#
# MATCHEM_REQUIRE_CANN=1 时，找不到 CANN 直接失败——否则编译阶段会被静默 SKIP，
# 得到"绿但没编译"的假结果（这正是此前 CI 假绿的成因）。
source_cann_env() {
    local env_script="" path
    for path in \
        "${ASCEND_HOME_PATH:+${ASCEND_HOME_PATH}/set_env.sh}" \
        "${ASCEND_TOOLKIT_HOME:+${ASCEND_TOOLKIT_HOME}/set_env.sh}" \
        "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
        "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" \
        /home/*/Ascend/ascend-toolkit/set_env.sh \
        /home/*/Ascend/ascend-toolkit/bin/set_env.sh \
        /home/*/Ascend/*/set_env.sh; do
        if [ -n "$path" ] && [ -f "$path" ]; then
            env_script="$path"
            break
        fi
    done

    if [ -n "$env_script" ]; then
        # shellcheck disable=SC1090
        source "$env_script" 2>/dev/null || true
        log_info "CANN env: sourced ${env_script}"
    fi

    if [ "${MATCHEM_REQUIRE_CANN:-0}" = "1" ] && ! command -v ccec >/dev/null 2>&1; then
        die "MATCHEM_REQUIRE_CANN=1 但找不到 CANN（ccec 不在 PATH）。请 source <CANN>/set_env.sh，
或设置 ASCEND_CANN_PACKAGE_PATH / ASCEND_TOOLKIT_HOME。"
    fi

    log_info "CANN env: ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>} ccec=$(command -v ccec || echo -)"
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
