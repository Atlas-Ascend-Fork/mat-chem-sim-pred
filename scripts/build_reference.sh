#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 参考实现测试：遍历算子的 tests/ 目录运行 pytest。
# 交付契约：有 tests/ 必须提供 reference/reference.py（或 reference/__init__.py）。

reference_required_for_op() {
    local op="$1"
    case "$REFERENCE_MODE" in
        required) return 0 ;;
        auto)     op_tests_use_reference "$op" ;;
        off)      return 1 ;;
    esac
}

# reference 测试的"共享目录是否有变更"判定。
# 只认调用方传入的 --common-dirs：编译侧的 GLOBAL_BUILD_PATHS（顶层 CMakeLists / cmake/）
# 语义不适用于 reference 测试——否则任何触及 cmake/ 的 PR 都会全量跑一遍，
# 把上游缺 reference 的算子全暴露成失败（与旧实现行为不一致）。
reference_common_changed() {
    local filelist="$1" dirs="$2" cdir
    [ -f "$filelist" ] || return 1
    [ -n "$dirs" ] || return 1
    for cdir in $dirs; do
        grep -q "^${cdir%/}/" "$filelist" 2>/dev/null && return 0
    done
    return 1
}

run_reference_tests() {
    local filelist="$1" filter_prefix="$2" common_dirs_str="$3"
    local overall_rc=0 ran_test=false python_bin op
    local -a missing_ref=()

    local common_changed=false
    if reference_common_changed "$filelist" "$common_dirs_str"; then
        common_changed=true
    fi

    if [ -n "${ASCEND_TOOLKIT_HOME:-}" ] && [ -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]; then
        source "${ASCEND_TOOLKIT_HOME}/set_env.sh" 2>/dev/null || true
    fi

    python_bin="${PYTHON_BIN:-python3}"
    "$python_bin" -m pip install pytest numpy 2>/dev/null || true
    "$python_bin" - <<'PY'
import importlib.util
import subprocess
import sys

required = ["numpy", "pytest", "torch"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY

    for op in "${ALL_OPERATORS[@]}"; do
        matches_prefix "$op" "$filter_prefix" || continue
        [ -d "$op/tests" ] || continue
        if [ "$common_changed" = false ] && [ -n "$filelist" ] && [ -f "$filelist" ]; then
            grep -q "^${op}/" "$filelist" 2>/dev/null || continue
        fi
        # 契约校验：required 一律要求；auto 只在测试真的 import reference 时要求。
        if reference_required_for_op "$op"; then
            if [ ! -f "$op/reference/reference.py" ] && [ ! -f "$op/reference/__init__.py" ]; then
                echo "[REF-MISSING] $op — tests/ 存在但缺少 reference/reference.py"
                missing_ref+=("$op")
                overall_rc=1
                continue
            fi
        elif [ "$REFERENCE_MODE" = auto ]; then
            echo "[REF-SKIP] $op — tests 未引用 reference，按 --reference=auto 放行契约检查"
        fi
        ran_test=true
        echo "=== Reference test: $op ==="
        if ! ( cd "$op" && "$python_bin" -X faulthandler -m pytest tests -v --tb=short ); then
            overall_rc=1
        fi
    done

    finish_reference_run "$overall_rc" "$ran_test" "${missing_ref[@]}"
}
