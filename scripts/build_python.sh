#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 判断 Python 算子是否必须跑在 NPU 上。
python_op_needs_npu() {
    local op="$1" nop
    for nop in "${NPU_PYTHON_OPERATORS[@]}"; do
        [ "$op" = "$nop" ] && return 0
    done
    return 1
}

# 聚合模式：跑受影响的 Python 算子测试 + 受影响的 reference 测试。
# device=cpu 跳过 NPU 算子；device=npu 只跑 NPU 算子且不做 reference 校验。
run_python_tests_all() {
    local filelist="$1" device="${2:-cpu}" rc=0 op ran=0
    for op in "${PYTHON_OPERATORS[@]}"; do
        case "$device" in
            cpu) python_op_needs_npu "$op" && continue ;;
            npu) python_op_needs_npu "$op" || continue ;;
        esac
        if [ -n "$filelist" ] && [ -f "$filelist" ]; then
            grep -q "^${op}/" "$filelist" 2>/dev/null || continue
        fi
        ran=$((ran + 1))
        run_python_test "$op" "" || rc=1
    done
    if [ "$ran" -eq 0 ] && [ -z "$filelist" ]; then
        echo "[WARN] No Python operators matched device=${device} in cmake/variables.cmake"
    fi
    if [ "$device" != npu ]; then
        run_reference_tests "$filelist" "" "$COMMON_DIRS" || rc=1
    fi
    return $rc
}

run_python_test() {
    local op="$1" filelist="$2" python_bin
    [ -d "$op" ] || die "Python operator not found: $op"

    # 增量检测：filelist 中该算子无变更则跳过
    if [ -n "$filelist" ] && [ -f "$filelist" ]; then
        if ! grep -q "^${op}/" "$filelist" 2>/dev/null; then
            echo "[SKIP] No ${op} files changed; Python tests are skipped."
            return 0
        fi
    fi

    echo "[PYTEST] $op"
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

    # DAO 依赖 pypto（tile 框架）：仅当源码确实引用时按需安装
    if grep -RqsE '^[[:space:]]*(import pypto|from pypto)' "$op" --include='*.py'; then
        "$python_bin" - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("pypto") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypto"])
PY
    fi

    if [ -f "${op}/test_ops.py" ]; then
        "$python_bin" -X faulthandler -m pytest "${op}/test_ops.py" -v --tb=short
    elif [ -f "${op}/test_gpr.py" ]; then
        "$python_bin" -X faulthandler -m pytest "${op}/test_gpr.py" -v --tb=short
    elif [ -d "${op}/tests" ]; then
        ( cd "$op" && "$python_bin" -X faulthandler -m pytest tests -v --tb=short )
    else
        echo "[WARN] No Python test file found in $op"
        return 1
    fi
}
