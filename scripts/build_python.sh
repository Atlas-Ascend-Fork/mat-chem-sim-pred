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
# Python 算子测试：自动探测 test_ops.py / test_gpr.py / tests/。

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
