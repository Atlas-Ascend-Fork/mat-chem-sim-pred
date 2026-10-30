#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 统一入口的新实现（迁移期由 build.sh 分发进来）。
#
# 已接管：--list / --help / --python <op> / --python-tests[=cpu|npu|all]
# 未接管：--build / --test / --reference-test / --package 等仍走 build.sh 的旧实现，
# 随后逐个搬迁，每搬一个就删掉旧代码对应的一段。
# 配置的唯一来源仍是 cmake/variables.cmake（经 scripts/lib/config.sh 解析）。

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
cd "$REPO_ROOT"

for _lib in \
    lib/common.sh lib/config.sh lib/ops.sh \
    build_options.sh build_single.sh build_python.sh build_reference.sh; do
    # shellcheck source=/dev/null
    source "$REPO_ROOT/scripts/$_lib"
done
unset _lib

export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"

usage() {
    cat <<'EOF'
用法: bash build.sh [选项]

已由新入口接管：
  --list [--backend=ascendc|pypto|all]   列出参与构建的算子
  --python <operator>                    Python 算子 pytest
  --python-tests[=cpu|npu|all]           跑受影响的 Python 与 reference 测试
  --help                                 显示本帮助

其余选项（--build/--test/--reference-test/--package 等）仍由 build.sh 的旧实现处理。
EOF
}

list_operators() {
    echo "Discovered ${#ALL_OPERATORS[@]} CMake operators:"
    printf '  %s\n' "${ALL_OPERATORS[@]}"
    echo ""
    echo "Python operators (use --python):"
    printf '  %s\n' "${PYTHON_OPERATORS[@]:-}"
    if [ ${#SKIP_OPERATORS[@]} -gt 0 ]; then
        echo ""
        echo "Excluded from build (WIP):"
        printf '  %s\n' "${SKIP_OPERATORS[@]}"
    fi
}

main() {
    if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
        usage
        return 0
    fi

    load_repo_config
    discover_operators
    parse_args "$@"

    if [ "$LIST_ONLY" = true ]; then
        list_operators
        return 0
    fi

    if [ -n "$PYTHON_OP" ]; then
        run_python_test "$PYTHON_OP" "$FILELIST"
        return $?
    fi

    if [ "$PYTHON_TESTS" = true ]; then
        run_python_tests_all "$FILELIST" "$PYTHON_TESTS_DEVICE"
        return $?
    fi

    if [ "$REFERENCE_TEST" = true ]; then
        [ -n "$COMMON_DIRS" ] || COMMON_DIRS="${DEFAULT_COMMON_DIRS[*]:-}"
        # 位置参数是前缀过滤（如 prediction/ProcessControl/），与旧实现一致
        run_reference_tests "$FILELIST" "${OPERATORS[0]:-}" "$COMMON_DIRS"
        return $?
    fi

    echo "build_main.sh: 该子命令尚未迁移到新入口: $*" >&2
    return 2
}

main "$@"
