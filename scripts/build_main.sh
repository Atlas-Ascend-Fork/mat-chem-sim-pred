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
# 当前只接管只读子命令（--list / --help），它们不产生编译副作用，便于先验证配置层；
# 编译、测试、打包等子命令仍留在 build.sh 的旧实现里，随后逐个搬过来并把旧代码删掉。
# 配置的唯一来源仍是 cmake/variables.cmake（经 scripts/lib/config.sh 解析）。

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

for _lib in lib/common.sh lib/config.sh lib/ops.sh; do
    # shellcheck source=/dev/null
    source "$REPO_ROOT/scripts/$_lib"
done
unset _lib

usage() {
    cat <<'EOF'
用法: bash build.sh [选项]

只读命令：
  --list [--backend=ascendc|pypto|all]   列出参与构建的算子
  --help                                  显示本帮助

其余命令（--build / --test / --python-tests / --package 等）当前仍由 build.sh
的旧实现处理，迁移完成后统一由本脚本接管。
EOF
}

parse_backend() {
    local backend="all" arg
    for arg in "$@"; do
        case "$arg" in
            --backend=*) backend="${arg#--backend=}" ;;
        esac
    done
    echo "$backend"
}

list_operators() {
    local backend
    backend="$(parse_backend "$@")"
    load_repo_config
    discover_operators
    OPERATORS=("${ALL_OPERATORS[@]:-}")
    [ "$backend" = "all" ] || filter_by_backend "$backend" >/dev/null
    echo "Discovered ${#OPERATORS[@]} operators (backend=${backend}):"
    printf '  %s\n' "${OPERATORS[@]}"
}

main() {
    case "${1:-}" in
        --list|--list-ops) list_operators "$@" ;;
        --help|-h)         usage ;;
        *)                 echo "build_main.sh: 未接管的参数: $*" >&2; exit 2 ;;
    esac
}

main "$@"
