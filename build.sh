#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# mat-chem-sim-pred 统一编译入口。
# 配置（算子族 / 例外 / Python 算子 / 共享目录）的唯一来源是 cmake/variables.cmake；
# 本脚只做参数解析与编排，实现分散在 scripts/。
# 用法: bash build.sh --help
# ============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
# shellcheck source=scripts/lib/config.sh
source scripts/lib/config.sh
# shellcheck source=scripts/lib/ops.sh
source scripts/lib/ops.sh
# shellcheck source=scripts/build_options.sh
source scripts/build_options.sh
# shellcheck source=scripts/build_single.sh
source scripts/build_single.sh
# shellcheck source=scripts/build_python.sh
source scripts/build_python.sh
# shellcheck source=scripts/build_reference.sh
source scripts/build_reference.sh

export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"

load_repo_config
discover_operators

parse_args "$@"
source_cann_env

# ---------------------------------------------------------------------------
# 列举算子
# ---------------------------------------------------------------------------
if [ "$LIST_ONLY" = true ]; then
    echo "Discovered ${#ALL_OPERATORS[@]} CMake operators:"
    printf '  %s\n' "${ALL_OPERATORS[@]}"
    echo ""
    echo "Python operators (use --python):"
    printf '  %s\n' "${PYTHON_OPERATORS[@]}"
    if [ ${#SKIP_OPERATORS[@]} -gt 0 ]; then
        echo ""
        echo "Excluded from build (WIP):"
        printf '  %s\n' "${SKIP_OPERATORS[@]}"
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Python 算子模式
# ---------------------------------------------------------------------------
if [ -n "$PYTHON_OP" ]; then
    run_python_test "$PYTHON_OP" "$FILELIST"
    exit $?
fi

# ---------------------------------------------------------------------------
# 参考测试模式（位置参数可作前缀过滤，如 prediction/ProcessControl/）
# ---------------------------------------------------------------------------
if [ "$REFERENCE_TEST" = true ]; then
    run_reference_tests "$FILELIST" "${OPERATORS[0]:-}" "$COMMON_DIRS"
    exit $?
fi

# ---------------------------------------------------------------------------
# -f 增量模式
# ---------------------------------------------------------------------------
if [ -n "$FILELIST" ]; then
    [ -f "$FILELIST" ] || die "file list not found: $FILELIST"
    echo "[INFO] Reading file list: $FILELIST"

    # 纯 .md 变更短路：既不编译也不测试
    if [ "$SKIP_CHECK" = false ] && should_skip_docs_only "$FILELIST"; then
        echo "[SKIP] File list contains only .md files, skip build and test"
        mkdir -p build_out 2>/dev/null || true
        touch build_out/skip_build.run 2>/dev/null || true
        exit 0
    fi

    FILELIST_RESULT=()
    while IFS= read -r resolved; do
        [ -n "$resolved" ] && FILELIST_RESULT+=("$resolved")
    done < <(resolve_operators_from_filelist "$FILELIST" "$COMMON_DIRS")

    # ALL（共享目录变更）按 --scope 限定，只重建该批算子而非全仓
    if [ ${#FILELIST_RESULT[@]} -gt 0 ] && [ "${FILELIST_RESULT[0]}" = "ALL" ]; then
        echo "[INFO] Common directory changed, building operators in scope: ${SCOPE:-<all>}"
        for op in "${ALL_OPERATORS[@]}"; do
            if [ -z "$SCOPE" ] || [[ "$op" == "$SCOPE"* ]]; then
                OPERATORS+=("$op")
            fi
        done
    else
        for op in "${FILELIST_RESULT[@]}"; do
            [ -z "$op" ] && continue
            if [ -z "$SCOPE" ] || [[ "$op" == "$SCOPE"* ]]; then
                OPERATORS+=("$op")
            fi
        done
    fi

    if [ ${#OPERATORS[@]} -eq 0 ]; then
        echo "[SKIP] No affected operators found in file list"
        mkdir -p build_out 2>/dev/null || true
        touch build_out/skip_build.run 2>/dev/null || true
        exit 0
    fi

    echo "[INFO] Affected operators: ${#OPERATORS[@]}"
    printf '  %s\n' "${OPERATORS[@]}"
fi

# ---------------------------------------------------------------------------
# --ops= 按算子名
# ---------------------------------------------------------------------------
if [ -n "$OPS_BY_NAME" ]; then
    IFS=',' read -ra name_arr <<< "$OPS_BY_NAME"
    for name in "${name_arr[@]}"; do
        name="${name#"${name%%[![:space:]]*}"}"   # 去前导空格（不依赖 xargs）
        name="${name%"${name##*[![:space:]]}"}"   # 去尾部空格
        if ! resolved="$(resolve_operator "$name")"; then
            log_warn "Operator not found: $name"
            continue
        fi
        OPERATORS+=("$resolved")
    done
fi

# ---------------------------------------------------------------------------
# 确定操作列表
# ---------------------------------------------------------------------------
if [ "$BUILD_ALL" = true ] || { [ "$CLEAN_MODE" = true ] && [ ${#OPERATORS[@]} -eq 0 ]; }; then
    [ ${#ALL_OPERATORS[@]} -gt 0 ] || die "no operators discovered — check cmake/variables.cmake"
    OPERATORS=("${ALL_OPERATORS[@]}")
fi

if [ ${#OPERATORS[@]} -eq 0 ]; then
    usage
    exit 0
fi

# ---------------------------------------------------------------------------
# 清理模式
# ---------------------------------------------------------------------------
if [ "$CLEAN_MODE" = true ]; then
    for op in "${OPERATORS[@]}"; do
        clean_single "$op"
    done
    echo "----------------------------------------------------------------"
    echo "Clean done: ${#OPERATORS[@]} operator(s)."
    exit 0
fi

# ---------------------------------------------------------------------------
# 编译模式
# ---------------------------------------------------------------------------
if gtest_available; then
    HAS_GTEST=true
fi
if [ "$RUN_TEST" = true ] && [ "$HAS_GTEST" != true ]; then
    for op in "${OPERATORS[@]}"; do
        if [ -d "$op/tests" ]; then
            log_warn "GTest not found; unit tests are skipped (-DBUILD_TESTS=OFF)"
            break
        fi
    done
fi

failed=0
succeeded=0
for op in "${OPERATORS[@]}"; do
    if build_single "$op"; then
        succeeded=$((succeeded + 1))
    else
        failed=$((failed + 1))
    fi
done

echo "----------------------------------------------------------------"
echo "Done: $succeeded succeeded, $failed failed."
[ "$failed" -eq 0 ] || exit 1
