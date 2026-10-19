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
# 算子发现与增量解析。全部基于 cmake/variables.cmake，新增算子无需改脚本。

# 动态发现算子：族目录的直接子目录 + CMakeLists.txt，排除 MATCHEM_SKIP_OPERATORS。
discover_operators() {
    local root dir op skip skip_op
    ALL_OPERATORS=()
    for root in "${OPERATOR_ROOTS[@]}"; do
        [ -d "$root" ] || continue
        for dir in "$root"/*/; do
            [ -f "${dir}CMakeLists.txt" ] || continue
            op="${dir%/}"
            skip=false
            for skip_op in "${SKIP_OPERATORS[@]}"; do
                if [ "$op" = "$skip_op" ]; then
                    skip=true
                    break
                fi
            done
            [ "$skip" = true ] || ALL_OPERATORS+=("$op")
        done
    done
}

# 算子名 → 算子路径映射
resolve_operator() {
    local input="$1" op
    if [ -f "$input/CMakeLists.txt" ]; then
        echo "$input"
        return 0
    fi
    for op in "${ALL_OPERATORS[@]}"; do
        if [ "${op##*/}" = "${input##*/}" ]; then
            echo "$op"
            return 0
        fi
    done
    return 1
}

# 判断算子是否匹配前缀列表（支持逗号分隔，如 "pid,time_series,molecular"）。
matches_prefix() {
    local path="$1" prefixes="$2" prefix
    local IFS=','
    [ -n "$prefixes" ] || return 0
    for prefix in $prefixes; do
        prefix="${prefix%/}"
        [ -n "$prefix" ] || return 0
        case "$path" in
            "$prefix"|"$prefix"/*) return 0 ;;
        esac
    done
    return 1
}

# 判断目录是否为 DirectMatchDir（忽略调用方传入的尾部斜杠）
is_direct_match_dir() {
    local candidate="${1%/}" dir
    for dir in "${DIRECT_MATCH_DIRS[@]}"; do
        if [ "$candidate" = "${dir%/}" ]; then
            return 0
        fi
    done
    return 1
}

# 共享目录是否有变更：DirectMatchDir 只认直接文件，普通目录按前缀。
common_dirs_changed() {
    local filelist="$1" common_dirs_str="$2" cdir prefix
    [ -n "$common_dirs_str" ] && [ -f "$filelist" ] || return 1
    for cdir in $common_dirs_str; do
        prefix="${cdir%/}"
        if is_direct_match_dir "$prefix"; then
            grep -qE "^${prefix}/[^/]+$" "$filelist" 2>/dev/null && return 0
        else
            grep -q "^${prefix}/" "$filelist" 2>/dev/null && return 0
        fi
    done
    return 1
}

# 从文件列表提取受影响算子（去重）；共享目录有变更时输出 ALL 标记。
resolve_operators_from_filelist() {
    local filelist="$1" common_dirs_str="$2" line op seen=" "
    [ -f "$filelist" ] || die "file list not found: $filelist"

    if common_dirs_changed "$filelist" "$common_dirs_str"; then
        echo "ALL"
        return 0
    fi

    # || [ -n "$line" ] 兜底无换行结尾的最后一行
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"          # 兼容 CRLF
        [ -z "$line" ] && continue
        case "$line" in \#*) continue ;; esac
        for op in "${ALL_OPERATORS[@]}"; do
            if [[ "$line" == "$op/"* ]]; then
                case "$seen" in
                    *" $op "*) ;;
                    *) seen="${seen}${op} "; echo "$op" ;;
                esac
                break
            fi
        done
    done < "$filelist"
}

# 纯文档变更判定：filelist 中非空行全部以 .md 结尾。
# tests/ 变更不在快路径内——用例改动必须重编后再跑 ctest。
should_skip_docs_only() {
    local filelist="$1" non_doc_count
    non_doc_count=$(tr -d '\r' < "$filelist" 2>/dev/null | grep -vE '^[[:space:]]*$' | grep -cvE '\.md$' || true)
    non_doc_count=${non_doc_count:-0}
    [ "$non_doc_count" -eq 0 ]
}
