#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 解析 cmake/variables.cmake —— 让 CMake 与 bash 共用同一份算子族配置。
# 约定：set(NAME ... ) 到独占一行的 ")" 之间为列表体，元素为一行一个双引号字符串。

# 输出某个 set(...) 块内的双引号字符串，一行一个。
read_cmake_list() {
    local name="$1"
    awk -v name="$name" '
        $0 ~ "^[[:space:]]*set\\(" name "([[:space:]]|$)" { inside = 1; next }
        inside && /^[[:space:]]*\)/ { exit }
        inside { print }
    ' "${REPO_ROOT}/cmake/variables.cmake" | sed -n 's/^[[:space:]]*"\([^"]*\)".*/\1/p'
}

read_cmake_scalar() {
    local name="$1"
    awk -v name="$name" '
        $0 ~ "^[[:space:]]*set\\(" name "[[:space:]]" { print; exit }
    ' "${REPO_ROOT}/cmake/variables.cmake" | sed -n 's/.*"\([^"]*\)".*/\1/p'
}

load_repo_config() {
    local value
    [ -f "${REPO_ROOT}/cmake/variables.cmake" ] || die "missing cmake/variables.cmake"

    OPERATOR_ROOTS=()
    while IFS= read -r value; do
        [ -n "$value" ] && OPERATOR_ROOTS+=("$value")
    done < <(read_cmake_list MATCHEM_OP_CATEGORIES)

    SKIP_OPERATORS=()
    while IFS= read -r value; do
        [ -n "$value" ] && SKIP_OPERATORS+=("$value")
    done < <(read_cmake_list MATCHEM_SKIP_OPERATORS)

    PYTHON_OPERATORS=()
    while IFS= read -r value; do
        [ -n "$value" ] && PYTHON_OPERATORS+=("$value")
    done < <(read_cmake_list MATCHEM_PYTHON_OPERATORS)

    NPU_PYTHON_OPERATORS=()
    while IFS= read -r value; do
        [ -n "$value" ] && NPU_PYTHON_OPERATORS+=("$value")
    done < <(read_cmake_list MATCHEM_NPU_PYTHON_OPERATORS)

    DIRECT_MATCH_DIRS=()
    while IFS= read -r value; do
        [ -n "$value" ] && DIRECT_MATCH_DIRS+=("$value")
    done < <(read_cmake_list MATCHEM_DIRECT_MATCH_DIRS)

    GLOBAL_BUILD_PATHS=()
    while IFS= read -r value; do
        [ -n "$value" ] && GLOBAL_BUILD_PATHS+=("$value")
    done < <(read_cmake_list MATCHEM_GLOBAL_BUILD_PATHS)

    value="$(read_cmake_scalar MATCHEM_DEFAULT_SOC)"
    [ -n "$value" ] && DEFAULT_SOC="$value"

    [ ${#OPERATOR_ROOTS[@]} -gt 0 ] || die "MATCHEM_OP_CATEGORIES is empty in cmake/variables.cmake"

    # 默认共享目录：每个族的 common/ + DirectMatchDir 根。
    # CI 不再手写 --common-dirs；新增族自动纳入。
    DEFAULT_COMMON_DIRS=()
    local root
    for root in "${OPERATOR_ROOTS[@]}"; do
        [ -d "${root}/common" ] && DEFAULT_COMMON_DIRS+=("${root}/common")
    done
    for value in "${DIRECT_MATCH_DIRS[@]}"; do
        DEFAULT_COMMON_DIRS+=("${value}")
    done
}
