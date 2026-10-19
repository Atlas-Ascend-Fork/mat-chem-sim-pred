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
# 单算子编译 / 清理。
# 每个算子仍是独立 CMake 工程（-S <op>），这是当前阶段的契约。

build_single() {
    local op="$1" want_tests=false
    if [ ! -f "$op/CMakeLists.txt" ]; then
        echo "[SKIP] $op — CMakeLists.txt not found"
        return 1
    fi
    echo "[BUILD] $op (SOC=$SOC_VERSION, type=$BUILD_TYPE, jobs=$JOBS)"

    # 仅 --test、算子含 tests/ 且 GTest 可用时才编译单元测试
    if [ "$BUILD_ONLY" != "true" ] && [ "$RUN_TEST" = true ] && [ -d "$op/tests" ] && [ "$HAS_GTEST" = true ]; then
        want_tests=true
    fi

    local args=(-S "$op" -B "$op/build"
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DSOC_VERSION="$SOC_VERSION")
    if [ "$want_tests" = true ]; then
        args+=(-DBUILD_TESTS=ON)
    else
        args+=(-DBUILD_TESTS=OFF)
    fi

    cmake "${args[@]}" \
        || { echo "[FAIL] $op — cmake configure failed"; return 1; }

    cmake --build "$op/build" -j "$JOBS" \
        || { echo "[FAIL] $op — compile failed"; return 1; }

    if [ "$want_tests" = true ]; then
        echo "[TEST] $op"
        # 用 cd 而非 ctest --test-dir：后者需要 CMake >= 3.20，本仓下限是 3.16
        ( cd "$op/build" && ctest --output-on-failure -j "$JOBS" ) \
            || { echo "[FAIL] $op — ctest failed"; return 1; }
    fi
    echo "[ OK ] $op"
    return 0
}

clean_single() {
    local op="$1"
    if [ -d "$op/build" ]; then
        echo "[CLEAN] $op/build"
        rm -rf "$op/build"
    fi
}
