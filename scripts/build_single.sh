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

# PyPTO / Python 后端：命令由算子自己的 op.yaml 声明，build.sh 只负责调度。
build_single_pypto() {
    local op="$1" backend build_cmd test_cmd
    backend="$(op_backend "$op")"
    [ -f "$op/op.yaml" ] || { echo "[FAIL] $op — op.yaml not found"; return 1; }

    build_cmd="$(op_field "$op" build 2>/dev/null || true)"
    test_cmd="$(op_field "$op" test 2>/dev/null || true)"
    build_cmd="${build_cmd:-${PYTHON_BIN:-python3} -m pip install -e .}"
    test_cmd="${test_cmd:-${PYTHON_BIN:-python3} -m pytest tests -q}"

    echo "[BUILD] $op (backend=$backend)"
    ( cd "$op" && bash -c "$build_cmd" ) \
        || { echo "[FAIL] $op — backend build failed: $build_cmd"; return 1; }

    if [ "$BUILD_ONLY" != "true" ] && [ "$RUN_TEST" = true ] && [ "$UT_MODE" != "off" ]; then
        echo "[TEST] $op (backend=$backend)"
        ( cd "$op" && bash -c "$test_cmd" ) \
            || { echo "[FAIL] $op — backend test failed: $test_cmd"; return 1; }
    fi
    echo "[ OK ] $op"
    return 0
}

build_single() {
    local op="$1" want_tests=false
    if [ "$(op_backend "$op")" != ascendc ]; then
        build_single_pypto "$op"
        return $?
    fi
    if [ ! -f "$op/CMakeLists.txt" ]; then
        echo "[SKIP] $op — CMakeLists.txt not found"
        return 1
    fi
    echo "[BUILD] $op (SOC=$SOC_VERSION, type=$BUILD_TYPE, jobs=$JOBS)"

    # UT 策略：--ut=off 不编 UT；required 缺 GTest 直接失败；auto 缺 GTest 跳过。
    if [ "$BUILD_ONLY" != "true" ] && [ "$RUN_TEST" = true ] && [ "$UT_MODE" != "off" ] && [ -d "$op/tests" ]; then
        if [ "$HAS_GTEST" = true ]; then
            want_tests=true
        elif [ "$UT_MODE" = required ]; then
            echo "[FAIL] $op — GTest is required (--ut=required) but not found"
            return 1
        fi
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
    if [ "$(op_backend "$op")" != ascendc ]; then
        echo "[CLEAN] $op (backend=$(op_backend "$op"))"
        rm -rf "$op/build" "$op/dist" "$op"/*.egg-info
        return 0
    fi
    if [ -d "$op/build" ]; then
        echo "[CLEAN] $op/build"
        rm -rf "$op/build"
    fi
}
