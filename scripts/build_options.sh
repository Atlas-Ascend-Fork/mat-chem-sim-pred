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
# 命令行参数与帮助。变量为全局，供 build.sh 主流程消费。

SOC_VERSION="${SOC_VERSION:-}"
BUILD_TYPE="Release"
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
RUN_TEST=false
HAS_GTEST=false
BUILD_ALL=false
LIST_ONLY=false
CLEAN_MODE=false
SKIP_CHECK=false
BUILD_ONLY="${BUILD_ONLY:-false}"
FILELIST=""
OPS_BY_NAME=""
PYTHON_OP=""
PYTHON_TESTS=false
PYTHON_TESTS_DEVICE="cpu"
COMMON_DIRS=""
SCOPE=""
REFERENCE_TEST=false
PACKAGE_MODE=false
PACKAGE_OUT="build_out/opp"
PACKAGE_FRAMEWORK="tf"
PACKAGE_REQUIRE_ALL=false
UT_MODE="auto"           # auto | required | off
REFERENCE_MODE="required" # auto | required | off
SHARD_INDEX=1
SHARD_TOTAL=1
OPERATORS=()

usage() {
    cat <<'EOF'
mat-chem-sim-pred Build Script

Usage:
  bash build.sh <operator>...            Build specified operators (path or name)
  bash build.sh --all                    Build all discovered operators
  bash build.sh --list                   List discovered operators
  bash build.sh --clean [operator]...    Clean build artifacts
  bash build.sh --test [operator]...     Build and run ctest
  bash build.sh -f <filelist>            Incremental build from file list
  bash build.sh --ops=op1,op2,...        Build by operator names
  bash build.sh --python <operator>      Python operator pytest test
  bash build.sh --python-tests           Run all affected Python + reference tests
  bash build.sh --reference-test [prefix] Run pytest reference tests for operators

Options:
  --soc=<SOC>        SOC version (default: cmake/variables.cmake MATCHEM_DEFAULT_SOC)
  --debug            Debug build type (CMAKE_BUILD_TYPE=Debug)
  -O<level>          Optimization level (O0/O1/O2/O3, maps to Release)
  -j<N>              Parallel build jobs (default: nproc)
  -f <filelist>      File list for incremental build (one path per line)
  --ops=<ops>        Operator names, comma-separated
  --scope=<prefixes> Path prefix filter, comma-separated (-f mode only)
  --common-dirs=<dirs>  Shared dirs (space-separated); default: all <family>/common
                     plus the DirectMatchDir roots from cmake/variables.cmake
  --python <op>      Python operator test via pytest
  --python-tests[=cpu|npu|all]  Run affected Python + reference tests (default: cpu)
  --reference-test   Run pytest reference tests for CMake operators with tests/ dirs
  --reference=<mode> required | auto | off (default: required)
                     auto: only enforce reference/ when tests actually import it
  --ut=<mode>        auto | required | off (default: auto)
                     auto: skip UT when GTest is missing; required: fail instead
  --shard=<i>/<N>    Build only shard i of N from the affected operator list
  --package          Build CANN custom opp packages instead of normal build
  --package-out=<dir> Package output directory (default: build_out/opp)
  --framework=<name>  msopgen framework for --package (default: tf)
  --package-require-all  Fail if any selected operator has no msopgen/*.json
  --build-only       Build only, skip ctest
  --skip-check       Skip "md-only" check for -f mode
  -h, --help         Show this help

Examples:
  bash build.sh --list
  bash build.sh md/lennard_jones
  bash build.sh --ops=lennard_jones,pinn --test
  bash build.sh -f pr_filelist.txt --test --ut=required
  bash build.sh -f pr_filelist.txt --test --shard=1/2
  bash build.sh -f pr_filelist.txt --python-tests --reference=auto
  bash build.sh --package --soc=Ascend910B1 time_series/cfc_scan_fused
  bash build.sh --all -j16

Note:
  - Operator families / exclusions / Python operators live in cmake/variables.cmake.
  - Adding an operator needs no change to this script or the CI workflow.
EOF
}

parse_shard() {
    local value="$1" index total
    case "$value" in
        */*) index="${value%%/*}"; total="${value##*/}" ;;
        *) die "--shard expects <i>/<N>, got: $value" ;;
    esac
    case "$index" in ''|*[!0-9]*) die "--shard index must be an integer" ;; esac
    case "$total" in ''|*[!0-9]*) die "--shard total must be an integer" ;; esac
    [ "$total" -ge 1 ] && [ "$total" -le 16 ] || die "--shard total must be 1..16"
    [ "$index" -ge 1 ] && [ "$index" -le "$total" ] || die "--shard index must be 1..$total"
    SHARD_INDEX="$index"
    SHARD_TOTAL="$total"
}

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --all)           BUILD_ALL=true ;;
            --list|--list-ops) LIST_ONLY=true ;;
            --test)          RUN_TEST=true ;;
            --clean)         CLEAN_MODE=true ;;
            --debug)         BUILD_TYPE="Debug" ;;
            --build-only)    BUILD_ONLY=true ;;
            --skip-check)    SKIP_CHECK=true ;;
            --soc=*)         SOC_VERSION="${1#--soc=}" ;;
            --ops=*)         OPS_BY_NAME="${1#--ops=}" ;;
            --scope=*)       SCOPE="${1#--scope=}" ;;
            --common-dirs=*) COMMON_DIRS="${1#--common-dirs=}" ;;
            --python)
                shift
                PYTHON_OP="${1:-}"
                [ -n "$PYTHON_OP" ] || die "--python requires an operator path"
                ;;
            --python-tests)  PYTHON_TESTS=true ;;
            --python-tests=*)
                PYTHON_TESTS=true
                PYTHON_TESTS_DEVICE="${1#--python-tests=}"
                case "$PYTHON_TESTS_DEVICE" in
                    cpu|npu|all) ;;
                    *) die "--python-tests must be cpu|npu|all" ;;
                esac
                ;;
            --reference-test) REFERENCE_TEST=true ;;
            --reference=*)
                REFERENCE_MODE="${1#--reference=}"
                case "$REFERENCE_MODE" in
                    auto|required|off) ;;
                    *) die "--reference must be auto|required|off" ;;
                esac
                ;;
            --ut=*)
                UT_MODE="${1#--ut=}"
                case "$UT_MODE" in
                    auto|required|off) ;;
                    *) die "--ut must be auto|required|off" ;;
                esac
                ;;
            --shard=*)       parse_shard "${1#--shard=}" ;;
            --package)       PACKAGE_MODE=true ;;
            --package-out=*) PACKAGE_OUT="${1#--package-out=}" ;;
            --framework=*)   PACKAGE_FRAMEWORK="${1#--framework=}" ;;
            --package-require-all) PACKAGE_REQUIRE_ALL=true ;;
            -f)
                shift
                FILELIST="${1:-}"
                [ -n "$FILELIST" ] || die "-f requires a file list path"
                ;;
            -O[0-3])         BUILD_TYPE="Release" ;;
            -j*)             JOBS="${1#-j}"; [ -n "$JOBS" ] || JOBS=8 ;;
            -h|--help)       usage; exit 0 ;;
            *)               OPERATORS+=("$1") ;;
        esac
        shift
    done
    [ -n "$SOC_VERSION" ] || SOC_VERSION="$DEFAULT_SOC"
}
