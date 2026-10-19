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
COMMON_DIRS=""
SCOPE=""
REFERENCE_TEST=false
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
  bash build.sh --reference-test [prefix] Run pytest reference tests for operators

Options:
  --soc=<SOC>        SOC version (default: cmake/variables.cmake MATCHEM_DEFAULT_SOC)
  --debug            Debug build type (CMAKE_BUILD_TYPE=Debug)
  -O<level>          Optimization level (O0/O1/O2/O3, maps to Release)
  -j<N>              Parallel build jobs (default: nproc)
  -f <filelist>      File list for incremental build (one path per line)
  --ops=<ops>        Operator names, comma-separated
  --scope=<prefixes> Path prefix filter, comma-separated (-f mode only); common-dir
                     changes only rebuild operators under these prefixes
  --common-dirs=<dirs>  Shared dirs (space-separated); changes trigger full rebuild
  --python <op>      Python operator test via pytest
  --reference-test   Run pytest reference tests for CMake operators with tests/ dirs
  --build-only       Build only, skip ctest
  --skip-check       Skip "md/tests-only" check for -f mode
  -h, --help         Show this help

Examples:
  bash build.sh --list
  bash build.sh md/lennard_jones
  bash build.sh lennard_jones pinn
  bash build.sh --ops=lennard_jones,pinn --test
  bash build.sh -f pr_filelist.txt --test
  bash build.sh --all -j16
  bash build.sh --python experimental/SmallData -f pr_filelist.txt
  bash build.sh --reference-test pid/,time_series/,molecular/ -f pr_filelist.txt

Note:
  - Operator families / exclusions / Python operators live in cmake/variables.cmake.
  - Adding an operator needs no change to this script.
EOF
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
            --reference-test) REFERENCE_TEST=true ;;
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
