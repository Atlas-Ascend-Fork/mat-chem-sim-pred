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
# 安装构建依赖（不安装 CANN Toolkit，CANN 按官方指南单独安装）。
# 支持 Ubuntu / Debian（apt）与 RHEL / CentOS / openEuler（dnf / yum）。
# ============================================================================
set -euo pipefail

PACKAGES_APT=(cmake make gcc g++ python3 python3-pip git dos2unix)
PACKAGES_RPM=(cmake make gcc gcc-c++ python3 python3-pip git dos2unix)

run_as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        echo "ERROR: 需要 root 或 sudo 安装依赖" >&2
        exit 1
    fi
}

install_with_apt() {
    run_as_root apt-get update
    run_as_root apt-get install -y "${PACKAGES_APT[@]}"
}

install_with_dnf() {
    run_as_root dnf install -y "${PACKAGES_RPM[@]}"
}

install_with_yum() {
    run_as_root yum install -y "${PACKAGES_RPM[@]}"
}

main() {
    if command -v apt-get >/dev/null 2>&1; then
        install_with_apt
    elif command -v dnf >/dev/null 2>&1; then
        install_with_dnf
    elif command -v yum >/dev/null 2>&1; then
        install_with_yum
    else
        echo "ERROR: 未识别的包管理器（需要 apt-get / dnf / yum）" >&2
        echo "请手动安装：cmake >= 3.16, gcc/g++, make, python3, python3-pip, git, dos2unix" >&2
        exit 1
    fi

    echo "----------------------------------------------------------------"
    echo "构建依赖已安装。"
    echo "CANN Toolkit 需按昇腾社区指南单独安装，并 source set_env.sh。"
    echo "Python 依赖: python3 -m pip install -r requirements.txt"
}

main "$@"
