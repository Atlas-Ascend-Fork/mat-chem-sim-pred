# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""目录契约：算子包的输入与 SOC 清单。不需要 CANN。"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_packager_script_exists():
    script = ROOT / "scripts" / "package_opp.sh"
    assert script.is_file(), "missing scripts/package_opp.sh"
    assert script.stat().st_mode & 0o111, "scripts/package_opp.sh must be executable"


def test_soc_list_declares_910b_910c_950():
    text = (ROOT / "cmake" / "variables.cmake").read_text()
    assert "MATCHEM_SOCS" in text
    for soc in ("Ascend910B1", "Ascend910C", "Ascend950PR_9579"):
        assert soc in text, f"MATCHEM_SOCS missing {soc}"


def test_backends_declared():
    text = (ROOT / "cmake" / "variables.cmake").read_text()
    assert "MATCHEM_BACKENDS" in text
    for backend in ("ascendc", "pypto"):
        assert backend in text, f"MATCHEM_BACKENDS missing {backend}"


def test_msopgen_ops_have_sources():
    jsons = sorted(ROOT.rglob("msopgen/*.json"))
    if not jsons:
        pytest.skip("当前没有 msopgen 算子包（旧布局算子尚未提供）")
    for js in jsons:
        op = js.parent.parent
        assert (op / "op_host").is_dir(), f"{op} missing op_host/"
        assert (op / "op_kernel").is_dir(), f"{op} missing op_kernel/"
