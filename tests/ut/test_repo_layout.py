# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""仓库目录契约的棘轮测试。

不依赖 CANN / NPU：只读文件系统与 cmake/variables.cmake，可在任意机器上跑。

规则：
  R1 每个算子目录必须有 CMakeLists.txt（发现规则本身）
  R2 算子目录名必须是小写下划线
  R3 每个算子必须有 README.md
  R4 每个算子必须有 tests/
  R5 有 tests/ 的算子必须有 reference/（AGENTS.md 的交付契约）
  R6 仓库不得跟踪构建产物
  R7 每个算子必须有 msopgen/*.json（算子包契约）

已存在的缺口记录在 tests/ut/layout_exceptions.json。测试只允许缺口缩小，
不允许新增：新算子若违反 R2–R7 会直接失败。
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIABLES = REPO_ROOT / "cmake" / "variables.cmake"
EXCEPTIONS = Path(__file__).with_name("layout_exceptions.json")

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
BUILD_ARTIFACTS = re.compile(
    r"(^|/)(build|build_out)/|(__pycache__|\.pytest_cache|\.ruff_cache)/"
)


def read_cmake_list(name: str) -> list[str]:
    """读取 variables.cmake 中 `set(NAME "a" "b")` 形式的字符串列表。"""
    text = VARIABLES.read_text(encoding="utf-8")
    match = re.search(rf"set\(\s*{name}\s*(.*?)\n\)", text, re.S)
    assert match, f"{name} not found in {VARIABLES}"
    return re.findall(r'"([^"]*)"', match.group(1))


def discover_operators() -> list[str]:
    categories = read_cmake_list("MATCHEM_OP_CATEGORIES")
    skipped = set(read_cmake_list("MATCHEM_SKIP_OPERATORS"))
    found = []
    for category in categories:
        category_dir = REPO_ROOT / category
        assert category_dir.is_dir(), f"category not found: {category}"
        for candidate in sorted(category_dir.iterdir()):
            if (candidate / "CMakeLists.txt").is_file():
                relative = candidate.relative_to(REPO_ROOT).as_posix()
                if relative not in skipped:
                    found.append(relative)
    return found


@pytest.fixture(scope="module")
def operators() -> list[str]:
    return discover_operators()


@pytest.fixture(scope="module")
def exceptions() -> dict[str, list[str]]:
    return json.loads(EXCEPTIONS.read_text(encoding="utf-8"))


def assert_no_new_gaps(
    operators: list[str], exceptions: dict[str, list[str]], rule: str, predicate
) -> None:
    """actual 必须是 baseline 的子集：允许偿还欠账，不允许新增。"""
    actual = {op for op in operators if predicate(REPO_ROOT / op)}
    baseline = set(exceptions.get(rule, []))
    new_gaps = sorted(actual - baseline)
    assert not new_gaps, (
        f"{rule}: 新增契约缺口 {new_gaps}。"
        f"请补齐交付件，而不是把它们加进 layout_exceptions.json。"
    )


def test_r1_at_least_one_operator(operators: list[str]) -> None:
    assert operators, "no operators discovered — check cmake/variables.cmake"


def test_r1_operator_names_are_unique(operators: list[str]) -> None:
    names = [Path(op).name for op in operators]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"duplicate operator directory names: {duplicates}"


def test_r2_snake_case(operators: list[str], exceptions: dict[str, list[str]]) -> None:
    assert_no_new_gaps(
        operators,
        exceptions,
        "non_snake_case",
        lambda path: not SNAKE_CASE.match(path.name),
    )


def test_r3_readme(operators: list[str], exceptions: dict[str, list[str]]) -> None:
    assert_no_new_gaps(
        operators,
        exceptions,
        "missing_readme",
        lambda path: not (path / "README.md").is_file(),
    )


def test_r4_tests(operators: list[str], exceptions: dict[str, list[str]]) -> None:
    assert_no_new_gaps(
        operators,
        exceptions,
        "missing_tests",
        lambda path: not (path / "tests").is_dir(),
    )


def test_r5_reference(operators: list[str], exceptions: dict[str, list[str]]) -> None:
    assert_no_new_gaps(
        operators,
        exceptions,
        "tests_without_reference",
        lambda path: (path / "tests").is_dir() and not (path / "reference").is_dir(),
    )


def test_r7_msopgen(operators: list[str], exceptions: dict[str, list[str]]) -> None:
    assert_no_new_gaps(
        operators,
        exceptions,
        "missing_msopgen",
        lambda path: not list((path / "msopgen").glob("*.json")),
    )


def test_r6_no_tracked_build_artifacts() -> None:
    try:
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    offenders = [path for path in tracked if BUILD_ARTIFACTS.search(path)]
    assert not offenders, f"build artifacts are tracked: {offenders[:10]}"


def test_skipped_operators_are_not_discovered(operators: list[str]) -> None:
    skipped = set(read_cmake_list("MATCHEM_SKIP_OPERATORS"))
    assert not (skipped & set(operators)), (
        "MATCHEM_SKIP_OPERATORS leaked into discovery"
    )


def test_python_operators_exist() -> None:
    missing = [
        op
        for op in read_cmake_list("MATCHEM_PYTHON_OPERATORS")
        if not (REPO_ROOT / op).is_dir()
    ]
    assert not missing, f"MATCHEM_PYTHON_OPERATORS entries not found: {missing}"


def test_docs_op_list_covers_every_operator(operators: list[str]) -> None:
    op_list = REPO_ROOT / "docs" / "zh" / "op_list.md"
    if not op_list.is_file():
        pytest.skip("docs/zh/op_list.md not present")
    text = op_list.read_text(encoding="utf-8")
    missing = [op for op in operators if Path(op).name not in text]
    assert not missing, f"operators missing from docs/zh/op_list.md: {missing}"
