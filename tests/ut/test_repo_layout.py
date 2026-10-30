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

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIABLES = REPO_ROOT / "cmake" / "variables.cmake"

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
BUILD_ARTIFACTS = re.compile(
    r"(^|/)(build|build_out)/|(__pycache__|\.pytest_cache|\.ruff_cache)/"
)


# 目录契约的存量缺口基线：只允许缩小，不允许新增。新增缺口会让下面的测试失败。
# 迁移到新布局（md/ pde/ graph/ geometry/ pid/ time_series/）时按新路径重新登记，
# 旧路径条目随之删除。这里用 Python 常量而不是 .json 文件，避免给"数据文件"套版权头规则。
BASELINE: dict[str, list[str]] = {
    "non_snake_case": [],
    "missing_readme": [],
    "missing_tests": [],
    "tests_without_reference": [
        "prediction/ProcessControl/TimeSeriesForecast/selective_scan_1d",
        "prediction/ProcessControl/NonTemporalPrediction/csr_arma_stack_propagate_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_chebyshev_basis_k3_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_gcn2_residual_propagate_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_hypergraph_attention_two_stage_propagate_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_tagcn_basis_k3_fused",
        "prediction/ProcessControl/NonTemporalPrediction/dimenet_triplet_angle_fused",
        "prediction/ProcessControl/NonTemporalPrediction/dimenet_triplet_enumerate_fused",
    ],
    "missing_msopgen": [
        "prediction/ProcessControl/TimeSeriesForecast/selective_scan_1d",
        "prediction/ProcessControl/NonTemporalPrediction/attentional_aggregation_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_agnn_cosine_attention_aggregate_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_arma_stack_propagate_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_chebyshev_basis_k3_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_film_modulated_mean_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_gat_attention_aggregate_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_gatv2_dynamic_attention_aggregate_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_gcn2_residual_propagate_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_genconv_softmax_aggregate_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_hypergraph_attention_two_stage_propagate_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_lightgcn_k2_weighted_sum_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_point_transformer_attention_aggregate_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_sage_mean_root_relu_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_signed_cross_mean_pack_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_supergat_dot_attention_aggregate_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_tagcn_basis_k3_fused",
        "prediction/ProcessControl/NonTemporalPrediction/csr_transformer_dot_attention_aggregate_fused",
        "prediction/ProcessControl/NonTemporalPrediction/dimenet_triplet_angle_fused",
        "prediction/ProcessControl/NonTemporalPrediction/dimenet_triplet_enumerate_fused",
        "prediction/ProcessControl/NonTemporalPrediction/farthest_point_sampling_fused",
        "prediction/ProcessControl/NonTemporalPrediction/point_ball_query_fused",
        "prediction/ProcessControl/NonTemporalPrediction/ppf_point_pair_features_fused",
        "prediction/ProcessControl/NonTemporalPrediction/sort_aggregation_topk_fused",
    ],
}


def read_cmake_list(name: str) -> list[str]:
    """读取 variables.cmake 中 `set(NAME "a" "b")` 形式的字符串列表。"""
    text = VARIABLES.read_text(encoding="utf-8")
    match = re.search(rf"set\(\s*{name}\s*(.*?)\n\)", text, re.DOTALL)
    assert match, f"{name} not found in {VARIABLES}"
    return re.findall(r'"([^"]*)"', match.group(1))


def operator_dirs(category: str) -> list[Path]:
    """某个族目录下的算子目录；族目录尚未建立时返回空（迁移期正常态）。"""
    category_dir = REPO_ROOT / category
    if not category_dir.is_dir():
        return []
    return [
        c for c in sorted(category_dir.iterdir()) if (c / "CMakeLists.txt").is_file()
    ]


def discover_operators() -> list[str]:
    skipped = set(read_cmake_list("MATCHEM_SKIP_OPERATORS"))
    found = []
    for category in read_cmake_list("MATCHEM_OP_CATEGORIES"):
        for candidate in operator_dirs(category):
            relative = candidate.relative_to(REPO_ROOT).as_posix()
            if relative not in skipped:
                found.append(relative)
    return found


def load_baseline() -> dict[str, list[str]]:
    return BASELINE


def assert_no_new_gaps(rule: str, predicate) -> None:
    """actual 必须是 baseline 的子集：允许偿还欠账，不允许新增。"""
    actual = set()
    for op in discover_operators():
        if predicate(REPO_ROOT / op):
            actual.add(op)
    allowed = set(load_baseline().get(rule, []))
    new_gaps = sorted(actual - allowed)
    assert not new_gaps, (
        f"{rule}: 新增契约缺口 {new_gaps}。"
        f"请补齐交付件，而不是把它们加进 layout_exceptions.json。"
    )


def gap_non_snake_case(path: Path) -> bool:
    return not SNAKE_CASE.match(path.name)


def gap_missing_readme(path: Path) -> bool:
    return not (path / "README.md").is_file()


def gap_missing_tests(path: Path) -> bool:
    return not (path / "tests").is_dir()


def gap_tests_without_reference(path: Path) -> bool:
    if not (path / "tests").is_dir():
        return False
    return not (path / "reference").is_dir()


def test_r1_at_least_one_operator() -> None:
    assert discover_operators(), "no operators discovered — check cmake/variables.cmake"


def test_r1_operator_names_are_unique() -> None:
    names = [Path(op).name for op in discover_operators()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"duplicate operator directory names: {duplicates}"


def test_r2_snake_case() -> None:
    assert_no_new_gaps("non_snake_case", gap_non_snake_case)


def test_r3_readme() -> None:
    assert_no_new_gaps("missing_readme", gap_missing_readme)


def test_r4_tests() -> None:
    assert_no_new_gaps("missing_tests", gap_missing_tests)


def test_r5_reference() -> None:
    assert_no_new_gaps("tests_without_reference", gap_tests_without_reference)


def op_backend(path: Path) -> str:
    manifest = path / "op.yaml"
    if not manifest.is_file():
        return "ascendc"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("backend:"):
            return stripped.split(":", 1)[1].strip().strip("\"'") or "ascendc"
    return "ascendc"


def gap_missing_msopgen(path: Path) -> bool:
    if op_backend(path) != "ascendc":
        return False
    return not list((path / "msopgen").glob("*.json"))


def test_r7_msopgen() -> None:
    assert_no_new_gaps("missing_msopgen", gap_missing_msopgen)


def test_pypto_contract() -> None:
    pypto_ops = [
        op for op in discover_operators() if op_backend(REPO_ROOT / op) == "pypto"
    ]
    for op in pypto_ops:
        path = REPO_ROOT / op
        assert (path / "pyproject.toml").is_file(), f"{op} missing pyproject.toml"
        assert (path / "tests").is_dir(), f"{op} missing tests/"


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


def test_skipped_operators_are_not_discovered() -> None:
    skipped = set(read_cmake_list("MATCHEM_SKIP_OPERATORS"))
    assert not (skipped & set(discover_operators())), (
        "MATCHEM_SKIP_OPERATORS leaked into discovery"
    )


def test_python_operators_exist() -> None:
    missing = []
    for op in read_cmake_list("MATCHEM_PYTHON_OPERATORS"):
        if not (REPO_ROOT / op).is_dir():
            missing.append(op)
    assert not missing, f"MATCHEM_PYTHON_OPERATORS entries not found: {missing}"


def test_docs_op_list_covers_every_operator() -> None:
    op_list = REPO_ROOT / "docs" / "zh" / "op_list.md"
    if not op_list.is_file():
        pytest.skip("docs/zh/op_list.md not present")
    text = op_list.read_text(encoding="utf-8")
    missing = []
    for op in discover_operators():
        if Path(op).name not in text:
            missing.append(op)
    assert not missing, f"operators missing from docs/zh/op_list.md: {missing}"
