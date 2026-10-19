# tests — 全仓测试

本目录放**全仓级**测试，不放单个算子的用例；算子用例仍在各自的 `<op>/tests/`。

```text
tests/
├── ut/
│   ├── layout_exceptions.json   # 目录契约缺口的基线（只允许缩小）
│   └── test_repo_layout.py      # 目录契约棘轮测试（无需 CANN/NPU）
└── README.md
```

## 运行

```bash
python -m pytest tests/ut -q
```

`test_repo_layout.py` 校验：

- R1 算子发现（族目录 + CMakeLists.txt）与命名唯一性；
- R2 算子目录名小写下划线；
- R3 每个算子有 README；
- R4 每个算子有 tests/；
- R5 有 tests/ 的算子有 reference/；
- R6 仓库不跟踪构建产物。

已存在的缺口写在 `layout_exceptions.json`。规则是**棘轮**：缺口可以修掉，但不能新增——
给新算子豁免会被测试直接拒绝。

## 后续

Step 3 会把 `reference/` 迁移为 `tests/assets/golden.py`（对齐 CANN 算子仓惯例），
届时 R5 与 `layout_exceptions.json` 同步修改，不在本阶段提前改口径。
