# experimental — 未成熟能力的隔离区

本目录收纳**还不是正式算子**的东西：模型原型、尚未满足目录契约的实现、规划中的验证代码。
默认不参与构建、不计入算子数量、不进入发布包。

## 准入

- 有明确问题、输入输出和验证方式，但还没有稳定的 aclnn / Host API；
- 或者是从论文 / 上游模型迁移过来的参考实现，尚未做 NPU 适配；
- 目录内必须有 `README.md` 写清：目的、当前状态、验证命令、毕业条件。

## 准出（毕业为正式算子）

同时满足以下条件才可迁出到算子族：

1. 独立目录、独立 Host API、独立 kernel 入口；
2. 有 CPU reference 与事先声明的精度阈值；
3. 有 `tests/`，并在 CI 中可发现、可运行；
4. `README.md` / `docs/algorithm.md` / `docs/api_reference.md` 齐备；
5. 在 `cmake/variables.cmake` 的族目录下有一席之地（必要时新增族）。

## 当前内容

| 内容 | 说明 | 运行方式 |
|---|---|---|
| `DAO/` | 材料性质预测的 PyTorch 基础算子库 | `bash build.sh --python experimental/DAO` |
| `SmallData/` | GP 回归 + 贝叶斯优化参考实现 | `bash build.sh --python experimental/SmallData` |
| `MaterialPropertyPrediction.md` | 材料性质预测与结构生成的方向综述 | 阅读 |

这些内容不参与算子计数与 CMake 构建。roadmap 中"规划中"的 ML-MD / 材料性质预测条目在落地前也放这里。
