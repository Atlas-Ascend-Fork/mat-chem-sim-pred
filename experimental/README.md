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

## 当前状态

骨架阶段：本目录只有本说明，尚未迁入任何内容。
计划迁入：`simulation/MaterialPropertyPrediction/DAO`、`prediction/SmallData`（PyTorch 参考实现），
以及 roadmap 中"规划中"的 ML-MD / 材料性质预测条目。迁移动作放在 roadmap Step 2。
