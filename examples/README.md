# examples — 跨算子示例

本目录放**跨算子**的端到端示例（多个算子串成一条模型子图）。单算子的 ACL/ACLNN smoke 示例仍在
各自算子的 `examples/` 下。

## 当前状态

骨架阶段：本目录只有占位说明。`add_example/`（可编译的最小算子骨架，替代纯 Markdown 模板）
随 roadmap Step 3 落地，届时同时替换 `template/operator_example.md`。

## 规划形态

```text
examples/
├── add_example/            # 最小可编译算子：CMakeLists + op_host + op_kernel + examples + tests
├── pid_pipeline/           # PID 整定全链路：fit -> diagnose -> tune -> rollout -> metrics
└── graph_message_passing/  # GNN 子图：CSR 注意力 + 聚合 + 池化
```

约定：示例必须可独立运行，命令写进示例自己的 README，并在 CI 中可发现。
