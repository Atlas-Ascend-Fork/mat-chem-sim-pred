# 目录结构

本页说明每个目录"放什么、不放什么"。新增文件前先对照本页；与 [CONTRIBUTING.md](../../../CONTRIBUTING.md) 冲突时以 CONTRIBUTING 为准。

## 顶层

```text
mat-chem-sim-pred/
├── .gitcode/                       # Issue / PR 模板与 CI 工作流
├── cmake/
│   ├── variables.cmake             # 唯一配置源：算子族、例外算子、Python 算子、共享目录
│   └── ops.cmake                   # 算子发现（CMake 侧，与 build.sh 同口径）
├── common/                         # 跨算子公共层
│   ├── inc/{op_host,op_kernel,op_api}/
│   ├── src/                        # 确有实现时新增，当前为空
│   └── tests/                      # 公共层 UT
├── docs/zh/                        # 文档中心（跨算子文档）
├── examples/                       # 跨算子示例；当前占位，Step 3 落 add_example 骨架
├── experimental/                   # 未成熟能力隔离区，默认不参与构建
├── scripts/
│   ├── lib/{common,config,ops}.sh  # 日志 / 配置解析 / 算子发现
│   ├── build_options.sh            # 参数与帮助
│   ├── build_single.sh             # 单算子编译与清理
│   ├── build_python.sh             # Python 算子测试
│   └── build_reference.sh          # 参考实现测试
├── simulation/
│   ├── AI4MD/                      # 分子动力学算子族（6）
│   ├── AI4PDE/                     # PDE 算子族（3）
│   └── MaterialPropertyPrediction/ # PyTorch 参考实现（非算子）
├── prediction/
│   ├── ProcessControl/
│   │   ├── PIDModelFit/            # PID 整定算子族（12）
│   │   ├── TimeSeriesForecast/     # 时序预测算子族（14）
│   │   └── NonTemporalPrediction/  # 图/点云/分子几何算子族（26）
│   └── SmallData/                  # GP 回归 / BO 参考实现（非算子）
├── template/                       # 文档与代码模板
├── tests/
│   ├── README.md
│   └── ut/                         # 全仓目录契约测试（无需 CANN）
├── tutorial/                       # 教程 notebook
├── CMakeLists.txt                  # 顶层入口：公共层 + 可选算子聚合
├── build.sh                        # 统一构建入口（薄壳）
├── install_deps.sh                 # 构建依赖安装
├── requirements.txt                # Python 测试/参考实现依赖
├── version.cmake                   # 版本与 CANN 依赖声明
├── classify_rule.yaml              # 开源/交付范围分级
├── CHANGELOG.md
├── README.md  CONTRIBUTING.md  AGENTS.md  SECURITY.md  FAQ.md  roadmap.md
├── LICENSE  OAT.xml  .clang-format  .pre-commit-config.yaml  .gitattributes
└── Third_Party_Open_Source_Software_List.yaml / _Notice
```

## 算子目录（当前契约）

```text
<op>/
├── CMakeLists.txt        # 当前是独立 project()，可 cmake -S <op> -B <op>/build
├── README.md             # 功能、输入输出、构建与测试命令、性能摘要
├── docs/
│   ├── algorithm.md      # 数学定义、边界条件、NPU 实现策略
│   ├── api_reference.md  # API、shape、dtype、workspace、错误条件
│   ├── benchmark.md      # 环境、命令、规模、计时口径、数据
│   └── test_report.md    # 较大变更必须提供
├── op_host/              # 参数校验、tiling、workspace、Host launcher
├── op_kernel/            # Ascend C device kernel 与 __aicore__ 入口
├── examples/             # 可独立运行的 ACL/ACLNN smoke 示例
├── reference/            # CPU 参考实现（唯一正确性基准）
└── tests/                # 正确性、C++ UT、benchmark、E2E
```

约定：

- 算子目录名小写下划线；存量 6 个不合规目录记录在 `tests/ut/layout_exceptions.json`，只减不增。
- `tests/` 存在则必须有 `reference/`；CI 的 `--reference-test` 会据此判红。
- 一个算子一个目录；不要在同一 PR 混入探索性原型（放 `experimental/`）。

## 族级 `common/`

某些族已有族内共享实现（如 `PIDModelFit/common/`、`TimeSeriesForecast/common/`），只服务本族。
提升到顶层 `common/` 的条件：出现第二个族的使用方，且语义完全一致。

## 规划中的调整（roadmap Step 2）

当前仍是 `simulation/` + `prediction/` 两层分类。Step 2 会把算子族收到一级目录：

```text
md/  pde/  molecular/  process/
```

并将 `DAO`、`SmallData` 迁入 `experimental/`。本页在迁移完成后同步更新。
