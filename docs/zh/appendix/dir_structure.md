# 目录结构

本页说明每个目录"放什么、不放什么"。新增文件前先对照本页；与 [CONTRIBUTING.md](../../../CONTRIBUTING.md) 冲突时以 CONTRIBUTING 为准。

## 顶层

```text
mat-chem-sim-pred/
├── .gitcode/                       # Issue / PR 模板与 CI 工作流
├── cmake/
│   ├── variables.cmake             # 唯一配置源：算子族、例外算子、Python 算子、共享目录
│   ├── ops.cmake                   # 算子发现（CMake 侧，与 build.sh 同口径）
│   └── op_prelude.cmake            # 算子工程前置：CANN 探测 + 顶层/独立两种模式
├── common/                         # 跨算子公共层
│   ├── inc/{op_host,op_kernel,op_api}/
│   ├── src/                        # 确有实现时新增，当前为空
│   └── tests/                      # 公共层 UT
├── md/                             # 分子动力学算子族（6）
├── pde/                            # AI for PDE 算子族（3 + pde_common）
├── molecular/                      # 图 / 点云 / 分子几何算子族（26）
├── pid/                            # PID 整定算子族（12 + common）
├── time_series/                    # 时序预测算子族（14 + common）
├── docs/zh/                        # 文档中心（跨算子文档）
├── examples/                       # 跨算子示例（当前占位）
├── experimental/                   # 未成熟能力与 PyTorch 参考实现（DAO、SmallData）
├── scripts/
│   ├── lib/{common,config,ops}.sh  # 日志 / 配置解析 / 算子发现
│   ├── build_options.sh            # 参数与帮助
│   ├── build_single.sh             # 单算子编译与清理
│   ├── build_python.sh             # Python 算子测试
│   └── build_reference.sh          # 参考实现测试
├── template/                       # 文档与代码模板
├── tests/
│   ├── README.md
│   └── ut/                         # 全仓目录契约测试（无需 CANN）
├── tutorial/                       # 教程 notebook
├── CMakeLists.txt                  # 顶层入口：公共层 + 算子聚合
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

## 算子族

一级目录即能力边界，不再按"科学计算 / 预测优化"分桶：

| 族 | 阶段 | 服务对象 | 典型上游 |
|---|---|---|---|
| `md/` | 研发 | 经典分子动力学 | LAMMPS / GROMACS 力场与积分器 |
| `pde/` | 研发 | 连续介质与 PDE 仿真 | PINN / FNO / DeepONet、结构/流体/传热代理 |
| `molecular/` | 研发 | 图、点云、分子几何 | PyG、PointNet++、DimeNet、GemNet；可复用于 CAD / 扫描 |
| `pid/` | 生产 | 过程模型辨识与整定 | FOPDT / IPDT / SOPDT、ZN / IMC / CC |
| `time_series/` | 生产 | 生产时序与设备信号 | Mamba、Autoformer、Reformer、CfC |

族的增删改在 `cmake/variables.cmake` 的 `MATCHEM_OP_CATEGORIES` 登记，构建脚本与 CMake 自动跟随。

## 算子目录

```text
<op>/
├── CMakeLists.txt        # 父级片段；单独 cmake -S <op> 时由 op_prelude 自举
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

- 算子目录名小写下划线；契约测试会拒绝新增的不合规命名。
- `tests/` 存在则必须有 `reference/`；CI 的 `--reference-test` 会据此判红。
- 一个算子一个目录；不要在同一 PR 混入探索性原型（放 `experimental/`）。

## 族级 `common/`

`pid/common/`、`time_series/common/` 只服务本族；`molecular/` 的共享头直接放在族根。
提升到顶层 `common/` 的条件：出现第二个族的使用方，且语义完全一致。

## 构建入口

```bash
bash build.sh <op>                 # 单算子，独立 configure（推荐日常）
cmake -S . -B build                # 顶层聚合，默认编译全部算子（需 CANN）
cmake -S . -B build -DMATCHEM_OP=molecular/<op>   # 只聚合一个算子
```

新增算子或算子族都不需要修改 `build.sh` / 顶层 `CMakeLists.txt`。
