<div align="center">

# mat-chem-sim-pred

[![license](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![docs](https://img.shields.io/badge/docs-项目文档-green.svg)](docs/zh/README.md)
[![contributing](https://img.shields.io/badge/贡献指南-teal)](CONTRIBUTING.md)

</div>

---

## 🔥 最新动态

- [2026/10] 仓库定位扩展为**面向工业制造的研发与生产**，不再限定流程工业；新增研发域 / 生产域能力映射、行业覆盖矩阵与质量 / 设备 / 运行三类规划方向（见 [roadmap.md](roadmap.md)）。
- [2026/10] 仓库骨架对齐 CANN 算子仓：新增顶层 CMake 入口、`common/` 公共层、`docs/` 文档中心与目录契约测试；算子族 / 例外 / 共享目录收敛到 `cmake/variables.cmake` 单一配置源。
- [2026/10] 补齐 7 个已入库算子的清单登记（Mamba-2 SSD 分块扫描、AttentionalAggregation、AGNN / GENConv / SuperGAT、EdgePooling、SortAggregation），算子总数口径统一为 61。

## 🚀 概述

mat-chem-sim-pred 是 [CANN](https://hiascend.com/software/cann) 面向**工业制造**的领域算子仓：以材料、化学与过程工业为起点，服务工业制造中**研发**（设计、仿真、试制）与**生产**（控制、质量、运行优化）两类负载，把工业模型里"框架不支持或需要多次 kernel launch"的子图，下沉为可独立编译、可参考验证、可 benchmark 的 Ascend C 算子。

核心定位：**工业制造模型的 NPU 融合算子底座**。上游是材料与分子模拟、PDE / 结构仿真、图与几何表征、过程控制、生产时序等工业模型；下游是 CANN 上的单 kernel 替换。

仓库名 `mat-chem-sim-pred` 保留历史（材料 / 化学 / 模拟 / 预测）。能力边界已从流程工业扩展到工业制造的研发与生产；目录按能力族组织，不按行业切分——同一族可以服务材料、流程工业或离散制造。

边界：

- 收入：有明确工业任务、有 aclnn / Host API、有 CPU reference 与精度阈值、有性能口径的算子。
- 不收：行业应用、工艺包、模型训练框架、数据集与一次性实验代码；这些以 reference 或示例的形态留在仓外或 `experimental/`。
- 只解决"计算子图在 NPU 上跑得快且可验证"，不承担产线编排、MES/APC 集成或业务闭环。

算子总数与逐项说明见 [docs/zh/op_list.md](docs/zh/op_list.md)。

## 🏭 研发与生产两类负载

| 阶段 | 典型任务 | 已落地算子族 | 规划方向 |
|---|---|---|---|
| 研发（设计 / 仿真 / 试制） | 分子与材料模拟、PDE / 结构 / 流体仿真、几何与结构表征、小数据寻优 | `md/`、`pde/`、`molecular/`、`experimental/` | ML 势、材料性质与结构生成、CAD / 网格几何、仿真代理 |
| 生产（控制 / 质量 / 运行） | 过程控制与整定、生产时序预测、质量检测、设备健康、排产与能耗优化 | `pid/`、`time_series/` | 质量与检测、设备健康、生产运行优化 |

行业覆盖（需求来源，不是目录划分）：

| 行业 | 研发 | 生产 |
|---|---|---|
| 材料与化工 | 已覆盖：`md/`、`molecular/` | 已覆盖：`pid/`、`time_series/` |
| 流程工业（冶金、建材、造纸等） | 部分覆盖：`pde/` | 已覆盖：`pid/`、`time_series/`；质量与运行规划中 |
| 离散制造（装备、汽车、电子等） | 规划：`pde/`、`molecular/` 的几何能力可复用 | 规划：质量检测、设备健康、排产调度 |

> 规划项的完整清单与状态见 [roadmap.md](roadmap.md)；已入库算子见 [docs/zh/op_list.md](docs/zh/op_list.md)。

## 📦 能力矩阵

| 能力族 | 目录 | 阶段 | 算子数 | 服务场景 |
|---|---|---|---:|---|
| 分子与材料模拟（AI4MD） | `md/` | 研发 | 6 | 经典力场、静电求和、约束、积分、耗散粒子动力学 |
| 连续介质仿真（AI4PDE） | `pde/` | 研发 | 3 | PINN / FNO / DeepONet 推理；结构、流体、传热仿真代理的基础 |
| 图 / 点云 / 分子几何（NonTemporalPrediction） | `molecular/` | 研发 | 26 | GNN 传播与注意力聚合、点云邻域、分子多体几何；可复用于 CAD / 扫描表征 |
| PID 整定全流程（PIDModelFit） | `pid/` | 生产 | 12 | 模型辨识 → 诊断 → 整定 → 仿真选优 → 评估 |
| 生产时序预测（TimeSeriesForecast） | `time_series/` | 生产 | 17 | SSM 扫描、lag 聚合、LSH 分桶、TimesNet/KAN/Anomaly Transformer 融合；可用于工艺与设备信号 |
| **合计** | | | **64** | 其中 63 个纳入构建；DPD 实现未完成，暂缓 CI |

> 阶段说明：研发 = 设计、仿真、试制；生产 = 控制、质量、运行。规划中的生产域方向（质量检测、设备健康、运行优化）尚未形成算子族，见 roadmap。

另有两个 Python 参考实现，不计入算子数：`experimental/DAO`、`experimental/SmallData`（GP 回归 / 贝叶斯优化）。

> 规划中的方向（ML-MD、材料性质预测与结构生成等）见 [roadmap.md](roadmap.md)。

## 🧱 仓库结构

```text
mat-chem-sim-pred/
├── cmake/                  # 构建配置与算子族单一配置源（variables.cmake）
├── common/                 # 跨算子公共层（inc / src / tests）
├── md/                     # 研发域：分子与材料模拟算子族（6）
├── pde/                    # 研发域：连续介质仿真算子族（3 + pde_common）
├── molecular/              # 研发域：图 / 点云 / 分子几何算子族（26）
├── pid/                    # 生产域：PID 整定算子族（12）
├── time_series/            # 生产域：生产时序算子族（17）
├── docs/zh/                # 文档中心：快速入门、算子清单、目录说明
├── examples/               # 跨算子示例（当前仅占位）
├── experimental/           # 未成熟能力与 PyTorch 参考实现（DAO、SmallData）
├── scripts/                # build.sh 的实现：构建、增量、Python / reference 测试
├── tests/ut/               # 全仓目录契约测试（无需 CANN/NPU）
├── template/               # 算子文档与代码模板
├── tutorial/               # 教程 notebook
├── CMakeLists.txt          # 顶层入口：公共层 + 算子聚合
├── build.sh                # 统一构建入口
└── .gitcode/               # issue / PR 模板与 CI 工作流
```

完整目录说明见 [docs/zh/appendix/dir_structure.md](docs/zh/appendix/dir_structure.md)。

## 📌 版本配套

本仓源码随 CANN 社区版本发布。具体 tag 与 CANN 版本的对应关系，在 release 流程建立后登记到本表；
在此之前请以 `.gitcode/workflows/` 使用的编译镜像与 [release-management](https://gitcode.com/cann/release-management) 为准。

## 🛠️ 环境准备

- CANN Toolkit（含 Ascend C 编译链）；使用前 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。
- Python 3.9+（参考实现与测试）。
- CMake ≥ 3.16、支持 C++17 的编译器；GTest 可选（缺失时跳过 C++ UT）。
- 基础依赖一键安装（Ubuntu / Debian、RHEL / CentOS）：

```bash
bash install_deps.sh
```

无 NPU 的机器仍可运行目录契约测试与 Python 参考测试，见 [快速开始](#-快速开始)。

## ⬇️ 源码下载

```bash
git clone https://gitcode.com/cann/mat-chem-sim-pred.git
cd mat-chem-sim-pred
```

## ⚡ 快速开始

```bash
# 1. 看仓库里有哪些算子（无需 CANN）
bash build.sh --list

# 2. 目录契约测试（无需 CANN / NPU）
python -m pytest tests/ut -q

# 3. 编译单个算子（需要 CANN）
bash build.sh md/lennard_jones

# 4. 编译并跑 ctest（需要 CANN；GTest 缺失时自动跳过 UT）
bash build.sh --ops=lennard_jones,pinn --test

# 5. Python 参考实现测试（无需 NPU）
bash build.sh --python experimental/SmallData

# 6. 按文件清单增量跑参考测试
bash build.sh --reference-test pid/,time_series/,molecular/ -f pr_filelist.txt -j8

# 7. 顶层聚合编译（需要 CANN；默认全部算子，也可 -DMATCHEM_OP=<path> 只编一个）
cmake -S . -B build && cmake --build build -j
```

更多参数：`bash build.sh --help`；增量与 CI 用法见 [docs/zh/QUICKSTART.md](docs/zh/QUICKSTART.md)。

## 📖 学习教程

| 文档 | 说明 |
|---|---|
| [快速入门](docs/zh/QUICKSTART.md) | 环境、构建、测试与增量门禁 |
| [算子清单](docs/zh/op_list.md) | 64 个算子的逐项说明与统计口径 |
| [目录结构](docs/zh/appendix/dir_structure.md) | 每个目录的职责与放置规则 |
| [贡献模板](template/) | 算法说明、API、测试架构与算子骨架模板 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 目录契约、编码规范、PR 要求 |
| [AGENTS.md](AGENTS.md) | 面向 AI Agent 的可执行约束 |

## 🤝 贡献

Fork → 开发测试 → 提交 PR → SIG Review → 合入。提交前请至少运行：

```bash
git diff --check
pre-commit run --all-files
python -m pytest tests/ut -q
```

新增算子不需要修改 `build.sh` 或顶层 `CMakeLists.txt`：把算子目录放进 `cmake/variables.cmake` 登记的族目录即可被发现。
目录命名、必备交付件与参考实现要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📚 参考资源

| 类别 | 仓库 |
|---|---|
| 基础框架 | [opbase](https://gitcode.com/cann/opbase)（算子开发基础库） |
| 算子库 | [ops-math](https://gitcode.com/cann/ops-math)、[ops-nn](https://gitcode.com/cann/ops-nn)、[ops-blas](https://gitcode.com/cann/ops-blas)、[ops-sparse](https://gitcode.com/cann/ops-sparse) |
| 开发与测试 | [cann-samples](https://gitcode.com/cann/cann-samples)、[ops-test-kit](https://gitcode.com/cann/ops-test-kit)、[oam-tools](https://gitcode.com/cann/oam-tools) |
| 迁移与生成 | [pypto](https://gitcode.com/cann/pypto)、[cann-learning-hub](https://gitcode.com/cann/cann-learning-hub) |

## 👥 维护团队

Maintainer：黄剑兴、张玉橙、周吉彬、高菲

Committer：刘非、刘海东、高梓博、马博文、刘达林、赵俊、郑柳琪、张强豪、李姝漫

## 📄 许可证

[Apache License 2.0](LICENSE)。安全声明见 [SECURITY.md](SECURITY.md)，变更记录见 [CHANGELOG.md](CHANGELOG.md)。
