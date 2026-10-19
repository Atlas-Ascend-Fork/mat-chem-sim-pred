# CHANGELOG

> 记录仓库级重要变更，按时间倒序。算子自身的算法/性能变更写在该算子的 `docs/` 与 PR 中。

## v0.1.0 — 2026-10-19

仓库骨架对齐 CANN 算子仓（ops-math / ops-nn）范式，不改变任何算子的编译产物与对外 API。

### 新增

- 顶层 `CMakeLists.txt`：配置 `common/` 公共层；`-DMATCHEM_BUILD_OPS=ON` 可选聚合算子工程。
- `cmake/variables.cmake`：算子族、暂缓算子、Python 算子、共享目录的唯一配置源。
- `cmake/ops.cmake`：CMake 侧算子发现，与 `build.sh` 同口径。
- `common/`：跨算子公共层的 include 挂载点（INTERFACE，暂不产出目标）。
- `docs/zh/`：文档中心（快速入门、算子清单、目录结构、环境安装）。
- `tests/ut/`：目录契约棘轮测试，无需 CANN/NPU。
- `experimental/`：未成熟能力隔离区说明。
- `scripts/`：`build.sh` 的实现拆分（配置解析、算子发现、单算子编译、Python / reference 测试）。
- `install_deps.sh`、`requirements.txt`、`version.cmake`、`classify_rule.yaml`、第三方依赖清单。
- `.gitcode/ISSUE_TEMPLATE/`：bug / 文档 / 特性 / 提问四类模板。

### 变更

- `build.sh` 从 628 行收敛为薄入口；算子族不再硬编码在脚本里，新增算子无需改脚本。
- 顶层支持 macOS 自带 bash 3.2（移除 `mapfile` 与关联数组依赖）。
- README 改为"概述 / 能力矩阵 / 目录 / 版本配套 / 环境 / 下载 / 快速开始 / 教程 / 贡献"结构；
  逐算子清单移到 `docs/zh/op_list.md`，数量口径统一为 61 个算子目录、60 个纳入构建。
- 共享目录列表移除两个不存在的目录（`ContinuousPrediction`、`DiscretePrediction`）。

### 已知缺口（后续 Step）

- 算子目录仍是 `simulation/` + `prediction/` 两层分类；`NonTemporalPrediction` 下的图/点云算子尚未归位。
- 6 个存量算子目录名不合规（大写 / 连字符），记录在 `tests/ut/layout_exceptions.json`。
- 36 个有 `tests/` 的算子尚无 `reference/`，契约测试以基线例外记录。
- 算子 `CMakeLists.txt` 仍是独立 `project()`；顶层聚合为实验性，Step 3 完成契约迁移后转正。
