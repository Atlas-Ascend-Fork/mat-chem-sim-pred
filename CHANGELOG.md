# CHANGELOG

> 记录仓库级重要变更，按时间倒序。算子自身的算法/性能变更写在该算子的 `docs/` 与 PR 中。

## v0.2.0 — 2026-10-19

算子族收平为一级目录，构建入口收敛到顶层 CMake。

### 变更

- CI：workflow 收薄为 4 个 job（x86 聚合测试、ARM 2 分片编译、NPU Python）；算子发现、共享目录、UT/reference 强度全部下沉到 `build.sh`，新增 `--python-tests[=cpu|npu|all]`、`--ut=auto|required|off`、`--reference=auto|required|off`、`--shard=i/N`，`--common-dirs` 默认由 `cmake/variables.cmake` 推导。
- 算子包：新增 `scripts/package_opp.sh`、`build.sh --package` 和手动流水线 `opp_package.yml`，支持把带 `msopgen/*.json` 的算子打成 CANN custom opp `.run`；`MATCHEM_SOCS` 登记 910B/910C/950。
- 治理：新增目录契约 R7（算子必须有 `msopgen/*.json`）和《算子标准与治理》文档；存量 48 个缺口登记为例外，只允许缩小。
- 后端：新增 `op.yaml` 契约与 `--backend=ascendc|pypto` 分派；PyPTO 复用 Compile/UT 拓扑，默认复用 Ascend C ARM 镜像，`PYPTO_ENABLED` 控制是否启用。
- 文档：仓库定位从"材料/化学/流程工业"扩展为**工业制造的研发与生产**；README 新增研发域/生产域负载与行业覆盖矩阵；roadmap 重构为研发域（设计与仿真）与生产域（控制、质量与运行），新增质量检测、设备健康、生产运行优化三类规划方向；AGENTS/CONTRIBUTING/目录说明同步口径。
- 算子族从 `simulation/<方向>/` + `prediction/ProcessControl/<方向>/` 收平为
  `md/`、`pde/`、`molecular/`、`pid/`、`time_series/` 五个一级目录。
- 6 个存量算子目录改为小写下划线：`lennard_jones`、`gaff2`、`pme`、`shake`、
  `velocity_verlet`、`dissipative_particle_dynamics`。
- `DAO`、`SmallData` 迁入 `experimental/`；材料性质预测综述迁为 `experimental/MaterialPropertyPrediction.md`；
  原 ProcessControl 汇总页迁为 `docs/zh/appendix/process-control.md`。
- `build.sh --scope` 与 `--reference-test` 支持逗号分隔多前缀（`pid/,time_series/,molecular/`）。
- 算子 `CMakeLists.txt` 从独立 `project()` 改为父级片段：CANN 探测与工程初始化收敛到
  `cmake/op_prelude.cmake`，顶层 `CMakeLists.txt` 默认聚合全部 60 个算子；
  单独 `cmake -S <op>` 仍可通过 prelude 自举。
- `tests/ut/layout_exceptions.json` 收缩：命名例外清零。

### 验证

- `bash build.sh --list` 60 个算子；`pytest tests/ut` 10 项通过。
- 顶层聚合配置在 CANN stub 下发现并配置 60 个算子，275 个目标；单算子 `cmake -S <op>` 抽样 7 个通过。
- 参考实现抽样：`molecular/csr_gatv2_dynamic_attention_aggregate_fused` 7 项通过；
  `experimental/SmallData` 10 项通过（`DAO` 依赖 pypto，本机无法运行）。
- 未做真实 Ascend C 编译与 NPU 用例（本机无 CANN / NPU）。

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

- ~~算子目录两层分类、图/点云算子未归位~~（v0.2.0 已收平为五个一级族）。
- ~~6 个存量算子目录名不合规~~（v0.2.0 已改名）。
- 36 个有 `tests/` 的算子尚无 `reference/`，契约测试以基线例外记录（待补）。
- ~~算子 `CMakeLists.txt` 仍是独立 `project()`~~（v0.2.0 已改为父级片段，顶层聚合转正）。
