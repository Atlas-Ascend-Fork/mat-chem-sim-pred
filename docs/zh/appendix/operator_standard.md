# 算子标准化与治理

存量算子不标准是事实，治理方式不是一次性大改，而是“先定契约，再用棘轮收敛”。

## 1. 正规算子的六条硬标准

一个算子只有同时满足以下六条，才算“可交付”：

| # | 标准 | 检查方式 |
|---|---|---|
| S1 | 有 `msopgen/*.json` 算子原型 | `tests/ut/test_repo_layout.py` R7 |
| S2 | 有 `op_host/`、`op_kernel/`，能独立构建 | `build.sh <op>` / 顶层 CMake |
| S3 | 有 `tests/`，且测试有真实断言 | UT stage / 人工评审 |
| S4 | 有 `reference/` 或等价 golden | R5 + 测试实际引用 |
| S5 | 能打成 custom opp `.run` | `build.sh --package` |
| S6 | 声明支持的 SOC | `MATCHEM_SOCS` + 包内 kernel |

R1–R7 的现状由 `tests/ut/layout_exceptions.json` 记录；例外只能缩小，不能新增。

## 2. 当前基线（2026-10-20）

| 指标 | 数值 |
|---|---|
| 纳入构建的算子 | 63 |
| 有 `msopgen/*.json` | 15（全部在 `time_series/`） |
| 缺 `msopgen` 的存量算子 | 48（已登记为例外） |
| UT cpp | 30，其中仅 1 个使用 GTest |
| 能打出的算子包 | 0（打包链路已就绪，待真机验证） |

所以现在的“正规”只能做到：

- 新算子：必须满足 S1–S6，否则 R7 直接失败；
- 存量算子：按族分批补，例外表逐批缩短；
- CI：先跑编译/UT，打包用手动流水线验证，不阻塞 PR。

## 3. 分批治理

按“离标准最近”的顺序推进：

| 批次 | 范围 | 现状 | 主要工作 | 完成判据 |
|---|---|---|---|---|
| B0 | 基础设施 | 打包脚本、SOC 清单、R7 | 真机跑通一个 `.run` | cfc_scan_fused 可安装、可调用 |
| B1 | `time_series/` 17 个 | 15 个有 msopgen JSON | 补 2 个 JSON、补真 UT、逐 op 打包 | 17/17 有 JSON，至少 3 个有断言 UT |
| B2 | `molecular/` 26 个 | 0 个 JSON，26 个有 model_e2e | 补 JSON、把 e2e 收编成 UT/ST | 26/26 有 JSON，UT 覆盖 Graph 类 |
| B3 | `pid/` 12 个 | 0 个 JSON，12 个有 Python tests | 补 JSON、把 Python 测试转成 C++ UT 或保留为 reference | 12/12 有 JSON |
| B4 | `md/` + `pde/` 8 个 | 0 个 JSON，部分缺 tests/reference | 补 JSON、补 tests/reference | 8/8 满足 S1–S5 |

每批的退出条件一致：

1. 该族所有算子出现在 `msopgen` 覆盖列表；
2. `--package-require-all --shard` 能对该族打包；
3. 至少一个算子有真实断言的 UT，并在 UT stage 跑过；
4. `layout_exceptions.json` 对应条目删除。

## 4. CI 门禁的翻转顺序

  阶段 1（现在）
    Compile: --build-only
    UT:      --test --ut=auto
    Package: 手动流水线
    reference: 不跑

  阶段 2（B1 完成后）
    Compile: --build-only
    UT:      --test --ut=required（镜像装 GTest）
    Package: 手动流水线 + 每 PR 对 time_series 跑 --package
    reference: 仍在 UT 内部消费，不单列 stage

  阶段 3（B2/B3 完成后）
    Compile: --package + 上传 OBS
    UT:      下载包 + 安装 + ctest
    PreSmoke: 对 master/标签跑最小真机用例

原则：先在“被改动的路径”上强制，再逐步扩到全仓；不搞一次性全量整改。

## 5. 新算子的提交前检查

```bash
# 1. 目录与原型
ls <op>/msopgen/*.json <op>/op_host <op>/op_kernel

# 2. 构建与 UT
bash build.sh <op> --test --ut=required

# 3. 打包（有 CANN 的 ARM 环境）
bash build.sh --package --soc=Ascend910B1 <op>

# 4. 契约棘轮
python3 -m pytest tests/ut -q
```

R7 失败说明新算子没有 `msopgen/*.json`；不要把它加进例外表，补原型定义。
