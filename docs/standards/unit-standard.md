# Unit Standard v1.2

> 本文件定义 mat-chem-sim-pred 的**基本单元**：一个算子目录必须长什么样、可以缺什么、
> 成熟度怎么标、哪些规则不能破。
>
> **本版的边界**（先读这一段，避免按别的版本执行）：
> - **不引入「模型层」目录**。目录仍是 `<算法族>/<算子>/` 两层；模型归属只作为 UNIT.yaml
>   的可选字段记录，不改变目录形状。
> - **UNIT.yaml 登记是自愿的**。仓库不要求 60 个存量单元都补，也不设 CI 门禁；登记与否
>   只影响该单元能否被自动统计、索引和定级。
> - 定级（L0–L4）是**标注**，不是门禁。当前没有任何流水线会因为级别低而拦 PR。
>
> v1.0 于 2026-10-02 发布后随分支删除；本版是针对当前六族骨架重写并降级落地要求的版本。

## 1. 什么是一个「单元」

一个单元 = **一个可独立编译、独立测试的算子目录**，位于
`md/ pde/ graph/ geometry/ pid/ time_series/` 六个算法族之一下，目录名小写下划线。

判定一个目录是不是单元，只有一条机械标准：

```text
<family>/<unit>/CMakeLists.txt  存在  →  它是一个 ascendc 单元
<family>/<unit>/op.yaml         存在且 backend: pypto  →  它是一个 pypto 单元
```

`cmake/variables.cmake` 的 `MATCHEM_OP_CATEGORIES` 定义族清单，`MATCHEM_SKIP_OPERATORS`
定义暂不纳入构建的例外（当前只有 `md/dissipative_particle_dynamics`）。

## 2. 目录契约

```text
<family>/<unit>/
├── UNIT.yaml                 # 可选：单元元数据（见第 3 节）。不登记也能构建
├── README.md                 # 必需：功能 / 场景 / 输入输出 / 构建测试命令 / 性能摘要
├── CMakeLists.txt            # 必需：声明式构建入口（add_matchem_operator），不引用单元目录外路径
├── docs/
│   ├── algorithm.md          # 数学定义、边界条件、NPU 实现策略、精度阈值前置声明
│   ├── api_reference.md      # API、shape、dtype、workspace、错误条件
│   ├── benchmark.md          # 环境、命令、规模、计时口径、数据
│   └── test_report.md        # 实测：max_abs_err / max_rel_err / mismatch_count / nan_count
├── op_host/
│   ├── <op>_def.cpp          # 算子原型注册
│   ├── <op>_tiling.cpp       # 硬约定：参与编译的 tiling 文件必须含 _tiling
│   └── <op>_tiling.h
├── op_kernel/
│   ├── <op>.cpp              # __aicore__ 入口
│   └── <op>.h
├── examples/
│   └── test_aclnn_<op>.cpp   # 可独立运行的 ACLNN smoke
├── reference/
│   └── reference.py          # 唯一正确性基准（见第 6 节硬规则 1）
└── tests/
    ├── test_<op>.py          # 四类用例：小规模确定性 / 随机 / 边界 / 非法参数
    ├── benchmark_<op>.py
    └── ut/op_kernel/test_<op>.cpp   # GTest 单元测试
```

必需项只有三个：`README.md`、`CMakeLists.txt`、`op_host/` + `op_kernel/` 的编译入口。

`CMakeLists.txt` 采用社区形态：**只声明、不引用单元目录外的路径**。工具链探测与目标定义由
仓根 `cmake/op_prelude.cmake` 和 `cmake/operator.cmake` 提供（顶层 include 一次），单元里只写：

```cmake
add_matchem_operator(NAME <op>)
```

单算子构建经仓根入口，不再进入算子目录独立配置：

```bash
bash build.sh <op>                                   # 推荐
cmake -S . -B build/<op> -DMATCHEM_OP=<family>/<op>  # 等价的手工方式
```

为什么不用 `cmake -S <unit>`：那样算子 CMakeLists 必须自己 include 仓外的 prelude，
而 CANN CodeCheck 明确判这种写法 no pass（探针实测，见第 9 节演进记录）。
社区 `cann/ops-nn`（`add_modules_sources`）、`cann/ops-math`（`add_all_modules_sources`）
都是同一形态，本仓与之对齐。

存量算子的目标组合实测有 24 种，因此 `cmake/operator.cmake` 除了通用的
`add_matchem_operator()`，另有三个**形态函数**，只覆盖归一后完全一致的三组：
`matchem_add_kernel_host_opdef()`（graph 11 个）、`matchem_add_kernel_opdef()`（time_series 扫描类 7 个）、
`matchem_add_kernel_host_opdef_cli()`（pid 指标类 5 个）。其余形态保持逐条书写——
宁可有 24 种明确写法，也不做一个参数几十个的万能宏。

`docs/`、`examples/`、`reference/`、`tests/ut/` 属于**按单元形态裁剪**的项，但裁剪必须在
README 里显式声明「本单元没有 X，因为 Y」，不得无声缺席。

`op_host/` 的 `_tiling` 命名是 CANN 编译系统的硬约定，不是风格偏好：参与编译的 tiling
文件不带 `_tiling` 时构建系统不识别，会表现为「编译过了但 tiling 没生效」这类假绿。

## 3. UNIT.yaml（可选登记）

字段的权威定义是下面的 schema 片段。当前仓内还没有独立的 schema 文件，等登记量起来
再拆出 `docs/standards/UNIT.schema.json`（见第 9 节）。

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["name", "family", "owner", "maturity", "soc_verified", "cann",
               "dtypes", "implementation", "precision", "last_verified", "scenarios"],
  "properties": {
    "name":         {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
    "family":       {"type": "string", "enum": ["md", "pde", "graph", "geometry", "pid", "time_series"]},
    "owner":        {"type": "string", "minLength": 1},
    "maturity":     {"type": "string", "enum": ["L0", "L1", "L2", "L3", "L4"]},
    "soc_verified": {"type": "object", "required": ["ascend910b", "ascend910_93", "ascend950"],
                     "properties": {"ascend910b": {"type": "boolean"},
                                    "ascend910_93": {"type": "boolean"},
                                    "ascend950": {"type": "boolean"}}},
    "cann":         {"type": "string", "pattern": "^[0-9]+\\.[0-9]+(\\.[0-9]+)?$"},
    "dtypes":       {"type": "array", "minItems": 1,
                     "items": {"type": "string", "enum": ["float16", "float32", "bfloat16",
                                                          "int32", "int64", "uint8", "bool"]}},
    "implementation": {"type": "array", "minItems": 1,
                       "items": {"type": "string", "enum": ["ascend-c", "pypto"]}},
    "precision":    {"type": "object", "required": ["max_abs_err", "max_rel_err"],
                     "properties": {"max_abs_err": {"type": ["number", "null"]},
                                    "max_rel_err": {"type": ["number", "null"]}}},
    "last_verified": {"type": "string", "format": "date"},
    "scenarios":    {"type": "array", "minItems": 1, "items": {"type": "string"}},

    "benchmark_baseline": {"type": ["object", "null"]},
    "model":        {"type": "string"},
    "paper":        {"type": ["string", "null"]},
    "upstream":     {"type": ["string", "null"]}
  }
}
```

必填 11 项，可选 4 项。三条填写纪律：

1. **`precision` 只填实测值**。没测就填 `null`，并在 PR 说明缺什么；禁止把估计值写成实测值。
2. **`soc_verified` 只填实机验证过的结果**。编译通过不算验证。
3. **`paper` / `model` 可留空，禁止编造**。查不到出处就 `null`。

带注释的完整示例：

```yaml
name: lennard_jones                 # 必须等于目录名
family: md                          # 必须是路径第一段
owner: 刘非 (@Magic_LF)             # 责任到人，不接受空值
maturity: L2                        # 见第 5 节

soc_verified:
  ascend910b: true                  # 910B
  ascend910_93: false               # 910C
  ascend950: false                  # 950

cann: "8.0"                         # 验证所用 CANN 版本
dtypes:
  - float32
implementation:
  - ascend-c

precision:                          # 与 docs/algorithm.md 的前置声明一致
  max_abs_err: 4.21e-06
  max_rel_err: 2.1e-03

benchmark_baseline: null            # 无真实基线时必须 null
model: null                         # 可选：服务哪个模型，仅作文档索引，不改目录
paper: null                         # 可选：arXiv / DOI / 会议名
upstream: null                      # 可选：PyG 类名 / HF 模型 id

last_verified: "2026-02-01"
scenarios:
  - md_molecular_simulation
```

## 4. 登记什么时候做

登记是**自愿**的，但下面三种情况建议登记，因为不登记会让别人多问一轮：

| 场景 | 为什么建议登记 |
|---|---|
| 新增单元 | 新目录本来就要过 review，顺手填一份成本最低 |
| 要参与对外统计 / 旗舰 E2E | `docs/zh/op_list.md`、README 能力矩阵的成熟度标注要有依据 |
| 精度或 SOC 覆盖发生变化 | 这类变化不看代码看不出来，只有 `precision` / `soc_verified` 能表达 |

存量单元不登记不需要任何理由。缺口靠正常 PR 流量收敛，不做一次性补齐运动。

## 5. 成熟度分级（标注用，非门禁）

| 级别 | 门槛 |
|---|---|
| **L0 experimental** | 目录骨架 + algorithm.md 草稿 + owner |
| **L1 alpha** | Host + kernel 编译通过；reference 存在；参数校验完整；小规模确定性输入对齐；910B 单 SOC |
| **L2 beta** | 文档四件套初稿；随机 / 边界 / 非法参数用例齐全；精度达标（阈值在 algorithm.md 前置声明）；benchmark 初测；910B CI 绿灯 |
| **L3 production** | 文档定稿 + test_report.md；全量精度指标；性能不劣于基线；910B + 910C + 950 三代 SOC 验证 |
| **L4 certified** | 至少一个旗舰场景 E2E 集成通过；owner 维护承诺；安全与许可证扫描全过 |

规则：级别只升不降，但可冻结；连续两个季度无 owner 响应的 L3 / L4 降为 archived。

## 6. 三条硬语义规则

1. **reference 是唯一正确性基准**。禁止为了让测试通过去改 reference 反向拟合错误实现。
2. **Host 职责不下沉 kernel**。参数、shape / dtype、workspace、tiling 校验属于 Host；
   kernel 只处理已验证的设备端计算。
3. **测试必须覆盖四类用例**：小规模确定性、随机、边界、非法参数。

这三条与登记无关——它们对**所有**单元始终有效，不管有没有 UNIT.yaml。

## 7. 命名与术语

- 单元目录：小写下划线，描述**计算阶段**而不是模型名（`dimenet_triplet_angle_fused`，
  不是 `dimenet`）。
- `op_host/` 参与编译的 tiling 文件：必须含 `_tiling`。
- 只使用官方名称 **Ascend C** / **PyPTO**，禁止自造缩写与已废弃的内部简称。
- 「单元」= 一个算子目录；「登记」= 在该目录放 UNIT.yaml；「定级」= 写 `maturity`。

## 8. 存量缺口（2026-10 实测，60 个单元）

| 项 | 覆盖 | 缺口 |
|---|---|---|
| UNIT.yaml | 0 / 60 | 本版不强制，属正常状态 |
| README.md | 57 / 60 | `pde/pinn`、`pde/fno`、`pde/deeponet` |
| docs/algorithm.md | 52 / 60 | 8 个 |
| docs/api_reference.md | 49 / 60 | 11 个 |
| docs/benchmark.md | 47 / 60 | 13 个 |
| docs/test_report.md | 39 / 60 | 21 个（L3 门槛） |
| reference/ | 21 / 60 | 39 个，与硬规则 1 冲突最大 |
| tests/test_<op>.py | 56 / 60 | 4 个 |
| tests/ut/op_kernel | 29 / 60 | 31 个 |
| tests/benchmark_* | 30 / 60 | 30 个 |
| `_tiling` 命名 | 18 / 60 | 需逐个确认（部分单元确无 tiling 阶段） |
| README 引论文出处 | 3 / 60 | 模型溯源几乎为零 |

缺口最大、也最该先补的是 **reference/（21/60）**：它是三条硬规则里唯一一条「没有就无法
判定对错」的。

## 9. 后续演进（本版明确不做）

| 候选 | 前置条件 |
|---|---|
| 拆分 `docs/standards/UNIT.schema.json` + 校验脚本 | 登记量 ≥ 10 个，才值得维护第二份定义 |
| 在 CI 里校验已登记的 UNIT.yaml | 先有 schema 文件与脚本，且社区 CodeCheck 的规则对齐后再启用 |
| 引入「模型层」目录（`<family>/<model>/<unit>/`） | 模型归属稳定、且同模型 ≥2 单元成为常态 |
| 把定级接成门禁（R7 棘轮） | 见上一行，且需要先补齐 reference/ 与 test_report/ |

## 10. 变更记录

| 版本 | 变更 |
|---|---|
| v1.0（2026-10-02，已废弃） | 首版：目录契约、UNIT.yaml 12 字段、L0–L4、三条硬规则 |
| v1.2（本版） | 对齐六族骨架；去掉模型层要求；UNIT.yaml 改为自愿登记；定级降为标注；补存量缺口实测与演进前置条件 |
