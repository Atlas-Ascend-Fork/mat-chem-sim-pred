# 算子后端（Ascend C / PyPTO）

仓库不再假设“算子 = CMake + Ascend C”。后端（backend）与能力族（family）是两个独立维度：

- family：`md/`、`pde/`、`graph/`、`geometry/`、`pid/`、`time_series/`，按业务能力划分；
- backend：`ascendc`、`pypto`，按实现与构建方式划分。

## 1. 算子声明

每个算子可在根目录放 `op.yaml`：

```yaml
backend: pypto
soc:
  - Ascend910B1
  - Ascend910C
build: python3 -m pip install -e .
test: python3 -m pytest tests -q
package: python3 -m build --wheel
```

- 没有 `op.yaml` 的存量算子按 `backend: ascendc` 兼容；
- `build` / `test` / `package` 是 shell 命令，在算子目录内执行；
- Ascend C 算子不需要写这些命令，继续走 CMake 路径；
- PyPTO 算子必须有 `pyproject.toml` 和 `tests/`。

## 2. build.sh 的分派

```bash
# 只处理 Ascend C 算子
bash build.sh -f pr_filelist.txt --backend=ascendc --build-only

# 只处理 PyPTO 算子
bash build.sh -f pr_filelist.txt --backend=pypto --test

# 全部（默认）
bash build.sh --list
```

分派规则：

| backend | 发现条件 | 编译 | 测试 | 打包 |
|---|---|---|---|---|
| ascendc | `CMakeLists.txt` | cmake + ascendc | GTest + ctest | msopgen custom opp `.run` |
| pypto | `op.yaml` | `op.yaml: build` | `op.yaml: test` | `op.yaml: package` |

`--backend` 先过滤，再 `--shard` 分片；PyPTO job 不会混进 Ascend C 的 shard。

## 3. CI

主流水线新增两个 job，默认关闭：

```yaml
env:
  PYPTO_ENABLED: ${{ vars.PYPTO_ENABLED }}
```

- `JOB_compile_pypto`：stage2 Compile，`--backend=pypto --build-only`
- `JOB_ut_pypto`：stage3 UT，`--backend=pypto --test`

两个 job 默认复用 Ascend C 的 ARM 镜像：

```yaml
image: ${{ vars.PYPTO_IMAGE || format('.../ci_cann/{0}', jobs.JOB_image.outputs.Compile_Ascend_ARM) }}
```

`PYPTO_IMAGE` 只是可选覆盖，不是必填。启用条件：

1. 仓库变量 `PYPTO_ENABLED=true`；
2. Ascend C 的 ARM 镜像里预装与 CANN 匹配的 pypto（推荐），或设置
   `PYPTO_IMAGE` 指向同版本、带 pypto 的镜像；
3. 至少有一个 `backend: pypto` 的算子。

两个 job 在构建前都会跑：

```bash
python3 -c "import pypto; print(pypto.__version__)"
```

导入失败或版本不匹配会直接失败，不会静默跳过。

如果 PyPTO 只能 x86 编译，把两个 job 的 `runs-on` 改回 `arch=x64`，并设置
`PYPTO_IMAGE` 为同版本 x86 镜像；环境版本仍保持同一套，不维护两套无关环境。

## 4. 依赖与版本

PyPTO 与 CANN 版本强相关，CI 必须在 job 开头做版本校验：

```bash
python3 -c "import pypto; print(pypto.__version__)"
```

建议：

- 在镜像里预装 pypto，而不是 job 里临时 `pip install`；
- 若必须安装，使用内网 wheel 或 `requirements/pypto-<cann>.txt` 钉版本；
- 版本不匹配直接失败，不要让它退化成 `ModuleNotFoundError` 或静默跳过。

## 5. 打包形态

先确定 PyPTO 算子的交付物：

- 产出 custom opp `.run`：`package` 命令调用对应打包工具，产物仍按
  `ASCEND_CUSTOM_OPP_PATH` 安装；
- 产出 Python wheel/sdist：`package` 命令默认 `python3 -m build --wheel`，
  产物在 `build_out/pypto/`，用 pip 安装。

不要假设 PyPTO 一定能复用 `package_opp.sh`；它只负责 Ascend C 的 msopgen 流程。

## 6. 新增第一个 PyPTO 算子

```bash
mkdir -p time_series/my_pypto_op/tests
# 写 op.yaml、pyproject.toml、实现和 tests
bash build.sh --backend=pypto --list          # 应能看到它
bash build.sh time_series/my_pypto_op --test  # 本地跑 build + test
python3 -m pytest tests/ut -q                 # R7 对 pypto 不要求 msopgen
```

然后在仓库变量里打开 `PYPTO_ENABLED`，CI 才会真正跑 PyPTO job。
