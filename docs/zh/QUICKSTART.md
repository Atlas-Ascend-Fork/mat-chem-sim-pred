# 快速入门

本文只讲"怎么把它跑起来"。算子原理、API 与性能数据见各算子目录的 `docs/`。

## 1. 环境

| 依赖 | 要求 | 用途 |
|---|---|---|
| CANN Toolkit | ≥ 7.0，含 Ascend C 编译链 | 编译算子、跑 NPU 用例 |
| Python | 3.9+ | 参考实现、pytest |
| CMake | ≥ 3.16 | 每个算子独立配置 |
| 编译器 | 支持 C++17 | Host 侧 |
| GTest | 可选 | C++ UT；缺失时自动跳过 |

安装与 `set_env.sh` 的细节见 [install/quick_install.md](install/quick_install.md)。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 2. 不需要 CANN 就能跑的检查

```bash
# 算子发现（读 cmake/variables.cmake，不调 cmake）
bash build.sh --list

# 目录契约测试：命名、README、tests、reference、构建产物
python -m pytest tests/ut -q

# Python 参考实现
bash build.sh --python experimental/SmallData
```

`--list` 会打印三类：可构建算子、Python 算子、暂缓构建的算子。

## 3. 编译单个算子

```bash
bash build.sh md/lennard_jones
bash build.sh --ops=lennard_jones,pinn -j8
bash build.sh --soc=Ascend910B3 md/gaff2
```

产物在 `<op>/build/`。清理：

```bash
bash build.sh --clean --all
```

## 4. 跑测试

```bash
# 编译 + ctest（需要 GTest；没有则只编译）
bash build.sh --ops=lennard_jones --test

# 只编译，不跑 ctest
bash build.sh lennard_jones --build-only

# 参考实现（CPU / pytest）
bash build.sh --reference-test pid/,time_series/,molecular/ -f pr_filelist.txt -j8
```

`--reference-test` 的契约：有 `tests/` 的算子必须提供 `reference/reference.py` 或 `reference/__init__.py`；
缺失会打印 `[REF-MISSING]` 并以非零退出，不允许"空跑假绿"。

## 5. 增量门禁（CI 同款）

CI 通过 `pr_filelist.txt` 驱动 `build.sh -f`：

```bash
# 生成文件清单（与 CI 的 get-pr 同口径）
git diff --name-only origin/master...HEAD > pr_filelist.txt

# 只构建受影响的算子
bash build.sh -f pr_filelist.txt --test

# 限定范围 + 声明共享目录
bash build.sh -f pr_filelist.txt --test \
  --scope=pid/,time_series/,molecular/ \
  --common-dirs="molecular/"

# 纯文档 PR：直接短路，不编译
bash build.sh -f pr_filelist.txt
```

规则：

- 文件清单全是 `.md` → 直接退出 0，并写 `build_out/skip_build.run`。
- 共享目录变更 → 该 scope 下所有算子重建；`DirectMatchDir` 只认目录下的直接文件。
- 清单里没有算子路径 → 退出 0，避免"没变更却报红"。

## 6. 新增算子

1. 在 `cmake/variables.cmake` 已登记的族目录下建算子目录（小写下划线）；
2. 至少包含 `CMakeLists.txt`、`README.md`、`op_host/`、`op_kernel/`、`examples/`、`tests/`、`reference/`；
3. `bash build.sh --list` 应能发现它，`python -m pytest tests/ut -q` 应通过；
4. 不需要改 `build.sh`、顶层 `CMakeLists.txt` 或 CI 工作流。

目录细节见 [appendix/dir_structure.md](appendix/dir_structure.md)，交付要求见 [CONTRIBUTING.md](../../CONTRIBUTING.md)。

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `cmake: command not found` | 未装 CMake | `bash install_deps.sh` 或自行安装 ≥ 3.16 |
| `Cannot find ascendc_kernel_cmake` | 未 source CANN 环境 | `source .../set_env.sh`，或设 `ASCEND_CANN_PACKAGE_PATH` |
| `[WARN] GTest not found` | 镜像无 GTest | 只编译不跑 UT，或在 CI 镜像内运行 |
| `[REF-MISSING]` | 有 tests 无 reference | 补 `reference/reference.py`，见 CONTRIBUTING |
| 改动没触发构建 | 文件不在任何算子目录下 | 确认路径，或用 `--common-dirs` 声明共享目录 |
