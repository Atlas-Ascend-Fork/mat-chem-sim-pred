# 环境安装

## 1. CANN Toolkit

按昇腾社区 CANN 安装指南安装 Toolkit（含 Ascend C 编译链），并确认：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "$ASCEND_TOOLKIT_HOME"
which bisheng   # 或 ccec，取决于 CANN 版本
```

若 Toolkit 不在默认路径，设置：

```bash
export ASCEND_CANN_PACKAGE_PATH=/path/to/ascend-toolkit/latest
```

`build.sh` 与各算子 `CMakeLists.txt` 都优先读取该变量，其次 `ASCEND_TOOLKIT_HOME`。

## 2. 构建依赖

```bash
bash install_deps.sh
```

脚本按发行版安装：`cmake`、`gcc/g++`、`make`、`python3`、`python3-pip`、`git`、`dos2unix`。
GTest 为可选依赖：缺失时 `build.sh --test` 只编译、不跑 C++ UT，并打印一次告警。

## 3. Python 依赖

```bash
python3 -m pip install -r requirements.txt
```

参考实现使用 NumPy / PyTorch；DAO 还依赖 `pypto`，`build.sh --python` 会在源码确实引用时按需安装。

## 4. 无 NPU 的机器能做什么

| 操作 | 是否需要 NPU | 说明 |
|---|---|---|
| `bash build.sh --list` | 否 | 只读目录与配置 |
| `python -m pytest tests/ut -q` | 否 | 目录契约测试 |
| `bash build.sh --python <op>` | 否 | CPU / pytest |
| `bash build.sh --reference-test ...` | 否 | CPU / pytest |
| `bash build.sh <op>` | 需要 CANN（不一定要卡） | 交叉编译可用 CI 镜像 |
| `ctest` 中下设备的用例 | 需要 | 无卡时只跑 CPU 侧断言 |

## 5. 容器与 CI

CI 使用 CANN 社区镜像（见 `.gitcode/workflows/`）。本地复刻门禁的脚本在 `.local-ci/`（不随仓提交），
其编排与 CI 一致：`pr_filelist.txt` + `build.sh`。
