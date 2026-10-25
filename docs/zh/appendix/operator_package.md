# 算子包（custom opp）说明

## 1. 为什么需要算子包

仓库日常的 `build.sh` 只产出每个算子的 `.so` 编译产物，不是可安装的 CANN 算子包。
这带来三个问题：

- UT 只能从源码重新编译产品，Compile/UT 两个 stage 会重复编译；
- PreSmoke/ST 无法“装包后在真机上调 aclnn”；
- 对外没有可安装的交付物，多芯片也无法按 SOC 分发。

算子包是把算子以 CANN custom opp 的布局打包成 `.run`，安装后通过
`ASCEND_CUSTOM_OPP_PATH` 生效。

## 2. 当前能力与边界

已实现：

- `scripts/package_opp.sh`：msopgen 生成工程 → 覆盖仓内 `op_host/op_kernel` →
  调用生成工程的 `build.sh` → 收集 `custom_opp_*.run`；
- `build.sh --package`：按 filelist/算子列表/分片选择算子后打包；
- `.gitcode/workflows/opp_package.yml`：手动 ARM 打包流水线，上传 OBS。
- `cmake/variables.cmake` 的 `MATCHEM_SOCS`：910B / 910C / 950 的目标清单。

边界：

- 只有带 `msopgen/*.json` 的算子能打包。当前是 `time_series/` 下的 15 个算子；
  其余算子没有原型定义，`--package` 会跳过，`--package-require-all` 会报错。
- 真实打包依赖 ARM + CANN 的 `msopgen` 和生成工程的 `build.sh`；本机没有 CANN 时
  只能做编排级验证（fake msopgen）。
- 打包尚未接入 PR 门禁；等 P0 在真机验证通过、所有目标算子补齐 msopgen JSON 后，
  再把 Compile stage 改成“产包 + 上传”，UT stage 改成“下载 + 装包”。

## 3. 使用方式

单个算子：

```bash
bash build.sh --package --soc=Ascend910B1 \
  --package-out=build_out/opp \
  time_series/cfc_scan_fused
```

一组算子（分片）：

```bash
bash build.sh -f pr_filelist.txt --package --soc=Ascend910B1 \
  --package-out=build_out/opp --shard=1/2
```

要求所有选中算子都必须有 msopgen JSON：

```bash
bash build.sh --package --package-require-all --soc=Ascend910B1 \
  time_series/cfc_scan_fused
```

输出：

```text
build_out/opp/
├── cfc_scan_fused_custom_opp_linux_aarch64.run
└── manifest.json
```

`manifest.json` 记录 SOC、framework、算子与包文件名的对应关系。

## 4. 打包流程

```text
msopgen gen -i <op>/msopgen/*.json -f tf -c ai_core-<soc> -lan cpp -out <project>
  → 覆盖 <project>/op_host、<project>/op_kernel 为仓内实现
  → 复制族 common/ 到 project 的 ../common（供 ../../common 头文件使用）
  → cd <project> && bash build.sh
  → 收集 <project>/build_out/custom_opp_*.run
```

## 5. 安装与消费

```bash
./cfc_scan_fused_custom_opp_linux_aarch64.run --install-path=/tmp/matchem-opp
source /tmp/matchem-opp/vendors/customize/bin/set_env.bash
echo "$ASCEND_CUSTOM_OPP_PATH"
```

安装后的目录形如：

```text
/tmp/matchem-opp/vendors/customize/
├── framework/
├── op_api/{include,lib}
├── op_impl/ai_core/tbe/{config,kernel,op_tiling}
└── op_proto/{inc,lib}
```

UT/PreSmoke 消费包时，应在安装后 `source set_env.bash`，再编译/运行测试，
而不是继续使用源码 build 目录。

## 6. 多芯片

`cmake/variables.cmake` 的 `MATCHEM_SOCS` 定义目标：

| id | SOC_VERSION | 门控 |
|---|---|---|
| 910b | Ascend910B1 | pr |
| 910c | Ascend910C | master |
| 950 | Ascend950PR_9579 | master |

每个 SOC 单独打包、单独上传：

```bash
bash build.sh --package --soc=Ascend910B1 --package-out=build_out/opp_910b <ops>
bash build.sh --package --soc=Ascend910C  --package-out=build_out/opp_910c <ops>
bash build.sh --package --soc=Ascend950PR_9579 --package-out=build_out/opp_950 <ops>
```

CI 侧手动流水线 `opp_package.yml` 用 `soc` 输入选择目标；后续接 PR 门禁时，
每个 SOC 一个 Compile job，UT/PreSmoke 按 SOC 消费对应包。

## 7. 手动流水线

`.gitcode/workflows/opp_package.yml`：

- 触发：GitCode 流水线页面手动运行；
- 输入：`soc`、`framework`、`ops`（空 = 所有 time_series 算子）；
- 运行环境：`vars.OPP_ARM_IMAGE` 指定的 ARM + CANN 镜像；
- 输出：`build_out/opp_<soc>.tar.gz` 上传到
  `${{ vars.OBS_PATH }}/opp/<soc>/`。

该流水线不参与 PR 门禁，失败不影响合入；用于验证打包链路和产出可安装包。
