# common — 跨算子公共层

本目录只放**真正跨算子复用**的头文件与实现，目标是把"同一份逻辑在 N 个算子目录里各抄一遍"收敛到一处。

## 目录

```text
common/
├── inc/
│   ├── op_host/      # 参数校验、tiling、workspace 辅助
│   ├── op_kernel/    # device 侧公共结构体、索引与数学工具
│   └── op_api/       # aclnn 入口共用的类型与宏
├── src/              # 确有实现时新增；当前为空
└── tests/            # 公共层的 UT；当前为空
```

## 收什么、不收什么

收：

- 被 3 个及以上算子使用，且语义已经完全一致的逻辑；
- 纯头文件工具（shape 推导、索引计算、错误码包装）；
- 与具体算子业务无关的常量与类型。

不收：

- 只被 1–2 个算子使用的"顺手抽出来"的代码——留在算子自己的 `op_host/` 或族级 `common/`；
- 带业务分支的"万能 kernel"；
- 与 CANN 版本强耦合、需要按 SOC 分叉的实现（这类放 `cmake/` 或算子自身 `op_kernel/arch*/`）。

## 与算子目录内 `common/` 的关系

族级 `common/`（例如 `pid/common/`）只服务本族；
本目录服务全仓。提升路径：族级 `common/` → 出现第二个族的使用方 → 迁到本目录，并补 UT。

## 构建

顶层 `CMakeLists.txt` 通过 `add_subdirectory(common)` 挂载 `matchem_common`（INTERFACE）。
算子工程若需使用，显式 `target_link_libraries(<target> PRIVATE matchem_common)`；
在算子仍为独立 `project()` 的阶段，用 `-DMATCHEM_COMMON_INC=<repo>/common/inc/...` 或相对路径包含。
