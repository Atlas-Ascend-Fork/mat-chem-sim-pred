<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Mamba2SsdChunkScan API 参考

## 1. 算子基本信息

| 项目 | 内容 |
|---|---|
| OpDef 名称 | `Mamba2SsdChunkScan` |
| ACLNN API | `aclnnMamba2SsdChunkScan` |
| 输入 1 (`u`) | `float32 [B, L, H, P]`, Format: `ND` (主干输入特征) |
| 输入 2 (`dt`) | `float32 [B, L, H]`, Format: `ND` (连续时间离散化步长 $\Delta$) |
| 输入 3 (`a`) | `float32 [H]`, Format: `ND` (对角衰减参数，通常为负实数) |
| 输入 4 (`b`) | `float32 [B, L, H, N]`, Format: `ND` (状态投影矩阵 $B$) |
| 输入 5 (`c`) | `float32 [B, L, H, N]`, Format: `ND` (状态观测矩阵 $C$) |
| 输入 6 (`d`) | `float32 [H, P]`, Format: `ND` (跳跃直通连接权重 $D$) |
| 输出 (`out`) | `float32 [B, L, H, P]`, Format: `ND` (扫描融合输出) |
| 属性 (`chunk_size`) | `int64`, 默认值: 64 (分块扫描块长度 $Q$) |
| Workspace 大小 | 0 Byte (片上 UB 缓存流转，无需额外全局显存) |
| 目标硬件平台 | `Ascend910B` (`ai_core-ascend910b`) |

## 2. ACLNN C++ 调用签名

```cpp
#include "acl/acl.h"
#include "aclnn_mamba2_ssd_chunk_scan.h"

// 阶段 1: 查询所需 Workspace 显存并生成算子执行器
aclnnStatus aclnnMamba2SsdChunkScanGetWorkspaceSize(
    const aclTensor* u,
    const aclTensor* dt,
    const aclTensor* a,
    const aclTensor* b,
    const aclTensor* c,
    const aclTensor* d,
    int64_t chunkSize,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

// 阶段 2: 发射至昇腾硬件 Stream 执行
aclnnStatus aclnnMamba2SsdChunkScan(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

### 返回值说明
- `ACL_SUCCESS (0)`: 执行成功。
- `ACL_ERROR_INVALID_PARAM`: 参数校验失败（如张量为空、形状不符合约束、数据类型非 float32 等）。
- `ACL_ERROR_INTERNAL_ERROR`: CANN 内部运行时异常。

## 3. Python Reference 接口

供纯 CPU 数值基准与离线算法校验使用：

```python
from reference import reference

# u: [B, L, H, P], np.float32
# dt: [B, L, H], np.float32
# a: [H], np.float32
# b: [B, L, H, N], np.float32
# c: [B, L, H, N], np.float32
# d: [H, P], np.float32 (可选，若为 None 则默认全零)
# chunk_size: int, 默认 64
out = reference(u, dt, a, b, c, d=d, chunk_size=64)
```

## 4. 参数约束与校验规则

1. **输入张量形状合同**：
   - `u.dim == 4`: 对应 `(B, L, H, P)`，且各维度严格大于 0；
   - `dt.dim == 3`: 对应 `(B, L, H)`，必须与 `u` 的前三维完全一致；
   - `a.dim == 1`: 对应 `(H,)`，必须与多头数 $H$ 严格相等；
   - `b.dim == 4` 且 `c.dim == 4`: 对应 `(B, L, H, N)`，状态维度 $N \ge 1$；
   - `d.dim == 2`: 对应 `(H, P)`，与多头数及头维度严格匹配；
   - `out.dim == 4`: 对应 `(B, L, H, P)`，与输入 `u` 的形状严格一致。
2. **数据类型**：
   - 统一支持 `ACL_FLOAT` (float32)。
3. **数据格式**：
   - 必须为连续存储（Contiguous）的 `ACL_FORMAT_ND`。
4. **属性合法性**：
   - `chunk_size` 必须大于 0 且 $\le 512$。

## 5. ACLNN 工程代码生成命令

在安装有 CANN Toolkit 的环境中使用 `msopgen` 生成标准构建框架：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msopgen gen -i msopgen/mamba2_ssd_chunk_scan_msopgen.json \
    -f aclnn \
    -c ai_core-ascend910b \
    -out build/msopgen_mamba2_ssd_chunk_scan \
    -lan cpp
```
