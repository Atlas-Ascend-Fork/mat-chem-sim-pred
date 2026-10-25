# geometry — 点云与分子几何算子族

把 PointNet++、DimeNet、GemNet 等上游模型里“框架不支持或需要多次
kernel launch”的几何算子，下沉为单个 Ascend C kernel。

## 范围

- 点云邻域与采样：FarthestPointSampling、BallQuery；
- 点对特征：PPF（Point Pair Feature）；
- 分子图多体几何：DimeNet 三元组枚举与角度、GemNet 四元组几何。

图传播、图注意力与图池化在 [`graph/`](../graph/README.md)；其中的
`csr_point_transformer_attention_aggregate_fused` 因为走 CSR 消息传递，归在 graph。

不在本族范围：过程控制（`pid/`）、时序预测（`time_series/`）、分子动力学力场（`md/`）。

## 目录约定

```text
geometry/
├── README.md                       # 本页
├── geometry_math.h                 # 族内共享：向量/距离/角度
├── pointnet2_geometry_*common.*    # 族内共享：点云邻域与 benchmark
├── acl_smoke_helpers.h             # 族内共享：ACL 冒烟测试
└── <op>/                           # 每个算子：CMakeLists + op_host + op_kernel + tests + reference
```

共享头只服务本族；出现第二个族的使用方时提升到顶层 `common/`。

## 构建与测试

```bash
bash build.sh geometry/farthest_point_sampling_fused
bash build.sh --ops=farthest_point_sampling_fused --test
bash build.sh --reference-test geometry/ -f pr_filelist.txt
```

顶层聚合（需 CANN）：

```bash
cmake -S . -B build -DMATCHEM_OP=geometry/farthest_point_sampling_fused
cmake --build build -j
```

逐算子说明与状态见 [docs/zh/op_list.md](../docs/zh/op_list.md)。
