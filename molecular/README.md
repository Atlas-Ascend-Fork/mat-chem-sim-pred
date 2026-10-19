# molecular — 图、点云与分子几何算子族

把 PyTorch Geometric、PointNet++、DimeNet/GemNet 等上游模型里"框架不支持或需要多次 kernel launch"
的子图，下沉为单个 Ascend C kernel。

## 范围

- 稀疏图传播与注意力聚合：GAT / GATv2 / GraphSAGE / LightGCN / Signed GCN / ARMA / APPNP /
  FiLM / Hypergraph / ChebNet / GCNII / TAGCN / Graph Transformer / AGNN / GENConv / SuperGAT；
- 图池化：AttentionalAggregation、EdgePooling、SortAggregation；
- 点云与三维几何：FarthestPointSampling、BallQuery、Point Transformer、PPF；
- 分子图多体几何：DimeNet 三元组枚举与角度、GemNet 四元组几何。

不在本族范围：过程控制（`pid/`）、时序预测（`time_series/`）、分子动力学力场（`md/`）。

## 目录约定

```text
molecular/
├── README.md                 # 本页
├── attention_*_common.h      # 族内共享：注意力打分/softmax/聚合
├── graph_message_*_common.h  # 族内共享：CSR 消息传递
├── spectral_*_common.*       # 族内共享：谱基与 benchmark
├── pointnet2_*_common.*      # 族内共享：点云几何
└── <op>/                     # 每个算子：CMakeLists + op_host + op_kernel + tests + reference
```

共享头只服务本族；出现第二个族的使用方时提升到顶层 `common/`。

## 构建与测试

```bash
bash build.sh molecular/csr_gat_attention_aggregate_fused
bash build.sh --ops=csr_gat_attention_aggregate_fused --test
bash build.sh --reference-test molecular/ -f pr_filelist.txt
```

顶层聚合（需 CANN）：

```bash
cmake -S . -B build -DMATCHEM_OP=molecular/csr_gat_attention_aggregate_fused
cmake --build build -j
```

逐算子说明与状态见 [docs/zh/op_list.md](../docs/zh/op_list.md)。
