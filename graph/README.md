# graph — 图学习算子族

把 PyTorch Geometric 等上游模型里“框架不支持或需要多次 kernel launch”的稀疏图算子，
下沉为单个 Ascend C kernel。

## 范围

- 稀疏图传播与注意力聚合：GraphSAGE / LightGCN / Signed GCN / ARMA / APPNP /
  Hypergraph / GCNII / TAGCN / Graph Transformer / AGNN / GENConv / SuperGAT；
- 图池化：AttentionalAggregation、EdgePooling、SortAggregation。

GAT / GATv2 / ChebNet / FiLM 四个算子因外部环境验证有问题，已移到
`experimental/`，暂不纳入本族构建与计数。

点云与分子几何（FPS、BallQuery、PPF、Point Transformer、DimeNet、GemNet）在
[`geometry/`](../geometry/README.md)。

不在本族范围：过程控制（`pid/`）、时序预测（`time_series/`）、分子动力学力场（`md/`）。

## 目录约定

```text
graph/
├── README.md                 # 本页
├── attention_*_common.h      # 族内共享：注意力打分/softmax/聚合
├── graph_message_*_common.h  # 族内共享：CSR 消息传递
├── spectral_*_common.*       # 族内共享：谱基与 benchmark
└── <op>/                     # 每个算子：CMakeLists + op_host + op_kernel + tests + reference
```

共享头只服务本族；出现第二个族的使用方时提升到顶层 `common/`。

## 构建与测试

```bash
bash build.sh graph/csr_sage_mean_root_relu_fused
bash build.sh --ops=csr_sage_mean_root_relu_fused --test
bash build.sh --reference-test graph/ -f pr_filelist.txt
```

顶层聚合（需 CANN）：

```bash
cmake -S . -B build -DMATCHEM_OP=graph/csr_sage_mean_root_relu_fused
cmake --build build -j
```

逐算子说明与状态见 [docs/zh/op_list.md](../docs/zh/op_list.md)。
