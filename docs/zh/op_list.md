# 算子清单

> 本页是仓库全部算子的唯一清单。README 只保留能力矩阵，数量口径以本页为准。

## 统计口径

- 算子目录：60 个（`op_host/` + `op_kernel/` 齐备的独立算子目录）。
- 纳入构建：59 个；`md/dissipative_particle_dynamics` 实现未完成，暂缓 CI（`cmake/variables.cmake` 的 `MATCHEM_SKIP_OPERATORS`）。
- 另有两个 Python 参考实现不进算子计数：`experimental/DAO`、`experimental/SmallData`。
- 目录命名以实际路径为准；命名不合规的存量算子记录在 `tests/ut/layout_exceptions.json`，新增算子必须小写下划线。
- 阶段口径：研发域 = `md/`、`pde/`、`molecular/`（设计、仿真、试制）；生产域 = `pid/`、`time_series/`（控制、质量、运行）。

## 分子动力学（md）

目录：`md/`；算子数：6。经典力场的力、静电、约束、积分与耗散粒子动力学。为 ML-MD 提供底层计算能力；不含机器学习势推理。

| 算子 | 说明 | 状态 |
|---|---|---|
| `lennard_jones` | Lennard-Jones 力场 | ✅ Ascend C |
| `gaff2` | GAFF2 力场 | ✅ Ascend C |
| `pme` | PME 静电求和 | ✅ Ascend C |
| `shake` | SHAKE 约束算法 | ✅ Ascend C |
| `velocity_verlet` | Velocity Verlet 积分器 | ✅ Ascend C |
| `dissipative_particle_dynamics` | DPD 耗散粒子动力学 | ✅ Ascend C（暂缓 CI） |

## AI for PDE（pde）

目录：`pde/`；算子数：3。PINN / FNO / DeepONet 的推理算子。

| 算子 | 说明 | 状态 |
|---|---|---|
| `pinn` | PINN 物理信息神经网络推理 | ✅ Ascend C |
| `fno` | FNO 傅里叶神经网络算子推理 | ✅ Ascend C |
| `deeponet` | DeepONet 深度算子网络推理 | ✅ Ascend C |

## PID 整定（pid）

目录：`pid/`；算子数：12。12 个算子按模型辨识 → 诊断 → 整定 → 仿真选优 → 评估组织。

| 算子 | 说明 | 状态 |
|---|---|---|
| `pid_fopdt_basis_gemm_fit` | FOPDT 模型辨识（basis-GEMM 加速） | ✅ Ascend C |
| `pid_ipdt_basis_gemm_fit` | IPDT 模型辨识（basis-GEMM 加速） | ✅ Ascend C |
| `pid_sopdt_basis_gemm_fit` | SOPDT 模型辨识（basis-GEMM 加速） | ✅ Ascend C |
| `pid_residual_diagnostics` | 残差诊断 | ✅ Ascend C |
| `pid_windowed_residual_diagnostics` | 窗口残差诊断 | ✅ Ascend C |
| `pid_tuning_rule_batch` | 批量整定规则（Ziegler-Nichols / IMC / Cohen-Coon） | ✅ Ascend C |
| `pid_fopdt_batch_rollout_score` | FOPDT 候选闭环仿真 + 评分 + best 选择 | ✅ Ascend C |
| `pid_ipdt_batch_rollout_score` | IPDT 候选闭环仿真 + 评分 + best 选择 | ✅ Ascend C |
| `pid_sopdt_batch_rollout_score` | SOPDT 候选闭环仿真 + 评分 + best 选择 | ✅ Ascend C |
| `pid_step_response_features` | 阶跃响应特征提取（上升时间/超调/稳态误差等） | ✅ Ascend C |
| `pid_control_performance_metrics` | 控制性能指标（IAE/ISE/ITAE 等） | ✅ Ascend C |
| `pid_process_capability_metrics` | 过程能力指数（Cp/Cpk 等） | ✅ Ascend C |

## 时序预测（time_series）

目录：`time_series/`；算子数：17。面向 DCS/传感器时序：SSM 扫描、lag 聚合、LSH 分桶与框架不支持路径下沉。

| 算子 | 说明 | 状态 |
|---|---|---|
| `selective_scan_1d` | 时间维选择性扫描融合 | ✅ Ascend C |
| `s6scan_fused` | S6 扫描 + softplus + 累积融合 | ✅ Ascend C |
| `autocorr_fused_aggregate` | 自相关筛选 + lag 权重 + 时移聚合 | ✅ Ascend C |
| `batch_spd_inv_fp32` | 批量 SPD 矩阵求逆，替换 `torch.linalg.lstsq` CPU fallback | ✅ Ascend C |
| `tirex_slstm_cell` | sLSTM 门控 + 稳定化 + 状态更新融合 | ✅ Ascend C |
| `cfc_scan_fused` | CfC 递归扫描融合 | ✅ Ascend C |
| `cornn_scan_fused` | coRNN 递归扫描融合 | ✅ Ascend C |
| `sru_scan_fused` | SRU 递归扫描融合 | ✅ Ascend C |
| `unicornn_scan_fused` | UnICORNN 递归扫描融合 | ✅ Ascend C |
| `ltc_scan_fused` | LTC 递归扫描融合 | ✅ Ascend C |
| `reformer_lsh_bucket_sort` | LSH 分桶排序 + sticker/inverse 构造融合 | ✅ Ascend C |
| `reformer_lsh_qkv_gather` | 复用 LSH index 整理 query/key 与 value | ✅ Ascend C |
| `autoformer_inference_aggregate_fused` | 均值归约 + TopK + softmax + 循环移位加权聚合融合 | ✅ Ascend C |
| `mamba2_ssd_chunk_scan` | Mamba-2 SSD 分块扫描的四阶段 Cube-Vector 融合（910B/950） | ✅ Ascend C |
| `timesnet2d_period_fold_aggregate` | TimesNet 二维多周期折叠 + stencil 滤波 + softmax 幅度聚合 | ✅ Ascend C |
| `fused_kan_spline_basis_linear` | TimeKAN 三阶 Cox-de Boor B 样条基 + SiLU 残差收缩融合 | ✅ Ascend C |
| `anomaly_association_discrepancy_kl_fused` | Anomaly Transformer prior/series 对称 KL 差异融合 | ✅ Ascend C |

## 图、点云与分子几何（molecular）

目录：`molecular/`；算子数：22。把 PyG / PointNet++ / DimeNet / GemNet 的稀疏图传播、注意力聚合、邻域查询与多体几何下沉为单 kernel。

| 算子 | 说明 | 状态 |
|---|---|---|
| `csr_sage_mean_root_relu_fused` | 邻居均值 + root 分支 + ReLU 融合 | ✅ Ascend C |
| `csr_lightgcn_k2_weighted_sum_fused` | K=2 对称归一化两跳传播 + 层加权求和 | ✅ Ascend C |
| `csr_signed_cross_mean_pack_fused` | 第二层正/负邻域交叉均值打包 | ✅ Ascend C |
| `csr_arma_stack_propagate_fused` | K 路并行栈式加权传播 + root/bias + ReLU | ✅ Ascend C |
| `csr_appnp_propagate_fused` | 定步数传播阶段乒乓 workspace 融合 | ✅ Ascend C |
| `csr_hypergraph_attention_two_stage_propagate_fused` | 节点-超边-节点两阶段注意力传播 | ✅ Ascend C |
| `csr_tagcn_basis_k3_fused` | `K=3` 四阶特征基逐级传播 | ✅ Ascend C |
| `csr_gcn2_residual_propagate_fused` | 传播 + 初始残差融合 | ✅ Ascend C |
| `csr_transformer_dot_attention_aggregate_fused` | 缩放点积多头注意力 + CSR softmax + 聚合 | ✅ Ascend C |
| `farthest_point_sampling_fused` | 点云最远点采样（FPS）中心点选取 | ✅ Ascend C |
| `point_ball_query_fused` | 球半径邻域查询 + 距离过滤 + 有效计数 | ✅ Ascend C |
| `csr_point_transformer_attention_aggregate_fused` | 位置注意力打分 + 分段 softmax + 位置增量聚合 | ✅ Ascend C |
| `ppf_point_pair_features_fused` | 点对特征（距离 + 三个法向角）计算 | ✅ Ascend C |
| `dimenet_triplet_enumerate_fused` | 方向性 `k→j→i` 三元组枚举 | ✅ Ascend C |
| `dimenet_triplet_angle_fused` | 三元组角度 `atan2` 几何计算 | ✅ Ascend C |
| `gemnet_quadruplet_geometry_fused` | 官方 GemNet 四元组几何（角度与投影扭转角） | ✅ Ascend C |
| `attentional_aggregation_fused` | PyG AttentionalAggregation 的稳定 softmax 加权图池化 | ✅ Ascend C |
| `csr_agnn_cosine_attention_aggregate_fused` | AGNN 余弦打分 + CSR 分段 softmax + 加权聚合 | ✅ Ascend C |
| `csr_genconv_softmax_aggregate_fused` | GENConv + SoftmaxAggregation 的消息与聚合融合 | ✅ Ascend C |
| `csr_supergat_dot_attention_aggregate_fused` | SuperGAT-SD 缩放点积注意力消息阶段融合 | ✅ Ascend C |
| `edgepool_sorted_rank_contract_fused` | PyG EdgePooling 的贪心端点匹配与簇 ID 生成 | ✅ Ascend C |
| `sort_aggregation_topk_fused` | DGCNN SortAggregation 的稳定排序 + TopK + 零填充 | ✅ Ascend C |

---

合计 60 个算子目录。
