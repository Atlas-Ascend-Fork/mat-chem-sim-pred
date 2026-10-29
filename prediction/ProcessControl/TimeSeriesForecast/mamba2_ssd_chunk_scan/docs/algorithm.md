<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Mamba-2 SSD 分块扫描算法原理与 Ascend C Cube-Vector 异构流水设计

## 1. 算法背景与数学表述

**Mamba-2（State Space Duality, SSD）** 通过引入标量时间步衰减因子 $a_t = \exp(\Delta_t A_h) \in (0, 1)$（同一个头 $h$ 内所有状态维度 $N$ 共享同一个衰减率），将传统状态空间模型（SSM）的序列递推等价变换为**分块半可分矩阵乘法（Block-Structured Semiseparable Matrix Multiplication）**。

设序列总长度为 $L$，将其均匀切分为 $K = L / Q$ 个长度为 $Q=64$ 的时间分块（Chunk）。对于第 $c \in [0, K-1]$ 个时间分块及任意 $(b, h)$ 对，记输入块 $X^{(c)} \in \mathbb{R}^{Q \times P}$、离散步长 $\Delta^{(c)} \in \mathbb{R}^{Q}$、状态投影矩阵 $B^{(c)}, C^{(c)} \in \mathbb{R}^{Q \times N}$ 以及进入当前分块的跨块隐状态 $H^{(c-1)} \in \mathbb{R}^{N \times P}$（初始 $H^{(-1)} = \mathbf{0}$）。

单分块内的计算由以下 **4 个核心数学阶段** 组成：

### 阶段 1：块内一维衰减前缀和与输入缩放（AI Vector）
1. 计算对数衰减步长 $d_i = \Delta_i^{(c)} \cdot A_h \le 0$ 及其块内前缀和：
   $$S_i = \sum_{k=0}^{i} d_k, \quad i \in [0, Q-1]$$
2. 构造块内离散化缩放输入 $\widetilde{X} \in \mathbb{R}^{Q \times P}$ 与块末状态贡献输入 $\widehat{X} \in \mathbb{R}^{Q \times P}$：
   $$\widetilde{X}_{i, p} = \Delta_i^{(c)} X_{i, p}^{(c)}, \qquad \widehat{X}_{i, p} = \exp(S_{Q-1} - S_i) \widetilde{X}_{i, p}$$

### 阶段 2：块内语法耦合矩阵乘与跨块状态注入（AI Cube Wave 1）
利用 AI Cube 同时计算块内耦合矩阵 $G \in \mathbb{R}^{Q \times Q}$ 与跨块状态注入项 $Y_{\text{off}} \in \mathbb{R}^{Q \times P}$：
$$G = C^{(c)} \big(B^{(c)}\big)^T \in \mathbb{R}^{Q \times Q}, \qquad Y_{\text{off}}^{\text{raw}} = C^{(c)} H^{(c-1)} \in \mathbb{R}^{Q \times P}$$

### 阶段 3：因果指数衰减掩码（AI Vector）与对偶块内输出/状态增量（AI Cube Wave 2）
1. AI Vector 对耦合矩阵 $G$ 施加下三角指数衰减掩码，生成对偶核矩阵 $M \in \mathbb{R}^{Q \times Q}$：
   $$M_{i, j} = \begin{cases} G_{i, j} \exp(S_i - S_j), & \text{if } i \ge j \\ 0, & \text{if } i < j \endend{cases}$$
2. AI Cube 同时计算块内对偶输出 $Y_{\text{diag}} \in \mathbb{R}^{Q \times P}$ 与当前分块对隐状态的增量贡献 $\Delta H \in \mathbb{R}^{N \times P}$：
   $$Y_{\text{diag}} = M \widetilde{X} \in \mathbb{R}^{Q \times P}, \qquad \Delta H = \big(B^{(c)}\big)^T \widehat{X} \in \mathbb{R}^{N \times P}$$

### 阶段 4：总输出合成与片上常驻隐状态递推更新（AI Vector）
1. 合成当前分块最终输出 $Y^{(c)} \in \mathbb{R}^{Q \times P}$：
   $$Y_{i, p}^{(c)} = \exp(S_i) \big(Y_{\text{off}}^{\text{raw}}\big)_{i, p} + \big(Y_{\text{diag}}\big)_{i, p} + D_h X_{i, p}^{(c)}$$
2. 在片上 UB 内存中原地更新跨分块隐状态 $H^{(c)} \in \mathbb{R}^{N \times P}$（无需写回 HBM，直接服务第 $c+1$ 个分块）：
   $$H^{(c)} = \exp(S_{Q-1}) H^{(c-1)} + \Delta H$$

---

## 2. 昇腾 910B `KERNEL_TYPE_MIX_AIC_1_2` 硬件映射与异步乒乓流水

在昇腾 910B（`ascend910b`）上，核函数通过 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 请求 `1:2` 的 AI Cube 与 AI Vector 异构协同调度（详见 [`op_kernel/mamba2_ssd_chunk_scan_kernel.cpp`](../op_kernel/mamba2_ssd_chunk_scan_kernel.cpp)）：

### 2.1 $N=64, P=64, Q=64$ 的 4 级异步流水 (`ProcessChunkMix64Pipelined`)
```mermaid
sequenceDiagram
    participant HBM as HBM2e
    participant AIV as AI Vector Core (UB 192KB)
    participant L2 as L2 Cache / GM Workspace (14 Slots)
    participant AIC as AI Cube Core (Matmul Engine)

    HBM->>AIV: DMA Load Chunk c (X, dt, B, C)
    AIV->>AIV: Brcb 向量广播计算 S_i, X_tilde, X_hat
    AIV->>L2: MTE3 写出 B, C, X_tilde, X_hat, H_prev
    par Wave 1: 双路 Cube Matmul 异步并发
        AIV->>AIC: mm_cb_.IterateAll(C, B^T -> ws_g0, false)
        AIV->>AIC: mm_ch_.IterateAll(C, H_prev -> ws_yoff0, false)
    end
    AIC-->>L2: 写回 G = C * B^T
    L2->>AIV: MTE2 读回 G, 在 UB 内施加下三角因果掩码 M_{i,j} = G_{i,j} exp(S_i - S_j)
    AIV->>L2: MTE3 写出 M
    par Wave 2: 双路 Cube Matmul + 下一分块 HBM 预取重叠
        AIV->>AIC: mm_mx_.IterateAll(M, X_tilde -> ws_ydiag, false)
        AIV->>AIC: mm_bx_.IterateAll(B^T, X_hat -> ws_dh0, false)
        HBM->>AIV: IssueLoadChunkInputs(c+1) 异步预取下一分块
    end
    AIC-->>L2: 写回 Y_diag, dH
    L2->>AIV: 读回 Y_off, Y_diag, dH; 合成 Y^{(c)} 并更新 UB 常驻 H^{(c)}
    AIV->>HBM: DMA Store Y^{(c)}
```

### 2.2 $N=128, P=64, Q=64$ 的 7 级无别名异步流水 (`ProcessChunkMix128Pipelined`)
当状态容量扩展至 $N=128$ 时，我们将状态维度切分为低半区 (`lo: 0..63`) 与高半区 (`hi: 64..127`)，并在 Host 侧注册 **7 个独立 `AscendC::matmul::Matmul` 引擎**：
- **Wave 1（4 路 Cube 并发）**：同时非阻塞挂起 `mm_cb_` ($C_{\text{lo}} B_{\text{lo}}^T \to \text{ws\_g0}$)、`mm_cb_hi_` ($C_{\text{hi}} B_{\text{hi}}^T \to \text{ws\_g1}$)、`mm_ch_` ($C_{\text{lo}} H_{\text{lo}} \to \text{ws\_yoff0}$)、`mm_ch_hi_` ($C_{\text{hi}} H_{\text{hi}} \to \text{ws\_yoff1}$)；
- **Vector 归约与掩码**：AIV 从 `ws_g0` 与 `ws_g1` 读回两部分耦合矩阵，相加得到完整的 $G = C_{\text{lo}} B_{\text{lo}}^T + C_{\text{hi}} B_{\text{hi}}^T$，并施加下三角因果掩码得到 $M$；
- **Wave 2（3 路 Cube 并发 + 跨块 DMA 预取）**：同时非阻塞挂起 `mm_mx_` ($M \widetilde{X} \to \text{ws\_ydiag}$)、`mm_bx_` ($B_{\text{lo}}^T \widehat{X} \to \text{ws\_dh0}$)、`mm_bx_hi_` ($B_{\text{hi}}^T \widehat{X} \to \text{ws\_dh1}$)，同时预取分块 $c+1$ 的输入数据，消除半区串行等待。
