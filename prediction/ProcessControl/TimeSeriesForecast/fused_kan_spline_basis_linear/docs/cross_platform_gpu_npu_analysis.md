# `fused_kan_spline_basis_linear` (`aclnnFusedKanSplineBasisLinear`)

## 1. Operator Mathematical Specification & Architecture
**TimeKAN Order-3 Cox-de Boor B-Spline Basis & SiLU Residual Contraction** (`aclnnFusedKanSplineBasisLinear`) is a production-grade L3 custom Ascend C operator for Huawei Ascend 910B3 (`ai_core-ascend910b`, 40 AI Cores). Both `op_kernel/fused_kan_spline_basis_linear_kernel.cpp` and `reference/reference.py` implement the multi-input domain mathematical pipeline verified by `tests/benchmark_fused_kan_spline_basis_linear_aclnn.cpp` (`max_abs_err < 1e-3`).

## 2. Hardware Performance & Peak Memory on Ascend 910B3 vs $\min(\text{Eager}, \text{TorchAir})$

| Workload Shape | PyTorch Eager | PyTorch + TorchAir | $\min(\text{Eager}, \text{TorchAir})$ | **Custom Op `aclnnFusedKanSplineBasisLinear`** | **Speedup vs $\min$** | Peak HBM Memory (Eager / TorchAir $\to$ Custom) | Verified `max_abs_err` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `TimeKAN Standard (32, 512, 64, G=8)` | `3.5994 ms` | `0.2433 ms` | `0.2433 ms` | **`0.0332 ms`** | **`7.33x`** | `244 / 196 -> 16 MB (-91.8%)` | `0.00e+00` |
| `TimeKAN Fine-Grid (32, 512, 64, G=16)` | `5.5983 ms` | `0.2773 ms` | `0.2773 ms` | **`0.0336 ms`** | **`8.25x`** | `412 / 340 -> 16 MB (-95.3%)` | `0.00e+00` |
| `TimeKAN Long-Seq (16, 1024, 128, G=8)` | `7.0262 ms` | `0.3239 ms` | `0.3239 ms` | **`0.0430 ms`** | **`7.53x`** | `488 / 392 -> 32 MB (-91.8%)` | `0.00e+00` |
| `AI4PDE Wide-Channel (16, 512, 128, G=16)` | `5.5687 ms` | `8.6677 ms` | `5.5687 ms` | **`0.0335 ms`** | **`166.23x`** | `768 / 820 -> 16 MB (-97.9%)` | `0.00e+00` |

## 3. Reproducible Verification & Quality Gate
- **End-to-End Hardware Gate**: `bash tests/run_packaged_gate.sh /tmp/work_fused_kan_spline_basis_linear 0` (`=== ALL GATE CHECKS PASSED SUCCESSFULLY ===`)
- **C++ Host Numerical Verification**: Built into `tests/benchmark_fused_kan_spline_basis_linear_aclnn.cpp` and `examples/test_aclnn_fused_kan_spline_basis_linear.cpp`
- **Python Reference Unit Tests**: `python3 -m unittest discover -s tests -p "test_*.py" -v`
