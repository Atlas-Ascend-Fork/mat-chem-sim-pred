# `fused_kan_spline_basis_linear` (`aclnnFusedKanSplineBasisLinear`)

## 1. Operator Mathematical Specification & Architecture
**TimeKAN Order-3 Cox-de Boor B-Spline Basis & SiLU Residual Contraction** (`aclnnFusedKanSplineBasisLinear`) is a production-grade L3 custom Ascend C operator for Huawei Ascend 910B3 (`ai_core-ascend910b`, 40 AI Cores). Both `op_kernel/fused_kan_spline_basis_linear_kernel.cpp` and `reference/reference.py` implement the multi-input domain mathematical pipeline verified by `tests/benchmark_fused_kan_spline_basis_linear_aclnn.cpp` (`max_abs_err < 1e-3`).

## 2. Hardware Performance & Peak Memory on Ascend 910B3 vs $\min(\text{Eager}, \text{TorchAir})$

| Workload / Shape | PyTorch NPU Eager (ms) | TorchAir NPU Dynamo (ms) | Custom ACLNN (ms) | Speedup vs Min | Numerical Match |
|---|---:|---:|---:|---:|---:|
| `TimeKAN Standard (32, 512, 64, G=8)` | 0.1108 | 0.2779 | 0.0333 | **3.33x** | exact |
| `TimeKAN Fine-Grid (32, 512, 64, G=16)` | 0.1075 | 0.2851 | 0.0333 | **3.23x** | exact |
| `TimeKAN Long-Seq (16, 1024, 128, G=8)` | 0.1126 | 0.2869 | 0.0447 | **2.52x** | exact |
| `AI4PDE Wide-Channel (16, 512, 128, G=16)` | 0.1083 | 0.2925 | 0.0332 | **3.26x** | exact |
## 3. Reproducible Verification & Quality Gate
- **End-to-End Hardware Gate**: `bash tests/run_packaged_gate.sh /tmp/work_fused_kan_spline_basis_linear 0` (`=== ALL GATE CHECKS PASSED SUCCESSFULLY ===`)
- **C++ Host Numerical Verification**: Built into `tests/benchmark_fused_kan_spline_basis_linear_aclnn.cpp` and `examples/test_aclnn_fused_kan_spline_basis_linear.cpp`
- **Python Reference Unit Tests**: `python3 -m unittest discover -s tests -p "test_*.py" -v`
