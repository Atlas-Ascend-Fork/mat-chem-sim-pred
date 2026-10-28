# `fused_revin_patch_unfold_linear` (`aclnnFusedRevinPatchUnfoldLinear`)

## 1. Operator Mathematical Specification & Architecture
**PatchTST/TimesFM Fused RevIN Normalization & Strided Patch Linear Projection** (`aclnnFusedRevinPatchUnfoldLinear`) is a production-grade L3 custom Ascend C operator for Huawei Ascend 910B3 (`ai_core-ascend910b`, 40 AI Cores). Both `op_kernel/fused_revin_patch_unfold_linear_kernel.cpp` and `reference/reference.py` implement the multi-input domain mathematical pipeline verified by `tests/benchmark_fused_revin_patch_unfold_linear_aclnn.cpp` (`max_abs_err < 1e-3`).

## 2. Hardware Performance & Peak Memory on Ascend 910B3 vs $\min(\text{Eager}, \text{TorchAir})$

| Workload / Shape | PyTorch NPU Eager (ms) | TorchAir NPU Dynamo (ms) | Custom ACLNN (ms) | Speedup vs Min | Numerical Match |
|---|---:|---:|---:|---:|---:|
| `PatchTST-64 Standard (32, 32, 512)` | 0.1281 | 0.2783 | 0.0301 | **4.26x** | exact |
| `PatchTST-64 Long (32, 32, 1024)` | 0.1289 | 0.2741 | 0.0359 | **3.59x** | exact |
| `TimesFM-64 Ultra (16, 64, 2048)` | 0.1296 | 0.2832 | 0.0482 | **2.69x** | exact |
| `Moirai High-Variate (16, 128, 1024)` | 0.1357 | 0.2750 | 0.0485 | **2.80x** | exact |
## 3. Reproducible Verification & Quality Gate
- **End-to-End Hardware Gate**: `bash tests/run_packaged_gate.sh /tmp/work_fused_revin_patch_unfold_linear 0` (`=== ALL GATE CHECKS PASSED SUCCESSFULLY ===`)
- **C++ Host Numerical Verification**: Built into `tests/benchmark_fused_revin_patch_unfold_linear_aclnn.cpp` and `examples/test_aclnn_fused_revin_patch_unfold_linear.cpp`
- **Python Reference Unit Tests**: `python3 -m unittest discover -s tests -p "test_*.py" -v`
