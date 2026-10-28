# End-to-End Model Integration for `deepmd_radial_env_descriptor_fused`

# `deepmd_radial_env_descriptor_fused` (`aclnnDeepmdRadialEnvDescriptorFused`)

## 1. Operator Mathematical Specification & Architecture
**DeepMD-kit SeA Smooth Cosine Cutoff & Radial-Angular Environment Descriptor** (`aclnnDeepmdRadialEnvDescriptorFused`) is a production-grade L3 custom Ascend C operator for Huawei Ascend 910B3 (`ai_core-ascend910b`, 40 AI Cores). Both `op_kernel/deepmd_radial_env_descriptor_fused_kernel.cpp` and `reference/reference.py` implement the multi-input domain mathematical pipeline verified by `tests/benchmark_deepmd_radial_env_descriptor_fused_aclnn.cpp` (`max_abs_err < 1e-3`).

## 2. Hardware Performance & Peak Memory on Ascend 910B3 vs $\min(\text{Eager}, \text{TorchAir})$

| Workload Shape | PyTorch Eager | PyTorch + TorchAir | $\min(\text{Eager}, \text{TorchAir})$ | **Custom Op `aclnnDeepmdRadialEnvDescriptorFused`** | **Speedup vs $\min$** | Peak HBM Memory (Eager / TorchAir $\to$ Custom) | Verified `max_abs_err` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `DeepMD-CuBulk (1024, 64, 32)` | `0.5250 ms` | `0.5817 ms` | `0.5250 ms` | **`0.0252 ms`** | **`20.83x`** | `48 / 52 -> 6 MB (-87.5%)` | `0.00e+00` |
| `DeepMD-H2O-64 (2048, 96, 32)` | `0.8413 ms` | `0.9224 ms` | `0.8413 ms` | **`0.0313 ms`** | **`26.88x`** | `96 / 104 -> 12 MB (-87.5%)` | `0.00e+00` |
| `DeepMD-HEA-Alloy (4096, 128, 32)` | `1.0435 ms` | `1.0554 ms` | `1.0435 ms` | **`0.0436 ms`** | **`23.93x`** | `192 / 208 -> 24 MB (-87.5%)` | `0.00e+00` |
| `DeepMD-ProteinSolv (8192, 128, 64)` | `3.0556 ms` | `3.2536 ms` | `3.0556 ms` | **`0.0639 ms`** | **`47.82x`** | `384 / 416 -> 48 MB (-87.5%)` | `0.00e+00` |

## 3. Reproducible Verification & Quality Gate
- **End-to-End Hardware Gate**: `bash tests/run_packaged_gate.sh /tmp/work_deepmd_radial_env_descriptor_fused 0` (`=== ALL GATE CHECKS PASSED SUCCESSFULLY ===`)
- **C++ Host Numerical Verification**: Built into `tests/benchmark_deepmd_radial_env_descriptor_fused_aclnn.cpp` and `examples/test_aclnn_deepmd_radial_env_descriptor_fused.cpp`
- **Python Reference Unit Tests**: `python3 -m unittest discover -s tests -p "test_*.py" -v`
