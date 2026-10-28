# `deepmd_radial_env_descriptor_fused` (`aclnnDeepmdRadialEnvDescriptorFused`)

## 1. Operator Mathematical Specification & Architecture
**DeepMD-kit SeA Smooth Cosine Cutoff & Radial-Angular Environment Descriptor** (`aclnnDeepmdRadialEnvDescriptorFused`) is a production-grade L3 custom Ascend C operator for Huawei Ascend 910B3 (`ai_core-ascend910b`, 40 AI Cores). Both `op_kernel/deepmd_radial_env_descriptor_fused_kernel.cpp` and `reference/reference.py` implement the multi-input domain mathematical pipeline verified by `tests/benchmark_deepmd_radial_env_descriptor_fused_aclnn.cpp` (`max_abs_err < 1e-3`).

## 2. Hardware Performance & Peak Memory on Ascend 910B3 vs $\min(\text{Eager}, \text{TorchAir})$

| Workload / Shape | PyTorch NPU Eager (ms) | TorchAir NPU Dynamo (ms) | Custom ACLNN (ms) | Speedup vs Min | Numerical Match |
|---|---:|---:|---:|---:|---:|
| `DeepMD-CuBulk (1024, 64, 32)` | 0.1069 | 0.2799 | 0.0242 | **4.42x** | exact |
| `DeepMD-H2O-64 (2048, 96, 32)` | 0.1765 | 0.2964 | 0.0315 | **5.60x** | exact |
| `DeepMD-HEA-Alloy (4096, 128, 32)` | 0.7717 | 0.9026 | 0.0429 | **17.99x** | exact |
| `DeepMD-ProteinSolv (8192, 128, 64)` | 3.0591 | 3.1539 | 0.0627 | **48.79x** | exact |
## 3. Reproducible Verification & Quality Gate
- **End-to-End Hardware Gate**: `bash tests/run_packaged_gate.sh /tmp/work_deepmd_radial_env_descriptor_fused 0` (`=== ALL GATE CHECKS PASSED SUCCESSFULLY ===`)
- **C++ Host Numerical Verification**: Built into `tests/benchmark_deepmd_radial_env_descriptor_fused_aclnn.cpp` and `examples/test_aclnn_deepmd_radial_env_descriptor_fused.cpp`
- **Python Reference Unit Tests**: `python3 -m unittest discover -s tests -p "test_*.py" -v`
