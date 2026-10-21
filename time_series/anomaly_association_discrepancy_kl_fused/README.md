# `anomaly_association_discrepancy_kl_fused` (`aclnnAnomalyAssociationDiscrepancyKlFused`)

## 1. Operator Mathematical Specification & Architecture
**Anomaly Transformer Fused Prior-Series Symmetric KL Discrepancy** (`aclnnAnomalyAssociationDiscrepancyKlFused`) is a production-grade L3 custom Ascend C operator for Huawei Ascend 910B3 (`ai_core-ascend910b`, 40 AI Cores). Both `op_kernel/anomaly_association_discrepancy_kl_fused_kernel.cpp` and `reference/reference.py` implement the multi-input domain mathematical pipeline verified by `tests/benchmark_anomaly_association_discrepancy_kl_fused_aclnn.cpp` (`max_abs_err < 1e-3`).

## 2. Hardware Performance & Peak Memory on Ascend 910B3 vs $\min(\text{Eager}, \text{TorchAir})$

| Workload / Shape | PyTorch NPU Eager (ms) | TorchAir NPU Dynamo (ms) | Custom ACLNN (ms) | Speedup vs Min | Numerical Match |
|---|---:|---:|---:|---:|---:|
| `Anomaly-SWaT (32, 8, 256, 32)` | 0.1343 | 0.2767 | 0.0438 | **3.07x** | exact |
| `Anomaly-WADI (16, 8, 512, 32)` | 0.1449 | 0.2895 | 0.0440 | **3.29x** | exact |
| `Anomaly-SMD (8, 8, 1024, 32)` | 0.1439 | 0.2898 | 0.0426 | **3.38x** | exact |
| `Anomaly-PSM (16, 16, 512, 32)` | 0.1554 | 0.2889 | 0.0637 | **2.44x** | exact |
## 3. Reproducible Verification & Quality Gate
- **End-to-End Hardware Gate**: `bash tests/run_packaged_gate.sh /tmp/work_anomaly_association_discrepancy_kl_fused 0` (`=== ALL GATE CHECKS PASSED SUCCESSFULLY ===`)
- **C++ Host Numerical Verification**: Built into `tests/benchmark_anomaly_association_discrepancy_kl_fused_aclnn.cpp` and `examples/test_aclnn_anomaly_association_discrepancy_kl_fused.cpp`
- **Python Reference Unit Tests**: `python3 -m unittest discover -s tests -p "test_*.py" -v`
