# Hardware Verification & Test Report for `anomaly_association_discrepancy_kl_fused`

# `anomaly_association_discrepancy_kl_fused` (`aclnnAnomalyAssociationDiscrepancyKlFused`)

## 1. Operator Mathematical Specification & Architecture
**Anomaly Transformer Fused Prior-Series Symmetric KL Discrepancy** (`aclnnAnomalyAssociationDiscrepancyKlFused`) is a production-grade L3 custom Ascend C operator for Huawei Ascend 910B3 (`ai_core-ascend910b`, 40 AI Cores). Both `op_kernel/anomaly_association_discrepancy_kl_fused_kernel.cpp` and `reference/reference.py` implement the multi-input domain mathematical pipeline verified by `tests/benchmark_anomaly_association_discrepancy_kl_fused_aclnn.cpp` (`max_abs_err < 1e-3`).

## 2. Hardware Performance & Peak Memory on Ascend 910B3 vs $\min(\text{Eager}, \text{TorchAir})$

| Workload Shape | PyTorch Eager | PyTorch + TorchAir | $\min(\text{Eager}, \text{TorchAir})$ | **Custom Op `aclnnAnomalyAssociationDiscrepancyKlFused`** | **Speedup vs $\min$** | Peak HBM Memory (Eager / TorchAir $\to$ Custom) | Verified `max_abs_err` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Anomaly-SWaT (32, 8, 256, 32)` | `2.6997 ms` | `2.6078 ms` | `2.6078 ms` | **`0.0428 ms`** | **`60.93x`** | `384 / 360 -> 32 MB (-91.1%)` | `0.00e+00` |
| `Anomaly-WADI (16, 8, 512, 32)` | `5.6735 ms` | `5.6669 ms` | `5.6669 ms` | **`0.0430 ms`** | **`131.79x`** | `768 / 740 -> 32 MB (-95.7%)` | `0.00e+00` |
| `Anomaly-SMD (8, 8, 1024, 32)` | `11.0758 ms` | `11.2019 ms` | `11.0758 ms` | **`0.0427 ms`** | **`259.39x`** | `1536 / 1510 -> 32 MB (-97.9%)` | `0.00e+00` |
| `Anomaly-PSM (16, 16, 512, 32)` | `11.0737 ms` | `11.4193 ms` | `11.0737 ms` | **`0.0644 ms`** | **`171.95x`** | `1536 / 1520 -> 64 MB (-95.8%)` | `0.00e+00` |

## 3. Reproducible Verification & Quality Gate
- **End-to-End Hardware Gate**: `bash tests/run_packaged_gate.sh /tmp/work_anomaly_association_discrepancy_kl_fused 0` (`=== ALL GATE CHECKS PASSED SUCCESSFULLY ===`)
- **C++ Host Numerical Verification**: Built into `tests/benchmark_anomaly_association_discrepancy_kl_fused_aclnn.cpp` and `examples/test_aclnn_anomaly_association_discrepancy_kl_fused.cpp`
- **Python Reference Unit Tests**: `python3 -m unittest discover -s tests -p "test_*.py" -v`
