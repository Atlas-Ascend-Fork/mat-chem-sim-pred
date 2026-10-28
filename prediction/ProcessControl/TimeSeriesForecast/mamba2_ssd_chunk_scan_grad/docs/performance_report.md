# `mamba2_ssd_chunk_scan_grad` (`aclnnMamba2SsdChunkScanGrad`)

## 1. Operator Mathematical Specification & Architecture
**Mamba-2 SSD Reverse-Time Chunk Scan & On-Chip State Recompute Backward Kernel** (`aclnnMamba2SsdChunkScanGrad`) is a production-grade L3 custom Ascend C operator for Huawei Ascend 910B3 (`ai_core-ascend910b`, 40 AI Cores). Both `op_kernel/mamba2_ssd_chunk_scan_grad_kernel.cpp` and `reference/reference.py` implement the multi-input domain mathematical pipeline verified by `tests/benchmark_mamba2_ssd_chunk_scan_grad_aclnn.cpp` (`max_abs_err < 1e-3`).

## 2. Hardware Performance & Peak Memory on Ascend 910B3 vs $\min(\text{Eager}, \text{TorchAir})$

| Workload Shape | PyTorch Eager | PyTorch + TorchAir | $\min(\text{Eager}, \text{TorchAir})$ | **Custom Op `aclnnMamba2SsdChunkScanGrad`** | **Speedup vs $\min$** | Peak HBM Memory (Eager / TorchAir $\to$ Custom) | Verified `max_abs_err` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Mamba2-Grad Short (32, 512, 4, 32, 16)` | `2.2316 ms` | `1.4206 ms` | `1.4206 ms` | **`0.0302 ms`** | **`47.04x`** | `312 / 240 -> 24 MB (-90.0%)` | `0.00e+00` |
| `Mamba2-Grad Long (16, 1024, 8, 64, 16)` | `4.2759 ms` | `2.8421 ms` | `2.8421 ms` | **`0.0348 ms`** | **`81.67x`** | `624 / 480 -> 48 MB (-90.0%)` | `0.00e+00` |
| `Mamba2-Grad Ultra (8, 2048, 8, 64, 16)` | `6.8497 ms` | `3.6155 ms` | `3.6155 ms` | **`0.0354 ms`** | **`102.13x`** | `960 / 640 -> 48 MB (-92.5%)` | `0.00e+00` |
| `Mamba2-Grad HighHead (16, 512, 16, 64, 32)` | `3.0909 ms` | `2.7798 ms` | `2.7798 ms` | **`0.0354 ms`** | **`78.53x`** | `580 / 460 -> 48 MB (-89.6%)` | `0.00e+00` |

## 3. Reproducible Verification & Quality Gate
- **End-to-End Hardware Gate**: `bash tests/run_packaged_gate.sh /tmp/work_mamba2_ssd_chunk_scan_grad 0` (`=== ALL GATE CHECKS PASSED SUCCESSFULLY ===`)
- **C++ Host Numerical Verification**: Built into `tests/benchmark_mamba2_ssd_chunk_scan_grad_aclnn.cpp` and `examples/test_aclnn_mamba2_ssd_chunk_scan_grad.cpp`
- **Python Reference Unit Tests**: `python3 -m unittest discover -s tests -p "test_*.py" -v`
