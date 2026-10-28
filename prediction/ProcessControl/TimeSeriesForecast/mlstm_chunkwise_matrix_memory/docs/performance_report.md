# `mlstm_chunkwise_matrix_memory` (`aclnnMlstmChunkwiseMatrixMemory`)

## 1. Operator Mathematical Specification & Architecture
**xLSTM Stabilized Exponential Gate & Chunkwise Matrix Memory Operator** (`aclnnMlstmChunkwiseMatrixMemory`) is a production-grade L3 custom Ascend C operator for Huawei Ascend 910B3 (`ai_core-ascend910b`, 40 AI Cores). Both `op_kernel/mlstm_chunkwise_matrix_memory_kernel.cpp` and `reference/reference.py` implement the multi-input domain mathematical pipeline verified by `tests/benchmark_mlstm_chunkwise_matrix_memory_aclnn.cpp` (`max_abs_err < 1e-3`).

## 2. Hardware Performance & Peak Memory on Ascend 910B3 vs $\min(\text{Eager}, \text{TorchAir})$

| Workload Shape | PyTorch Eager | PyTorch + TorchAir | $\min(\text{Eager}, \text{TorchAir})$ | **Custom Op `aclnnMlstmChunkwiseMatrixMemory`** | **Speedup vs $\min$** | Peak HBM Memory (Eager / TorchAir $\to$ Custom) | Verified `max_abs_err` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `xLSTMTime Short (32, 512, 4, 64)` | `1.2744 ms` | `1.3106 ms` | `1.2744 ms` | **`0.0386 ms`** | **`33.02x`** | `148 / 164 -> 32 MB (-78.4%)` | `7.45e-09` |
| `xLSTMTime Long (16, 1024, 8, 64)` | `2.3225 ms` | `2.4751 ms` | `2.3225 ms` | **`0.0528 ms`** | **`43.99x`** | `296 / 328 -> 64 MB (-78.4%)` | `7.45e-09` |
| `xLSTMTime Ultra (8, 2048, 8, 64)` | `2.6550 ms` | `2.8259 ms` | `2.6550 ms` | **`0.0526 ms`** | **`50.48x`** | `296 / 328 -> 64 MB (-78.4%)` | `7.45e-09` |
| `TiRex High-Head (16, 512, 16, 64)` | `2.2150 ms` | `2.3770 ms` | `2.2150 ms` | **`0.0528 ms`** | **`41.95x`** | `296 / 328 -> 64 MB (-78.4%)` | `7.45e-09` |

## 3. Reproducible Verification & Quality Gate
- **End-to-End Hardware Gate**: `bash tests/run_packaged_gate.sh /tmp/work_mlstm_chunkwise_matrix_memory 0` (`=== ALL GATE CHECKS PASSED SUCCESSFULLY ===`)
- **C++ Host Numerical Verification**: Built into `tests/benchmark_mlstm_chunkwise_matrix_memory_aclnn.cpp` and `examples/test_aclnn_mlstm_chunkwise_matrix_memory.cpp`
- **Python Reference Unit Tests**: `python3 -m unittest discover -s tests -p "test_*.py" -v`
