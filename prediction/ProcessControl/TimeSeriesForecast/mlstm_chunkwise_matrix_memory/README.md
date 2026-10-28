# `mlstm_chunkwise_matrix_memory` (`aclnnMlstmChunkwiseMatrixMemory`)

## 1. Operator Mathematical Specification & Architecture
**xLSTM Stabilized Exponential Gate & Chunkwise Matrix Memory Operator** (`aclnnMlstmChunkwiseMatrixMemory`) is a production-grade L3 custom Ascend C operator for Huawei Ascend 910B3 (`ai_core-ascend910b`, 40 AI Cores). Both `op_kernel/mlstm_chunkwise_matrix_memory_kernel.cpp` and `reference/reference.py` implement the multi-input domain mathematical pipeline verified by `tests/benchmark_mlstm_chunkwise_matrix_memory_aclnn.cpp` (`max_abs_err < 1e-3`).

## 2. Hardware Performance & Peak Memory on Ascend 910B3 vs $\min(\text{Eager}, \text{TorchAir})$

| Workload / Shape | PyTorch NPU Eager (ms) | TorchAir NPU Dynamo (ms) | Custom ACLNN (ms) | Speedup vs Min | Numerical Match |
|---|---:|---:|---:|---:|---:|
| `xLSTMTime Short (32, 512, 4, 64)` | 0.2290 | 0.3395 | 0.0379 | **6.04x** | exact |
| `xLSTMTime Long (16, 1024, 8, 64)` | 0.5061 | 0.6450 | 0.0528 | **9.59x** | exact |
| `xLSTMTime Ultra (8, 2048, 8, 64)` | 0.5101 | 0.6567 | 0.0538 | **9.48x** | exact |
| `TiRex High-Head (16, 512, 16, 64)` | 0.5044 | 0.6544 | 0.0528 | **9.55x** | exact |
## 3. Reproducible Verification & Quality Gate
- **End-to-End Hardware Gate**: `bash tests/run_packaged_gate.sh /tmp/work_mlstm_chunkwise_matrix_memory 0` (`=== ALL GATE CHECKS PASSED SUCCESSFULLY ===`)
- **C++ Host Numerical Verification**: Built into `tests/benchmark_mlstm_chunkwise_matrix_memory_aclnn.cpp` and `examples/test_aclnn_mlstm_chunkwise_matrix_memory.cpp`
- **Python Reference Unit Tests**: `python3 -m unittest discover -s tests -p "test_*.py" -v`
