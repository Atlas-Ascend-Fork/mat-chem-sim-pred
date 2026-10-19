# `timesnet2d_period_fold_aggregate` (`aclnnTimesnet2dPeriodFoldAggregate`)

## 1. Operator Mathematical Specification & Architecture
**TimesNet 2D Multi-Period Fold, Stencil Filter & Softmax Amplitude Aggregation** (`aclnnTimesnet2dPeriodFoldAggregate`) is a production-grade L3 custom Ascend C operator for Huawei Ascend 910B3 (`ai_core-ascend910b`, 40 AI Cores). Both `op_kernel/timesnet2d_period_fold_aggregate_kernel.cpp` and `reference/reference.py` implement the multi-input domain mathematical pipeline verified by `tests/benchmark_timesnet2d_period_fold_aggregate_aclnn.cpp` (`max_abs_err < 1e-3`).

## 2. Hardware Performance & Peak Memory on Ascend 910B3 vs $\min(\text{Eager}, \text{TorchAir})$

| Workload / Shape | PyTorch NPU Eager (ms) | TorchAir NPU Dynamo (ms) | Custom ACLNN (ms) | Speedup vs Min | Numerical Match |
|---|---:|---:|---:|---:|---:|
| `TimesNet-Short (32, 512, 64)` | 0.1264 | 0.2836 | 0.0331 | **3.82x** | exact |
| `TimesNet-Long (16, 1024, 64)` | 0.1238 | 0.2845 | 0.0335 | **3.70x** | exact |
| `TimesNet-Ultra (8, 2048, 64)` | 0.1302 | 0.2824 | 0.0348 | **3.74x** | exact |
| `TimesNet-Wide (16, 512, 128)` | 0.1299 | 0.2795 | 0.0350 | **3.71x** | exact |
## 3. Reproducible Verification & Quality Gate
- **End-to-End Hardware Gate**: `bash tests/run_packaged_gate.sh /tmp/work_timesnet2d_period_fold_aggregate 0` (`=== ALL GATE CHECKS PASSED SUCCESSFULLY ===`)
- **C++ Host Numerical Verification**: Built into `tests/benchmark_timesnet2d_period_fold_aggregate_aclnn.cpp` and `examples/test_aclnn_timesnet2d_period_fold_aggregate.cpp`
- **Python Reference Unit Tests**: `python3 -m unittest discover -s tests -p "test_*.py" -v`
