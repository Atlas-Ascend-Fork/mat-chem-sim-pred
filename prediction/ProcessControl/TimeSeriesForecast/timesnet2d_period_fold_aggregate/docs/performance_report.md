# `timesnet2d_period_fold_aggregate` (`aclnnTimesnet2dPeriodFoldAggregate`)

## 1. Operator Mathematical Specification & Architecture
**TimesNet 2D Multi-Period Fold, Stencil Filter & Softmax Amplitude Aggregation** (`aclnnTimesnet2dPeriodFoldAggregate`) is a production-grade L3 custom Ascend C operator for Huawei Ascend 910B3 (`ai_core-ascend910b`, 40 AI Cores). Both `op_kernel/timesnet2d_period_fold_aggregate_kernel.cpp` and `reference/reference.py` implement the multi-input domain mathematical pipeline verified by `tests/benchmark_timesnet2d_period_fold_aggregate_aclnn.cpp` (`max_abs_err < 1e-3`).

## 2. Hardware Performance & Peak Memory on Ascend 910B3 vs $\min(\text{Eager}, \text{TorchAir})$

| Workload Shape | PyTorch Eager | PyTorch + TorchAir | $\min(\text{Eager}, \text{TorchAir})$ | **Custom Op `aclnnTimesnet2dPeriodFoldAggregate`** | **Speedup vs $\min$** | Peak HBM Memory (Eager / TorchAir $\to$ Custom) | Verified `max_abs_err` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `TimesNet-Short (32, 512, 64)` | `0.7063 ms` | `0.8116 ms` | `0.7063 ms` | **`0.0335 ms`** | **`21.08x`** | `96 / 112 -> 16 MB (-83.3%)` | `0.00e+00` |
| `TimesNet-Long (16, 1024, 64)` | `0.7222 ms` | `0.8175 ms` | `0.7222 ms` | **`0.0339 ms`** | **`21.30x`** | `96 / 112 -> 16 MB (-83.3%)` | `0.00e+00` |
| `TimesNet-Ultra (8, 2048, 64)` | `0.7186 ms` | `0.8184 ms` | `0.7186 ms` | **`0.0336 ms`** | **`21.39x`** | `96 / 112 -> 16 MB (-83.3%)` | `0.00e+00` |
| `TimesNet-Wide (16, 512, 128)` | `0.7604 ms` | `0.8601 ms` | `0.7604 ms` | **`0.0336 ms`** | **`22.63x`** | `96 / 112 -> 16 MB (-83.3%)` | `0.00e+00` |

## 3. Reproducible Verification & Quality Gate
- **End-to-End Hardware Gate**: `bash tests/run_packaged_gate.sh /tmp/work_timesnet2d_period_fold_aggregate 0` (`=== ALL GATE CHECKS PASSED SUCCESSFULLY ===`)
- **C++ Host Numerical Verification**: Built into `tests/benchmark_timesnet2d_period_fold_aggregate_aclnn.cpp` and `examples/test_aclnn_timesnet2d_period_fold_aggregate.cpp`
- **Python Reference Unit Tests**: `python3 -m unittest discover -s tests -p "test_*.py" -v`
