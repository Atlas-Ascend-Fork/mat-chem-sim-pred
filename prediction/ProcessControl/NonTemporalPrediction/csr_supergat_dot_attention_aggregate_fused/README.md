# CsrSuperGatDotAttentionAggregateFused

AscendC fusion of the SuperGAT `attention_type="SD"` message stage: scaled dot
score, LeakyReLU, destination-segment softmax, and weighted neighbor aggregate.
This is not the earlier GAT/GATv2 fusion; SuperGAT-SD uses a scaled projected
feature dot product and has no learned left/right attention vector.

The FP32 Cora x4 model benchmark is exact to `7.45e-08` max absolute error and
runs in 3.395 ms versus the strongest correct eager path at 6.001 ms (1.77x).
See the [benchmark summary](docs/benchmark.md) for the recorded measurements.

Supported shapes are `[N,H,C]` with `H<=8`, `C<=64`, non-empty int32 CSR, and
maximum destination degree 512. The ACLNN forward accepts contiguous FP32, FP16,
and BF16 inputs with FP32 softmax/message accumulation. The supplied
`torch.library` binding supports training through a numerically checked autograd
recomputation; it is not yet a dedicated AscendC backward kernel. The caller
owns output/workspace and supplies the stream; distinct in-flight calls require
distinct output/workspace buffers.

```bash
cmake -S . -B build -DSOC_VERSION=Ascend910B1 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
python -m pytest tests -q
```

The directory contains the AscendC kernel, ACLNN host API, GE op definition,
NumPy oracle, dispatch guards, unit tests, and reproducible model evidence.

The [GE op definition](op_host/csr_supergat_dot_attention_aggregate_fused_def.cpp)
is FP32-only. A TorchAir `torch.compile(..., fullgraph=True)` run on Ascend910B3
recorded `2.38e-7` maximum error, but the probe script is not included in this
repository. Direct ACLNN remains the FP32/FP16/BF16 path.
