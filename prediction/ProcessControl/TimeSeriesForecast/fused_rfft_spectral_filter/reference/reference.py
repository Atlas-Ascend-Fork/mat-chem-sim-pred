# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def reference(x, weights):
    """
    Pure NumPy CPU Reference Implementation for Fused RFFT Spectral Filter.

    Args:
        x (np.ndarray): Real input time series tensor of shape [B, L, D], dtype float32
        weights (np.ndarray): Complex spectral weight tensor stored in interleaved
                              format [L//2 + 1, D, 2], where [..., 0] is real part
                              and [..., 1] is imaginary part.

    Returns:
        np.ndarray: Filtered real time series tensor of shape [B, L, D], dtype float32
    """
    if not isinstance(x, np.ndarray) or not isinstance(weights, np.ndarray):
        raise TypeError("Inputs x and weights must be numpy ndarrays.")
    if x.ndim != 3 or weights.ndim != 3:
        raise ValueError(f"Expected 3D inputs, got x.ndim={x.ndim}, weights.ndim={weights.ndim}")

    batch_size, seq_len, d_model = x.shape
    freq_len = seq_len // 2 + 1

    if weights.shape[0] != freq_len or weights.shape[1] != d_model or weights.shape[2] != 2:
        raise ValueError(
            f"Expected weights shape ({freq_len}, {d_model}, 2), got {weights.shape}"
        )

    # 1. 1D Real Fast Fourier Transform along sequence dimension L
    x_fft = np.fft.rfft(x, axis=1)  # shape: [batch_size, freq_len, d_model], complex64

    # 2. Reconstruct complex weights
    w_complex = weights[..., 0] + 1j * weights[..., 1]  # shape: [freq_len, d_model]

    # 3. Spectral multiplication with broadcasting along batch dimension
    out_fft = x_fft * w_complex[np.newaxis, :, :]  # shape: [batch_size, freq_len, d_model]

    # 4. Inverse Real Fast Fourier Transform
    out = np.fft.irfft(out_fft, n=seq_len, axis=1)  # shape: [batch_size, seq_len, d_model]

    return out.astype(np.float32)
