# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def _validate_types_and_dims(tensors, chunk_size):
    u_arr, dt_arr, a_arr, b_arr, c_arr = tensors
    for label, tensor in [
        ("u", u_arr),
        ("dt", dt_arr),
        ("a", a_arr),
        ("b", b_arr),
        ("c", c_arr),
    ]:
        if not isinstance(tensor, np.ndarray):
            raise TypeError(f"Input {label} must be a numpy ndarray")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    if (u_arr.ndim, dt_arr.ndim, a_arr.ndim, b_arr.ndim, c_arr.ndim) != (4, 3, 1, 4, 4):
        raise ValueError("Invalid input dimensions for Mamba-2 SSD")


def _validate_mamba2_shapes(tensors, d_skip):
    u_arr, dt_arr, a_arr, b_arr, c_arr = tensors
    batch_size, seq_len, num_heads, head_dim = u_arr.shape
    state_dim = b_arr.shape[-1]
    expected_shapes = (
        (batch_size, seq_len, num_heads),
        (num_heads,),
        (batch_size, seq_len, num_heads, state_dim),
        (batch_size, seq_len, num_heads, state_dim),
    )
    if (dt_arr.shape, a_arr.shape, b_arr.shape, c_arr.shape) != expected_shapes:
        raise ValueError("Mismatched tensor shapes in Mamba-2 SSD")
    if d_skip is not None and d_skip.shape != (num_heads, head_dim):
        raise ValueError(
            f"Expected d shape ({num_heads}, {head_dim}), got {d_skip.shape}"
        )
    return batch_size, seq_len, num_heads, head_dim, state_dim


def _step_scan(state, *step_tensors):
    u_step, dt_step, alpha_step, b_step, c_step, d_skip = step_tensors
    bu_outer = b_step[:, :, :, np.newaxis] * u_step[:, :, np.newaxis, :]
    new_state = alpha_step * state + dt_step * bu_outer
    y_step = np.sum(c_step[:, :, :, np.newaxis] * new_state, axis=2)
    if d_skip is not None:
        y_step += d_skip[np.newaxis, :, :] * u_step
    return new_state, y_step


def reference(*args, **kwargs):
    """Pure NumPy CPU Reference Implementation for Mamba-2 State Space Duality (SSD)."""
    names = ("u", "dt", "a", "b", "c", "d", "chunk_size")
    bound = dict(zip(names, args))
    bound.update(kwargs)
    u_arr, dt_arr, a_arr = bound["u"], bound["dt"], bound["a"]
    b_arr, c_arr = bound["b"], bound["c"]
    d_skip = bound.get("d", None)
    chunk_size = bound.get("chunk_size", 64)

    tensors = (u_arr, dt_arr, a_arr, b_arr, c_arr)
    _validate_types_and_dims(tensors, chunk_size)
    batch_size, seq_len, num_heads, head_dim, state_dim = _validate_mamba2_shapes(
        tensors, d_skip
    )

    out = np.zeros_like(u_arr, dtype=np.float32)
    state = np.zeros((batch_size, num_heads, state_dim, head_dim), dtype=np.float32)

    for step_idx in range(seq_len):
        dt_step = dt_arr[:, step_idx, :, np.newaxis, np.newaxis]
        alpha = np.exp(dt_arr[:, step_idx, :] * a_arr[np.newaxis, :])[
            :, :, np.newaxis, np.newaxis
        ]
        state, y_step = _step_scan(
            state,
            u_arr[:, step_idx, :, :],
            dt_step,
            alpha,
            b_arr[:, step_idx, :, :],
            c_arr[:, step_idx, :, :],
            d_skip,
        )
        out[:, step_idx, :, :] = y_step

    return out
