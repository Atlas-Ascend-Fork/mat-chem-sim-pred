#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -eo pipefail

OP_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK_ROOT=${1:?usage: run_packaged_gate.sh <new-work-dir> [physical-device]}
DEVICE=${2:-0}
if [ -e "$WORK_ROOT" ]; then
    echo "work directory already exists: $WORK_ROOT" >&2
    exit 17
fi
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
elif [ -f /home/developer/Ascend/ascend-toolkit/set_env.sh ]; then
    source /home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
fi
DRIVER_COMMON="/usr/local/Ascend/driver/lib64/common"
DRIVER_DIR="/usr/local/Ascend/driver/lib64/driver"
export LD_LIBRARY_PATH="/home/developer/Ascend/cann-9.0.0/lib64:$DRIVER_COMMON:$DRIVER_DIR:$LD_LIBRARY_PATH"

mkdir -p "$WORK_ROOT/common"
chmod go-w "$WORK_ROOT" "$WORK_ROOT/common"
cp "$OP_ROOT/msopgen/mlstm_chunkwise_matrix_memory_msopgen.json" "$WORK_ROOT/schema.json"
chmod go-w "$WORK_ROOT/schema.json"

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
    -s "$OP_ROOT/tests" -p 'test_*.py' -v

msopgen gen -i "$WORK_ROOT/schema.json" -f aclnn -c ai_core-ascend910b \
    -out "$WORK_ROOT/project" -lan cpp

PROJECT="$WORK_ROOT/project"
if [ -f "$PROJECT/op_host/mlstm_chunkwise_matrix_memory.cpp" ]; then
    mv "$PROJECT/op_host/mlstm_chunkwise_matrix_memory.cpp" \
        "$PROJECT/op_host/mlstm_chunkwise_matrix_memory.generated.cpp.stub"
fi

cp "$OP_ROOT/op_host/mlstm_chunkwise_matrix_memory_host.cpp" "$PROJECT/op_host/"
cp "$OP_ROOT/op_host/mlstm_chunkwise_matrix_memory_def.cpp" "$PROJECT/op_host/"
cp "$OP_ROOT/op_host/mlstm_chunkwise_matrix_memory_tiling.h" "$PROJECT/op_host/"
cp "$OP_ROOT/op_host/mlstm_chunkwise_matrix_memory_host.h" "$PROJECT/op_host/"
cp "$OP_ROOT/op_kernel/mlstm_chunkwise_matrix_memory_kernel.cpp" \
    "$PROJECT/op_kernel/mlstm_chunkwise_matrix_memory.cpp"

if [ -f "$OP_ROOT/op_host/op_def_utils.h" ]; then
    cp "$OP_ROOT/op_host/op_def_utils.h" "$PROJECT/op_host/"
    cp "$OP_ROOT/op_host/op_def_utils.h" "$WORK_ROOT/common/"
fi
if [ -f "$OP_ROOT/../common/op_def_utils.h" ]; then
    cp "$OP_ROOT/../common/op_def_utils.h" "$WORK_ROOT/common/"
fi
if [ -f "$OP_ROOT/op_host/op_host_utils.h" ]; then
    cp "$OP_ROOT/op_host/op_host_utils.h" "$PROJECT/op_host/"
    cp "$OP_ROOT/op_host/op_host_utils.h" "$WORK_ROOT/common/"
fi
if [ -f "$OP_ROOT/../../common/op_host_utils.h" ]; then
    cp "$OP_ROOT/../../common/op_host_utils.h" "$WORK_ROOT/common/"
fi
if [ -f "$OP_ROOT/../common/op_host_utils.h" ]; then
    cp "$OP_ROOT/../common/op_host_utils.h" "$WORK_ROOT/common/"
fi

cmake -S "$PROJECT" -B "$PROJECT/build_formal" \
    -DASCEND_COMPUTE_UNIT=ascend910b -DCMAKE_BUILD_TYPE=Release
cmake --build "$PROJECT/build_formal" --target binary -- -j2
cmake --build "$PROJECT/build_formal" --target install -- -j2

VENDOR_DIR="$PROJECT/build_formal/packages/vendors/customize"
if [ -d "$VENDOR_DIR/op_impl/ai_core/tbe/kernel" ]; then
    ln -sf ascend910b "$VENDOR_DIR/op_impl/ai_core/tbe/kernel/ascend910_93" 2>/dev/null || true
fi
if [ -d "$VENDOR_DIR/op_impl/ai_core/tbe/kernel/config" ]; then
    ln -sf ascend910b "$VENDOR_DIR/op_impl/ai_core/tbe/kernel/config/ascend910_93" 2>/dev/null || true
fi

TOOLKIT_ROOT="/usr/local/Ascend/ascend-toolkit/latest"
if [ ! -d "$TOOLKIT_ROOT" ] && [ -d "/home/developer/Ascend/ascend-toolkit/latest" ]; then
    TOOLKIT_ROOT="/home/developer/Ascend/ascend-toolkit/latest"
fi
OP_API_DIR="$VENDOR_DIR/op_api"
DRIVER_PATHS="-L/usr/local/Ascend/driver/lib64/driver -L/usr/local/Ascend/driver/lib64/common \
    -Wl,-rpath,/usr/local/Ascend/driver/lib64/driver -Wl,-rpath,/usr/local/Ascend/driver/lib64/common"

g++ -std=c++17 -O2 "$OP_ROOT/examples/test_aclnn_mlstm_chunkwise_matrix_memory.cpp" \
    -I"$OP_API_DIR/include" -I"$TOOLKIT_ROOT/include" \
    -L"$OP_API_DIR/lib" -L"$TOOLKIT_ROOT/lib64" $DRIVER_PATHS \
    -Wl,-rpath,"$OP_API_DIR/lib" -Wl,-rpath,"$TOOLKIT_ROOT/lib64" \
    -Wl,--copy-dt-needed-entries -lcust_opapi -lnnopbase -lascendcl -ldl \
    -o "$WORK_ROOT/test_aclnn_mlstm_chunkwise_matrix_memory"

g++ -std=c++17 -O2 "$OP_ROOT/tests/benchmark_mlstm_chunkwise_matrix_memory_aclnn.cpp" \
    -I"$OP_API_DIR/include" -I"$TOOLKIT_ROOT/include" \
    -L"$OP_API_DIR/lib" -L"$TOOLKIT_ROOT/lib64" $DRIVER_PATHS \
    -Wl,-rpath,"$OP_API_DIR/lib" -Wl,-rpath,"$TOOLKIT_ROOT/lib64" \
    -Wl,--copy-dt-needed-entries -lcust_opapi -lnnopbase -lascendcl -ldl \
    -o "$WORK_ROOT/benchmark_mlstm_chunkwise_matrix_memory_aclnn"

OP_TILING_LIB="$VENDOR_DIR/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"
OP_PROTO_LIB="$VENDOR_DIR/op_proto/lib/linux/aarch64"
RUN_LD_PATH="$OP_API_DIR/lib:$OP_TILING_LIB:$OP_PROTO_LIB:$PROJECT/build_formal/autogen:${LD_LIBRARY_PATH:-}"

export ASCEND_CUSTOM_OPP_PATH="$VENDOR_DIR"
ASCEND_RT_VISIBLE_DEVICES="$DEVICE" LD_LIBRARY_PATH="$RUN_LD_PATH" \
    "$WORK_ROOT/test_aclnn_mlstm_chunkwise_matrix_memory" 0

ASCEND_RT_VISIBLE_DEVICES="$DEVICE" LD_LIBRARY_PATH="$RUN_LD_PATH" \
    "$WORK_ROOT/benchmark_mlstm_chunkwise_matrix_memory_aclnn" 0 > "$WORK_ROOT/bench_result.json"

cat "$WORK_ROOT/bench_result.json"
echo "=== ALL GATE CHECKS PASSED SUCCESSFULLY ==="
