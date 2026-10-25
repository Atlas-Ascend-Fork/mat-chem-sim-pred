#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# PyPTO / Python 后端的打包：执行 op.yaml 的 package 命令，收集 wheel/sdist。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# 复用 op.yaml 读取函数（op_field / op_backend）。
# shellcheck source=scripts/lib/ops.sh
source "$REPO_ROOT/scripts/lib/ops.sh"

OUT_DIR="build_out/pypto"
OPS=()

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[package-pypto] $*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --out=*) OUT_DIR="${1#--out=}" ;;
        --out)   shift; OUT_DIR="${1:-}" ;;
        -h|--help)
            echo "Usage: bash scripts/package_pypto.sh [--out=<dir>] <op>..."
            exit 0
            ;;
        -*) die "unknown option: $1" ;;
        *)  OPS+=("$1") ;;
    esac
    shift
done

[ ${#OPS[@]} -gt 0 ] || die "at least one operator path is required"
mkdir -p "$OUT_DIR"
RECORDS="$(mktemp)"
trap 'rm -f "$RECORDS"' EXIT

packed=0
for op in "${OPS[@]}"; do
    op="${op%/}"
    [ -f "$op/op.yaml" ] || die "$op — missing op.yaml"
    [ "$(op_backend "$op")" = pypto ] || die "$op — not a pypto operator"

    cmd="$(op_field "$op" package 2>/dev/null || true)"
    cmd="${cmd:-${PYTHON_BIN:-python3} -m build --wheel}"
    log "package $op: $cmd"
    ( cd "$op" && rm -rf dist && bash -c "$cmd" )

    found=false
    for artifact in "$op"/dist/*.whl "$op"/dist/*.tar.gz; do
        [ -f "$artifact" ] || continue
        dest="$OUT_DIR/$(basename "$op")_$(basename "$artifact")"
        cp "$artifact" "$dest"
        printf '%s\t%s\n' "$op" "$(basename "$dest")" >> "$RECORDS"
        log "packed $dest"
        found=true
        packed=$((packed + 1))
    done
    [ "$found" = true ] || die "$op — package command produced no wheel/sdist under dist/"
done

python3 - "$OUT_DIR/manifest_pypto.json" "$RECORDS" <<'PY'
import json
import sys
from pathlib import Path

out, records = sys.argv[1:3]
items = []
for line in Path(records).read_text().splitlines():
    if line.strip():
        op, package = line.split("\t", 1)
        items.append({"op": op, "package": package})
Path(out).write_text(
    json.dumps({"count": len(items), "packages": items}, indent=2, ensure_ascii=False) + "\n"
)
print(f"[package-pypto] manifest: {out} ({len(items)} package(s))")
PY

log "done: packed=$packed out=$OUT_DIR"
