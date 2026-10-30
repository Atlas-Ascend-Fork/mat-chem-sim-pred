#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 把算子打成 CANN custom opp 包：
#   msopgen gen -> 覆盖 op_host/op_kernel -> 生成工程 build.sh -> 收集 custom_opp_*.run
#
# 依赖：CANN toolkit（msopgen + 生成工程的 build.sh）。本脚本只做编排，不复制 CANN 工具。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SOC=""
FRAMEWORK="tf"
OUT_DIR="build_out/opp"
REQUIRE_ALL=false
KEEP_WORK=false
OPS=()

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[package] $*"; }

usage() {
    cat <<'EOF'
Package selected operators as CANN custom opp packages.

Usage:
  bash scripts/package_opp.sh --soc=<SOC> [options] <op>...

Options:
  --soc=<SOC>            Target SOC version, e.g. Ascend910B1
  --framework=<name>     msopgen framework argument (default: tf)
  --out=<dir>            Output directory (default: build_out/opp)
  --require-all          Fail if any selected operator has no msopgen/*.json
  --keep-work            Keep the generated msopgen work directory
  -h, --help             Show this help

Each operator needs:
  <op>/msopgen/*.json    Operator prototype definition
  <op>/op_host/          Host implementation
  <op>/op_kernel/        Kernel implementation

Output:
  <out>/<op>_custom_opp_<os>_<arch>.run
  <out>/manifest.json
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --soc=*)        SOC="${1#--soc=}" ;;
        --soc)          shift; SOC="${1:-}" ;;
        --framework=*)  FRAMEWORK="${1#--framework=}" ;;
        --framework)    shift; FRAMEWORK="${1:-}" ;;
        --out=*)        OUT_DIR="${1#--out=}" ;;
        --out)          shift; OUT_DIR="${1:-}" ;;
        --require-all)  REQUIRE_ALL=true ;;
        --keep-work)    KEEP_WORK=true ;;
        -h|--help)      usage; exit 0 ;;
        -*)             die "unknown option: $1" ;;
        *)              OPS+=("$1") ;;
    esac
    shift
done

[ -n "$SOC" ] || die "--soc is required"
[ ${#OPS[@]} -gt 0 ] || die "at least one operator path is required"

MSOPGEN_BIN="${MSOPGEN_BIN:-}"
if [ -z "$MSOPGEN_BIN" ]; then
    MSOPGEN_BIN="$(command -v msopgen || command -v msOpGen || true)"
fi
[ -n "$MSOPGEN_BIN" ] && [ -x "$MSOPGEN_BIN" ] \
    || die "msopgen not found; source CANN set_env.sh or set MSOPGEN_BIN"

mkdir -p "$OUT_DIR"
WORK_DIR="$OUT_DIR/.work"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

SOC_LOWER="$(printf '%s' "$SOC" | tr '[:upper:]' '[:lower:]')"
SOC_ARG="ai_core-${SOC_LOWER}"
RECORDS="$(mktemp)"
trap 'rm -f "$RECORDS"' EXIT

packed=0
skipped=0
for op in "${OPS[@]}"; do
    op="${op%/}"
    [ -d "$op" ] || die "operator directory not found: $op"
    name="$(basename "$op")"

    json="$(find "$op/msopgen" -maxdepth 1 -name '*.json' -print 2>/dev/null | head -1 || true)"
    if [ -z "$json" ]; then
        echo "[package][SKIP] $op — no msopgen/*.json"
        skipped=$((skipped + 1))
        continue
    fi
    [ -d "$op/op_host" ] || die "$op — missing op_host/"
    [ -d "$op/op_kernel" ] || die "$op — missing op_kernel/"

    project="$WORK_DIR/$name"
    log "msopgen gen $op -> $project"
    rm -rf "$project"
    "$MSOPGEN_BIN" gen -i "$json" -f "$FRAMEWORK" -c "$SOC_ARG" -lan cpp -out "$project"

    # msopgen 生成工程只带骨架；用仓内实现覆盖 op_host/op_kernel。
    cp -rf "$op/op_host/." "$project/op_host/"
    cp -rf "$op/op_kernel/." "$project/op_kernel/"

    # 族共享头：op_host 里的 "../../common/..." 依赖族 common/。
    family="$(dirname "$op")"
    if [ -d "$family/common" ]; then
        mkdir -p "$WORK_DIR/common"
        cp -rf "$family/common/." "$WORK_DIR/common/"
    fi

    log "build $name (SOC=$SOC)"
    ( cd "$project" && bash build.sh )

    run_file="$(find "$project/build_out" -maxdepth 2 -name '*.run' -print 2>/dev/null | head -1 || true)"
    [ -n "$run_file" ] || die "$op — build finished but no custom_opp_*.run under build_out/"
    dest="$OUT_DIR/${name}_$(basename "$run_file")"
    cp "$run_file" "$dest"
    printf '%s\t%s\n' "$op" "$(basename "$dest")" >> "$RECORDS"
    packed=$((packed + 1))
    log "packed $dest"
done

if [ "$REQUIRE_ALL" = true ] && [ "$skipped" -gt 0 ]; then
    die "$skipped operator(s) have no msopgen/*.json; --require-all forbids skipping"
fi

python3 - "$OUT_DIR/manifest.json" "$SOC" "$FRAMEWORK" "$RECORDS" <<'PY'
import json
import sys
from pathlib import Path

out, soc, framework, records = sys.argv[1:5]
items = []
for line in Path(records).read_text().splitlines():
    if not line.strip():
        continue
    op, package = line.split("\t", 1)
    items.append({"op": op, "package": package})
manifest = {
    "soc": soc,
    "framework": framework,
    "count": len(items),
    "packages": items,
}
Path(out).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print(f"[package] manifest: {out} ({len(items)} package(s))")
PY

if [ "$KEEP_WORK" != true ]; then
    rm -rf "$WORK_DIR"
fi

log "done: packed=$packed skipped=$skipped out=$OUT_DIR"
