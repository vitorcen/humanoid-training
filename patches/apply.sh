#!/usr/bin/env bash
# Apply all local patches under patches/*.patch to their submodules.
# _Idempotent: skips patches already applied._
set -euo pipefail
cd "$(dirname "$0")/.."

apply_one() {
    local target_dir="$1"
    local patch_file
    patch_file="$(realpath "$2")"
    if (cd "$target_dir" && git apply --reverse --check "$patch_file" 2>/dev/null); then
        echo "✓ already applied: $2"
    elif (cd "$target_dir" && git apply --check "$patch_file" 2>/dev/null); then
        (cd "$target_dir" && git apply "$patch_file")
        echo "✓ applied: $2"
    else
        echo "✗ cannot apply (conflict or already partial): $2" >&2
        return 1
    fi
}

apply_one dependencies/dr-q patches/dr-q-save-agent.patch
apply_one dependencies/humanoid-bench patches/g1-pos-control.patch
apply_one dependencies/humanoid-bench patches/humanoid-bench-g1-blocked-hands.patch
