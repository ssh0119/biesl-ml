#!/usr/bin/env bash
# Run the full classical + 1D-CNN pipeline at two new window configurations
# (40 s / 20 s overlap, 30 s / 15 s overlap), saving JSONs into per-variant
# subdirectories. Edits src/features.py's WINDOW_S between runs and restores
# it to 60.0 at the end (even on failure).
#
# Output:
#   outputs/win40_ov20/{local_loro,local_randomsplit,dl_local_loro,dl_local_randomsplit}.json
#   outputs/win30_ov15/{...same four...}.json
#
# Usage:  bash scripts/experiments/run_window_variants.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

FEATURES_PY="src/features.py"

set_window() {
    local w="$1"
    sed -i "s/^WINDOW_S = .*/WINDOW_S = ${w}           # set by scripts\/run_window_variants.sh/" "$FEATURES_PY"
    echo "[driver] patched WINDOW_S = ${w}"
    grep "^WINDOW_S" "$FEATURES_PY"
}

restore() {
    set_window "60.0"
    echo "[driver] restored WINDOW_S = 60.0"
}
trap restore EXIT

set_window "40.0"
uv run python scripts/experiments/run_one_window_variant.py outputs/win40_ov20

set_window "30.0"
uv run python scripts/experiments/run_one_window_variant.py outputs/win30_ov15

# trap restores 60.0 on exit
echo "[driver] all variants done"
