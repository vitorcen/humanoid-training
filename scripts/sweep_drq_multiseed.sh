#!/usr/bin/env bash
# Multi-seed validation on tasks where DR.Q seed 0 already hit ≥50%.
# _10 ep × 3 seeds — this is the Step 2 acceptance run._
set -e

TASKS=(
  h1-crawl-v0 h1-pole-v0 h1-run-v0 h1-stand-v0 h1-walk-v0
  h1-sit_hard-v0 h1-sit_simple-v0
  h1hand-sit_simple-v0 h1hand-sit_hard-v0
)

mkdir -p results/drq_multiseed
for t in "${TASKS[@]}"; do
  echo "=== $t ==="
  conda run -n humanoidbench --no-capture-output python scripts/eval.py \
    --task "$t" --driver drq --eval 10 --seed_list 0,10,20 --action_repeat 2 \
    --out "results/drq_multiseed/${t}.jsonl" 2>&1 | tail -2
done

echo ""
echo "=== Multi-seed summary (N=30 ep) ==="
python - <<'PY'
import json
from pathlib import Path
rows = []
for p in sorted(Path("results/drq_multiseed").glob("*.jsonl")):
    with p.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("_summary"):
                rows.append((d["task"], d["success_rate"], d["mean_return"], d["timeout_rate"]))
print(f"{'task':<32}  {'succ':>6}  {'mean_ret':>9}  {'timeout':>7}")
for task, sr, mr, to in rows:
    flag = "🟢" if sr >= 0.5 else ("🟡" if sr >= 0.2 else "🔴")
    print(f"{task:<32}  {sr*100:5.0f}%  {mr:9.1f}  {to*100:6.0f}%  {flag}")
PY
