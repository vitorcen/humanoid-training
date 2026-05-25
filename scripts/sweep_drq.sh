#!/usr/bin/env bash
# Sweep DR.Q across all 28 tasks, seed 0 only, 5 ep each.
# _Quick scan to find tasks where DR.Q seed 0 already reaches ≥50%._
set -e

TASKS=(
  h1-balance_hard-v0 h1-balance_simple-v0 h1-crawl-v0 h1-hurdle-v0
  h1-maze-v0 h1-pole-v0 h1-reach-v0 h1-run-v0
  h1-sit_hard-v0 h1-sit_simple-v0 h1-slide-v0 h1-stair-v0
  h1-stand-v0 h1-walk-v0
  h1hand-basketball-v0 h1hand-bookshelf_hard-v0 h1hand-bookshelf_simple-v0
  h1hand-crawl-v0 h1hand-door-v0 h1hand-pole-v0 h1hand-reach-v0
  h1hand-run-v0 h1hand-sit_hard-v0 h1hand-sit_simple-v0
  h1hand-slide-v0 h1hand-stair-v0 h1hand-stand-v0 h1hand-walk-v0
)

mkdir -p results/drq_sweep_seed0
for t in "${TASKS[@]}"; do
  echo "=== $t ==="
  conda run -n humanoidbench --no-capture-output python scripts/eval.py \
    --task "$t" --driver drq --eval 5 --seed_list 0 --action_repeat 2 \
    --out "results/drq_sweep_seed0/${t}.jsonl" 2>&1 | tail -2
done

echo ""
echo "=== Summary across all tasks ==="
python - <<'PY'
import json
from pathlib import Path
rows = []
for p in sorted(Path("results/drq_sweep_seed0").glob("*.jsonl")):
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
