#!/usr/bin/env bash
# Print current G1 / DR.Q training status — milestones, dense ASCII curve, live verdict.
# _Usage: bash scripts/train_status.sh runs/g1_pilot/DRQ/HBench-g1-walk-v0/r0_

set -u
RUN="${1:-runs/g1_pilot/DRQ/HBench-g1-walk-v0/r0}"
STATUS="$RUN/auto_eval.status.json"
MILE="$RUN/auto_eval.csv"
DENSE="$RUN/eval_dense.csv"

if [[ ! -f "$STATUS" ]]; then
    echo "❌ watcher not running — no $STATUS. Start it with:"
    echo "    python scripts/train_watcher.py --run $RUN --train_pid <PID> --total_steps 500000 --success_bar 700"
    exit 1
fi

echo "=== Live status / 当前状态 ==="
python3 -c "
import json
s = json.load(open('$STATUS'))
print(f'  evals seen       : {s[\"evals_seen\"]}')
print(f'  ~step            : {s[\"approx_step\"]:,}  ({s[\"approx_progress_pct\"]:.1f}%)')
print(f'  last return      : {s[\"last_return\"]}')
print(f'  peak return      : {s[\"peak_return\"]} @ eval#{s[\"peak_at_eval\"]}')
print(f'  fit status       : {s[\"status\"]}')
print(f'  note             : {s[\"note\"]}')
"

if [[ -f "$MILE" ]]; then
    echo
    echo "=== Milestones / 10-slice 拐点 ==="
    column -t -s, "$MILE"
fi

if [[ -f "$DENSE" ]]; then
    n=$(wc -l < "$DENSE")
    if [[ "$n" -gt 1 ]]; then
        echo
        echo "=== Reward curve (last 60 evals) ==="
        python3 - <<PY
import csv
rows = list(csv.DictReader(open("$DENSE")))
rows = rows[-60:]
vals = [float(r["eval_return"]) for r in rows]
if not vals: raise SystemExit
lo, hi = min(vals), max(vals)
span = max(1e-9, hi - lo)
width = 50
for r in rows:
    v = float(r["eval_return"])
    n = int((v - lo) / span * width)
    bar = "█" * n + "·" * (width - n)
    print(f"  step {int(r['step']):>7}  {v:7.2f}  |{bar}|")
print(f"  range [lo={lo:.1f}, hi={hi:.1f}]  span={span:.1f}")
PY
    fi
fi

if [[ -f "$RUN/.eval_abort" ]]; then
    echo
    echo "🛑 .eval_abort marker present — watcher signaled abort"
fi
