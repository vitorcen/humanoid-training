"""DR.Q training watcher — LeIsaac-style auto-eval + early stop + over/under-fit detection.

_Polls DR.Q's eval log every POLL_S, aggregates per-slice CSV, detects
 underfit / overfit / dead-policy, can SIGTERM training on persistent failure._

LeIsaac rule (verbatim): split total steps into 10 slices, quick-eval after each.
DR.Q already evals every `eval_freq` (5000 steps) — denser than LeIsaac. This
watcher converts that stream into:

    1. milestone CSV  `<RUN>/auto_eval.csv`  (one row per 10%-slice)
    2. dense CSV      `<RUN>/eval_dense.csv` (one row per DR.Q eval)
    3. status JSON    `<RUN>/auto_eval.status.json` (live status for shell)
    4. abort marker   `<RUN>/.eval_abort`     (early-stop trigger)

Detection rules:
    - DEAD          : last 50 evals all < success_bar/10  (= "policy never moves")
    - UNDERFIT      : last 30 evals' max didn't improve >5% over prior 30
    - OVERFIT       : peak reward seen ≥ 20 evals ago AND current < 0.7 × peak
    - PROGRESS      : otherwise (still climbing or near peak)

Usage:
    python scripts/train_watcher.py \\
        --run runs/g1_pilot/DRQ/HBench-g1-walk-v0/r0 \\
        --train_pid 648471 \\
        --total_steps 500000 \\
        --success_bar 700

Run in background; tails eval log; touches abort marker on DEAD streak;
parent shell can `kill $(cat $RUN/.train_pid)` when marker appears.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


def read_eval_log(p: Path) -> list[float]:
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(float(line))
        except ValueError:
            pass
    return out


def classify(returns: list[float], success_bar: float) -> dict:
    """LeIsaac-inspired fit-status detection."""
    n = len(returns)
    if n == 0:
        return {"status": "WARMUP", "note": "no eval yet"}
    peak = max(returns)
    peak_idx = returns.index(peak)
    last = returns[-1]
    floor = success_bar / 10  # "dead policy" threshold

    if n >= 50 and all(r < floor for r in returns[-50:]):
        return {"status": "DEAD", "note": f"last 50 evals all < {floor:.1f}"}

    if n >= 60:
        recent = max(returns[-30:])
        prior = max(returns[-60:-30])
        if recent < prior * 1.05:
            return {"status": "UNDERFIT", "note": f"max(last30)={recent:.1f} ≤ 1.05×max(prior30)={prior*1.05:.1f}"}

    if n >= 20 and peak_idx <= n - 20 and last < 0.7 * peak:
        return {"status": "OVERFIT", "note": f"peak={peak:.1f} at eval#{peak_idx}, last={last:.1f} (<0.7×peak)"}

    return {"status": "PROGRESS", "note": f"last={last:.1f} peak={peak:.1f}@#{peak_idx}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="DR.Q run dir, e.g. runs/g1_pilot/DRQ/HBench-g1-walk-v0/r0")
    p.add_argument("--eval_log", default=None, help="override eval log path (default: <run>/../../evals/<project>.txt)")
    p.add_argument("--project_name", default=None, help="override project name (default: auto-derive)")
    p.add_argument("--train_pid", type=int, default=None, help="training PID to SIGTERM on DEAD")
    p.add_argument("--total_steps", type=int, default=500_000)
    p.add_argument("--eval_freq", type=int, default=5_000,
                   help="DR.Q HBench eval frequency in agent steps")
    p.add_argument("--success_bar", type=float, default=700.0,
                   help="task.success_bar, used for DEAD threshold = bar/10")
    p.add_argument("--poll_s", type=float, default=30.0)
    p.add_argument("--slices", type=int, default=10, help="number of milestones to log")
    p.add_argument("--dead_kills", action="store_true",
                   help="actually SIGTERM the train PID on DEAD (default: just mark abort)")
    args = p.parse_args()

    run = Path(args.run).resolve()
    drq_root = run.parents[1]  # .../DRQ/
    project = args.project_name or f"DRQ+{run.parents[0].name}+{run.name.lstrip('r')}"
    eval_log = Path(args.eval_log) if args.eval_log else drq_root / "evals" / f"{project}.txt"

    run.mkdir(parents=True, exist_ok=True)
    csv_milestone = run / "auto_eval.csv"
    csv_dense = run / "eval_dense.csv"
    status_json = run / "auto_eval.status.json"
    abort_marker = run / ".eval_abort"
    log = run / "watcher.log"

    if not csv_milestone.exists():
        csv_milestone.write_text("slice,step,eval_return,peak_so_far,status,note\n")
    if not csv_dense.exists():
        csv_dense.write_text("eval_idx,step,eval_return\n")

    def write_log(msg: str):
        with log.open("a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        print(msg, flush=True)

    write_log(f"watcher start  run={run}")
    write_log(f"  eval_log={eval_log}  project={project}")
    write_log(f"  total_steps={args.total_steps}  eval_freq={args.eval_freq}  success_bar={args.success_bar}")
    write_log(f"  train_pid={args.train_pid}  dead_kills={args.dead_kills}")

    slice_step = args.total_steps // args.slices
    last_dense_n = 0
    last_milestone_step = -1
    abort_fired = False

    while True:
        if abort_marker.exists() and not abort_fired:
            write_log("abort marker exists — exiting watcher")
            break

        returns = read_eval_log(eval_log)
        n = len(returns)

        # dense rows
        if n > last_dense_n:
            with csv_dense.open("a") as f:
                for i in range(last_dense_n, n):
                    f.write(f"{i+1},{(i+1)*args.eval_freq},{returns[i]:.4f}\n")
            last_dense_n = n

        # milestone rows (one per slice crossed)
        approx_step = n * args.eval_freq
        slice_idx = approx_step // slice_step
        for s in range(int(last_milestone_step // slice_step) + 1, int(slice_idx) + 1):
            milestone_step = s * slice_step
            sub_returns = returns[: (milestone_step // args.eval_freq)]
            cls = classify(sub_returns, args.success_bar)
            peak = max(sub_returns) if sub_returns else float("nan")
            last = sub_returns[-1] if sub_returns else float("nan")
            with csv_milestone.open("a") as f:
                f.write(f"{s}/{args.slices},{milestone_step},{last:.4f},{peak:.4f},{cls['status']},{cls['note']}\n")
            write_log(f"slice {s}/{args.slices} @ step {milestone_step:>7} | last={last:7.2f} peak={peak:7.2f} | {cls['status']} :: {cls['note']}")
        last_milestone_step = max(last_milestone_step, slice_idx * slice_step)

        # live status
        cls = classify(returns, args.success_bar)
        status = {
            "timestamp": int(time.time()),
            "evals_seen": n,
            "approx_step": approx_step,
            "approx_progress_pct": min(100.0, 100.0 * approx_step / args.total_steps),
            "last_return": returns[-1] if returns else None,
            "peak_return": max(returns) if returns else None,
            "peak_at_eval": (returns.index(max(returns)) + 1) if returns else None,
            **cls,
        }
        status_json.write_text(json.dumps(status, indent=2))

        # abort logic
        if cls["status"] == "DEAD" and not abort_fired:
            write_log(f"⚠ DEAD detected — {cls['note']}")
            abort_marker.touch()
            abort_fired = True
            if args.dead_kills and args.train_pid:
                write_log(f"sending SIGTERM to train pid {args.train_pid}")
                try:
                    os.kill(args.train_pid, signal.SIGTERM)
                except ProcessLookupError:
                    write_log("  pid already gone")

        # exit if training finished
        if approx_step >= args.total_steps:
            write_log(f"reached total_steps {args.total_steps} — final status: {cls['status']}")
            break

        # also exit if train pid died
        if args.train_pid:
            try:
                os.kill(args.train_pid, 0)
            except ProcessLookupError:
                write_log(f"train pid {args.train_pid} no longer alive — exiting")
                break

        time.sleep(args.poll_s)

    write_log("watcher done")


if __name__ == "__main__":
    main()
