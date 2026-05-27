"""TD-MPC2 per-ckpt auto-eval daemon.

_Polls a TD-MPC2 `models/` dir for new `step_*.pt` files; runs N-ep eval per ckpt._

Companion to `scripts/train_watcher.py` (which only tails the eval log). Together they form
the same triple-process pattern used for DR.Q:
    A) train.py            — emits step_XXXXXXXX.pt every eval_freq
    B) train_watcher.py    — early-stops on DEAD/UNDERFIT/OVERFIT (reads training eval log)
    C) ckpt_eval_loop_tdmpc2.py — runs the *deterministic* N-ep eval per new ckpt

Usage:
    python scripts/ckpt_eval_loop_tdmpc2.py \\
        --task humanoid_h1-walk-v0 \\
        --models_dir runs/h1_tdmpc2_pilot/logs/humanoid_h1-walk-v0/0/h1_tdmpc2_pilot/models \\
        --train_pid 12345 \\
        --out runs/h1_tdmpc2_pilot/ckpt_eval.csv \\
        --eval_eps 3
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEP_RE = re.compile(r"step_(\d+)\.pt$")


def discover_ckpts(models_dir: Path, seen: set[str]) -> list[Path]:
    if not models_dir.is_dir():
        return []
    new = []
    for p in sorted(models_dir.glob("step_*.pt")):
        if p.name in seen:
            continue
        # skip very fresh files (still being written)
        if time.time() - p.stat().st_mtime < 5:
            continue
        new.append(p)
    return new


def quick_eval(task: str, ckpt: Path, seed: int, eval_eps: int) -> dict:
    out_path = ROOT / f"/tmp/_tdmpc2_ckpt_eval_{ckpt.stem}_{int(time.time())}.jsonl"
    cmd = [
        "conda", "run", "-n", "humanoidbench", "--no-capture-output",
        "python", str(ROOT / "scripts/tdmpc2_eval.py"),
        "--task", task, "--ckpt", str(ckpt),
        "--seed", str(seed), "--eval", str(eval_eps),
        "--out", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout)[-400:]}
    if not out_path.is_file():
        return {"error": "no jsonl produced"}
    summary = None
    for line in out_path.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("_summary"):
            summary = d
            break
    out_path.unlink(missing_ok=True)
    return summary or {"error": "no _summary row"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="tdmpc2 task name (humanoid_h1-walk-v0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--models_dir", required=True)
    p.add_argument("--train_pid", type=int, default=None)
    p.add_argument("--out", default="runs/h1_tdmpc2_pilot/ckpt_eval.csv")
    p.add_argument("--eval_eps", type=int, default=3)
    p.add_argument("--poll_s", type=float, default=60.0)
    args = p.parse_args()

    models_dir = Path(args.models_dir)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_csv.with_suffix(".log")

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    if not out_csv.exists():
        with out_csv.open("w") as f:
            csv.writer(f).writerow([
                "timestamp", "ckpt_mtime", "agent_train_step",
                "success_rate", "mean_return", "mean_steps",
                "timeout_rate", "n_ep", "note", "ckpt_file",
            ])

    log(f"daemon start  task={args.task}  seed={args.seed}")
    log(f"  models_dir={models_dir}")
    log(f"  out={out_csv}  poll={args.poll_s}s  eval_eps={args.eval_eps}")

    seen: set[str] = set()
    best = (-1e18, None)
    while True:
        if args.train_pid:
            try:
                os.kill(args.train_pid, 0)
                train_alive = True
            except ProcessLookupError:
                train_alive = False
        else:
            train_alive = True

        new = discover_ckpts(models_dir, seen)
        for ckpt in new:
            seen.add(ckpt.name)
            m = STEP_RE.search(ckpt.name)
            train_step = int(m.group(1)) if m else -1
            mtime = ckpt.stat().st_mtime
            log(f"new ckpt {ckpt.name} step={train_step} mtime={time.strftime('%H:%M:%S', time.localtime(mtime))}")
            summary = quick_eval(args.task, ckpt, args.seed, args.eval_eps)
            if "error" in summary:
                log(f"  ❌ eval error: {summary['error']}")
                with out_csv.open("a") as f:
                    csv.writer(f).writerow([int(time.time()), int(mtime), train_step,
                                             "", "", "", "", args.eval_eps,
                                             f"error: {summary['error'][:80]}", ckpt.name])
                continue
            sr = summary.get("success_rate", 0.0)
            mr = summary.get("mean_return", 0.0)
            tag = ""
            if mr > best[0]:
                best = (mr, ckpt.name)
                tag = "  🏆 NEW BEST"
            log(f"  ✅ step={train_step:>7}  success={sr*100:.0f}%  mean={mr:.2f}{tag}")
            with out_csv.open("a") as f:
                csv.writer(f).writerow([
                    int(time.time()), int(mtime), train_step,
                    f"{sr:.4f}", f"{mr:.2f}",
                    f"{summary.get('mean_steps', 0):.1f}",
                    f"{summary.get('timeout_rate', 0):.4f}",
                    args.eval_eps, "ok", ckpt.name,
                ])

        if not train_alive:
            log(f"train pid {args.train_pid} gone — exit daemon")
            log(f"  best mean_return so far: {best[0]:.2f} @ {best[1]}")
            break

        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
