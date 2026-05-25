"""Auto-eval daemon — polls DR.Q checkpoint dir, quick-evals each new ckpt.

_LeIsaac eval_watcher pattern adapted for DR.Q: for each new ckpt copy the
3 critical files to HF cache location so drq_viewer/eval.py can resolve it,
then run N=3 ep eval and append to ckpt_eval.csv._

Designed to run alongside train_watcher.py:
    - train_watcher.py  → reads eval log, monitors fit status, early-stop
    - ckpt_eval_loop.py → reads ckpt dir, runs ACTUAL deterministic eval per ckpt

Usage:
    python scripts/ckpt_eval_loop.py \\
        --task g1-walk-v0 --seed 0 \\
        --ckpt_dir runs/g1_pilot/DRQ/checkpoint/DRQ+HBench-g1-walk-v0+0 \\
        --train_pid 722151 \\
        --out runs/g1_pilot/ckpt_eval.csv \\
        --eval_eps 3
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dependencies" / "dr-q" / "DRQ"))  # for pickle of Hyperparameters


CRITICAL = ("policy.pt", "encoder.pt", "agent_var.npy")


def ensure_hf_cache_link(ckpt_dir: Path, task: str, seed: int) -> Path:
    """Copy ckpt's 3 critical files into a uniquely-named directory under HF cache
    so drq_viewer.resolve_ckpt(...) can find it.

    Strategy: use a custom 'snapshots/<train_run_tag>/' instead of the upstream snapshot dir.
    Returns the path drq_viewer will resolve.
    """
    project = f"DRQ+HBench-{task}+{seed}"
    snap = Path.home() / ".cache/huggingface/hub/models--dmux--DR.Q/snapshots/_local_train"
    snap.mkdir(parents=True, exist_ok=True)
    target = snap / project
    target.mkdir(exist_ok=True)
    for f in CRITICAL:
        src = ckpt_dir / f
        if not src.is_file():
            raise FileNotFoundError(f"missing {f} in {ckpt_dir}")
        shutil.copy2(src, target / f)
    return target


def quick_eval(task: str, seed: int, eval_eps: int, action_repeat: int = 2) -> dict:
    """Run scripts/eval.py for N ep × 1 seed; return aggregate from JSONL summary."""
    out_path = ROOT / "runs/g1_pilot" / f"_ckpt_eval_{task}_{seed}_{int(time.time())}.jsonl"
    cmd = [
        "conda", "run", "-n", "humanoidbench", "--no-capture-output",
        "python", str(ROOT / "scripts/eval.py"),
        "--task", task, "--driver", "drq",
        "--eval", str(eval_eps), "--seed_list", str(seed),
        "--action_repeat", str(action_repeat),
        "--out", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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


def discover_ckpts(ckpt_dir: Path) -> list[Path]:
    """Return list of ckpt dirs that have all 3 critical files.

    DR.Q saves into a flat dir (overwriting each save_freq); to capture each
    save, train_watcher should be configured to checkpoint elsewhere. Here we
    just return [ckpt_dir] if complete.
    """
    if not ckpt_dir.is_dir():
        return []
    if all((ckpt_dir / f).is_file() for f in CRITICAL):
        return [ckpt_dir]
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--train_pid", type=int, default=None,
                   help="exit when this pid is gone")
    p.add_argument("--out", default="runs/g1_pilot/ckpt_eval.csv")
    p.add_argument("--eval_eps", type=int, default=3)
    p.add_argument("--poll_s", type=float, default=60.0)
    p.add_argument("--action_repeat", type=int, default=2)
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
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
            w = csv.writer(f)
            w.writerow(["timestamp", "ckpt_mtime", "agent_train_step",
                        "success_rate", "mean_return", "mean_steps",
                        "timeout_rate", "n_ep", "note"])

    log(f"daemon start  task={args.task}  seed={args.seed}")
    log(f"  ckpt_dir={ckpt_dir}")
    log(f"  out={out_csv}  poll={args.poll_s}s  eval_eps={args.eval_eps}")

    last_mtime = None
    while True:
        # train alive check
        if args.train_pid:
            try:
                os.kill(args.train_pid, 0)
                train_alive = True
            except ProcessLookupError:
                train_alive = False
        else:
            train_alive = True

        ckpts = discover_ckpts(ckpt_dir)
        if ckpts:
            ck = ckpts[0]
            mtime = (ck / "policy.pt").stat().st_mtime
            if mtime != last_mtime:
                log(f"new ckpt detected mtime={time.strftime('%H:%M:%S', time.localtime(mtime))}")
                # extract training_steps from agent_var.npy
                try:
                    import numpy as np
                    var = np.load(ck / "agent_var.npy", allow_pickle=True).item()
                    ts = int(var.get("training_steps", -1))
                except Exception as e:
                    ts = -1
                    log(f"  could not read training_steps: {e}")

                cached = ensure_hf_cache_link(ck, args.task, args.seed)
                log(f"  mirrored to {cached}")
                summary = quick_eval(args.task, args.seed, args.eval_eps, args.action_repeat)
                if "error" in summary:
                    log(f"  ❌ eval error: {summary['error']}")
                    with out_csv.open("a") as f:
                        csv.writer(f).writerow([int(time.time()), int(mtime), ts,
                                                "", "", "", "", args.eval_eps,
                                                f"error: {summary['error'][:80]}"])
                else:
                    sr = summary.get("success_rate", 0.0)
                    mr = summary.get("mean_return", 0.0)
                    log(f"  ✅ step≈{ts:>7}  success={sr*100:.0f}%  mean_return={mr:.2f}")
                    with out_csv.open("a") as f:
                        csv.writer(f).writerow([
                            int(time.time()), int(mtime), ts,
                            f"{sr:.4f}", f"{mr:.2f}",
                            f"{summary.get('mean_steps', 0):.1f}",
                            f"{summary.get('timeout_rate', 0):.4f}",
                            args.eval_eps, "ok"])
                last_mtime = mtime

        if not train_alive:
            log(f"train pid {args.train_pid} gone — exit daemon")
            break

        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
