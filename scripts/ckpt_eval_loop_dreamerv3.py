"""DreamerV3 per-ckpt auto-eval daemon.

DreamerV3 keeps a single rolling `logdir/checkpoint.ckpt` (overwritten every
`--run.save_every` seconds). We watch its mtime, snapshot to
`pending_eval/step_<N>.ckpt`, and run a deterministic N-ep `eval_only` per
snapshot. VRAM-aware: if train + JAX hold >X GB, defer eval into the queue
and drain after training exits.

Outputs to `<run_dir>/eval_history.jsonl` — one row per completed eval, with
`status ∈ {DEAD, UNDERFIT, OVERFIT, PROGRESS}`.

Usage:
    python scripts/ckpt_eval_loop_dreamerv3.py \\
        --run-dir runs/h1hand_window_dreamer_pilot \\
        --task humanoid_h1hand-window-v0 \\
        --seed 0 --eval-n 5 --success-bar 650 \\
        [--vram-mode parallel|deferred|auto]
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def gpu_free_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0


def parse_metrics_for_step(logdir: Path) -> int:
    """Read tail of metrics.jsonl to get current training step."""
    m = logdir / "metrics.jsonl"
    if not m.is_file():
        return 0
    last = 0
    with m.open() as f:
        for line in f:
            try:
                d = json.loads(line)
                if "step" in d:
                    last = max(last, int(d["step"]))
            except json.JSONDecodeError:
                continue
    return last


def run_eval(ckpt: Path, task: str, seed: int, n_eps: int, eval_logdir: Path) -> dict:
    """Spawn embodied.agents.dreamerv3.train --run.script eval_only."""
    eval_logdir.mkdir(parents=True, exist_ok=True)
    target_steps = max(2000, n_eps * 1000)
    cmd = [
        "conda", "run", "-n", "humanoidbench-jax", "--no-capture-output",
        "python", "-m", "embodied.agents.dreamerv3.train",
        "--configs", "humanoid_benchmark", "small",
        "--method", "dreamer",
        "--logdir", str(eval_logdir),
        "--task", task,
        "--seed", str(seed),
        "--run.from_checkpoint", str(ckpt),
        "--run.script", "eval_only",
        "--run.steps", str(target_steps),
        "--run.num_envs", "1",
        "--jax.prealloc", "False",
        "--run.wandb", "False",
    ]
    env = {**os.environ, "MUJOCO_GL": "egl"}
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    scores_jsonl = eval_logdir / "scores.jsonl"
    if not scores_jsonl.is_file():
        return {"error": (proc.stderr or proc.stdout)[-400:]}
    eps = []
    for line in scores_jsonl.read_text().splitlines():
        try:
            d = json.loads(line)
            if "episode/score" in d:
                eps.append(float(d["episode/score"]))
        except json.JSONDecodeError:
            continue
    if not eps:
        return {"error": "no episodes"}
    eps = eps[-n_eps:] if len(eps) >= n_eps else eps
    return {
        "n_eps": len(eps),
        "mean_return": mean(eps),
        "max_return": max(eps),
        "min_return": min(eps),
    }


def classify(history: list[dict], success_bar: float) -> tuple[str, str]:
    """Return (status, note). status ∈ {DEAD, UNDERFIT, OVERFIT, PROGRESS}."""
    if not history:
        return "PROGRESS", "no eval yet"
    last = history[-1]["mean_return"]
    if len(history) >= 5:
        recent_max = max(h["mean_return"] for h in history[-5:])
        if recent_max < success_bar * 0.1:
            return "DEAD", f"5 evals all < {success_bar*0.1:.1f}"
    peak = max(h["mean_return"] for h in history)
    peak_idx = max(range(len(history)), key=lambda i: history[i]["mean_return"])
    if len(history) >= 8 and len(history) - peak_idx >= 4 and last < 0.7 * peak:
        return "OVERFIT", f"peak {peak:.1f} {len(history)-peak_idx} evals ago, now {last:.1f}"
    if len(history) >= 8:
        prev = max(h["mean_return"] for h in history[-8:-4])
        curr = max(h["mean_return"] for h in history[-4:])
        if curr <= prev * 1.05:
            return "UNDERFIT", f"plateau: prev_max={prev:.1f} curr_max={curr:.1f}"
    return "PROGRESS", f"last={last:.1f} peak={peak:.1f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-n", type=int, default=5)
    p.add_argument("--success-bar", type=float, required=True)
    p.add_argument("--vram-mode", choices=["auto", "parallel", "deferred"], default="auto")
    p.add_argument("--vram-min-mb", type=int, default=4000,
                   help="if free VRAM < this, defer eval")
    p.add_argument("--train-pid", type=int, default=None)
    p.add_argument("--poll-s", type=float, default=120.0)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    logdir = run_dir / "logdir"
    pending = run_dir / "pending_eval"
    pending.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "eval_history.jsonl"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    log(f"daemon start  task={args.task}  vram_mode={args.vram_mode}")
    log(f"  run_dir={run_dir}  ckpt={logdir/'checkpoint.ckpt'}")
    log(f"  success_bar={args.success_bar}  eval_n={args.eval_n}")

    ckpt_path = logdir / "checkpoint.ckpt"
    last_mtime = 0.0
    history: list[dict] = []
    if history_path.exists():
        for line in history_path.read_text().splitlines():
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        log(f"  resumed history: {len(history)} prior evals")

    while True:
        train_alive = True
        if args.train_pid:
            try:
                os.kill(args.train_pid, 0)
            except ProcessLookupError:
                train_alive = False

        if ckpt_path.is_file():
            mt = ckpt_path.stat().st_mtime
            if mt > last_mtime + 1:
                step = parse_metrics_for_step(logdir)
                snap = pending / f"step_{step:08d}.ckpt"
                if not snap.exists():
                    shutil.copy2(ckpt_path, snap)
                    log(f"snapshot step={step} → {snap.name}")
                last_mtime = mt

        queue = sorted(pending.glob("step_*.ckpt"))
        mode = args.vram_mode
        if mode == "auto":
            free = gpu_free_mb()
            mode = "deferred" if free < args.vram_min_mb else "parallel"

        if queue and (mode == "parallel" or not train_alive):
            snap = queue[0]
            step = int(snap.stem.split("_")[1])
            eval_logdir = run_dir / f"eval_step_{step:08d}"
            log(f"eval ckpt step={step}  (mode={mode}, free={gpu_free_mb()}MB)")
            t0 = time.time()
            result = run_eval(snap, args.task, args.seed, args.eval_n, eval_logdir)
            dt = time.time() - t0
            if "error" in result:
                log(f"  ❌ {result['error'][:200]}")
                row = {"step": step, "status": "ERROR", "note": result["error"][:200],
                       "wall_time_s": dt}
            else:
                history.append({"step": step, **result})
                status, note = classify(history, args.success_bar)
                sr = sum(1 for _ in range(result["n_eps"])
                         if result["mean_return"] >= args.success_bar) / max(1, result["n_eps"])
                row = {"step": step, "mean_return": result["mean_return"],
                       "max_return": result["max_return"], "n_eps": result["n_eps"],
                       "success_rate_approx": sr, "status": status, "note": note,
                       "wall_time_s": dt}
                log(f"  ✅ mean={result['mean_return']:.2f} max={result['max_return']:.2f} "
                    f"status={status}  ({dt:.0f}s)")
                if status == "DEAD":
                    (run_dir / ".eval_abort").touch()
                    log("  🛑 .eval_abort touched")
            with history_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            snap.unlink()
        elif queue and mode == "deferred":
            log(f"deferred: {len(queue)} ckpts queued, free={gpu_free_mb()}MB < "
                f"{args.vram_min_mb}MB  (will drain when train exits)")

        if not train_alive and not queue:
            log(f"train exited and queue drained — daemon done ({len(history)} evals)")
            break

        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
