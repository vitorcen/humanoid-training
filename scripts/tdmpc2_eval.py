"""TD-MPC2 multi-episode evaluator — mirrors scripts/eval.py JSONL contract.

_Same per-episode + _summary rows so ckpt_eval_loop / dashboards can be reused._

Usage:
    python scripts/tdmpc2_eval.py --task humanoid_h1-walk-v0 \\
        --ckpt runs/h1_tdmpc2_pilot/.../models/step_00050000.pt \\
        --seed 0 --eval 3 --out /tmp/_tdmpc2_eval.jsonl
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# silence MuJoCo GL chatter before tdmpc2 import
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LAZY_LEGACY_OP", "0")

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dependencies" / "humanoid-bench" / "tdmpc2"))
sys.path.insert(0, str(ROOT / "scripts"))

from tdmpc2.common.parser import parse_cfg
from tdmpc2.envs import make_env
from tdmpc2.tdmpc2 import TDMPC2

DEFAULT_CFG_PATH = ROOT / "dependencies/humanoid-bench/tdmpc2/tdmpc2/config.yaml"
KNOWN_BAR = {
    "walk": 700, "run": 700, "stand": 800, "sit_simple": 750, "sit_hard": 750,
    "crawl": 700, "pole": 700, "stair": 700, "slide": 700, "hurdle": 700,
    "balance_simple": 800, "balance_hard": 800, "maze": 1200, "reach": 12000,
}


@dataclass
class EpisodeStats:
    task: str
    driver: str
    seed: int
    ep_idx: int
    ep_steps: int = 0
    ep_return: float = 0.0
    success: bool = False
    subtasks_max: int = 0
    time_to_success_s: float | None = None
    timed_out: bool = False
    wall_time_s: float = 0.0


def get_success_bar(task: str) -> float | None:
    for k, v in KNOWN_BAR.items():
        if f"-{k}-" in task:
            return float(v)
    return None


def load_cfg(task: str, seed: int) -> OmegaConf:
    """Build minimal cfg for TDMPC2 init without hydra cwd-juggling."""
    cfg = OmegaConf.load(DEFAULT_CFG_PATH)
    cfg.task = task
    cfg.seed = seed
    cfg.exp_name = "eval"
    cfg.data_dir = "/tmp/_tdmpc2_eval_buffer"
    cfg.checkpoint = ""
    cfg.disable_wandb = True
    cfg.save_csv = False
    cfg.save_agent = False
    cfg.save_video = False
    cfg.multitask = False
    cfg.model_size = 5
    # parse_cfg uses hydra.utils.get_original_cwd(); fake it
    import hydra.utils
    hydra.utils.get_original_cwd = lambda: str(ROOT)
    cfg = parse_cfg(cfg)
    return cfg


def run_episode(env, agent, task_name: str, seed: int, ep_idx: int,
                success_bar: float | None) -> EpisodeStats:
    # TensorWrapper.reset() drops seed kwarg; seed via unwrapped + space RNGs instead
    s = seed + ep_idx
    env.unwrapped.action_space.seed(s)
    env.unwrapped.observation_space.seed(s)
    try:
        env.unwrapped.reset(seed=s)
    except TypeError:
        pass
    obs, _ = env.reset()
    stats = EpisodeStats(task=task_name, driver="tdmpc2", seed=seed, ep_idx=ep_idx)
    dt = float(env.unwrapped.model.opt.timestep)
    t0_step = time.time()
    success_step = None
    done = False
    t = 0
    while not done:
        action = agent.act(obs, t0=(t == 0), eval_mode=True)
        obs, reward, term, trunc, info = env.step(action)
        stats.ep_steps += 1
        stats.ep_return += float(reward)
        if isinstance(info, dict):
            sub = info.get("success_subtasks")
            if sub is not None:
                stats.subtasks_max = max(stats.subtasks_max, int(sub))
            if info.get("success") and success_step is None:
                success_step = stats.ep_steps
                stats.success = True
        done = bool(term or trunc)
        if done:
            stats.timed_out = bool(trunc and not term)
        t += 1
    if not stats.success and success_bar is not None and stats.ep_return >= success_bar:
        stats.success = True
    if success_step is not None:
        stats.time_to_success_s = success_step * dt
    stats.wall_time_s = time.time() - t0_step
    return stats


def aggregate(rows: list[EpisodeStats]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n_episodes": 0}
    succ = [r for r in rows if r.success]
    ttf = [r.time_to_success_s for r in succ if r.time_to_success_s is not None]
    return {
        "n_episodes": n,
        "success_rate": len(succ) / n,
        "mean_return": float(np.mean([r.ep_return for r in rows])),
        "mean_steps": float(np.mean([r.ep_steps for r in rows])),
        "mean_subtasks": float(np.mean([r.subtasks_max for r in rows])),
        "mean_ttf_success_s": float(np.mean(ttf)) if ttf else None,
        "timeout_rate": float(np.mean([r.timed_out for r in rows])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True,
                   help="tdmpc2 task name, e.g. humanoid_h1-walk-v0")
    p.add_argument("--ckpt", required=True, help=".pt produced by TDMPC2.save()")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", dest="n_eval", type=int, default=10)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_cfg(args.task, args.seed)
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    agent.load(args.ckpt)

    bench_task = args.task.removeprefix("humanoid_")
    bar = get_success_bar(bench_task)
    print(f"[tdmpc2_eval] task={args.task} ckpt={Path(args.ckpt).name} "
          f"seed={args.seed} success_bar={bar}")

    out_path = Path(args.out) if args.out else None
    out_f = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w")

    all_rows = []
    t0_total = time.time()
    try:
        for ep in range(args.n_eval):
            row = run_episode(env, agent, bench_task, args.seed, ep, bar)
            all_rows.append(row)
            if out_f:
                out_f.write(json.dumps(asdict(row)) + "\n")
                out_f.flush()
            tag = "✅" if row.success else ("⏱" if row.timed_out else "·")
            print(f"  {tag} ep{ep:02d} steps={row.ep_steps:4d} "
                  f"return={row.ep_return:8.2f} sub={row.subtasks_max} "
                  f"wall={row.wall_time_s:.1f}s")
    finally:
        agg = aggregate(all_rows)
        agg.update(task=bench_task, driver="tdmpc2", seeds=[args.seed],
                   eval_per_seed=args.n_eval,
                   total_wall_s=time.time() - t0_total)
        if out_f:
            out_f.write(json.dumps({"_summary": True, **agg}) + "\n")
            out_f.close()

    print("\n=== Aggregate ===")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
