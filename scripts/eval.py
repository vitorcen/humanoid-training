"""HumanoidBench multi-episode evaluator — no render, JSONL output, aggregate stats.

_Reusable scale before weighing — Step 1 of `project_benchmark_validation.md`._

Usage:
    # DR.Q on a locomotion task, 10 ep × 3 seeds
    python scripts/eval.py --task h1hand-walk-v0 --driver drq --eval 10 --seeds 3 \
        --out results/h1hand-walk-v0.jsonl

    # Random baseline (lower bound)
    python scripts/eval.py --task h1hand-stand-v0 --driver random --eval 5 --seeds 2

    # Reach skill on a supported manipulation task
    python scripts/eval.py --task h1hand-push-v0 --driver reach --eval 5 --seeds 2 \
        --policy_path dependencies/humanoid-bench/data/reach_two_hands/torch_model.pt \
        --mean_path   dependencies/humanoid-bench/data/reach_two_hands/mean.npy \
        --var_path    dependencies/humanoid-bench/data/reach_two_hands/var.npy

Per-episode JSONL fields:
    task, driver, seed, ep_idx, ep_steps, ep_return, success,
    subtasks_max, time_to_success_s, timed_out, wall_time_s

Aggregate (printed + appended to JSONL as a final summary row with "_summary": true):
    success_rate, mean_return, mean_steps, mean_ttf_success, timeout_rate, n_episodes
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import humanoid_bench  # noqa: F401,E402  registers gym envs


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


def make_env(task: str, driver: str, args):
    """gym.make + optional reach-skill kwargs."""
    env_kwargs = {}
    if driver == "reach":
        if not args.policy_path:
            sys.exit("❌ --driver reach 需要 --policy_path / --mean_path / --var_path")
        env_kwargs.update(
            policy_path=args.policy_path,
            mean_path=args.mean_path,
            var_path=args.var_path,
            policy_type=args.policy_type or "reach_double_relative",
        )
    try:
        return gym.make(task, **env_kwargs)
    except AttributeError as e:
        if "htarget_low" in str(e) and env_kwargs:
            sys.exit(
                f"❌ task '{task}' 不支持 reach wrapper (没有 htarget_low/high). "
                f"换 --driver random/zero/drq."
            )
        raise


def get_success_bar(env: gym.Env) -> float | None:
    """Read task.success_bar — walks through BlockedHandsLocoWrapper if present.

    HumanoidEnv.task can be a wrapper (e.g., BlockedHandsLocoWrapper) which proxies
    the real task as ._env's task attr — but `HumanoidEnv.task` is replaced by the
    wrapper, creating a cycle. So we have to look for the actual class with
    success_bar attribute via __class__.__mro__ inspection or wrapper chain.
    """
    base = env.unwrapped
    task = getattr(base, "task", None)
    if task is None:
        return None
    # direct hit
    bar = getattr(task, "success_bar", None)
    if bar is not None:
        return float(bar)
    # BlockedHandsLocoWrapper case: it sets ._env to the gym env (which itself owns .task = the wrapper)
    # so we need to check the wrapper's wrapped raw task class — look at task.__class__.__mro__[1]
    # for the underlying task class
    for parent in type(task).__mro__:
        bar = getattr(parent, "success_bar", None)
        if bar is not None:
            return float(bar)
    # fallback: task name → known bar table (HumanoidBench)
    task_name = getattr(env.spec, "id", "") if env.spec else ""
    known = {
        "walk": 700, "run": 700, "stand": 800, "sit_simple": 750, "sit_hard": 750,
        "crawl": 700, "pole": 700, "stair": 700, "slide": 700, "hurdle": 700,
        "balance_simple": 800, "balance_hard": 800, "maze": 1200, "reach": 12000,
    }
    for k, v in known.items():
        if f"-{k}-" in task_name:
            return float(v)
    return None


def build_policy(driver: str, task: str, seed: int, env: gym.Env, args) -> Callable[[np.ndarray], np.ndarray]:
    """Return act(obs)->action for the chosen driver."""
    act_space = env.action_space
    act_shape = act_space.shape

    if driver == "random":
        rng = np.random.default_rng(seed)
        low, high = act_space.low, act_space.high
        return lambda _obs: rng.uniform(low, high).astype(np.float32)

    if driver == "zero":
        zero = np.zeros(act_shape, dtype=np.float32)
        return lambda _obs: zero

    if driver == "reach":
        # In hierarchical mode the env's action space is high-level reach targets;
        # sample them randomly (low-level skill handles joint control).
        rng = np.random.default_rng(seed)
        low, high = act_space.low, act_space.high
        return lambda _obs: rng.uniform(low, high).astype(np.float32)

    if driver == "drq":
        import torch
        from drq_viewer import build_agent, resolve_ckpt
        ckpt_dir = resolve_ckpt(task, seed, auto_download=not args.no_download)
        device = torch.device(args.device)
        obs_dim = env.observation_space.shape[0]
        action_dim = act_shape[0]
        encoder, policy = build_agent(ckpt_dir, obs_dim, action_dim, device)

        @torch.no_grad()
        def act(obs: np.ndarray) -> np.ndarray:
            t = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
            zs = encoder.zs(t)
            return policy.act(zs).cpu().numpy().reshape(action_dim)

        return act

    sys.exit(f"❌ unknown --driver '{driver}'")


def run_episode(env, act, task: str, driver: str, seed: int, ep_idx: int,
                success_bar: float | None, dt: float, action_repeat: int = 1) -> EpisodeStats:
    obs, _ = env.reset(seed=seed + ep_idx)
    stats = EpisodeStats(task=task, driver=driver, seed=seed, ep_idx=ep_idx)
    t0 = time.time()
    success_step = None
    done = False
    while not done:
        action = act(obs)
        for _ in range(action_repeat):
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
            if term or trunc:
                stats.timed_out = bool(trunc and not term)
                done = True
                break
    if not stats.success and success_bar is not None and stats.ep_return >= success_bar:
        stats.success = True
    if success_step is not None:
        stats.time_to_success_s = success_step * dt
    stats.wall_time_s = time.time() - t0
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
    p.add_argument("--task", required=True)
    p.add_argument("--driver", choices=["random", "zero", "reach", "drq"], default="random")
    p.add_argument("--eval", dest="n_eval", type=int, default=10,
                   help="episodes per seed")
    p.add_argument("--seeds", type=int, default=3,
                   help="number of seeds when --seed_list not given")
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--seed_stride", type=int, default=1,
                   help="stride between auto-generated seeds (DR.Q uses 10)")
    p.add_argument("--seed_list", default=None,
                   help="comma-separated explicit seed list, overrides --seeds/--seed_start/--seed_stride")
    p.add_argument("--out", default=None, help="JSONL output path")
    # drq
    p.add_argument("--device", default=None)
    p.add_argument("--no_download", action="store_true")
    p.add_argument("--action_repeat", type=int, default=1,
                   help="repeat each action N times (DR.Q trained with 2)")
    # reach
    p.add_argument("--policy_path", default=None)
    p.add_argument("--mean_path", default=None)
    p.add_argument("--var_path", default=None)
    p.add_argument("--policy_type", default=None)
    args = p.parse_args()

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    out_path = Path(args.out) if args.out else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w")
    else:
        out_f = None

    if args.seed_list:
        seeds = [int(s) for s in args.seed_list.split(",")]
    else:
        seeds = [args.seed_start + i * args.seed_stride for i in range(args.seeds)]

    all_rows: list[EpisodeStats] = []
    overall_t0 = time.time()
    try:
        for seed in seeds:
            env = make_env(args.task, args.driver, args)
            dt = float(env.unwrapped.model.opt.timestep)
            success_bar = get_success_bar(env)
            act = build_policy(args.driver, args.task, seed, env, args)
            print(f"[seed {seed}] task={args.task} driver={args.driver} "
                  f"success_bar={success_bar} dt={dt:.4f}")
            action_repeat = args.action_repeat if args.driver == "drq" else 1
            for ep in range(args.n_eval):
                row = run_episode(env, act, args.task, args.driver, seed, ep,
                                  success_bar, dt, action_repeat=action_repeat)
                all_rows.append(row)
                if out_f:
                    out_f.write(json.dumps(asdict(row)) + "\n")
                    out_f.flush()
                tag = "✅" if row.success else ("⏱" if row.timed_out else "·")
                print(f"  {tag} ep{ep:02d} steps={row.ep_steps:4d} "
                      f"return={row.ep_return:8.2f} sub={row.subtasks_max} "
                      f"ttf={row.time_to_success_s} wall={row.wall_time_s:.1f}s")
            env.close()
    finally:
        agg = aggregate(all_rows)
        agg["task"] = args.task
        agg["driver"] = args.driver
        agg["seeds"] = seeds
        agg["eval_per_seed"] = args.n_eval
        agg["total_wall_s"] = time.time() - overall_t0
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
