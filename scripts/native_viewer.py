"""Launch a HumanoidBench env in MuJoCo's native interactive viewer.

Usage:
    # 随机动作 / Random action
    python scripts/native_viewer.py --env h1hand-walk-v0

    # 零动作（看初始姿态）/ Zero action
    python scripts/native_viewer.py --env h1hand-stand-v0 --action zero

    # 官方 reach 低层技能 (hierarchical) / Built-in reach skill (hierarchical)
    python scripts/native_viewer.py --env h1hand-push-v0 \
        --policy_path dependencies/humanoid-bench/data/reach_two_hands/torch_model.pt \
        --mean_path   dependencies/humanoid-bench/data/reach_two_hands/mean.npy \
        --var_path    dependencies/humanoid-bench/data/reach_two_hands/var.npy \
        --policy_type reach_double_relative

Controls (MuJoCo viewer):
    Left-drag      rotate camera
    Right-drag     pan
    Scroll         zoom
    Space          pause / resume
    Esc / close    quit
"""

import argparse
import sys
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

import humanoid_bench  # noqa: F401  registers gym envs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="h1hand-walk-v0")
    p.add_argument("--action", choices=["random", "zero"], default="random",
                   help="action driver when no hierarchical policy is set")
    p.add_argument("--fps", type=float, default=30.0,
                   help="visualization target fps; physics still steps at env dt")
    # hierarchical reach-skill kwargs — passed straight to gym.make
    p.add_argument("--policy_path", default=None,
                   help="path to torch low-level skill checkpoint (.pt)")
    p.add_argument("--mean_path", default=None,
                   help="path to obs-mean normalization (.npy)")
    p.add_argument("--var_path", default=None,
                   help="path to obs-var normalization (.npy)")
    p.add_argument("--policy_type", default=None,
                   choices=[None, "flat", "reach_single",
                            "reach_double_absolute", "reach_double_relative"],
                   help="enable hierarchical wrapper inside the env")
    args = p.parse_args()

    env_kwargs = {}
    if args.policy_type and args.policy_type != "flat":
        assert args.policy_path, "--policy_path required when --policy_type is set"
        env_kwargs.update(
            policy_path=args.policy_path,
            mean_path=args.mean_path,
            var_path=args.var_path,
            policy_type=args.policy_type,
        )

    try:
        env = gym.make(args.env, **env_kwargs)
    except AttributeError as e:
        if "htarget_low" in str(e) and env_kwargs:
            sys.exit(
                f"❌ env '{args.env}' 不支持 hierarchical reach wrapper "
                f"(task class 没定义 htarget_low/high).\n"
                f"   '{args.env}' does not declare htarget_low/htarget_high — reach wrapper不可用.\n"
                f"   支持 reach_double_relative 的 task / Supported tasks:\n"
                f"     reach · walk · run · stand · sit · balance · stair · slide · hurdle · crawl\n"
                f"     maze · pole · package · truck · bookshelf · push\n"
                f"   去掉 --policy_type/--policy_path 即可用 random/zero 看场景."
            )
        raise
    env.reset()
    base = env.unwrapped
    model, data = base.model, base.data

    mode = "hierarchical" if env_kwargs else args.action
    print(f"env={args.env}  obs_dim={env.observation_space.shape}  "
          f"act_dim={env.action_space.shape}  dt={model.opt.timestep}  driver={mode}")

    dt = 1.0 / args.fps
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            if args.action == "zero" and not env_kwargs:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                # hierarchical mode also gets random high-level targets here,
                # which sends arms toward random reachable points
                action = env.action_space.sample()
            _, _, term, trunc, _ = env.step(action)
            viewer.sync()
            if term or trunc:
                env.reset()
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
