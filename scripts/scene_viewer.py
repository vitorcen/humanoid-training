"""Open a HumanoidBench task in MuJoCo's native viewer at the initial state — **no policy, no rollout**.

_Pure scene preview: spawn env, reset, hand the underlying mjModel/mjData to mujoco.viewer.
Robot stays in its initial pose. Drag/zoom to inspect the scene._

Usage:
    python scripts/scene_viewer.py --task h1hand-walk-v0
    python scripts/scene_viewer.py --task g1-walk-v0 --seed 7
"""

import argparse
import os
import sys
from pathlib import Path

# Native window backend — must be set before mujoco imports anything else.
os.environ.setdefault("MUJOCO_GL", "glfw")

import gymnasium as gym
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import humanoid_bench  # noqa: F401  registers HumanoidBench gym envs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="e.g. h1hand-walk-v0, g1-cube-v0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = gym.make(args.task)
    env.reset(seed=args.seed)
    mj = env.unwrapped

    print(f"Scene preview: {args.task}  (seed={args.seed})")
    print("  Drag = rotate · Right-drag = pan · Scroll = zoom · Esc = quit")
    mujoco.viewer.launch(mj.model, mj.data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
