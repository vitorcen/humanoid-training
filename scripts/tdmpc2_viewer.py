"""Load a TD-MPC2 checkpoint and drive a HumanoidBench env in MuJoCo's native viewer.

_Counterpart to drq_viewer.py — same UX, different driver._

Usage:
    DISPLAY=:0 python scripts/tdmpc2_viewer.py \\
        --task humanoid_h1-walk-v0 \\
        --ckpt runs/h1_tdmpc2_pilot/logs/humanoid_h1-walk-v0/0/h1_tdmpc2_pilot/models/step_00800098.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Use GLFW for native viewer window (default — tdmpc2.envs.humanoid only sets egl
# if MUJOCO_GL is unset, so this preempts it).
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("LAZY_LEGACY_OP", "0")

import mujoco
import mujoco.viewer
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dependencies" / "humanoid-bench" / "tdmpc2"))
sys.path.insert(0, str(ROOT / "scripts"))

from tdmpc2_eval import load_cfg  # noqa: E402
from tdmpc2.envs import make_env  # noqa: E402
from tdmpc2.tdmpc2 import TDMPC2  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="humanoid_h1-walk-v0",
                   help="tdmpc2 task name (humanoid_h1-walk-v0, humanoid_g1-walk-v0, ...)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()

    cfg = load_cfg(args.task, args.seed)
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    agent.load(args.ckpt)
    print(f"📂 ckpt = {args.ckpt}")

    base = env.unwrapped
    model, data = base.model, base.data
    print(f"env={args.task}  obs_dim={cfg.obs_shape}  act_dim={cfg.action_dim}  "
          f"device={agent.device}")

    obs, _ = env.reset()
    dt = 1.0 / args.fps
    ep_return, ep_t, ep_num = 0.0, 0, 0
    t_step = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            action = agent.act(obs, t0=(t_step == 0), eval_mode=True)
            obs, reward, term, trunc, _ = env.step(action)
            ep_return += float(reward)
            ep_t += 1
            t_step += 1
            viewer.sync()
            if term or trunc:
                ep_num += 1
                print(f"  ep {ep_num} | steps={ep_t} | return={ep_return:.2f}")
                obs, _ = env.reset()
                ep_return, ep_t, t_step = 0.0, 0, 0
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
