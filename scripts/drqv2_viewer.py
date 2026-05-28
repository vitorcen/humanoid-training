"""DrQ-v2 GUI viewer — load snapshot.pt, run policy with live mujoco window.

Usage:
    DISPLAY=:0 conda run -n humanoidbench --no-capture-output \\
        python scripts/drqv2_viewer.py \\
        --task humanoid_walk \\
        --ckpt runs/drqv2_west_pull/frame_1300000.pt \\
        --seed 0
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "glfw")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dependencies/drqv2"))

import dmc

import mujoco
import mujoco.viewer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="humanoid_walk", help="dm_control suite task")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--action-repeat", type=int, default=2)
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=1000)
    args = p.parse_args()

    print(f"📂 ckpt = {args.ckpt}")
    print(f"🎯 task = {args.task}  (frame_stack={args.frame_stack}, action_repeat={args.action_repeat})")

    env = dmc.make(args.task, args.frame_stack, args.action_repeat, args.seed)
    print(f"env obs_spec: {env.observation_spec()}")
    print(f"env act_spec: {env.action_spec()}")

    snapshot = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(snapshot, dict) and "agent" in snapshot:
        agent = snapshot["agent"]
    else:
        agent = snapshot
    print(f"📦 agent = {type(agent).__name__}")
    agent.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for net_name in ["encoder", "actor", "critic", "critic_target"]:
        net = getattr(agent, net_name, None)
        if net is not None:
            net.to(agent.device)
    print(f"   device = {agent.device}")

    inner = env
    while not hasattr(inner, "physics"):
        inner = inner._env
    physics = inner.physics
    model = physics.model.ptr
    data = physics.data.ptr

    print("\n🎬 launching mujoco viewer ...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(0.5)
        for ep in range(1, args.num_episodes + 1):
            time_step = env.reset()
            ep_return = 0.0
            ep_steps = 0
            t0 = time.time()
            while not time_step.last() and ep_steps < args.max_steps:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                    obs = time_step.observation
                    action = agent.act(obs, step=1_000_000, eval_mode=True)
                time_step = env.step(action)
                ep_return += float(time_step.reward or 0.0)
                ep_steps += 1
                viewer.sync()
                if not viewer.is_running():
                    print("viewer closed by user, exiting")
                    return
                target_dt = 0.04
                dt = time.time() - t0
                if dt < target_dt * ep_steps:
                    time.sleep(target_dt * ep_steps - dt)
            print(f"  ep {ep}  steps={ep_steps}  return={ep_return:.2f}  "
                  f"wall={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
