"""Minimal LeRobot G1 sim-replay that **handles nested action names**.

_Workaround for the lerobot 0.5.2 bug in `lerobot/scripts/lerobot_replay.py:122`
where `features[ACTION]["names"]` is iterated as flat strings, but DEX3-style
multi-modality datasets store it as `[[name1, name2, ...]]` (list-of-list).
Iterating that yields a list → `TypeError: unhashable type: 'list'`._

This script:
  1. Loads dataset via LeRobotDataset
  2. Auto-flattens one level of name nesting (no-op for flat schemas)
  3. Runs the same replay loop as lerobot-replay, but never crashes on dex3 schema

Usage (run with the lerobot env's python so unitree_sdk2py / loguru / mujoco are importable):

    DISPLAY=:0 ~/miniconda3/envs/lerobot/bin/python scripts/replay_g1.py \\
        --repo-id Breno-de-Angelo/unitree-g1-dex3-1-pick-kettle-v3 \\
        --episode 0 --fps 10
"""

import argparse
import sys
import time

from lerobot.configs.types import FeatureType  # noqa: F401  (force-load schema enums)
from lerobot.datasets import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots import make_robot_from_config
from lerobot.robots.unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_


# lerobot's UnitreeG1.publish_lowcmd ONLY writes joints whose key matches
# f"{G1_29_JointIndex(i).name}.q". Dex3 hand joints (kLeftHandThumb0 etc.)
# are NOT in G1_29_JointIndex — they need a separate DDS topic that
# lerobot's sim path does not yet implement (Nov 2025 — May 2026). So we
# publish body joints via UnitreeG1.publish_lowcmd and hand joints via the
# DEX3 hand DDS topics that the MuJoCo bridge already subscribes to.
G1_29_NAMES = {G1_29_JointIndex(i).name for i in range(29)}
DEX3_LEFT_NAMES = [
    "kLeftHandThumb0",
    "kLeftHandThumb1",
    "kLeftHandThumb2",
    "kLeftHandMiddle0",
    "kLeftHandMiddle1",
    "kLeftHandIndex0",
    "kLeftHandIndex1",
]
DEX3_RIGHT_NAMES = [
    "kRightHandThumb0",
    "kRightHandThumb1",
    "kRightHandThumb2",
    "kRightHandMiddle0",
    "kRightHandMiddle1",
    "kRightHandIndex0",
    "kRightHandIndex1",
]


def flatten_action_names(names):
    """One-level flatten. Returns flat list of strings.

    LeRobot v0.5 stores Dex3-style schemas as `[[a, b, ...]]` and flat schemas as `[a, b, ...]`.
    """
    if not names:
        return []
    if isinstance(names[0], list):
        flat = []
        for g in names:
            flat.extend(g if isinstance(g, list) else [g])
        return flat
    return list(names)


def normalize_action_key(name: str) -> str | None:
    """Convert dataset action key → lerobot G1 publish_lowcmd key, or None to drop.

    - `kLeftShoulderPitch`     → `kLeftShoulderPitch.q`   (publishes to sim)
    - `kLeftShoulderPitch.q`   → `kLeftShoulderPitch.q`   (already correct)
    - `kLeftHandThumb0`        → None                      (dex3 hand, sim publish unsupported)
    - `remote.lx`, `remote.ly` → `remote.lx`, ...          (controller stick, used by GR00T/Holosoma controllers)
    """
    if name.startswith("remote."):
        return name
    base = name[:-2] if name.endswith(".q") else name
    if base in G1_29_NAMES:
        return base + ".q"
    return None


def make_hand_publisher(topic: str):
    publisher = ChannelPublisher(topic, HandCmd_)
    publisher.Init()
    msg = unitree_hg_msg_dds__HandCmd_()
    for cmd in msg.motor_cmd:
        cmd.mode = 1
        cmd.kp = 1.5
        cmd.kd = 0.05
    return publisher, msg


def publish_handcmd(publisher, msg, values: list[float]) -> None:
    for cmd, q in zip(msg.motor_cmd, values, strict=True):
        cmd.q = float(q)
        cmd.dq = 0.0
        cmd.tau = 0.0
    publisher.Write(msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--fps", type=int, default=None, help="override; defaults to dataset.fps")
    p.add_argument("--diagnostics", action="store_true", help="print action/state ranges while replaying")
    args = p.parse_args()

    print(f"📦 Loading dataset: {args.repo_id} episode={args.episode}")
    ds = LeRobotDataset(args.repo_id, episodes=[args.episode])
    actions = ds.select_columns(ACTION)

    raw_names = flatten_action_names(ds.features[ACTION]["names"])
    # Map raw → publishable key (None means drop)
    mapped = [(i, raw, normalize_action_key(raw)) for i, raw in enumerate(raw_names)]
    publishable = [(i, raw, k) for (i, raw, k) in mapped if k is not None]
    left_hand = [(raw_names.index(name), name) for name in DEX3_LEFT_NAMES if name in raw_names]
    right_hand = [(raw_names.index(name), name) for name in DEX3_RIGHT_NAMES if name in raw_names]
    hand_indices = {i for i, _ in left_hand + right_hand}
    dropped = [(i, raw) for (i, raw, k) in mapped if k is None and i not in hand_indices]
    print(f"   action_dim = {ds.features[ACTION]['shape']}  ({len(raw_names)} raw names)")
    print(f"   publishable to G1 sim: {len(publishable)} / {len(raw_names)}")
    if left_hand or right_hand:
        print(f"   publishable to DEX3 hands: left={len(left_hand)} right={len(right_hand)}")
    if dropped:
        print(f"   dropped (no sim channel): {[r for _, r in dropped]}")
    print(f"   total frames = {ds.num_frames}, fps = {ds.fps}")

    print("🤖 Spawning UnitreeG1 in sim ...")
    robot = make_robot_from_config(UnitreeG1Config(is_simulation=True))
    robot.connect()
    left_hand_pub = left_hand_msg = right_hand_pub = right_hand_msg = None
    if left_hand:
        left_hand_pub, left_hand_msg = make_hand_publisher("rt/dex3/left/cmd")
    if right_hand:
        right_hand_pub, right_hand_msg = make_hand_publisher("rt/dex3/right/cmd")

    fps = args.fps or ds.fps
    processor = make_default_robot_action_processor()

    print(f"▶ Replaying {ds.num_frames} frames @ {fps} fps "
          f"(≈ {ds.num_frames / fps:.1f} s wall-clock)")
    try:
        for idx in range(ds.num_frames):
            t0 = time.perf_counter()
            arr = actions[idx][ACTION]
            # Only build dict entries for keys G1 sim actually consumes.
            action = {key: float(arr[i]) for (i, _raw, key) in publishable}
            obs = robot.get_observation()
            processed = processor((action, obs))
            robot.send_action(processed)
            if left_hand_pub is not None:
                publish_handcmd(left_hand_pub, left_hand_msg, [float(arr[i]) for i, _ in left_hand])
            if right_hand_pub is not None:
                publish_handcmd(right_hand_pub, right_hand_msg, [float(arr[i]) for i, _ in right_hand])
            precise_sleep(max(1 / fps - (time.perf_counter() - t0), 0.0))
            if idx % max(1, ds.num_frames // 20) == 0:
                if args.diagnostics:
                    ls = robot.get_observation()
                    print(
                        f"   step {idx}/{ds.num_frames} "
                        f"LSP={ls.get('kLeftShoulderPitch.q', float('nan')):.3f} "
                        f"RSP={ls.get('kRightShoulderPitch.q', float('nan')):.3f}",
                        flush=True,
                    )
                else:
                    print(f"   step {idx}/{ds.num_frames}", flush=True)
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        robot.disconnect()
        print("✅ done")


if __name__ == "__main__":
    main()
