"""Load a DR.Q checkpoint from HF cache and drive a HumanoidBench env in MuJoCo's native viewer.

_Minimal eval-only loader — skips the 367 MB replay buffer required by DR.Q's full `Agent.load`._

Usage:
    python scripts/drq_viewer.py --task h1hand-walk-v0 --seed 0
    python scripts/drq_viewer.py --task h1-run-v0 --seed 10 --fps 30

The checkpoint is auto-resolved from the HuggingFace default cache:
    ~/.cache/huggingface/hub/models--dmux--DR.Q/snapshots/<sha>/DRQ+HBench-<task>+<seed>/

If not present, download with:
    hf download dmux/DR.Q \\
        --include 'DRQ+HBench-h1hand-walk-v0+0/policy.pt' \\
        --include 'DRQ+HBench-h1hand-walk-v0+0/encoder.pt' \\
        --include 'DRQ+HBench-h1hand-walk-v0+0/agent_var.npy'
"""

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import torch

# Make DR.Q importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dependencies" / "dr-q" / "DRQ"))

import models  # noqa: E402  from dependencies/dr-q/DRQ/models.py
import DRQ as _drq_module  # noqa: E402  has Hyperparameters
# DR.Q's published checkpoints were saved when the repo was still named "REP".
# The pickled Hyperparameters reference module "REP", so alias it to "DRQ" here.
sys.modules.setdefault("REP", _drq_module)

import humanoid_bench  # noqa: F401,E402  registers gym envs


CRITICAL_FILES = ("policy.pt", "encoder.pt", "agent_var.npy")


def resolve_ckpt(task: str, seed: int, auto_download: bool = True) -> Path:
    """Locate DRQ+HBench-<task>+<seed>/ inside HF cache.

    If missing and auto_download=True, pull the 3 critical files (~13 MB) on demand.
    Skips the 367 MB replay buffer and *_target / *_optimizer .pt files (training-only).
    """
    project = f"DRQ+HBench-{task}+{seed}"
    cache_root = Path.home() / ".cache/huggingface/hub/models--dmux--DR.Q/snapshots"

    def _existing() -> Path | None:
        if not cache_root.is_dir():
            return None
        for snap in cache_root.iterdir():
            cand = snap / project
            if cand.is_dir() and all((cand / f).is_file() for f in CRITICAL_FILES):
                return cand
        return None

    found = _existing()
    if found is not None:
        return found

    if not auto_download:
        sys.exit(
            f"❌ '{project}' not present under {cache_root}\n"
            f"   Run with --auto_download (default) or:\n"
            f"   hf download dmux/DR.Q "
            + " ".join(f"--include '{project}/{f}'" for f in CRITICAL_FILES)
        )

    print(f"📥 ckpt '{project}' not in cache — fetching critical files (~13 MB) ...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="dmux/DR.Q",
        allow_patterns=[f"{project}/{f}" for f in CRITICAL_FILES],
    )
    found = _existing()
    if found is None:
        sys.exit(
            f"❌ download finished but '{project}' still not resolvable under {cache_root}.\n"
            f"   The task/seed combo likely does not exist on the Hub. Check "
            f"https://huggingface.co/dmux/DR.Q/tree/main"
        )
    print(f"✅ ckpt ready at {found}")
    return found


def build_agent(ckpt_dir: Path, obs_dim: int, action_dim: int, device: torch.device):
    """Construct minimal Encoder + Policy and load weights. Skip buffer / optimizers / targets."""
    var = np.load(ckpt_dir / "agent_var.npy", allow_pickle=True).item()
    hp = var["hp"]  # DR.Q Hyperparameters dataclass instance

    encoder = models.Encoder(
        state_dim=obs_dim, action_dim=action_dim, pixel_obs=False,
        num_bins=hp.num_bins, zs_dim=hp.zs_dim, za_dim=hp.za_dim, zsa_dim=hp.zsa_dim,
        hdim=hp.enc_hdim, activ=hp.enc_activ,
    ).to(device).eval()
    encoder.load_state_dict(torch.load(ckpt_dir / "encoder.pt", map_location=device, weights_only=True))

    policy = models.Policy(
        action_dim=action_dim, discrete=False,
        gumbel_tau=hp.gumbel_tau, zs_dim=hp.zs_dim,
        hdim=hp.policy_hdim, activ=hp.policy_activ,
    ).to(device).eval()
    policy.load_state_dict(torch.load(ckpt_dir / "policy.pt", map_location=device, weights_only=True))

    return encoder, policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="h1hand-walk-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--action_repeat", type=int, default=2,
                   help="HBenchPreprocessing default = 2; keep aligned with training")
    p.add_argument("--no_download", action="store_true",
                   help="fail if ckpt missing instead of auto-fetching from HF Hub")
    p.add_argument("--ckpt_dir", default=None,
                   help="explicit path to DRQ+HBench-<task>+<seed>/ — bypass cache lookup. "
                        "Use this to load ckpts from non-dmux repos (e.g. wsagi/HumanoidBench-DR.Q).")
    args = p.parse_args()

    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
        missing = [f for f in CRITICAL_FILES if not (ckpt_dir / f).is_file()]
        if missing:
            sys.exit(f"❌ --ckpt_dir {ckpt_dir} missing required files: {missing}")
    else:
        ckpt_dir = resolve_ckpt(args.task, args.seed, auto_download=not args.no_download)
    print(f"📂 ckpt = {ckpt_dir}")

    env = gym.make(args.task)
    env.reset(seed=args.seed)
    base = env.unwrapped
    model, data = base.model, base.data
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"env={args.task}  obs_dim={obs_dim}  act_dim={action_dim}  device={args.device}")

    device = torch.device(args.device)
    encoder, policy = build_agent(ckpt_dir, obs_dim, action_dim, device)

    @torch.no_grad()
    def act(state_np: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(state_np, dtype=torch.float32, device=device).reshape(1, -1)
        zs = encoder.zs(state)
        return policy.act(zs).cpu().numpy().reshape(action_dim)

    obs, _ = env.reset(seed=args.seed)
    dt = 1.0 / args.fps
    ep_return, ep_t, ep_num = 0.0, 0, 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            action = act(obs)
            for _ in range(args.action_repeat):
                obs, reward, term, trunc, _ = env.step(action)
                ep_return += float(reward)
                ep_t += 1
                if term or trunc:
                    break
            viewer.sync()
            if term or trunc:
                ep_num += 1
                print(f"  ep {ep_num} | steps={ep_t} | return={ep_return:.2f}")
                obs, _ = env.reset()
                ep_return, ep_t = 0.0, 0
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
