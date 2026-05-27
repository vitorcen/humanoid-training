"""Launch lerobot sim replay (mode A) or Rerun dataset viz (mode B) on a HF G1 dataset.

_Locates the `lerobot-replay` / `lerobot-dataset-viz` CLI (PATH first, then known conda
envs), builds the argv, then `os.execvpe`'s into the CLI — the notebook button can
Popen this script and forget about it._

Usage:
    python scripts/dataset_preview.py --repo-id X --mode A --episode 0
    python scripts/dataset_preview.py --repo-id X --mode B --episode 5
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

LEROBOT_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"


def find_lerobot_bin(name: str) -> str | None:
    """PATH lookup first, then ~/miniconda3/envs/{lerobot, lerobot-v040}/bin/."""
    p = shutil.which(name)
    if p:
        return p
    for env in ("lerobot", "lerobot-v040"):
        cand = Path.home() / "miniconda3" / "envs" / env / "bin" / name
        if cand.is_file():
            return str(cand)
    return None


def preflight(repo_id: str, episode: int) -> int | None:
    """Sanity-check before exec'ing lerobot. Returns exit code on failure, None on OK."""
    info_p = LEROBOT_CACHE_ROOT / repo_id / "meta" / "info.json"
    if not info_p.is_file():
        print(f"❌ meta/info.json missing at {info_p}\n"
              f"   run: python scripts/dataset_download.py --repo-id {repo_id}",
              file=sys.stderr, flush=True)
        return 1
    info = json.loads(info_p.read_text())
    cv = info.get("codebase_version", "?")
    if not str(cv).startswith("v3"):
        print(f"❌ codebase_version={cv} — lerobot requires v3.x (this dataset is too old/new)",
              file=sys.stderr, flush=True)
        return 1
    n_ep = info.get("total_episodes")
    if isinstance(n_ep, int) and not (0 <= episode < n_ep):
        print(f"❌ episode {episode} out of range — dataset has {n_ep} episodes (valid 0..{n_ep-1})",
              file=sys.stderr, flush=True)
        return 1
    print(f"✓ preflight OK  repo={repo_id}  episode={episode}/{n_ep}  cv={cv}", flush=True)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--mode", choices=["A", "B"], required=True,
                    help="A=lerobot-replay (sim) · B=lerobot-dataset-viz (Rerun)")
    ap.add_argument("--episode", type=int, default=0)
    args = ap.parse_args()

    if (rc := preflight(args.repo_id, args.episode)) is not None:
        return rc

    if args.mode == "A":
        binary = find_lerobot_bin("lerobot-replay")
        if not binary:
            print("❌ lerobot-replay not found — install lerobot[unitree_g1]", file=sys.stderr)
            return 1
        cmd = [
            binary,
            "--robot.type=unitree_g1",
            "--robot.is_simulation=true",
            f"--dataset.repo_id={args.repo_id}",
            f"--dataset.episode={args.episode}",
        ]
    else:
        binary = find_lerobot_bin("lerobot-dataset-viz")
        if not binary:
            print("❌ lerobot-dataset-viz not found — install lerobot", file=sys.stderr)
            return 1
        cmd = [
            binary,
            f"--repo-id={args.repo_id}",
            f"--episode-index={args.episode}",
        ]

    # rerun-sdk's `rr.init(...)` internally fork+execs a separate `rerun` viewer
    # binary, looking for it on PATH. The notebook kernel runs in `humanoidbench`
    # env so child inherits that PATH and can't find `~/miniconda3/envs/lerobot/bin/rerun`.
    # Prepend the lerobot env's bin/ so the rerun spawn succeeds.
    bin_dir = str(Path(binary).parent)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "DISPLAY": os.environ.get("DISPLAY", ":0"),
        "PYTHONUNBUFFERED": "1",
    }
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    raise SystemExit(main())
