"""Download a HuggingFace dataset to LeRobot's expected cache path.

_LeRobot reads datasets from `~/.cache/huggingface/lerobot/<repo_id>/` (direct files
with `meta/info.json` at root), **not** the standard HF hub cache layout
(`datasets--ORG--NAME/snapshots/<sha>/...`). If we use `snapshot_download` with no
`local_dir`, files land in hub cache and `LeRobotDataset(repo_id)` later fails with
`FileNotFoundError: meta/info.json`. So pin `local_dir` to lerobot's path._

_Also wipes `<repo>/.cache/` after download — that subdir is HF datasets'
arrow cache; a previous failed `lerobot-dataset-viz` run can leave an empty
cache that makes subsequent loads fail with `Instruction "train" corresponds to no data`._

Usage:
    python scripts/dataset_download.py --repo-id jnsungp/unitree-g1-...
"""

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# LeRobot's hard-coded cache root (see lerobot.datasets.constants.HF_LEROBOT_HOME).
LEROBOT_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--repo-type", default="dataset")
    args = ap.parse_args()

    local_dir = LEROBOT_CACHE_ROOT / args.repo_id
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"▶ snapshot_download  repo={args.repo_id}  →  {local_dir}", flush=True)
    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_dir=str(local_dir),
        )
    except Exception as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1

    info = local_dir / "meta" / "info.json"
    print(f"✅ cached at: {local_dir}", flush=True)
    print(f"   meta/info.json {'exists' if info.is_file() else 'MISSING'}", flush=True)

    # Wipe HF datasets' arrow cache to avoid "Instruction train corresponds to no data"
    # from a stale failed-load cache. The cache rebuilds on next preview (1-2 s overhead).
    stale_cache = local_dir / ".cache"
    if stale_cache.is_dir():
        shutil.rmtree(stale_cache, ignore_errors=True)
        print(f"   wiped stale arrow cache: {stale_cache}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
