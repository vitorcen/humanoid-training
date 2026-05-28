#!/usr/bin/env bash
# Set up conda env `humanoidbench-jax` for DreamerV3 + HumanoidBench (JAX-only).
#
# _Idempotent: re-running just re-pins versions._
#
# Why these exact pins (after 5+ failed combos diagnosed by codex gpt-5.5 2026-05-27):
#   - jax/jaxlib 0.4.30 + cudnn 9.1.0.70 (9.22 segfaults on this hardware)
#   - nvidia-cuda-nvcc-cu12 12.1.105 (12.9 ptxas crashes)
#   - numpy 2.1.3 (2.4 hits matplotlib NoneType bug; 1.x is too old for jax 0.4.30)
#   - matplotlib 3.9.4 (3.8.x + 3.9.0-3.9.3 have "duplicate parameter name"
#     ValueError on Py 3.11 inside dreamerv3's import chain; 3.7 forces numpy<2)
#   - Pillow 10.4.0 (12.x has ExifTags enum bug on Py 3.11)
#   - scipy 1.13.1 (1.14+ FunctionDoc/_docscrape deepcopy regression breaks
#     tensorflow_probability lazy scipy.special import in subprocess spawn)
#   - tensorflow REMOVED — dreamerv3 only needs tfp via JAX substrate
#   - torch REMOVED — humanoid_bench's flax_to_torch lazy-import patch makes it optional
#
# Prereq: NVIDIA driver supporting CUDA 12.x (verify with `nvidia-smi`).
set -euo pipefail

ENV_NAME="${ENV_NAME:-humanoidbench-jax}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> [1/5] ensure conda env '$ENV_NAME' exists with python 3.11"
if ! conda env list | grep -qE "^$ENV_NAME\s"; then
    conda create -n "$ENV_NAME" python=3.11 -y
fi

PIP="conda run -n $ENV_NAME --no-capture-output pip"

echo "==> [2/5] install JAX 0.4.30 + cuda12 plugin (no deps to avoid version drift)"
$PIP install --force-reinstall --no-deps \
    "jax==0.4.30" "jaxlib==0.4.30" "jax-cuda12-pjrt==0.4.30" "jax-cuda12-plugin==0.4.30"

echo "==> [3/5] pin NVIDIA CUDA libs (avoid 12.9 nvcc ptxas crash, force cudnn 9.1)"
$PIP install --force-reinstall --no-deps \
    "nvidia-cuda-runtime-cu12==12.1.105" \
    "nvidia-cuda-nvrtc-cu12==12.1.105" \
    "nvidia-cuda-cupti-cu12==12.1.105" \
    "nvidia-cuda-nvcc-cu12==12.1.105" \
    "nvidia-cuda-runtime-cu12==12.1.105" \
    "nvidia-cublas-cu12==12.1.3.1" \
    "nvidia-cusolver-cu12==11.4.5.107" \
    "nvidia-cusparse-cu12==12.1.0.106" \
    "nvidia-cufft-cu12==11.0.2.54" \
    "nvidia-curand-cu12==10.3.2.106" \
    "nvidia-cudnn-cu12==9.1.0.70" \
    "nvidia-nccl-cu12==2.20.5" \
    "nvidia-nvjitlink-cu12==12.9.86" \
    "nvidia-nvtx-cu12==12.1.105"

echo "==> [4/5] install DreamerV3 deps (no tensorflow, no torch — see header note)"
# Strip out the bad `--editable dreamerv3` line in upstream requirements + drop tensorflow.
grep -v -E '^(dreamerv3|tensorflow|--editable)' \
    "$REPO_ROOT/dependencies/humanoid-bench/requirements_dreamer.txt" \
    > /tmp/_req_dreamer_clean.txt
$PIP install -r /tmp/_req_dreamer_clean.txt
# Override versions that requirements_dreamer.txt's deps mis-pull.
$PIP install --force-reinstall --no-deps \
    "matplotlib==3.10.0" "numpy==2.1.3" "ml-dtypes==0.5.4" "Pillow==10.4.0" \
    "scipy==1.13.1" \
    "tensorflow-probability==0.24.0"

echo "==> [5/5] editable install humanoid_bench + dreamerv3"
$PIP install -e "$REPO_ROOT/dependencies/humanoid-bench" \
              -e "$REPO_ROOT/dependencies/humanoid-bench/dreamerv3" --no-deps

echo ""
echo "==> sanity check"
conda run -n "$ENV_NAME" --no-capture-output python -c "
import jax; assert jax.devices()[0].platform == 'cuda', f'JAX not on CUDA: {jax.devices()}'
print(f'jax {jax.__version__} on {jax.devices()}')
import humanoid_bench, gymnasium as gym
env = gym.make('h1hand-window-v0')
print(f'env obs={env.observation_space.shape} act={env.action_space.shape}')
import embodied
print('embodied (dreamerv3) imported OK')
print('✅ env ready for: python -m embodied.agents.dreamerv3.train ...')
"
