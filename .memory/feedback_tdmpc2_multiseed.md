---
name: tdmpc2-multiseed-parallel
description: "Run N independent TD-MPC2 seeds in N tmux sessions on the same GPU — natural GPU time-slicing pushes util 15% → 98%, near-zero per-seed slowdown"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 12be2311-45e9-45b0-9189-4b3c0c491e3e
---

When training TD-MPC2 (or any single-process single-env MuJoCo RL) on a beefy GPU that single-seed underutilizes, **spawn N independent seeds in N tmux sessions instead of patching the trainer for parallelism**. The OS gives you the parallel primitive for free.

**Why:** Single-seed TD-MPC2 alternates `CPU step (MuJoCo physics) → GPU plan (MPC)` with the GPU idle during physics — measured 15% GPU util on a 4080S 32 GB / 4090 24 GB. Three independent processes naturally time-slice the GPU (98% util) because their CPU/GPU phases interleave. Net: <b>2.7× total throughput with each seed only ~10% slower individually</b> (2026-05-27, cloud AutoDL 4080S, humanoid_g1-walk-v0). Also gives the multi-seed averages that RL papers actually report — single seed has no statistical meaning (DR.Q H1-walk measured ±40% return spread across seeds).

**How to apply:** Whenever the GPU is loaded < 30% by a single TD-MPC2 / DR.Q / SAC / similar process. Pair with [[parallel-jax-torch-gpu]] if mixing frameworks; pair with [[train-with-watcher]] for per-seed eval daemons (one watcher per seed).

## Recipe

```bash
for SEED in 0 10 20; do
  tmux new-session -d -s "g1train_s${SEED}" -x 220 -y 50
  tmux send-keys -t "g1train_s${SEED}" "
    cd /path/to/runs/<exp> &&
    MUJOCO_GL=egl conda run -n <env> --no-capture-output \
      python -m tdmpc2.train \
      task=<task> model_size=5 steps=1000000 eval_freq=50000 \
      seed=${SEED} exp_name=<exp> \
      data_dir=/tmp/_buf_s${SEED} \
      checkpoint=null disable_wandb=true save_agent=true save_csv=true \
      2>&1 | tee train_s${SEED}.log
  " Enter
done
```

## Hard constraints (each one is a trap that bit us)

- **`data_dir` per seed** — `/tmp/_buf_s0`, `_buf_s10`, … must be DIFFERENT directories. Sharing one corrupts statistical independence and triggers replay-buffer race conditions.
- **`MUJOCO_GL=egl`** — required in headless containers (e.g. AutoDL); otherwise `mujoco.FatalError: gladLoadGL`.
- **TD-MPC2 work_dir auto-separates seeds** — output goes to `logs/<task>/<seed>/<exp_name>/`, so `exp_name` can be the same across seeds; only `data_dir` needs to differ.
- **tmux ≠ SSH** — tmux session survives SSH disconnect by design; ssh reconnect + `tmux attach -t <name>` resumes view. No `nohup` needed.
- **`checkpoint=null` for fresh init** — strict paper-comparable multi-seed averaging requires all seeds fresh-from-random. If one seed resumes from a `.pt` checkpoint, mark it as fine-tune in the report — don't pool it into the mean blindly.

## When NOT to do this

- **CPU-bound runs hitting 100% on physics already** — adding more seeds won't help; you'll just thrash. Symptom: single-seed GPU util already > 70%.
- **GPU memory < 6 GB per process** — most workloads fit; check the heavy-tail (BC + huge encoders, DreamerV3 medium = ~10 GB).
- **More seeds than CPU cores / 3** — TD-MPC2's MuJoCo physics is essentially single-core per process. With 12 CPU cores in our cgroup, > 6 seeds starts thrashing.

## Measured net effect (humanoid_g1-walk-v0, 4080S 32 GB, 12 CPU cores)

| Config | sps/seed | total sps | GPU mem | GPU util | wall to 1M steps |
| --- | --- | --- | --- | --- | --- |
| 1 seed | 11 | 11 | 2 GB | 15% | ~25 h |
| 3 seeds parallel | ~10 | **30** | 6 GB | 98% | **~28 h** (each seed 1 M)|
| 3 seeds serial | 11 | 11 (one-at-a-time) | 2 GB | 15% | 75 h |

**Full write-up**: `docs/tdmpc2_multiseed_parallel.html` (含 SVG 时序图 + 启动配方 + 缺点列表).

Related: [[parallel-jax-torch-gpu]], [[train-with-watcher]], [[tmux-codex-invocation]].
