# patches

Submodule 本地补丁集中存放点。

_Local patches against pinned submodules — apply after `git submodule update`._

| Patch | 目标 / Target | 作用 / What |
| --- | --- | --- |
| `dr-q-save-agent.patch` | `dependencies/dr-q/` | 让 DR.Q 训练真正保存 agent 权重（上游默认注释掉了） · skip 367 MB replay buffer dump |
| `g1-pos-control.patch` | `dependencies/humanoid-bench/` | 把 G1 motor → PD position 控制（参照 H1 配置），让 DR.Q sample-efficiency 与 H1 同量级。生成器：`patches/build_g1_pos.py` |
| `humanoid-bench-g1-and-lazy.patch` | `dependencies/humanoid-bench/` | 合并补丁：(1) `BlockedHandsLocoWrapper` 支持 G1（act 37→23，屏蔽 14 维手指），`__init__.py` 自动给 g1-* task 启用；(2) lazy-import torch in `wrappers.py` + `envs/{push,package,reach}.py` 让 `humanoidbench-jax` (JAX-only) env 不装 torch 也能 `gym.make` |
| `dreamerv3-logger-lazy-mpl.patch` | `dependencies/humanoid-bench/` | DreamerV3 的 `logger.py` 改 lazy matplotlib + Agg backend + try/except，绕过 mpl 3.7-3.10 + Python 3.11 在 multiprocessing.spawn 子进程里的 "duplicate parameter name" bug |
| `tdmpc2-save-agent.patch` | `dependencies/humanoid-bench/tdmpc2/` | 上游 logger 在 wandb 禁用时把 `save_agent` 一刀切关掉；本补丁保留 flag + 在 `online_trainer.py` 每 `eval_freq` 步落盘 `step_XXXXXXXX.pt`，配合 `scripts/ckpt_eval_loop.py` 做 slice-based auto-eval |

## 应用 / Apply

```bash
bash patches/apply.sh
```

## 还原 / Revert

```bash
cd dependencies/dr-q && git checkout -- DRQ/ && cd -
```

## DreamerV3 conda env

JAX-only env `humanoidbench-jax` (separate from `humanoidbench` to avoid torch↔jax cudnn conflict). One-shot install:

```bash
bash patches/dreamerv3-env-setup.sh
```

See the script header for the rationale behind each version pin (5 failed combos diagnosed before landing this set).
