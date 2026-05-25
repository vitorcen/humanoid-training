---
name: feedback-train-with-watcher
description: 长时训练必须分 slice + 后台 auto-eval + 早停 + 过/欠拟合检测 — 从 LeIsaac 项目学到的教训
metadata:
  type: feedback
---

任何超过 1 小时的训练任务，都必须**同时启动一个 watcher 后台进程**做分 slice 持续 eval，**不能开了训练就走人等结果**。

**Why:** 在 [[project-benchmark-validation]] 的 G1 自训阶段，用户引用 `/home/david/work/isaaclab-experience/LeIsaac/scripts/training/eval_watcher.sh` 的成熟做法。LeIsaac CLAUDE.md 记的血泪教训：DP v0.4.0 训了 10h 100k 步、loss 从 0.554 降到 0.011（看起来完美收敛），结果 eval 0/15 全 0——`crop_shape=(84,84)` 裁掉了 1.7% 关键画面，policy 学成了 "do nothing"。**一次 10k 步 quick eval 就能在 1h 内发现，省下 9h**。

**How to apply** — 启动任何 RL / SL 训练时同步部署：

### 1) 拆 slice
- 总步数拆 10 份：`SAVE_FREQ = total / 10`
- DR.Q 默认每 `eval_freq=5000` 步自己 eval 10 ep（比 LeIsaac 还密），可直接复用
- SL 训练（ACT / Diffusion Policy）则要手动 set `--save_freq=$((TOTAL/10))`

### 2) Watcher 后台进程（**必须从训练启动起就开**）
- DR.Q: `scripts/train_watcher.py`（本仓库已实现）— 读 DR.Q 的 evals/*.txt 流，输出：
  - `auto_eval.csv`: 一行一 slice（10%/20%/.../100%）
  - `eval_dense.csv`: 一行一 DR.Q eval（每 5000 步）
  - `auto_eval.status.json`: 实时状态供 shell 查询
  - `.eval_abort`: 早停 marker
- LeRobot 系: 复用 `LeIsaac/scripts/training/eval_watcher.sh`（poll checkpoints/ + 跑 X-VLA 3-round quick eval）

### 3) 三种诊断状态（重要！）
- **DEAD** = 连续 50 evals 全 < `success_bar/10` → policy 没学会动 → 立刻杀
- **UNDERFIT** = 最近 30 evals 最大值 ≤ 之前 30 的 1.05× → 学不动了，超参可能错
- **OVERFIT** = peak 已经过去 ≥20 eval 且 current < 0.7 × peak → catastrophic forgetting / dist shift
- **PROGRESS** = 还在爬升或近顶峰

注意 RL 的 "overfit" 不是 SL 意义上的过拟合，而是 **reward 见顶后回落**（buffer staleness、策略漂移、价值估计崩塌）。

### 4) 早停 marker → SIGTERM 训练
- watcher 检测到 DEAD 时 `touch .eval_abort`
- 训练 launcher（lerobot_finetune.sh 或本仓库的 wrapper）poll 这个 marker，SIGTERM 训练进程
- 默认行为应该是 **触发 abort 但不强杀**（`--dead_kills=False`）；需要时显式 opt-in，否则误判会丢工作

### 5) 一键查看脚本
- `scripts/train_status.sh <run_dir>` — 打印 live status + 10-slice milestones + ASCII reward 曲线
- ASCII 曲线用 unicode `█/·`，能在 ssh 终端直接看，不依赖 tensorboard

### 6) Tensorboard 也开但不作为唯一信号
- TB 看曲线方便，但**人不会守着浏览器**；watcher CSV + 早停 marker 才能在你睡觉时挡住烧钱

### 反模式 / Anti-patterns
- ❌ 启动训练就 `nohup` 走人，第二天回来才看结果
- ❌ 只看 train loss / reward，不在训练过程中做 eval
- ❌ 训练完了一次性大 eval，发现 0% 才知道白训
- ❌ 把 eval 嵌进训练主循环（拖慢训练速度，watcher 应该是独立进程读 ckpt / log）

### 文件位置
- 本仓库实现：`scripts/train_watcher.py` · `scripts/train_status.sh`
- 参考实现：`/home/david/work/isaaclab-experience/LeIsaac/scripts/training/eval_watcher.sh`
- LeIsaac 规则定义：`/home/david/work/isaaclab-experience/LeIsaac/CLAUDE.md` 第 5-9 行
