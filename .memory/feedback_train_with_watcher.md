---
name: feedback-train-with-watcher
description: 长时训练必须分 slice + 后台 auto-eval + 早停 + 过/欠拟合检测 — 从 LeIsaac 项目学到的教训
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 12be2311-45e9-45b0-9189-4b3c0c491e3e
---

任何超过 1 小时的训练任务，都必须**同时启动一个 watcher 后台进程**做分 slice 持续 eval，**不能开了训练就走人等结果**。**Mandatory**——launcher 脚本不挂 watcher 就不算"启动训练"，是 bug，要补。

**Why:** 在 [[project-benchmark-validation]] 的 G1 自训阶段，用户引用 `/home/david/work/isaaclab-experience/LeIsaac/scripts/training/eval_watcher.sh` 的成熟做法。LeIsaac CLAUDE.md 记的血泪教训：DP v0.4.0 训了 10h 100k 步、loss 从 0.554 降到 0.011（看起来完美收敛），结果 eval 0/15 全 0——`crop_shape=(84,84)` 裁掉了 1.7% 关键画面，policy 学成了 "do nothing"。**一次 10k 步 quick eval 就能在 1h 内发现，省下 9h**。

第二个血泪教训（2026-05-27）：本机 4090 跑 DreamerV3 small h1hand-window-v0 1M 步，**没挂 watcher**，理由是"小试一下 manipulation gap"。结果训完才知道 ep score 5–9 / success_bar=650 → success_rate≈0%，**6h+ wall time 烧光，零可解释信号**。如果挂了 watcher，250k step 时就该看出 DEAD（score 长期 < 65）→ 提前 4.5h 杀掉换策略。**"探路式实验"不是不挂 watcher 的理由，反而更需要——失败要快**。

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

### 6) Ckpt 保留策略：稀疏 cp 保留 5–30 份

训练算法对 ckpt 处理千差万别——有的每 step 存独立文件（TD-MPC2 `step_<N>.pt`、DR.Q），有的只滚动覆盖一个（DrQ-v2/DreamerV3 `snapshot.pt`/`checkpoint.ckpt`）。**统一规则：每个训练最终留 5–30 份 ckpt，按模型大小 + 磁盘空间灵活选**。

| 模型大小 / ckpt 单份 | 推荐份数 | 总占用 |
|---|---|---|
| < 100MB（TD-MPC2 small, DR.Q） | 20–30 份 | < 3 GB |
| 100–500MB（DrQ-v2, DreamerV3 small） | 10–20 份 | 1–10 GB |
| > 500MB（DreamerV3 medium / pixel + replay buffer） | 5–10 份 | 看盘 |

**Why:** 5 份下界是为了能看出 learning curve 形状（peak/plateau/dive 至少要 3 个点）；30 份上界是 AutoDL 50GB 数据盘的实际约束（× 100MB = 3 GB OK，× 500MB = 15 GB 占大头）。**关键产物是曲线，不是 ckpt 本身**——所以宁可稀疏 + 完整覆盖训练全程，也别密集只覆盖中间。

**实现两种**：

1. **自带 step-by-step ckpt 的算法**（TD-MPC2 / DR.Q）：watcher 直接消费，无需 cp。设 `eval_freq = total_steps / 25` 即可。
2. **滚动覆盖的算法**（DrQ-v2 `snapshot.pt`、DreamerV3 `checkpoint.ckpt`）：必须起独立 **cp daemon**，按 step / frame 桶触发：
   - 桶大小 = `total_steps / 25`（DrQ-v2 humanoid 3M frame → 桶 120k，约 25 份）
   - 触发源是训练 eval log（eval.csv / metrics.jsonl），桶号跨边界时 `cp snapshot.pt snapshots/<step>.pt`
   - 实现见 `/tmp/drqv2_snap_copy.sh`（独立 bash 脚本 + tmux session，已在 west 验证）

### 7) 显存协调 / VRAM-aware eval queuing（**仅限本机 train + eval 共卡场景**）

**适用边界**：watcher 跟 train 抢**同一张 GPU** 时。云端 AutoDL 每个 VM 一张卡、watcher 在 laptop 上跑或者根本不挂 watcher（cp daemon 只搬文件不吃显存）——这种**不适用本节**。

本机一张 4090/4080 同时 train + eval 才需要协调：

- **训练 VRAM < 60% 卡容量** → watcher 直接并行起 eval 子进程（DR.Q on 4090 24GB：train 2GB + eval 2GB，富余很多）
- **训练 60–80%** → watcher 跑 **轻量 eval（N=3 ep, action_repeat=2）** 拿个 noisy 但能区分 DEAD/PROGRESS 的信号
- **训练 ≥80%（DreamerV3 medium / 多 seed TD-MPC2 / pixel RL）** → watcher 进入 **deferred / 续训模式**：
  1. 不并行跑 eval
  2. 只 cp ckpt 到 `pending_eval/` 队列（mtime + step 命名）
  3. 训练结束 / 训练进程退出后，watcher 自动消费 queue，按 step 顺序回放跑 N=10 eval
  4. 这就是"间隔 eval + 续训"——eval 让出显存让训练跑完，然后续跑 eval

**Why:** 本机 4090 24GB 跑 DreamerV3 medium + JAX prealloc 会吃光显存，并行 eval 会 OOM 杀训练。**deferred 模式让 eval 不影响训练 throughput**，代价是失去实时观察。云端 4080S 32GB 跑 3 seed TD-MPC2 GPU util 98% / mem 6GB 算并行模式，不算 deferred。

接力 caveat: 当前云端 DrQ-v2 / DreamerV3 没挂 watcher 已经在跑——retrofit 方案是 **训练结束后用 ckpt_eval_loop 一次性扫所有 saved ckpt**（前提是有 cp daemon 保留了多份 snapshot；否则只能 eval 最终那个）。

### 8) 过/欠拟合监测的实操产物

watcher 实时维护两个文件让人能"瞟一眼就懂"：

- `eval_history.jsonl` — 每行一次 eval：`{step, eval_return, success_rate, n_eps, status: DEAD|UNDERFIT|OVERFIT|PROGRESS, wall_time_s}`
- `learning_curve.txt` — ASCII sparkline，每 5 行刷新，`step  return  ████░░░░  PROGRESS`

`scripts/train_status.sh <run_dir>` 把 jsonl 转 curve 并算当前 status。这是 SSH 终端唯一不可替代的可视化（tensorboard 需要浏览器 + port forward）。

### 9) Tensorboard 也开但不作为唯一信号
- TB 看曲线方便，但**人不会守着浏览器**；watcher CSV + 早停 marker 才能在你睡觉时挡住烧钱

### 反模式 / Anti-patterns
- ❌ 启动训练就 `nohup` 走人，第二天回来才看结果
- ❌ 只看 train loss / reward，不在训练过程中做 eval
- ❌ 训练完了一次性大 eval，发现 0% 才知道白训
- ❌ 把 eval 嵌进训练主循环（拖慢训练速度，watcher 应该是独立进程读 ckpt / log）
- ❌ "探路式实验所以省点事不挂 watcher"——错，正因为不确定才更要早杀
- ❌ 训练时 OOM-killer 杀进程比 eval 错过几次更糟——显存吃紧时用 deferred queue 不要硬上
- ❌ 滚动覆盖 ckpt 的算法（DrQ-v2/DreamerV3）不挂 cp daemon——训完只有最后那一份，OVERFIT 时没法回退到 peak
- ❌ 把 deferred 模式套到云端独立卡场景——云端 train + watcher 各张卡，本来不抢显存，硬上 deferred 反而丢实时性

### 文件位置
- 本仓库实现：
  - `scripts/launch_train.sh` — 统一 launcher，强制挂 watcher（mandatory wrapper）
  - `scripts/train_watcher.py` — DR.Q watcher
  - `scripts/ckpt_eval_loop_tdmpc2.py` — TD-MPC2 ckpt 队列 eval daemon
  - `scripts/ckpt_eval_loop_dreamerv3.py` — DreamerV3 rolling-ckpt snapshot + VRAM-aware deferred + DEAD/UNDERFIT/OVERFIT 分类
  - `scripts/snap_copy.sh` — 滚动 snapshot 稀疏 cp daemon（bash + tmux 模板，参数: run_dir + bucket_size；可移植到任何 single-rolling-ckpt 算法）
  - `scripts/train_status.sh` — live status + ASCII curve
- 参考实现：`/home/david/work/isaaclab-experience/LeIsaac/scripts/training/eval_watcher.sh`
- LeIsaac 规则定义：`/home/david/work/isaaclab-experience/LeIsaac/CLAUDE.md` 第 5-9 行

### 接力时的快速校验

启动训练前 / 接手他人的训练时，先 grep：

```bash
ps -ef | grep -E "watcher|ckpt_eval_loop" | grep -v grep
```

如果训练 PID 在跑但**没有**对应的 watcher PID，立刻：
1. retrofit 一个 watcher（poll ckpt dir + 跑 deterministic N=3 eval per new ckpt）
2. 在 [[project-benchmark-validation]] 的"当前并行训练"段标 ⚠️ no-watcher，提醒接力人 ckpt 没 live 评估
