---
name: project-benchmark-validation
description: HumanoidBench 验证目标 — 让若干任务 multi-run eval ≥50% 成功率，按 brainstorm HTML 的 8 路径优先级实战
metadata:
  type: project
---

## 验证目标 / Acceptance criteria

跑出至少 **3 个 HumanoidBench 任务**的 multi-run eval，**success_rate ≥ 50%** (N≥10 episodes / 3 seeds)。
"Success" 用 task class 自带信号：
- `info['success']==True` 任意一步出现（cabinet / kitchen 有）
- 或 `ep_return ≥ task.success_bar`（其他任务）

## 起点状态（compact 前已确认）/ State at compact

| 项 | 状态 | 文件 |
|---|---|---|
| conda env `humanoidbench` (Python 3.11, mujoco 3.1.6, torch 2.3.1) | ✅ | `~/miniconda3/envs/humanoidbench/` |
| humanoid-bench submodule | ✅ | `dependencies/humanoid-bench/` |
| DR.Q submodule | ✅ | `dependencies/dr-q/` |
| `scripts/native_viewer.py` (random/zero/reach skill, +friendly error for unsupported task) | ✅ | + 任务白名单 |
| `scripts/drq_viewer.py` (auto-download DR.Q ckpt to HF cache, minimal eval loader) | ✅ | 实测 `h1hand-walk` ep1 return=557 ∈ 论文 [371,652] |
| `HumanoidBench.ipynb` (53 cells, 两轴预览) | ✅ | + bilingual style |
| Brainstorm HTML（Opus + Codex 三方）| ✅ | `docs/manipulation_policy_brainstorm.html` |
| DR.Q ckpt cache | ✅ | `~/.cache/huggingface/hub/models--dmux--DR.Q/snapshots/.../DRQ+HBench-h1hand-walk-v0+0/` |
| `scripts/eval.py`（多 ep + JSONL 输出 + 聚合统计） | ❌ 未写 | TODO #1 |
| Multi-seed multi-run benchmark 数据 | ❌ 未跑 | TODO #2 |

**Why:** compact 后会丢具体 in-progress 状态，留这个 memory 让下一轮 Claude 能立即知道"从哪一步接"。

## 验证执行顺序（按 brainstorm HTML §5 的 ROI 排序）

**How to apply** — 接力时按这个顺序，每步跑完更新本 memory 的 "进度" 段：

### Step 1 — 实现 eval 工具（先有秤再称重）
- 写 `scripts/eval.py`：扩 `native_viewer.py` 和 `drq_viewer.py` 的 episode loop，加 `--eval N --seeds S --out results/<task>.jsonl --no-render` flag
- 输出 per-episode 行：`{task, seed, ep_steps, ep_return, success, subtasks_max, time_to_success_s, timed_out, wall_time_s}`
- 末尾聚合：success_rate, mean_return, mean_steps, mean_ttf, timeout_rate
- Acceptance：跑 `python scripts/eval.py --task h1hand-walk-v0 --eval 10 --seeds 3 --driver drq` 落 JSONL + 打印聚合

### Step 2 — DR.Q locomotion baseline（已有 ckpt，最快）
- 跑 DR.Q 在 14 个 h1hand locomotion task 上 N=10 eps × 3 seeds
- 期望：walk/run/stand/balance ≥50%（论文范围内），stair/hurdle 等可能<50%
- 这一步就是"验证 ≥50%"的第一批样本

### Step 3 — 5 个支持 reach 的 manipulation task（Opus 路径 A）
- `push · package · truck · bookshelf_simple · bookshelf_hard` 跑现有 reach skill eval
- 预期 success_rate 多半很低（reach skill 不会"完成任务"），作为下界
- 但视觉收益高，能确认 eval 框架对 manipulation 的 `info['success']` / subtasks 信号也工作

### Step 4 — 选 1 个易任务自训（路径 6）
- 在 4090 上跑 **`h1hand-window-v0`** Dreamer medium（window 是文中最易 manipulation）
- 用 `dependencies/humanoid-bench/dreamerv3/`：`--configs humanoid_benchmark medium --task humanoid_h1hand-window-v0`
- 跑 ~12 h，看 reward 曲线，eval 看是否 ≥50%

### Step 5 — UMI-on-Air 适配（路径 3，Codex 推荐）
- 下 `LeCAR-Lab/umi-on-air_checkpoints` 的 cabinet ckpt
- 写 obs/action adapter（H1 Shadow Hand 动作空间映射）
- eval 在 `h1hand-cabinet-v0` 上看能不能解锁 subtask

### Step 6 — 把通过的任务发回 HF（如果有富余）
- 任何 ≥50% 的自训 ckpt 都发 `wsagi/humanoidbench-<task>-<algo>`，填补 carlosferrazza/humanoid-bench Issue #66 的社区需求

## 进度 / Progress log

（compact 后 Claude 接力时在这里更新，每步标 ✅/❌/🟡 + 一行结果）

- [x] **Step 1 — eval.py ✅** `scripts/eval.py` 完成，drivers: random/zero/reach/drq；输出 JSONL + 聚合统计。验证：walk-v0 + DR.Q seed 0 ar=2 ep 10 跑出 success=3/10 mean_return≈530（论文 [371,652] 内）
- [x] **Step 2 — DR.Q baseline ✅ 超额完成** `scripts/sweep_drq.sh` 扫 28 task seed 0 N=5; `scripts/sweep_drq_multiseed.sh` 对候选扩 N=10×3seeds。**9/9 task ≥50%（5 个 100%）**：h1-crawl/pole/sit_simple/stand 100%, h1hand-sit_simple 100%, h1-run/sit_hard 97%, h1hand-sit_hard 77%, h1-walk 73%。**用户 ≥3 task 目标超 3 倍完成**。结果在 `results/drq_multiseed/*.jsonl`
- [x] **Step 4(替代) — 自训 H1-walk-v0 流水线验证 ✅** patch DR.Q (main.py:222 + DRQ.py:278) + `scripts/train_watcher.py` + `scripts/ckpt_eval_loop.py` 全链路。**500k 步 / 6.6h wall on RTX 4090**。最终 ckpt: **success_rate 90% N=10 ep, mean_return 801**（公开 seed 0 仅 ~530 / ~30%）。**首个自训通关 ckpt**。Ckpt 备份在 `runs/h1_walk_pilot/DRQ/checkpoint/DRQ+HBench-h1-walk-v0+0/`。
- [x] **Step 4b — G1-walk-v0 自训通关 ✅** 三轮实验最终成功。耗费 RTX 4090 ~3h（含 brainstorm 调研）。**关键发现**：
  - Round 1 (G1 torque baseline)：1M 步 mean 100 success 0%，DEAD
  - Round 2 (PD-only Tier S，patches/g1-pos-control.patch)：500k 步 mean 435 success 0%（4.3× 改善但未通关）
  - **Round 3 (PD+BlockedHands Tier S'，patches/humanoid-bench-g1-blocked-hands.patch)：500k 步 success 70% mean 711 ✅** Ckpt 备份在 `runs/g1_walk_pdbh_pilot/`
  - **真正根因**（OpenCode deepseek-v4-pro 诊断）：G1 37D act 含 14 维手指与行走完全无关，DR.Q 同方差 σ=0.2 noise 在 37D 几乎每 transition 都有手指扰动 → encoder dynamics loss 被迫学手指 → 250k catastrophic forgetting。屏蔽手指（act 23D）+ PD 控制双管齐下才通关
  - **Why:** 解决方案是 PD（接口稳定性）+ BlockedHands（去除无关 noise）的**组合**。单独 PD 解决 80%，单独 BlockedHands 也不够。三方 brainstorm 见 `docs/g1_training_strategies.html`
- [ ] Step 3 — reach skill manipulation baseline（低优先级）
- [ ] Step 5 — UMI-on-Air cabinet adapter
- [ ] **Step 6 — HF release** 候选: `wsagi/humanoidbench-h1-walk-v0-drq-selftrained` 含 policy.pt + encoder.pt + agent_var.npy（9 个 .pt，~80MB），mean 801 success 90% N=10 — **比公开 ckpt 强**，值得发回 carlosferrazza/humanoid-bench Issue #66

### Compact 后新发现 / Findings post-compact

1. **DR.Q 实际覆盖 28 个 task**，包括 manipulation：`h1hand-{basketball, bookshelf_hard, bookshelf_simple, door}-v0`。Brainstorm HTML §3 当时漏了，订正：DR.Q manipulation 也有 ckpt，**先跑这 4 个看能不能 ≥50%**，比自训省时间。完整 task list：`h1/h1hand × {balance_hard, balance_simple, crawl, hurdle, maze, pole, reach, run, sit_hard, sit_simple, slide, stair, stand, walk}` 共 28 个（h1hand 多 4 个 manipulation）。
2. **DR.Q seed 命名是 0/10/20/.../90**，不是 0/1/2。eval 时必须用 `--seed_list 0,10,20`，不是 `--seeds 3`。
3. **DR.Q action_repeat=2**（HBenchPreprocessing 默认），eval 时必须传 `--action_repeat 2`，否则 step rate 错位，return 减半。
4. **Seed 间方差极大**：walk-v0 上 seed 0 mean≈530，seed 10≈295，seed 20≈382。论文 [371, 652] 是"10 seeds 最佳范围"，不是"任意 seed 平均"。先用 seed 0 扫所有 task 找出能 ≥50% 的，再扩到 multi-seed。

## 接力时的"立刻可用"命令 / Drop-in commands

```bash
cd /home/david/work/humanoid-training

# 已验证可跑的 viewer（不需要新代码）
DISPLAY=:0 conda run -n humanoidbench --no-capture-output \
    python scripts/drq_viewer.py --task h1hand-walk-v0 --seed 0

# DR.Q ckpt 已经在 cache，cabinet/kitchen 等 manipulation 的 ckpt **不存在**，别试
# DR.Q cache location:
ls ~/.cache/huggingface/hub/models--dmux--DR.Q/snapshots/*/

# humanoid-bench task class 的 success_bar 表见 brainstorm HTML §4
# success 信号字段：info['success'] (cabinet/kitchen), info['success_subtasks']
```

## 不要重复的事 / Do not redo

- ~~不要重新调研「有没有 manipulation ckpt」~~ **订正**：DR.Q 实际覆盖 4 个 manipulation (basketball/bookshelf_hard/bookshelf_simple/door)，brainstorm HTML §3 漏了。Gap 任务（无任何公开 ckpt）实际是：cube/kitchen/cabinet/window/spoon/insert/highbar 共 7 个，不是 9 个。
- 不要用旧 `huggingface-cli`（已 deprecated），用 `hf download`
- 不要试 `cube/kitchen/cabinet/window/spoon/insert/highbar` 配 reach skill —— task class 没 `htarget_low/high`，`native_viewer.py` 会给友好报错（**door/basketball 现在已确认有 DR.Q ckpt，优先用 DR.Q**）
- 不要再加 `--local-dir` 到 HF download —— 默认 cache 路径是规则
- 不要用 `--seeds 3` 跑 DR.Q —— seed 命名是 0/10/.../90，要 `--seed_list 0,10,20`
- 不要漏 `--action_repeat 2` —— DR.Q 训练时 HBenchPreprocessing wrap 了 ActionRepeat(2)，eval 不加 return 减半

### DR.Q submodule patch 走 patches/ 目录

`dependencies/dr-q/` 是上游 submodule，本地有 2 处改动（让 agent.save 真存权重 + 跳过 buffer 写盘）。补丁集中在 `patches/dr-q-save-agent.patch`，clone 后跑 `bash patches/apply.sh` 即可，幂等。**不再手改 submodule 源码**。

**Why:** 用户首次启动 G1 训练跑了 50min 才发现 ckpt 没存权重——上游 main.py 注释掉了 `agent.save`。修过来后又改了 DRQ.py 跳过 367MB buffer 写盘。这两个改动通过 patch 文件维护比记到 memory 里更可靠（apply.sh 幂等检查 + git 可还原）。
