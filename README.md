# humanoid-training

_A training workbench for humanoid robot policies._

人形机器人策略训练工坊 。

---

## 🎯 目标

_Goal_

训练出在指标上**真正完成任务**的人形策略：复刻已有基线、微调预训练模型、从零训练。

_Train humanoid policies that **measurably solve tasks** — reproduce baselines, fine-tune pretrained models, train from scratch._

---

## 🏗️ 覆盖范围

_Scope_

| 维度 / Axis                  | 内容 / Content                                                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **算法 / Algorithms**  | RL (DR.Q · DreamerV3 · TD-MPC2 · SAC · PPO) · IL (BC · DP · ACT) · VLA (π0.5 · GR00T · OpenVLA) · scripted / hybrid |
| **机型 / Embodiments** | Unitree H1 / H1Hand / G1 · 可扩展到其它人形 / extensible to other humanoids                                                    |
| **仿真 / Simulators**  | MuJoCo · MJX · Isaac Sim · 可扩展 / extensible                                                                               |
| **任务 / Tasks**       | locomotion · manipulation · whole-body coordination · 任意 benchmark suite                                                   |
| **产出 / Outputs**     | reproducible scripts · multi-seed eval reports · HF checkpoints                                                               |

---

## 📂 仓库结构

_Repository layout_

```
humanoid-training/
├── HumanoidBench.ipynb          # 多任务 × 多策略一键预览 / multi-task × multi-policy preview
├── HumanoidBenchShowcase.ipynb  # 训练结果与可视化 / training results & visualization
├── scripts/
│   ├── native_viewer.py         # MuJoCo 原生预览（任务通用）/ native viewer
│   ├── drq_viewer.py            # DR.Q checkpoint 加载与回放 / DR.Q ckpt loader
│   ├── eval.py                  # 多 seed × 多 episode 评测 / multi-seed eval harness
│   ├── sweep_drq.sh             # 单 seed 扫所有 task / single-seed sweep
│   ├── sweep_drq_multiseed.sh   # 多 seed 候选任务扩展 / multi-seed expansion
│   ├── train_watcher.py         # 分 slice auto-eval + 早停 + 过/欠拟合检测 / slice-based auto-eval & early-stop
│   ├── ckpt_eval_loop.py        # 后台 daemon：每出 ckpt 自动 mirror + N=3 eval / per-ckpt auto-eval daemon
│   ├── train_status.sh          # 一键 ASCII 曲线 + 状态 / one-shot ASCII curve & verdict
│   └── build_showcase_nb.py     # 一键生成展示 notebook / showcase notebook generator
├── docs/                        # 调研与计划 HTML 文档 / research & planning docs
├── patches/                     # submodule 本地补丁 + apply.sh / local submodule patches
├── runs/                        # 训练产出 / training outputs (gitignored)
│   └── h1_walk_pilot/           # 首个自训通关 ckpt 实验 / first self-trained ckpt run
└── dependencies/
    ├── humanoid-bench/          # submodule
    └── dr-q/                    # submodule
```

---

## 🚀 快速开始

_Quickstart_

```bash
# 1. 克隆（含 submodule）/ clone with submodules
git clone --recursive git@github.com:vitorcen/humanoid-training.git
cd humanoid-training

# 2. 建 conda 环境 / set up conda env
conda create -n humanoidbench python=3.11 -y
conda activate humanoidbench
pip install -e dependencies/humanoid-bench

# 3. 启动 MuJoCo 原生预览 / launch native preview
DISPLAY=:0 python scripts/native_viewer.py --env h1hand-walk-v0 --action random

# 4. 加载 DR.Q 预训练 checkpoint（自动从 HF 下载） / load DR.Q ckpt (auto-download)
DISPLAY=:0 python scripts/drq_viewer.py --task h1hand-walk-v0 --seed 0

# 5. 多 seed × 多 episode 评测 / multi-seed eval
python scripts/eval.py --task h1-walk-v0 --driver drq \
    --eval 10 --seed_list 0,10,20 --action_repeat 2 \
    --out results/h1-walk-v0.jsonl

# 6. 从零自训（含 patch + watcher + auto-eval 全链路） / self-train with full monitoring
#    详见 "自训流水线" 段
```

详细工作流见 `HumanoidBench.ipynb`，自训成果展示见 `HumanoidBenchShowcase.ipynb`。

_Full workflow in `HumanoidBench.ipynb`; training results in `HumanoidBenchShowcase.ipynb`._

---

## 🔁 自训流水线

_Self-train pipeline (LeIsaac-inspired slice-based auto-eval & early-stop)_

任何超过 1h 的训练**必须**配合 watcher 跑，不能"启动完就走人"。流水线一次跑通包含三个并行进程：

_Any >1h training **must** run alongside a watcher — never "fire-and-forget". Three parallel processes:_

```bash
# A) 训练主进程 / training
cd dependencies/dr-q/DRQ && nohup python main.py \
    --env HBench-h1-walk-v0 --seed 0 \
    --total_timesteps 500000 --save_freq 50000 \
    --base_folder $PWD/../../../runs/h1_walk_pilot/ \
    --save_experiment > runs/h1_walk_pilot/train.log 2>&1 &

# B) Slice watcher：实时分 10 段聚合 eval 流 + 早停 / live milestone aggregator + early-stop
nohup python scripts/train_watcher.py \
    --run runs/h1_walk_pilot/DRQ/HBench-h1-walk-v0/r0 \
    --train_pid <TRAIN_PID> --total_steps 500000 --success_bar 700 \
    > runs/h1_walk_pilot/watcher.out 2>&1 &

# C) Ckpt daemon：新 ckpt → mirror 到 HF cache → N=3 deterministic eval / per-ckpt auto-eval
nohup python scripts/ckpt_eval_loop.py \
    --task h1-walk-v0 --seed 0 \
    --ckpt_dir runs/h1_walk_pilot/DRQ/checkpoint/DRQ+HBench-h1-walk-v0+0 \
    --train_pid <TRAIN_PID> --eval_eps 3 \
    --out runs/h1_walk_pilot/ckpt_eval.csv \
    > runs/h1_walk_pilot/ckpt_eval_daemon.out 2>&1 &

# 任意时刻一键看进度 / inspect any time
bash scripts/train_status.sh runs/h1_walk_pilot/DRQ/HBench-h1-walk-v0/r0
```

四种诊断状态：`PROGRESS / UNDERFIT / OVERFIT / DEAD`。详见 `.claude/memory/feedback_train_with_watcher.md`。

_Four diagnostic states: PROGRESS / UNDERFIT / OVERFIT / DEAD. See memory file for rules._

> Submodule 本地补丁统一放在 `patches/`，clone 后跑 `bash patches/apply.sh` 即可。
> _Local submodule patches live in `patches/` — run `bash patches/apply.sh` after clone._

---

## 📊 当前进度

_Current progress_

- ✅ **DR.Q baseline 9/9 ≥50% 成功率**（5 个 100%）on H1 / H1Hand 运动类任务
  _DR.Q baseline reaches **≥50% success on 9/9 locomotion tasks** (5 at 100%)._
- 🏆 **首个自训通关 ckpt** — `h1-walk-v0` from-scratch 500k 步 / 6.6h on RTX 4090：
  **success 90% (N=10 ep), mean_return 801** ← 公开 ckpt seed 0 仅 ~30% / ~530
  _First self-trained passing ckpt: H1-walk reaches **90% success / 801 mean** vs HF public 30% / 530._
- 🛑 G1 自训早停 — `g1-walk-v0` 1M 步 torque 控制达不到 success_bar，已记录到 memory 不再重试
  _G1-walk torque control found insufficient in 1M steps; documented to avoid re-attempt._
- 🟡 Manipulation gap 任务待攻关：cube · kitchen · cabinet · window · spoon · insert · highbar
  _Manipulation gap tasks pending: cube · kitchen · cabinet · window · spoon · insert · highbar._
- 📋 行动计划：`docs/manipulation_policy_brainstorm.html`（Opus + GPT-5.5 + DeepSeek 三方头脑风暴汇总）
  _Action plan in `docs/manipulation_policy_brainstorm.html` (triangulated across Opus + GPT-5.5 + DeepSeek)._

---

## 📚 相关工作

_Related work_

- [carlosferrazza/humanoid-bench](https://github.com/carlosferrazza/humanoid-bench) — upstream benchmark suite
- [dmux/DR.Q](https://huggingface.co/dmux/DR.Q) — DR.Q pretrained checkpoints
- `../mujoco-experience` · `../isaaclab-experience` — sister repos in this family

---

## 📄 License

MIT
