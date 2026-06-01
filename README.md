# humanoid-training

_A training workbench for humanoid robot policies._

人形机器人策略训练工坊 。

---

## 🎯 目标

_Goal_

训练出在指标上**真正完成任务**的人形策略：复刻已有基线、微调预训练模型、从零训练。

_Train humanoid policies that **measurably solve tasks** — reproduce baselines, fine-tune pretrained models, train from scratch._

---

## 🤗 已发布 checkpoints

_Released checkpoints_

两个 HF 仓库，**5 个 self-trained ckpt 全部 ≥ 30% success**，对比公开 baseline 多倍提升：

_Two HF repos, **all 5 self-trained ckpts pass the 30% bar**, with multi-fold gains over public baselines:_

### 🔗 [wsagi/HumanoidBench-DrQ](https://huggingface.co/wsagi/HumanoidBench-DrQ)

| Task           | 自训 / Self-trained                     | 公开 baseline                 | 提升                   |
| -------------- | --------------------------------------- | ----------------------------- | ---------------------- |
| `h1-walk-v0` | **success 90% / mean 801** (N=10) | dmux/DR.Q seed 0: ~30% / ~530 | **3× 成功率**   |
| `g1-walk-v0` | **success 70% / mean 711** (N=10) | DR.Q torque: 0% / mean ~100   | **7.1× return** |

### 🔗 [wsagi/HumanoidBench-TD-MPC2](https://huggingface.co/wsagi/HumanoidBench-TD-MPC2)

| Task           | 自训 / Self-trained                     | 同任务 DR.Q 对比              |
| -------------- | --------------------------------------- | ----------------------------- |
| `h1-walk-v0` | **success 100% / mean 817** (N=3) | DR.Q 90% / 801 (TD-MPC2 略胜) |
| `g1-walk-v0` | **success 50% / mean 602** (N=6)  | DR.Q 70% / 711 (DR.Q 仍领先)  |

均含完整权重、`train.log`、GUI MP4 演示。一键下载与本地观看见 `HumanoidBench-SelfTrained.ipynb`。

_All ckpts include full weights, `train.log`, and GUI MP4 demos. See `HumanoidBench-SelfTrained.ipynb` for one-click download + local playback._

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
├── HumanoidBench.ipynb              # 多任务 × 多策略一键预览 / multi-task × multi-policy preview
├── HumanoidBench-Showcase.ipynb     # baseline (dmux/DR.Q) 9 task 通关展示 / baseline showcase
├── HumanoidBench-SelfTrained.ipynb  # 自训 ckpt HF 下载 + 内嵌视频预览 / self-trained ckpt HF pull + inline MP4
├── Datasets.ipynb                  # G1 公开数据集一键 LeRobot sim 预览 / G1 dataset one-click sim replay
├── scripts/
│   ├── native_viewer.py            # MuJoCo 原生预览（任务通用）/ native viewer
│   ├── drq_viewer.py               # DR.Q checkpoint GUI / DR.Q ckpt GUI
│   ├── tdmpc2_viewer.py            # TD-MPC2 checkpoint GUI / TD-MPC2 ckpt GUI
│   ├── drqv2_viewer.py             # DrQ-v2 checkpoint GUI (dm_control suite)
│   ├── eval.py                     # 多 seed × 多 episode 评测 / multi-seed eval harness
│   ├── tdmpc2_eval.py              # TD-MPC2 N-ep deterministic JSONL eval
│   ├── sweep_drq.sh                # 单 seed 扫所有 task / single-seed sweep
│   ├── sweep_drq_multiseed.sh      # 多 seed 候选任务扩展 / multi-seed expansion
│   ├── launch_train.sh             # 统一 RL launcher，强制挂 watcher / unified train launcher (mandatory watcher)
│   ├── train_watcher.py            # DR.Q slice auto-eval + 早停 / DR.Q slice-based auto-eval & early-stop
│   ├── ckpt_eval_loop.py           # DR.Q per-ckpt auto-eval daemon
│   ├── ckpt_eval_loop_tdmpc2.py    # TD-MPC2 per-ckpt N-ep eval daemon
│   ├── ckpt_eval_loop_dreamerv3.py # DreamerV3 rolling-ckpt snapshot + VRAM-aware deferred eval
│   ├── snap_copy.sh                # 滚动覆盖 ckpt 稀疏 cp daemon (DrQ-v2 / DreamerV3) / sparse ckpt copier
│   ├── train_status.sh             # 一键 ASCII 曲线 + 状态 / one-shot ASCII curve & verdict
│   ├── replay_g1.py                # G1 dataset replay / G1 LeRobot dataset sim replay
│   └── build_showcase_nb.py        # 一键生成展示 notebook / showcase notebook generator
├── docs/                           # 调研与计划 HTML 文档 / research & planning docs
├── patches/                        # submodule 本地补丁 + apply.sh / local submodule patches
├── runs/                           # 训练产出 / training outputs (gitignored)
│   ├── h1_walk_pilot/              # 首个 DR.Q 自训通关 / first DR.Q passing ckpt
│   ├── g1_walk_pdbh_pilot/         # G1-walk DR.Q PDBH 自训通关 / G1-walk DR.Q passing ckpt
│   ├── h1_tdmpc2_pilot/            # H1-walk TD-MPC2 自训通关 (100%) / H1-walk TD-MPC2 passing
│   └── g1_tdmpc2_east_pull/        # G1-walk TD-MPC2 3-seed (AutoDL east) / G1-walk TD-MPC2 multi-seed
└── dependencies/
    ├── humanoid-bench/             # submodule
    ├── dr-q/                       # submodule
    └── drqv2/                      # submodule (dm_control baseline, pinned c0c650b)
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

四种诊断状态：`PROGRESS / UNDERFIT / OVERFIT / DEAD`。详见 `.memory/feedback_train_with_watcher.md`。

_Four diagnostic states: PROGRESS / UNDERFIT / OVERFIT / DEAD. See memory file for rules._

> Submodule 本地补丁统一放在 `patches/`，clone 后跑 `bash patches/apply.sh` 即可。
> _Local submodule patches live in `patches/` — run `bash patches/apply.sh` after clone._

---

## 📊 当前进度

_Current progress_

- ✅ **DR.Q baseline 9/9 ≥50% 成功率**（5 个 100%）on H1 / H1Hand 运动类任务
  _DR.Q baseline reaches **≥50% success on 9/9 locomotion tasks** (5 at 100%)._
- 🏆 **DR.Q 自训** — H1-walk **90% / mean 801**（N=10）+ G1-walk PDBH **70% / mean 711**（N=10，PD + BlockedHands 两层 patch，vs torque baseline 7.1× 提升）
  _DR.Q self-trained: H1-walk 90% / 801, G1-walk PDBH 70% / 711 (two-layer patch, 7.1× over torque baseline)._
- 🏆 **TD-MPC2 自训** ⭐ — H1-walk **100% / mean 817**（N=3，1M 步 / 4090）+ G1-walk PDBH **50% / mean 602**（N=6，1M 步 / AutoDL 4080S 32G，3-seed parallel）
  _TD-MPC2 self-trained: H1-walk 100% / 817 (1M steps / 4090), G1-walk PDBH 50% / 602 (1M / AutoDL 4080S, 3-seed parallel)._
- 🟡 **DreamerV3 探路** — H1Hand-window-v0 small 1M 步 DEAD（0%，未挂 watcher 浪费 6h）；教训写进 `feedback_train_with_watcher.md` 强制 mandatory watcher
  _DreamerV3 trial: H1Hand-window-v0 small 1M steps DEAD (0%, no watcher — wasted 6h). Lesson codified as mandatory watcher rule._
- 🤗 全部自训 ckpt 已发布到 HF 两个仓库（详见顶部 [已发布 checkpoints](#-已发布-checkpoints) 段）
  _All self-trained ckpts published to two HF repos — see the **Released checkpoints** section._
- 🟡 Manipulation gap 任务待攻关：cube · kitchen · cabinet · window · spoon · insert · highbar
  _Manipulation gap tasks pending: cube · kitchen · cabinet · window · spoon · insert · highbar._
- 📋 行动计划：`docs/manipulation_policy_brainstorm.html`（Opus + GPT-5.5 + DeepSeek 三方头脑风暴汇总）
  _Action plan in `docs/manipulation_policy_brainstorm.html` (triangulated across Opus + GPT-5.5 + DeepSeek)._

---

## 📚 相关工作

_Related work_

- [carlosferrazza/humanoid-bench](https://github.com/carlosferrazza/humanoid-bench) — upstream benchmark suite
- [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) — TD-MPC2 algorithm
- [facebookresearch/drqv2](https://github.com/facebookresearch/drqv2) — DrQ-v2 (pixel-based, dm_control baseline)
- [dmux/DR.Q](https://huggingface.co/dmux/DR.Q) — DR.Q pretrained checkpoints (baseline)
- 🤗 [wsagi/HumanoidBench-DrQ](https://huggingface.co/wsagi/HumanoidBench-DrQ) — **DR.Q self-trained**: H1-walk 90% / G1-walk 70%
- 🤗 [wsagi/HumanoidBench-TD-MPC2](https://huggingface.co/wsagi/HumanoidBench-TD-MPC2) — **TD-MPC2 self-trained**: H1-walk 100% / G1-walk 50%
- `../mujoco-experience` · `../isaaclab-experience` — sister repos in this family

---

## 📄 License

MIT
