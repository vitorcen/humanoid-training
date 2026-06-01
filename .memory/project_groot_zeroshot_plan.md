---
name: groot-zeroshot-plan
description: GR00T-N1.7 zero-shot on HumanoidBench manipulation gap (7 tasks) — resume point + first-3-steps verification checklist
metadata:
  type: project
---

## 目标 / Goal

跑 **`nvidia/GR00T-N1.7-3B`** zero-shot 到 HumanoidBench manipulation gap 7 task：
`cube · kitchen · cabinet · window · spoon · insert · highbar`。

**Why:** DR.Q 攻这些 task 大概率不行（state-only RL 对接触密集 manipulation 弱），VLA 是现实路径；GR00T 专为 humanoid 设计；本机 4090 24GB 推 3B 模型够。

## 起点状态 / State at recording

| 项 | 状态 | 位置 |
|---|---|---|
| HF cache: `nvidia/GR00T-N1.7-3B` (官方 base) | 🟡 下载中（2026-05-26 启动，~6.4 GB safetensors，启动时 ~3.5 G 完成） | `models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b9.../` |
| HF cache: `nvidia/GR00T-N1.6-3B` | ✅ (备选 base，N1.7 出问题时回退) | `models--nvidia--GR00T-N1.6-3B` |
| HF cache: `nvidia/GR00T-N1.5-3B` | ✅ (备选 base) | `models--nvidia--GR00T-N1.5-3B` |
| HF cache: `hi-space/GR00T-N1.7-3B-Pick-Orange` | ✅ (PickOrange finetune，参考其 statistics.json / experiment_cfg 看 prompt schema) | `models--hi-space--GR00T-N1.7-3B-Pick-Orange/snapshots/887def6d.../` |
| HF cache: `wsagi/GR00T-N1.6-PickOrange` | ✅ (用户之前自训的 1.6 PickOrange，可对照看 finetune diff) | `models--wsagi--GR00T-N1.6-PickOrange` |
| HumanoidBench RGB obs 支持 | ✅ 已确认 | `dependencies/humanoid-bench/humanoid_bench/env.py:105` `render_modes` 含 `rgb_array` |
| `docs/manipulation_policy_brainstorm.html` | ✅ 三方头脑风暴文档已存在 | `docs/` |
| Adapter 脚本 `scripts/groot_zeroshot.py` | ❌ 未写 | TODO #1 |

**Why state pinned:** compact 后第一时间不用重新调研有没有 ckpt / 有没有 RGB；直接进 adapter 编写。

## 首 3 步验证清单 / First-3-steps verification checklist

按顺序，每步跑完打勾：

### Step 1 — GR00T-N1.7 model card / processor schema
- 读 `~/.cache/huggingface/hub/models--hi-space--GR00T-N1.7-3B-Pick-Orange/snapshots/*/processor_config.json` 和 `config.json`，弄清：
  - 期望 RGB 输入 shape（HxW，channel order）
  - prompt / language token 格式（chat template? raw text? action tag?）
  - action head 维度（PickOrange 是 7-DoF arm+gripper？需确认）
  - `embodiment_id.json` 和 `experiment_cfg` 怎么用
- Acceptance：能用 `transformers` 或 GR00T SDK 加载模型 + 跑一次 dummy forward 不报错

### Step 2 — HumanoidBench obs/action vs GR00T 对齐
- `python -c "import gym, humanoid_bench; e=gym.make('h1hand-cube-v0', render_mode='rgb_array'); o,_=e.reset(); print(type(o), o.shape if hasattr(o,'shape') else 'dict'); print(e.action_space)"`
- 关键差距：
  - GR00T action_dim（PickOrange ~7-8）vs H1Hand action_dim（61）
  - GR00T 期望多视角相机；HumanoidBench 默认单相机
- 决定 adapter 策略：把 GR00T 输出的 7-DoF arm action **只控 H1 的对应 arm joints**，其余 leg/torso 用 reach skill 或 zero / hold

### Step 3 — 最小 rollout (`scripts/groot_zeroshot.py`)
- 加载 ckpt → reset env → 取 RGB → 调 GR00T `.predict()` → action 经 adapter 映射到 H1 → env.step → 看一步内不崩
- 不要求成功，只要求**前向不报错 + action 在合法 range**
- Acceptance：能跑完整 episode（trunc 出来即可），存一个 GIF/MP4 看初始行为是否合理

## 起手 task 优先级 / Task ordering

| Order | Task | Why first / Why last |
|---|---|---|
| 1 | `h1hand-cube-v0` | 单目标 pick，最接近 PickOrange 训练分布；最可能 zero-shot 出效果 |
| 2 | `h1hand-window-v0` | 单手开窗，brainstorm 文档列为"最易" manipulation |
| 3 | `h1hand-spoon-v0` | 单物体抓取 |
| 后 | `kitchen/cabinet/insert` | 多 subtask 序列，zero-shot 大概率不通 |
| 最后 | `highbar` | 全身协调，超出 VLA 训练分布 |

**Why:** 用最相近分布（PickOrange → cube）做 sanity check，再扩；省 GPU 时间。

## 已知阻塞 + 应对 / Known blockers & mitigations

1. **Action 维度不匹配**（GR00T ~7D arm vs H1Hand 61D）
   → 适配：GR00T action 映射到 H1Hand 的 right_arm joints (7 DoF)，剩余 54D 用 reach skill 控制（reach skill `dependencies/humanoid-bench/data/reach_two_hands/` 已有）
2. **相机视角不一致**
   → adapter 写法：复用 HumanoidBench `task_info.camera_name` + `default_camera_config`，resize / crop 到 GR00T 期望 shape
3. **Prompt schema 未知**
   → Step 1 读 `chat_template.json` 和 README；如缺则参考 `nvidia/GR00T-N1.5-3B` 的 example
4. **OOM 风险**
   → 4090 24GB 推 3B fp16 应该够；如不够用 bf16 + flash-attn；最后选项 N1.5 而非 N1.7
5. **PickOrange finetune 漂移**
   → N1.7-Pick-Orange 是窄分布 finetune，对 cube 可能比 base 强，对 kitchen 可能更弱；若 cube 不行，回退 `nvidia/GR00T-N1.5-3B` base

## 参考资料 / References

- `docs/manipulation_policy_brainstorm.html` — Opus + GPT-5.5 + DeepSeek 三方分析（VLA 路径优劣 + 7 个 gap task 详情）
- `docs/g1_training_strategies.html` — 同模型 brainstorm 流程模板，可参考结构
- `~/work/isaaclab-experience/` — sister repo，可能有 GR00T 推理示例脚本（待查）
- HumanoidBench task class：`dependencies/humanoid-bench/humanoid_bench/tasks/*.py`，每个 task 的 obs/reward/success 定义

## 不要重复的事 / Do not redo

- 不要重新下 GR00T ckpt — N1.5/1.6/1.7 + PickOrange variants 全在 cache（[[no-secrets-in-memory]] 已确认无需任何 HF token 写入 memory）
- 不要改 HumanoidBench env 加 RGB — 已支持 `render_mode="rgb_array"`
- 不要用 DR.Q 思路设计 noise / exploration — VLA 是 IL pretrain，零探索；直接 deterministic rollout

## 衔接 [[project-benchmark-validation]]

- 本计划 = `project_benchmark_validation.md` 的 **Step 7（新增）**：VLA zero-shot 攻 manipulation gap
- 完成定义：至少 1 个 manipulation task 的 GR00T zero-shot rollout success_rate > 0（不强求 ≥50%；zero-shot baseline 能跑出数值即算开局成功）
- 不通过则进入 Step 8：用 DR.Q 自训 ckpt rollout 生成 demo → GR00T finetune（需 H100 算力，可能要租）
