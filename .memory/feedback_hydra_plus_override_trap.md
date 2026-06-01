---
name: hydra-plus-override-trap
description: "Hydra `+task=foo` 加 nested key 不 override toplevel；DrQ-v2 上 6h+×3 seed 误训成 quadruped_walk 而非 humanoid_walk 的血泪教训"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 12be2311-45e9-45b0-9189-4b3c0c491e3e
---

启动 hydra 项目时 **不要无脑加 `+`**——`+key=value` 是**添加** nested 子树，不是 override 顶层。如果代码直接读 toplevel 变量（不解析嵌套），override 失效但**无报错**，会一直跑错的任务。

**Why（2026-05-27 事故）**：DrQ-v2 在 west AutoDL 用 `python train.py +task=humanoid_walk seed=1 ...` 启动 3 seed，hydra 的 `+task=humanoid_walk` 添加了 `cfg.task = {humanoid_walk.yaml 内容}` 这个 nested dict，但 **drqv2/train.py 代码读 `cfg.task_name`（toplevel，来自 default config）= "quadruped_walk"**。结果：
- 3 个 seed 各跑了 **1.3M frame × ~6h × 3 = 18 GPU 小时**
- 训练命名 `humanoid_walk_s1` / experiment field 误导
- eval reward R=817 看起来很漂亮（接近 dm_control "solved"），其实是 quadruped_walk 的分数
- **直到本地启动 GUI viewer 才发现 actor 输出 12 维（quadruped 4 legs × 3 joints），不是 humanoid 21 维**

诊断捷径——saved checkpoint 的 `actor.policy[-1].out_features` 揭穿一切：
```python
agent = torch.load(snap).get('agent', None)
print(agent.actor.policy[-1])   # Linear(in=1024, out=12)  → quadruped
```

**How to apply**：

1. **DrQ-v2 正确启动方式**：`task=humanoid_walk`（无 `+`，覆盖 default `task@_global_`）。或 `task_name=humanoid_walk` 直接覆盖顶层。
2. **任何 hydra 项目启动后第一时间 grep `.hydra/config.yaml`**：
   ```bash
   grep -E "task|task_name|domain" runs/<exp>/.hydra/config.yaml
   ```
   确认顶层 task_name 是你想要的。如果出现 `task: {nested...}` + `task_name: <default>` 同时存在 → **`+` 用错了**。
3. **训练第一个 ckpt 出来立刻 inspect actor/policy 维度**——和 env action_spec 对照。1 行 sanity check 省下 18 GPU 小时。
4. **不要相信 experiment / logdir 命名**——那些是人填的字符串，跟实际 train 任务可以完全不一致。

**Anti-patterns**：
- ❌ "我命名写 humanoid_walk 那肯定就是 humanoid_walk" — 完全不靠谱
- ❌ 盲信 hydra override 生效 — 不到 GUI/eval 阶段你看不出错
- ❌ Reward 趋势看起来正常就以为是对的 — quadruped_walk paper baseline 也是 ~800

**适用范围**：任何用 hydra 的训练框架（drqv2 / dreamerv3 / tdmpc2 等多个 RL repo），尤其代码混用 `cfg.foo` 和 `cfg.foo.bar`时。

Related: [[feedback-train-with-watcher]]（watcher 也救不了任务名错——它只看 reward 趋势，不验任务身份）。
