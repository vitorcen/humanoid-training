# patches

Submodule 本地补丁集中存放点。

_Local patches against pinned submodules — apply after `git submodule update`._

| Patch | 目标 / Target | 作用 / What |
| --- | --- | --- |
| `dr-q-save-agent.patch` | `dependencies/dr-q/` | 让 DR.Q 训练真正保存 agent 权重（上游默认注释掉了） · skip 367 MB replay buffer dump |
| `g1-pos-control.patch` | `dependencies/humanoid-bench/` | 把 G1 motor → PD position 控制（参照 H1 配置），让 DR.Q sample-efficiency 与 H1 同量级。生成器：`patches/build_g1_pos.py` |
| `humanoid-bench-g1-blocked-hands.patch` | `dependencies/humanoid-bench/` | 扩展 `BlockedHandsLocoWrapper` 支持 G1（act 37→23，屏蔽 14 维手指），`__init__.py` 自动给 g1-* task 启用。OpenCode 诊断：手指 noise 污染 encoder dynamics loss |

## 应用 / Apply

```bash
bash patches/apply.sh
```

## 还原 / Revert

```bash
cd dependencies/dr-q && git checkout -- DRQ/ && cd -
```
