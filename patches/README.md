# patches

Submodule 本地补丁集中存放点。

_Local patches against pinned submodules — apply after `git submodule update`._

| Patch | 目标 / Target | 作用 / What |
| --- | --- | --- |
| `dr-q-save-agent.patch` | `dependencies/dr-q/` | 让 DR.Q 训练真正保存 agent 权重（上游默认注释掉了） · skip 367 MB replay buffer dump |

## 应用 / Apply

```bash
bash patches/apply.sh
```

## 还原 / Revert

```bash
cd dependencies/dr-q && git checkout -- DRQ/ && cd -
```
