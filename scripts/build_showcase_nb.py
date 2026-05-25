"""Generate HumanoidBenchShowcase.ipynb — 一键查看 DR.Q 通关情况 + 模型信息.

_One-shot notebook builder: aggregates JSONL results, lists model params,
exposes one-click GUI launch buttons per task._

Run:
    conda run -n humanoidbench python scripts/build_showcase_nb.py
"""

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "HumanoidBenchShowcase.ipynb"


def md(src: str) -> dict:
    return nbf.v4.new_markdown_cell(src.lstrip("\n"))


def code(src: str) -> dict:
    return nbf.v4.new_code_cell(src.lstrip("\n"))


cells: list[dict] = []

cells.append(md("""
# HumanoidBench 通关展示  ·  *DR.Q Task Completion Showcase*

> 一键查看 **DR.Q 公开 checkpoint 在 HumanoidBench 上能稳定通关哪些任务**，并启动 MuJoCo GUI 现场观看.
>
> _One-click view of which HumanoidBench tasks the public DR.Q checkpoints reliably solve, plus on-demand MuJoCo GUI playback._

本 notebook 不训练模型，**只展示已经能通关的任务**。验证数据来自 `results/drq_multiseed/*.jsonl`（每任务 N=10 ep × 3 seeds = 30 episodes）.

_This notebook trains nothing — it just showcases tasks that already pass. Numbers come from `results/drq_multiseed/*.jsonl` (N=10 ep × 3 seeds = 30 episodes per task)._
"""))

cells.append(md("""
## §1  用到什么模型 / *What model is used*

**算法**：[DR.Q](https://github.com/dmksjfl/DR.Q) — ICML 2026 投稿，TD3 家族 + 模型化表征学习. 公开权重在 [`dmux/DR.Q`](https://huggingface.co/dmux/DR.Q).

_Algorithm: **DR.Q** (ICML 2026 submission) — TD3-family actor-critic + model-based representation learning. Public weights at `dmux/DR.Q`._

**为什么选 DR.Q？**
- 是目前唯一公开发布 HumanoidBench 训练完成 checkpoint 的项目（28 task × 10 seed = 280 ckpt）
- 论文报告在 locomotion 上达到 SOTA；社区缺 manipulation 完整 ckpt

_Why DR.Q: only public source of trained HumanoidBench ckpts (28 tasks × 10 seeds = 280 ckpt). Paper reports SOTA on locomotion; manipulation ckpts remain a community gap._
"""))

cells.append(code("""
# Eval 时实际加载的模块：Encoder + Policy（跳过 367 MB replay buffer / target nets / optimizers）
# _Eval-time load: Encoder + Policy only — skips the 367 MB replay buffer, targets, optimizers._
import sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "dependencies" / "dr-q" / "DRQ"))

import models, DRQ
hp = DRQ.Hyperparameters()

# H1 (无 hand): obs_dim=51, act_dim=19  ·  H1Hand: obs_dim=151, act_dim=61
enc = models.Encoder(state_dim=51, action_dim=19, pixel_obs=False,
    num_bins=hp.num_bins, zs_dim=hp.zs_dim, za_dim=hp.za_dim, zsa_dim=hp.zsa_dim,
    hdim=hp.enc_hdim, activ=hp.enc_activ)
pol = models.Policy(action_dim=19, discrete=False, gumbel_tau=hp.gumbel_tau,
    zs_dim=hp.zs_dim, hdim=hp.policy_hdim, activ=hp.policy_activ)

def count(m): return sum(p.numel() for p in m.parameters())
print(f"Encoder params : {count(enc):>11,}")
print(f"Policy  params : {count(pol):>11,}")
print(f"Eval total     : {count(enc)+count(pol):>11,}  (~{(count(enc)+count(pol))/1e6:.2f} M)")
print()
print("DR.Q 关键超参 / Key hyperparameters")
for k in ["zs_dim", "zsa_dim", "za_dim", "enc_hdim", "policy_hdim",
          "discount", "num_bins", "enc_horizon", "Q_horizon"]:
    print(f"  {k:<14} = {getattr(hp, k)}")
"""))

cells.append(md("""
## §2  通关成绩单 / *Acceptance results table*

下方读取 `results/drq_multiseed/*.jsonl`（已经跑完，无需重跑），按 success_rate 降序排.

_Loads from `results/drq_multiseed/*.jsonl` (already produced — no re-run). Sorted by success_rate desc._
"""))

cells.append(code("""
import json
from pathlib import Path

rows = []
for p in sorted(Path("results/drq_multiseed").glob("*.jsonl")):
    with p.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("_summary"):
                rows.append(d)

rows.sort(key=lambda r: -r["success_rate"])

try:
    import pandas as pd
    df = pd.DataFrame([{
        "task": r["task"],
        "success%": f"{r['success_rate']*100:.0f}%",
        "mean_return": f"{r['mean_return']:.1f}",
        "timeout%": f"{r['timeout_rate']*100:.0f}%",
        "N (eps × seeds)": f"{r['eval_per_seed']} × {len(r['seeds'])}",
        "seeds": r["seeds"],
    } for r in rows])
    display(df)
except ImportError:
    print(f"{'task':<32} {'succ':>6} {'mean_ret':>9} {'timeout':>7}")
    for r in rows:
        flag = "🟢" if r["success_rate"] >= 0.5 else "🔴"
        print(f"{r['task']:<32} {r['success_rate']*100:5.0f}% {r['mean_return']:9.1f} {r['timeout_rate']*100:6.0f}%  {flag}")

print(f"\\n通关 task 数 / Tasks ≥50% success: {sum(1 for r in rows if r['success_rate']>=0.5)} / {len(rows)}")
"""))

cells.append(md("""
## §3  任务说明 / *Task descriptions*

每个 task 的 success 判定：
- 多数任务：`ep_return ≥ task.success_bar`（自带阈值）
- `cabinet/kitchen/door/basketball/push/package/bookshelf`：`info['success']==True`

_Success criteria per task: most use `ep_return ≥ success_bar`; cabinet/kitchen/etc use `info['success']`._
"""))

cells.append(code("""
TASK_INFO = {
    # locomotion (h1 / h1hand 共用描述)
    "walk":         ("走路 / Walk forward",                        700),
    "run":          ("跑步 / Run forward",                         700),
    "stand":        ("站立平衡 / Standing balance",                800),
    "sit_simple":   ("坐到椅子上（简单） / Sit on chair (easy)",    750),
    "sit_hard":     ("坐到椅子上（困难） / Sit on chair (hard)",    750),
    "crawl":        ("钻洞爬行 / Crawl under obstacle",            700),
    "pole":         ("绕杆穿行 / Slalom around poles",             700),
    "stair":        ("上楼梯 / Climb stairs",                      700),
    "slide":        ("滑行平衡 / Slide stance",                    700),
    "hurdle":       ("跨栏 / Hurdle jump",                         700),
    "balance_simple":("窄板平衡（易）/ Narrow-beam balance",        800),
    "balance_hard": ("窄板平衡（难）/ Narrow-beam balance hard",    800),
    "maze":         ("走迷宫 / Maze navigation",                  1200),
    "reach":        ("手部到达目标 / Reach hand to target",       12000),
}

# 直接对照成绩单看每个 task 在做什么
import json
from pathlib import Path
rows = []
for p in sorted(Path("results/drq_multiseed").glob("*.jsonl")):
    for line in p.open():
        d = json.loads(line)
        if d.get("_summary"):
            rows.append(d)

print(f"{'task':<28} {'succ':>5}  description")
print("-" * 80)
for r in sorted(rows, key=lambda x: -x["success_rate"]):
    short = r["task"].replace("h1hand-", "").replace("h1-", "").replace("-v0", "")
    desc, bar = TASK_INFO.get(short, ("?", "?"))
    print(f"{r['task']:<28} {r['success_rate']*100:4.0f}%  {desc}  [bar={bar}]")
"""))

cells.append(md("""
## §4  一键 GUI 观看 / *One-click GUI playback*

点下方按钮启动 MuJoCo 原生窗口，看 DR.Q 怎么完成任务. 窗口里的操作：

| 操作 / Action | 效果 / Effect |
|---|---|
| 左键拖 / left-drag | 旋转相机 / rotate camera |
| 右键拖 / right-drag | 平移 / pan |
| 滚轮 / scroll | 缩放 / zoom |
| 空格 / space | 暂停/继续 / pause-resume |
| Esc / close | 退出 / quit |

每次按下按钮会启动一个新进程，**关掉窗口才能再启动下一个**.

_Each click launches a new viewer process — close the window before launching another._
"""))

cells.append(code("""
import subprocess
from ipywidgets import Button, HBox, VBox, Output, Label
from IPython.display import display

# 按 success_rate 排过的可通关任务
PASSING_TASKS = [
    ("h1-sit_simple-v0",       0, "坐下 / Sit (100%)"),
    ("h1hand-sit_simple-v0",   0, "Hand 版坐下 / Sit with Shadow Hand (100%)"),
    ("h1-stand-v0",            0, "站立 / Stand (100%)"),
    ("h1-crawl-v0",            0, "爬行 / Crawl (100%)"),
    ("h1-pole-v0",             0, "绕杆 / Pole slalom (100%)"),
    ("h1-run-v0",              0, "跑步 / Run (97%)"),
    ("h1-sit_hard-v0",         0, "坐下-难 / Sit hard (97%)"),
    ("h1-walk-v0",             0, "走路 / Walk (73%)"),
    ("h1hand-sit_hard-v0",     0, "Hand 版坐下-难 / Sit hard with Hand (77%)"),
]

out = Output()

def launch(task, seed):
    def _click(_btn):
        with out:
            print(f"▶ launching  task={task}  seed={seed}  ...")
        # 不阻塞 notebook：detach process
        subprocess.Popen(
            ["conda", "run", "-n", "humanoidbench", "--no-capture-output",
             "python", "scripts/drq_viewer.py",
             "--task", task, "--seed", str(seed),
             "--action_repeat", "2", "--fps", "60"],
            env={"DISPLAY": ":0", **__import__("os").environ},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    return _click

rows = []
for task, seed, label in PASSING_TASKS:
    b = Button(description=f"▶ {label}", layout={"width": "420px"})
    b.on_click(launch(task, seed))
    rows.append(b)

display(VBox(rows), out)
"""))

cells.append(md("""
## §5  完整 eval 复跑 / *Reproduce the full eval*

如果想自己复现成绩单（每 task ~30 sec on RTX 4090）：

_To reproduce the table (about 30 sec per task on an RTX 4090):_

```bash
# 28-task quick scan, seed 0, 5 ep
bash scripts/sweep_drq.sh

# Multi-seed validation on candidates that passed seed-0 scan
bash scripts/sweep_drq_multiseed.sh

# 单 task 自定义
python scripts/eval.py --task h1-sit_simple-v0 --driver drq \\
    --eval 10 --seed_list 0,10,20 --action_repeat 2 \\
    --out results/_my_sit_simple.jsonl
```
"""))

cells.append(md("""
## §6  局限 / *Limitations*

- ❌ Manipulation 复杂任务 (`cube · kitchen · cabinet · window · spoon · insert · highbar`) **没有任何公开 ckpt**，需要自训.
- ⚠️  `door / basketball / bookshelf` 虽然 DR.Q 有 ckpt，但 seed 0 也未达 50%——任务本身困难，社区训练未收敛.
- ⚠️  Seed 间方差大：walk-v0 seed 0/10/20 mean return 分别 530 / 295 / 382. 不要用单 seed 评估.
- ⚠️  Eval 必须传 `--action_repeat 2`，否则 step rate 错位，return 减半.

_Caveats: 7 manipulation tasks have no public ckpt; door/basketball/bookshelf have DR.Q ckpts but didn't reach 50% even on seed 0; large seed variance — never trust a single seed; remember `--action_repeat 2`._
"""))


nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "humanoidbench", "display_name": "Python (humanoidbench)"},
    "language_info": {"name": "python"},
})

OUT.write_text(nbf.writes(nb))
print(f"✅ wrote {OUT} ({len(cells)} cells)")
