"""Generate HumanoidBench-Showcase.ipynb — 公开 dmux/DR.Q ckpt 在 27 HumanoidBench task 上的完整状态.

_One-shot notebook builder: aggregates JSONL results across the full HumanoidBench
task suite, lists model params, exposes one-click GUI launch buttons for tasks that pass._

Run:
    conda run -n humanoidbench python scripts/build_showcase_nb.py
"""

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "HumanoidBench-Showcase.ipynb"


def md(src: str) -> dict:
    return nbf.v4.new_markdown_cell(src.lstrip("\n"))


def code(src: str) -> dict:
    return nbf.v4.new_code_cell(src.lstrip("\n"))


cells: list[dict] = []

cells.append(md("""
# HumanoidBench 公开 ckpt 全景  ·  *Public DR.Q Coverage Across HumanoidBench*

> 一站式看 **`dmux/DR.Q` 公开 checkpoint 在 HumanoidBench 全部 27 task 上的真实表现**——通关哪些、卡在哪些、哪些根本没 ckpt——并可一键打开 MuJoCo 查看每个任务的场景.
>
> _One-stop view of how the public `dmux/DR.Q` checkpoints fare across **all 27 HumanoidBench tasks** — what passes, what stalls, what has no public ckpt at all — plus one-click MuJoCo scene previews for every task._

本 notebook **只展示公开 ckpt**（`dmux/DR.Q`），不训练任何模型. 自训成绩（H1-walk DR.Q 90% / G1-walk DR.Q 70% / TD-MPC2 H1-walk 100%）见姊妹 notebook **`HumanoidBench-SelfTrained.ipynb`**.

_This notebook only shows **public** ckpts (`dmux/DR.Q`) and trains nothing. For self-trained numbers (H1-walk DR.Q 90% / G1-walk DR.Q 70% / TD-MPC2 H1-walk 100%) see the sibling notebook **`HumanoidBench-SelfTrained.ipynb`**._

数据来源 / Data sources:
- `results/drq_sweep_seed0/*.jsonl` — 28 task × seed 0 × 5 ep 全扫
- `results/drq_multiseed/*.jsonl`   — 9 task × 3 seeds × 10 ep 多种子复验
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
## §2  一键场景预览 / *One-click scene preview*

点下方按钮启动 MuJoCo 原生窗口，**只看任务初始场景**——不跑 policy，机器人维持初始姿态. 想看 DR.Q 真的能动起来的话，去姊妹 notebook `HumanoidBench-SelfTrained.ipynb`.

_Each button opens the MuJoCo viewer with the task at its **initial pose** — no policy, no rollout. Use this to inspect what each task scene looks like; for animated rollouts of self-trained ckpts, see `HumanoidBench-SelfTrained.ipynb`._

| 操作 / Action | 效果 / Effect |
|---|---|
| 左键拖 / left-drag | 旋转相机 / rotate camera |
| 右键拖 / right-drag | 平移 / pan |
| 滚轮 / scroll | 缩放 / zoom |
| Esc / close | 退出 / quit |

按钮覆盖 HumanoidBench 全部 27 个任务（h1hand 实施例 + 4 个 h1-only locomotion），按类别分组. 每次按下按钮启动一个新进程，**关掉窗口才能再启动下一个**.

_Buttons cover all 27 HumanoidBench tasks (h1hand embodiment + 4 h1-only locomotion), grouped by category. Each click spawns a new process — close the window before launching another._
"""))

cells.append(code("""
import subprocess, os
from ipywidgets import Button, VBox, HBox, Label, Output, Layout
from IPython.display import display

# Grouped by category — scene preview doesn't need ckpts, so we cover the full task suite.
SCENE_GROUPS = [
    ("🚶 Locomotion (h1)", [
        ("h1-walk-v0",            "走路 / Walk"),
        ("h1-run-v0",             "跑步 / Run"),
        ("h1-stand-v0",           "站立 / Stand"),
        ("h1-sit_simple-v0",      "坐下-易 / Sit easy"),
        ("h1-sit_hard-v0",        "坐下-难 / Sit hard"),
        ("h1-crawl-v0",           "爬行 / Crawl"),
        ("h1-pole-v0",            "绕杆 / Pole"),
        ("h1-stair-v0",           "上楼梯 / Stair"),
        ("h1-slide-v0",           "滑行 / Slide"),
        ("h1-hurdle-v0",          "跨栏 / Hurdle"),
        ("h1-maze-v0",            "迷宫 / Maze"),
        ("h1-balance_simple-v0",  "平衡-易 / Balance easy"),
        ("h1-balance_hard-v0",    "平衡-难 / Balance hard"),
        ("h1-reach-v0",           "到达 / Reach"),
    ]),
    ("✋ Locomotion (h1hand, w/ Shadow Hand)", [
        ("h1hand-walk-v0",        "走路 / Walk"),
        ("h1hand-run-v0",         "跑步 / Run"),
        ("h1hand-stand-v0",       "站立 / Stand"),
        ("h1hand-sit_simple-v0",  "坐下-易 / Sit easy"),
        ("h1hand-sit_hard-v0",    "坐下-难 / Sit hard"),
        ("h1hand-crawl-v0",       "爬行 / Crawl"),
        ("h1hand-pole-v0",        "绕杆 / Pole"),
        ("h1hand-stair-v0",       "上楼梯 / Stair"),
        ("h1hand-slide-v0",       "滑行 / Slide"),
        ("h1hand-reach-v0",       "到达 / Reach"),
    ]),
    ("🧰 Manipulation (h1hand)", [
        ("h1hand-door-v0",                "开门 / Door"),
        ("h1hand-basketball-v0",          "投篮 / Basketball"),
        ("h1hand-bookshelf_simple-v0",    "书架-易 / Bookshelf easy"),
        ("h1hand-bookshelf_hard-v0",      "书架-难 / Bookshelf hard"),
        ("h1hand-push-v0",                "推箱 / Push"),
        ("h1hand-package-v0",             "搬包裹 / Package"),
        ("h1hand-cube-v0",                "堆方块 / Cube"),
        ("h1hand-window-v0",              "擦窗 / Window"),
        ("h1hand-spoon-v0",               "勺运球 / Spoon"),
        ("h1hand-insert_normal-v0",       "插孔-常规 / Insert normal"),
        ("h1hand-insert_small-v0",        "插孔-小 / Insert small"),
        ("h1hand-kitchen-v0",             "厨房 / Kitchen"),
        ("h1hand-cabinet-v0",             "开柜 / Cabinet"),
        ("h1-highbar_simple-v0",          "单杠-易(h1) / Highbar easy"),
        ("h1strong-highbar_hard-v0",      "单杠-难(h1strong) / Highbar hard"),
        ("h1hand-truck-v0",               "卡车 / Truck"),
    ]),
]

out = Output()

def launch(task):
    def _click(_btn):
        with out:
            print(f"▶ scene  task={task}")
        subprocess.Popen(
            ["conda", "run", "-n", "humanoidbench", "--no-capture-output",
             "python", "scripts/scene_viewer.py", "--task", task],
            env={"DISPLAY": ":0", **os.environ},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    return _click

groups_ui = []
for title, items in SCENE_GROUPS:
    groups_ui.append(Label(value=title))
    row = []
    for task, label in items:
        b = Button(description=f"▶ {label}",
                   layout=Layout(width="220px"),
                   tooltip=task)
        b.on_click(launch(task))
        row.append(b)
        if len(row) == 3:
            groups_ui.append(HBox(row)); row = []
    if row: groups_ui.append(HBox(row))

display(VBox(groups_ui), out)
"""))

cells.append(md("""
## §3  全 27 task 成绩单 / *Full 27-task coverage table*

合并展示 **每一个 HumanoidBench task** 的状态 (3 类)：

| 状态 / Status | 含义 / Meaning |
|---|---|
| ✅ **passed (multi-seed)** | `dmux/DR.Q` ckpt 在 3 seeds × 10 ep 复验 ≥50% 成功率 |
| 🟡 **seed-0 only** | 仅 seed 0 × 5 ep 扫描结果（未做多种子复验） |
| ❌ **no public ckpt** | `dmux/DR.Q` 没发布对应权重 |

_Each row = one HumanoidBench task. Multi-seed validated rows take precedence; seed-0-only rows fill in the rest; manipulation tasks with no public ckpt are listed at the bottom._
"""))

cells.append(code("""
import json
from pathlib import Path

# ---- HumanoidBench 27 official tasks (per paper, h1hand + h1 variants) ----
# manipulation 7 + locomotion (h1) 12 + locomotion (h1hand) 8 = 27
HB_TASKS = [
    # h1 locomotion (no hand)
    "h1-walk-v0", "h1-run-v0", "h1-stand-v0", "h1-sit_simple-v0",
    "h1-sit_hard-v0", "h1-crawl-v0", "h1-pole-v0", "h1-stair-v0",
    "h1-slide-v0", "h1-hurdle-v0", "h1-maze-v0",
    "h1-balance_simple-v0", "h1-balance_hard-v0", "h1-reach-v0",
    # h1hand locomotion (Shadow Hand attached)
    "h1hand-walk-v0", "h1hand-run-v0", "h1hand-stand-v0",
    "h1hand-sit_simple-v0", "h1hand-sit_hard-v0", "h1hand-crawl-v0",
    "h1hand-pole-v0", "h1hand-stair-v0", "h1hand-slide-v0",
    "h1hand-reach-v0",
    # h1hand manipulation
    "h1hand-basketball-v0", "h1hand-door-v0",
    "h1hand-bookshelf_simple-v0", "h1hand-bookshelf_hard-v0",
    # manipulation tasks with NO public ckpt in dmux/DR.Q
    "h1hand-push-v0", "h1hand-package-v0", "h1hand-cube-v0",
    "h1hand-window-v0", "h1hand-spoon-v0", "h1hand-insert_normal-v0",
    "h1hand-insert_small-v0", "h1hand-kitchen-v0", "h1hand-cabinet-v0",
    "h1hand-highbar_hard-v0", "h1hand-highbar_simple-v0",
    "h1hand-truck-v0",
]


def load_summaries(dirpath):
    out = {}
    for p in sorted(Path(dirpath).glob("*.jsonl")):
        for line in p.open():
            d = json.loads(line)
            if d.get("_summary"):
                out[d["task"]] = d
                break
    return out


multiseed = load_summaries("results/drq_multiseed")     # 9 tasks, N=10 × 3 seeds
seed0     = load_summaries("results/drq_sweep_seed0")   # 28 tasks, N=5 × seed 0

rows = []
for task in HB_TASKS:
    if task in multiseed:
        r = multiseed[task]
        rows.append({
            "task": task,
            "status": "✅ passed" if r["success_rate"] >= 0.5 else "🟡 multi-seed",
            "success%": f"{r['success_rate']*100:.0f}%",
            "mean_return": f"{r['mean_return']:.1f}",
            "N": f"{r['eval_per_seed']} × {len(r['seeds'])} seeds",
            "source": "multiseed",
        })
    elif task in seed0:
        r = seed0[task]
        flag = "🟢 seed-0 pass" if r["success_rate"] >= 0.5 else "🔴 seed-0 fail"
        rows.append({
            "task": task,
            "status": flag,
            "success%": f"{r['success_rate']*100:.0f}%",
            "mean_return": f"{r['mean_return']:.1f}",
            "N": f"{r['n_episodes']} × 1 seed",
            "source": "seed0",
        })
    else:
        rows.append({
            "task": task,
            "status": "❌ no ckpt",
            "success%": "—",
            "mean_return": "—",
            "N": "—",
            "source": "none",
        })

# Sort: passed → multi-seed → seed-0 pass → seed-0 fail → no ckpt;
# within each bucket, by mean_return desc (numeric where possible).
status_rank = {
    "✅ passed": 0, "🟡 multi-seed": 1, "🟢 seed-0 pass": 2,
    "🔴 seed-0 fail": 3, "❌ no ckpt": 4,
}
def _ret(r):
    try: return -float(r["mean_return"])
    except ValueError: return 0.0
rows.sort(key=lambda r: (status_rank[r["status"]], _ret(r)))

try:
    import pandas as pd
    df = pd.DataFrame(rows)
    display(df)
except ImportError:
    print(f"{'task':<32} {'status':<18} {'succ':>5} {'mean':>9} {'N':>16}")
    for r in rows:
        print(f"{r['task']:<32} {r['status']:<18} {r['success%']:>5} {r['mean_return']:>9} {r['N']:>16}")

n_passed = sum(1 for r in rows if r["status"] == "✅ passed")
n_seed0  = sum(1 for r in rows if r["status"] == "🟢 seed-0 pass")
n_noctk  = sum(1 for r in rows if r["status"] == "❌ no ckpt")
print(f"\\n通关 / Passed (multi-seed ≥50%) : {n_passed} / {len(rows)}")
print(f"仅 seed-0 通过 / Seed-0 only pass: {n_seed0} / {len(rows)}  (需多种子复验 / pending multiseed)")
print(f"无公开 ckpt / No public ckpt    : {n_noctk} / {len(rows)}  (manipulation gap)")
"""))

cells.append(md("""
## §4  任务说明 / *Task descriptions*

每个 task 的 success 判定：
- **locomotion / reach**：`ep_return ≥ task.success_bar`（自带阈值）
- **manipulation** (`door · basketball · bookshelf · push · package · cube · window · spoon · insert · kitchen · cabinet · highbar · truck`)：`info['success']==True`（来自任务自带 reward / event）

_Success criteria: locomotion uses return threshold; manipulation uses `info['success']` flag from the task itself._
"""))

cells.append(code("""
TASK_INFO = {
    # ── locomotion ──
    "walk":             ("走路 / Walk forward",                        700),
    "run":              ("跑步 / Run forward",                         700),
    "stand":            ("站立平衡 / Standing balance",                800),
    "sit_simple":       ("坐到椅子上（简单） / Sit on chair (easy)",    750),
    "sit_hard":         ("坐到椅子上（困难） / Sit on chair (hard)",    750),
    "crawl":            ("钻洞爬行 / Crawl under obstacle",            700),
    "pole":             ("绕杆穿行 / Slalom around poles",             700),
    "stair":            ("上楼梯 / Climb stairs",                      700),
    "slide":            ("滑行平衡 / Slide stance",                    700),
    "hurdle":           ("跨栏 / Hurdle jump",                         700),
    "balance_simple":   ("窄板平衡（易） / Narrow-beam balance",        800),
    "balance_hard":     ("窄板平衡（难） / Narrow-beam balance hard",   800),
    "maze":             ("走迷宫 / Maze navigation",                  1200),
    "reach":            ("手部到达目标 / Reach hand to target",      12000),
    # ── manipulation (info['success']) ──
    "basketball":       ("投篮 / Shoot basketball through hoop",     "info"),
    "door":             ("开门 / Open the door",                     "info"),
    "bookshelf_simple": ("书架取书（简单） / Pick book from shelf (easy)", "info"),
    "bookshelf_hard":   ("书架取书（困难） / Pick book from shelf (hard)", "info"),
    "push":             ("推箱子到目标 / Push box to target",         "info"),
    "package":          ("搬包裹 / Carry package",                    "info"),
    "cube":             ("双手堆方块 / Stack cubes with both hands",  "info"),
    "window":           ("擦窗 / Wipe window",                        "info"),
    "spoon":            ("勺子运球 / Move ball on spoon",             "info"),
    "insert_normal":    ("插孔（常规） / Peg insertion (normal)",     "info"),
    "insert_small":     ("插孔（小孔） / Peg insertion (small)",      "info"),
    "kitchen":          ("厨房多步任务 / Kitchen multi-step",         "info"),
    "cabinet":          ("开柜门 / Open cabinet",                     "info"),
    "highbar_simple":   ("单杠悬挂（易） / High-bar hang (easy)",      "info"),
    "highbar_hard":     ("单杠引体（难） / High-bar pull-up (hard)",   "info"),
    "truck":            ("装卸卡车 / Truck loading",                  "info"),
}

# 对照成绩单看每个 task 在做什么
import json
from pathlib import Path
all_summary = {}
for p in sorted(Path("results/drq_multiseed").glob("*.jsonl")):
    for line in p.open():
        d = json.loads(line)
        if d.get("_summary"): all_summary[d["task"]] = (d, "multiseed"); break
for p in sorted(Path("results/drq_sweep_seed0").glob("*.jsonl")):
    if any(p.name == k + ".jsonl" for k in all_summary): continue
    for line in p.open():
        d = json.loads(line)
        if d.get("_summary"): all_summary[d["task"]] = (d, "seed0"); break

print(f"{'task':<32} {'succ':>5} {'src':<10} description")
print("-" * 95)
for task, (r, src) in sorted(all_summary.items(), key=lambda x: -x[1][0]["success_rate"]):
    short = task.replace("h1hand-", "").replace("h1-", "").replace("-v0", "")
    desc, bar = TASK_INFO.get(short, ("?", "?"))
    print(f"{task:<32} {r['success_rate']*100:4.0f}% {src:<10} {desc}  [bar={bar}]")
"""))

cells.append(md("""
## §5  完整 eval 复跑 / *Reproduce the full eval*

如果想自己复现成绩单（每 task ~30 sec on RTX 4090）：

_To reproduce the table (about 30 sec per task on an RTX 4090):_

```bash
# 28-task quick scan, seed 0, 5 ep   (results/drq_sweep_seed0/)
bash scripts/sweep_drq.sh

# Multi-seed validation on candidates that passed seed-0  (results/drq_multiseed/)
bash scripts/sweep_drq_multiseed.sh

# 单 task 自定义
python scripts/eval.py --task h1-sit_simple-v0 --driver drq \\
    --eval 10 --seed_list 0,10,20 --action_repeat 2 \\
    --out results/_my_sit_simple.jsonl
```
"""))

cells.append(md("""
## §6  局限 / *Limitations*

- ❌ **12 个 manipulation task 完全没有公开 ckpt** (`push · package · cube · window · spoon · insert_normal · insert_small · kitchen · cabinet · highbar_simple · highbar_hard · truck`)，必须自训.
- ⚠️  **4 个 manipulation 有 ckpt 但未达 50%** (`basketball · door · bookshelf_simple · bookshelf_hard`)：任务本身困难，社区训练未收敛.
- ⚠️  **12 个 locomotion seed-0 失败**：`stair / slide / hurdle / maze / reach / balance_simple / balance_hard` 以及全部 `h1hand-*` 行走类（带 Shadow Hand 后 act_dim 61，更难训）.
- ⚠️  Seed 间方差大：`walk-v0` seed 0/10/20 mean return 530 / 295 / 382. 不要用单 seed 评估.
- ⚠️  Eval 必须传 `--action_repeat 2`，否则 step rate 错位，return 减半.
- ➡️  本 notebook **只 show 场景**——想看 policy 真的动起来（DR.Q 自训 H1-walk 90% / G1-walk 70% / TD-MPC2 H1-walk 100%），请打开姊妹 notebook **`HumanoidBench-SelfTrained.ipynb`**.

_Caveats: 12 manipulation tasks have no public ckpt; 4 manipulation tasks have ckpts that don't reach 50%; 12 locomotion seed-0 fails (mostly h1hand variants); large seed variance; remember `--action_repeat 2`. This notebook only previews scenes — for animated policy rollouts (self-trained DR.Q / TD-MPC2 ckpts that beat baseline), see the sibling notebook._
"""))


nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "humanoidbench", "display_name": "Python (humanoidbench)"},
    "language_info": {"name": "python"},
})

OUT.write_text(nbf.writes(nb))
print(f"✅ wrote {OUT} ({len(cells)} cells)")
