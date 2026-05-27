"""Generate Datasets.ipynb — G1 公开数据集一键下载 + 多场景预览.

_One-shot notebook builder: thin ipywidgets buttons that delegate to
`scripts/dataset_download.py` and `scripts/dataset_preview.py`._

Run:
    conda run -n humanoidbench python scripts/build_datasets_nb.py
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Datasets.ipynb"


def md(src: str) -> dict:
    return nbf.v4.new_markdown_cell(src.lstrip("\n"))


def code(src: str) -> dict:
    return nbf.v4.new_code_cell(src.lstrip("\n"))


cells: list[dict] = []

cells.append(md("""
# Unitree G1 公开数据集一键预览 · *G1 Datasets One-Click Preview*

配套文档 / Companion doc: [`docs/g1_lerobot_datasets.html`](docs/g1_lerobot_datasets.html)

本 notebook 从 HF 拉取 G1 公开数据集，启动 LeRobot 仿真预览. 三阶段：

_Three-stage flow:_

| § | 阶段 / Stage | 说明 / What |
| --- | --- | --- |
| §1 | 环境检查 / env check | CLI + 4 个 hidden runtime deps |
| **§2** | **下载 / download** | 按分类多按钮，调 `scripts/dataset_download.py` 拉到 HF 默认 cache (`~/.cache/huggingface/hub/`) |
| **§3** | **预览 / preview** | 每个数据集 2–4 个 episode × A/B mode 按钮，调 `scripts/dataset_preview.py` |
| §4 | 完整候选清单 / live HF list | 所有 `unitree_g1` 关键词 dataset |
| §5 | Headless inline 预览 / fallback | 无 GUI 时用 `LeRobotDataset` API |
| §6 | 环境快速安装 / quick install | conda + lerobot[unitree_g1] |

> ⚠️ 数据集都下载到 **HF 默认 cache** (`$HF_HOME/hub/` 或 `~/.cache/huggingface/hub/`)，已 cache 第二次点击秒返回.
> ⚠️ 按钮调用的脚本：`scripts/dataset_{download,preview}.py`，notebook 只负责 UI 胶水.
"""))

# ───────────────── §1 env check ─────────────────
cells.append(md("""
## §1  环境检查 / *Environment check*

检测 `lerobot-replay` / `lerobot-dataset-viz` CLI **+ 4 个隐藏 python runtime deps**（`unitree_sdk2py` · `loguru` · `msgpack_numpy` · `mujoco`）.

⚠️ **A 路径必踩的坑**：即使 `--robot.is_simulation=true`，lerobot G1 **依然强制 import `unitree_sdk2py`**（DDS bridge 设计耦合）. 缺以上任一 → 子进程闪退 ~12 秒，log 才能看到 `ModuleNotFoundError`. 下一格自动给出一键修复命令.

_Beyond CLI presence: watch 4 hidden deps. Missing any → sim dies in ~12 s; you only see the error in `.viewer_logs/`. The fix cell below prints copy-pasteable install commands._
"""))

cells.append(code("""
from scripts.dataset_ui import check_env
check_env()
"""))

# ───────────────── §2 download ─────────────────
cells.append(md("""
## §2  一键下载 / *One-click download*

按数据集分类分组，每个按钮调 `scripts/dataset_download.py --repo-id X`，拉到 HF 默认 cache.

_Buttons grouped by dataset category. Each click spawns `scripts/dataset_download.py` to pull the repo into HF's default cache (`~/.cache/huggingface/hub/`)._

| 分类 / Category | 含义 / Meaning |
| --- | --- |
| 🎮 **Sim/Synthetic** | 仿真生成（RoboCasa contact-aware · ManiSkill recovery） |
| ✋ **Dex3-1 teleop** | Unitree Dex3-1 五指灵巧手真机遥操作 |
| 📦 **Other teleop** | 其它手部/末端 teleop |
| 🧪 **Sanity** | 最小数据集，用于 GUI/pipeline 烟测 |

点击后 stdout 实时显示 cache 路径；已下载的第二次秒返回.
"""))

cells.append(code("""
# UI 逻辑 + 数据表都在 scripts/dataset_ui.py — 改数据集列表去那里
# _UI logic and dataset tables live in scripts/dataset_ui.py — edit there to add datasets_
from scripts.dataset_ui import build_download_panel
build_download_panel()
"""))

# ───────────────── §3 preview ─────────────────
cells.append(md("""
## §3  多场景预览 / *Multi-scene preview*

每个数据集多个 episode × **B (Rerun GUI)** 按钮. 调 `scripts/dataset_preview.py --repo-id X --mode B --episode N`，子进程独立日志.

_Each dataset exposes 1–3 episode buttons in mode B (Rerun viz). Each click spawns `scripts/dataset_preview.py` as a detached subprocess._

| 标记 / Tag | 后端 / Backend | 状态 / Status |
|---|---|---|
| 🔵 **B** | `lerobot-dataset-viz` + Rerun | ✅ 已验证 GUI 真起来 (Vulkan + RTX4090 detected) |
| 🟠 ~~A~~ | `lerobot-replay` + MuJoCo sim | ⚠️ 当前 lerobot 触发 `TypeError: unhashable type: 'list'`（action 处理 bug），按钮**暂时移除** |

### 已知约束 / Known constraints

1. **数据集必须有 `v3.0` git tag**：lerobot `CODEBASE_VERSION="v3.0"` 硬编码于 `dataset_metadata.py:55`. 缺 tag → `RevisionNotFoundError`（HF lib API 变更又触发 `TypeError: response missing`）；只有 `v2.0/v2.1` → `BackwardCompatibilityError → NotImplementedError("Contact maintainer on Discord")`. 已剔除 `jnsungp/...robocasa-...` · `afabisch/...leave_room` (无 tag) 和 `johnMinelli/ManiSkill_...` (v2.0 太老).
2. **数据集必须先下载到 lerobot cache** (`~/.cache/huggingface/lerobot/<repo_id>/`)：`scripts/dataset_download.py` 已默认指向这里，§2 按钮点完即可用. 如果之前在标准 HF hub cache 下过 (`datasets--*/snapshots/...`)，先去 §2 重新点一次下载（秒级，已 cache 文件会复用，但 lerobot 期望的路径要建好）.

⚠️ 前置：先在 §2 下载对应数据集（已 cache 第二次秒返回）.

_Caveats: (1) lerobot's `get_safe_version` requires a version tag — tagless repos crash with a cascading `TypeError`. (2) Data must live in `~/.cache/huggingface/lerobot/<repo>/`, not the standard HF hub cache — `scripts/dataset_download.py` writes to the right place by default._
"""))

cells.append(code("""
# 同上：场景 + ep 索引列表都在 scripts/dataset_ui.py 的 PREVIEW_SCENES
# _Scene/episode list lives in scripts/dataset_ui.py (PREVIEW_SCENES table)_
from scripts.dataset_ui import build_preview_panel
build_preview_panel()
"""))

# ───────────────── §4 candidate list ─────────────────
cells.append(md("""
## §4  完整候选清单 / *Full candidate list*

下游表格直接从 HF API 拉，包含所有 `unitree_g1` 关键词 dataset. 可自行复制 repo_id 到 §2/§3 里手动添加.

_All datasets with `unitree_g1` keyword (live from HF API)._
"""))

cells.append(code("""
from scripts.dataset_ui import show_hf_candidates
show_hf_candidates(keyword="unitree_g1")
"""))

# ───────────────── §5 inline headless preview ─────────────────
cells.append(md("""
## §5  Inline 数据浏览 / *Inline preview without sim*

极简兜底：当 GUI 不可用（headless server / X11 失败）时，用 `LeRobotDataset` API 直接 inline 显示 dataset 元信息.

_Headless fallback: use `LeRobotDataset` API to print episode/frame counts inline._
"""))

cells.append(code("""
# 默认查 Breno DEX3 kettle (确认能加载) ；想看其它的传 repo_id=
from scripts.dataset_ui import inline_dataset_info
inline_dataset_info(repo_id="Breno-de-Angelo/unitree-g1-dex3-1-pick-kettle-v3")
"""))

# ───────────────── §6 quick install ─────────────────
cells.append(md("""
## §6  附：环境快速安装 / *Quick install*

参考 sister repo `/home/david/work/isaaclab-experience/lerobot/docs/source/unitree_g1.mdx`:

```bash
conda create -y -n lerobot python=3.12 && conda activate lerobot
conda install -c conda-forge 'pinocchio>=3.0.0,<4.0.0' ffmpeg -y

# Unitree SDK (PyPI 不发布，必须从 GitHub)
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python && pip install -e . && cd ..

# LeRobot with G1 extras
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e '.[unitree_g1]'

# MuJoCo sim deps
pip install mujoco loguru msgpack msgpack-numpy
```

完成后，§1 应该能找到 CLI；§2/§3 按钮可一键启动.
"""))


nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "humanoidbench", "display_name": "Python (humanoidbench)"},
    "language_info": {"name": "python"},
})
OUT.write_text(nbf.writes(nb))
print(f"✅ wrote {OUT} ({len(cells)} cells)")
