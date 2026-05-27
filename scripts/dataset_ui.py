"""All UI/helper logic for Datasets.ipynb — env check, download + preview panels, HF list, inline info.

_Notebook cells call thin entry points; data tables at bottom of file._

    from scripts.dataset_ui import (
        check_env,               # §1
        build_download_panel,    # §2
        build_preview_panel,     # §3
        show_hf_candidates,      # §4
        inline_dataset_info,     # §5
    )
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

from ipywidgets import Button, HBox, Label, Layout, Output, VBox

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / ".viewer_logs"
LOG_DIR.mkdir(exist_ok=True)


# ────────────────── §1: env check ──────────────────

def _find_lerobot_bin(name: str) -> str | None:
    p = shutil.which(name)
    if p:
        return p
    for env in ("lerobot", "lerobot-v040"):
        cand = Path.home() / "miniconda3" / "envs" / env / "bin" / name
        if cand.is_file():
            return str(cand)
    return None


def check_env() -> None:
    """Detect lerobot CLI + 4 hidden runtime deps; print one-line fix if missing."""
    replay = _find_lerobot_bin("lerobot-replay")
    viz    = _find_lerobot_bin("lerobot-dataset-viz")
    print("=== CLI ===")
    print(f"  lerobot-replay      = {replay or '❌ NOT FOUND'}")
    print(f"  lerobot-dataset-viz = {viz or '❌ NOT FOUND'}")

    lerobot_py = str(Path(replay or viz or "").parent / "python") if (replay or viz) else None
    required = ["lerobot", "mujoco", "loguru", "msgpack_numpy", "unitree_sdk2py"]
    missing: list[str] = []
    if lerobot_py and Path(lerobot_py).is_file():
        print(f"\n=== runtime deps (in {lerobot_py}) ===")
        for mod in required:
            r = subprocess.run([lerobot_py, "-c", f"import {mod}"], capture_output=True, timeout=15)
            ok = r.returncode == 0
            print(f"  {mod:18s} = {'✅' if ok else '❌'}")
            if not ok:
                missing.append(mod)
    else:
        print("\n⚠️  没找到 lerobot python — 先按 §6 装环境")

    if missing:
        print(f"\n🔧 缺 {len(missing)} 个 / Missing — 一键修复:")
        pip_pkgs = [{"msgpack_numpy": "msgpack-numpy"}.get(m, m)
                    for m in missing if m != "unitree_sdk2py"]
        if pip_pkgs:
            print(f"  {lerobot_py} -m pip install {' '.join(pip_pkgs)}")
        if "unitree_sdk2py" in missing:
            print("  git clone https://github.com/unitreerobotics/unitree_sdk2_python /tmp/usdk2 && \\")
            print(f"    {lerobot_py} -m pip install --no-deps -e /tmp/usdk2")
    elif lerobot_py:
        print("\n✅ 全部就位 — §2 可以一键下载、§3 可以一键预览")


# ────────────────── §2: download panel ──────────────────

_dl_running: dict[str, subprocess.Popen] = {}


def _pump_download(proc: subprocess.Popen, log_path: Path,
                   repo_id: str, btn: Button, out: Output) -> None:
    """Mirror child stdout to widget (thread-safe via append_stdout)."""
    btn.button_style = "warning"
    btn.description = btn.description.replace("⬇", "⏳", 1)
    tag = repo_id.split("/")[-1][:32]
    with log_path.open("w") as lf:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            lf.write(line + "\n"); lf.flush()
            out.append_stdout(f"   [{tag}] {line}\n")
    rc = proc.wait()
    out.append_stdout(f"{'✅' if rc == 0 else '❌'} exit {rc}  {repo_id}\n\n")
    btn.button_style = "success" if rc == 0 else "danger"
    btn.description = btn.description.replace("⏳", "✅" if rc == 0 else "❌", 1)
    _dl_running.pop(repo_id, None)


def _on_download_click(repo_id: str, btn: Button, out: Output) -> None:
    if repo_id in _dl_running:
        out.append_stdout(f"⚠️  already running: {repo_id}\n")
        return
    log = LOG_DIR / f"download__{repo_id.replace('/', '__')}.log"
    out.append_stdout(f"▶ download  {repo_id}\n")
    proc = subprocess.Popen(
        [sys.executable, "-u", "scripts/dataset_download.py", "--repo-id", repo_id],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    _dl_running[repo_id] = proc
    out.append_stdout(f"   pid={proc.pid}  log={log}\n")
    threading.Thread(
        target=_pump_download, args=(proc, log, repo_id, btn, out), daemon=True
    ).start()


def build_download_panel() -> VBox:
    """Render §2 download buttons grouped by category."""
    out = Output()
    ui: list = []
    for cat_title, items in DATASETS_BY_CATEGORY:
        ui.append(Label(value=cat_title))
        for repo, desc in items:
            b = Button(description=f"⬇ {desc}",
                       tooltip=repo, button_style="info",
                       layout=Layout(width="780px"))
            b.on_click(lambda _btn, r=repo: _on_download_click(r, _btn, out))
            ui.append(b)
    return VBox([*ui, out])


# ────────────────── §3: preview panel ──────────────────

_pv_running: dict[tuple, subprocess.Popen] = {}


def _pump_preview(proc: subprocess.Popen, log_path: Path,
                  repo_id: str, mode: str, ep: int,
                  btn: Button, out: Output, max_lines: int = 40) -> None:
    """Mirror first N lines of preview child to widget; rest goes to log only."""
    btn.button_style = "warning"
    tag = f"{repo_id.split('/')[-1][:24]} {mode}·ep{ep}"
    seen = 0
    with log_path.open("w") as lf:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            lf.write(line + "\n"); lf.flush()
            seen += 1
            if seen <= max_lines:
                out.append_stdout(f"   [{tag}] {line}\n")
                if seen == max_lines:
                    out.append_stdout(
                        f"   [{tag}] … (truncated, tail -f {log_path})\n"
                    )
    rc = proc.wait()
    if rc == 0:
        out.append_stdout(f"✅ exit 0  {tag}\n\n"); btn.button_style = "success"
    else:
        out.append_stdout(f"❌ exit {rc}  {tag}  → {log_path}\n\n"); btn.button_style = "danger"
    _pv_running.pop((repo_id, mode, ep), None)


def _on_preview_click(repo_id: str, mode: str, ep: int,
                     btn: Button, out: Output) -> None:
    key = (repo_id, mode, ep)
    if key in _pv_running:
        out.append_stdout(f"⚠️  already running: {repo_id} {mode}·ep{ep}\n")
        return
    log = LOG_DIR / f"preview__{repo_id.replace('/', '__')}__{mode}_ep{ep}.log"
    out.append_stdout(f"▶ preview  repo={repo_id}  mode={mode}  ep={ep}\n")
    proc = subprocess.Popen(
        [sys.executable, "-u", "scripts/dataset_preview.py",
         "--repo-id", repo_id, "--mode", mode, "--episode", str(ep)],
        cwd=str(ROOT),
        env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0"),
             "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    _pv_running[key] = proc
    out.append_stdout(f"   pid={proc.pid}  stop: !kill {proc.pid}\n")
    threading.Thread(
        target=_pump_preview, args=(proc, log, repo_id, mode, ep, btn, out), daemon=True
    ).start()


def show_hf_candidates(keyword: str = "unitree_g1", limit: int = 50) -> None:
    """§4: live HF API list of datasets matching `keyword`, rendered as HTML table."""
    from IPython.display import HTML, display

    url = f"https://huggingface.co/api/datasets?search={keyword}&limit={limit}"
    data = json.loads(urllib.request.urlopen(url, timeout=10).read())

    rows: list[tuple] = []
    for d in data:
        rid = d["id"]; dl = d.get("downloads", 0); tags = d.get("tags", [])
        size = next((t.split(":", 1)[1] for t in tags if t.startswith("size_categories:")), "—")
        lic  = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), "—")
        is_lerobot = "LeRobot" in tags or any("lerobot" in t.lower() for t in tags)
        rows.append((rid, dl, size, lic, "✅" if is_lerobot else ""))
    rows.sort(key=lambda x: -x[1])

    html = ["<table style='font-family:monospace;font-size:0.85em'>"]
    html.append("<tr><th>repo_id</th><th>dl</th><th>size</th><th>license</th><th>LeRobot tag</th></tr>")
    for r in rows:
        html.append(
            f"<tr><td><a href='https://huggingface.co/datasets/{r[0]}' target='_blank'>{r[0]}</a></td>"
            f"<td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td></tr>"
        )
    html.append("</table>")
    display(HTML("".join(html)))
    print(f"\n共 {len(rows)} 个 dataset（live from HF API, keyword={keyword!r}）")


def inline_dataset_info(repo_id: str = "afabisch/unitree_g1_leave_room_old") -> None:
    """§5: print dataset metadata via `LeRobotDataset` API (headless fallback, no GUI)."""

    def _find_lerobot_python():
        try:
            import lerobot  # noqa
            return sys.executable
        except ImportError:
            pass
        for env in ("lerobot", "lerobot-v040"):
            cand = Path.home() / "miniconda3" / "envs" / env / "bin" / "python"
            if cand.is_file():
                return str(cand)
        return None

    py = _find_lerobot_python()
    print(f"lerobot python = {py}")
    if py is None:
        print("❌ 没找到能 import lerobot 的 python — 装 lerobot 后重试")
        return

    script = f"""
import json
from lerobot.datasets import LeRobotDataset
ds = LeRobotDataset({repo_id!r})
print(json.dumps({{
    'num_episodes':    ds.num_episodes,
    'num_frames':      ds.num_frames,
    'fps':             ds.fps,
    'features':        sorted(ds.features.keys()),
    'episode_lengths': [int(x) for x in ds.episode_data_index['to'][:5].tolist()],
}}, indent=2))
"""
    try:
        r = subprocess.run([py, "-c", script], capture_output=True, text=True, timeout=120)
        print(r.stdout)
        if r.returncode != 0:
            print("--- stderr ---"); print(r.stderr[-2000:])
    except subprocess.TimeoutExpired:
        print("⚠️  timeout — dataset 加载超 120s")


def build_preview_panel() -> VBox:
    """Render §3 preview buttons: per-dataset row of (mode, ep) buttons."""
    out = Output()
    ui: list = []
    for cat_title, datasets in PREVIEW_SCENES:
        ui.append(Label(value=cat_title))
        for repo, label, scenes in datasets:
            ui.append(Label(value=f"  └ {label}  ({repo})"))
            row: list = []
            for mode, ep in scenes:
                color = "warning" if mode == "A" else "info"
                tag = "🟠" if mode == "A" else "🔵"
                b = Button(description=f"{tag} {mode}·ep{ep}",
                           tooltip=f"{repo}  mode={mode}  episode={ep}",
                           button_style=color,
                           layout=Layout(width="150px"))
                b.on_click(
                    lambda _btn, r=repo, m=mode, e=ep:
                        _on_preview_click(r, m, e, _btn, out)
                )
                row.append(b)
            ui.append(HBox(row))
    return VBox([*ui, out])


# ────────────────── data tables ──────────────────
# ⚠️ lerobot CODEBASE_VERSION="v3.0" — git tag AND meta/info.json's codebase_version
# both must start with "v3". `memory-vla/press_knob` has v3.0 tag but v2.1 info.json
# → BackwardCompatibilityError, excluded. Episode counts below come from each
# dataset's meta/info.json after first download.

DATASETS_BY_CATEGORY = [
    ("✋ Dex3-1 teleop (5指灵巧手)", [
        ("Breno-de-Angelo/unitree-g1-dex3-1-pick-kettle-v3",
         "DEX3-1 pick kettle · 30 ep · v3.0"),
        ("unitreerobotics/G1_Dex3_Pouring_Dataset",
         "Unitree official · Dex3 倒水 · 311 ep · v3.0"),
    ]),
    ("📦 General teleop / community", [
        ("nepyope/unitree_box_move_blue_full",
         "Box move · Homunculus exo · 550 ep · v3.0"),
        ("zeeshaan-ai/unitree-g1-dataset-apple-pick-place",
         "苹果抓放 · 103 ep · v3.0"),
    ]),
    ("🏢 Unitree official", [
        ("unitreerobotics/G1_BlockStacking_Dataset",
         "Unitree · 堆方块 · 301 ep · v3.0"),
    ]),
]

# Only mode B (Rerun). Mode A triggers lerobot's
# `TypeError: unhashable type: 'list'` in replay_loop — internal bug, suppressed.
PREVIEW_SCENES = [
    ("✋ Dex3-1 teleop", [
        ("Breno-de-Angelo/unitree-g1-dex3-1-pick-kettle-v3",
         "DEX3-1 kettle (30 ep)",
         [("B", 0), ("B", 15), ("B", 29)]),
        ("unitreerobotics/G1_Dex3_Pouring_Dataset",
         "Unitree Dex3 倒水 (311 ep)",
         [("B", 0), ("B", 150), ("B", 310)]),
    ]),
    ("📦 General teleop / community", [
        ("nepyope/unitree_box_move_blue_full",
         "Box move Homunculus (550 ep)",
         [("B", 0), ("B", 275), ("B", 549)]),
        ("zeeshaan-ai/unitree-g1-dataset-apple-pick-place",
         "苹果抓放 (103 ep)",
         [("B", 0), ("B", 50), ("B", 102)]),
    ]),
    ("🏢 Unitree official", [
        ("unitreerobotics/G1_BlockStacking_Dataset",
         "Unitree 堆方块 (301 ep)",
         [("B", 0), ("B", 150), ("B", 300)]),
    ]),
]
