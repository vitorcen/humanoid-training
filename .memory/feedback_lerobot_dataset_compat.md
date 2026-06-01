---
name: lerobot-dataset-compat
description: lerobot dataset loading is brittle — needs v3.0 git tag + lerobot cache path; surface this early when wiring HF→lerobot pipelines
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0182c875-57e6-499e-a050-108e8fa8ab3e
---

LeRobot 加载 HF 数据集踩三个隐藏坑，全要满足才不崩：

1. **必须 `~/.cache/huggingface/lerobot/<repo_id>/` 路径**，不是 HF hub 默认的 `~/.cache/huggingface/hub/datasets--*/snapshots/<sha>/`. `snapshot_download(local_dir=...)` 必须显式指定. 否则 `LeRobotDataset(repo)` 拿不到 `meta/info.json`.

2. **必须有 `v3.0` git tag 且 `meta/info.json` 里的 `codebase_version` 也是 v3.x**. `CODEBASE_VERSION = "v3.0"` 在 `lerobot/datasets/dataset_metadata.py:55`. 两层都要查 —— git tag 写 v3.0 但 info.json 仍 v2.1 的仓库（eg `memory-vla/unitree_g1_dex3_press_knob`）依然会 `BackwardCompatibilityError`. `v2.0`/`v2.1` 触发 `BackwardCompatibilityError` → `NotImplementedError("Contact maintainer on Discord")`. 完全无 tag 触发 `RevisionNotFoundError` → 再撞 huggingface_hub API 变更 bug `TypeError: HfHubHTTPError.__init__() missing 'response'`. **下载后立即校验 info.json**：`python -c "import json; print(json.load(open('~/.cache/huggingface/lerobot/<repo>/meta/info.json'))['codebase_version'])"`

3. **Mode A (`lerobot-replay` sim) 不可用**：触发 `TypeError: unhashable type: 'list'` 在 `replay_loop` 的 `action[name] = action_array[i]`. 是 lerobot 内部 action 处理 bug，跟数据集无关. Mode B (`lerobot-dataset-viz` + Rerun) 稳定可用.

4. **跨 env 调 lerobot-dataset-viz 必须 fix PATH**：rerun-sdk 的 `rr.init()` 内部会 fork+exec 一个独立的 `rerun` viewer binary（GUI 渲染进程），靠 `$PATH` 查找. 如果父进程不在 lerobot env (eg notebook kernel 在 humanoidbench)，子进程继承父 PATH 找不到 `~/miniconda3/envs/lerobot/bin/rerun` → `RuntimeError: Failed to find Rerun Viewer executable in PATH`. 必须显式 `PATH=lerobot_env_bin:$PATH` 传给 Popen.

5. **失败 load 会污染 HF datasets 的 arrow cache**：lerobot-dataset-viz 内部 `Dataset.from_parquet(paths, filters=isin([ep]))` 会在 `~/.cache/huggingface/lerobot/<repo>/.cache/huggingface/` 下建 arrow 缓存. 如果第一次因为别的 bug (PATH/version) 中途崩了，缓存写了个空目录. 之后再 load 同 episode → `ValueError: Instruction "train" corresponds to no data!`（HF datasets builder 把空缓存当成"没数据"返回）. 修复：下载完后 `shutil.rmtree(<repo>/.cache, ignore_errors=True)`，重建只 1-2s 开销.

**Why:** 这三个 bug 调试一上午（用户点按钮没反应 → log 才看到错误）. 任何 HF→lerobot pipeline 都会撞上.

**How to apply:**
- 给 lerobot 写下载脚本一律 `local_dir=~/.cache/huggingface/lerobot/<repo>/`
- 列数据集前先 `HfApi.list_repo_refs(repo, repo_type='dataset')` 过滤 `v3*` tag
- 暂时只暴露 mode B (Rerun) 按钮，mode A 等 lerobot 修
- 用户没 GUI 反馈时检查 daemon thread 用 `Output.append_stdout()` 而非 `with output: print()`（后者跨线程不推送）

相关上下文：[[feedback_train_with_watcher]] 类似的"GUI 进度不可见"教训.
