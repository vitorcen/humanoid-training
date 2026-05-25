---
name: feedback-bilingual-docs
description: Docs/notebooks must carry italic English captions alongside Chinese for key descriptions, following isaaclab-experience/README.md style
metadata:
  type: feedback
---

When writing docs, notebooks, and README content for this project family (humanoid-training, mujoco-experience, isaaclab-experience), **always pair key Chinese descriptions with an italic English caption** for open-source collaboration.

**Why:** 用户在做开源人形机器人工作，文档面向中英文双语读者；同时已在全局 CLAUDE.md 写下「中英对照」规则。`/home/david/work/isaaclab-experience/README.md` 是 canonical reference style — every major section there carries the bilingual pattern.

**How to apply:**

1. **Section headers** — 中文标题，空一行，下一行 `_Italic English subtitle_`. Example:
   ```markdown
   ## 2. 创建并安装 conda 环境（幂等）

   _Create and install the conda env (idempotent)_
   ```

2. **TL;DR / key paragraphs** — write Chinese sentence, then a separate line of `_Italic English equivalent_`:
   ```markdown
   官方要求 Python 3.11，setup.py 钉死 mujoco==3.1.6。
   _Upstream requires Python 3.11 and pins mujoco==3.1.6 in setup.py._
   ```

3. **Callout / warning blocks** — title is bilingual on the same line; body has Chinese line + italic-English line:
   ```markdown
   > ⚠️ **独立 env 是刚需 / Isolated env is mandatory**
   >
   > humanoid-bench 的 mujoco/torch 版本会和 KungfuBot 冲突...
   >
   > _humanoid-bench's mujoco/torch versions clash with KungfuBot..._
   ```

4. **Tables** — header cells bilingual `| 中文 / English |`; body cells follow `中文 · english` or full bilingual where helpful.

5. **FAQ entries** — Q line bilingual `**Q: 中文 / English**`; answer paragraph Chinese + italic English follow-up.

6. **Inline list items (key bullets)** — `**中文 / English**: ...` or for longer items, Chinese line then `  _italic English line_` underneath.

7. **What NOT to do:**
   - Don't add italics to throwaway phrases, code comments, or trivial labels.
   - Don't put both languages mid-sentence — keep them on separate lines or separated by `/` / `·`.
   - Don't translate filenames, env IDs, CLI flags — those stay verbatim.

**Concrete reference:** Read `/home/david/work/isaaclab-experience/README.md` (especially §VLA 推理实测 and §BEHAVIOR-1K) for the canonical implementation. `/home/david/work/humanoid-training/HumanoidBench.ipynb` is the in-project example following this rule.

Related: this is a project-specific concretization of the global `中英对照` rule in `~/.claude/CLAUDE.md`. The global rule says *what*; this memory says *what shape, with examples, anchored to a reference file*.
