---
name: feedback-commit-style
description: "User commits manually — base two short English Conventional Commit candidates on the actual uncommitted diff (not abstract milestones)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 7d4c3d5c-6c21-470f-acd6-6e6460eaf62e
---

When a commit point is reached, do NOT run `git commit` yourself. Instead, **inspect what is actually uncommitted** and present **two** English short-sentence options in Conventional Commits format: `feat(scope): xxx xxx ... xxx xxx` (or `fix`, `chore`, `docs`, `refactor`, etc.). The user picks one and commits themselves.

**Why:** User prefers to author their own commits but wants Claude to draft message candidates that accurately reflect the **current uncommitted diff**, not an abstract "what we've been doing." Past sessions hit cases where the suggested message covered work that was already committed in earlier commits — confusing and inaccurate.

**How to apply:**
- **Before suggesting**, check the working tree: run `git status` + `git diff --stat` (or `git diff --cached --stat` if anything is staged) to confirm exactly which files / hunks are about to enter the commit. Do not rely on memory of "what we just did".
- Filter out files that are already tracked & unchanged from the description — only describe what the commit will actually contain.
- Trigger whenever you judge the working tree is at a natural commit boundary (initial setup, feature done, fix complete, etc.) OR whenever the user signals they're about to commit.
- Always offer **two** options so the user can choose / mix. Make them genuinely different (different scope, different verb, or different emphasis) — not two near-identical phrasings.
- Keep them short, imperative, English, Conventional-Commits prefixed with scope.
- Do not execute `git commit` — just print the two candidate messages.

**Scope conventions used in this project family** (humanoid-training, mujoco-experience, isaaclab-experience):
- `viewer` · `drq` · `humanoid` · `notebook` · `repo` · `init` · `docs` · `scripts` · `deps`
- Match scope to the touched directory / surface, not to the broader feature.
