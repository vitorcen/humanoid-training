---
name: feedback-commit-style
description: "User commits manually — when a commit point is reached, offer two short English Conventional Commit suggestions (do not commit for them)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7d4c3d5c-6c21-470f-acd6-6e6460eaf62e
---

When a commit point is reached, do NOT run `git commit` yourself. Instead, present **two** English short-sentence options in Conventional Commits format: `feat(scope): xxx xxx ... xxx xxx` (or `fix`, `chore`, `docs`, `refactor`, etc. as appropriate). The user will pick one and commit themselves.

**Why:** User prefers to author their own commits but wants Claude to draft message candidates in a consistent style.

**How to apply:**
- Trigger whenever you judge the working tree is at a natural commit boundary (initial setup, feature done, fix complete, etc.).
- Always offer **two** options so the user can choose / mix.
- Keep them short, imperative, English, Conventional-Commits prefixed with scope.
- Do not execute `git commit` — just print the two candidate messages.
