---
name: tmux-codex-invocation
description: "Recipe for invoking `codex exec -m gpt-5.5` inside a tmux session for long-running tool-using delegation"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0182c875-57e6-499e-a050-108e8fa8ab3e
---

When delegating to **codex gpt-5.5** in tmux on this machine, follow this exact recipe. Three traps caught us; each takes minutes to diagnose otherwise.

**Why:** Got DreamerV3 install delegated to codex only after burning ~30 min on PATH + sandbox + prompt-length failures (2026-05-27). Save the recipe so the next delegation is one-shot.

**How to apply:** Use whenever a long install / debug task is best handed to codex (heavy iteration, repeated pip churn, multi-step diagnosis). Pair with `[[tmux-tty-external-models]]` style usage from G1 brainstorm session.

## 1 · Absolute path to new codex (PATH trap)

tmux's PATH puts `~/.npm-global/bin/codex` (old 0.113) ahead of `~/.nvm/.../codex` (latest). `which codex` in tmux returns the old one and `gpt-5.5` rejects. **Always use the absolute path** in wrapper scripts:

```bash
CODEX=/home/david/.nvm/versions/node/v22.22.1/bin/codex
$CODEX --version  # confirm 0.134.0+
```

## 2 · Sandbox flag (model-routing trap)

ChatGPT-account codex with `--sandbox danger-full-access` routes `gpt-5.5` to a tier (`gpt-5.5-codex`) that returns:
> ERROR: The 'gpt-5.5' model requires a newer version of Codex.

The fix is the alias flag `--dangerously-bypass-approvals-and-sandbox` which writes/executes freely AND lets `gpt-5.5` through:

```bash
$CODEX exec -m gpt-5.5 \
  --dangerously-bypass-approvals-and-sandbox \
  --skip-git-repo-check \
  "<short prompt>"
```

(For read-only / non-shell prompts, `--sandbox read-only` also works with `gpt-5.5`.)

## 3 · Short prompt + brief file (length trap)

A long prompt passed as argv (`"$(cat brief.md)"`) ALSO triggers the same "newer Codex" rejection — likely OpenAI routes long prompts to a different model. **Pass a one-sentence prompt that points at a brief file** for the agent to read:

```bash
$CODEX exec -m gpt-5.5 --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check \
  "Install X in env Y. Read brief at /tmp/brief.md and follow ALL its constraints. When done write report to /tmp/report.md."
```

Brief content stays in the file. Codex reads it with `sed -n '1,220p' /tmp/brief.md` as one of its first steps.

## 4 · Wrapper + tmux launch

Write a wrapper to avoid quoting pain through `tmux send-keys`:

```bash
cat > /tmp/run_codex.sh <<WRAP
#!/usr/bin/env bash
exec /home/david/.nvm/versions/node/v22.22.1/bin/codex exec -m gpt-5.5 \\
  --dangerously-bypass-approvals-and-sandbox \\
  --skip-git-repo-check \\
  "<one-sentence prompt + brief pointer>"
WRAP
chmod +x /tmp/run_codex.sh

tmux kill-session -t <name> 2>/dev/null
tmux new-session -d -s <name> -x 220 -y 50
tmux send-keys -t <name> "/tmp/run_codex.sh 2>&1 | tee /tmp/codex_<name>.log" Enter
```

## 5 · Inspecting progress

`tmux capture-pane` without flags often shows nothing (small scrollback). Use:

```bash
tmux capture-pane -t <name>:0 -p -S -200 | tail -30
```

Or follow the tee'd file: `tail -f /tmp/codex_<name>.log`.

## 6 · Brief contents that work

A good brief includes:
- **Goal + success criterion** (e.g. "smoke test prints ≥5 [Agent Step N] lines")
- **What's been tried (failed)** — full version-pin history so codex doesn't repeat your mistakes
- **CRITICAL — DO NOT TOUCH** section listing PIDs / envs / dirs that must survive
- **Reusable smoke-test command** verbatim
- **Constraints** (which env to use, which `conda run` invocation pattern)
- **Deliverable** — where to write report + version pins

Related: [[train-with-watcher]] for the parallel-process pattern; [[no-secrets-in-memory]] — never put auth tokens in the brief.
