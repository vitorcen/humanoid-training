# humanoid-training

## Init

Memory lives in-repo at `.claude/memory/`. On a fresh clone, symlink the global path once:

```bash
G="$HOME/.claude/projects/-home-david-work-humanoid-training"
mkdir -p "$G" && ln -sfn "$PWD/.claude/memory" "$G/memory"
```
