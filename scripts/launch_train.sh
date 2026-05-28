#!/usr/bin/env bash
# Unified train launcher — MANDATORY watcher.
# Refuses to start training without an eval watcher (see feedback-train-with-watcher).
#
# Usage:
#   bash scripts/launch_train.sh \
#       --algo {drq|tdmpc2|dreamerv3} \
#       --task <task_name> \
#       --run-dir <runs/...> \
#       --total-steps <int> \
#       --success-bar <float> \
#       [--seed 0] \
#       [--vram-mode {auto|parallel|deferred}] \
#       [--eval-n 5] \
#       [--no-watcher  # ONLY for debug; you will see a big WARNING]
#
# Spawns:
#   tmux session "<algo>_<task>_s<seed>"     — train process
#   tmux session "<algo>_<task>_s<seed>_w"   — watcher process
#
# Why: see [[feedback-train-with-watcher]]. DreamerV3 1M run without watcher
#      wasted 6h+ wall time on a DEAD policy (2026-05-27).

set -euo pipefail

ALGO=""; TASK=""; RUN_DIR=""; TOTAL_STEPS=""; SUCCESS_BAR=""
SEED=0; VRAM_MODE="auto"; EVAL_N=5; NO_WATCHER=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algo) ALGO="$2"; shift 2;;
        --task) TASK="$2"; shift 2;;
        --run-dir) RUN_DIR="$2"; shift 2;;
        --total-steps) TOTAL_STEPS="$2"; shift 2;;
        --success-bar) SUCCESS_BAR="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --vram-mode) VRAM_MODE="$2"; shift 2;;
        --eval-n) EVAL_N="$2"; shift 2;;
        --no-watcher) NO_WATCHER=1; shift;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

for v in ALGO TASK RUN_DIR TOTAL_STEPS SUCCESS_BAR; do
    if [[ -z "${!v}" ]]; then
        lc=${v,,}; flag=${lc//_/-}
        echo "missing --${flag}" >&2
        exit 2
    fi
done

if [[ "$NO_WATCHER" == "1" ]]; then
    echo "⚠️  ⚠️  ⚠️  WATCHER DISABLED — debug only. See feedback-train-with-watcher."
    echo "⚠️  Continuing in 5s. Ctrl-C to abort." >&2
    sleep 5
fi

mkdir -p "$RUN_DIR"
TAG="${ALGO}_$(echo "$TASK" | tr '/' '_')_s${SEED}"

# 1) Auto-detect VRAM mode if requested
if [[ "$VRAM_MODE" == "auto" ]]; then
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    # rough threshold: < 8 GB free → deferred
    if [[ "$FREE_MB" -lt 8000 ]]; then
        VRAM_MODE="deferred"
    else
        VRAM_MODE="parallel"
    fi
    echo "auto VRAM mode → $VRAM_MODE (free=${FREE_MB}MB)"
fi

# 2) Algo-specific train command
case "$ALGO" in
    drq)
        TRAIN_CMD="cd dependencies/dr-q && MUJOCO_GL=egl conda run -n humanoidbench --no-capture-output \
            python main.py task=$TASK seed=$SEED save_dir=$PWD/$RUN_DIR \
            num_train_frames=$TOTAL_STEPS";;
    tdmpc2)
        TRAIN_CMD="MUJOCO_GL=egl conda run -n humanoidbench --no-capture-output \
            python -m tdmpc2.train task=$TASK seed=$SEED \
            steps=$TOTAL_STEPS data_dir=/tmp/_buf_${TAG} \
            checkpoint=null disable_wandb=true save_agent=true save_csv=true";;
    dreamerv3)
        TRAIN_CMD="MUJOCO_GL=egl conda run -n humanoidbench-jax --no-capture-output \
            python -m embodied.agents.dreamerv3.train --configs humanoid_benchmark small \
            --method dreamer --logdir $RUN_DIR/logdir --task $TASK --seed $SEED \
            --run.steps $TOTAL_STEPS --jax.prealloc False --run.wandb False";;
    *) echo "unknown algo: $ALGO" >&2; exit 2;;
esac

# 3) Start train in tmux
tmux kill-session -t "$TAG" 2>/dev/null || true
tmux new-session -d -s "$TAG" -x 220 -y 50
tmux send-keys -t "$TAG" "$TRAIN_CMD 2>&1 | tee $RUN_DIR/train_s${SEED}.log" Enter
sleep 2
TRAIN_PID=$(pgrep -nf "$(echo "$TRAIN_CMD" | head -c 80)" || echo "")
echo "✅ train tmux: $TAG  (pid hint: $TRAIN_PID)"

# 4) Start watcher (unless --no-watcher)
if [[ "$NO_WATCHER" != "1" ]]; then
    case "$ALGO" in
        drq)
            WATCH_CMD="python scripts/train_watcher.py --run $RUN_DIR --train_pid $TRAIN_PID \
                --total_steps $TOTAL_STEPS --success_bar $SUCCESS_BAR";;
        tdmpc2)
            # tdmpc2 watcher uses older CLI (--models_dir / --eval_eps / --out)
            MODELS="$RUN_DIR/logs/$TASK/$SEED/$(basename $RUN_DIR)/models"
            WATCH_CMD="python scripts/ckpt_eval_loop_tdmpc2.py --task $TASK --seed $SEED \
                --models_dir $MODELS --train_pid $TRAIN_PID --eval_eps $EVAL_N \
                --out $RUN_DIR/ckpt_eval.csv";;
        dreamerv3)
            WATCH_CMD="python scripts/ckpt_eval_loop_dreamerv3.py --run-dir $RUN_DIR --task $TASK \
                --seed $SEED --eval-n $EVAL_N --success-bar $SUCCESS_BAR --vram-mode $VRAM_MODE";;
    esac
    tmux kill-session -t "${TAG}_w" 2>/dev/null || true
    tmux new-session -d -s "${TAG}_w" -x 200 -y 30
    tmux send-keys -t "${TAG}_w" "$WATCH_CMD 2>&1 | tee $RUN_DIR/watcher.log" Enter
    echo "✅ watcher tmux: ${TAG}_w  (vram-mode=$VRAM_MODE)"
fi

cat <<EOF

Train + watcher both up. Live view:
    tmux attach -t $TAG          # train
    tmux attach -t ${TAG}_w      # watcher
    bash scripts/train_status.sh $RUN_DIR

Detach with Ctrl-b d. Sessions survive SSH disconnect.
EOF
