#!/usr/bin/env bash
# Sparse snapshot copier for DrQ-v2: cp on every 100k-frame boundary in eval.csv.
set -u
RUN="${1:-/root/autodl-tmp/drqv2_runs/humanoid_walk_s1}"
BUCKET_SIZE="${2:-100000}"
mkdir -p "$RUN"
cd "$RUN"
mkdir -p snapshots

# wait until train has emitted eval.csv + snapshot.pt at least once
while [ ! -f eval.csv ] || [ ! -f snapshot.pt ]; do
    echo "[$(date +%H:%M)] waiting for eval.csv + snapshot.pt to appear in $RUN ..."
    sleep 30
done

cur_frame=$(tail -1 eval.csv | cut -d, -f4 | cut -d. -f1)
cp snapshot.pt "snapshots/frame_${cur_frame}.pt"
last_bucket=$((cur_frame / BUCKET_SIZE))
echo "[$(date +%H:%M)] init cp frame_${cur_frame}.pt bucket=${last_bucket} (size=$(du -sh snapshots | cut -f1))"

while true; do
    if [ ! -f eval.csv ]; then sleep 60; continue; fi
    frame=$(tail -1 eval.csv | cut -d, -f4 | cut -d. -f1)
    bucket=$((frame / BUCKET_SIZE))
    if [ "$bucket" -gt "$last_bucket" ]; then
        cp snapshot.pt "snapshots/frame_${frame}.pt"
        echo "[$(date +%H:%M)] saved frame_${frame}.pt (bucket=${bucket}, total=$(ls snapshots | wc -l), size=$(du -sh snapshots | cut -f1))"
        last_bucket=$bucket
    fi
    sleep 60
done
