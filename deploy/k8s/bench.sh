#!/usr/bin/env bash
# Scaling benchmark: renders the same book at several pod counts, from a
# clean parts directory each time, and writes deploy/k8s/bench.csv.
# render_seconds is the render Job's wall time (scheduling + model load +
# synthesis), not just synthesis -- what a user would actually wait.
#
# Usage: BOOK=bench.txt deploy/k8s/bench.sh [pod counts...]   (default: 1 2 4 8)
# Assumes run.sh has been run once (cluster up, image loaded, cast written).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
if (($#)); then counts=("$@"); else counts=(1 2 4 8); fi

ensure_cluster
export CHAPTERS="$(chapter_count)"
csv="$K8S_DIR/bench.csv"
echo "pods,chapters,render_seconds,audio_seconds,speedup,realtime_factor" >"$csv"

baseline=""
for pods in "${counts[@]}"; do
  export PARALLELISM=$pods
  rm -rf "$DATA_DIR/parts/$STEM"
  log "benchmark: $pods pod(s), $CHAPTERS chapters"
  run_job render-job.yaml alebrijes-render
  seconds=$JOB_SECONDS
  run_job assemble-job.yaml alebrijes-assemble
  audio=$(audio_seconds)
  baseline=${baseline:-$seconds}
  python3 - "$pods" "$CHAPTERS" "$seconds" "$audio" "$baseline" >>"$csv" <<'EOF'
import sys
pods, chapters, seconds, audio, baseline = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
print(f"{pods},{chapters},{seconds},{audio:.0f},{baseline / seconds:.2f},{audio / seconds:.2f}")
EOF
  tail -1 "$csv"
done

log "results ($csv):"
column -s, -t "$csv"
