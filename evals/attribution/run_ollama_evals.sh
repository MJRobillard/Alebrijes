#!/usr/bin/env bash
# Runs ollama_eval.py model by model (fastest first) across novels, so each
# model's results land before the next starts. Safe to re-run after an
# interruption: every quotation is checkpointed, and finished model/novel
# pairs make no requests at all.
#
# Usage: evals/attribution/run_ollama_evals.sh PATH_TO_PDNC [novels...]
set -euo pipefail
pdnc=${1:?usage: run_ollama_evals.sh PATH_TO_PDNC [novels...]}
shift
novels=("${@:-PrideAndPrejudice TheSportOfTheGods}")
read -r -a novels <<<"${novels[*]}"
models=(
  qwen2.5:1.5b-instruct
  llama3.1:8b
  richardyoung/llama-3.1-8b-instruct-abliterated:latest
  hermes3:8b
)
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for model in "${models[@]}"; do
  for novel in "${novels[@]}"; do
    echo "=== $model / $novel $(date +%T)"
    python "$here/ollama_eval.py" --pdnc "$pdnc" --novel "$novel" --models "$model"
  done
done
