#!/usr/bin/env bash
# Step 2 on the GPU machine: for every test scene the story-world adapter and the soul adapters
# of both base models, and of one base model again on the pseudonym data (memorization control).
# An adapter that already exists is skipped. Every run yields to the other job of the machine
# (guarded in gpu_gate.sh): it waits until that job has finished and the GPU has been free for
# QUIET_CHECKS checks, and a run interrupted by the other job is started anew.
set -u
cd "$(dirname "$0")/.."
source scripts/gpu_gate.sh
PY=${TRAIN_PY:-/media/w/DATA1/envs/charsim/bin/python}
MODELS=${MODELS_DIR:-/media/w/DATA1/models}
declare -A BASE=([llama]=$MODELS/Meta-Llama-3.1-8B-Instruct-bnb-4bit [qwen]=$MODELS/Qwen3-8B-bnb-4bit)
RUNS=${RUNS:-"llama: qwen: qwen:_pseudo"}
# batch 2 keeps the peak below about 10 GB, so that the other jobs of the machine still fit
TRAIN_ARGS=${TRAIN_ARGS:-"--batch 2 --accum 8"}
QUIET_CHECKS=${QUIET_CHECKS:-3}

train() {
  local out=$1
  shift
  # the training log is written after the adapter, so it marks a complete adapter
  if [ -f "$out/train_log.json" ]; then
    echo "exists: $out"
    return
  fi
  mkdir -p "$(dirname "$out")"
  echo "$(date '+%F %T') train $out"
  # shellcheck disable=SC2086
  guarded "$out.log" "$PY" scripts/train_lora.py --out "$out" $TRAIN_ARGS "$@" || echo "FAILED: $out (see $out.log)"
}

evaluate() {
  local out=results/fidelity/$1/$2
  if [ -f "$out/summary.json" ]; then
    echo "exists: $out"
    return
  fi
  mkdir -p "$out"
  echo "$(date '+%F %T') evaluate $out"
  guarded "$out/log.txt" "$PY" scripts/eval_fidelity.py --base "$3" --adapters "adapters/$1/$2" --eval "$4" --out "$out" ||
    echo "FAILED: $out (see $out/log.txt)"
}

for run in $RUNS; do
  b=${run%%:*}
  sfx=${run#*:}
  for d in data/train$sfx/c*; do
    A=adapters/$b$sfx/${d##*/}
    train "$A/world" --base "${BASE[$b]}" --world "$d/world.txt"
    for f in "$d"/soul_*.jsonl; do
      train "$A/$(basename "$f" .jsonl)" --base "${BASE[$b]}" --soul "$f" --world-adapter "$A/world"
    done
    evaluate "$b$sfx" "${d##*/}" "${BASE[$b]}" "data/eval$sfx/${d##*/}"
  done
done
echo STEP2_DONE
