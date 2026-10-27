#!/usr/bin/env bash
# Step 7 on the GPU machine, the third follow-up of 2026-10-26 (asked for by the authors: more
# tools until the characters reach what they want; not pre-registered): soul agents with the
# tools of step 6 and the consistency check T7, the choice T8 and the turn-taking T9
# (charsim/scene.py), all with the T5 texts of config/state_texts2.json, the base cards and tools
# on, seeds 100-109. Every tool for both bases first (judged and summarized), then without T9
# and without T8. Every run yields to the other job of the machine and resumes where it stopped.
set -u
cd "$(dirname "$0")/.."
source scripts/gpu_gate.sh
PY=${TRAIN_PY:-/media/w/DATA1/envs/charsim/bin/python}
MODELS=${MODELS_DIR:-/media/w/DATA1/models}
declare -A BASE=([llama]=$MODELS/Meta-Llama-3.1-8B-Instruct-bnb-4bit [qwen]=$MODELS/Qwen3-8B-bnb-4bit)
QUIET_CHECKS=${QUIET_CHECKS:-3}
export CHARSIM_NO_CACHE=${CHARSIM_NO_CACHE:-1}
JUDGES=${JUDGES:-"phi4:14b gemma3:12b mistral-nemo:12b"}

sim() {
  local run=$1 scene=$2
  shift 2
  mkdir -p "results/sim/$run"
  echo "$(date '+%F %T') simulate $run $scene --seeds $SEEDS $*"
  guarded "results/sim/$run/$scene.$(echo "$*" | tr ' :/' '___').log" "$PY" scripts/simulate.py \
    --base "${BASE[${run%%_*}]}" --run "$run" --scene "$scene" --seeds "$SEEDS" "$@" || echo "FAILED: $run $scene $*"
}

judge_all() {
  mkdir -p results/judge
  echo "$(date '+%F %T') judge"
  # shellcheck disable=SC2086
  guarded results/judge/judge.log python3 scripts/judge.py --judges $JUDGES --inputs results/sim/*/*/* ||
    echo "FAILED: judges"
  unload_ollama
}

T="--modes soul --tools on --state-texts config/state_texts2.json --t4 --t5 --t6 --t7"
grid() {  # one set of tools (the arguments) for both bases and scenes
  local run scene
  for run in qwen llama; do
    for scene in pearls sell_the_land; do
      # shellcheck disable=SC2086
      sim "$run" "$scene" $T "$@"
    done
  done
}

summary() {
  echo "$(date '+%F %T') summary"
  nice -n 19 "$PY" scripts/analyze.py > results/summary_print.txt 2>&1 || echo "FAILED: summary"
}

SEEDS=100-109
grid --t8 --t9
judge_all
summary
grid --t8
grid --t9
judge_all
summary
echo STEP7_DONE
