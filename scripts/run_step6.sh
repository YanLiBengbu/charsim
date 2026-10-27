#!/usr/bin/env bash
# Step 6 on the GPU machine, the second follow-up of 2026-10-26 (asked for by the authors after
# the first follow-up; not pre-registered): soul agents with all three tools of the characters
# (the plan T4, the state T5 and the scene check T6, charsim/scene.py), with T4 alone and with T6
# alone, with the base cards and tools on, seeds 100-109 for both bases, each base judged and
# summarized. The plan prompt and the scene check were revised after two smoke tests
# (results/sim_smoke_t45, results/sim_smoke_t456). Every run yields to the other job of the
# machine and resumes where it stopped.
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
  guarded "results/sim/$run/$scene.$(echo "$*" | tr ' :' '__').log" "$PY" scripts/simulate.py \
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

grid() {  # the conditions of the tools of one base
  local run=$1 scene
  for scene in pearls sell_the_land; do
    sim "$run" "$scene" --modes soul --tools on --t4 --t5 --t6
    sim "$run" "$scene" --modes soul --tools on --t4
    sim "$run" "$scene" --modes soul --tools on --t6
  done
}

summary() {
  echo "$(date '+%F %T') summary"
  nice -n 19 "$PY" scripts/analyze.py > results/summary_print.txt 2>&1 || echo "FAILED: summary"
}

SEEDS=100-109
grid qwen
judge_all
summary
grid llama
judge_all
summary
echo STEP6_DONE
