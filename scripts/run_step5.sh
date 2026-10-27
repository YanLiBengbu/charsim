#!/usr/bin/env bash
# Step 5 on the GPU machine, the follow-up of 2026-10-26 after the main results (not
# pre-registered): soul agents with the rich cards, soul agents with a director, CoSER-style
# acting (prompted characters with the rich cards who know the plot of the original scene),
# prompted characters with the rich cards, and soul agents with the rich cards and a director;
# tools on. A follow-up session takes about twice as long as a main one (longer prompts, one
# more generation per turn with a baseline), so both bases run seeds 100-109 first, as the
# ablations, and are judged and summarized; Qwen3 8B then runs seeds 110-119, so that its
# follow-up conditions have 20 seeds as the main grid. Every run yields to the other job of the
# machine and resumes where it stopped. The director prompt was revised in smoke tests
# (results/sim_smoke*) before the grid; every scene runs for 20 moves (two ways to let the base
# model end a scene were tried and dropped, see charsim/scene.py).
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

grid() {  # the follow-up conditions of one base
  local run=$1 scene
  for scene in pearls sell_the_land; do
    sim "$run" "$scene" --modes soul --tools on --rich
    sim "$run" "$scene" --modes soul --tools on --baseline director
    sim "$run" "$scene" --modes prompted --tools on --rich --baseline coser
    sim "$run" "$scene" --modes prompted --tools on --rich
    sim "$run" "$scene" --modes soul --tools on --rich --baseline director
  done
}

summary() {
  echo "$(date '+%F %T') summary"
  nice -n 19 "$PY" scripts/analyze.py > results/summary_print.txt 2>&1 || echo "FAILED: summary"
}

SEEDS=100-109
grid qwen
judge_all
grid llama
judge_all
summary
SEEDS=110-119
grid qwen
judge_all
summary
echo STEP5_DONE
