#!/usr/bin/env bash
# Step 3 on the GPU machine: the simulations of the main experiment with the seeds of the
# experiments (100-119; seeds below 100 were used while the engine was developed):
# {prompted, soul} x {tools off, tools on} and the story-world adapter alone, for both base
# models and both scenes, and the pseudonym control of Qwen3 8B. The scene of chapter 19 runs
# first; the supporting-cast souls that chapter 34 needs are trained before its scene. Every
# run yields to the other job of the machine (guarded in gpu_gate.sh) and resumes where it
# stopped.
set -u
cd "$(dirname "$0")/.."
source scripts/gpu_gate.sh
PY=${TRAIN_PY:-/media/w/DATA1/envs/charsim/bin/python}
MODELS=${MODELS_DIR:-/media/w/DATA1/models}
declare -A BASE=([llama]=$MODELS/Meta-Llama-3.1-8B-Instruct-bnb-4bit [qwen]=$MODELS/Qwen3-8B-bnb-4bit)
SEEDS=${SEEDS:-100-119}
QUIET_CHECKS=${QUIET_CHECKS:-3}
# the intention scores without the cached prompt, as in the fidelity evaluation (the cached
# scores differed by up to 0.023 in bf16 on the test of 2026-10-22)
export CHARSIM_NO_CACHE=${CHARSIM_NO_CACHE:-1}

sim() {
  local run=$1 scene=$2
  shift 2
  mkdir -p "results/sim/$run"
  echo "$(date '+%F %T') simulate $run $scene $*"
  guarded "results/sim/$run/$scene.$(echo "$*" | tr ' :' '__').log" "$PY" scripts/simulate.py \
    --base "${BASE[${run%%_*}]}" --run "$run" --scene "$scene" --seeds "$SEEDS" "$@" || echo "FAILED: $run $scene $*"
}

scene_grid() {
  local scene=$1
  for run in llama qwen; do
    sim "$run" "$scene" --modes soul prompted --tools on off
    sim "$run" "$scene" --modes world --tools on
  done
  sim qwen_pseudo "$scene" --modes soul prompted --tools on
}

scene_grid pearls
RUNS="llama: qwen: qwen:_pseudo" bash scripts/run_step2.sh  # only the missing supporting-cast souls are trained
scene_grid sell_the_land
echo STEP3_MAIN_DONE
