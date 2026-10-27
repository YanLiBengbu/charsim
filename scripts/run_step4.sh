#!/usr/bin/env bash
# Step 4 on the GPU machine, after the main experiment: the judges of the main experiment and of
# the calibration controls, the persona vectors, the personality ablation (every pre-registered
# trait edit up and down, as a card edit and as steering with coefficient +1 or -1, seeds
# 100-109, soul agents with tools), the component ablation (T1 only, T2 only, T3 shown to the
# characters, seeds 100-109), and the judges of all ablation sessions. Qwen3 8B runs before
# Llama 3.1 8B. Every run yields to the other job of the machine and resumes where it stopped.
set -u
cd "$(dirname "$0")/.."
source scripts/gpu_gate.sh
PY=${TRAIN_PY:-/media/w/DATA1/envs/charsim/bin/python}
MODELS=${MODELS_DIR:-/media/w/DATA1/models}
declare -A BASE=([llama]=$MODELS/Meta-Llama-3.1-8B-Instruct-bnb-4bit [qwen]=$MODELS/Qwen3-8B-bnb-4bit)
SEEDS=${SEEDS:-100-109}
QUIET_CHECKS=${QUIET_CHECKS:-3}
export CHARSIM_NO_CACHE=${CHARSIM_NO_CACHE:-1}
JUDGES=${JUDGES:-"phi4:14b gemma3:12b mistral-nemo:12b"}

sim() {
  local run=$1 scene=$2
  shift 2
  mkdir -p "results/sim/$run"
  echo "$(date '+%F %T') simulate $run $scene $*"
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

edits() {  # character:trait of every pre-registered trait edit of a scene
  python3 -c "import json, sys; [print(e['character'] + ':' + e['trait']) for s in json.load(open('config/scenes.json'))['scenes'] if s['id'] == sys.argv[1] for e in s['trait_edits']]" "$1"
}

judge_all
for run in qwen llama; do
  for scene in pearls sell_the_land; do
    if [ ! -d "results/steer/$run/$scene" ]; then
      guarded "results/steer_$run.$scene.log" "$PY" scripts/persona_vectors.py --base "${BASE[$run]}" --run "$run" \
        --scene "$scene" || echo "FAILED: persona vectors $run $scene"
    fi
    mapfile -t E < <(edits "$scene")
    for e in "${E[@]}"; do
      for d in up down; do
        sim "$run" "$scene" --modes soul --tools on --edit "$e:$d"
      done
      for a in 1 -1; do
        sim "$run" "$scene" --modes soul --tools on --steer "$e:$a"
      done
    done
    sim "$run" "$scene" --modes soul --tools t1 t2
    sim "$run" "$scene" --modes soul --tools on --t3-to-agents
  done
done
judge_all
echo STEP4_DONE
