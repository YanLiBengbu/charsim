#!/usr/bin/env bash
# Step 1 on the GPU machine: LLM annotation of the novel, merging into the narrative tools,
# and the memorization probe of the base models. Resumable: finished windows are skipped.
# Before every LLM step the GPU gate yields to the other training job of the machine.
set -u
cd "$(dirname "$0")/.."
source scripts/gpu_gate.sh
PY=${PY:-python3}
URL=${OLLAMA_BASE_URL:-http://127.0.0.1:11434}
W=${WORKERS:-2}
EXTRACTORS=${EXTRACTORS:-"qwen2.5:14b gemma3:12b"}
ADJ=${ADJUDICATOR:-phi4:14b}
PROBE=${PROBE_MODELS:-"llama3.1:8b qwen3:8b"}

for m in $EXTRACTORS; do
  for t in attribution beats events relations; do
    wait_for_gpu 1
    $PY scripts/extract_llm.py --task "$t" --model "$m" --base-url "$URL" --workers "$W"
  done
done
wait_for_gpu 1
$PY scripts/extract_llm.py --task attribution --model "$ADJ" --base-url "$URL" --workers "$W"
set -- $EXTRACTORS
$PY scripts/merge_annotations.py --models "$1" "$2" --adjudicator "$ADJ"
wait_for_gpu 1
$PY scripts/probe_memory.py --models $PROBE --base-url "$URL"
unload_ollama
echo STEP1_DONE
