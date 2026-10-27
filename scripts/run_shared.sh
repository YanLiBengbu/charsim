#!/usr/bin/env bash
# Steps of the GPU machine (the arguments, default 3 and 4; the watchdog names the steps that
# are left) while the reinforcement-learning trainings of the machine share the GPU. Those
# trainings are started by a queue (run_queue.py --max-total N) that starts one only while
# fewer than N processes with "nezhasim_space.train" and "--out" in their command line run.
# SLOTS placeholder processes with these arguments hold that many of the N places, so the queue
# runs at most N - SLOTS trainings (never stopping a running one), and a simulation starts
# while the processes other than Ollama use less than FREE_MIB MiB. CHARSIM_GPU_MIB caps the
# memory of the simulation itself. The placeholders end with this script. Since 2026-10-26 the
# user also starts off-policy trainings (nezhasim_space.offpolicy, about 1.9 GB each) outside
# the queue: FREE_MIB 6500 and the cap of 9000 MiB let a simulation run beside three of them on
# the 16 GB GPU.
set -u
cd "$(dirname "$0")/.."
SLOTS=${SLOTS:-8}
STEPS=${*:-3 4}
export FREE_MIB=${FREE_MIB:-6500} CHARSIM_GPU_MIB=${CHARSIM_GPU_MIB:-9000}
for _ in $(seq "$SLOTS"); do
  bash -c 'while kill -0 "$1" 2>/dev/null; do sleep 30; done' charsim_gpu_slot $$ nezhasim_space.train --out none &
done
echo "$(date '+%F %T') holding $SLOTS queue places, FREE_MIB=$FREE_MIB CHARSIM_GPU_MIB=$CHARSIM_GPU_MIB, steps $STEPS"
for s in $STEPS; do
  bash "scripts/run_step$s.sh" >> "step$s.log" 2>&1
done
echo "$(date '+%F %T') done"
