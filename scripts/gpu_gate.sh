#!/usr/bin/env bash
# Shared GPU gate: the GPU counts as free when no other training job (OTHER_JOB, a pgrep -f
# pattern) is running and the processes other than Ollama use less than FREE_MIB MiB.
# wait_for_gpu blocks until it has been free at QUIET_CHECKS consecutive checks CHECK_SEC
# seconds apart, unloading the Ollama models first so that the other job gets the whole GPU.
# anchored at the start of the command line, so that a shell command that only mentions the
# script name is not taken for the job
OTHER_JOB=${OTHER_JOB:-'^[^ ]*python[0-9.]* run_trashcan\.py'}
FREE_MIB=${FREE_MIB:-2000}
CHECK_SEC=${CHECK_SEC:-300}

other_mib() {
  local own
  own=" $(pgrep -f 'ollama' | tr '\n' ' ') "
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits |
    awk -F', *' -v own="$own" 'index(own, " " $1 " ") == 0 {s += $2} END {print s + 0}'
}

gpu_free() {
  local used
  used=$(other_mib)
  echo "$(date '+%F %T') GPU memory used by other processes ${used} MiB, other job running: $(pgrep -f "$OTHER_JOB" >/dev/null && echo yes || echo no)"
  ! pgrep -f "$OTHER_JOB" >/dev/null && [ "$used" -lt "$FREE_MIB" ]
}

unload_ollama() {
  ollama ps 2>/dev/null | awk 'NR > 1 {print $1}' | while read -r m; do ollama stop "$m"; done
}

# Run a GPU command with its output in a log file. If the other job starts meanwhile, the
# command is stopped at once and started anew after the other job has finished. Once the other
# job has been seen, the GPU must be free at QUIET_CHECKS consecutive checks before a start.
guarded() {
  local log=$1 pid rc yielded need=1
  shift
  while true; do
    pgrep -f "$OTHER_JOB" >/dev/null && need=${QUIET_CHECKS:-3}
    wait_for_gpu "$need"
    unload_ollama
    "$@" > "$log" 2>&1 &
    pid=$!
    yielded=0
    while kill -0 "$pid" 2>/dev/null; do
      if pgrep -f "$OTHER_JOB" >/dev/null; then
        kill "$pid"
        yielded=1
        echo "$(date '+%F %T') other job started, stopped $log"
        break
      fi
      sleep "${WATCH_SEC:-10}"
    done
    wait "$pid"
    rc=$?
    [ "$yielded" = 0 ] && return "$rc"
    need=${QUIET_CHECKS:-3}
  done
}

wait_for_gpu() {
  local need=${1:-1} ok=0
  if pgrep -f "$OTHER_JOB" >/dev/null; then
    unload_ollama
  fi
  while [ "$ok" -lt "$need" ]; do
    if gpu_free; then ok=$((ok + 1)); else ok=0; unload_ollama; fi
    [ "$ok" -lt "$need" ] && sleep "$CHECK_SEC"
  done
}
