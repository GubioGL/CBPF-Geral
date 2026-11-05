#!/usr/bin/env bash
# Classical PINN at N_col = 20..90, with the same setup as the HQPINN runs (run_qubit_sweep.py --fixed-validation).
# It completes the blue curves of Fig. 3. Runs already done are skipped, so it can be started again at any time.
#
#   bash run/run_classical_ncol.sh                  # n = 10, 15, 7, 5, 3 with 12 processes
#   WIDTHS="10" JOBS=10 bash run/run_classical_ncol.sh
#
# The sweep has no internal parallelism: each process runs one (n, seed) on one CPU thread. On the i7-13620H the
# throughput grows up to about 12 processes, and more threads per process do not help. Keep the laptop plugged in:
# below MIN_BATTERY percent on battery no new job starts. Sleep and the lid switch are inhibited while it runs.
set -u
cd "$(dirname "$0")/.."

if [[ -z "${RUN_INHIBITED:-}" ]] && command -v systemd-inhibit > /dev/null \
   && systemd-inhibit --what=idle:sleep:handle-lid-switch --who=hqpinn --why=sweep true 2> /dev/null; then
  RUN_INHIBITED=1 exec systemd-inhibit --what=idle:sleep:handle-lid-switch --who=hqpinn --why="classical sweep" bash "$0" "$@"
fi

export PY=${PYTHON:-.venv/bin/python}
export OUT=data/Forward/classical_same_protocol_ncol
export MIN_BATTERY=${MIN_BATTERY:-25}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
JOBS=${JOBS:-12}
WIDTHS=${WIDTHS:-10 15 7 5 3}
mkdir -p "$OUT/logs"

low_battery() {
  local bat=$(ls -d /sys/class/power_supply/BAT* 2> /dev/null | head -1)
  [[ -n "$bat" && "$(cat "$bat/status")" == "Discharging" && "$(cat "$bat/capacity")" -lt "$MIN_BATTERY" ]]
}
run_job() {
  if low_battery; then echo "SKIPPED n=$1 seed=$2: battery below $MIN_BATTERY%"; return; fi
  nice -n 10 "$PY" run/run_qubit_sweep.py --fixed-validation --models classical_mlp --qubits "$1" \
    --n-collocation 20 30 40 50 60 70 80 90 --tfin 0.1 1.0 --seeds "$2" --output-dir "$OUT" \
    > "$OUT/logs/n$1_s$2.log" 2>&1 || echo "FAILED n=$1 seed=$2 (see $OUT/logs/n$1_s$2.log)"
}
export -f low_battery run_job

for n in $WIDTHS; do for s in $(seq 0 49); do echo "$n $s"; done; done | xargs -P "$JOBS" -L 1 bash -c 'run_job "$0" "$1"'
"$PY" run/run_qubit_sweep.py --rebuild-only --fixed-validation --output-dir "$OUT" 2>&1 | tail -1
echo "runs done: $(find "$OUT/raw_runs" -name 'seed_*.json' | wc -l) (16 per n and seed; 4000 for all widths)"
