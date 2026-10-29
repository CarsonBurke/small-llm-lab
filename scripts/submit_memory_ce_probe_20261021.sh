#!/usr/bin/env bash
# Teacher-forced memory probe (postraining/memory_ce_probe.py): does dense
# next-token CE on gold drill traces adopt the gated-delta memory carry that
# RL left at a 2-9% read share? NOTES 2026-10-21, "Teacher-forced memory
# probe". $1 selects the job:
#   calib  40 steps of the memory arm and the costliest arm, for measured
#          step and eval times
#   calib_trunk  60 trunk_only steps at TRUNK_LR_SCALE: the SFT lr_scale 0.1
#          raised CE at this probe's ~7k supervised tokens per step (calib_r3)
#   a, b   the real arms, split so each job fits the 75m limit
# STEPS is the step count; a and b must use the same value (analyze refuses
# trained arms with different step counts). Set it from the calibration.
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
RUNS="$REPO/postraining/runs"
DATA="$REPO/postraining/data/memory_ce_probe_drills_20261021_v1.parquet"
PANEL="$REPO/postraining/data/memory_ce_probe_hazard_panel_20261021_v2.parquet"
JOB=${1:?usage: $0 calib|a|b}
case "$JOB" in
  calib) ARMS=(mem mem_plus_trunk); STEPS=${STEPS:-40}; LIMIT=30m ;;
  calib_trunk) ARMS=(trunk_only); STEPS=${STEPS:-60}; LIMIT=20m ;;
  a) ARMS=(off mem mem_lr_quarter mem_lr_x4); STEPS=${STEPS:?set STEPS from the calibration}; LIMIT=75m ;;
  b) ARMS=(mem_shuffled mem_midlayer trunk_only mem_plus_trunk); STEPS=${STEPS:?set STEPS from the calibration}; LIMIT=75m ;;
  *) echo "unknown job $JOB" >&2; exit 2 ;;
esac
# ATTEMPT (r1, r2, ...) names a resubmission; earlier run directories stay.
ATTEMPT=${ATTEMPT:-r1}
NAME=memory_ce_probe_20261021_$JOB
[ "$ATTEMPT" = r1 ] || NAME="${NAME}_$ATTEMPT"
exec mlq submit --name "$NAME" --cwd "$REPO" --max-parallel-runs 1 --priority 11 \
  --time-limit "$LIMIT" ${AFTER:+--after-success "$AFTER"} --idempotency-key "$NAME" -- \
  "$REPO/.venv/bin/python" -u -m postraining.memory_ce_probe train \
  --train-data "$DATA" --hazard-panel "$PANEL" \
  --arms "${ARMS[@]}" --steps "$STEPS" \
  --batch-trajectories 32 --macro-batches 4 --log-every 25 --train-slot-budget 12288 \
  --fixed-point-passes 2 --fresh-learning-rate 3e-3 --write-layer 4 --seed 1337 \
  ${TRUNK_LR_SCALE:+--trunk-lr-scale "$TRUNK_LR_SCALE"} \
  --output "$RUNS/$NAME"
