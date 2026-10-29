#!/usr/bin/env bash
# Full-capture benchmark traces (all 512 panel problems x 4 samples, the run's
# exact bench contract and eval seed) from the SFT start and the stage-1 MAC
# snapshots, to see what decimal arithmetic became as the run collapsed. Each
# command is the run's recorded argv minus its stop rule and output; a
# snapshot's actor is branched in with a fresh critic the benchmark never
# reads, the start evaluates the SFT actor as the run's step-0 bench did.
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
RUN=mac_online_allstates_drill_20261019
OUT="$REPO/postraining/runs/${RUN}_traces"
for step in ${STEPS:-0 60 120 140}; do
  if [ "$step" -eq 0 ]; then policy=start; else policy=$(printf 'step_%06d' "$step"); fi
  mapfile -d '' args < <("$REPO/.venv/bin/python" - "$REPO/postraining/runs/$RUN" "$policy" <<'EOF'
import json, sys
from pathlib import Path
run, policy = Path(sys.argv[1]), sys.argv[2]
argv = json.loads((run / "manifest.json").read_text())["source_provenance"]["argv"][1:]
drop = {"--bench-stop-floor": 1, "--bench-stop-evals": 1, "--output": 1}
if policy != "start":
    drop["--study-critic-init"] = 1
kept, index = [], 0
while index < len(argv):
    if argv[index] in drop:
        index += 1 + drop[argv[index]]
        continue
    kept.append(argv[index])
    index += 1
if policy != "start":
    # Only the policy is evaluated: branch its actor, leave the critic fresh.
    kept += ["--actor-init", str(run / "snapshots" / f"{policy}.pt"), "--fresh-critic"]
sys.stdout.write("\0".join(kept) + "\0")
EOF
  )
  mlq submit --name "${RUN}_trace_${policy}" --cwd "$REPO" --max-parallel-runs 1 --priority 10 \
    --idempotency-key "${RUN}_trace_${policy}_v3" -- \
    "$REPO/.venv/bin/python" -u -m postraining.train_latent_vapo "${args[@]}" \
    --bench-only --bench-capture-problems 512 --bench-capture-samples 4 \
    --output "$OUT/$policy"
done
