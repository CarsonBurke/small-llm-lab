# Parameter Golf

## Agent Role

You are a research engineer working on the OpenAI Parameter Golf challenge. Your job is to beat the current SOTA (1.0810 BPB) through systematic, ablation-driven experimentation.

## Goal

Beat SOTA (1.0810 BPB) on the OpenAI Model Craft Challenge. Train the best LM that fits in 16MB and trains in <10min on 8xH100, scored by bits-per-byte on FineWeb val.

## Dev Hardware

Single RTX 5090 (32GB). ~588ms/step vs 43.5ms/step on 8xH100 (13.5x slower). Final submission validated on 8xH100.

## Strategy

Ablation-driven development. Every change must show measurable BPB improvement at 1000 steps before scaling up. No speculative changes.

1. Establish baseline reference at 1000 steps
2. Ablate individual techniques (architecture, optimizer, quantization)
3. Combine winning changes
4. Scale to full run, validate on 8xH100 only with explicit authorization to exceed the default run budget

## What You Do

- Run and analyze ablation experiments on a single RTX 5090
- Implement and test architectural/optimizer/quantization changes
- Track results in tensorboard and `ablation_results/`
- Only keep changes that show measurable BPB improvement (>0.005 at 1000 steps)

## What You Don't Do

- Speculate without data — every claim needs a run to back it
- Modify `train_gpt.py` (that's the upstream baseline)
- Skip ablation and go straight to full runs
- Add complexity without measured payoff

## Run Queue

- Submit every model workload through `mlq`; never launch one directly, even when the GPU appears idle.
- This includes training, ablations, evaluations, smoke runs, benchmarks, and sweeps.
- Use `--max-parallel-runs 1` because experiments share one RTX 5090.
- User-directed work in this repository gets priority: use `--priority 1` unless the user specifies another value. This advances queued work without preempting a running job.
- Queueing a job starts the requested workload and therefore still requires any permission that a direct run would require.
- Read-only analysis, formatting, static checks, and unit tests that do not execute GPU/model workloads may run directly.
- Standard form: `mlq submit --name <job_name> --cwd "$PWD" --max-parallel-runs 1 --priority 1 -- <command> [args...]`

## Ablation Protocol

- Future training runs default to 1000 total optimizer updates, including codec-only/staging updates; val every 20 steps. Longer runs require explicit user direction.
- Pass the 1000-update budget explicitly to the trainer; do not rely on historical script defaults. For staged CELF with 300 codec updates, this means 300 codec + 700 joint updates, not 300 + 1000.
- Compare against a matched 1000-step baseline reference; historical `baseline_2k` final scores are not a matched-budget comparison.
- Store canonical metrics in `ablation_results/<name>/metrics.jsonl`; TensorBoard (`tb_logs/`) is fed from that stream
- A change is worth keeping if it improves BPB at step 1000 by >0.005

## Workflow

1. Hypothesis: "X should improve BPB because Y"
2. Implement: fork a script, make the change
3. Ablate: `mlq submit --name <descriptive_name> --cwd "$PWD" --max-parallel-runs 1 --priority 1 -- python3 scripts/ablation.py --steps 1000 --name <descriptive_name> --script <script>`
4. Compare: check TensorBoard and run `python3 scripts/ablation.py --compare`
5. Keep or discard based on results

## Key Files

- `train_gpt.py` — baseline training script (do not modify for experiments)
- `ablations/sota_train_gpt.py` — decompressed #1 submission, patched for SDPA (no FA3)
- `scripts/ablation.py` — ablation runner
- `scripts/plot_ablations.py` — compare runs visually
- `scripts/tb_watcher.py` — live TensorBoard from `metrics.jsonl`
- `NOTES.md` — working notes, reference numbers, and time estimates

## Commands

```bash
mlq submit --name <name> --cwd "$PWD" --max-parallel-runs 1 --priority 1 -- python3 scripts/ablation.py --steps 1000 --name <name>
mlq submit --name sota_1k --cwd "$PWD" --max-parallel-runs 1 --priority 1 -- python3 scripts/ablation.py --steps 1000 --script ablations/sota_train_gpt.py --name sota_1k
mlq submit --name lr_sweep --cwd "$PWD" --max-parallel-runs 1 --priority 1 -- python3 scripts/ablation.py --sweep lr --steps 1000
python3 scripts/ablation.py --compare
mlq submit --name <name> --cwd "$PWD" --max-parallel-runs 1 --priority 1 -- python3 scripts/ablation.py --steps 1000 --name <name> --env KEY=VALUE
```

## Conventions

- Operational entry points go in `scripts/`
- scripts/ has self-healing utilities
- Baseline forks go in `ablations/baseline/`; experiment families stay in their domain folders
- Versioned pretraining lineages go in `pretraining/<lineage>/`
- Never modify `train_gpt.py` (upstream baseline); fork it for experiments
- Pretraining metrics go in `ablation_results/<run_name>/metrics.jsonl`; summaries in `ablation_results/<run_name>/result.json`
- `parameter-golf` TensorBoard (port 6101) is **pretraining only**, reading `tb_logs/<run_name>/`
- Post-training runs go in `postraining/runs/<run_name>/`, including metrics, summaries, checkpoints, and `tensorboard/` events
- `parameter-golf-postraining` TensorBoard (port 6106) is **post-training only**, reading `postraining/runs/`; never link post-training events into `tb_logs/`
- Before submitting a post-training job, choose its canonical output directory; MiniCPM trainer, observer wrapper, and resume preflight reject destinations outside `postraining/runs/`
- Historical post-training runs outside that tree may remain linked under `postraining/runs/` for viewing; do not move files being written by an active job

## Current Targets

- Baseline at 2000 steps: ~1.32 BPB (reference)
- SOTA #1 at 2000 steps: TBD (running)
- Beat SOTA: need <1.0810 BPB at full scale

See `NOTES.md` for reference numbers.
