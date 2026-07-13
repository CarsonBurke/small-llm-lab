# Parameter Golf

## Goal
Beat SOTA (1.0810 BPB) on the OpenAI Model Craft Challenge. Train the best LM that fits in 16MB and trains in <10min on 8xH100, scored by bits-per-byte on FineWeb val.

## Dev Hardware
Single RTX 5090 (32GB). ~588ms/step vs 43.5ms/step on 8xH100 (13.5x slower). Final submission validated on 8xH100.

## Strategy
Ablation-driven development. Every change must show measurable BPB improvement at 2000 steps before scaling up. No speculative changes.

1. Establish baseline reference at 2000 steps
2. Ablate individual techniques (architecture, optimizer, quantization)
3. Combine winning changes
4. Scale to full run, validate on 8xH100

## Ablation Protocol
- Default: 2000 steps, val every 20 steps
- Compare against baseline_2k reference
- Store canonical metrics in `ablation_results/<name>/metrics.jsonl`; TensorBoard (`tb_logs/`) is fed from that stream
- A change is worth keeping if it improves BPB at step 2000 by >0.005

## Key Files
- `train_gpt.py` — baseline training script (do not modify for experiments)
- `sota_train_gpt.py` — decompressed #1 submission, patched for SDPA (no FA3)
- `ablation.py` — run experiments: `python3 ablation.py --steps 2000 --name <name>`
- `plot_ablations.py` — compare runs visually
- `tb_watcher.py` — live tensorboard from `metrics.jsonl`
- `NOTES.md` — working notes, reference numbers, time estimates

## Commands
```bash
python3 ablation.py --steps 2000 --name <name>                    # single run
python3 ablation.py --script sota_train_gpt.py --name sota_2k     # run SOTA
python3 ablation.py --sweep lr --steps 2000                       # sweep
python3 ablation.py --compare                                     # table
python3 ablation.py --name <name> --env KEY=VALUE                 # override
```

## Conventions
- Experiment scripts go in repo root, named descriptively
- Never modify `train_gpt.py` (upstream baseline) — fork for experiments
- Metrics in `ablation_results/<run_name>/metrics.jsonl`
- Summary in `ablation_results/<run_name>/result.json`
- Tensorboard in `tb_logs/<run_name>/`
