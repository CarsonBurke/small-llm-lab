# small-llm-lab

Research on small language models: pretraining, recurrent architectures,
supervised fine-tuning, and reinforcement learning with verifiable rewards.
The focus is learning reasoning that transfers to held-out problems under
realistic memory, compute, and context budgets.

The active line combines a small hybrid Kimi Delta Attention (KDA) backbone
with supervised reasoning traces and verifier-guided post-training. Other
experiment families explore recurrent memory, latent reasoning, byte-native
diffusion, tokenization, and efficient GPU execution. These are research
implementations; an experiment's presence is not evidence that it improves
the model.

This project grew out of [OpenAI Parameter Golf](https://github.com/openai/parameter-golf).
Its scope now extends beyond the original 16 MB artifact and ten-minute
training challenge. Upstream baselines and submission records remain as
historical references.

## Research areas

- **Pretraining:** nanoGPT-derived models, hybrid attention/recurrent
  backbones, corpus mixtures, and context curricula.
- **Post-training:** reasoning-trace SFT, VAPO-based reinforcement learning,
  math and code verifiers, and token/latent carry experiments.
- **Data and evaluation:** problem registries, overlap audits, corpus
  provenance, held-out capability gates, and rollout inspection.
- **Systems:** Triton kernels, cached recurrent decoding, compiled rollout
  execution, profiling, and checkpoint/resume validation.

The current KDA8 line and the separate MiniCPM experiments use different
checkpoints and training contracts. Consult their run documentation before
reusing a checkpoint or launching a recipe. Training reward alone is not a
measure of generalization; compare held-out accuracy, termination behavior,
and retained base-model capability under matched evaluation settings.

## Repository map

| Path | Contents |
| --- | --- |
| [`pretraining/`](pretraining/README.md) | Model lineages, corpus configuration, and pretraining recipes |
| [`postraining/`](postraining/README.md) | SFT, RL, verifiers, rollout engines, and evaluation; directory spelling retained for existing paths |
| [`scripts/`](scripts/README.md) | Dataset builders, launchers, diagnostics, benchmarks, and reports |
| [`tokenization/`](tokenization/README.md) | Tokenizer experiments |
| [`energy_readout/`](energy_readout/), [`xlayer/`](xlayer/) | Experimental readout and cross-layer components |
| [`shared/`](shared/) | Shared utilities |
| [`tests/`](tests/), [`pretraining/tests/`](pretraining/tests/), [`postraining/tests/`](postraining/tests/) | Test suites |
| [`docs/`](docs/) | Research assessments and design documentation |
| [`NOTES.md`](NOTES.md) | Experiment history, measurements, and operational notes |
| [`ablations/`](ablations/) | Historical ablations and Parameter Golf experiments |

`train_gpt.py` and `train_gpt_mlx.py` are preserved upstream entry points.
New experiments belong in their domain directories.

## Environment

Development targets Linux with an NVIDIA GPU; the local research workstation
uses a single RTX 5090 with 32 GB of VRAM. GPU kernels and training recipes
have hardware-specific requirements. The dependency list is a research
environment, not a portable inference package.

Clone the repository and create an environment:

```bash
git clone https://github.com/CarsonBurke/small-llm-lab.git
cd small-llm-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use a CUDA-compatible PyTorch installation for the selected GPU. Individual
experiments may require additional dependencies or kernel setup; consult the
relevant domain documentation. Datasets, prepared corpora, and trained
checkpoints are separate artifacts: cloning the source does not reproduce an
existing local run.

## Running experiments

Read [`AGENTS.md`](AGENTS.md) for local execution constraints and the
[script index](scripts/README.md) for entry points. Launch commands from the
repository root so relative data and checkpoint paths resolve consistently.

All local model workloads—including training, evaluation, and benchmarks—go
through the separately installed `mlq` queue. Use one concurrent run on the
shared GPU and priority 1 for user-directed work. For example, a historical
baseline comparison with an explicit budget is:

```bash
mlq submit --name baseline_1k --cwd "$PWD" \
  --max-parallel-runs 1 --priority 1 -- \
  "$PWD/.venv/bin/python" scripts/ablation.py \
    --steps 1000 --val-every 20 --name baseline_1k
```

New training runs default to 1,000 total optimizer updates, including staged
updates; longer runs require explicit authorization. Historical recipes may
show larger budgets and should not be launched unchanged by default.
For the active work, start with the [pretraining guide](pretraining/README.md)
or [post-training guide](postraining/README.md). Select the corpus, checkpoint,
evaluation protocol, and output directory before submitting a job.

## Results and reproducibility

| Run type | Canonical outputs | TensorBoard events |
| --- | --- | --- |
| Pretraining | `ablation_results/<run_name>/metrics.jsonl` and `result.json` | `tb_logs/<run_name>/` |
| Post-training | `postraining/runs/<run_name>/`, including metrics, summaries, and checkpoints | `postraining/runs/<run_name>/tensorboard/` |

The local dashboards retain their historical service names:
`parameter-golf` on port 6101 for pretraining and
`parameter-golf-postraining` on port 6106 for post-training. Keep their event
trees separate. Checkpoints and manifests bind runs to their data, model,
and prompt contracts; review those contracts before resuming or comparing
runs. See the domain guides for exact requirements and artifact locations.

## Origins and licensing

The repository began with OpenAI's Parameter Golf code and includes work
adapted from modded-nanogpt. The original copyright notices are preserved in
[`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
Dataset and external model licenses are separate from this repository's MIT
license; consult the corresponding source manifests and notices.
