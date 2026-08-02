"""On-policy self-distillation from a pretrained or trace-SFT checkpoint.

Implements Zhao et al., "Self-Distilled Reasoner: On-Policy
Self-Distillation for Large Language Models" (arXiv:2601.18734v3): the
student samples one response from question-only context, a frozen step-0 copy
of the same model scores that response with a verified solution in context,
and only the student's logits receive the full-vocabulary, pointwise-clipped
forward-KL gradient.

The current VAPO and SFT trainers are independent and unchanged. OPSD exports
``opsd_final_model.pt`` in the same bare-backbone payload shape consumed by
``postraining.model_io.load_model`` and retains nested SFT provenance for
downstream fence reconstruction.

GPU workload -- submit through mlq:

    mlq submit --name opsd_v1 --cwd "$PWD" --max-parallel-runs 1 -- \
        .venv/bin/python -m postraining.train_opsd \
        --name opsd_v1 \
        --checkpoint \
          postraining/runs/sft_v4_answer_canonical_hfonly_e3/sft_final_model.pt \
        --dataset postraining/data/opsd_dapo17k_train.parquet \
        --reference-column solution \
        --data-manifest postraining/data/opsd_dapo17k.manifest.json

Static compatibility validation is CPU-only and may run directly:

    CUDA_VISIBLE_DEVICES='' .venv/bin/python -m postraining.train_opsd \
        --name opsd_v1 --validate-only
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from postraining.opsd.config import build_arg_parser, validate_args
from postraining.opsd.trainer import (
    OPSDTrainer,
    infer_source_contract,
    validate_source_contract,
    write_manifest,
)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        source_payload = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        if not isinstance(source_payload, dict):
            raise ValueError("source checkpoint must be a metadata payload")
        infer_source_contract(args, source_payload)
        validate_args(args)
        contract = validate_source_contract(args, source_payload)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.validate_only:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return
    output = Path("postraining/runs") / args.name
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error(
            f"output run {output} already exists and is nonempty; use a new "
            "--name or pass --resume"
        )
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output, args, contract)
    trainer = OPSDTrainer(args, source_payload)
    del source_payload
    trainer.train()


if __name__ == "__main__":
    main()
