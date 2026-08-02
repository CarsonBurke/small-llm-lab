"""Teacher-forced SFT on modern-teacher reasoning traces, before RL.

Fine-tunes the pretrained nano backbone on ``sft_traces_v1.parquet``
documents so the base policy acquires a samplable multi-step reasoning
repertoire — the ingredient the RL-from-base run measurably lacked (mode
collapse: ~98% of prompts produced 8 identical samples, so GRPO/VAPO
advantages were ~zero). The gate for this stage is therefore NOT loss:
it is sampling diversity at temperature 1.0 — within-group reward spread
and the fraction of prompts with mixed correct/incorrect samples — which
is what gives the RL restart gradient signal.

Document framing matches the RL episode exactly. Each document is

    {problem}{INSTRUCTION_SUFFIX}\n{reasoning}\nAnswer: {final}

and is tokenized the way RL frames it: ``encode_prompt`` for the prompt
([BOS] + problem + suffix, encoded standalone) and a separately encoded
completion, because BPE is not concatenation-stable and the RL sampler
never sees a jointly encoded prompt+completion. Documents pack whole
(never split) into ``--seq-len`` rows separated by single 50256 tokens —
pretraining's document-boundary convention (BOS == EOS == <|endoftext|>,
no doubled separators) — and each row ends with one terminal 50256 so
every document trains its stop decision. Loss covers completion tokens
plus that stop target; prompt and instruction tokens are context only.

Optimizer mirrors pretraining's update geometry (Muon on block matrices;
AdamW groups for embed / readout / scalars / KDA convs / KDA decay
params) with every rate scaled by ``--lr-scale`` from the pretraining
values, linear warmup then linear decay to zero, weight decay 0 (decay
would only shrink pretrained weights over a short fine-tune). The Muon
rate carries POLAR_EXPRESS_STEP_COMPENSATION so the realized trunk step
keeps its pretraining ratio to the AdamW groups, and Muon momentum warms
0.85 -> 0.95 over 500 steps exactly as pretraining did.

Held-out evaluation splits by NORMALIZED problem identity (the same
160-char key that matched GSM8K provenance), so trace variants of one
problem can never straddle the split. Held-out completion CE streams to
``metrics.jsonl`` (tb_watcher schema; ``val_bpb`` is completion CE in
bits per TOKEN, not bytes — documents, not FineWeb bytes, are the unit
here). The final checkpoint is saved in the exact payload shape
``model_io.load_model`` consumes, so the RL trainer can initialize from
it directly, then the sampling gate runs through the production eval
stack (``evaluate_latent_math`` with ``pin_emit=True`` on a fresh
zero-init wrapper — the combiner path is bypassed, so the gate measures
the pure token policy at temperature 1.0 / top-p 1.0).

GPU workload — submit through mlq:

    mlq submit --name sft_traces_v1 --cwd "$PWD" --max-parallel-runs 1 -- \
        .venv/bin/python -m postraining.sft_trace_train --name sft_traces_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import train_gpt as baseline  # noqa: F401  (import order: patches must load first)
from fresh_lejepa_train import FreshHyperparameters
from postraining.core import (
    GPT2BPETokenizer,
    encode_prompt,
    load_posttraining_tokenizer,
    structural_format_ok,
)
from postraining.latent_eval import evaluate_latent_math
from postraining.latent_thought import LatentThoughtModel
from postraining.model_io import load_model
from postraining.muon import Muon
from postraining.prepare_sft_traces import (
    INSTRUCTION_SUFFIX,
    INSTRUCTION_SUFFIX_ANSWER,
    normalize_problem,
)
from postraining.math_prompt import ANSWER_FENCE_PROMPT_SCHEMA

SFT_CHECKPOINT_SCHEMA = "sft_trace_train/v1"

# Pretraining AdamW/Muon rates (nanogpt_mini_gpt2vocab_kda_3to1_pm_train.py);
# --lr-scale multiplies all of them uniformly. The Muon rate additionally
# carries POLAR_EXPRESS_STEP_COMPENSATION: postraining's Muon orthogonalizes
# with Polar Express and drops the rectangular scale, which shrinks the
# realized trunk step ~2-4x at equal nominal LR — without the compensation
# the AdamW groups would keep their pretraining ratio while the trunk
# silently under-moves (see the derivation in postraining/vapo/config.py).
PRETRAIN_LRS = {
    "embed": 0.7,
    "readout": 0.004,
    "scalar": 0.015,
    "delta_conv": 0.004,
    "delta_decay": 0.015,
    "muon": 0.025,
}

# Pretraining's Muon momentum warmup (0.85 -> 0.95 over 500 steps). A short
# fine-tune keeps the same trajectory rather than starting cold at full
# momentum with empty buffers.
MUON_MOMENTUM_START = 0.85
MUON_MOMENTUM = 0.95
MUON_MOMENTUM_WARMUP_STEPS = 500

IGNORE_INDEX = -100


def register_special_tokens(backbone, tokenizer: GPT2BPETokenizer) -> None:
    """Prepare the padded-vocab slack rows for the registered special tokens.

    The rows already exist (pretraining pads 50257 -> 50304), but their
    READOUT rows were anti-trained: never a CE target, they only ever
    appeared in the softmax denominator, so pretraining pushed their logits
    toward the softcap floor. Zeroing each registered readout row restarts
    it neutral. The embedding rows keep their pretraining init — inputs are
    RMS-normalized immediately, so any full-rank init is a fine starting
    type embedding for the SFT gradient to shape.
    """
    special_ids = [
        token_id
        for token_id in (
            tokenizer.think_open_id,
            tokenizer.think_close_id,
            tokenizer.answer_open_id,
            tokenizer.answer_close_id,
        )
        if token_id is not None
    ]
    if not special_ids:
        raise ValueError("tokenizer has no special tokens registered")
    vocab_rows = backbone.proj.weight.size(0)
    if max(special_ids) >= vocab_rows:
        raise ValueError(
            f"special token ids exceed the checkpoint vocab ({vocab_rows})"
        )
    with torch.no_grad():
        for token_id in special_ids:
            backbone.proj.weight[token_id].zero_()


def muon_momentum_at(step: int) -> float:
    """Pretraining's linear momentum warmup, clamped at the final value."""
    progress = min(1.0, step / MUON_MOMENTUM_WARMUP_STEPS)
    return MUON_MOMENTUM_START + (MUON_MOMENTUM - MUON_MOMENTUM_START) * progress


@dataclass(frozen=True)
class TokenizedDocument:
    """One trace, framed exactly as an RL episode."""

    prompt_length: int  # [BOS] + problem + instruction suffix
    ids: tuple[int, ...]  # prompt then completion, no terminal separator


def load_documents(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    kept = [row for row in rows if row["verified"]]
    if not kept:
        raise ValueError(f"{path} holds no verified documents")
    return kept


def split_holdout(
    documents: list[dict], holdout_problems: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic problem-level split: (train_docs, holdout_docs, panel).

    Problems are keyed by their normalized identity (the GSM8K-matching
    key), so trace variants of one problem never straddle the split. The
    holdout panel carries one row per problem for the sampling gate.
    """
    by_problem: dict[str, list[dict]] = {}
    for document in documents:
        by_problem.setdefault(
            normalize_problem(document["problem"]), []
        ).append(document)
    ranked = sorted(
        by_problem,
        key=lambda key: hashlib.sha256(key.encode("utf-8")).digest(),
    )
    if not 0 < holdout_problems < len(ranked):
        raise ValueError(
            f"--holdout-problems must be in (0, {len(ranked)}); "
            f"got {holdout_problems}"
        )
    held = set(ranked[:holdout_problems])
    train_docs, holdout_docs, panel = [], [], []
    for key in ranked:
        docs = by_problem[key]
        if key in held:
            holdout_docs.extend(docs)
            panel.append(docs[0])
        else:
            train_docs.extend(docs)
    return train_docs, holdout_docs, panel


def tokenize_documents(
    tokenizer,
    documents: list[dict],
    seq_len: int,
    min_completion_tokens: int = 0,
    instruction_suffix: str = INSTRUCTION_SUFFIX,
) -> list[TokenizedDocument]:
    """RL-exact framing: prompt encoded standalone, completion separately.

    ``min_completion_tokens`` drops ultra-terse traces (about a third of the
    opus46_10k source is under 30 completion tokens) — an ablation knob for
    the risk that near-instant derivations reinforce the answer-first
    brevity this stage exists to unlearn.

    ``instruction_suffix`` must match the corpus's composition suffix
    (the answer-fenced corpus uses ``INSTRUCTION_SUFFIX_ANSWER``); the
    startswith check below fails loudly on a mismatch.
    """
    tokenized = []
    for document in documents:
        prompt_text = document["problem"] + instruction_suffix
        body = document["document"]
        if not body.startswith(prompt_text):
            raise ValueError(
                "document does not start with its own prompt: "
                f"{body[:80]!r}"
            )
        prompt_ids = encode_prompt(tokenizer, prompt_text)
        completion_ids = tokenizer.encode(body[len(prompt_text):])
        if not completion_ids:
            raise ValueError(f"empty completion in document {body[:80]!r}")
        if len(completion_ids) < min_completion_tokens:
            continue
        ids = tuple(prompt_ids) + tuple(completion_ids)
        # +1 for the stop/separator target after the completion.
        if len(ids) + 1 > seq_len:
            continue
        tokenized.append(TokenizedDocument(len(prompt_ids), ids))
    if not tokenized:
        raise ValueError("no documents fit --seq-len")
    return tokenized


def answer_fence_document_fraction(
    documents: list[TokenizedDocument], tokenizer
) -> float:
    """Fraction of completions with the exact anchored gate shape.

    Measured on token ids with the RL gate itself (a terminal separator
    stands in for the stop token, exactly where ``pack_rows`` puts one).
    Red-team round 2: the RL trainer's provenance guard used to trust the
    ``--answer-fence`` FLAG, which an operator can set against a corpus
    with no answer fences at all — the guard must verify the capability,
    not the claim.
    """
    think = (tokenizer.think_open_id, tokenizer.think_close_id)
    answer = (tokenizer.answer_open_id, tokenizer.answer_close_id)
    compliant = sum(
        1
        for document in documents
        if structural_format_ok(
            list(document.ids[document.prompt_length:])
            + [tokenizer.eos_id()],
            think,
            answer,
        )
    )
    return compliant / max(len(documents), 1)


def think_span_token_percentiles(
    documents: list[TokenizedDocument], tokenizer
) -> dict[str, int] | None:
    """Inner think-span token-length distribution over completions.

    Recorded in checkpoint provenance so the RL trainer can refuse a
    ``--think-min-tokens`` floor the corpus never taught (red-team
    round 2, finding D: the fence fraction validates the SHAPE, not the
    span length — an anchored-but-terse corpus measures 1.0 yet earns
    all-zero reward at a floor above its spans). Documents without a
    single well-formed think span are skipped; ``None`` when none have
    one.
    """
    from postraining.core import single_fence_span

    fences = (tokenizer.think_open_id, tokenizer.think_close_id)
    lengths = sorted(
        span[1] - span[0] - 1
        for document in documents
        if (
            span := single_fence_span(
                document.ids[document.prompt_length:], fences
            )
        )
        is not None
    )
    if not lengths:
        return None
    return {
        "min": lengths[0],
        "p1": lengths[round(0.01 * (len(lengths) - 1))],
        "p50": lengths[round(0.50 * (len(lengths) - 1))],
    }


def pack_rows(
    documents: list[TokenizedDocument],
    seq_len: int,
    separator: int,
    rng: random.Random | None,
) -> list[tuple[list[int], list[int]]]:
    """Greedy whole-document packing into (input, target) rows.

    Documents follow each other directly — each document starts with BOS,
    which doubles as the previous document's stop target, exactly like the
    pretraining shards. The last document in a row gets one terminal
    separator so its stop decision is trained too; the remainder is padding
    with all targets ignored. Targets cover completion tokens plus the stop
    slot; prompt positions predict nothing.
    """
    order = list(range(len(documents)))
    if rng is not None:
        rng.shuffle(order)
    rows: list[tuple[list[int], list[int]]] = []
    tokens: list[int] = []
    spans: list[tuple[int, int, int]] = []  # (offset, prompt_length, length)

    def close_row() -> None:
        tokens.append(separator)
        targets = [IGNORE_INDEX] * seq_len
        for offset, prompt_length, length in spans:
            for position in range(offset + prompt_length - 1, offset + length):
                targets[position] = tokens[position + 1]
        tokens.extend([separator] * (seq_len - len(tokens)))
        rows.append((list(tokens), targets))
        tokens.clear()
        spans.clear()

    for index in order:
        document = documents[index]
        if len(tokens) + len(document.ids) + 1 > seq_len:
            close_row()
        spans.append(
            (len(tokens), document.prompt_length, len(document.ids))
        )
        tokens.extend(document.ids)
    if tokens:
        close_row()
    return rows


def build_optimizers(backbone, lr_scale: float) -> list[torch.optim.Optimizer]:
    """Pretraining's exact optimizer partition, scaled for fine-tuning.

    Muon takes block matrices (KDA depthwise conv windows excluded — their
    per-channel geometry is not a Muon update); AdamW covers embed, readout,
    scalars, conv windows, and the KDA decay parameters. Weight decay is 0
    everywhere: pretraining's decay regularizes a from-scratch run, but over
    a short fine-tune it would only shrink the pretrained weights.
    """
    from postraining.train_latent_vapo import (
        muon_matrix_parameters,
        non_trunk_parameter_ids,
    )

    delta_conv, delta_decay = [], []
    for block in backbone.blocks:
        if getattr(block, "use_kda", False):
            attention = block.attn
            delta_conv.extend(
                conv.weight
                for conv in (
                    attention.q_conv1d, attention.k_conv1d, attention.v_conv1d
                )
            )
            delta_decay.extend((attention.A_log, attention.dt_bias))
    delta_decay_ids = {id(parameter) for parameter in delta_decay}
    matrix_parameters = muon_matrix_parameters(
        backbone.blocks, non_trunk_parameter_ids(backbone)
    )
    matrix_ids = {id(parameter) for parameter in matrix_parameters}
    scalars = [
        parameter
        for parameter in backbone.parameters()
        if parameter.ndim < 2 and id(parameter) not in delta_decay_ids
    ]
    adamw = torch.optim.AdamW(
        [
            {"params": [backbone.embed.weight],
             "lr": PRETRAIN_LRS["embed"] * lr_scale},
            {"params": [backbone.proj.weight],
             "lr": PRETRAIN_LRS["readout"] * lr_scale},
            {"params": scalars, "lr": PRETRAIN_LRS["scalar"] * lr_scale},
            {"params": delta_conv,
             "lr": PRETRAIN_LRS["delta_conv"] * lr_scale},
            {"params": delta_decay,
             "lr": PRETRAIN_LRS["delta_decay"] * lr_scale},
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
        fused=all(
            parameter.is_cuda for parameter in backbone.parameters()
        ),
    )
    from postraining.vapo.config import POLAR_EXPRESS_STEP_COMPENSATION

    muon = Muon(
        matrix_parameters,
        lr=PRETRAIN_LRS["muon"] * lr_scale * POLAR_EXPRESS_STEP_COMPENSATION,
        weight_decay=0.0,
        mu=MUON_MOMENTUM_START,
    )
    optimizers = [adamw, muon]
    owned = [
        parameter
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(owned) != len({id(parameter) for parameter in owned}):
        raise AssertionError("a parameter is owned by two optimizer groups")
    if {id(parameter) for parameter in owned} != {
        id(parameter) for parameter in backbone.parameters()
    }:
        raise AssertionError(
            "optimizer groups must exactly partition the backbone parameters"
        )
    if matrix_ids & {id(parameter) for parameter in delta_conv}:
        raise AssertionError("KDA conv windows leaked into the Muon group")
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers


def lr_scale_at(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup to 1.0, then linear decay to 0 at ``total_steps``."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    remaining = total_steps - warmup_steps
    if remaining <= 0:
        return 1.0
    return max(0.0, (total_steps - step) / remaining)


def masked_ce_sum(backbone, inputs: torch.Tensor, targets: torch.Tensor):
    """(summed CE over supervised positions, supervised position count).

    The vocab readout renders ONLY the supervised positions: full-sequence
    logits at 50k vocab retain multiple [B*S, V] fp32 tensors through the
    softcap's autograd graph — the exact linear-in-slots term the replay
    path bounds with its slot budget after a measured OOM on this card
    (NOTES.md). Gathering features before ``logits_from_features`` is
    exact (the readout is positionwise) and cuts that memory to the ~66%
    of packed slots that carry loss.
    """
    supervised = targets != IGNORE_INDEX
    with torch.autocast("cuda", dtype=torch.bfloat16):
        token_latent = backbone.embed_tokens(inputs)
        belief = backbone.temporal_belief_from_token_latent(token_latent)
        features = torch.cat((token_latent, belief), dim=-1)[supervised]
        logits = backbone.logits_from_features(features)
    loss = F.cross_entropy(
        logits.float(), targets[supervised], reduction="sum"
    )
    return loss, int(supervised.sum())


@torch.no_grad()
def holdout_ce(
    backbone,
    rows: list[tuple[list[int], list[int]]],
    rows_per_batch: int,
    device: torch.device,
) -> float:
    """Mean completion CE (nats/token) over the held-out packed rows."""
    was_training = backbone.training
    backbone.eval()
    total, count = 0.0, 0
    for start in range(0, len(rows), rows_per_batch):
        batch = rows[start:start + rows_per_batch]
        inputs = torch.tensor(
            [tokens for tokens, _ in batch], dtype=torch.long, device=device
        )
        targets = torch.tensor(
            [targets for _, targets in batch], dtype=torch.long, device=device
        )
        loss, supervised = masked_ce_sum(backbone, inputs, targets)
        total += float(loss)
        count += supervised
    if was_training:
        backbone.train()
    return total / count


def gate_metrics_from_counts(
    counts: list[int], samples: int
) -> dict[str, float]:
    """The RL-restart signal: is there within-group variance to learn from?"""
    if not counts:
        raise ValueError("empty prompt panel")
    mixed = sum(1 for count in counts if 0 < count < samples)
    rates = [count / samples for count in counts]
    return {
        "accuracy": sum(counts) / (len(counts) * samples),
        "mixed_prompt_fraction": mixed / len(counts),
        "within_group_reward_std_mean": sum(
            math.sqrt(rate * (1.0 - rate)) for rate in rates
        ) / len(counts),
        "prompts_all_correct": sum(1 for c in counts if c == samples),
        "prompts_all_wrong": sum(1 for c in counts if c == 0),
    }


def run_sampling_gate(
    backbone,
    tokenizer,
    panel: list[dict],
    args,
    device: torch.device,
    instruction_suffix: str = INSTRUCTION_SUFFIX,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Temperature-1.0 group sampling on held-out problems, RL-identically.

    A fresh zero-init wrapper with ``pin_emit=True`` routes generation
    through the production rollout/eval stack while bypassing the combiner
    entirely — the measured policy is exactly the SFT'd token policy.
    """
    wrapper = LatentThoughtModel(backbone).to(device)
    wrapper.eval()
    rows = [
        {
            "prompt": [
                {"content": document["problem"] + instruction_suffix}
            ],
            "reward_model": {
                "ground_truth": document["final_answer"],
                "style": (
                    "rule"
                    if document.get("source") == "a1_deepmind"
                    else "rule-lighteval/MATH_v2"
                ),
            },
        }
        for document in panel[: args.gate_prompts]
    ]
    captured: list[dict[str, object]] = []
    think_fence_ids = (
        (tokenizer.think_open_id, tokenizer.think_close_id)
        if getattr(tokenizer, "think_open_id", None) is not None
        else None
    )
    answer_fence_ids = (
        (tokenizer.answer_open_id, tokenizer.answer_close_id)
        if getattr(tokenizer, "answer_open_id", None) is not None
        else None
    )
    metrics = evaluate_latent_math(
        wrapper,
        tokenizer,
        rows,
        args.gate_samples,
        args.gate_max_new_tokens,
        args.gate_max_new_tokens,
        args.gate_samples,
        args.seed,
        device,
        args.gate_prompt_tokens,
        captured_attempts=captured,
        capture_problem_count=min(16, len(rows)),
        capture_samples_per_problem=min(4, args.gate_samples),
        temperature=1.0,
        top_p=1.0,
        pin_emit=True,
        think_fence_ids=think_fence_ids,
        answer_fence_ids=answer_fence_ids,
        min_think_tokens=getattr(args, "gate_think_min_tokens", 1),
    )
    count_key = (
        "contract_prompt_correct_counts"
        if think_fence_ids is not None and answer_fence_ids is not None
        else "prompt_correct_counts"
    )
    gate = gate_metrics_from_counts(
        list(metrics[count_key]), args.gate_samples
    )
    gate.update(
        {
            "samples_per_prompt": args.gate_samples,
            "prompts": len(rows),
            "temperature": 1.0,
            "top_p": 1.0,
            "think_min_tokens": getattr(args, "gate_think_min_tokens", 1),
            "raw_accuracy": metrics["accuracy"],
            "contract_accuracy": metrics["contract_accuracy"],
            "structural_format_fraction": metrics[
                "structural_format_fraction"
            ],
            "ended_fraction": metrics["ended_fraction"],
            "emitted_tokens_mean": metrics["emitted_tokens_mean"],
            "emitted_tokens_p95": metrics["emitted_tokens_p95"],
        }
    )
    return gate, captured


def save_backbone_checkpoint(backbone, path: Path, sft_metadata: dict) -> None:
    """The exact payload shape ``model_io.load_model`` consumes, atomically."""
    payload = {
        "model": {
            key: value.detach().cpu()
            for key, value in backbone.state_dict().items()
        },
        "model_config": backbone.model_config,
        "architecture": backbone.architecture,
        "train_seq_len": backbone.train_context_tokens,
        "sft": sft_metadata,
    }
    staging = path.with_name(path.name + ".tmp")
    torch.save(payload, staging)
    os.replace(staging, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def create_fresh_run_dir(path: Path) -> None:
    """Create one immutable run root; this trainer has no resume semantics."""
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing SFT run directory {path}"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--checkpoint", default="logs/k3_quality_20k_ctx8k_final_model.pt"
    )
    parser.add_argument(
        "--traces", default="postraining/data/sft_traces_v1.parquet"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=4096)
    # 2x4096 rows keep grad-enabled vocab slots (~5.4k supervised) inside
    # the replay path's measured-safe 8192 slot budget; the step-wide loss
    # normalization makes the 2x4 accumulation bit-comparable to 4x2.
    parser.add_argument("--rows-per-micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr-scale", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--min-completion-tokens", type=int, default=0)
    # Think-fenced corpus (sft_traces_v2_think.parquet): registers the
    # <think>/</think> special tokens in the padded-vocab slack and trains
    # the fence as part of every completion.
    parser.add_argument("--think-tokens", action="store_true")
    # Answer-fenced corpus: additionally registers <answer>/</answer> so
    # completions carry both fences; requires --think-tokens. Named
    # --answer-fence to match the RL trainer, where --answer-tokens is
    # already the numeric answer-budget argument.
    parser.add_argument("--answer-fence", action="store_true")
    parser.add_argument("--holdout-problems", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--gate-prompts", type=int, default=128)
    parser.add_argument("--gate-samples", type=int, default=8)
    # Reference completions on the gate panel run to 674 tokens (p95 467);
    # a 512 budget would truncate ~5% of solvable groups and register
    # length-driven "mixed" groups — contaminating the exact diversity
    # metric the gate reads. 768 clears the panel's maximum.
    parser.add_argument("--gate-max-new-tokens", type=int, default=768)
    parser.add_argument("--gate-prompt-tokens", type=int, default=512)
    parser.add_argument("--gate-think-min-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    for field in (
        "epochs", "seq_len", "rows_per_micro_batch", "grad_accum",
        "holdout_problems", "eval_every", "gate_prompts", "gate_samples",
        "gate_max_new_tokens", "gate_prompt_tokens",
        "gate_think_min_tokens",
    ):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if not 0.0 < args.lr_scale <= 1.0:
        parser.error("--lr-scale must be in (0, 1]")
    if args.min_completion_tokens < 0:
        parser.error("--min-completion-tokens must be nonnegative")

    run_dir = Path("postraining/runs") / args.name
    try:
        create_fresh_run_dir(run_dir)
    except FileExistsError as error:
        parser.error(str(error))
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("")

    backbone = load_model(args.checkpoint, device)
    if args.seq_len > backbone.train_context_tokens:
        parser.error(
            f"--seq-len {args.seq_len} exceeds the checkpoint's pretraining "
            f"context ({backbone.train_context_tokens})"
        )
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    if args.answer_fence and not args.think_tokens:
        parser.error("--answer-fence requires --think-tokens")
    if args.think_tokens:
        if "gpt2vocab" not in backbone.architecture:
            parser.error("--think-tokens requires a gpt2vocab checkpoint")
        tokenizer = GPT2BPETokenizer(
            think_tokens=True, answer_tokens=args.answer_fence
        )
        register_special_tokens(backbone, tokenizer)
    else:
        tokenizer = load_posttraining_tokenizer(
            backbone.architecture, FreshHyperparameters.tokenizer_path
        )

    instruction_suffix = (
        INSTRUCTION_SUFFIX_ANSWER if args.answer_fence else INSTRUCTION_SUFFIX
    )
    documents = load_documents(Path(args.traces))
    train_docs, holdout_docs, panel = split_holdout(
        documents, args.holdout_problems
    )
    train_tokens = tokenize_documents(
        tokenizer, train_docs, args.seq_len, args.min_completion_tokens,
        instruction_suffix=instruction_suffix,
    )
    # The holdout keeps every trace regardless of the filter: CE must be
    # measured on the corpus distribution, not the filtered one.
    holdout_tokens = tokenize_documents(
        tokenizer, holdout_docs, args.seq_len,
        instruction_suffix=instruction_suffix,
    )
    fence_fraction: float | None = None
    if args.answer_fence:
        fence_fraction = answer_fence_document_fraction(
            train_tokens + holdout_tokens, tokenizer
        )
        # Verify the capability, not the flag: an SFT run whose corpus
        # lacks the anchored fence shape would still stamp
        # answer_fence=True into provenance, and the RL guard would then
        # admit a checkpoint that earns all-zero structural reward.
        if fence_fraction < 0.99:
            parser.error(
                f"--answer-fence: only {fence_fraction:.1%} of documents "
                "carry the anchored <think>...</think><answer>...</answer> "
                "completion shape; this corpus cannot teach the "
                "structural gate (want an --answer-tags parquet from "
                "prepare_sft_traces)"
            )
    think_span_percentiles: dict[str, int] | None = None
    if args.think_tokens:
        think_span_percentiles = think_span_token_percentiles(
            train_tokens + holdout_tokens, tokenizer
        )
        if think_span_percentiles is not None:
            print(
                "think-span tokens: "
                f"min {think_span_percentiles['min']}, "
                f"p1 {think_span_percentiles['p1']}, "
                f"p50 {think_span_percentiles['p50']}"
            )
    holdout_rows = pack_rows(
        holdout_tokens, args.seq_len, tokenizer.eos_id(), rng=None
    )
    print(
        f"documents: {len(train_tokens)} train / {len(holdout_tokens)} "
        f"held-out ({len(panel)} held-out problems); "
        f"train tokens {sum(len(d.ids) for d in train_tokens)/1e6:.2f}M"
    )

    # Pack every epoch up front (distinct shuffles) so the schedule length
    # is exact before the first step.
    rows_per_step = args.rows_per_micro_batch * args.grad_accum
    epoch_rows = [
        pack_rows(
            train_tokens,
            args.seq_len,
            tokenizer.eos_id(),
            random.Random(args.seed + epoch),
        )
        for epoch in range(args.epochs)
    ]
    steps_per_epoch = [
        math.ceil(len(rows) / rows_per_step) for rows in epoch_rows
    ]
    total_steps = sum(steps_per_epoch)
    print(
        f"schedule: {total_steps} steps "
        f"({rows_per_step} rows x {args.seq_len} tokens per step)"
    )

    optimizers = build_optimizers(backbone, args.lr_scale)
    backbone.train()
    started = time.perf_counter()
    step = 0

    def log(entry: dict) -> None:
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")

    last_val_ce: float | None = None

    def log_val(at_step: int) -> float:
        nonlocal last_val_ce
        ce = holdout_ce(
            backbone, holdout_rows, args.rows_per_micro_batch, device
        )
        last_val_ce = ce
        log(
            {
                "type": "val",
                "step": at_step,
                "val_loss": ce,
                # Bits per supervised TOKEN (completion CE), not bytes;
                # the key name keeps tb_watcher streaming.
                "val_bpb": ce / math.log(2),
                "train_time_ms": (time.perf_counter() - started) * 1000,
            }
        )
        print(f"step {at_step}: holdout completion ce {ce:.4f}", flush=True)
        return ce

    log_val(0)
    for rows in epoch_rows:
        for start in range(0, len(rows), rows_per_step):
            step_rows = rows[start:start + rows_per_step]
            supervised_total = sum(
                sum(1 for target in targets if target != IGNORE_INDEX)
                for _, targets in step_rows
            )
            scale = lr_scale_at(step, total_steps, args.warmup_steps)
            momentum = muon_momentum_at(step)
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * scale
                    if isinstance(optimizer, Muon):
                        group["mu"] = momentum
            step_loss = 0.0
            for micro_start in range(
                0, len(step_rows), args.rows_per_micro_batch
            ):
                micro = step_rows[
                    micro_start:micro_start + args.rows_per_micro_batch
                ]
                inputs = torch.tensor(
                    [tokens for tokens, _ in micro],
                    dtype=torch.long, device=device,
                )
                targets = torch.tensor(
                    [targets for _, targets in micro],
                    dtype=torch.long, device=device,
                )
                loss, _ = masked_ce_sum(backbone, inputs, targets)
                (loss / supervised_total).backward()
                step_loss += float(loss) / supervised_total
            for optimizer in optimizers:
                optimizer.step()
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            step += 1
            log(
                {
                    "type": "train",
                    "step": step,
                    "train_loss": step_loss,
                    "lr_scale": scale,
                    "train_time_ms": (time.perf_counter() - started) * 1000,
                }
            )
            if step % args.eval_every == 0:
                log_val(step)

    if step % args.eval_every:
        log_val(step)
    final_ce = last_val_ce
    checkpoint_path = run_dir / "sft_final_model.pt"
    save_backbone_checkpoint(
        backbone,
        checkpoint_path,
        {
            "schema": SFT_CHECKPOINT_SCHEMA,
            "base_checkpoint": args.checkpoint,
            "traces": args.traces,
            "traces_sha256": file_sha256(Path(args.traces)),
            "traces_manifest": (
                str(Path(args.traces).with_suffix(".manifest.json"))
                if Path(args.traces).with_suffix(".manifest.json").is_file()
                else None
            ),
            "args": vars(args),
            "steps": step,
            # Measured on token ids, not asserted by flag; the RL
            # trainer's --answer-fence guard keys off this value.
            "answer_fence_document_fraction": fence_fraction,
            "answer_fence_prompt_schema": (
                ANSWER_FENCE_PROMPT_SCHEMA if args.answer_fence else None
            ),
            # The RL trainer checks --think-min-tokens against these
            # (a floor above the corpus median is refused at startup).
            "think_span_token_percentiles": think_span_percentiles,
        },
    )
    print(f"saved {checkpoint_path}")

    gate, captured = run_sampling_gate(
        backbone, tokenizer, panel, args, device,
        instruction_suffix=instruction_suffix,
    )
    print(
        f"sampling gate: accuracy {gate['accuracy']:.4f}, "
        f"mixed prompts {gate['mixed_prompt_fraction']:.3f}, "
        f"within-group reward std {gate['within_group_reward_std_mean']:.4f}",
        flush=True,
    )
    (run_dir / "gate_transcripts.json").write_text(
        json.dumps(captured, indent=2)
    )
    result = {
        "schema": SFT_CHECKPOINT_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "steps": step,
        "holdout_completion_ce": final_ce,
        "sampling_gate": gate,
        "answer_fence_prompt_schema": (
            ANSWER_FENCE_PROMPT_SCHEMA if args.answer_fence else None
        ),
        "args": vars(args),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    log({"type": "diag", "step": step, **{
        f"gate_{key}": value
        for key, value in gate.items()
        if isinstance(value, (int, float))
    }})


if __name__ == "__main__":
    main()
