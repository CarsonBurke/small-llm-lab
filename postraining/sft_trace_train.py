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
import contextlib
import functools
import hashlib
import json
import itertools
import math
import numpy as np
import os
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import train_gpt as baseline  # noqa: F401  (import order: patches must load first)
from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters
from pretraining.latent_moe_training import (
    apply_accumulated_quantile_balance,
    enable_quantile_balance_collection,
    reset_quantile_balance_accumulators,
)
from postraining.core import (
    encode_many,
    encode_prompt,
    frame_prompt,
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
from postraining.math_prompt import (
    ANSWER_FENCE_PROMPT_SCHEMA,
    require_answer_fence_prompt_schema,
)

SFT_CHECKPOINT_SCHEMA = "sft_trace_train/v1"

# Pretraining AdamW/Muon rates (pretraining/nanogpt_mini/nanogpt_mini_gpt2vocab_kda_3to1_pm_train.py);
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


def register_special_tokens(backbone, tokenizer) -> None:
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
    # int32 array, NOT a tuple of Python ints: token ids exceed the small-int
    # cache, so every element would be a ~28-byte object behind an 8-byte
    # pointer. A 900k-document corpus is ~355M tokens, i.e. ~13 GB as Python
    # objects against ~1.4 GB here.
    ids: np.ndarray  # prompt then completion, no terminal separator


def load_documents(path: Path, allow_unverified: bool = False) -> list[dict]:
    """Rows from a corpus parquet, honouring the ``verified`` column.

    The curated trace corpora carry per-row verification, and admitting an
    unverified row from them silently would defeat the point of the column.
    Large published instruction corpora carry model-generated solutions that
    were never re-verified here, so admitting them is a deliberate choice the
    caller states rather than something the truthiness of the column's
    contents decides -- a corpus that stored the string "False" would
    otherwise pass this filter.
    """
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    for row in rows:
        if "gradeable" in row and not isinstance(row["gradeable"], bool):
            raise ValueError(
                f"{path} stores 'gradeable' as "
                f"{type(row['gradeable']).__name__}, not bool"
            )
        if not isinstance(row["verified"], bool):
            raise ValueError(
                f"{path} stores 'verified' as "
                f"{type(row['verified']).__name__}, not bool; a non-empty "
                "string would pass a truthiness filter regardless of value"
            )
    if allow_unverified:
        return rows
    kept = [row for row in rows if row["verified"]]
    if not kept:
        raise ValueError(
            f"{path} holds no verified documents; pass --allow-unverified to "
            "train on a corpus whose solutions were not re-verified here"
        )
    if len(kept) < len(rows):
        # A mixed-domain build marks only its sandbox-verified code rows
        # verified; filtering would silently train on that slice alone.
        raise ValueError(
            f"{path} mixes {len(kept)} verified and {len(rows) - len(kept)} "
            "unverified documents; pass --allow-unverified to train on all of "
            "them, or build a corpus from the verified sources alone"
        )
    return kept


def split_holdout(
    documents: list[dict], holdout_problems: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic problem-level split: (train_docs, holdout_docs, panel).

    Problems are keyed by their normalized identity (the GSM8K-matching
    key), so trace variants of one problem never straddle the split. The
    holdout panel carries one row per problem for the sampling gate.

    Only rows the math verifier can grade enter that panel. A corpus may mix
    domains -- a code row's final answer is a program graded by executing
    tests, not by comparing an answer string -- and the gate scores every
    panel row with the MATH verifier. Admitting a code row would therefore
    count it wrong by construction, deflating gate accuracy and the
    mixed-group rate the RL stage is sized from. Such rows still train and
    still count toward holdout CE; they just cannot certify the run.
    Corpora without the column are math-only and wholly gradeable.
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
            gradeable = [
                document for document in docs
                if document.get("gradeable", True)
            ]
            if gradeable:
                panel.append(gradeable[0])
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
    # Chunked so the batch encoder's Python lists (~28 bytes per token) never
    # hold more than one chunk; each chunk is packed to int32 immediately.
    chunk = 4096
    for chunk_start in range(0, len(documents), chunk):
        batch = documents[chunk_start:chunk_start + chunk]
        prompt_texts = [
            document["problem"] + instruction_suffix for document in batch
        ]
        completion_texts = []
        for document, prompt_text in zip(batch, prompt_texts, strict=True):
            body = document["document"]
            if not body.startswith(prompt_text):
                raise ValueError(
                    "document does not start with its own prompt: "
                    f"{body[:80]!r}"
                )
            completion_texts.append(body[len(prompt_text):])
        for prompt_ids, completion_ids, body in zip(
            encode_many(tokenizer, prompt_texts),
            encode_many(tokenizer, completion_texts),
            (document["document"] for document in batch),
            strict=True,
        ):
            prompt_ids = frame_prompt(tokenizer, prompt_ids)
            if not completion_ids:
                raise ValueError(
                    f"empty completion in document {body[:80]!r}"
                )
            if len(completion_ids) < min_completion_tokens:
                continue
            # +1 for the stop/separator target after the completion.
            if len(prompt_ids) + len(completion_ids) + 1 > seq_len:
                continue
            ids = np.fromiter(
                itertools.chain(prompt_ids, completion_ids),
                dtype=np.int32,
                count=len(prompt_ids) + len(completion_ids),
            )
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
    stop = np.array([tokenizer.eos_id()], dtype=np.int32)
    compliant = sum(
        1
        for document in documents
        if structural_format_ok(
            np.concatenate((document.ids[document.prompt_length:], stop)),
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
) -> list[tuple[np.ndarray, np.ndarray]]:
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
    # Rows are int32 arrays rather than Python int lists. At seq_len 5120 a
    # packed corpus is tens of thousands of rows and each row carried two
    # 5120-element int lists, ~25 GB of interpreter objects for a 900k-
    # document corpus; as int32 the same content is ~2.8 GB.
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    row_tokens = np.full(seq_len, separator, dtype=np.int32)
    filled = 0
    spans: list[tuple[int, int, int]] = []  # (offset, prompt_length, length)

    def close_row() -> None:
        nonlocal filled
        row_tokens[filled] = separator
        length_with_stop = filled + 1
        row_tokens[length_with_stop:] = separator
        targets = np.full(seq_len, IGNORE_INDEX, dtype=np.int32)
        for offset, prompt_length, length in spans:
            start = offset + prompt_length - 1
            stop = offset + length
            # Targets are the next token, exactly as the list version read
            # tokens[position + 1]; the stop slot is covered because the
            # separator was written at ``filled`` above.
            targets[start:stop] = row_tokens[start + 1:stop + 1]
        rows.append((row_tokens.copy(), targets))
        row_tokens[:] = separator
        filled = 0
        spans.clear()

    for index in order:
        document = documents[index]
        length = int(document.ids.shape[0])
        if filled + length + 1 > seq_len:
            close_row()
        spans.append((filled, document.prompt_length, length))
        row_tokens[filled:filled + length] = document.ids
        filled += length
    if filled:
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
        id(parameter)
        for parameter in backbone.parameters()
        if parameter.requires_grad
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


@dataclass(frozen=True)
class DeviceBatch:
    """One micro-batch on the device, with its supervision fixed on the host.

    The supervised slots are read from the packed numpy targets on the host,
    so the device never has to report how many there are: a boolean-mask
    gather would size its output from device data, which stalls the host on
    every micro-batch and hands the compiler a data-dependent shape.
    """

    inputs: torch.Tensor  # [B, T] int64
    positions: torch.Tensor  # [N] int64 flat indices into B*T
    labels: torch.Tensor  # [N] int64 next-token targets
    supervised: int  # N


def device_batch(
    rows: list[tuple[np.ndarray, np.ndarray]], device: torch.device
) -> DeviceBatch:
    tokens = np.stack([tokens for tokens, _ in rows])
    targets = np.stack([targets for _, targets in rows]).reshape(-1)
    positions = np.flatnonzero(targets != IGNORE_INDEX)
    labels = targets[positions]

    def upload(array: np.ndarray) -> torch.Tensor:
        host = torch.from_numpy(array)
        if device.type == "cuda":
            host = host.pin_memory()
        return host.to(device=device, non_blocking=True).long()

    return DeviceBatch(
        inputs=upload(tokens),
        positions=upload(positions),
        labels=upload(labels),
        supervised=int(positions.size),
    )


def readout_ce_sum(
    backbone, features: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Summed CE of the renderer's softcapped logits at the given features.

    The backbone's fused readout keeps only the bf16 readout GEMM output and
    a logsumexp per position for backward, not the [N, V] fp32 logits.
    """
    with torch.autocast(features.device.type, dtype=torch.bfloat16):
        return -backbone.target_logprobs_from_features(features, labels).sum()


class SupervisedCE:
    """Summed completion CE over a packed micro-batch.

    The vocab readout renders ONLY the supervised positions: the readout is
    positionwise, so gathering features first is exact, and it keeps the
    ~1/3 of packed slots that carry no loss off the model's largest matmul.
    Full-sequence logits at 50k vocab would also retain multiple [B*S, V]
    fp32 tensors through the softcap's autograd graph.

    ``compiled`` compiles each residual block independently, as pretraining
    does, so an opaque FLA recurrence splits the graph at one mixer instead
    of forcing the whole model eager; the readout, softcap and cross-entropy
    compile as one region with a dynamic supervised-slot count, so the
    [N, V] logits are produced and consumed by fused kernels. Blocks are
    compiled as modules, so only their ``__call__`` changes: the sampling
    gate's decode path calls ``block.attn``/``block.mlp`` directly and is
    unaffected.
    """

    def __init__(self, backbone, compiled: bool):
        self.backbone = backbone
        self.readout = readout_ce_sum
        self.dynamo_config = contextlib.nullcontext
        if compiled:
            for block in backbone.blocks:
                block.compile(dynamic=False)
            self.readout = torch.compile(readout_ce_sum, dynamic=True)
            # Every block shares ``Block.forward``'s code object, whose cache
            # holds one graph per {KDA, attention} x {grad, no-grad} x
            # micro-batch row count (the full split, each epoch's tail step,
            # the holdout tail): 8 in a single epoch, exactly Dynamo's
            # default budget. Past the budget Dynamo would silently run the
            # block eagerly, so a shape nobody anticipated stops the run
            # instead. Scoped to this loss: the sampling gate compiles its
            # own decode graphs in the same process.
            self.dynamo_config = functools.partial(
                torch._dynamo.config.patch,
                recompile_limit=64,
                fail_on_recompile_limit_hit=True,
            )

    def __call__(self, batch: DeviceBatch) -> torch.Tensor:
        with self.dynamo_config():
            return self._supervised_ce(batch)

    def _supervised_ce(self, batch: DeviceBatch) -> torch.Tensor:
        backbone = self.backbone
        with torch.autocast(batch.inputs.device.type, dtype=torch.bfloat16):
            token_latent = backbone.embed_tokens(batch.inputs)
            belief = backbone.temporal_belief_from_token_latent(token_latent)
            features = torch.cat(
                (
                    token_latent.flatten(0, 1).index_select(
                        0, batch.positions
                    ),
                    belief.flatten(0, 1).index_select(0, batch.positions),
                ),
                dim=-1,
            )
        return self.readout(backbone, features, batch.labels)


class AsyncScalar:
    """A device scalar copied to the host without draining the stream."""

    def __init__(self, value: torch.Tensor):
        if value.device.type != "cuda":
            self._host, self._event = value.detach().cpu(), None
            return
        self._host = torch.empty(
            (), dtype=value.dtype, device="cpu", pin_memory=True
        )
        self._host.copy_(value.detach(), non_blocking=True)
        self._event = torch.cuda.Event()
        self._event.record()

    def read(self) -> float:
        if self._event is not None:
            self._event.synchronize()
        return float(self._host)


def accumulate_step_gradients(
    loss_fn: SupervisedCE,
    step_rows: list[tuple[np.ndarray, np.ndarray]],
    rows_per_micro_batch: int,
    device: torch.device,
) -> torch.Tensor:
    """Backward one optimizer step's rows; returns its mean CE on device.

    The loss is normalized by the step-wide supervised count, so any
    micro-batch split of ``step_rows`` computes the same gradient. Nothing
    here synchronizes with the device; the caller reads the returned scalar
    when it logs.
    """
    supervised_total = sum(
        int(np.count_nonzero(targets != IGNORE_INDEX))
        for _, targets in step_rows
    )
    step_loss = torch.zeros((), device=device)
    for micro_start in range(0, len(step_rows), rows_per_micro_batch):
        batch = device_batch(
            step_rows[micro_start:micro_start + rows_per_micro_batch], device
        )
        loss = loss_fn(batch) / supervised_total
        loss.backward()
        step_loss += loss.detach()
    return step_loss


@torch.no_grad()
def holdout_ce(
    loss_fn: SupervisedCE,
    rows: list[tuple[np.ndarray, np.ndarray]],
    rows_per_batch: int,
    device: torch.device,
) -> float:
    """Mean completion CE (nats/token) over the held-out packed rows."""
    backbone = loss_fn.backbone
    was_training = backbone.training
    backbone.eval()
    total = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    for start in range(0, len(rows), rows_per_batch):
        batch = device_batch(rows[start:start + rows_per_batch], device)
        total += loss_fn(batch)
        count += batch.supervised
    if was_training:
        backbone.train()
    return float(total) / count


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


def assert_gate_panel_fits(panel, tokenizer, args, instruction_suffix) -> None:
    """Refuse a gate panel whose prompts would be front-truncated.

    ``encode_prompt`` keeps BOS plus the LAST ``max_tokens-1`` tokens, so a
    problem that overflows the budget loses its opening and the gate would
    report accuracy on a question the policy never saw in full. The gate is
    the run's certificate, so this is checked at startup as well as here --
    a one-epoch pass over a 900k-document corpus must not spend hours before
    discovering its panel cannot be graded.
    """
    overflow = [
        length
        for length in (
            len(encode_prompt(tokenizer, document["problem"] + instruction_suffix))
            for document in panel[: args.gate_prompts]
        )
        if length > args.gate_prompt_tokens
    ]
    if overflow:
        raise ValueError(
            f"{len(overflow)} of {min(args.gate_prompts, len(panel))} gate "
            f"prompts exceed --gate-prompt-tokens {args.gate_prompt_tokens} "
            f"(longest {max(overflow)}); encode_prompt would drop their "
            "opening tokens and the gate would grade a truncated question"
        )


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
    assert_gate_panel_fits(panel, tokenizer, args, instruction_suffix)
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
        batch_trajectories=args.gate_batch_trajectories,
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


def save_backbone_checkpoint(
    backbone, path: Path, sft_metadata: dict, trained_seq_len: int
) -> None:
    """The exact payload shape ``model_io.load_model`` consumes, atomically."""
    payload = {
        "model": {
            key: value.detach().cpu()
            for key, value in backbone.state_dict().items()
        },
        "model_config": backbone.model_config,
        "architecture": backbone.architecture,
        "train_seq_len": backbone.train_context_tokens,
        # train_seq_len stays the PRETRAINED bound: every downstream loader
        # derives its budgets from it and RL stage 2 must stay inside 1024.
        # trained_seq_len records the window this SFT pass actually trained.
        "trained_seq_len": trained_seq_len,
        "context_extension": trained_seq_len > backbone.train_context_tokens,
        "sft": sft_metadata,
    }
    staging = path.with_name(path.name + ".tmp")
    torch.save(payload, staging)
    os.replace(staging, path)


RESUME_STATE_SCHEMA = "sft_exact_resume/v1"
RESUME_STATE_NAME = "resume.pt"
# Arguments that steer how a run is checkpointed, not what it trains, so a
# resume may change them.
RESUME_EXEMPT_ARGS = frozenset({"resume", "checkpoint_interval_seconds"})
# EX_TEMPFAIL: the run stopped at a checkpoint and is meant to be resumed.
EXIT_CHECKPOINTED = 75
# Everything that executes on both sides of a resume boundary: the update
# path, the model, the optimizers and the schedule. A resumed run whose code
# differs is a different run. The sampling gate runs after the last step in
# one process, so its modules need no binding.
RESUME_BOUND_MODULES = (
    "train_gpt",
    "postraining.sft_trace_train",
    "postraining.core",
    "postraining.model_io",
    "postraining.muon",
    "postraining.kda_backbone",
    "postraining.nano_backbone",
    "postraining.vapo.config",
    "pretraining.latent_moe_training",
    "pretraining.nanogpt_mini.nanogpt_mini_kda_model",
)


def resume_source_sha256() -> dict[str, str]:
    """Byte hashes of the modules a resumed run must share."""
    import importlib

    return {
        name: file_sha256(Path(importlib.import_module(name).__file__))
        for name in RESUME_BOUND_MODULES
    }


def packed_schedule_sha256(
    epoch_rows: list[list[tuple[np.ndarray, np.ndarray]]],
) -> str:
    """Digest of every packed row in step order, targets included.

    Resume recomputes the schedule from the corpus rather than storing
    gigabytes of rows, so this is what proves the recomputation reproduced
    the interrupted run's data order bit for bit.
    """
    digest = hashlib.sha256()
    for rows in epoch_rows:
        digest.update(len(rows).to_bytes(8, "little"))
        for tokens, targets in rows:
            digest.update(tokens.tobytes())
            digest.update(targets.tobytes())
    return digest.hexdigest()


def resume_contract(args: argparse.Namespace, **identity) -> dict:
    """What a resumed run must share with the run that wrote the state."""
    return {
        "resume_schema": RESUME_STATE_SCHEMA,
        "sft_checkpoint_schema": SFT_CHECKPOINT_SCHEMA,
        "torch_version": torch.__version__,
        "args": {
            key: value
            for key, value in sorted(vars(args).items())
            if key not in RESUME_EXEMPT_ARGS
        },
        **identity,
    }


def rng_state() -> dict:
    """Every global generator, for a resumed run that draws any randomness.

    Training draws none today and the sampling gate reseeds, so this is a
    guard for future stochastic training (dropout, sampled packing) rather
    than a current dependency.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_rng_state(state: dict) -> None:
    """Inverse of ``rng_state``."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def fsync_path(path: Path) -> None:
    """Flush a file's (or directory's) data and metadata to stable storage."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_resume_state(
    path: Path,
    *,
    contract: dict,
    backbone,
    optimizers: list[torch.optim.Optimizer],
    step: int,
    progress: dict,
) -> None:
    """Everything the next step depends on, written atomically.

    ``progress`` is the loop's own bookkeeping (elapsed time, last holdout
    CE, the metrics byte offset, resume history), restored verbatim.
    """
    payload = {
        "contract": contract,
        "step": step,
        "model": backbone.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "rng": rng_state(),
        "progress": progress,
    }
    staging = path.with_name(path.name + ".tmp")
    torch.save(payload, staging)
    # Durable before it replaces the previous state: after a host crash the
    # rename must never expose a truncated file in place of a good one.
    fsync_path(staging)
    os.replace(staging, path)
    fsync_path(path.parent)


def resume_contract_mismatch(saved: dict, current: dict) -> list[str]:
    """Dotted names of the contract fields that differ."""
    changed = []
    for key in sorted(saved.keys() | current.keys()):
        before, after = saved.get(key), current.get(key)
        if before == after:
            continue
        if isinstance(before, dict) and isinstance(after, dict):
            changed.extend(
                f"{key}.{name}"
                for name in resume_contract_mismatch(before, after)
            )
        else:
            changed.append(key)
    return changed


def require_resume_contract(path: Path, contract: dict, saved: dict) -> None:
    changed = resume_contract_mismatch(saved, contract)
    if changed:
        raise ValueError(
            f"refusing to resume {path}: the run contract changed "
            f"({', '.join(changed)})"
        )


def load_resume_contract(path: Path) -> dict:
    """The saved contract alone; mmap keeps the tensors on disk."""
    return torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )["contract"]


def load_resume_state(
    path: Path,
    *,
    contract: dict,
    backbone,
    optimizers: list[torch.optim.Optimizer],
) -> tuple[int, dict]:
    """Restore a run in place; refuse any change to what it trains."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require_resume_contract(path, contract, payload["contract"])
    backbone.load_state_dict(payload["model"], strict=True)
    if len(payload["optimizers"]) != len(optimizers):
        raise ValueError(f"{path} holds a different optimizer layout")
    for optimizer, state in zip(optimizers, payload["optimizers"]):
        optimizer.load_state_dict(state)
    restore_rng_state(payload["rng"])
    return payload["step"], payload["progress"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_trace_manifest(
    traces: str | Path, *, answer_fence: bool
) -> Path | None:
    """Bind answer-fenced SFT to current-schema immutable corpus bytes."""
    if not answer_fence:
        return None
    traces = Path(traces)
    manifest_path = traces.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(
            f"answer-fenced SFT requires corpus manifest {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid trace manifest {manifest_path}: {error}"
        ) from error
    require_answer_fence_prompt_schema(
        manifest, answer_fence=True, source=str(manifest_path)
    )
    expected_hash = manifest.get("output_sha256")
    actual_hash = file_sha256(traces)
    if expected_hash != actual_hash:
        raise ValueError(
            f"trace manifest hash mismatch for {traces}: "
            f"{expected_hash!r} != {actual_hash!r}"
        )
    return manifest_path


def create_fresh_run_dir(path: Path) -> None:
    """Create one run root; only ``--resume`` may reopen it."""
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
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="admit rows whose 'verified' column is False. Required for the "
        "large published instruction corpora, whose solutions are "
        "model-generated and are not re-verified by this repository",
    )
    # One epoch. Repeated passes over a small trace set drive holdout
    # completion CE down (3.9449 -> 0.9009 over three epochs on the retired v6
    # traces) while the derived policy still scored 0.00 on every DeepMind
    # interpolate module: that gap is memorisation of the trace set, not
    # reasoning. A single pass over a large corpus is the idiomatic setting.
    parser.add_argument("--epochs", type=int, default=1)
    # None = the checkpoint's own pretraining context. Packing past that
    # window makes the model attend at positions RoPE never saw.
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument(
        "--allow-context-extension",
        action="store_true",
        help="train past the checkpoint's pretraining context. Off by "
        "default: a longer window runs without a code change because RoPE is "
        "computed from torch.arange(T), so the failure would otherwise be "
        "silent extrapolation rather than an error. Setting it makes the "
        "extension deliberate and records it in the checkpoint's metadata.",
    )
    # The step-wide loss normalization makes any micro-batch split of a
    # step's rows compute the same gradient, so the split is a memory knob.
    # One micro-batch is fastest: 8x5120 rows measured 110 ms/step at 9.4 GiB
    # peak (scripts/benchmark_sft_step.py, job 9155) against 130 ms for 2x4
    # (job 9152).
    parser.add_argument("--rows-per-micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
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
    # None = context-derived: keep the 768-token generation budget the
    # panel needs and give the prompt whatever the window has left,
    # capped at the 512 a bare math problem can never exceed (p99 is
    # 136 tokens). A short-context checkpoint therefore samples inside
    # its window instead of extrapolating RoPE mid-gate.
    parser.add_argument("--gate-max-new-tokens", type=int, default=None)
    parser.add_argument("--gate-prompt-tokens", type=int, default=None)
    parser.add_argument("--gate-think-min-tokens", type=int, default=1)
    # The gate's GPU rollout width. ``evaluate_latent_math`` packs
    # ``batch_trajectories // gate_samples`` problem groups into each rollout,
    # so 128 runs a 128-prompt panel as 8 sequential passes and the device has
    # ample room for more at 64M parameters. It stays at 128 anyway: the gate
    # is sampled with ``pin_emit=True``, which disables the counter-based
    # per-row uniform in ``latent_rollout`` and draws from the global RNG at
    # the live batch width, so this knob changes which tokens are sampled and
    # therefore the reported gate accuracy. 128 is the width job 9040's
    # numbers were measured at and every comparison against the canonical
    # base depends on matching it. The whole panel costs ~30 s at the measured
    # 13.5k tok/s (job 9082), so there is no throughput case for breaking that
    # comparability. Widen it only alongside a batch-invariant sampling path.
    parser.add_argument("--gate-batch-trajectories", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    # Wall-clock, not steps: the point is to bound the work a stop loses
    # whatever the step time. A save is ~1 GB (weights plus both optimizers'
    # state) and takes seconds, so five minutes costs well under 1%. Named
    # as in train_latent_vapo, whose rolling checkpoints use the same knob.
    parser.add_argument(
        "--checkpoint-interval-seconds", type=float, default=300.0
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"continue postraining/runs/<name> from its {RESUME_STATE_NAME}. "
        "Every argument except the checkpoint cadence must match, and the "
        "corpus, base checkpoint and recomputed packed schedule must hash "
        "identically. SIGTERM (mlq cancel) saves a checkpoint at the next "
        f"step boundary and exits {EXIT_CHECKPOINTED}.",
    )
    args = parser.parse_args()
    if not args.checkpoint_interval_seconds > 0:
        parser.error("--checkpoint-interval-seconds must be positive")
    for field in (
        "epochs", "seq_len", "rows_per_micro_batch", "grad_accum",
        "holdout_problems", "eval_every", "gate_prompts", "gate_samples",
        "gate_max_new_tokens", "gate_prompt_tokens",
        "gate_think_min_tokens", "gate_batch_trajectories",
    ):
        value = getattr(args, field)
        if value is not None and value < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if not 0.0 < args.lr_scale <= 1.0:
        parser.error("--lr-scale must be in (0, 1]")
    if args.min_completion_tokens < 0:
        parser.error("--min-completion-tokens must be nonnegative")

    try:
        traces_manifest = validate_trace_manifest(
            args.traces, answer_fence=args.answer_fence
        )
    except ValueError as error:
        parser.error(str(error))

    run_dir = Path("postraining/runs") / args.name
    resume_path = run_dir / RESUME_STATE_NAME
    metrics_path = run_dir / "metrics.jsonl"
    # A run is finished when its gate result exists: the final checkpoint is
    # written before the gate, so a stop during the gate still resumes.
    if args.resume:
        if not resume_path.is_file():
            parser.error(f"--resume: no {resume_path}")
        if (run_dir / "result.json").exists():
            parser.error(f"--resume: {run_dir} already finished")
    elif run_dir.exists():
        # Checked now, created only once preparation is done: a stop during
        # the minutes of tokenization must not strand an empty run dir that
        # neither a fresh run nor --resume will open.
        parser.error(f"refusing to overwrite existing SFT run directory {run_dir}")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    backbone = load_model(args.checkpoint, device)
    if not hasattr(backbone, "target_logprobs_from_features"):
        parser.error(
            f"{type(backbone).__name__} has no fused target readout; SFT "
            "trains the nano/KDA post-training backbones"
        )
    context_tokens = backbone.train_context_tokens
    if args.seq_len is None:
        args.seq_len = context_tokens
    # The window the run actually trains and gates at. Beyond the pretrained
    # context this is extrapolation for the full-attention layers until the
    # run itself trains those positions, which is why it is opt-in.
    effective_context = context_tokens
    if args.seq_len > context_tokens:
        if not args.allow_context_extension:
            parser.error(
                f"--seq-len {args.seq_len} exceeds the checkpoint's "
                f"pretraining context ({context_tokens}); pass "
                "--allow-context-extension to train the longer window "
                "deliberately"
            )
        effective_context = args.seq_len
    # The prompt budget is floored, and the response budget yields to it.
    # Deriving max_new first and giving the prompt whatever remains looks
    # safe because the result is always >= 1, but for a context in the
    # 770..1023 range it leaves a 2..255-token prompt window -- and
    # encode_prompt keeps BOS plus the LAST max_tokens-1 tokens, so problems
    # would silently lose their opening and the gate would report accuracy on
    # truncated questions. A gate that cannot see the whole prompt is not a
    # measurement, so the run fails instead.
    GATE_PROMPT_FLOOR = 256
    if args.gate_prompt_tokens is None:
        args.gate_prompt_tokens = min(
            512, max(GATE_PROMPT_FLOOR, effective_context // 4)
        )
    if args.gate_max_new_tokens is None:
        args.gate_max_new_tokens = min(
            768, effective_context - args.gate_prompt_tokens
        )
    if args.gate_max_new_tokens < 1:
        parser.error(
            f"gate context ({effective_context}) cannot hold a "
            f"{args.gate_prompt_tokens}-token gate prompt and any response"
        )
    gate_window = args.gate_prompt_tokens + args.gate_max_new_tokens
    if gate_window > effective_context:
        parser.error(
            f"gate budget {args.gate_prompt_tokens}+"
            f"{args.gate_max_new_tokens} exceeds the training window "
            f"({effective_context})"
        )
    # Every argument is resolved now, so the whole contract except the
    # packed schedule is known: a resume that would be refused fails here,
    # before minutes of tokenization, not after.
    traces_sha256 = file_sha256(Path(args.traces))
    base_checkpoint_sha256 = file_sha256(Path(args.checkpoint))
    contract = resume_contract(
        args,
        traces_sha256=traces_sha256,
        base_checkpoint_sha256=base_checkpoint_sha256,
        source_sha256=resume_source_sha256(),
    )
    if args.resume:
        saved = load_resume_contract(resume_path)
        saved.pop("packed_schedule_sha256", None)
        try:
            require_resume_contract(resume_path, contract, saved)
        except ValueError as error:
            parser.error(str(error))
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    frozen_moe_router_parameters = (
        backbone.freeze_moe_routing_()
        if hasattr(backbone, "freeze_moe_routing_")
        else 0
    )
    if frozen_moe_router_parameters:
        enable_quantile_balance_collection(backbone, num_bins=1000)
    if args.answer_fence and not args.think_tokens:
        parser.error("--answer-fence requires --think-tokens")
    tokenizer = load_posttraining_tokenizer(
        backbone.architecture,
        FreshHyperparameters.tokenizer_path,
        think_tokens=args.think_tokens,
        answer_tokens=args.answer_fence,
        tokenizer_provenance=backbone.model_config.get(
            "tokenizer_provenance"
        ),
    )
    if args.think_tokens:
        register_special_tokens(backbone, tokenizer)

    instruction_suffix = (
        INSTRUCTION_SUFFIX_ANSWER if args.answer_fence else INSTRUCTION_SUFFIX
    )
    documents = load_documents(Path(args.traces), args.allow_unverified)
    train_docs, holdout_docs, panel = split_holdout(
        documents, args.holdout_problems
    )
    # The panel holds only gradeable problems, so a mixed-domain corpus can
    # yield fewer than --holdout-problems. The gate silently truncating to a
    # short panel would change what its accuracy means between runs, so the
    # shortfall is an error the caller resolves by raising
    # --holdout-problems.
    if len(panel) < args.gate_prompts:
        raise ValueError(
            f"sampling gate needs {args.gate_prompts} gradeable held-out "
            f"problems but the split yielded {len(panel)} from "
            f"{args.holdout_problems} held-out problems; raise "
            "--holdout-problems or lower --gate-prompts"
        )
    # Checked here, not only inside the gate: the gate runs after the whole
    # training pass, and an ungradeable panel must not cost a full epoch to
    # discover.
    assert_gate_panel_fits(panel, tokenizer, args, instruction_suffix)
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
        if fence_fraction != 1.0:
            parser.error(
                f"--answer-fence: only {fence_fraction:.1%} of documents "
                "carry the anchored <think>...</think><answer>...</answer> "
                "completion shape; every row must match the exact prompt "
                "boundary and this corpus cannot safely teach the "
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
    # One (rows, offset) slice per optimizer step, in order, so a resumed
    # run continues at exactly the slice the interrupted one would have run.
    step_slices = [
        (rows, start)
        for rows in epoch_rows
        for start in range(0, len(rows), rows_per_step)
    ]
    total_steps = len(step_slices)
    print(
        f"schedule: {total_steps} steps "
        f"({rows_per_step} rows x {args.seq_len} tokens per step)"
    )
    contract["packed_schedule_sha256"] = packed_schedule_sha256(epoch_rows)

    optimizers = build_optimizers(backbone, args.lr_scale)
    step = 0
    progress = {
        "elapsed_seconds": 0.0,
        "last_val_ce": None,
        "metrics_bytes": 0,
        "resumed_at_steps": [],
    }
    if args.resume:
        step, progress = load_resume_state(
            resume_path,
            contract=contract,
            backbone=backbone,
            optimizers=optimizers,
        )
        # Entries logged after the checkpoint describe steps this run is
        # about to redo; the stream must hold each step exactly once. The
        # save fsyncs the stream first, so a shorter file means it was lost
        # or edited, and truncate() would pad it with NULs.
        if metrics_path.stat().st_size < progress["metrics_bytes"]:
            raise ValueError(
                f"{metrics_path} is shorter than the {progress['metrics_bytes']} "
                f"bytes {resume_path} recorded"
            )
        with metrics_path.open("r+b") as stream:
            stream.truncate(progress["metrics_bytes"])
        progress["resumed_at_steps"].append(step)
        print(f"resumed {resume_path} at step {step}/{total_steps}")
    loss_fn = SupervisedCE(backbone, compiled=True)
    backbone.train()
    started = time.perf_counter() - progress["elapsed_seconds"]

    def log(entry: dict) -> None:
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")

    last_val_ce: float | None = progress["last_val_ce"]

    def log_val(at_step: int) -> float:
        nonlocal last_val_ce
        ce = holdout_ce(
            loss_fn, holdout_rows, args.rows_per_micro_batch, device
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

    pending_train_entry: dict | None = None

    def flush_train_entry() -> None:
        nonlocal pending_train_entry
        if pending_train_entry is None:
            return
        log(
            {
                key: (
                    value.read()
                    if isinstance(value, AsyncScalar)
                    else float(value) if torch.is_tensor(value) else value
                )
                for key, value in pending_train_entry.items()
            }
        )
        pending_train_entry = None

    def save_checkpoint() -> None:
        flush_train_entry()
        fsync_path(metrics_path)
        progress.update(
            elapsed_seconds=time.perf_counter() - started,
            last_val_ce=last_val_ce,
            metrics_bytes=metrics_path.stat().st_size,
        )
        save_resume_state(
            resume_path,
            contract=contract,
            backbone=backbone,
            optimizers=optimizers,
            step=step,
            progress=progress,
        )

    # mlq cancel sends SIGTERM. From creating the run dir until the last
    # step's checkpoint, stop at the next step boundary with a checkpoint
    # rather than dying mid-step with up to a cadence of work unsaved; the
    # handler only sets a flag, so the step in flight finishes. Outside that
    # window the default handler applies: preparation has nothing to save,
    # and after the last step's checkpoint a stop resumes into the final
    # holdout, save and gate.
    stop_requested = False

    def request_stop(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    def stop_if_requested() -> None:
        if stop_requested:
            print(
                f"SIGTERM: {resume_path} holds step {step}/{total_steps}; "
                "continue with --resume",
                flush=True,
            )
            raise SystemExit(EXIT_CHECKPOINTED)

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    if step == 0:
        create_fresh_run_dir(run_dir)
        metrics_path.write_text("")
        log_val(0)
        # Resumable from the first moment the run dir exists.
        save_checkpoint()
        stop_if_requested()
    last_checkpoint = time.perf_counter()
    for rows, start in step_slices[step:]:
        step_rows = rows[start:start + rows_per_step]
        scale = lr_scale_at(step, total_steps, args.warmup_steps)
        momentum = muon_momentum_at(step)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * scale
                if isinstance(optimizer, Muon):
                    group["mu"] = momentum
        if frozen_moe_router_parameters:
            reset_quantile_balance_accumulators(backbone)
        step_loss = accumulate_step_gradients(
            loss_fn, step_rows, args.rows_per_micro_batch, device
        )
        if frozen_moe_router_parameters:
            moe_load_cv, moe_max_load = apply_accumulated_quantile_balance(
                backbone
            )
        for optimizer in optimizers:
            optimizer.step()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        step += 1
        # Logged one step late through an async copy: ``float`` on a
        # device tensor waits for everything queued on the stream, so
        # reading this step's loss now would drain the queue and idle
        # the GPU while the host prepares the next step. The event
        # waits only for this step's copy, which the next flush finds
        # long finished.
        flush_train_entry()
        pending_train_entry = {
            "type": "train",
            "step": step,
            "train_loss": AsyncScalar(step_loss),
            "lr_scale": scale,
            # Host time once the step is queued; the device finishes it
            # later by however much work is still in the launch queue.
            "train_time_ms": (time.perf_counter() - started) * 1000,
            **(
                {
                    "moe_load_cv2": AsyncScalar(moe_load_cv),
                    "moe_max_load": AsyncScalar(moe_max_load),
                }
                if frozen_moe_router_parameters
                else {}
            ),
        }
        if step % args.eval_every == 0:
            flush_train_entry()
            log_val(step)
        if (
            stop_requested
            or step == total_steps
            or time.perf_counter() - last_checkpoint
            >= args.checkpoint_interval_seconds
        ):
            save_checkpoint()
            last_checkpoint = time.perf_counter()
            print(f"step {step}: saved {resume_path}", flush=True)
            stop_if_requested()
    signal.signal(signal.SIGTERM, previous_sigterm)

    flush_train_entry()
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
            "traces_sha256": traces_sha256,
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "traces_manifest": (
                str(traces_manifest) if traces_manifest is not None else None
            ),
            "args": vars(args),
            "steps": step,
            "resumed_at_steps": progress["resumed_at_steps"],
            "moe_router_frozen_parameters": frozen_moe_router_parameters,
            "moe_quantile_balance_bins": (
                1000 if frozen_moe_router_parameters else None
            ),
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
        trained_seq_len=args.seq_len,
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
