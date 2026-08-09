#!/usr/bin/env python3
"""Two-stage Bolmo byteification of the latest KDA/NoPE source checkpoint.

All expensive invocations of this script must go through ``mlq``.  A stopped
run is a chronological prefix of the paper-scaled schedule: for the default
2,000-update plan, ``STOP_AFTER_STEP=1000`` completes 667 Stage-1 updates and
then executes the first 333 of 1,333 planned Stage-2 updates.  Each phase uses
its own complete warmup-plus-linear-decay horizon.
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterator, Mapping, Sequence

import torch
from torch import Tensor

from pretraining.bolmo import (
    BolmoArchitecture,
    BolmoBatch,
    BolmoLoss,
    BolmoModel,
    BolmoTeacher,
    BolmoValidationStatistics,
    DEFAULT_TEACHER_POSITIONS_PER_CHUNK,
    SOURCE_EOT_ID,
    parameter_report,
)
from pretraining.bolmo_data import (
    BOLMO_DATA_SCHEMA,
    load_example_shard,
    scored_source_token_count,
)
from pretraining.nanogpt_mini.nanogpt_mini_kda_model import (
    CausalSelfAttention,
    KimiDeltaAttention,
)
from pretraining.source_muon import (
    SOURCE_MUON_ALGORITHM,
    Muon as SourceMuon,
    PerHeadMuon,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "logs/k3_v8_armC_kda8_cosine_nextlat_nope_stop1k_final_model.pt"
DEFAULT_SOURCE_SHA256 = "b0ce261e7471b8092d1c859bf676b09fad46568991a908518cd010c690c76cc3"
DEFAULT_SOURCE_CONTEXT = 2048
# Both entries run the same architecture and the same two-stage procedure. They
# differ only in the optimizer, learning rates and schedule: ``paper`` is
# Table 8 of arXiv:2512.15586v2, ``source`` is the recipe the KDA/NoPE trunk was
# pretrained under, so that a Bolmo arm can be compared against the source model
# without confounding architecture with training recipe.
TRAINING_RECIPES = {
    "paper": "bolmo_paper_v3",
    "source": "bolmo_source_muon_v1",
}


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(data_dir: Path) -> dict:
    path = data_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != BOLMO_DATA_SCHEMA:
        raise ValueError(
            f"{path} has schema {manifest.get('schema')!r}, expected {BOLMO_DATA_SCHEMA!r}"
        )
    expected = manifest.get("payload_sha256")
    payload = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    if actual != expected:
        raise ValueError(f"{path} payload hash mismatch: {actual} != {expected}")
    return manifest


def artifact_paths(data_dir: Path, manifest: dict, split: str) -> list[Path]:
    try:
        artifacts = manifest["splits"][split]["artifacts"]
    except KeyError as exc:
        raise KeyError(f"Bolmo dataset has no split {split!r}") from exc
    examples = manifest["examples"]
    expected_source_width = (
        int(examples["training_stored_source_width"])
        if split == "train"
        else int(examples["validation_stored_source_width"])
    )
    paths = [data_dir / artifact["path"] for artifact in artifacts]
    for path, artifact in zip(paths, artifacts, strict=True):
        if artifact.get("source_width") != expected_source_width:
            raise ValueError(
                f"{split} artifact {path} declares source width "
                f"{artifact.get('source_width')} != {expected_source_width}"
            )
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != artifact["sha256"]:
            raise ValueError(f"{path} hash mismatch")
    return paths


def validate_paper_data_manifest(manifest: dict) -> None:
    """Require the released BOS/skip-last row geometry for every split."""

    examples = manifest.get("examples")
    splits = manifest.get("splits")
    if not isinstance(examples, dict) or not isinstance(splits, dict):
        raise ValueError("Bolmo manifest lacks examples or splits")
    sequence_length = examples.get("source_sequence_length")
    if not isinstance(sequence_length, int) or sequence_length <= 1:
        raise ValueError("Bolmo source_sequence_length must exceed one")
    expected_examples = {
        "training_real_source_tokens": sequence_length - 1,
        "training_stored_source_width": sequence_length,
        "validation_stored_source_width": sequence_length + 2,
    }
    mismatches = {
        key: (examples.get(key), expected)
        for key, expected in expected_examples.items()
        if examples.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Bolmo manifest has non-paper row geometry: {mismatches}")
    for split, details in splits.items():
        if not isinstance(details, dict):
            raise ValueError(f"Bolmo split {split!r} is not a mapping")
        expected_context = 0 if split == "train" else 1
        expected_skip_last = split == "train"
        observed = (
            details.get("context_source_tokens"),
            details.get("skip_last_source_token"),
        )
        expected = (expected_context, expected_skip_last)
        if observed != expected:
            raise ValueError(
                f"Bolmo split {split!r} has non-paper context/skip-last "
                f"contract: {observed} != {expected}"
            )


def validate_checkpoint_tokenizer(checkpoint: dict, source_meta: dict) -> None:
    """Fail closed unless dataset source IDs use the checkpoint tokenizer."""

    checkpoint_meta = checkpoint.get("model_config", {}).get(
        "tokenizer_provenance"
    )
    if not isinstance(checkpoint_meta, dict):
        raise ValueError("source checkpoint does not bind tokenizer provenance")
    # The exact tokenizer.json and n-gram hashes bind ordered specials, ToaST
    # merges, numeric vocabulary, and TST settings. The explicit fields make
    # identity failures legible rather than reporting only a digest mismatch.
    expected = {
        "kind": source_meta.get("kind"),
        "vocab_size": source_meta.get("logical_vocab_size"),
        "eot_id": source_meta.get("eot_id"),
        "spec_sha256": source_meta.get("spec_file_sha256"),
        "ngrams_sha256": source_meta.get("ngrams_sha256"),
    }
    mismatches = {
        key: (checkpoint_meta.get(key), value)
        for key, value in expected.items()
        if checkpoint_meta.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "dataset tokenizer differs from source checkpoint: "
            f"{mismatches}"
        )
    if source_meta.get("tst_group_size") != 1 or not source_meta.get(
        "tst_compound"
    ):
        raise ValueError("Bolmo requires the checkpoint's compound TST N=1 tokenizer")


def validate_stage1_resume_checkpoint(
    checkpoint: dict,
    *,
    model: BolmoModel,
    source_sha256: str,
    data_manifest_sha256: str,
    training_recipe: str,
    planned_steps: int,
    planned_stage1: int,
    minimum_boundary_accuracy: float,
    allow_gate_failure: bool = False,
) -> None:
    """Fail closed before using a Stage-1 artifact as a Stage-2 starting point.

    ``allow_gate_failure`` is an explicit operator override for experimental
    Stage-2 arms. The gate is still measured and the failing result is still
    required and preserved; only the refusal is waived.
    """

    expected = {
        "architecture": "bolmo_kda_nope_v1",
        "training_recipe": training_recipe,
        "source_checkpoint_sha256": source_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "completed_steps": planned_stage1,
        "planned_steps": planned_steps,
        "stage1_planned_steps": planned_stage1,
        "stage1_completed_steps": planned_stage1,
        "stage2_completed_steps": 0,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    checkpoint_config = checkpoint.get("model_config")
    current_config = model.export_config()
    if isinstance(checkpoint_config, dict):
        checkpoint_config = json.loads(json.dumps(checkpoint_config))
        checkpoint_architecture = checkpoint_config.get("architecture", {})
        if isinstance(checkpoint_architecture, dict):
            checkpoint_architecture.pop("backend", None)
            checkpoint_architecture.pop("chunk_size", None)
            checkpoint_architecture.pop("autocast_kernel_dtype", None)
    current_config = json.loads(json.dumps(current_config))
    current_architecture = current_config.get("architecture", {})
    if isinstance(current_architecture, dict):
        current_architecture.pop("backend", None)
        current_architecture.pop("chunk_size", None)
        current_architecture.pop("autocast_kernel_dtype", None)
    if checkpoint_config != current_config:
        mismatches["model_config"] = (checkpoint_config, current_config)
    if mismatches:
        raise ValueError(f"Stage-1 resume checkpoint mismatch: {mismatches}")
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Stage-1 resume checkpoint has no model state")
    gate_accuracy = checkpoint.get("stage1_boundary_accuracy")
    if not isinstance(gate_accuracy, (int, float)):
        raise ValueError("Stage-1 resume checkpoint has no boundary gate result")
    if gate_accuracy < minimum_boundary_accuracy:
        if not allow_gate_failure:
            raise ValueError(
                "Stage-1 resume checkpoint failed the paper boundary gate: "
                f"{gate_accuracy:.6f} < {minimum_boundary_accuracy:.6f}"
            )
        print(
            "WARNING: Stage-2 resume overrides the paper boundary gate: "
            f"{gate_accuracy:.6f} < {minimum_boundary_accuracy:.6f}. This run "
            "is experimental and is not a paper-aligned Stage-2 arm.",
            flush=True,
        )
    for diagnostic in ("stage1_boundary_precision", "stage1_boundary_recall"):
        if not isinstance(checkpoint.get(diagnostic), (int, float)):
            raise ValueError(f"Stage-1 resume checkpoint has no {diagnostic}")
    for diagnostic in (
        "stage1_oracle_byte_bpb",
        "stage1_predicted_oracle_bpb_gap",
    ):
        if not isinstance(checkpoint.get(diagnostic), (int, float)):
            raise ValueError(
                f"Stage-1 resume checkpoint has no {diagnostic} diagnostic"
            )
    train_diagnostics = checkpoint.get("stage1_train_diagnostics")
    if not isinstance(train_diagnostics, dict) or not all(
        isinstance(train_diagnostics.get(name), (int, float))
        for name in (
            "boundary",
            "ce",
            "encoder_stitch",
            "encoder_stitch_cosine",
            "decoder_distill",
        )
    ):
        raise ValueError("Stage-1 resume checkpoint has incomplete train diagnostics")


def validate_training_contract(checkpoint: dict, expected: dict) -> None:
    observed = checkpoint.get("training_contract")
    if observed != expected:
        raise ValueError(
            "Stage-1 checkpoint training contract mismatch: "
            f"{observed!r} != {expected!r}"
        )
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, dict) or set(rng_state) != {
        "python",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("Stage-1 checkpoint has incomplete RNG state")


class ExampleStream:
    """Deterministic artifact-local shuffle with bounded CPU memory."""

    def __init__(
        self,
        paths: Sequence[Path],
        *,
        seed: int,
        repeat: bool,
        shuffle: bool,
        source_vocab_size: int | None = None,
        source_pad_id: int | None = None,
        atomic_vocab_size: int | None = None,
        context_source_tokens: int | None = None,
    ):
        if not paths:
            raise ValueError("an example stream requires at least one artifact")
        self.paths = tuple(paths)
        self.seed = seed
        self.repeat = repeat
        self.shuffle = shuffle
        self.source_vocab_size = source_vocab_size
        self.source_pad_id = source_pad_id
        self.atomic_vocab_size = atomic_vocab_size
        self.context_source_tokens = context_source_tokens
        self.epoch = 0
        self._iterator = self._make_iterator()

    def _make_iterator(self) -> Iterator[dict[str, Tensor]]:
        rng = random.Random(self.seed + self.epoch)
        paths = list(self.paths)
        if self.shuffle:
            rng.shuffle(paths)
        for path in paths:
            payload = load_example_shard(
                path,
                source_vocab_size=self.source_vocab_size,
                source_pad_id=self.source_pad_id,
                atomic_vocab_size=self.atomic_vocab_size,
                context_source_tokens=self.context_source_tokens,
            )
            rows = list(range(payload["source_ids"].shape[0]))
            if self.shuffle:
                rng.shuffle(rows)
            for row in rows:
                yield {name: tensor[row] for name, tensor in payload.items()}

    def take(self, count: int) -> list[dict[str, Tensor]]:
        result: list[dict[str, Tensor]] = []
        while len(result) < count:
            try:
                result.append(next(self._iterator))
            except StopIteration:
                if not self.repeat:
                    break
                self.epoch += 1
                self._iterator = self._make_iterator()
        if len(result) != count and self.repeat:
            raise AssertionError("repeating stream failed to provide a full batch")
        return result

    def take_exact(self, count: int) -> list[dict[str, Tensor]]:
        result = self.take(count)
        if len(result) != count:
            raise RuntimeError(
                "Bolmo data stream exhausted before the requested unique batch: "
                f"{len(result)} != {count}"
            )
        return result

    def skip(self, count: int) -> int:
        """Advance deterministically without retaining skipped tensor rows."""

        if count < 0:
            raise ValueError("skip count must be nonnegative")
        skipped = 0
        while skipped < count:
            try:
                next(self._iterator)
                skipped += 1
            except StopIteration:
                if not self.repeat:
                    break
                self.epoch += 1
                self._iterator = self._make_iterator()
        if skipped != count and self.repeat:
            raise AssertionError("repeating stream failed to skip the requested rows")
        return skipped


def collate_rows(rows: Sequence[dict[str, Tensor]], *, byte_pad_id: int, source_pad_id: int) -> BolmoBatch:
    if not rows:
        raise ValueError("cannot collate no rows")
    source_width = rows[0]["source_ids"].numel()
    if any(row["source_ids"].numel() != source_width for row in rows):
        raise ValueError("source widths differ within a batch")
    valid_lengths = [int(row["valid_mask"].sum()) for row in rows]
    byte_width = max(valid_lengths)
    byte_width = (byte_width + 127) // 128 * 128
    count = len(rows)
    source_ids = torch.full((count, source_width), source_pad_id, dtype=torch.long)
    source_valid = torch.zeros((count, source_width), dtype=torch.bool)
    byte_ids = torch.full((count, byte_width), byte_pad_id, dtype=torch.long)
    expanded_ids = torch.full((count, byte_width), source_pad_id, dtype=torch.long)
    boundaries = torch.zeros((count, byte_width), dtype=torch.bool)
    valid = torch.zeros((count, byte_width), dtype=torch.bool)
    score = torch.zeros((count, byte_width), dtype=torch.bool)
    patch_lens = torch.zeros((count, source_width), dtype=torch.long)
    for index, (row, length) in enumerate(zip(rows, valid_lengths, strict=True)):
        source_ids[index].copy_(row["source_ids"].long())
        source_valid[index].copy_(row["source_valid_mask"])
        byte_ids[index, :length].copy_(row["byte_ids"][:length].long())
        expanded_ids[index, :length].copy_(row["expanded_ids"][:length].long())
        boundaries[index, :length].copy_(row["boundary_mask"][:length])
        valid[index, :length] = True
        score[index, :length].copy_(row["score_mask"][:length])
        patch_lens[index].copy_(row["patch_lens"].long())
    return BolmoBatch(
        source_ids=source_ids,
        source_valid_mask=source_valid,
        byte_ids=byte_ids,
        expanded_ids=expanded_ids,
        oracle_boundaries=boundaries,
        valid_mask=valid,
        score_mask=score,
        patch_lens=patch_lens,
    )


def chunks(values: Sequence, size: int) -> Iterator[Sequence]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def split_parameter_groups(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    weight_decay: float,
    lr: float,
) -> list[dict]:
    decay, no_decay = [], []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        # The released recipe exempts only the local embedding tables. All
        # other parameters receive the paper's 0.1 AdamW decay.
        if "embedding" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


def stage1_optimizer(model: BolmoModel, peak_lr: float) -> torch.optim.AdamW:
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(("local_encoder.", "local_decoder."))
    ]
    return torch.optim.AdamW(
        split_parameter_groups(named, weight_decay=0.1, lr=peak_lr),
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )


def stage2_optimizer(
    model: BolmoModel,
    *,
    local_peak_lr: float,
    global_peak_lr: float,
) -> torch.optim.AdamW:
    local_named, global_named = [], []
    for name, parameter in model.named_parameters():
        if name.startswith("global_blocks."):
            global_named.append((name, parameter))
        else:
            local_named.append((name, parameter))
    groups = [
        *split_parameter_groups(local_named, weight_decay=0.1, lr=local_peak_lr),
        *split_parameter_groups(global_named, weight_decay=0.1, lr=global_peak_lr),
    ]
    return torch.optim.AdamW(
        groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )


def linear_warmup_decay(step: int, planned_steps: int, warmup_fraction: float = 0.1) -> float:
    warmup_steps = max(1, round(planned_steps * warmup_fraction))
    if step <= warmup_steps:
        return step / warmup_steps
    return max(0.0, (planned_steps - step) / max(1, planned_steps - warmup_steps))


def cosine_warmup_decay(
    step: int, planned_steps: int, warmup_fraction: float = 0.01
) -> float:
    """The source trainer's cosine schedule, on Bolmo's 1-indexed stage step.

    ``nanogpt_mini_gpt2vocab_kda_3to1_pm_train.set_hparams`` indexes from zero,
    so the conversion here is the whole difference; the warmup ramp and the
    cosine body are evaluated exactly as they were for the source run.
    """

    if not 1 <= step <= planned_steps:
        raise ValueError(f"stage step {step} is outside 1..{planned_steps}")
    zero_based = step - 1
    progress = zero_based / planned_steps
    warmup_steps = max(1, round(planned_steps * warmup_fraction))
    if progress < warmup_fraction:
        # The source compares against the unrounded fraction but divides by the
        # rounded step count, so a horizon whose warmup rounds *down* spends its
        # last warmup step above the peak. The source's own horizons all round
        # exactly, so clamping is parity there and removes the spike elsewhere.
        return min(1.0, step / warmup_steps)
    post_warmup = (progress - warmup_fraction) / (1 - warmup_fraction)
    return 0.5 * (1 + math.cos(math.pi * post_warmup))


@dataclass(frozen=True)
class SourceOptimizerRecipe:
    """The optimizer, learning rates and schedule of the source KDA run.

    The defaults are the values recorded in the source checkpoint's own
    ``training_config``.  Adopting them is a deliberate departure from Table 8
    of arXiv:2512.15586v2, which trains every Bolmo parameter with AdamW at
    7e-4 (Stage 1) and 2.6e-5 / 5.2e-5 (Stage 2) on a warmup-plus-linear
    schedule.  Under that recipe the retained trunk moves about three orders of
    magnitude less than the Muon baseline it is measured against, so a
    paper-recipe Bolmo and the source model differ in architecture *and* in
    training recipe.  This arm removes the second difference.
    """

    embed_lr: float = 0.7
    proj_lr: float = 0.004
    delta_conv_lr: float = 0.004
    scalar_lr: float = 0.015
    muon_lr: float = 0.025
    adam_betas: tuple[float, float] = (0.8, 0.95)
    adam_eps: float = 1e-10
    adam_weight_decay: float = 0.001
    muon_weight_decay: float = 0.05
    muon_momentum: float = 0.95
    muon_momentum_warmup_start: float = 0.85
    muon_momentum_warmup_steps: int = 500
    warmup_fraction: float = 0.01
    # Structural choices that define the arm as much as the learning rates do,
    # so they travel in the contract and a resume rejects a run that changed
    # them. ``per_head_muon`` mirrors the source's ``PER_HEAD_MUON``.
    #
    # ``schedule_scope`` chooses between the two defensible readings of the
    # source's single annealed cosine. ``per_stage`` is the paper's shape: each
    # stage gets its own horizon and its own anneal, which is the honest choice
    # given the stages optimize different parameter sets under different
    # objectives — Stage 1 holds the trunk frozen. But it hands Bolmo two
    # complete anneals where the source got one, and restarts the trunk at the
    # full 0.025 the source only ever applied to a random initialization.
    # ``continuous`` runs one horizon over the whole plan instead, so Bolmo
    # anneals once like the source and the trunk enters mid-schedule at a lower
    # rate. Neither is "matched" — the source had one stage — so the scope is
    # recorded rather than assumed.
    per_head_muon: bool = True
    schedule: str = "cosine"
    momentum_warmup_scope: str = "per_stage"
    schedule_scope: str = "per_stage"

    SCHEDULE_SCOPES = ("per_stage", "continuous")

    def __post_init__(self) -> None:
        if self.schedule != "cosine":
            raise ValueError(
                f"the source recipe is cosine-scheduled, got {self.schedule!r}"
            )
        for name in ("schedule_scope", "momentum_warmup_scope"):
            value = getattr(self, name)
            if value not in self.SCHEDULE_SCOPES:
                raise ValueError(
                    f"{name} must be one of {self.SCHEDULE_SCOPES}, got {value!r}"
                )
        # A continuous learning-rate horizon with a per-stage momentum ramp
        # would restart the momentum warmup in the middle of an unbroken decay,
        # which is neither of the two shapes above.
        if self.momentum_warmup_scope != self.schedule_scope:
            raise ValueError(
                "the Muon momentum warmup must follow the learning-rate "
                f"horizon, got {self.momentum_warmup_scope!r} against "
                f"{self.schedule_scope!r}"
            )

    @property
    def continuous_schedule(self) -> bool:
        return self.schedule_scope == "continuous"

    def as_contract(self) -> dict:
        payload = asdict(self)
        payload["adam_betas"] = list(self.adam_betas)
        payload["muon_algorithm"] = SOURCE_MUON_ALGORITHM
        return payload


# The source trainer keeps the depthwise short convolutions and the decay
# scalars out of Muon by identity, not by rank: ``q/k/v_conv1d.weight`` is
# rank three and would otherwise be orthogonalized as a matrix.
DELTA_CONV_SUFFIXES = (".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")
DELTA_DECAY_SUFFIXES = (".attn.A_log", ".attn.dt_bias")


def global_per_head_projections(
    model: BolmoModel,
) -> list[tuple[torch.nn.Parameter, int]]:
    """Q/K/V projections of the retained trunk, paired with their head counts.

    Mirrors the source trainer's per-head Muon selection.  An unrecognized
    attention type raises rather than silently falling back to whole-matrix
    orthogonalization, which would change the recipe without changing the
    recipe identifier.
    """

    projections: list[tuple[torch.nn.Parameter, int]] = []
    for index, block in enumerate(model.global_blocks):
        attention = block.attn
        if isinstance(attention, KimiDeltaAttention):
            linears = (attention.q_proj, attention.k_proj, attention.v_proj)
        elif isinstance(attention, CausalSelfAttention):
            linears = (attention.q, attention.k, attention.v)
        else:
            raise ValueError(
                f"global block {index} has attention type "
                f"{type(attention).__name__}, which the source per-head Muon "
                "recipe does not cover"
            )
        projections.extend(
            (linear.weight, attention.num_heads) for linear in linears
        )
    return projections


def source_recipe_optimizers(
    model: BolmoModel,
    *,
    stage: int,
    recipe: SourceOptimizerRecipe,
) -> list[torch.optim.Optimizer]:
    """Group Bolmo's parameters the way the source trainer grouped its own.

    Embeddings, the output head, the short convolutions and every rank-one
    parameter go to AdamW at their own learning rates; every remaining matrix
    goes to Muon, per head for the trunk's Q/K/V projections.  Stage 1 sees
    only the local modules because the trunk is frozen there, so it has no
    per-head group and no convolution or decay-scalar group.
    """

    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")
    # Resolved in both stages even though only Stage 2 trains the trunk: an
    # unsupported mixer must refuse at startup, not after Stage 1 has already
    # spent its share of the GPU budget and written a checkpoint.
    per_head_projections = (
        global_per_head_projections(model) if recipe.per_head_muon else []
    )
    per_head_ids = (
        {id(parameter) for parameter, _ in per_head_projections}
        if stage == 2
        else set()
    )
    embeddings, output_head, scalars = [], [], []
    delta_conv, delta_decay, muon_local, muon_global = [], [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_global = name.startswith("global_blocks.")
        if stage == 1 and is_global:
            raise ValueError(
                "Stage 1 trains only the local modules; the trunk must be "
                f"frozen, but {name} requires grad"
            )
        if name.endswith(DELTA_CONV_SUFFIXES):
            delta_conv.append(parameter)
        elif name.endswith(DELTA_DECAY_SUFFIXES):
            delta_decay.append(parameter)
        elif "embedding" in name:
            embeddings.append(parameter)
        elif name == "local_decoder.lm_head.weight":
            output_head.append(parameter)
        elif parameter.ndim < 2:
            scalars.append(parameter)
        elif id(parameter) in per_head_ids:
            continue
        elif is_global:
            muon_global.append(parameter)
        else:
            muon_local.append(parameter)

    adam_groups = [
        {"params": embeddings, "lr": recipe.embed_lr, "tag": "embed"},
        {"params": output_head, "lr": recipe.proj_lr, "tag": "proj"},
        {"params": scalars, "lr": recipe.scalar_lr, "tag": "scalar"},
        {"params": delta_conv, "lr": recipe.delta_conv_lr, "tag": "delta_conv"},
        # The source trainer exempts only the decay scalars from Adam decay.
        {
            "params": delta_decay,
            "lr": recipe.scalar_lr,
            "weight_decay": 0.0,
            "tag": "delta_decay",
        },
    ]
    adam_groups = [group for group in adam_groups if group["params"]]
    if not adam_groups:
        raise ValueError("the source recipe found no AdamW parameters")
    optimizers: list[torch.optim.Optimizer] = [
        torch.optim.AdamW(
            adam_groups,
            betas=recipe.adam_betas,
            eps=recipe.adam_eps,
            weight_decay=recipe.adam_weight_decay,
            fused=True,
        )
    ]
    for parameters, tag in ((muon_local, "muon_local"), (muon_global, "muon_global")):
        if not parameters:
            continue
        optimizer = SourceMuon(
            parameters,
            lr=recipe.muon_lr,
            weight_decay=recipe.muon_weight_decay,
            mu=recipe.muon_momentum_warmup_start,
        )
        for group in optimizer.param_groups:
            group["tag"] = tag
        optimizers.append(optimizer)
    if stage == 2 and per_head_projections:
        per_head = PerHeadMuon(
            per_head_projections,
            lr=recipe.muon_lr,
            weight_decay=recipe.muon_weight_decay,
            mu=recipe.muon_momentum_warmup_start,
        )
        for group in per_head.param_groups:
            group["tag"] = "muon_global"
        optimizers.append(per_head)
    return optimizers


class StageOptimizer:
    """One stage's optimizers, plus its schedule, behind one handle.

    The paper recipe is a single AdamW; the source recipe is an AdamW plus two
    or three Muons that must be stepped and scheduled together.  The training
    loop drives both through this object so the two arms differ only in how
    they are constructed.
    """

    def __init__(
        self,
        optimizers: Sequence[torch.optim.Optimizer],
        *,
        planned_steps: int,
        schedule,
        report: Mapping[str, tuple[str, str]],
        momentum_warmup_steps: int = 0,
        momentum_start: float = 0.0,
        momentum_end: float = 0.0,
    ) -> None:
        self.optimizers = list(optimizers)
        self.planned_steps = planned_steps
        self.schedule = schedule
        self.report = dict(report)
        self.momentum_warmup_steps = momentum_warmup_steps
        self.momentum_start = momentum_start
        self.momentum_end = momentum_end
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group["peak_lr"] = group["lr"]
                group.setdefault("tag", "local")

    @property
    def param_groups(self) -> list[dict]:
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def apply_schedule(self, stage_step: int) -> float:
        multiplier = self.schedule(stage_step, self.planned_steps)
        momentum = self.momentum_for(stage_step)
        for group in self.param_groups:
            group["lr"] = group["peak_lr"] * multiplier
            if "mu" in group:
                group["mu"] = momentum
        return multiplier

    def momentum_for(self, stage_step: int) -> float:
        if not self.momentum_warmup_steps:
            return self.momentum_end
        progress = min((stage_step - 1) / self.momentum_warmup_steps, 1.0)
        return self.momentum_start + progress * (
            self.momentum_end - self.momentum_start
        )

    def learning_rate_report(self) -> dict[str, float]:
        groups = self.param_groups
        values: dict[str, float] = {}
        for name, (aggregate, tag) in self.report.items():
            selected = [
                group["lr"] for group in groups if tag == "*" or group["tag"] == tag
            ]
            if not selected:
                values[name] = 0.0
            else:
                values[name] = max(selected) if aggregate == "max" else min(selected)
        if self.momentum_warmup_steps:
            mus = [group["mu"] for group in groups if "mu" in group]
            if mus:
                values["muon_mu"] = float(max(mus))
        return values


PAPER_LR_REPORT = {"lr_local": ("max", "*"), "lr_global": ("min", "*")}
SOURCE_LR_REPORT = {
    "lr_local": ("max", "muon_local"),
    "lr_global": ("max", "muon_global"),
    "lr_embed": ("max", "embed"),
    "lr_proj": ("max", "proj"),
    "lr_scalar": ("max", "scalar"),
}


def build_stage_optimizer(
    model: BolmoModel,
    *,
    stage: int,
    planned_steps: int,
    optimizer_recipe: str,
    source_recipe: SourceOptimizerRecipe,
    stage1_lr: float,
    stage2_local_lr: float,
    stage2_global_lr: float,
) -> StageOptimizer:
    if optimizer_recipe == "paper":
        optimizer = (
            stage1_optimizer(model, stage1_lr)
            if stage == 1
            else stage2_optimizer(
                model,
                local_peak_lr=stage2_local_lr,
                global_peak_lr=stage2_global_lr,
            )
        )
        # PAPER_LR_REPORT aggregates over every group, so the single AdamW
        # keeps reporting exactly the max/min pair earlier runs recorded.
        return StageOptimizer(
            [optimizer],
            planned_steps=planned_steps,
            schedule=linear_warmup_decay,
            report=PAPER_LR_REPORT,
        )
    if optimizer_recipe != "source":
        raise ValueError(
            f"BOLMO_OPTIMIZER_RECIPE must be 'paper' or 'source', got {optimizer_recipe!r}"
        )
    return StageOptimizer(
        source_recipe_optimizers(model, stage=stage, recipe=source_recipe),
        planned_steps=planned_steps,
        schedule=lambda step, planned: cosine_warmup_decay(
            step, planned, source_recipe.warmup_fraction
        ),
        report=SOURCE_LR_REPORT,
        momentum_warmup_steps=source_recipe.muon_momentum_warmup_steps,
        momentum_start=source_recipe.muon_momentum_warmup_start,
        momentum_end=source_recipe.muon_momentum,
    )


def paper_stage_schedule(
    planned_steps: int, stop_after_step: int
) -> tuple[int, int, int, int]:
    """Return Table-8-scaled stage horizons and their chronological prefix."""

    if not 0 < stop_after_step <= planned_steps:
        raise ValueError("STOP_AFTER_STEP must be within the planned schedule")
    planned_stage1 = round(planned_steps / 3)
    planned_stage2 = planned_steps - planned_stage1
    executed_stage1 = min(stop_after_step, planned_stage1)
    executed_stage2 = max(0, stop_after_step - planned_stage1)
    if min(planned_stage1, planned_stage2, executed_stage1) <= 0:
        raise ValueError("the Bolmo plan and its Stage-1 prefix must be positive")
    return planned_stage1, planned_stage2, executed_stage1, executed_stage2


def microbatch_objective(
    loss: BolmoLoss,
    counts: dict[str, int],
    totals: dict[str, int],
    *,
    stage: int,
) -> Tensor:
    objective = (
        4.0
        * loss.boundary
        * counts["boundary_positions"]
        / totals["boundary_positions"]
        + loss.ce * counts["target_bytes"] / totals["target_bytes"]
    )
    if stage == 1:
        assert loss.encoder_stitch is not None and loss.decoder_distill is not None
        objective = (
            objective
            + loss.encoder_stitch
            * counts["stitch_patches"]
            / totals["stitch_patches"]
            + loss.decoder_distill
            * counts["target_patches"]
            / totals["target_patches"]
        )
    return objective


def batch_totals(rows: Sequence[dict[str, Tensor]]) -> dict[str, int]:
    valid_bytes = sum(int(row["valid_mask"].sum()) for row in rows)
    target_bytes = valid_bytes - len(rows)
    boundary_positions = valid_bytes - len(rows)
    patches = sum(int(row["source_valid_mask"].sum()) for row in rows)
    # Match BolmoModel._teacher_patch_alignment literally. In particular,
    # patch zero is retained even when patch one is EOT. Exact counts keep
    # regrouped microbatches equal to the global aligned-patch mean.
    stitch_patches = 0
    for row in rows:
        keep = row["source_valid_mask"].clone()
        keep[:-1] &= ~(
            row["source_valid_mask"][1:]
            & row["source_ids"][1:].eq(SOURCE_EOT_ID)
        )
        keep[0] = True
        stitch_patches += int(keep.sum())
    target_patches = patches - len(rows)
    return {
        "valid_bytes": valid_bytes,
        "target_bytes": target_bytes,
        "boundary_positions": boundary_positions,
        "patches": patches,
        "stitch_patches": stitch_patches,
        "target_patches": target_patches,
    }


def aggregate_metrics(
    accumulator: dict[str, Tensor],
    loss: BolmoLoss,
    counts: dict[str, int],
) -> None:
    weights = {
        "ce": counts["target_bytes"],
        "boundary": counts["boundary_positions"],
        "encoder_stitch": counts["stitch_patches"],
        "encoder_stitch_cosine": counts["stitch_patches"],
        "decoder_distill": counts["target_patches"],
    }
    for name, weight in weights.items():
        value = getattr(loss, name)
        if value is not None:
            key = f"{name}_sum"
            weighted = value.detach().float() * weight
            accumulator[key] = accumulator.get(key, weighted.new_zeros(())) + weighted
            weight_key = f"{name}_weight"
            accumulator[weight_key] = (
                accumulator.get(weight_key, weighted.new_zeros(())) + weight
            )
    for name in (
        "boundary_correct",
        "boundary_true_positive",
        "boundary_false_positive",
        "boundary_false_negative",
        "boundary_positions",
        "valid_bytes",
        "predicted_patches",
    ):
        value = getattr(loss, name)
        if value is not None:
            key = f"{name}_sum"
            detached = value.detach().float()
            accumulator[key] = accumulator.get(key, detached.new_zeros(())) + detached


def finalize_metrics(accumulator: dict[str, Tensor], stage: int) -> dict[str, float]:
    metrics = {}
    for name in (
        "ce",
        "boundary",
        "encoder_stitch",
        "encoder_stitch_cosine",
        "decoder_distill",
    ):
        weight = accumulator.get(f"{name}_weight")
        if weight is not None and int(weight.item()):
            metrics[name] = float(
                (accumulator[f"{name}_sum"] / weight).item()
            )
    correct = accumulator["boundary_correct_sum"]
    true_positive = accumulator["boundary_true_positive_sum"]
    false_positive = accumulator["boundary_false_positive_sum"]
    false_negative = accumulator["boundary_false_negative_sum"]
    boundary_positions = accumulator["boundary_positions_sum"]
    metrics["boundary_accuracy"] = float(
        (correct / boundary_positions.clamp_min(1)).item()
    )
    metrics["boundary_precision"] = float(
        (true_positive / (true_positive + false_positive).clamp_min(1)).item()
    )
    metrics["boundary_recall"] = float(
        (true_positive / (true_positive + false_negative).clamp_min(1)).item()
    )
    metrics["bytes_per_patch"] = float(
        (
            accumulator["valid_bytes_sum"]
            / accumulator["predicted_patches_sum"].clamp_min(1)
        ).item()
    )
    total = metrics["ce"] + 4.0 * metrics["boundary"]
    if stage == 1:
        total += metrics["encoder_stitch"] + metrics["decoder_distill"]
    metrics["total"] = total
    return metrics


@dataclass(frozen=True)
class EvaluationMetrics:
    byte_loss: float
    byte_bpb: float
    joint_loss: float
    joint_bpb: float
    predicted_bytes_per_patch: float
    boundary_accuracy: float
    boundary_precision: float
    boundary_recall: float


@torch.no_grad()
def evaluate(
    model: BolmoModel,
    rows: Sequence[dict[str, Tensor]],
    *,
    microbatch_examples: int,
    byte_pad_id: int,
    source_pad_id: int,
    device: torch.device,
    oracle_boundaries: bool = False,
    fixed_stride: int | None = None,
    uniform_patching: str | None = None,
    causal_routing: bool = False,
) -> EvaluationMetrics:
    model.eval()
    totals: BolmoValidationStatistics | None = None
    sorted_rows = sorted(rows, key=lambda row: int(row["valid_mask"].sum()))
    for row_chunk in chunks(sorted_rows, microbatch_examples):
        batch = collate_rows(
            row_chunk, byte_pad_id=byte_pad_id, source_pad_id=source_pad_id
        ).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            chunk = model.validation_statistics(
                batch,
                oracle_boundaries=oracle_boundaries,
                fixed_stride=fixed_stride,
                uniform_patching=uniform_patching,
                causal_routing=causal_routing,
            )
        if totals is None:
            totals = chunk
        else:
            totals = BolmoValidationStatistics(
                byte_nll=totals.byte_nll + chunk.byte_nll,
                joint_nll=totals.joint_nll + chunk.joint_nll,
                scored_bytes=totals.scored_bytes + chunk.scored_bytes,
                valid_atoms=totals.valid_atoms + chunk.valid_atoms,
                predicted_patches=(
                    totals.predicted_patches + chunk.predicted_patches
                ),
                boundary_correct=totals.boundary_correct + chunk.boundary_correct,
                boundary_positions=(
                    totals.boundary_positions + chunk.boundary_positions
                ),
                boundary_true_positive=(
                    totals.boundary_true_positive
                    + chunk.boundary_true_positive
                ),
                boundary_false_positive=(
                    totals.boundary_false_positive
                    + chunk.boundary_false_positive
                ),
                boundary_false_negative=(
                    totals.boundary_false_negative
                    + chunk.boundary_false_negative
                ),
            )
    if totals is None:
        raise ValueError("validation selection is empty")
    byte_nll = float(totals.byte_nll.item())
    joint_nll = float(totals.joint_nll.item())
    raw_byte_count = int(totals.scored_bytes.item())
    valid_atom_count = int(totals.valid_atoms.item())
    patch_count = int(totals.predicted_patches.item())
    boundary_positions = int(totals.boundary_positions.item())
    true_positive = int(totals.boundary_true_positive.item())
    false_positive = int(totals.boundary_false_positive.item())
    false_negative = int(totals.boundary_false_negative.item())
    if raw_byte_count <= 0:
        raise ValueError("validation selection contains no literal bytes")
    if patch_count <= 0 or boundary_positions <= 0:
        raise ValueError("validation selection contains no patches or boundaries")
    model.train()
    return EvaluationMetrics(
        byte_loss=byte_nll / raw_byte_count,
        byte_bpb=byte_nll / (raw_byte_count * math.log(2.0)),
        joint_loss=joint_nll / raw_byte_count,
        joint_bpb=joint_nll / (raw_byte_count * math.log(2.0)),
        predicted_bytes_per_patch=valid_atom_count / patch_count,
        boundary_accuracy=(
            int(totals.boundary_correct.item()) / boundary_positions
        ),
        boundary_precision=true_positive / max(1, true_positive + false_positive),
        boundary_recall=true_positive / max(1, true_positive + false_negative),
    )


def save_checkpoint(
    path: Path,
    model: BolmoModel,
    *,
    source_checkpoint: Path,
    source_sha256: str,
    data_dir: Path,
    data_manifest: dict,
    training_recipe: str,
    completed_steps: int,
    planned_steps: int,
    stage1_planned_steps: int,
    stage1_completed_steps: int,
    stage2_completed_steps: int,
    stage1_boundary_accuracy: float | None = None,
    stage1_boundary_precision: float | None = None,
    stage1_boundary_recall: float | None = None,
    stage1_oracle_byte_bpb: float | None = None,
    stage1_predicted_oracle_bpb_gap: float | None = None,
    stage1_train_diagnostics: dict[str, float] | None = None,
    stage1_gate_override: bool = False,
    training_contract: dict | None = None,
    training_device_seconds: float | None = None,
    training_wall_seconds: float | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".working")
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.export_config(),
            "architecture": "bolmo_kda_nope_v1",
            "training_recipe": training_recipe,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": source_sha256,
            "data_dir": str(data_dir),
            "data_manifest_sha256": data_manifest["payload_sha256"],
            "completed_steps": completed_steps,
            "planned_steps": planned_steps,
            "stage1_planned_steps": stage1_planned_steps,
            "stage1_completed_steps": stage1_completed_steps,
            "stage2_completed_steps": stage2_completed_steps,
            "stage1_boundary_accuracy": stage1_boundary_accuracy,
            "stage1_boundary_precision": stage1_boundary_precision,
            "stage1_boundary_recall": stage1_boundary_recall,
            "stage1_oracle_byte_bpb": stage1_oracle_byte_bpb,
            "stage1_predicted_oracle_bpb_gap": stage1_predicted_oracle_bpb_gap,
            "stage1_train_diagnostics": stage1_train_diagnostics,
            # Kept outside training_contract, which stays byte-identical across
            # a Stage-2 resume; the override is a per-run operator decision.
            "stage1_gate_override": stage1_gate_override,
            "training_contract": training_contract,
            "rng_state": {
                "python": random.getstate(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
            "training_device_seconds": training_device_seconds,
            "training_wall_seconds": training_wall_seconds,
            "paper": "arXiv:2512.15586v2",
        },
        temporary,
    )
    temporary.replace(path)


def compile_hot_blocks(model: BolmoModel) -> None:
    """Compile fixed block bodies while leaving dynamic pool/depool orchestration eager."""

    mode = os.environ.get("BOLMO_COMPILE_MODE", "default")
    # The source-token axis and microbatch are fixed within a run, so the
    # retained trunk gets the same static specialization as the source
    # trainer. Byte lengths vary by microbatch and stay dynamic locally.
    for block in model.global_blocks:
        block.compile(dynamic=False, fullgraph=False, mode=mode)
    for block in (*model.local_encoder.blocks, *model.local_decoder.blocks):
        block.compile(dynamic=True, fullgraph=False, mode=mode)
    model.local_encoder.boundary_predictor.compile(
        dynamic=True, fullgraph=False, mode=mode
    )
    if os.environ.get("BOLMO_COMPILE_BYTE_HEAD", "1") == "1":
        # Match the source trainer's targeted vocabulary-head compilation
        # without pulling dynamic pooling or opaque recurrences into one graph.
        model.local_decoder.project_logits = torch.compile(  # type: ignore[method-assign]
            model.local_decoder.project_logits,
            dynamic=True,
            fullgraph=True,
            mode=mode,
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Bolmo training requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    run_id = os.environ.get("RUN_ID", "bolmo_kda_nope")
    source_checkpoint = Path(os.environ.get("SOURCE_CHECKPOINT", str(DEFAULT_SOURCE)))
    expected_source_hash = os.environ.get("SOURCE_CHECKPOINT_SHA256", DEFAULT_SOURCE_SHA256)
    source_hash = sha256_file(source_checkpoint)
    if source_hash != expected_source_hash:
        raise ValueError(
            f"source checkpoint hash mismatch: {source_hash} != {expected_source_hash}"
        )
    data_dir = Path(os.environ["BOLMO_DATA_PATH"])
    manifest = load_manifest(data_dir)
    validate_paper_data_manifest(manifest)
    source_meta = manifest["source_tokenizer"]
    atomic_meta = manifest["atomic_vocabulary"]
    num_specials = len(atomic_meta["specials"])
    source_pad_id = int(source_meta["source_pad_id"])
    byte_pad_id = int(atomic_meta["pad_id"])

    planned_steps = int(os.environ.get("ITERATIONS", "2000"))
    stop_after_step = int(os.environ.get("STOP_AFTER_STEP", str(planned_steps)))
    (
        default_planned_stage1,
        default_planned_stage2,
        _,
        _,
    ) = paper_stage_schedule(planned_steps, stop_after_step)
    planned_stage1 = int(
        os.environ.get("BOLMO_STAGE1_PLANNED_STEPS", default_planned_stage1)
    )
    planned_stage2 = planned_steps - planned_stage1
    if (planned_stage1, planned_stage2) != (
        default_planned_stage1,
        default_planned_stage2,
    ):
        raise ValueError(
            "paper-aligned Bolmo does not permit an overridden phase split: "
            f"{planned_stage1}/{planned_stage2} != "
            f"{default_planned_stage1}/{default_planned_stage2}"
        )
    if min(planned_stage1, planned_stage2) <= 0:
        raise ValueError("both planned Bolmo stages must be positive")
    executed_stage1 = min(stop_after_step, planned_stage1)
    executed_stage2 = max(0, stop_after_step - planned_stage1)
    if executed_stage2 > planned_stage2:
        raise ValueError("the executed Stage 2 exceeds its planned LR schedule")
    runtime_end_step = int(
        os.environ.get("BOLMO_END_AFTER_GLOBAL_STEP", str(stop_after_step))
    )
    if not 0 < runtime_end_step <= stop_after_step:
        raise ValueError("BOLMO_END_AFTER_GLOBAL_STEP must be within the executed schedule")

    seed = int(os.environ.get("SEED", "1337"))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    architecture = BolmoArchitecture(
        backend=os.environ.get("BOLMO_BACKEND", "triton"),
        chunk_size=int(os.environ.get("BOLMO_CHUNK_SIZE", "128")),
        autocast_kernel_dtype=os.environ.get(
            "BOLMO_AUTOCAST_KERNEL_DTYPE", "float32"
        ),
        num_special_tokens=num_specials,
    )
    model, teacher, source_payload = BolmoModel.from_source_checkpoint(
        source_checkpoint, architecture=architecture
    )
    validate_checkpoint_tokenizer(source_payload, source_meta)
    if model.local_encoder.source_vocab_size != source_pad_id:
        raise ValueError(
            "dataset source model vocabulary differs from checkpoint: "
            f"{source_pad_id} != {model.local_encoder.source_vocab_size}"
        )
    if byte_pad_id != architecture.byte_pad_id:
        raise ValueError(
            f"dataset atomic padding differs from model: {byte_pad_id} != {architecture.byte_pad_id}"
        )
    resume_path_value = os.environ.get("BOLMO_STAGE2_RESUME_CHECKPOINT")
    resume_stage2 = resume_path_value is not None
    minimum_stage1_boundary_accuracy = float(
        os.environ.get("BOLMO_STAGE1_MIN_BOUNDARY_ACCURACY", "0.99")
    )
    if not 0.99 <= minimum_stage1_boundary_accuracy <= 1.0:
        raise ValueError(
            "BOLMO_STAGE1_MIN_BOUNDARY_ACCURACY must be in [0.99, 1] "
            "for the paper-aligned recipe"
        )
    # The threshold itself stays pinned to the paper. An experimental arm opts
    # out of the refusal explicitly instead, so the measured gate result is
    # still produced, still required, and still preserved in run provenance.
    allow_stage1_gate_failure = (
        os.environ.get("BOLMO_ALLOW_STAGE1_GATE_FAILURE", "0") == "1"
    )
    optimizer_recipe = os.environ.get("BOLMO_OPTIMIZER_RECIPE", "paper")
    if optimizer_recipe not in TRAINING_RECIPES:
        raise ValueError(
            "BOLMO_OPTIMIZER_RECIPE must be one of "
            f"{sorted(TRAINING_RECIPES)}, got {optimizer_recipe!r}"
        )
    training_recipe = TRAINING_RECIPES[optimizer_recipe]
    schedule_scope = os.environ.get("BOLMO_SOURCE_SCHEDULE_SCOPE", "per_stage")
    source_recipe = SourceOptimizerRecipe(
        schedule_scope=schedule_scope, momentum_warmup_scope=schedule_scope
    )
    # One horizon over the whole plan, so Stage 2 continues Stage 1's decay
    # instead of restarting it. Only the source recipe has a single set of
    # learning rates to run a single horizon over; the paper's are per-stage by
    # construction (Table 8 gives Stage 1 and Stage 2 different peaks), so
    # asking for a continuous schedule there is a configuration error.
    continuous_schedule = source_recipe.continuous_schedule
    if continuous_schedule and optimizer_recipe != "source":
        raise ValueError(
            "BOLMO_SOURCE_SCHEDULE_SCOPE=continuous requires "
            f"BOLMO_OPTIMIZER_RECIPE=source, got {optimizer_recipe!r}"
        )
    if resume_stage2 and runtime_end_step <= planned_stage1:
        raise ValueError("Stage-2 resume must execute at least one Stage-2 update")
    resume_payload: dict | None = None
    if resume_path_value is not None:
        resume_path = Path(resume_path_value)
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_stage1_resume_checkpoint(
            resume_payload,
            model=model,
            source_sha256=source_hash,
            data_manifest_sha256=manifest["payload_sha256"],
            training_recipe=training_recipe,
            planned_steps=planned_steps,
            planned_stage1=planned_stage1,
            minimum_boundary_accuracy=minimum_stage1_boundary_accuracy,
            allow_gate_failure=allow_stage1_gate_failure,
        )
        model.load_state_dict(resume_payload["model"], strict=True)
    model.to(device)
    if resume_stage2:
        del teacher
        model.freeze_global(False)
    else:
        teacher.to(device)
        model.freeze_global(True)

    train_stream = ExampleStream(
        artifact_paths(data_dir, manifest, "train"),
        seed=seed,
        repeat=False,
        shuffle=True,
        source_vocab_size=int(source_meta["logical_vocab_size"]),
        source_pad_id=source_pad_id,
        atomic_vocab_size=architecture.atomic_vocab_size,
        context_source_tokens=0,
    )
    source_tokens_per_example = int(
        manifest["examples"]["source_sequence_length"]
    )
    expected_source_context = int(
        os.environ.get("BOLMO_SOURCE_CONTEXT", str(DEFAULT_SOURCE_CONTEXT))
    )
    if source_tokens_per_example != expected_source_context:
        raise ValueError(
            "Bolmo data must match the pinned source checkpoint context: "
            f"{source_tokens_per_example} != {expected_source_context}"
        )
    expected_atomic_cap = 6 * expected_source_context
    if manifest["examples"].get("max_atomic_tokens") != expected_atomic_cap:
        raise ValueError(
            "Bolmo data must preserve Table 8's 6x byte cap: "
            f"{manifest['examples'].get('max_atomic_tokens')} != "
            f"{expected_atomic_cap}"
        )
    canonical_val_source_tokens = int(
        os.environ.get("BOLMO_CANONICAL_VAL_SOURCE_TOKENS", str(2**21))
    )
    if canonical_val_source_tokens % source_tokens_per_example:
        raise ValueError(
            "canonical validation tokens must be divisible by source tokens per example"
        )
    canonical_val_examples = (
        canonical_val_source_tokens // source_tokens_per_example
    )
    proxy_val_examples = int(os.environ.get("BOLMO_PROXY_VAL_EXAMPLES", "256"))
    if not 0 < proxy_val_examples <= canonical_val_examples:
        raise ValueError(
            "BOLMO_PROXY_VAL_EXAMPLES must be within the canonical validation span"
        )
    val_stream = ExampleStream(
        artifact_paths(data_dir, manifest, "validation"),
        seed=0,
        repeat=False,
        shuffle=False,
        source_vocab_size=int(source_meta["logical_vocab_size"]),
        source_pad_id=source_pad_id,
        atomic_vocab_size=architecture.atomic_vocab_size,
        context_source_tokens=1,
    )
    canonical_val_rows = val_stream.take(canonical_val_examples)
    if len(canonical_val_rows) != canonical_val_examples:
        raise ValueError(
            "validation split is too short for the source-matched canonical span: "
            f"{len(canonical_val_rows)} != {canonical_val_examples} examples"
        )
    proxy_val_rows = canonical_val_rows[:proxy_val_examples]
    canonical_scored_source_tokens = sum(
        scored_source_token_count(
            row["source_valid_mask"],
            row["patch_lens"],
            row["valid_mask"],
            row["score_mask"],
            context_source_tokens=1,
        )
        for row in canonical_val_rows
    )
    if canonical_scored_source_tokens != canonical_val_source_tokens:
        raise ValueError(
            "canonical validation byte truncation changed the source-matched "
            f"score span: {canonical_scored_source_tokens} != "
            f"{canonical_val_source_tokens}"
        )
    proxy_scored_source_tokens = sum(
        scored_source_token_count(
            row["source_valid_mask"],
            row["patch_lens"],
            row["valid_mask"],
            row["score_mask"],
            context_source_tokens=1,
        )
        for row in proxy_val_rows
    )

    common_microbatch = os.environ.get("BOLMO_MICROBATCH_EXAMPLES")
    stage1_microbatch_examples = int(
        os.environ.get(
            "BOLMO_STAGE1_MICROBATCH_EXAMPLES", common_microbatch or "8"
        )
    )
    stage2_microbatch_examples = int(
        os.environ.get(
            "BOLMO_STAGE2_MICROBATCH_EXAMPLES", common_microbatch or "8"
        )
    )
    if 131_072 % source_tokens_per_example:
        raise ValueError(
            "source example width must divide Table 8's Stage-1 token batch"
        )
    default_stage1_global_examples = 131_072 // source_tokens_per_example
    stage1_global_examples = int(
        os.environ.get(
            "BOLMO_STAGE1_GLOBAL_EXAMPLES", str(default_stage1_global_examples)
        )
    )
    stage2_global_examples = int(
        os.environ.get(
            "BOLMO_STAGE2_GLOBAL_EXAMPLES",
            str(2 * default_stage1_global_examples),
        )
    )
    teacher_positions_per_chunk = int(
        os.environ.get(
            "BOLMO_TEACHER_POSITIONS_PER_CHUNK",
            str(DEFAULT_TEACHER_POSITIONS_PER_CHUNK),
        )
    )
    val_every = int(os.environ.get("VAL_LOSS_EVERY", "20"))
    log_every = int(os.environ.get("TRAIN_LOG_EVERY", "10"))
    # Table 8 clips at 0.5. The source run does not clip at all: Muon already
    # normalizes its updates, and a 0.5 ceiling would rescale the 0.7 embedding
    # group on essentially every step. Either way the norm is still measured
    # and logged, so the two arms stay comparable on gradient telemetry.
    max_grad_norm = float(
        os.environ.get(
            "BOLMO_MAX_GRAD_NORM", "0.5" if optimizer_recipe == "paper" else "inf"
        )
    )
    if min(
        stage1_microbatch_examples,
        stage2_microbatch_examples,
        stage1_global_examples,
        stage2_global_examples,
        teacher_positions_per_chunk,
    ) <= 0:
        raise ValueError("batch sizes must be positive")
    if stage1_global_examples % stage1_microbatch_examples:
        raise ValueError(
            "Stage-1 global examples must be divisible by its microbatch examples"
        )
    if stage2_global_examples % stage2_microbatch_examples:
        raise ValueError(
            "Stage-2 global examples must be divisible by its microbatch examples"
        )
    required_train_examples = (
        executed_stage1 * stage1_global_examples
        + executed_stage2 * stage2_global_examples
    )
    available_train_examples = int(manifest["splits"]["train"]["examples"])
    if available_train_examples < required_train_examples:
        raise ValueError(
            "Bolmo run would repeat its training data, unlike the paper: "
            f"{available_train_examples} < {required_train_examples} examples"
        )

    print(f"Bolmo source: {source_checkpoint} sha256:{source_hash}", flush=True)
    print(
        f"Bolmo schedule: stage1 {executed_stage1}/{planned_stage1}, "
        f"stage2 {executed_stage2}/{planned_stage2}, total {stop_after_step}/{planned_steps} "
        f"runtime_end={runtime_end_step} "
        f"stage1_boundary_gate={minimum_stage1_boundary_accuracy:.6f} "
        f"stage1_gate_override={int(allow_stage1_gate_failure)} "
        f"optimizer_recipe={optimizer_recipe} training_recipe={training_recipe}",
        flush=True,
    )
    if optimizer_recipe == "source":
        print(f"Bolmo source optimizer recipe: {source_recipe}", flush=True)
    print(f"Bolmo parameters: {json.dumps(parameter_report(model), sort_keys=True)}", flush=True)
    print(
        f"Bolmo batches: source_sequence_length={source_tokens_per_example} "
        f"stage1_examples={stage1_global_examples} stage2_examples={stage2_global_examples} "
        f"stage1_microbatch_examples={stage1_microbatch_examples} "
        f"stage2_microbatch_examples={stage2_microbatch_examples} "
        f"teacher_positions_per_chunk={teacher_positions_per_chunk} "
        f"proxy_val_examples={proxy_val_examples} "
        f"canonical_val_examples={canonical_val_examples}",
        flush=True,
    )
    if resume_stage2:
        skipped = train_stream.skip(planned_stage1 * stage1_global_examples)
        if skipped != planned_stage1 * stage1_global_examples:
            raise RuntimeError("training data is too short to reconstruct Stage-2 cursor")
        print(
            f"Bolmo Stage-2 resume: {resume_path_value} "
            f"start_step={planned_stage1 + 1} skipped_stage1_examples={skipped}",
            flush=True,
        )

    # Calibrate once on real source-aligned data, then train that same data as
    # the first Stage-1 batch so calibration does not consume extra examples.
    pending_rows: list[dict[str, Tensor]] | None = None
    if not resume_stage2:
        pending_rows = train_stream.take_exact(stage1_global_examples)
        calibration_batch = collate_rows(
            pending_rows[:stage1_microbatch_examples],
            byte_pad_id=byte_pad_id,
            source_pad_id=source_pad_id,
        ).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.calibrate_encoder_output(calibration_batch, teacher)  # type: ignore[name-defined]
        del calibration_batch
    compile_enabled = os.environ.get("BOLMO_COMPILE", "1") == "1"
    if compile_enabled:
        compile_hot_blocks(model)

    stage1_lr = float(os.environ.get("BOLMO_STAGE1_LR", "7e-4"))
    stage2_local_lr = float(os.environ.get("BOLMO_STAGE2_LOCAL_LR", "5.2e-5"))
    stage2_global_lr = float(os.environ.get("BOLMO_STAGE2_GLOBAL_LR", "2.6e-5"))
    # ``recipe`` already names the optimizer arm one-for-one, so the contract
    # stays byte-identical to what earlier paper runs wrote and their Stage-1
    # checkpoints remain resumable.
    training_contract = {
        "recipe": training_recipe,
        "data_shards_verified": True,
        "seed": seed,
        "planned_steps": planned_steps,
        "planned_stage1_steps": planned_stage1,
        "planned_stage2_steps": planned_stage2,
        "source_tokens_per_example": source_tokens_per_example,
        "max_atomic_tokens": expected_atomic_cap,
        "stage1_global_examples": stage1_global_examples,
        "stage2_global_examples": stage2_global_examples,
        "stage1_microbatch_examples": stage1_microbatch_examples,
        "stage2_microbatch_examples": stage2_microbatch_examples,
        "stage1_consumed_examples": planned_stage1 * stage1_global_examples,
        "canonical_val_source_tokens": canonical_val_source_tokens,
        "max_grad_norm": max_grad_norm,
        "stage1_min_boundary_accuracy": minimum_stage1_boundary_accuracy,
    }
    if optimizer_recipe == "paper":
        training_contract |= {
            "stage1_peak_lr": stage1_lr,
            "stage2_local_peak_lr": stage2_local_lr,
            "stage2_global_peak_lr": stage2_global_lr,
            "adam_betas": [0.9, 0.95],
            "adam_eps": 1e-8,
            "weight_decay": 0.1,
        }
    else:
        training_contract["source_optimizer"] = source_recipe.as_contract()
    if resume_payload is not None:
        validate_training_contract(resume_payload, training_contract)
    optimizer = build_stage_optimizer(
        model,
        stage=2 if resume_stage2 else 1,
        planned_steps=(
            planned_steps
            if continuous_schedule
            else (planned_stage2 if resume_stage2 else planned_stage1)
        ),
        optimizer_recipe=optimizer_recipe,
        source_recipe=source_recipe,
        stage1_lr=stage1_lr,
        stage2_local_lr=stage2_local_lr,
        stage2_global_lr=stage2_global_lr,
    )
    if resume_payload is not None:
        rng_state = resume_payload["rng_state"]
        random.setstate(rng_state["python"])
        torch.set_rng_state(rng_state["torch_cpu"])
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    resume_device_seconds = float(
        os.environ.get(
            "BOLMO_RESUME_TRAINING_DEVICE_SECONDS",
            str((resume_payload or {}).get("training_device_seconds") or 0.0),
        )
    )
    resume_wall_seconds = float(
        os.environ.get(
            "BOLMO_RESUME_TRAINING_WALL_SECONDS",
            str((resume_payload or {}).get("training_wall_seconds") or 0.0),
        )
    )
    training_time = resume_device_seconds
    training_wall_time = resume_wall_seconds
    stage_training_time = {1: 0.0, 2: 0.0}
    stage_wall_time = {1: 0.0, 2: 0.0}
    stage_source_tokens = {1: 0, 2: 0}
    stage_atomic_tokens = {1: 0, 2: 0}
    reported_stage_time = {1: 0.0, 2: 0.0}
    reported_stage_wall_time = {1: 0.0, 2: 0.0}
    reported_stage_source_tokens = {1: 0, 2: 0}
    reported_stage_atomic_tokens = {1: 0, 2: 0}
    pending_timings: list[
        tuple[torch.cuda.Event, torch.cuda.Event, int, int, int]
    ] = []
    interval_wall_start: float | None = None

    def flush_training_timing() -> None:
        nonlocal training_time, training_wall_time, interval_wall_start
        if not pending_timings:
            return
        pending_timings[-1][1].synchronize()
        for start, end, timing_stage, source_tokens, atomic_tokens in pending_timings:
            elapsed = start.elapsed_time(end) / 1000.0
            training_time += elapsed
            stage_training_time[timing_stage] += elapsed
            stage_source_tokens[timing_stage] += source_tokens
            stage_atomic_tokens[timing_stage] += atomic_tokens
        timing_stages = {item[2] for item in pending_timings}
        # Stage boundaries are mandatory log/flush points, so an interval can
        # never mix the frozen and end-to-end stages.
        if len(timing_stages) != 1:
            raise AssertionError("a timing interval crossed a Bolmo stage")
        timing_stage = next(iter(timing_stages))
        pending_timings.clear()
        assert interval_wall_start is not None
        wall_elapsed = time.perf_counter() - interval_wall_start
        training_wall_time += wall_elapsed
        stage_wall_time[timing_stage] += wall_elapsed
        interval_wall_start = None

    def run_validation(
        step: int,
        *,
        canonical: bool = False,
        stage1_gate: bool = False,
    ) -> tuple[
        EvaluationMetrics,
        EvaluationMetrics | None,
        EvaluationMetrics | None,
    ]:
        eval_started = time.perf_counter()
        proxy_validation = evaluate(
            model,
            proxy_val_rows,
            microbatch_examples=stage2_microbatch_examples,
            byte_pad_id=byte_pad_id,
            source_pad_id=source_pad_id,
            device=device,
        )
        canonical_validation = (
            evaluate(
                model,
                canonical_val_rows,
                microbatch_examples=stage2_microbatch_examples,
                byte_pad_id=byte_pad_id,
                source_pad_id=source_pad_id,
                device=device,
            )
            if canonical
            else None
        )
        oracle_validation = (
            evaluate(
                model,
                canonical_val_rows,
                microbatch_examples=stage2_microbatch_examples,
                byte_pad_id=byte_pad_id,
                source_pad_id=source_pad_id,
                device=device,
                oracle_boundaries=True,
            )
            if stage1_gate
            else None
        )
        # The boundary predictor is non-causal, so ``val_canonical_bpb`` scores
        # each byte under a routing decision derived from that byte and is not
        # a codelength. This one is, but it is a loose upper bound rather than
        # the comparand: the model is trained with non-causal routing, so
        # forcing every position onto a stale patch charges a train/eval
        # mismatch far larger than the leak (about 2.2 bpb on both measured
        # arms). ``val_canonical_joint_bpb`` is the tight valid codelength and
        # is what a subword model's bits-per-byte must be compared against.
        # Canonical checkpoints only, so the extra pass costs a few seconds
        # twice per run.
        causal_validation = (
            evaluate(
                model,
                canonical_val_rows,
                microbatch_examples=stage2_microbatch_examples,
                byte_pad_id=byte_pad_id,
                source_pad_id=source_pad_id,
                device=device,
                causal_routing=True,
            )
            if canonical
            else None
        )
        eval_seconds = time.perf_counter() - eval_started
        gate_extras = ""
        if canonical_validation is not None:
            gate_extras += (
                f"val_canonical_loss:{canonical_validation.byte_loss:.6f} "
                f"val_canonical_bpb:{canonical_validation.byte_bpb:.6f} "
                f"val_canonical_joint_bpb:{canonical_validation.joint_bpb:.6f} "
                f"val_canonical_boundary_accuracy:"
                f"{canonical_validation.boundary_accuracy:.6f} "
                f"val_canonical_boundary_precision:"
                f"{canonical_validation.boundary_precision:.6f} "
                f"val_canonical_boundary_recall:"
                f"{canonical_validation.boundary_recall:.6f} "
                f"val_canonical_predicted_bytes_per_patch:"
                f"{canonical_validation.predicted_bytes_per_patch:.4f} "
                f"val_canonical_source_tokens:{canonical_val_source_tokens} "
            )
        if causal_validation is not None:
            assert canonical_validation is not None
            gate_extras += (
                f"val_canonical_causal_bpb:{causal_validation.byte_bpb:.6f} "
                f"val_canonical_noncausal_routing_credit_bpb:"
                f"{causal_validation.byte_bpb - canonical_validation.byte_bpb:.6f} "
            )
        if oracle_validation is not None:
            assert canonical_validation is not None
            gate_extras = (
                f"{gate_extras}"
                f"val_canonical_oracle_bpb:{oracle_validation.byte_bpb:.6f} "
                f"val_canonical_predicted_oracle_bpb_gap:"
                f"{canonical_validation.byte_bpb - oracle_validation.byte_bpb:.6f} "
            )
        print(
            f"{step}/{planned_steps} val_loss: {proxy_validation.byte_loss:.6f} "
            f"val_bpb: {proxy_validation.byte_bpb:.6f} "
            f"val_joint_loss:{proxy_validation.joint_loss:.6f} "
            f"val_joint_bpb:{proxy_validation.joint_bpb:.6f} "
            f"val_boundary_accuracy:{proxy_validation.boundary_accuracy:.6f} "
            f"val_boundary_precision:{proxy_validation.boundary_precision:.6f} "
            f"val_boundary_recall:{proxy_validation.boundary_recall:.6f} "
            f"val_proxy_source_tokens:"
            f"{proxy_scored_source_tokens} "
            f"{gate_extras}"
            f"train_time:{training_wall_time * 1000:.0f}ms "
            f"train_wall_time_ms:{training_wall_time * 1000:.0f} "
            f"train_device_time_ms:{training_time * 1000:.0f} "
            f"eval_seconds:{eval_seconds:.3f} "
            f"predicted_bytes_per_patch:"
            f"{proxy_validation.predicted_bytes_per_patch:.4f}",
            flush=True,
        )
        return proxy_validation, canonical_validation, oracle_validation

    stage1_boundary_accuracy = (
        float(resume_payload["stage1_boundary_accuracy"])
        if resume_payload is not None
        else None
    )
    stage1_boundary_precision = (
        float(resume_payload["stage1_boundary_precision"])
        if resume_payload is not None
        else None
    )
    stage1_boundary_recall = (
        float(resume_payload["stage1_boundary_recall"])
        if resume_payload is not None
        else None
    )
    stage1_oracle_byte_bpb = (
        float(resume_payload["stage1_oracle_byte_bpb"])
        if resume_payload is not None
        else None
    )
    stage1_predicted_oracle_bpb_gap = (
        float(resume_payload["stage1_predicted_oracle_bpb_gap"])
        if resume_payload is not None
        else None
    )
    stage1_train_diagnostics = (
        dict(resume_payload["stage1_train_diagnostics"])
        if resume_payload is not None
        else None
    )
    if not resume_stage2:
        run_validation(0, canonical=True)
    first_global_step = planned_stage1 + 1 if resume_stage2 else 1
    for global_step in range(first_global_step, runtime_end_step + 1):
        stage = 1 if global_step <= planned_stage1 else 2
        if not resume_stage2 and global_step == planned_stage1 + 1:
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
            model.freeze_global(False)
            optimizer = build_stage_optimizer(
                model,
                stage=2,
                planned_steps=(
                    planned_steps if continuous_schedule else planned_stage2
                ),
                optimizer_recipe=optimizer_recipe,
                source_recipe=source_recipe,
                stage1_lr=stage1_lr,
                stage2_local_lr=stage2_local_lr,
                stage2_global_lr=stage2_global_lr,
            )
            # Stage 2 has no teacher and the output/input source tables are
            # intentionally not part of the inference model.
            del teacher
            gc.collect()
            if compile_enabled:
                # Stage 1 specializes the shared blocks under a frozen trunk,
                # teacher no-grad calls, and a different byte-length stream.
                # Clear those phase-specific guards before Stage 2 can hit
                # Dynamo's recompile limit and permanently fall back to eager.
                torch.compiler.reset()
            torch.cuda.empty_cache()

        if interval_wall_start is None:
            interval_wall_start = time.perf_counter()
        if stage == 1:
            stage_step = global_step
            if global_step == 1:
                assert pending_rows is not None
                rows = pending_rows
            else:
                rows = train_stream.take_exact(stage1_global_examples)
        else:
            stage_step = (
                global_step if continuous_schedule else global_step - planned_stage1
            )
            rows = train_stream.take_exact(stage2_global_examples)
        # Group similarly sized byte sequences inside the already sampled
        # global batch. This preserves the examples and exact weighted
        # objective while avoiding compute on the longest row's padding in
        # every microbatch.
        rows.sort(key=lambda row: int(row["valid_mask"].sum()))
        optimizer.apply_schedule(stage_step)
        totals = batch_totals(rows)
        optimizer.zero_grad(set_to_none=True)
        accumulator: dict[str, Tensor] = {}
        timing_start = torch.cuda.Event(enable_timing=True)
        timing_end = torch.cuda.Event(enable_timing=True)
        timing_start.record()
        if stage == 1:
            microbatches = chunks(rows, stage1_microbatch_examples)
        else:
            microbatches = chunks(rows, stage2_microbatch_examples)
        for row_chunk in microbatches:
            counts = batch_totals(row_chunk)
            batch = collate_rows(
                row_chunk,
                byte_pad_id=byte_pad_id,
                source_pad_id=source_pad_id,
            ).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if stage == 1:
                    loss = model.stage1(  # type: ignore[name-defined]
                        batch,
                        teacher,
                        teacher_positions_per_chunk=teacher_positions_per_chunk,
                    )
                else:
                    loss = model.stage2(batch)
                objective = microbatch_objective(loss, counts, totals, stage=stage)
            objective.backward()
            aggregate_metrics(accumulator, loss, counts)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        timing_end.record()
        pending_timings.append(
            (
                timing_start,
                timing_end,
                stage,
                totals["target_patches"],
                totals["target_bytes"],
            )
        )
        should_log = global_step % log_every == 0 or global_step in {
            1,
            planned_stage1,
            planned_stage1 + 1,
        }
        should_validate = global_step % val_every == 0 or global_step in {
            planned_stage1,
            runtime_end_step,
        }
        if should_log or should_validate:
            flush_training_timing()
        if should_log:
            metrics = finalize_metrics(accumulator, stage)
            if stage == 1:
                stage1_train_diagnostics = metrics
            extras = " ".join(
                f"bolmo_{name}:{value:.6f}" for name, value in metrics.items() if name != "total"
            )
            learning_rates = " ".join(
                f"{name}:{value:.8f}"
                for name, value in optimizer.learning_rate_report().items()
            )
            stage_seconds = stage_training_time[stage]
            stage_wall_seconds = stage_wall_time[stage]
            interval_seconds = stage_seconds - reported_stage_time[stage]
            interval_wall_seconds = (
                stage_wall_seconds - reported_stage_wall_time[stage]
            )
            interval_source_tokens = (
                stage_source_tokens[stage] - reported_stage_source_tokens[stage]
            )
            interval_atomic_tokens = (
                stage_atomic_tokens[stage] - reported_stage_atomic_tokens[stage]
            )
            print(
                f"step:{global_step}/{planned_steps} train_loss:{metrics['total']:.6f} "
                f"stage_id:{stage} {learning_rates} "
                f"grad_norm:{float(grad_norm):.6f} {extras} "
                f"train_time:{training_wall_time * 1000:.0f}ms "
                f"train_wall_time_ms:{training_wall_time * 1000:.0f} "
                f"train_device_time_ms:{training_time * 1000:.0f} "
                f"step_avg:{training_wall_time * 1000 / global_step:.2f}ms "
                f"stage_source_tokens_per_second:"
                f"{interval_source_tokens / interval_wall_seconds:.2f} "
                f"stage_atomic_tokens_per_second:"
                f"{interval_atomic_tokens / interval_wall_seconds:.2f} "
                f"stage_source_tokens_per_device_second:"
                f"{interval_source_tokens / interval_seconds:.2f} "
                f"stage_atomic_tokens_per_device_second:"
                f"{interval_atomic_tokens / interval_seconds:.2f} "
                f"stage_cumulative_source_tokens_per_second:"
                f"{stage_source_tokens[stage] / stage_wall_seconds:.2f} "
                f"stage_cumulative_atomic_tokens_per_second:"
                f"{stage_atomic_tokens[stage] / stage_wall_seconds:.2f} "
                f"peak_vram_allocated_mib:"
                f"{torch.cuda.max_memory_allocated() / 2**20:.0f} "
                f"peak_vram_reserved_mib:"
                f"{torch.cuda.max_memory_reserved() / 2**20:.0f}",
                flush=True,
            )
            reported_stage_time[stage] = stage_seconds
            reported_stage_wall_time[stage] = stage_wall_seconds
            reported_stage_source_tokens[stage] = stage_source_tokens[stage]
            reported_stage_atomic_tokens[stage] = stage_atomic_tokens[stage]
        if should_validate:
            proxy_validation, canonical_validation, oracle_validation = run_validation(
                global_step,
                canonical=(
                    global_step == planned_stage1
                    or global_step == runtime_end_step
                ),
                stage1_gate=global_step == planned_stage1,
            )
        else:
            proxy_validation, canonical_validation, oracle_validation = None, None, None
        if global_step == planned_stage1:
            assert proxy_validation is not None
            assert canonical_validation is not None
            assert oracle_validation is not None
            assert stage1_train_diagnostics is not None
            stage1_boundary_accuracy = canonical_validation.boundary_accuracy
            stage1_boundary_precision = canonical_validation.boundary_precision
            stage1_boundary_recall = canonical_validation.boundary_recall
            stage1_oracle_byte_bpb = oracle_validation.byte_bpb
            stage1_predicted_oracle_bpb_gap = (
                canonical_validation.byte_bpb - oracle_validation.byte_bpb
            )
            stage1_path = ROOT / "logs" / f"{run_id}_stage1_model.pt"
            save_checkpoint(
                stage1_path,
                model,
                source_checkpoint=source_checkpoint,
                source_sha256=source_hash,
                data_dir=data_dir,
                data_manifest=manifest,
                training_recipe=training_recipe,
                completed_steps=planned_stage1,
                planned_steps=planned_steps,
                stage1_planned_steps=planned_stage1,
                stage1_completed_steps=planned_stage1,
                stage2_completed_steps=0,
                stage1_boundary_accuracy=stage1_boundary_accuracy,
                stage1_boundary_precision=stage1_boundary_precision,
                stage1_boundary_recall=stage1_boundary_recall,
                stage1_oracle_byte_bpb=stage1_oracle_byte_bpb,
                stage1_predicted_oracle_bpb_gap=(
                    stage1_predicted_oracle_bpb_gap
                ),
                stage1_train_diagnostics=stage1_train_diagnostics,
                stage1_gate_override=allow_stage1_gate_failure,
                training_contract=training_contract,
                training_device_seconds=training_time,
                training_wall_seconds=training_wall_time,
            )
            print(f"saved Stage-1 checkpoint: {stage1_path}", flush=True)
            if stage1_boundary_accuracy < minimum_stage1_boundary_accuracy:
                if not allow_stage1_gate_failure:
                    raise RuntimeError(
                        "Stage 1 failed the paper boundary-emulation gate; refusing "
                        "to start Stage 2: "
                        f"{stage1_boundary_accuracy:.6f} < "
                        f"{minimum_stage1_boundary_accuracy:.6f}. Increase the "
                        "complete Stage-1 horizon and retrain from the source checkpoint."
                    )
                print(
                    "WARNING: Stage 1 failed the paper boundary-emulation gate "
                    f"({stage1_boundary_accuracy:.6f} < "
                    f"{minimum_stage1_boundary_accuracy:.6f}); continuing into "
                    "Stage 2 as an explicitly experimental arm.",
                    flush=True,
                )

    final_path = ROOT / "logs" / f"{run_id}_final_model.pt"
    save_checkpoint(
        final_path,
        model,
        source_checkpoint=source_checkpoint,
        source_sha256=source_hash,
        data_dir=data_dir,
        data_manifest=manifest,
        training_recipe=training_recipe,
        completed_steps=runtime_end_step,
        planned_steps=planned_steps,
        stage1_planned_steps=planned_stage1,
        stage1_completed_steps=min(runtime_end_step, planned_stage1),
        stage2_completed_steps=max(0, runtime_end_step - planned_stage1),
        stage1_boundary_accuracy=stage1_boundary_accuracy,
        stage1_boundary_precision=stage1_boundary_precision,
        stage1_boundary_recall=stage1_boundary_recall,
        stage1_oracle_byte_bpb=stage1_oracle_byte_bpb,
        stage1_predicted_oracle_bpb_gap=stage1_predicted_oracle_bpb_gap,
        stage1_train_diagnostics=stage1_train_diagnostics,
        stage1_gate_override=allow_stage1_gate_failure,
        training_contract=training_contract,
        training_device_seconds=training_time,
        training_wall_seconds=training_wall_time,
    )
    print(f"saved checkpoint: {final_path}", flush=True)


if __name__ == "__main__":
    main()
