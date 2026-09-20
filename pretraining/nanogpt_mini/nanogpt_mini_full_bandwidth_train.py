"""CUDA/BF16 full-bandwidth nanoGPT-mini pretraining (no import-time work).

The paper specifies aggregate pass frequencies, not exact stage boundaries. Our
local progressive interpretation uses 50% single-pass warmup, a 50/50 one/two
pass stage until 88%, then 50/25/25 one/two/three passes. Each stage is shuffled
independently with the run seed. At 2,000 steps this is exactly 1,500/440/60.
The single schedule is a matched token/step control, not a matched-FLOP control.

Canonical val_loss/val_bpb always score pass one; p2/p3 (and final p8) are
explicit diagnostics, including before the feedback channel has been trained.
Training-time accounting excludes validation and checkpoint writes but includes
compilation. token_equivalent_pass_compute counts global tokens times passes,
not measured FLOPs. Run through mlq; use scripts/ablation.py for metrics/TB.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import struct
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class PassSchedule:
    two_pass_start: float = 0.5
    three_pass_start: float = 0.88

    def __post_init__(self) -> None:
        if not 0 <= self.two_pass_start < self.three_pass_start <= 1:
            raise ValueError("pass stage boundaries must satisfy 0 <= two < three <= 1")

    @staticmethod
    @lru_cache(maxsize=16)
    def _sequence(
        total_steps: int, seed: int, two_pass_start: float, three_pass_start: float
    ) -> tuple[int, ...]:
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        two_start = int(total_steps * two_pass_start)
        three_start = int(total_steps * three_pass_start)
        middle_size = three_start - two_start
        final_size = total_steps - three_start
        middle_twos = middle_size // 2
        final_quarter = final_size // 4
        middle = [1] * (middle_size - middle_twos) + [2] * middle_twos
        final = (
            [1] * (final_size - 2 * final_quarter)
            + [2] * final_quarter
            + [3] * final_quarter
        )
        random.Random(f"fbt:{seed}:middle").shuffle(middle)
        random.Random(f"fbt:{seed}:final").shuffle(final)
        # In incomplete stages, leftover slots are single-pass. No rank-local
        # RNG or microbatch counter participates in this optimizer-step schedule.
        return tuple([1] * two_start + middle + final)

    def passes_at(self, step: int, total_steps: int, seed: int) -> int:
        if not 0 <= step < total_steps:
            raise ValueError("step must satisfy 0 <= step < total_steps")
        return self._sequence(
            total_steps, seed, self.two_pass_start, self.three_pass_start
        )[step]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compiled CUDA/BF16 nanoGPT-mini Full-bandwidth Transformer pretraining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Queue GPU runs with mlq. Environment variables provide defaults; CLI flags win. "
            "WARMDOWN_ITERS is intentionally ignored: this trainer uses WSD with a final "
            "25% cooldown. Canonical validation is always pass1; p2/p3/p8 are diagnostics."
        ),
    )

    def option(flag, env, default, type_=str, **kwargs):
        parser.add_argument(
            flag,
            type=type_,
            default=os.environ.get(env, str(default)),
            help=f"{kwargs.pop('help', '')} [{env}]".strip(),
            **kwargs,
        )

    option(
        "--schedule", "FBT_SCHEDULE", "progressive", choices=("progressive", "single")
    )
    option("--iterations", "ITERATIONS", 2000, int)
    option("--val-loss-every", "VAL_LOSS_EVERY", 20, int)
    option("--train-log-every", "TRAIN_LOG_EVERY", 10, int)
    option("--seed", "SEED", 1337, int)
    option(
        "--run-id", "RUN_ID", "", help="Output directory name; empty generates a UUID"
    )
    option("--data-path", "DATA_PATH", "data/datasets/fineweb_onepass_sp1024")
    option(
        "--tokenizer-path", "TOKENIZER_PATH", "data/tokenizers/fineweb_1024_bpe.model"
    )
    option("--vocab-size", "VOCAB_SIZE", 1024, int)
    option("--num-layers", "NUM_LAYERS", 6, int)
    option("--model-dim", "MODEL_DIM", 512, int)
    option(
        "--kv-heads",
        "FBT_KV_HEADS",
        0,
        int,
        help="Key/value heads; zero uses all query heads, otherwise must divide MODEL_DIM / 128",
    )
    option("--mlp-hidden", "MLP_HDIM", 0, int, help="Zero means four times model width")
    option("--noise", "FBT_NOISE", 0.02, float)
    option("--fusion", "FBT_FUSION", "glu", choices=("glu", "layerscale"))
    option(
        "--layerscale-init",
        "FBT_LAYERSCALE_INIT",
        0.1,
        float,
        help="Initial channelwise carry-residual scale for layerscale fusion",
    )
    option(
        "--detach-carry",
        "FBT_DETACH_CARRY",
        0,
        int,
        choices=(0, 1),
        help="Stop gradients into carried states; keep fusion and within-pass attention trainable",
    )
    option("--seq-len", "SEQ_LEN", 1024, int)
    option(
        "--batch-size",
        "BATCH_SIZE",
        524288,
        int,
        help="Global tokens per optimizer step",
    )
    option("--mbs", "MBS", 64, int, help="Per-rank training microbatch rows")
    option(
        "--val-mbs",
        "VAL_MBS",
        64,
        int,
        help="Per-rank parallel validation microbatch rows",
    )
    option(
        "--val-tokens", "VAL_TOKENS", 20 * 524288, int, help="Global validation tokens"
    )
    option("--matrix-lr", "FBT_MATRIX_LR", 0.01, float)
    option("--matrix-weight-decay", "FBT_MATRIX_WD", 0.01, float)
    option("--adam-lr", "FBT_ADAM_LR", 0.0005, float)
    option("--warmup-steps", "FBT_WARMUP_STEPS", 200, int)
    option("--cooldown-fraction", "FBT_COOLDOWN_FRAC", 0.25, float)
    option("--z-loss", "FBT_Z_LOSS", 1e-5, float, help="Enabled only during cooldown")
    option(
        "--ns-steps", "FBT_NS_STEPS", 10, int, help="NorMuon Newton-Schulz iterations"
    )
    option(
        "--seq-tokens",
        "FBT_SEQ_TOKENS",
        0,
        int,
        help="Global tokens for optional cached sequential evaluation; zero disables it",
    )
    option(
        "--seq-eval-len",
        "FBT_SEQ_LEN",
        128,
        int,
        help="Score prefixes this long for optional sequential validation",
    )
    option(
        "--seq-every",
        "FBT_SEQ_EVERY",
        0,
        int,
        help="Sequential evaluation interval in steps; zero means final-only if enabled",
    )
    option(
        "--seq-mbs", "FBT_SEQ_MBS", 1, int, help="Per-rank sequential evaluation rows"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check configuration, shard headers, and tokenizer without importing torch or using CUDA",
    )
    return parser


def _shard_tokens(path: Path) -> int:
    with path.open("rb") as stream:
        header = stream.read(1024)
    if len(header) != 1024:
        raise ValueError(f"short shard header: {path}")
    magic, version, count = struct.unpack_from("<iii", header)
    if magic != 20240520 or version != 1 or count < 1:
        raise ValueError(f"invalid token shard header: {path}")
    if path.stat().st_size != 1024 + 2 * count:
        raise ValueError(f"shard payload size disagrees with header: {path}")
    return count


def _preflight(args, world_size: int, rank: int, local_rank: int):
    """Read-only checks, including shard headers and tokenizer, before CUDA use."""
    positive = (
        "iterations",
        "val_loss_every",
        "train_log_every",
        "vocab_size",
        "num_layers",
        "model_dim",
        "seq_len",
        "batch_size",
        "mbs",
        "val_mbs",
        "val_tokens",
        "ns_steps",
        "seq_eval_len",
        "seq_mbs",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("mlp_hidden", "kv_heads", "warmup_steps", "seq_tokens", "seq_every"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in (
        "noise",
        "layerscale_init",
        "matrix_lr",
        "matrix_weight_decay",
        "adam_lr",
        "z_loss",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.schedule not in ("progressive", "single"):
        raise ValueError("FBT_SCHEDULE must be progressive or single")
    if args.fusion not in ("glu", "layerscale"):
        raise ValueError("FBT_FUSION must be glu or layerscale")
    if args.detach_carry not in (0, 1):
        raise ValueError("FBT_DETACH_CARRY must be 0 or 1")
    if not 0 < args.cooldown_fraction < 1:
        raise ValueError("cooldown_fraction must be between zero and one")
    cooldown_start = int(args.iterations * (1 - args.cooldown_fraction))
    if args.warmup_steps > cooldown_start:
        raise ValueError(
            "warmup_steps overlaps cooldown; shorten warmup for shorter runs"
        )
    if args.model_dim % 128:
        raise ValueError(
            "MODEL_DIM must be a multiple of the nanoGPT-mini head width (128)"
        )
    if args.kv_heads and (args.model_dim // 128) % args.kv_heads:
        raise ValueError(
            "FBT_KV_HEADS must be zero or a positive divisor of MODEL_DIM / 128"
        )
    if args.vocab_size > 65536:
        raise ValueError("uint16 token shards support at most 65536 vocabulary entries")
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("invalid torchrun WORLD_SIZE/RANK/LOCAL_RANK")
    if args.batch_size % (world_size * args.seq_len * args.mbs):
        raise ValueError(
            "BATCH_SIZE must divide evenly into WORLD_SIZE * SEQ_LEN * MBS"
        )
    if args.val_tokens % (world_size * args.seq_len * args.val_mbs):
        raise ValueError(
            "VAL_TOKENS must divide evenly into WORLD_SIZE * SEQ_LEN * VAL_MBS"
        )
    if args.seq_tokens:
        if args.seq_eval_len > args.seq_len:
            raise ValueError("FBT_SEQ_LEN cannot exceed SEQ_LEN")
        if args.seq_tokens % (world_size * args.seq_eval_len * args.seq_mbs):
            raise ValueError(
                "FBT_SEQ_TOKENS must divide into WORLD_SIZE * FBT_SEQ_LEN * FBT_SEQ_MBS"
            )
        if args.seq_tokens // args.seq_eval_len > args.val_tokens // args.seq_len:
            raise ValueError(
                "sequential evaluation requests more rows than the validation window"
            )
    if args.run_id and (
        Path(args.run_id).name != args.run_id or args.run_id in (".", "..")
    ):
        raise ValueError("RUN_ID must be one directory name")
    data_path = Path(args.data_path)
    train_files = [
        (path, _shard_tokens(path))
        for path in sorted(data_path.glob("fineweb_train_*.bin"))
    ]
    val_files = [
        (path, _shard_tokens(path))
        for path in sorted(data_path.glob("fineweb_val_*.bin"))
    ]
    if not train_files or not val_files:
        raise FileNotFoundError(
            f"expected fineweb_train_*.bin and fineweb_val_*.bin in {data_path}"
        )
    if any(count < args.batch_size + 1 for _, count in train_files):
        raise ValueError(
            "every training shard must contain a global batch plus its final target"
        )
    if sum(count for _, count in val_files) < args.val_tokens + 1:
        raise ValueError("validation shards do not contain VAL_TOKENS + 1 tokens")
    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor(model_file=str(Path(args.tokenizer_path)))
    if tokenizer.vocab_size() != args.vocab_size:
        raise ValueError("VOCAB_SIZE must match the SentencePiece tokenizer")
    return train_files, val_files, tokenizer


def _run(args, rank, local_rank, world_size, train_files, val_files, tokenizer):
    import torch
    import torch.distributed as dist

    from pretraining.nanogpt_mini.nanogpt_mini_full_bandwidth_model import (
        FullBandwidthGPT,
    )
    from shared.normuon import NorMuon

    if not torch.cuda.is_available():
        raise RuntimeError("full-bandwidth pretraining requires CUDA; no CPU fallback")
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("full-bandwidth pretraining requires native CUDA BF16")
    device = torch.device("cuda", local_rank)
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl", device_id=device)
    try:
        _train(
            args,
            rank,
            world_size,
            train_files,
            val_files,
            tokenizer,
            torch,
            dist,
            device,
            FullBandwidthGPT,
            NorMuon,
        )
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _train(
    args,
    rank,
    world_size,
    train_files,
    val_files,
    tokenizer,
    torch,
    dist,
    device,
    FullBandwidthGPT,
    NorMuon,
):
    distributed = world_size > 1

    def reduce_sum(tensor):
        if distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def synchronize():
        torch.cuda.synchronize(device)
        if distributed:
            dist.barrier()

    run_id = args.run_id or str(uuid.uuid4())
    if distributed:
        run_ids = [run_id if rank == 0 else None]
        dist.broadcast_object_list(run_ids, src=0, device=device)
        run_id = run_ids[0]
    output_dir = Path("ablation_results") / run_id
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    logfile = output_dir / "train.log"

    def log(message):
        if rank == 0:
            print(message, flush=True)
            with logfile.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")

    def read_tokens(path, count, offset=0):
        tokens = torch.empty(count, dtype=torch.uint16, pin_memory=True)
        with path.open("rb", buffering=0) as stream:
            stream.seek(1024 + 2 * offset)
            read_bytes = stream.readinto(tokens.numpy())
        if read_bytes != 2 * count:
            raise ValueError(f"short token payload: {path}")
        if int(tokens.numpy().max()) >= args.vocab_size:
            raise ValueError(f"token ID outside VOCAB_SIZE: {path}")
        return tokens

    local_batch = args.batch_size // world_size

    def training_batches():
        for path, count in itertools.cycle(train_files):
            tokens = read_tokens(path, count)
            for offset in range(0, count - args.batch_size, args.batch_size):
                start = offset + rank * local_batch
                buf = tokens[start : start + local_batch + 1]
                inputs = buf[:-1].to(
                    device=device, dtype=torch.int32, non_blocking=True
                )
                targets = buf[1:].to(
                    device=device, dtype=torch.int64, non_blocking=True
                )
                yield inputs.view(-1, args.seq_len), targets.view(-1, args.seq_len)

    # Fixed global validation window, partitioned into equal contiguous rank
    # slices. The extra token preserves each rank's final next-token target.
    local_val_tokens = args.val_tokens // world_size
    val_buffer = torch.empty(local_val_tokens + 1, dtype=torch.uint16, pin_memory=True)
    skip, written = rank * local_val_tokens, 0
    for path, count in val_files:
        if skip >= count:
            skip -= count
            continue
        take = min(count - skip, val_buffer.numel() - written)
        val_buffer[written : written + take].copy_(read_tokens(path, take, skip))
        written += take
        skip = 0
        if written == val_buffer.numel():
            break
    val_inputs = (
        val_buffer[:-1].to(device=device, dtype=torch.int32).view(-1, args.seq_len)
    )
    val_targets = (
        val_buffer[1:].to(device=device, dtype=torch.int64).view(-1, args.seq_len)
    )
    del val_buffer

    # Reuse the challenge metric, including special-token boundary accounting.
    from train_gpt import build_sentencepiece_luts

    byte_lut, space_lut, boundary_lut = build_sentencepiece_luts(
        tokenizer, vocab_size=args.vocab_size, device=device
    )
    val_bytes = byte_lut[val_targets] + (
        space_lut[val_targets] & ~boundary_lut[val_inputs.long()]
    )
    byte_count_tensor = val_bytes.sum(dtype=torch.float64)
    reduce_sum(byte_count_tensor)
    val_byte_count = float(byte_count_tensor)
    if val_byte_count <= 0:
        raise ValueError("validation window has no scored bytes")

    torch.manual_seed(args.seed)
    model = FullBandwidthGPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        mlp_hidden=args.mlp_hidden or None,
        num_kv_heads=args.kv_heads or None,
        noise=args.noise,
        detach_carry=bool(args.detach_carry),
        fusion=args.fusion,
        layerscale_init=args.layerscale_init,
    ).to(device=device, dtype=torch.float32)
    if distributed:
        for parameter in model.parameters():
            dist.broadcast(parameter.detach(), src=0)
    # Separate per-rank mixin/noise streams; schedule itself is rank independent.
    torch.manual_seed(args.seed + rank)
    compiled_train = torch.compile(model, fullgraph=True, dynamic=False)
    compiled_parallel = torch.compile(model.pass_losses, fullgraph=True, dynamic=False)
    compiled_sequential = (
        torch.compile(model.loss_sequential, fullgraph=True, dynamic=False)
        if args.seq_tokens
        else None
    )
    fusion_params = [model.fuse_value.weight, model.fuse_gate.weight]
    fusion_ids = {id(parameter) for parameter in fusion_params}
    inactive_feedback_ids = {
        id(parameter) for parameter in model.inactive_feedback_parameters()
    }
    matrix_params = [
        parameter
        for parameter in model.parameters()
        if parameter.ndim == 2
        and parameter is not model.embed.weight
        and id(parameter) not in fusion_ids
    ]
    adam_params = [
        parameter
        for parameter in model.parameters()
        if parameter.ndim < 2 or parameter is model.embed.weight
    ]
    # LayerScale is a vector: scalar AdamW LR, no decay. AdamW skips its
    # absent gradient on single-pass steps, including existing momentum state.
    adam = torch.optim.AdamW(
        adam_params,
        lr=args.adam_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        fused=True,
    )
    matrix_optimizer = NorMuon(
        matrix_params, lr=args.matrix_lr, momentum=0.95, backend_steps=args.ns_steps
    )
    # A separate shared optimizer lets K=1 genuinely skip fusion-matrix state
    # and updates, rather than manufacturing gradients or decaying unused weights.
    fusion_optimizer = NorMuon(
        fusion_params, lr=args.matrix_lr, momentum=0.95, backend_steps=args.ns_steps
    )
    schedule = PassSchedule()
    pass_plan = (
        [1] * args.iterations
        if args.schedule == "single"
        else [
            schedule.passes_at(step, args.iterations, args.seed)
            for step in range(args.iterations)
        ]
    )
    counts = Counter(pass_plan)
    cooldown_start = int(args.iterations * (1 - args.cooldown_fraction))
    training_config = dict(
        vars(args),
        run_id=run_id,
        world_size=world_size,
        architecture="nanogpt_mini_full_bandwidth_v1",
        schedule_two_pass_start=schedule.two_pass_start,
        schedule_three_pass_start=schedule.three_pass_start,
        schedule_interpretation="stagewise_seeded_exact_counts",
        schedule_pass_counts={str(k): counts[k] for k in (1, 2, 3)},
        schedule_passes=pass_plan,
        cooldown_start=cooldown_start,
        optimizer="shared.NorMuon+fused.AdamW",
        adam_betas=[0.9, 0.95],
        adam_eps=1e-8,
        adam_weight_decay=0.0,
        normuon_momentum=0.95,
        normuon_beta2=0.95,
        parameter_dtype="float32",
        activation_dtype="bfloat16",
        num_query_heads=model.model_dim // 128,
        num_kv_heads=model.num_kv_heads,
        kv_dim=model.kv_dim,
        validation_headline_pass=1,
        validation_prefix_length=1,
        residual_scale=1 / math.sqrt(2 * args.num_layers),
    )
    log("training_config " + json.dumps(training_config, sort_keys=True))
    log(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    log(
        f"device: {torch.cuda.get_device_name(device)} torch: {torch.__version__} world_size: {world_size}"
    )
    log(
        f"schedule: {args.schedule} planned_p1: {counts[1]} planned_p2: {counts[2]} planned_p3: {counts[3]}"
    )
    log(
        f"fusion: {model.fusion} detach_carry: {int(model.detach_carry)}"
        + (
            f" layerscale_init: {model.layerscale_init:g}"
            if model.fusion == "layerscale"
            else ""
        )
    )

    @torch.no_grad()
    def evaluate(step, feedback_trained):
        model.eval()
        final = step == args.iterations
        passes = 8 if final else 3
        sums = torch.zeros(passes, device=device, dtype=torch.float64)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for start in range(0, len(val_inputs), args.val_mbs):
                losses = compiled_parallel(
                    val_inputs[start : start + args.val_mbs],
                    val_targets[start : start + args.val_mbs],
                    passes,
                    prefix_mixin=False,
                    z_loss=0.0,
                )
                sums += torch.stack(losses).double()
        reduce_sum(sums)
        loss_values = (sums / args.val_tokens).tolist()
        metrics = {
            "val_loss": loss_values[0],
            "val_bpb": float(sums[0]) / (math.log(2) * val_byte_count),
            "val_headline_pass": 1,
            "feedback_trained": int(feedback_trained),
        }
        for k in (1, 2, 3, 8):
            if k <= passes:
                metrics[f"val_loss_p{k}"] = loss_values[k - 1]
                metrics[f"val_bpb_p{k}"] = float(sums[k - 1]) / (
                    math.log(2) * val_byte_count
                )
        sequential_due = args.seq_tokens and (
            final or (step > 0 and args.seq_every > 0 and step % args.seq_every == 0)
        )
        if sequential_due:
            rows = args.seq_tokens // (world_size * args.seq_eval_len)
            seq_sum = torch.zeros((), device=device, dtype=torch.float64)
            p1_sum = torch.zeros_like(seq_sum)
            seq_bytes = val_bytes[:rows, : args.seq_eval_len].sum(dtype=torch.float64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for start in range(0, rows, args.seq_mbs):
                    inputs = val_inputs[
                        start : start + args.seq_mbs, : args.seq_eval_len
                    ].contiguous()
                    targets = val_targets[
                        start : start + args.seq_mbs, : args.seq_eval_len
                    ].contiguous()
                    seq_sum += compiled_sequential(
                        inputs, targets, prefix_length=1
                    ).double()
                    p1_sum += compiled_parallel(
                        inputs, targets, 1, prefix_mixin=False, z_loss=0.0
                    )[0].double()
            stats = torch.stack((seq_sum, p1_sum, seq_bytes))
            reduce_sum(stats)
            sequential_loss, one_pass_loss, sequential_bytes = stats.tolist()
            if sequential_bytes <= 0:
                raise ValueError("sequential validation window has no scored bytes")
            metrics.update(
                val_loss_seq=sequential_loss / args.seq_tokens,
                val_bpb_seq=sequential_loss / (math.log(2) * sequential_bytes),
                val_loss_p1_seq=one_pass_loss / args.seq_tokens,
                val_bpb_p1_seq=one_pass_loss / (math.log(2) * sequential_bytes),
                val_seq_tokens=args.seq_tokens,
                val_seq_len=args.seq_eval_len,
            )
        model.train()
        return metrics

    token_equivalent_pass_compute = 0
    seen_counts = Counter()
    training_time = 0.0
    interval_start = None
    last_val_step = 0
    train_loader = training_batches()
    model.zero_grad(set_to_none=True)
    for step in range(args.iterations + 1):
        final = step == args.iterations
        if final or step % args.val_loss_every == 0:
            synchronize()
            elapsed = (
                0.0 if interval_start is None else time.perf_counter() - interval_start
            )
            training_time += elapsed
            step_avg_ms = (
                1000 * elapsed / (step - last_val_step) if step > last_val_step else 0.0
            )
            last_val_step = step
            # Save before final deep validation so a scoring failure cannot lose
            # the trained model. The state is FP32 and does not include wrappers.
            if final and rank == 0:
                checkpoint = output_dir / "final_model.pt"
                torch.save(
                    {
                        "architecture": "nanogpt_mini_full_bandwidth_v1",
                        "model": model.state_dict(),
                        "model_config": model.config,
                        "training_config": training_config,
                        "train_seq_len": args.seq_len,
                        "step": step,
                        "tokens_seen": step * args.batch_size,
                        "token_equivalent_pass_compute": token_equivalent_pass_compute,
                        "observed_pass_counts": dict(seen_counts),
                    },
                    checkpoint,
                )
                log(f"saved checkpoint: {checkpoint}")
            metrics = evaluate(step, seen_counts[2] + seen_counts[3] > 0)
            headline = f"step:{step}/{args.iterations} val_loss:{metrics.pop('val_loss'):.6f} val_bpb:{metrics.pop('val_bpb'):.6f}"
            extras = "".join(f" {key}:{value:.8g}" for key, value in metrics.items())
            log(
                headline
                + f" train_time:{1000 * training_time:.0f}ms step_avg:{step_avg_ms:.2f}ms"
                + f" token_equivalent_pass_compute:{token_equivalent_pass_compute}"
                + f" tokens_seen:{step * args.batch_size} schedule_p1:{seen_counts[1]}"
                + f" schedule_p2:{seen_counts[2]} schedule_p3:{seen_counts[3]}"
                + extras
            )
            synchronize()
            interval_start = time.perf_counter()
        if final:
            break

        passes = pass_plan[step]
        cooldown_scale = (
            (args.iterations - step) / (args.iterations - cooldown_start)
            if step >= cooldown_start
            else 1.0
        )
        warmup_scale = (
            min(1.0, (step + 1) / args.warmup_steps) if args.warmup_steps else 1.0
        )
        lr_scale = warmup_scale * cooldown_scale
        weight_decay = args.matrix_weight_decay * cooldown_scale
        z_loss = args.z_loss if step >= cooldown_start else 0.0
        adam.param_groups[0]["lr"] = args.adam_lr * lr_scale
        matrix_optimizer.param_groups[0]["lr"] = args.matrix_lr * lr_scale
        fusion_optimizer.param_groups[0]["lr"] = args.matrix_lr * lr_scale
        inputs, targets = next(train_loader)
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for start in range(0, len(inputs), args.mbs):
                loss = compiled_train(
                    inputs[start : start + args.mbs],
                    targets[start : start + args.mbs],
                    passes=passes,
                    z_loss=z_loss,
                )
                loss_sum += loss.detach().double()
                loss.backward()
        # Model heads return SUM losses. Manual DDP matches the existing mini
        # trainer, but divides the global gradient sum by the global token batch
        # exactly once (no implicit DDP average or microbatch-dependent scaling).
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                if passes != 1 or id(parameter) not in inactive_feedback_ids:
                    raise RuntimeError(f"unexpected absent gradient: {name}")
                continue
            reduce_sum(parameter.grad)
            parameter.grad.div_(args.batch_size)
        active_matrices = (
            matrix_params if passes == 1 else matrix_params + fusion_params
        )
        with torch.no_grad():
            # shared.NorMuon has no WD argument. Apply decoupled decay only to
            # participating matrices; never create a synthetic fusion gradient.
            decay = 1 - args.matrix_lr * lr_scale * weight_decay
            for parameter in active_matrices:
                parameter.mul_(decay)
        adam.step()
        matrix_optimizer.step()
        if passes > 1:
            fusion_optimizer.step()
        model.zero_grad(set_to_none=True)
        token_equivalent_pass_compute += args.batch_size * passes
        seen_counts[passes] += 1
        if (step + 1) % args.train_log_every == 0 or step + 1 == args.iterations:
            reduce_sum(loss_sum)
            synchronize()
            total_time = training_time + time.perf_counter() - interval_start
            log(
                f"step:{step + 1}/{args.iterations} train_loss:{float(loss_sum) / args.batch_size:.6f}"
                + f" train_time:{1000 * total_time:.0f}ms step_avg:{1000 * total_time / (step + 1):.2f}ms"
                + f" passes:{passes} lr_scale:{lr_scale:.8g} matrix_lr:{args.matrix_lr * lr_scale:.8g}"
                + f" adam_lr:{args.adam_lr * lr_scale:.8g} matrix_weight_decay:{weight_decay:.8g} z_loss:{z_loss:.8g}"
                + f" token_equivalent_pass_compute:{token_equivalent_pass_compute}"
                + f" tokens_seen:{(step + 1) * args.batch_size} schedule_p1:{seen_counts[1]}"
                + f" schedule_p2:{seen_counts[2]} schedule_p3:{seen_counts[3]}"
            )


def main(argv=None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    try:
        train_files, val_files, tokenizer = _preflight(
            args, world_size, rank, local_rank
        )
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    if args.preflight:
        schedule = PassSchedule()
        counts = Counter(
            1
            if args.schedule == "single"
            else schedule.passes_at(step, args.iterations, args.seed)
            for step in range(args.iterations)
        )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "preflight": "ok",
                        "schedule": args.schedule,
                        "pass_counts": {str(k): counts[k] for k in (1, 2, 3)},
                        "train_shards": len(train_files),
                        "val_shards": len(val_files),
                        "tokenizer_vocab_size": tokenizer.vocab_size(),
                        "world_size": world_size,
                        "iterations": args.iterations,
                        "num_query_heads": args.model_dim // 128,
                        "num_kv_heads": args.kv_heads or args.model_dim // 128,
                    },
                    sort_keys=True,
                )
            )
        return
    _run(args, rank, local_rank, world_size, train_files, val_files, tokenizer)


if __name__ == "__main__":
    main()
