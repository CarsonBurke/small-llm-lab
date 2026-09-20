"""Single-GPU, document-parallel, teacher-forced Full-bandwidth training.

Each token has one compiled forward/backward. Incoming carry and historical
K/V are detached; current-token K/V and the entire shared trunk are trained.
Only after backward does the static ring receive the new hidden and K/V.
There are no Jacobi passes, producer passes, or context truncations at page or
optimizer boundaries. The capacity includes the current token.

Validation is a deterministic document-stream panel, NOT packed challenge
validation. Its canonical BPB is val_bpb_document_stream; the metrics runner
displays it in the usual BPB chart while keeping its scope distinct from
packed challenge results.
Importing this module and --preflight do not import torch or execute a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

ARCHITECTURE = "nanogpt_mini_full_bandwidth_document_stream_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_SCOPE = (
    "deterministic document-stream panel; complete BOS-delimited source documents, "
    "fixed token budget with lane refill and potentially partial final documents; "
    "not packed parallel validation or challenge-comparable BPB"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compiled CUDA/BF16 document-parallel Full-bandwidth recurrence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Queue all GPU work with mlq. Single GPU only; no eager/CPU fallback or resume.",
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
        "--run-id",
        "RUN_ID",
        "",
        help="Output in ablation_results/RUN_ID; empty generates a UUID",
    )
    option("--data-path", "DATA_PATH", "data/datasets/fineweb_onepass_sp1024")
    option(
        "--tokenizer-path", "TOKENIZER_PATH", "data/tokenizers/fineweb_1024_bpe.model"
    )
    option("--seed", "SEED", 1337, int)
    option("--vocab-size", "VOCAB_SIZE", 1024, int)
    option("--num-layers", "NUM_LAYERS", 6, int)
    option("--model-dim", "MODEL_DIM", 512, int)
    option(
        "--kv-heads",
        "FBT_KV_HEADS",
        0,
        int,
        help="Zero matches query heads; one shares KV across all query heads",
    )
    option(
        "--mlp-hidden", "MLP_HDIM", 0, int, help="Zero means four times the model width"
    )
    option("--fusion", "FBT_FUSION", "glu", choices=("glu", "layerscale"))
    option("--layerscale-init", "FBT_LAYERSCALE_INIT", 0.1, float)
    option("--noise", "FBT_NOISE", 0.02, float)
    option(
        "--detach-carry",
        "FBT_DETACH_CARRY",
        1,
        int,
        choices=(1,),
        help="Mandatory detached incoming carry",
    )
    option(
        "--detach-kv-history",
        "FBT_DETACH_KV_HISTORY",
        1,
        int,
        choices=(1,),
        help="Mandatory detached historical keys/values",
    )
    option(
        "--document-batch",
        "FBT_DOCUMENT_BATCH",
        512,
        int,
        help="Independent document lanes per token tick",
    )
    option(
        "--cache-capacity",
        "FBT_CACHE_CAPACITY",
        1024,
        int,
        help="Attention window including the current token",
    )
    option(
        "--batch-size",
        "BATCH_SIZE",
        524288,
        int,
        help="Global scored tokens per optimizer update",
    )
    option(
        "--page-ticks",
        "FBT_PAGE_TICKS",
        32,
        int,
        help="Bounded CPU-prefetched ticks per data page; must divide ticks per update",
    )
    option("--iterations", "ITERATIONS", 1000, int)
    parser.add_argument(
        "--val-every",
        "--val-loss-every",
        dest="val_every",
        type=int,
        default=os.environ.get("VAL_LOSS_EVERY", "20"),
    )
    option(
        "--val-tokens",
        "VAL_TOKENS",
        1048576,
        int,
        help="Fixed document-stream panel tokens, not the old packed window",
    )
    option(
        "--val-document-batch",
        "FBT_VAL_DOCUMENT_BATCH",
        512,
        int,
        help="Fixed panel lanes, independent of training document_batch",
    )
    option(
        "--val-seed",
        "FBT_VAL_SEED",
        1338,
        int,
        help="Fixed validation panel seed, independent of the training seed",
    )
    option("--train-log-every", "TRAIN_LOG_EVERY", 10, int)
    option(
        "--checkpoint-every",
        "FBT_CHECKPOINT_EVERY",
        10,
        int,
        help="Atomic last_model.pt interval; also save after first update",
    )
    option("--matrix-lr", "FBT_MATRIX_LR", 0.01, float)
    option("--matrix-weight-decay", "FBT_MATRIX_WD", 0.01, float)
    option("--adam-lr", "FBT_ADAM_LR", 0.0005, float)
    option("--warmup-steps", "FBT_WARMUP_STEPS", 200, int)
    option("--cooldown-fraction", "FBT_COOLDOWN_FRAC", 0.25, float)
    option(
        "--z-loss",
        "FBT_Z_LOSS",
        1e-5,
        float,
        help="Applied once to current-token CE objective, only during cooldown",
    )
    option("--ns-steps", "FBT_NS_STEPS", 10, int)
    option(
        "--compile-mode",
        "FBT_COMPILE_MODE",
        "reduce-overhead",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
    )
    option("--cpu-threads", "FBT_CPU_THREADS", 8, int)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check tokenizer, shard headers, BOS index and configuration without torch/CUDA",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--benchmark",
        action="store_true",
        help="Full-shape warm-cache forward/backward/commit qualification only; no optimizer updates or quality claims",
    )
    modes.add_argument(
        "--eval-checkpoint",
        metavar="PATH",
        help="Evaluate a saved model_config/model on the identical fixed document panel; zero training updates",
    )
    option(
        "--benchmark-warmup",
        "FBT_BENCHMARK_WARMUP",
        32,
        int,
        help="Untimed compiled training ticks on already full synthetic history",
    )
    option(
        "--benchmark-ticks",
        "FBT_BENCHMARK_TICKS",
        256,
        int,
        help="Timed training ticks; no optimizer update",
    )
    return parser


def _preflight(args):
    import sentencepiece as spm

    from pretraining.future_credit_stream.data import DocumentIndex
    from pretraining.nanogpt_mini.nanogpt_mini_full_bandwidth_train import _shard_tokens

    if (
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
    ) != (1, 0, 0):
        raise ValueError(
            "document-stream training supports exactly one GPU/process (WORLD_SIZE=1, RANK=LOCAL_RANK=0)"
        )
    for name in (
        "vocab_size",
        "num_layers",
        "model_dim",
        "document_batch",
        "cache_capacity",
        "batch_size",
        "page_ticks",
        "iterations",
        "val_every",
        "val_tokens",
        "train_log_every",
        "checkpoint_every",
        "ns_steps",
        "cpu_threads",
        "benchmark_warmup",
        "benchmark_ticks",
        "val_document_batch",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("mlp_hidden", "seed", "val_seed", "warmup_steps"):
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
    if args.detach_carry != 1 or args.detach_kv_history != 1:
        raise ValueError("incoming carry and KV history must both be detached")
    if args.fusion not in ("glu", "layerscale"):
        raise ValueError("fusion must be glu or layerscale")
    if args.compile_mode not in (
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    ):
        raise ValueError("invalid compile mode; eager fallback is not supported")
    if not 0 < args.cooldown_fraction < 1:
        raise ValueError("cooldown_fraction must be between zero and one")
    if args.warmup_steps > int(args.iterations * (1 - args.cooldown_fraction)):
        raise ValueError("warmup_steps overlaps cooldown")
    if args.model_dim % 128 or not 2 <= args.vocab_size <= 65536:
        raise ValueError(
            "model_dim must be divisible by head width 128; vocabulary must fit uint16"
        )
    query_heads = args.model_dim // 128
    kv_heads = args.kv_heads or query_heads
    if kv_heads < 1 or query_heads % kv_heads:
        raise ValueError("kv_heads must be zero or a positive divisor of query heads")
    if args.batch_size % args.document_batch:
        raise ValueError("batch_size must divide evenly by document_batch")
    if (args.batch_size // args.document_batch) % args.page_ticks:
        raise ValueError("page_ticks must divide ticks per optimizer step")
    if args.val_tokens % args.val_document_batch:
        raise ValueError("val_tokens must divide evenly by val_document_batch")
    if args.eval_checkpoint and not Path(args.eval_checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.eval_checkpoint}")
    if args.run_id and (
        Path(args.run_id).name != args.run_id or args.run_id in (".", "..")
    ):
        raise ValueError("RUN_ID must be one directory name")
    tokenizer = spm.SentencePieceProcessor(model_file=str(Path(args.tokenizer_path)))
    if (
        tokenizer.vocab_size() != args.vocab_size
        or not 0 <= tokenizer.bos_id() < args.vocab_size
    ):
        raise ValueError("tokenizer vocabulary/BOS does not match the document stream")
    data = Path(args.data_path)
    for split in ("train", "val"):
        files = sorted(data.glob(f"fineweb_{split}_*.bin"))
        if not files:
            raise FileNotFoundError(f"no fineweb_{split}_*.bin shards in {data}")
        for path in files:
            _shard_tokens(path)
    indices = tuple(
        DocumentIndex(
            str(data / f"fineweb_{split}_*.bin"), tokenizer.bos_id(), args.vocab_size
        )
        for split in ("train", "val")
    )
    for split, index, batch in zip(
        ("train", "val"),
        indices,
        (args.document_batch, args.val_document_batch),
        strict=True,
    ):
        if index.document_count < batch:
            raise ValueError(
                f"{split} split needs at least {batch} complete BOS-delimited documents"
            )
    if indices[1].describe()["prediction_tokens_per_epoch"] < args.val_tokens:
        raise ValueError(
            "validation split has fewer complete-document prediction tokens than val_tokens"
        )
    return tokenizer, indices[0], indices[1]


class Engine:
    """AOT-compiled forward/backward with external, persistent accumulated grads."""

    def __init__(self, model, args, byte_luts, device):
        import torch
        import torch.nn.functional as F

        from pretraining.nanogpt_mini.full_bandwidth_stream import stream_forward

        self.model = model
        self.z_weight = torch.zeros((), device=device, dtype=torch.float32)
        torch._dynamo.mark_static_address(self.z_weight)
        byte_lut, space_lut, boundary_lut = byte_luts

        def forward(
            inputs, targets, previous, keys, values, positions, resets, z_weight
        ):
            logits, hidden, new_keys, new_values = stream_forward(
                model,
                inputs,
                previous,
                keys,
                values,
                positions,
                resets,
            )
            ce = F.cross_entropy(logits, targets, reduction="sum")
            z = logits.logsumexp(dim=-1).square().sum()
            # SUM / global tokens is exactly mean CE / ticks_per_update.
            # Z is weighted once, not again during gradient accumulation.
            loss = (ce + z_weight * z) / args.batch_size
            byte_count = byte_lut[targets] + (
                space_lut[targets] & ~boundary_lut[inputs]
            )
            stats = torch.stack(
                (
                    ce.detach().double(),
                    ce.new_tensor(inputs.numel(), dtype=torch.float64),
                    byte_count.sum(dtype=torch.float64),
                    z.detach().double(),
                )
            )
            return loss, hidden, new_keys, new_values, stats

        def validation(inputs, targets, previous, keys, values, positions, resets):
            logits, hidden, new_keys, new_values = stream_forward(
                model,
                inputs,
                previous,
                keys,
                values,
                positions,
                resets,
            )
            ce = F.cross_entropy(logits, targets, reduction="sum")
            byte_count = byte_lut[targets] + (
                space_lut[targets] & ~boundary_lut[inputs]
            )
            stats = torch.stack(
                (
                    ce.double(),
                    ce.new_tensor(inputs.numel(), dtype=torch.float64),
                    byte_count.sum(dtype=torch.float64),
                )
            )
            return hidden, new_keys, new_values, stats

        options = {"fullgraph": True, "dynamic": False, "mode": args.compile_mode}
        self.training_forward = torch.compile(forward, **options)
        self.validation_forward = torch.compile(validation, **options)

    def backward_page(self, page, state, totals, ticks=None):
        import torch

        steps = page["inputs"].shape[0] if ticks is None else ticks
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for offset in range(steps):
                torch.compiler.cudagraph_mark_step_begin()
                loss, hidden, new_keys, new_values, stats = self.training_forward(
                    page["inputs"][offset],
                    page["targets"][offset],
                    state.previous,
                    state.keys,
                    state.values,
                    state.positions,
                    page["resets"][offset],
                    self.z_weight,
                )
                loss.backward()
                totals.add_(stats)
                # Both forward and backward read the ring at its static addresses.
                # Graph-owned outputs must be consumed before the next mark_step.
                state.commit(hidden, new_keys, new_values, page["resets"][offset])
                del loss, hidden, new_keys, new_values, stats

    def validation_page(self, page, state, totals):
        import torch

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for offset in range(page["inputs"].shape[0]):
                torch.compiler.cudagraph_mark_step_begin()
                hidden, new_keys, new_values, stats = self.validation_forward(
                    page["inputs"][offset],
                    page["targets"][offset],
                    state.previous,
                    state.keys,
                    state.values,
                    state.positions,
                    page["resets"][offset],
                )
                totals.add_(stats)
                state.commit(hidden, new_keys, new_values, page["resets"][offset])
                del hidden, new_keys, new_values, stats


def _optimizers(model, args):
    import torch

    from shared.normuon import NorMuon

    fusion_params = [model.fuse_value.weight, model.fuse_gate.weight]
    fusion_ids = {id(parameter) for parameter in fusion_params}
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
    adam = torch.optim.AdamW(
        adam_params,
        lr=args.adam_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        fused=True,
    )
    matrices = NorMuon(
        matrix_params, lr=args.matrix_lr, momentum=0.95, backend_steps=args.ns_steps
    )
    fusion = NorMuon(
        fusion_params, lr=args.matrix_lr, momentum=0.95, backend_steps=args.ns_steps
    )
    return (adam, matrices, fusion), matrix_params + fusion_params


def _configuration(args, model, tokenizer, train_index, val_index, torch, device):
    sources = [
        Path(__file__),
        REPO_ROOT / "scripts/train_nanogpt_mini_full_bandwidth_stream.py",
        REPO_ROOT / "pretraining/nanogpt_mini/full_bandwidth_stream.py",
        REPO_ROOT / "pretraining/nanogpt_mini/full_bandwidth_stream_attention.py",
        REPO_ROOT / "pretraining/nanogpt_mini/nanogpt_mini_full_bandwidth_model.py",
        REPO_ROOT / "pretraining/nanogpt_mini/nanogpt_mini_model.py",
        REPO_ROOT / "pretraining/future_credit_stream/data.py",
        REPO_ROOT / "pretraining/future_credit_stream/training.py",
        REPO_ROOT / "shared/normuon.py",
        REPO_ROOT / "train_gpt.py",
    ]
    return dict(
        vars(args),
        architecture=ARCHITECTURE,
        model_config=model.config,
        world_size=1,
        source_sha256={
            str(path.relative_to(REPO_ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sources
            if path.exists()
        },
        tokenizer_sha256=hashlib.sha256(
            Path(args.tokenizer_path).read_bytes()
        ).hexdigest(),
        bos_id=tokenizer.bos_id(),
        train_data=train_index.describe(),
        val_data=val_index.describe(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device),
        optimizer=None
        if args.eval_checkpoint or args.benchmark
        else "shared.NorMuon+fused.AdamW",
        adam_betas=[0.9, 0.95],
        adam_eps=1e-8,
        adam_weight_decay=0.0,
        normuon_momentum=0.95,
        normuon_beta2=0.95,
        parameter_dtype="float32",
        activation_dtype="bfloat16",
        kv_dtype="bfloat16",
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        ticks_per_optimizer_step=args.batch_size // args.document_batch,
        pages_per_optimizer_step=args.batch_size
        // args.document_batch
        // args.page_ticks,
        carry_gradient_policy="no_grad_evaluation"
        if args.eval_checkpoint
        else "detached_incoming",
        kv_gradient_policy="no_grad_evaluation"
        if args.eval_checkpoint
        else "detached_history_current_kv_attached",
        cache_window="current token plus most recent capacity-1 tokens since actual BOS",
        rotary_positions="absolute position since actual BOS; never reset at page/update boundaries",
        state_crosses_optimizer_updates=True,
        state_updates_per_token=1,
        forwards_per_token=1,
        data_order="deterministic shard/document order, cyclic refill, shuffle=False; no corpus-sized permutation",
        validation_scope=EVALUATION_SCOPE,
        validation_primary="document-stream BPB (val_bpb_document_stream)",
        validation_bpb_metric="val_bpb_document_stream",
        validation_document_batch=args.val_document_batch,
        validation_ticks=args.val_tokens // args.val_document_batch,
        validation_seed=args.val_seed,
        validation_shuffle=False,
        cooldown_start=int(args.iterations * (1 - args.cooldown_fraction)),
        schedule="WSD: linear learning-rate warmup, stable plateau, final cooldown; matrix decay cools down",
        objective="sum(current-token CE + cooldown_weight*logsumexp(logits)^2)/global token budget",
        initialization_warmup="none; LR warmup changes rate, never recurrence/objective",
        optimizer_group_policy="trunk matrices NorMuon; fusion matrices separate NorMuon; tied embedding and 1D gains including carry_scale AdamW/no decay",
        zero_grad_policy="persistent FP32 grads zeroed in place once per optimizer update; BOS-only ticks allowed",
        checkpoint_scope="model/config/metrics only, not resumable; no hidden-state or data-cursor resume claim",
        model_constructor_source="saved checkpoint model_config"
        if args.eval_checkpoint
        else "training CLI/environment",
        execution_mode="checkpoint_evaluation"
        if args.eval_checkpoint
        else ("benchmark" if args.benchmark else "training"),
        query_heads=model.model_dim // 128,
        effective_kv_heads=model.num_kv_heads,
        kv_width=model.kv_dim,
        static_state_bytes=4
        * model.num_layers
        * args.document_batch
        * args.cache_capacity
        * model.kv_dim
        + 2 * args.document_batch * model.model_dim
        + 8 * args.document_batch,
        validation_state_bytes=4
        * model.num_layers
        * args.val_document_batch
        * args.cache_capacity
        * model.kv_dim
        + 2 * args.val_document_batch * model.model_dim
        + 8 * args.val_document_batch,
    )


def _evaluate(engine, args, index, state, device):
    import torch

    from pretraining.future_credit_stream.data import StreamingDocuments
    from pretraining.future_credit_stream.training import PrefetchedDocuments

    engine.model.eval()
    state.reset()
    # Recreate the same ordered lane panel each time, retaining only bounded pages.
    stream = StreamingDocuments(
        index, args.val_document_batch, args.val_seed, shuffle=False
    )
    prefetch = PrefetchedDocuments(stream, args.page_ticks, 0)
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    remaining = args.val_tokens // args.val_document_batch
    started = time.perf_counter()
    try:
        while remaining:
            page, _ = prefetch.next(device)
            count = min(remaining, args.page_ticks)
            if count != args.page_ticks:
                page = {key: value[:count] for key, value in page.items()}
            engine.validation_page(page, state, totals)
            remaining -= count
            del page
        torch.cuda.synchronize(device)
        ce_sum, tokens, byte_count = totals.tolist()
    finally:
        prefetch.close()
        engine.model.train()
    if (
        not all(math.isfinite(value) for value in (ce_sum, tokens, byte_count))
        or byte_count <= 0
    ):
        raise FloatingPointError(
            "document-stream validation has non-finite statistics or no scored bytes"
        )
    return {
        "val_loss_document_stream": ce_sum / tokens,
        "val_bpb_document_stream": ce_sum / (math.log(2) * byte_count),
        "val_ce_sum_document_stream": ce_sum,
        "val_tokens_document_stream": tokens,
        "val_bytes_document_stream": byte_count,
        "val_scope_document_stream": 1,
        "val_document_batch": args.val_document_batch,
        "val_ticks": args.val_tokens // args.val_document_batch,
        "val_panel_seed": args.val_seed,
        "val_cache_capacity": args.cache_capacity,
        "eval_seconds": time.perf_counter() - started,
    }


def _benchmark(engine, args, state, output_dir, config, log, device):
    import torch

    from pretraining.future_credit_stream.training import atomic_json

    # Synthetic full rings deliberately isolate the steady-state runtime. This
    # is not a no-grad producer pass, reduced model, or a validation/quality run.
    with torch.no_grad():
        state.previous.normal_()
        for cache in (*state.keys, *state.values):
            cache.normal_()
        state.positions.fill_(args.cache_capacity)
    raw = torch.randint(
        args.vocab_size - 1, (args.page_ticks, args.document_batch), device=device
    )
    inputs = raw + (raw >= config["bos_id"])
    page = {
        "inputs": inputs,
        "targets": torch.randint(args.vocab_size, inputs.shape, device=device),
        "resets": torch.zeros_like(inputs, dtype=torch.bool),
    }
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    engine.z_weight.fill_(args.z_loss)

    def ticks(count):
        while count:
            take = min(count, args.page_ticks)
            engine.backward_page(page, state, totals, ticks=take)
            count -= take

    log(
        "benchmark_scope full dimensions; synthetic full-capacity history; one forward/backward/commit per tick; no optimizer updates or quality evidence"
    )
    warmup_start = time.perf_counter()
    ticks(args.benchmark_warmup)
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_start
    engine.model.zero_grad(set_to_none=False)
    totals.zero_()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    ticks(args.benchmark_ticks)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    statistics = totals.tolist()
    if not all(math.isfinite(value) for value in statistics):
        raise FloatingPointError("non-finite benchmark objective statistics")
    result = {
        "architecture": ARCHITECTURE,
        "run_id": args.run_id,
        "benchmark_only": True,
        "quality_evidence": False,
        "optimizer_updates": 0,
        "shape": model_shape(engine.model, args),
        "benchmark_warmup": args.benchmark_warmup,
        "benchmark_ticks": args.benchmark_ticks,
        "warmup_seconds": warmup_seconds,
        "elapsed_seconds": elapsed,
        "tokens": args.benchmark_ticks * args.document_batch,
        "tokens_per_second": args.benchmark_ticks * args.document_batch / elapsed,
        "milliseconds_per_tick": 1000 * elapsed / args.benchmark_ticks,
        "estimated_tick_seconds_per_optimizer_step": elapsed
        / args.benchmark_ticks
        * (args.batch_size // args.document_batch),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "static_state_bytes": config["static_state_bytes"],
        "memory_scope": "model, persistent gradients, one full stream state, compiled forward/backward/commit; excludes optimizer state/workspace and separate validation ring",
        "objective_scope": "training CE + configured cooldown z-loss, normalized by global token budget",
        "history_scope": "synthetic BF16 full-capacity ring; no resets, no context shortening",
        "ce_sum": statistics[0],
        "scored_tokens": statistics[1],
        "scored_bytes": statistics[2],
    }
    atomic_json(output_dir / "benchmark.json", result)
    log("benchmark_result " + json.dumps(result, sort_keys=True, allow_nan=False))


def model_shape(model, args):
    return dict(
        model.config,
        document_batch=args.document_batch,
        cache_capacity=args.cache_capacity,
        batch_size=args.batch_size,
        page_ticks=args.page_ticks,
    )


def _train(
    engine,
    args,
    train_index,
    val_index,
    state,
    validation_state,
    output_dir,
    config,
    log,
    device,
):
    import torch

    from pretraining.future_credit_stream.data import StreamingDocuments
    from pretraining.future_credit_stream.training import PrefetchedDocuments

    optimizers, decayed = _optimizers(engine.model, args)
    stream = StreamingDocuments(
        train_index, args.document_batch, args.seed, shuffle=False
    )
    prefetch = PrefetchedDocuments(stream, args.page_ticks, 0)
    cumulative = torch.zeros(4, device=device, dtype=torch.float64)
    interval = torch.zeros_like(cumulative)
    train_seconds = interval_seconds = interval_wait = 0.0
    interval_steps = 0
    completed = 0

    def checkpoint(filename):
        path = output_dir / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "architecture": ARCHITECTURE,
                "model": engine.model.state_dict(),
                "model_config": engine.model.config,
                "training_config": config,
                "step": completed,
                "tokens_seen": completed * args.batch_size,
                "train_seconds": train_seconds,
                "cumulative_statistics": dict(
                    zip(("ce_sum", "tokens", "bytes", "z_sum"), cumulative.tolist())
                ),
                "resumable": False,
            },
            temporary,
        )
        temporary.replace(path)
        log(f"saved checkpoint: {path}")

    try:
        for step in range(args.iterations + 1):
            if step % args.val_every == 0 or step == args.iterations:
                torch.cuda.synchronize(device)
                metrics = _evaluate(engine, args, val_index, validation_state, device)
                loss = metrics["val_loss_document_stream"]
                extras = " ".join(
                    f"{key}:{value:.10g}" for key, value in metrics.items()
                )
                # Retain the explicit scope while making BPB the primary metric.
                log(
                    f"step:{step}/{args.iterations} val_loss:{loss:.8f} val_bpb_document_stream:{metrics['val_bpb_document_stream']:.10g} "
                    f"train_time:{1000 * train_seconds:.3f}ms tokens_seen:{step * args.batch_size} {extras}"
                )
            if step == args.iterations:
                break
            engine.model.train()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            engine.model.zero_grad(set_to_none=False)
            cooldown_start = config["cooldown_start"]
            cooldown = (
                (args.iterations - step) / (args.iterations - cooldown_start)
                if step >= cooldown_start
                else 1.0
            )
            warmup = (
                min(1.0, (step + 1) / args.warmup_steps) if args.warmup_steps else 1.0
            )
            lr_scale = warmup * cooldown
            weight_decay = args.matrix_weight_decay * cooldown
            z_weight = args.z_loss if step >= cooldown_start else 0.0
            engine.z_weight.fill_(z_weight)
            for optimizer, base_lr in zip(
                optimizers, (args.adam_lr, args.matrix_lr, args.matrix_lr), strict=True
            ):
                optimizer.param_groups[0]["lr"] = base_lr * lr_scale
            step_stats = torch.zeros_like(cumulative)
            for _ in range(config["pages_per_optimizer_step"]):
                page, wait_seconds = prefetch.next(device)
                interval_wait += wait_seconds
                engine.backward_page(page, state, step_stats)
                del page
            with torch.no_grad():
                decay = 1 - args.matrix_lr * lr_scale * weight_decay
                for parameter in decayed:
                    parameter.mul_(decay)
            for optimizer in optimizers:
                optimizer.step()
            cumulative.add_(step_stats)
            interval.add_(step_stats)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            train_seconds += elapsed
            interval_seconds += elapsed
            interval_steps += 1
            completed = step + 1
            if completed % args.train_log_every == 0 or completed == args.iterations:
                ce, tokens, byte_count, z_sum = interval.tolist()
                all_ce, all_tokens, all_bytes, _ = cumulative.tolist()
                if not all(
                    math.isfinite(value)
                    for value in (ce, tokens, byte_count, z_sum, all_ce)
                ):
                    raise FloatingPointError(
                        "non-finite document-stream training statistics"
                    )
                log(
                    f"step:{completed}/{args.iterations} train_loss:{ce / tokens:.8f} "
                    f"train_time:{1000 * train_seconds:.3f}ms step_avg:{1000 * interval_seconds / interval_steps:.3f}ms "
                    f"tokens_per_second:{tokens / interval_seconds:.3f} data_wait_ms:{1000 * interval_wait / interval_steps:.3f} "
                    f"train_ce_sum:{all_ce:.10g} train_tokens:{all_tokens:.0f} train_bytes:{all_bytes:.0f} "
                    f"tokens_seen:{completed * args.batch_size} token_equivalent_pass_compute:{completed * args.batch_size} "
                    f"state_updates:{completed * config['ticks_per_optimizer_step']} forwards_per_token:1 "
                    f"lr_scale:{lr_scale:.10g} matrix_lr:{args.matrix_lr * lr_scale:.10g} adam_lr:{args.adam_lr * lr_scale:.10g} "
                    f"matrix_weight_decay:{weight_decay:.10g} z_loss:{z_weight:.10g} z_mean:{z_sum / tokens:.10g} "
                    f"peak_vram_allocated_mib:{torch.cuda.max_memory_allocated(device) / 2**20:.3f} "
                    f"peak_vram_reserved_mib:{torch.cuda.max_memory_reserved(device) / 2**20:.3f}"
                )
                interval.zero_()
                interval_seconds = interval_wait = 0.0
                interval_steps = 0
            # Save before validation, including final validation, so scoring or
            # a later wall limit cannot erase the last completed model updates.
            if (
                completed == 1
                or completed % args.checkpoint_every == 0
                or completed == args.iterations
            ):
                checkpoint("last_model.pt")
            if completed == args.iterations:
                checkpoint("final_model.pt")
    finally:
        prefetch.close()


def _run(args, tokenizer, train_index, val_index):
    import torch

    from pretraining.future_credit_stream.training import MetricLog, atomic_json
    from pretraining.nanogpt_mini.full_bandwidth_stream import StreamState
    from pretraining.nanogpt_mini.nanogpt_mini_full_bandwidth_model import (
        FullBandwidthGPT,
    )
    from train_gpt import build_sentencepiece_luts

    if not torch.cuda.is_available():
        raise RuntimeError("document-stream training requires CUDA; no CPU fallback")
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("document-stream training requires native CUDA BF16")
    device = torch.device("cuda", 0)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    args.run_id = args.run_id or str(uuid.uuid4())
    output_dir = REPO_ROOT / "ablation_results" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(
        (output_dir / name).exists()
        for name in (
            "run_config.json",
            "last_model.pt",
            "final_model.pt",
            "benchmark.json",
            "evaluation.json",
        )
    ):
        raise FileExistsError(
            "run artifacts already exist: choose a fresh RUN_ID; resume is not supported"
        )
    metric_log = (
        None
        if args.benchmark
        else MetricLog(SimpleNamespace(output_dir=output_dir, run_id=args.run_id))
    )

    def log(message):
        if metric_log is None:
            print(message, flush=True)
        else:
            metric_log.emit(message)
        with (output_dir / "train.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    try:
        if args.eval_checkpoint:
            saved = torch.load(
                args.eval_checkpoint, map_location="cpu", weights_only=True
            )
            if (
                not isinstance(saved, dict)
                or not isinstance(saved.get("model_config"), dict)
                or "model" not in saved
            ):
                raise ValueError("checkpoint must contain model_config and model")
            if saved["model_config"].get("vocab_size") != tokenizer.vocab_size():
                raise ValueError(
                    "checkpoint vocabulary does not match evaluation tokenizer"
                )
            model = FullBandwidthGPT(**saved["model_config"]).to(
                device=device, dtype=torch.float32
            )
            model.load_state_dict(saved["model"], strict=True)
            checkpoint_step = int(saved.get("step", 0))
            del saved
            model.eval()
        else:
            model = FullBandwidthGPT(
                vocab_size=args.vocab_size,
                num_layers=args.num_layers,
                model_dim=args.model_dim,
                mlp_hidden=args.mlp_hidden or None,
                noise=args.noise,
                detach_carry=True,
                fusion=args.fusion,
                layerscale_init=args.layerscale_init,
                num_kv_heads=args.kv_heads or None,
            ).to(device=device, dtype=torch.float32)
            model.train()
            # Grad storage is allocated outside graph capture and is never replaced.
            # BOS-only ticks legitimately contribute zero to feedback parameters.
            for parameter in model.parameters():
                parameter.grad = torch.zeros_like(parameter)
        byte_luts = build_sentencepiece_luts(tokenizer, args.vocab_size, device)
        config = _configuration(
            args, model, tokenizer, train_index, val_index, torch, device
        )
        atomic_json(output_dir / "run_config.json", config)
        log("training_config " + json.dumps(config, sort_keys=True, allow_nan=False))
        log("validation_scope " + EVALUATION_SCOPE)
        engine = Engine(model, args, byte_luts, device)
        if args.eval_checkpoint:
            state = StreamState(
                model, args.val_document_batch, args.cache_capacity, device
            )
            metrics = _evaluate(engine, args, val_index, state, device)
            checkpoint_path = Path(args.eval_checkpoint).resolve()
            result = dict(
                architecture=ARCHITECTURE,
                evaluation_only=True,
                optimizer_updates=0,
                checkpoint_path=str(checkpoint_path),
                checkpoint_step=checkpoint_step,
                checkpoint_sha256=hashlib.sha256(
                    checkpoint_path.read_bytes()
                ).hexdigest(),
                model_config=model.config,
                validation_scope=EVALUATION_SCOPE,
                validation_index_fingerprint=val_index.fingerprint,
                tokenizer_sha256=config["tokenizer_sha256"],
                document_batch=args.val_document_batch,
                tokens=args.val_tokens,
                ticks=args.val_tokens // args.val_document_batch,
                panel_seed=args.val_seed,
                cache_capacity=args.cache_capacity,
                **metrics,
            )
            atomic_json(output_dir / "evaluation.json", result)
            loss = metrics["val_loss_document_stream"]
            extras = " ".join(f"{key}:{value:.10g}" for key, value in metrics.items())
            log(
                f"step:0/0 val_loss:{loss:.8f} val_bpb_document_stream:{metrics['val_bpb_document_stream']:.10g} train_time:0ms {extras}"
            )
            log(
                "evaluation_result "
                + json.dumps(result, sort_keys=True, allow_nan=False)
            )
        else:
            state = StreamState(model, args.document_batch, args.cache_capacity, device)
            if args.benchmark:
                _benchmark(engine, args, state, output_dir, config, log, device)
            else:
                validation_state = StreamState(
                    model, args.val_document_batch, args.cache_capacity, device
                )
                _train(
                    engine,
                    args,
                    train_index,
                    val_index,
                    state,
                    validation_state,
                    output_dir,
                    config,
                    log,
                    device,
                )
    finally:
        if metric_log is not None:
            metric_log.close()


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        tokenizer, train_index, val_index = _preflight(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    if args.preflight:
        print(
            json.dumps(
                {
                    "preflight": "ok",
                    "architecture": ARCHITECTURE,
                    "config": vars(args),
                    "world_size": 1,
                    "tokenizer_vocab_size": tokenizer.vocab_size(),
                    "bos_id": tokenizer.bos_id(),
                    "train_data": train_index.describe(),
                    "val_data": val_index.describe(),
                    "ticks_per_optimizer_step": args.batch_size // args.document_batch,
                    "static_state_bytes": 4
                    * args.num_layers
                    * args.document_batch
                    * args.cache_capacity
                    * ((args.kv_heads or args.model_dim // 128) * 128)
                    + 2 * args.document_batch * args.model_dim
                    + 8 * args.document_batch,
                    "evaluation_scope": EVALUATION_SCOPE,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    _run(args, tokenizer, train_index, val_index)


if __name__ == "__main__":
    main()
