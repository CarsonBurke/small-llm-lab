"""Prepare an explicit offline reasoning corpus and train Uno BEFORE VAPO RL.

Run CUDA work through mlq (max-parallel-runs 1). No dataset is downloaded by
this command. --prepare-only tokenizes local JSONL/text without loading a model.
A .txt file is one document; JSONL accepts text, messages, conversations
(OpenThoughts from/value turns), or prompt/response and problem/solution pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from checkpointing import RecoveryCheckpointPolicy, atomic_torch_save
from postraining.hf_runtime import prepare_text_only_transformers_runtime
from postraining.minicpm_vapo import (
    MINICPM5_MODEL_ID,
    MINICPM5_REVISION,
    LoRAConfig,
    TrajectoryRecord,
    inject_lora,
    load_adapter_state_dict,
    merge_lora_for_inference,
)
from postraining.uno import (
    UnoConfig,
    UnoAdapterBank,
    attach_uno_adapters,
    enable_uno_checkpointing,
    load_uno_adapter,
    make_uno_block_mask,
    uno_checkpoint_payload,
    uno_distillation_loss,
)

CORPUS_SCHEMA = "minicpm_uno_corpus/v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def identity_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def document_text(row: dict[str, Any], tokenizer: Any) -> str:
    turns = row.get("messages", row.get("conversations"))
    if turns is not None:
        if not isinstance(turns, list) or not turns:
            raise ValueError("conversation must contain at least one turn")
        messages = []
        roles = {
            "human": "user",
            "gpt": "assistant",
            "user": "user",
            "assistant": "assistant",
            "system": "system",
        }
        for turn in turns:
            role = roles.get(turn.get("role", turn.get("from")))
            content = turn.get("content", turn.get("value"))
            if role is None or not isinstance(content, str):
                raise ValueError(
                    "conversation needs supported role/from and string content/value"
                )
            messages.append({"role": role, "content": content})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    if isinstance(row.get("text"), str) and row["text"].strip():
        return row["text"]
    for question, answer in (
        ("prompt", "response"),
        ("problem", "solution"),
        ("question", "answer"),
    ):
        if isinstance(row.get(question), str) and isinstance(row.get(answer), str):
            return tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": row[question]},
                    {"role": "assistant", "content": row[answer]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
    raise ValueError(
        "record needs text, messages/conversations, or a prompt/response pair"
    )


def iter_documents(path: Path, tokenizer: Any):
    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"empty text corpus: {path}")
        yield text
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("record must be an object")
                    yield document_text(row, tokenizer)
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
    else:
        raise ValueError(
            f"unsupported corpus extension {path.suffix}; use .jsonl or .txt"
        )


def prepare_corpus(args, tokenizer) -> tuple[dict[str, Any], Path]:
    directory = Path(args.output) / "corpus"
    directory.mkdir(parents=True, exist_ok=True)
    paths = [Path(path).resolve() for path in args.data]
    heldout_paths = [Path(path).resolve() for path in args.validation_data]
    sources = {
        "train": [file_sha256(path) for path in paths],
        "validation": [file_sha256(path) for path in heldout_paths],
    }
    identity = {
        "schema": CORPUS_SCHEMA,
        "sources": sources,
        "model_id": args.model,
        "revision": args.revision,
        "tokenizer_sha256": identity_sha256(tokenizer.get_vocab()),
        "chat_template": tokenizer.chat_template,
        "seed": args.seed,
        "heldout_fraction": args.heldout_fraction,
        "eos_token_id": tokenizer.eos_token_id,
    }
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError(
                "existing prepared corpus identity differs; choose a fresh output directory"
            )
        for split in ("train", "validation"):
            if file_sha256(directory / f"{split}.bin") != manifest[split]["sha256"]:
                raise ValueError(f"prepared {split} corpus bytes changed")
        return manifest, directory
    if tokenizer.eos_token_id is None:
        raise ValueError("offline stream packing requires tokenizer EOS")
    counts = {"train": 0, "validation": 0}
    documents = {"train": 0, "validation": 0}
    temporary = {
        split: directory / f".{split}.{os.getpid()}.working" for split in counts
    }
    streams = {split: path.open("wb") for split, path in temporary.items()}
    # Assignment by rendered document hash keeps duplicate texts in one split.
    # Explicit heldout files are also checked against the training document set.
    heldout_hashes = set()
    try:
        for source_split, source_paths in (
            ("validation", heldout_paths),
            ("train", paths),
        ):
            for path in source_paths:
                for text in iter_documents(path, tokenizer):
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    if source_split == "validation":
                        heldout_hashes.add(digest)
                        split = "validation"
                    elif heldout_paths:
                        if digest in heldout_hashes:
                            raise ValueError(
                                "explicit heldout document also occurs in training data"
                            )
                        split = "train"
                    else:
                        score = (
                            int(
                                hashlib.sha256(
                                    f"{args.seed}:{digest}".encode()
                                ).hexdigest()[:16],
                                16,
                            )
                            / 2**64
                        )
                        split = (
                            "validation" if score < args.heldout_fraction else "train"
                        )
                    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                    if not ids or ids[-1] != tokenizer.eos_token_id:
                        ids.append(tokenizer.eos_token_id)
                    np.asarray(ids, dtype="<u4").tofile(streams[split])
                    counts[split] += len(ids)
                    documents[split] += 1
        for stream in streams.values():
            stream.close()
        for split in counts:
            if counts[split] < args.sequence_length:
                raise ValueError(
                    f"{split} has only {counts[split]} tokens; need >= sequence length. Supply --validation-data for small corpora."
                )
        manifest: dict[str, Any] = {"identity": identity}
        for split in counts:
            destination = directory / f"{split}.bin"
            os.replace(temporary[split], destination)
            manifest[split] = {
                "tokens": counts[split],
                "documents": documents[split],
                "sha256": file_sha256(destination),
            }
        manifest["corpus_sha256"] = identity_sha256(manifest)
        staged_manifest = directory / ".manifest.working"
        staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(staged_manifest, manifest_path)
        return manifest, directory
    finally:
        for stream in streams.values():
            stream.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)


class WindowStream:
    """Reproducible epoch permutations without a corpus-sized Python index list."""

    def __init__(self, path: Path, sequence_length: int, seed: int) -> None:
        self.tokens = np.memmap(path, mode="r", dtype="<u4")
        self.length = sequence_length
        self.windows = len(self.tokens) // sequence_length
        if self.windows < 1:
            raise ValueError("corpus does not contain one complete sequence")
        self.seed = seed

    def batch(self, cursor: int, batch_size: int, device: torch.device) -> torch.Tensor:
        result = np.empty((batch_size, self.length), dtype=np.int64)
        for row in range(batch_size):
            epoch, offset = divmod(cursor + row, self.windows)
            rng = random.Random(self.seed + epoch)
            stride = rng.randrange(1, self.windows + 1)
            while math.gcd(stride, self.windows) != 1:
                stride = stride % self.windows + 1
            index = (offset * stride + rng.randrange(self.windows)) % self.windows
            result[row] = self.tokens[index * self.length : (index + 1) * self.length]
        return torch.from_numpy(result).pin_memory().to(device, non_blocking=True)


def parse_curriculum(text: str) -> list[tuple[int, int]]:
    try:
        stages = [
            (int(block), int(tokens))
            for block, tokens in (item.split(":") for item in text.split(","))
        ]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "curriculum is block_size:tokens,..."
        ) from error
    if not stages or any(block < 2 or tokens < 1 for block, tokens in stages):
        raise argparse.ArgumentTypeError(
            "each curriculum stage needs block_size>=2 and positive tokens"
        )
    return stages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="local JSONL or .txt training documents",
    )
    parser.add_argument(
        "--validation-data",
        nargs="+",
        default=[],
        help="optional disjoint local heldout documents",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MINICPM5_MODEL_ID)
    parser.add_argument("--revision", default=MINICPM5_REVISION)
    parser.add_argument(
        "--teacher-checkpoint",
        help="optional VAPO v6 RL checkpoint supplying the frozen actor",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume", help="exact-recovery Uno checkpoint (normally OUTPUT/latest.pt)"
    )
    parser.add_argument("--rank", type=int, default=48)
    parser.add_argument("--alpha", type=float, default=3072.0)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=64,
        help="microbatches per optimizer update; default effective batch is 64",
    )
    parser.add_argument(
        "--curriculum",
        type=parse_curriculum,
        default=parse_curriculum("2:100000000,4:300000000"),
        help="block size and supervised-token budget per stage",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=None,
        help="linear warmup budget; default 2 percent of total curriculum tokens",
    )
    parser.add_argument("--head-chunk-size", type=int, default=32)
    parser.add_argument("--heldout-fraction", type=float, default=0.01)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval-seconds", type=float, default=480.0)
    return parser


def load_teacher(args, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="flex_attention",
        low_cpu_mem_usage=True,
    ).to(device)
    teacher_sha256 = identity_sha256(
        {"model_id": args.model, "revision": args.revision}
    )
    if args.teacher_checkpoint:
        teacher_sha256 = file_sha256(args.teacher_checkpoint)
        # VAPO recovery may contain this known repository dataclass, never permit
        # unrestricted pickle execution for externally supplied checkpoints.
        with torch.serialization.safe_globals([TrajectoryRecord]):
            source = torch.load(
                args.teacher_checkpoint, map_location="cpu", weights_only=True
            )
        policy = source.get("policy", {})
        if policy.get("schema") != "minicpm5_vapo_adapter/v6":
            raise ValueError("teacher checkpoint must use VAPO adapter/v6 schema")
        actor = policy["actor"]
        if (
            actor.get("model_id") != args.model
            or actor.get("revision") != args.revision
        ):
            raise ValueError("teacher checkpoint model identity differs")
        inject_lora(model, LoRAConfig(**actor["lora_config"]))
        model.to(device)  # Newly injected fp32 masters start on CPU.
        expected = {
            name: value
            for name, value in model.named_parameters()
            if name.endswith(("lora_a", "lora_b"))
        }
        if set(expected) != set(actor["adapter"]):
            raise ValueError("teacher adapter keys differ")
        for name, value in actor["adapter"].items():
            if value.shape != expected[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid teacher adapter: {name}")
        load_adapter_state_dict(model, actor["adapter"])
        merge_lora_for_inference(model)
        del source
    model.requires_grad_(False)
    model.config.use_cache = False
    if getattr(model.config, "attention_dropout", 0.0):
        raise ValueError("Uno paired teacher requires zero attention dropout")
    return model, teacher_sha256


def training_config(args) -> dict[str, Any]:
    return {
        name: value
        for name, value in vars(args).items()
        if name
        not in {
            "data",
            "validation_data",
            "output",
            "prepare_only",
            "resume",
            "checkpoint_interval_seconds",
            "teacher_checkpoint",
        }
    }


def main() -> None:
    args = build_parser().parse_args()
    config = UnoConfig(rank=args.rank, alpha=args.alpha)
    if args.warmup_tokens is None:
        args.warmup_tokens = int(0.02 * sum(tokens for _, tokens in args.curriculum))
    if (
        min(
            args.sequence_length,
            args.batch_size,
            args.gradient_accumulation_steps,
            args.head_chunk_size,
            args.validation_every,
            args.validation_batches,
        )
        < 1
    ):
        raise ValueError(
            "sequence, batch, accumulation, chunk and validation settings must be positive"
        )
    if (
        not 0 < args.heldout_fraction < 1
        or not math.isfinite(args.lr)
        or args.lr <= 0
        or args.warmup_tokens < 0
    ):
        raise ValueError("invalid heldout fraction, learning rate or warmup budget")
    if any(args.sequence_length % block for block, _ in args.curriculum):
        raise ValueError(
            "sequence length must be divisible by each curriculum block size"
        )
    if len(args.revision) != 40 or any(
        c not in "0123456789abcdef" for c in args.revision
    ):
        raise ValueError("--revision must be an immutable 40-character model commit")
    checkpoint_policy = RecoveryCheckpointPolicy(args.checkpoint_interval_seconds)
    prepare_text_only_transformers_runtime()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    manifest, corpus_path = prepare_corpus(args, tokenizer)
    print(
        json.dumps(
            {
                "corpus": str(corpus_path),
                "corpus_sha256": manifest["corpus_sha256"],
                "train_tokens": manifest["train"]["tokens"],
                "validation_tokens": manifest["validation"]["tokens"],
            }
        ),
        flush=True,
    )
    if args.prepare_only:
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Uno training requires a CUDA device with native bf16; no CPU/eager fallback"
        )
    device = torch.device("cuda", torch.cuda.current_device())
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, teacher_sha256 = load_teacher(args, device)
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match the model")
    if args.sequence_length > model.config.max_position_embeddings:
        raise ValueError("sequence length exceeds model's logical RoPE context")
    resume = None
    if args.resume:
        bank, resume = load_uno_adapter(
            args.resume, model, model_id=args.model, revision=args.revision
        )
        if bank.config != config or resume["teacher_sha256"] != teacher_sha256:
            raise ValueError("resume adapter configuration or teacher identity differs")
        if resume["training"].get("corpus_sha256") != manifest[
            "corpus_sha256"
        ] or resume["training"].get("config") != training_config(args):
            raise ValueError(
                "resume corpus or immutable training configuration differs"
            )
        if "recovery" not in resume:
            raise ValueError("adapter export has no exact training recovery state")
    else:
        if (Path(args.output) / "latest.pt").exists() or (
            Path(args.output) / "metrics.jsonl"
        ).exists():
            raise ValueError(
                "output already contains training; specify --resume or a new output"
            )
        bank = UnoAdapterBank(model, config)
    router = attach_uno_adapters(model, bank)
    enable_uno_checkpointing(model, router)
    # Only native FlexAttention is compiled by HF's WrappedFlexAttention. The
    # checkpoint context is intentionally outside whole-model Dynamo tracing.
    model.train()
    adapter_ids = {id(parameter) for parameter in bank.parameters()}
    if {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    } != adapter_ids:
        raise RuntimeError("only diffusion adapter parameters may be trainable")
    optimizer = torch.optim.AdamW(
        bank.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True
    )
    trained_tokens = step = cursor = 0
    recovery = None
    if resume is not None:
        recovery = resume["recovery"]
        optimizer.load_state_dict(recovery["optimizer"])
        trained_tokens, step, cursor = (
            resume["trained_tokens"],
            resume["step"],
            recovery["cursor"],
        )
        if (
            type(cursor) is not int
            or cursor < step * args.batch_size
            or trained_tokens > sum(tokens for _, tokens in args.curriculum)
        ):
            raise ValueError("resume data cursor or trained-token budget is invalid")
        del resume
    train_stream = WindowStream(
        corpus_path / "train.bin", args.sequence_length, args.seed
    )
    validation_stream = WindowStream(
        corpus_path / "validation.bin", args.sequence_length, args.seed + 1
    )
    masks = {
        block: make_uno_block_mask(args.sequence_length, block, device)
        for block, _ in args.curriculum
    }
    output = Path(args.output)
    from torch.utils.tensorboard import SummaryWriter

    metrics_path = output / "metrics.jsonl"
    if args.resume and metrics_path.exists():
        staged = output / ".metrics.working"
        with metrics_path.open() as old, staged.open("w") as new:
            for line in old:
                # A process killed during flush may leave an incomplete tail.
                if not line.endswith("\n"):
                    break
                record = json.loads(line)
                if record["step"] <= step:
                    new.write(line)
        os.replace(staged, metrics_path)
    writer = SummaryWriter(
        str(output / "tensorboard"), purge_step=step + 1 if args.resume else None
    )
    metrics_stream = metrics_path.open("a")
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    training = {
        "corpus_sha256": manifest["corpus_sha256"],
        "config": training_config(args),
        "objective": "full_vocab_probability_l1",
        "noise": "uniform_full_vocabulary",
        "attention": "compiled_torch_flex_attention",
        "dtype": "bf16_fp32_adapter_master",
        "source": "ifm-ai/uno@46fbdb66f026bae9c68a1e5a3f97a17c7805c778",
    }

    def save(path):
        payload = uno_checkpoint_payload(
            bank,
            model,
            model_id=args.model,
            revision=args.revision,
            trained_tokens=trained_tokens,
            step=step,
            teacher_sha256=teacher_sha256,
            training=training,
        )
        payload["recovery"] = {
            "optimizer": optimizer.state_dict(),
            "cursor": cursor,
            "python_rng": random.getstate(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        atomic_torch_save(payload, path)
        checkpoint_policy.committed((step, trained_tokens))

    total_budget = sum(tokens for _, tokens in args.curriculum)
    if recovery is not None:
        random.setstate(recovery["python_rng"])
        torch.set_rng_state(recovery["cpu_rng"])
        torch.cuda.set_rng_state_all(recovery["cuda_rng"])
    try:
        while trained_tokens < total_budget and not stop_requested:
            endpoint = 0
            block = args.curriculum[0][0]
            for block, budget in args.curriculum:
                endpoint += budget
                if trained_tokens < endpoint:
                    break
            microbatch_tokens = args.batch_size * args.sequence_length
            effective_tokens = min(
                microbatch_tokens * args.gradient_accumulation_steps,
                endpoint - trained_tokens,
            )
            microbatch_count = math.ceil(effective_tokens / microbatch_tokens)
            lr = args.lr * min(
                1.0, (trained_tokens + effective_tokens) / max(1, args.warmup_tokens)
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.monotonic()
            loss = torch.zeros((), device=device)
            for microbatch_index in range(microbatch_count):
                supervised_tokens = min(
                    microbatch_tokens,
                    effective_tokens - microbatch_index * microbatch_tokens,
                )
                batch = train_stream.batch(
                    cursor + microbatch_index * args.batch_size, args.batch_size, device
                )
                weights = (
                    (torch.arange(microbatch_tokens, device=device) < supervised_tokens)
                    .float()
                    .reshape_as(batch)
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    microbatch_loss = uno_distillation_loss(
                        model,
                        router,
                        batch,
                        masks[block],
                        chunk_size=args.head_chunk_size,
                        token_weights=weights,
                    )
                    weighted_loss = microbatch_loss * (
                        supervised_tokens / effective_tokens
                    )
                loss.add_(weighted_loss.detach())
                weighted_loss.backward()
            grad_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        torch.linalg.vector_norm(parameter.grad)
                        for parameter in bank.parameters()
                        if parameter.grad is not None
                    ]
                )
            )
            if not bool(torch.isfinite(loss) & torch.isfinite(grad_norm)):
                raise FloatingPointError(
                    "nonfinite Uno objective or adapter gradient; optimizer not advanced"
                )
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            step += 1
            cursor += args.batch_size * microbatch_count
            trained_tokens += effective_tokens
            metrics = {
                "step": step,
                "trained_tokens": trained_tokens,
                "train/l1": loss.item(),
                "train/tv": loss.item() / 2,
                "train/grad_norm": grad_norm.item(),
                "train/lr": lr,
                "train/block_size": block,
                "train/tokens_per_second": effective_tokens / elapsed,
                "train/step_seconds": elapsed,
            }
            if (
                step % args.validation_every == 0
                or trained_tokens == endpoint
                or stop_requested
            ):
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + 10_000
                )
                # Keep train mode for one compiled attention kernel; no-grad
                # disables checkpointing and there is no dropout to randomize.
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    validation = torch.zeros((), device=device)
                    for index in range(args.validation_batches):
                        heldout = validation_stream.batch(
                            index * args.batch_size, args.batch_size, device
                        )
                        validation += uno_distillation_loss(
                            model,
                            router,
                            heldout,
                            masks[block],
                            chunk_size=args.head_chunk_size,
                            generator=generator,
                        )
                    metrics["validation/l1"] = (
                        validation / args.validation_batches
                    ).item()
                    metrics["validation/tv"] = metrics["validation/l1"] / 2
            metrics_stream.write(json.dumps(metrics, sort_keys=True) + "\n")
            metrics_stream.flush()
            for name, value in metrics.items():
                if name != "step":
                    writer.add_scalar(name, value, step)
            writer.flush()
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if checkpoint_policy.due() or trained_tokens == endpoint:
                save(output / "latest.pt")
        if step and checkpoint_policy.terminal_due((step, trained_tokens)):
            save(output / "latest.pt")
        if trained_tokens == total_budget:
            export = uno_checkpoint_payload(
                bank,
                model,
                model_id=args.model,
                revision=args.revision,
                trained_tokens=trained_tokens,
                step=step,
                teacher_sha256=teacher_sha256,
                training=training,
            )
            atomic_torch_save(export, output / "adapter.pt")
    finally:
        router.close()
        metrics_stream.close()
        writer.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
