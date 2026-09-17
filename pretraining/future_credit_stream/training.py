"""Teacher-forced streaming FFNs with an online scalar TD value critic.

One combined backward trains current CE, differentiates the frozen critic into
the current producer, and fits the previous carried state's value to the newly
observed transition. No temporal graph, incoming-state VJP, or lookahead is used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import sentencepiece as spm
import torch
import torch.nn.functional as F

from pretraining.future_credit_stream.config import ARCHITECTURE, Config, REPO_ROOT
from pretraining.future_credit_stream.data import DocumentIndex, StreamingDocuments
from pretraining.future_credit_stream.model import StreamingFFNModel
from pretraining.future_credit_stream.objective import StreamingObjective
from train_gpt import build_sentencepiece_luts


class PrefetchedDocuments:
    """One CPU page ahead; checkpoints record the consumed, not speculative, cursor."""

    def __init__(self, stream: StreamingDocuments, steps: int):
        self.stream = stream
        self.steps = steps
        self.committed_state = stream.state_dict()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ffn-documents")
        self.future: Future = self.executor.submit(self._read)

    def _read(self):
        page = self.stream.next_chunk(self.steps)
        return page, self.stream.state_dict()

    def next(self, device: torch.device):
        start = time.perf_counter()
        page, self.committed_state = self.future.result()
        self.future = self.executor.submit(self._read)
        tensors = {
            key: torch.from_numpy(value).pin_memory().to(device, non_blocking=True)
            for key, value in page.items()
        }
        return tensors, time.perf_counter() - start

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


class MetricLog:
    def __init__(self, config: Config):
        import os
        self.writer = None
        if os.environ.get("ABLATION_RUNNER_OWNS_METRICS") != "1":
            from scripts.ablation import MetricsWriter
            self.writer = MetricsWriter(config.output_dir / "metrics.jsonl", config.run_id)

    def emit(self, line: str):
        print(line, flush=True)
        if self.writer is not None:
            from scripts.ablation import parse_log_line
            entry = parse_log_line(line)
            if entry is not None:
                if entry.get("type") == "metric_error":
                    raise RuntimeError(entry)
                self.writer.write_entry(entry)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes() -> dict:
    files = [Path(__file__)] + [Path(__file__).with_name(name) for name in
                              ("model.py", "objective.py", "data.py", "config.py")]
    files += [REPO_ROOT / "pretraining/nanogpt_mini/nanogpt_mini_model.py",
              REPO_ROOT / "pretraining/nextlat.py", REPO_ROOT / "train_gpt.py"]
    return {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def setup_device(config: Config) -> torch.device:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA GPU required; no CPU or eager model fallback")
    torch.set_num_threads(config.cpu_threads)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)
    return torch.device("cuda", 0)


class Engine:
    def __init__(self, model: StreamingFFNModel, config: Config, bos_id: int):
        self.model, self.config, self.bos_id = model, config, bos_id
        self.has_previous = False
        compile_options = dict(fullgraph=True, dynamic=False, mode=config.compile_mode)
        self.forward = torch.compile(model.forward, **compile_options)
        objective = StreamingObjective(model, config.discount, bos_id)
        self.training_forward = torch.compile(objective.forward, **compile_options)

    def backward_page(self, page: dict, carry: torch.Tensor) -> torch.Tensor:
        """Train observed transitions with one forward/backward per token.

        Carry is an external, detached buffer. The previous TD feature is that
        same buffer before document resets, not another state or saved graph.
        The first observation has no predecessor; page boundaries do not reset
        this fact. Resume derives it from the completed optimizer-step count.
        """
        steps = page["inputs"].shape[0]
        statistics = torch.zeros(4, device=carry.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for offset in range(steps):
                torch.compiler.cudagraph_mark_step_begin()
                loss, hidden, stats = self.training_forward(
                    page["inputs"][offset], carry, page["targets"][offset],
                    page["resets"][offset], self.has_previous,
                )
                (loss / steps).backward()
                statistics.add_(stats)
                carry.copy_(hidden.detach())
                self.has_previous = True
                del loss, hidden, stats
        return statistics


@torch.no_grad()
def evaluate(engine: Engine, page: dict, byte_luts: tuple) -> dict:
    """Exact causal codelength for the operational recurrent categorical policy."""
    model = engine.model
    model.eval()
    device = page["inputs"].device
    batch = page["inputs"].shape[1]
    carry = model.initial_state(batch, device)
    empty_carry = model.initial_state(batch, device)
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    base_bytes, leading_space, boundary = byte_luts
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for offset in range(page["inputs"].shape[0]):
            observed, target = page["inputs"][offset], page["targets"][offset]
            carry.masked_fill_(page["resets"][offset, :, None], 0)
            torch.compiler.cudagraph_mark_step_begin()
            logits, hidden = engine.forward(observed, carry)
            totals[0].add_(F.cross_entropy(logits, target, reduction="sum"))
            carry.copy_(hidden)
            del logits, hidden
            torch.compiler.cudagraph_mark_step_begin()
            logits, hidden = engine.forward(observed, empty_carry)
            totals[1].add_(F.cross_entropy(logits, target, reduction="sum"))
            del logits, hidden
            byte_count = base_bytes[target] + (leading_space[target] & ~boundary[observed]).to(torch.int16)
            totals[2].add_(byte_count.sum())
    loss_sum, reset_sum, byte_count = totals.tolist()
    count = page["targets"].numel()
    if byte_count <= 0 or not all(math.isfinite(value) for value in (loss_sum, reset_sum)):
        raise FloatingPointError("non-finite validation loss or empty byte denominator")
    denominator = math.log(2) * byte_count
    return {
        "val_loss": loss_sum / count,
        "val_bpb": loss_sum / denominator,
        "val_proxy_bpb": loss_sum / denominator,
        "val_bpb_reset_latent": reset_sum / denominator,
        "val_memory_gain_bpb": (reset_sum - loss_sum) / denominator,
        "val_tokens": count, "val_bytes": byte_count,
    }


def checkpoint(path: Path, *, engine: Engine, optimizer, carry, stream_state,
               step: int, train_seconds: float, best_bpb: float, metadata: dict):
    value = {
        "architecture": ARCHITECTURE, "config": asdict(engine.config),
        "model_config": engine.model.config, "model": engine.model.state_dict(),
        "optimizer": optimizer.state_dict(), "carry": carry.detach(),
        "stream": stream_state,
        "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state(),
        "step": step, "train_seconds": train_seconds, "best_bpb": best_bpb,
        "metadata": metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, config: Config) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint is not a streaming future-credit FFN model")
    previous, current = dict(value["config"]), asdict(config)
    previous.pop("run_id")
    current.pop("run_id")
    if previous != current:
        raise ValueError("resume configuration differs from saved training contract")
    if value["metadata"]["sources"] != source_hashes():
        raise ValueError("source code changed since checkpoint; exact resume refused")
    return value


def train(config: Config, resume: Path | None = None):
    device = setup_device(config)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and (output / "checkpoint.pt").exists():
        raise FileExistsError("checkpoint exists: choose a new RUN_ID or explicitly --resume")
    tokenizer_file = config.path(config.tokenizer_path)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_file))
    if tokenizer.vocab_size() != config.vocab_size or tokenizer.bos_id() < 0:
        raise ValueError("tokenizer vocabulary/BOS does not match configured stream")
    data = config.path(config.data_path)
    cache = data / ".nextlat_stream_index"
    index_started = time.perf_counter()
    print("Indexing complete documents with bounded shard reads...", flush=True)
    training_index = DocumentIndex(str(data / "fineweb_train_*.bin"), tokenizer.bos_id(), config.vocab_size, cache)
    validation_index = DocumentIndex(str(data / "fineweb_val_*.bin"), tokenizer.bos_id(), config.vocab_size, cache)
    print(f"document_index_seconds:{time.perf_counter() - index_started:.3f}", flush=True)
    stream = StreamingDocuments(training_index, config.document_batch, config.seed, shuffle=True)
    validation_stream = StreamingDocuments(validation_index, config.val_documents, config.seed + 1, shuffle=False)
    validation_page = {key: torch.from_numpy(value).to(device) for key, value in
                       validation_stream.next_chunk(config.val_stream_steps).items()}
    byte_luts = build_sentencepiece_luts(tokenizer, config.vocab_size, device)
    model = StreamingFFNModel(**config.model_config).to(device)
    engine = Engine(model, config, tokenizer.bos_id())
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": config.weight_decay},
         {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}],
        lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    # Keep accumulated gradients outside the CUDA-graph pool across replays.
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    carry = model.initial_state(config.document_batch, device)
    metadata = {
        "architecture": ARCHITECTURE, "config": asdict(config), "sources": source_hashes(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        "train_data": training_index.describe(), "val_data": validation_index.describe(),
        "train_fingerprint": training_index.fingerprint, "val_fingerprint": validation_index.fingerprint,
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
        "prediction_contract": "teacher-forced next-token CE; actual detached recurrent hidden; scalar TD critic is training-only and never sees future tokens",
        "evaluation_scope": "deterministic partial-document panel; not full challenge BPB or old nanoGPT validation window",
        "state_contract": "one vector/document injected before first FFN; detach every token; BOS resets; no attention/history/slots/noise",
        "objective": (
            "current CE plus discounted frozen-critic value into producer, plus previous-carry scalar TD regression; one combined backward per token"
            if model.critic is not None else "ordinary next-token CE with detached actual carry"
        ),
        "target_contract": (
            "previous carry predicts current per-example CE + discount * current scalar value, detached; document-reset targets zero, current-terminal future value zero"
            if model.critic is not None else "ground-truth next token"
        ),
        "page_boundary_contract": (
            "online TD continues across optimizer pages using the existing detached carry; first observation has no predecessor; no lookahead or pending graph"
            if model.critic is not None else "all local CE graphs consumed before update; no boundary lookahead"
        ),
        "optimizer": "fused AdamW beta=(0.9,0.95), not nanoGPT baseline Muon",
        "autocull": "disabled for first matched reference; mlq owns hard wall-clock limit",
    }
    start_step, train_seconds, best_bpb = 0, 0.0, math.inf
    if resume is not None:
        saved = load_checkpoint(resume, config)
        for key in ("train_fingerprint", "val_fingerprint", "tokenizer_sha256"):
            if saved["metadata"][key] != metadata[key]:
                raise ValueError(f"resume provenance mismatch: {key}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        carry.copy_(saved["carry"].to(device))
        stream.load_state_dict(saved["stream"])
        torch.set_rng_state(saved["torch_rng_state"])
        torch.cuda.set_rng_state(saved["cuda_rng_state"])
        start_step, train_seconds, best_bpb = saved["step"], saved["train_seconds"], saved["best_bpb"]
        engine.has_previous = saved["step"] > 0
        del saved
    atomic_json(output / "run_config.json", metadata)
    print(json.dumps({"parameters": metadata["parameters"], "document_batch": config.document_batch,
                      "tokens_per_step": config.tokens_per_step, "state_updates_per_token": 1,
                      "state_mib": carry.numel() * carry.element_size() / 2**20,
                      "train_documents": training_index.document_count,
                      "compile_mode": config.compile_mode, "objective": config.objective}), flush=True)
    log = MetricLog(config)
    prefetch = PrefetchedDocuments(stream, config.stream_steps)
    accumulation = torch.zeros(4, device=device)
    interval_steps, interval_seconds, interval_wait = 0, 0.0, 0.0
    last_validation = None
    completed_step = start_step
    try:
        for step in range(start_step, config.iterations + 1):
            if step == start_step or step == config.iterations or step % config.val_every == 0:
                torch.cuda.synchronize()
                eval_start = time.perf_counter()
                last_validation = evaluate(engine, validation_page, byte_luts)
                torch.cuda.synchronize()
                eval_seconds = time.perf_counter() - eval_start
                improved = last_validation["val_bpb"] < best_bpb
                best_bpb = min(best_bpb, last_validation["val_bpb"])
                extras = " ".join(f"{key}:{value:.8f}" for key, value in last_validation.items()
                                  if key not in ("val_loss", "val_bpb"))
                log.emit(f"step:{step}/{config.iterations} val_loss:{last_validation['val_loss']:.8f} "
                         f"val_bpb:{last_validation['val_bpb']:.8f} train_time:{train_seconds * 1000:.3f}ms "
                         f"eval_seconds:{eval_seconds:.4f} {extras}")
                # A resumed checkpoint can precede this step's validation.
                # Publish its improvement too, excluding only untrained step 0.
                if improved and step > 0:
                    checkpoint(output / "best.pt", engine=engine, optimizer=optimizer, carry=carry,
                               stream_state=prefetch.committed_state, step=step,
                               train_seconds=train_seconds, best_bpb=best_bpb, metadata=metadata)
                if step == config.iterations:
                    break
            model.train()
            torch.cuda.synchronize()
            started = time.perf_counter()
            page, wait_seconds = prefetch.next(device)
            optimizer.zero_grad(set_to_none=False)
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate_at(step)
            accumulation.add_(engine.backward_page(page, carry))
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            train_seconds += elapsed
            interval_seconds += elapsed
            interval_wait += wait_seconds
            interval_steps += 1
            completed_step = step + 1
            if completed_step % config.log_every == 0 or completed_step == config.iterations:
                values = accumulation.tolist()
                calls = interval_steps * config.stream_steps
                ce, td_loss, value_mean, target_mean = [value / calls for value in values]
                if not all(math.isfinite(value) for value in (ce, td_loss, value_mean, target_mean)):
                    raise FloatingPointError("non-finite streaming TD value")
                token_count = interval_steps * config.tokens_per_step
                log.emit(f"step:{completed_step}/{config.iterations} train_loss:{ce:.8f} "
                         f"train_time:{train_seconds * 1000:.3f}ms "
                         f"td_loss:{td_loss:.10g} td_value:{value_mean:.10g} "
                         f"td_target:{target_mean:.10g} "
                         f"tokens_per_second:{token_count / interval_seconds:.2f} "
                         f"step_avg_ms:{interval_seconds * 1000 / interval_steps:.3f} "
                         f"data_wait_ms:{interval_wait * 1000 / interval_steps:.3f} "
                         f"tokens_seen:{completed_step * config.tokens_per_step} "
                         f"state_updates:{completed_step * config.stream_steps} "
                         f"peak_vram_allocated_mib:{torch.cuda.max_memory_allocated() / 2**20:.2f} "
                         f"peak_vram_reserved_mib:{torch.cuda.max_memory_reserved() / 2**20:.2f}")
                accumulation.zero_()
                interval_steps, interval_seconds, interval_wait = 0, 0.0, 0.0
            if completed_step % config.checkpoint_every == 0 or completed_step == config.iterations:
                checkpoint(output / "checkpoint.pt", engine=engine, optimizer=optimizer, carry=carry,
                           stream_state=prefetch.committed_state, step=completed_step,
                           train_seconds=train_seconds, best_bpb=best_bpb, metadata=metadata)
        checkpoint(output / "checkpoint.pt", engine=engine, optimizer=optimizer, carry=carry,
                   stream_state=prefetch.committed_state, step=completed_step,
                   train_seconds=train_seconds, best_bpb=best_bpb, metadata=metadata)
        atomic_json(output / "result.json", {
            "name": config.run_id, "status": "completed", "returncode": 0,
            "steps": completed_step, "tokens_seen": completed_step * config.tokens_per_step,
            "training_seconds": train_seconds, "best_val_bpb": best_bpb,
            "final_val_bpb": None, "final_proxy_val_bpb": last_validation["val_bpb"],
            "final_val_loss": last_validation["val_loss"], "promotion_metric": "proxy_bpb",
            "validation": last_validation, "checkpoint": str(output / "checkpoint.pt"),
        })
    finally:
        prefetch.close()
        log.close()


@torch.no_grad()
def generate(checkpoint_path: Path, prompt: str, max_new_tokens: int):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved.get("architecture") != ARCHITECTURE:
        raise ValueError("not a streaming future-credit FFN checkpoint")
    config = Config(**saved["config"])
    device = setup_device(config)
    tokenizer = spm.SentencePieceProcessor(model_file=str(config.path(config.tokenizer_path)))
    if hashlib.sha256(config.path(config.tokenizer_path).read_bytes()).hexdigest() != saved["metadata"]["tokenizer_sha256"]:
        raise ValueError("checkpoint tokenizer changed")
    model = StreamingFFNModel(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    engine = Engine(model, config, tokenizer.bos_id())
    carry = model.initial_state(1, device)
    sampling_rng = torch.Generator(device=device).manual_seed(config.seed + 4)
    token = torch.tensor([tokenizer.bos_id()], device=device)
    prompt_tokens = tokenizer.encode(prompt, out_type=int)
    output = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for value in prompt_tokens:
            torch.compiler.cudagraph_mark_step_begin()
            _, hidden = engine.forward(token, carry)
            carry.copy_(hidden)
            token.fill_(value)
        for _ in range(max_new_tokens):
            torch.compiler.cudagraph_mark_step_begin()
            logits, hidden = engine.forward(token, carry)
            target = torch.multinomial(logits.softmax(-1), 1, generator=sampling_rng).squeeze(1)
            value = int(target.item())
            if value in (tokenizer.bos_id(), tokenizer.eos_id()):
                break
            carry.copy_(hidden)
            token.copy_(target)
            output.append(value)
    print(tokenizer.decode(prompt_tokens + output), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Streaming future-credit FFN pretraining (run through mlq)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path)
    group.add_argument("--generate", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--prompt", default="The reason")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.generate:
        if args.max_new_tokens < 0:
            parser.error("--max-new-tokens must be nonnegative")
        generate(args.generate, args.prompt, args.max_new_tokens)
    else:
        train(Config.from_env(), args.resume)


if __name__ == "__main__":
    main()
