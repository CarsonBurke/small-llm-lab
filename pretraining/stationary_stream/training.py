"""Teacher-forced streaming FFNs over a stationary buffer of detached hiddens.

Every tick is its own graph: the current CE trains the whole model, the
tick's post-final-norm hidden is detached into the newest buffer slot, and
the oldest slot falls off. No gradient crosses a tick and nothing judges the
buffer. The data pipeline, optimizer, schedule, validation panel, and
checkpoint contract are those of ``future_credit_stream``, so the CE
recursion arm of that line is a matched control: same seed, data order,
tokenizer, and (verified by test) bitwise-identical backbone initialization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import sentencepiece as spm
import torch
import torch.nn.functional as F

from pretraining.future_credit_stream.data import DocumentIndex, StreamingDocuments
from pretraining.future_credit_stream.training import (
    MetricLog,
    PrefetchedDocuments,
    atomic_json,
    setup_device,
)
from pretraining.stationary_stream.config import ARCHITECTURE, Config, REPO_ROOT
from pretraining.stationary_stream.model import RingState, StationaryFFNModel
from pretraining.stationary_stream.objective import StationaryObjective
from train_gpt import build_sentencepiece_luts


def source_hashes() -> dict:
    files = [Path(__file__)] + [Path(__file__).with_name(name) for name in
                              ("model.py", "objective.py", "config.py")]
    files += [REPO_ROOT / "pretraining/future_credit_stream/data.py",
              REPO_ROOT / "pretraining/future_credit_stream/training.py",
              REPO_ROOT / "pretraining/nanogpt_mini/nanogpt_mini_model.py",
              REPO_ROOT / "pretraining/nextlat.py", REPO_ROOT / "train_gpt.py"]
    return {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def build_optimizer(model: StationaryFFNModel, config: Config) -> torch.optim.AdamW:
    """Fused AdamW over the whole model, one base rate; decay only weight matrices.

    The control decays every parameter with two or more dims, all of which are
    weight matrices there. Here the per-head recency bias table is also 2-D
    and must not be pulled toward zero, so the rule is stated as intended:
    weight matrices decay, biases and gains do not.
    """
    decayed, plain = [], []
    for name, parameter in model.named_parameters():
        (decayed if parameter.ndim >= 2 and name.endswith(".weight") else plain).append(parameter)
    groups = [
        {"params": decayed, "weight_decay": config.weight_decay, "base_lr": config.learning_rate},
        {"params": plain, "weight_decay": 0.0, "base_lr": config.learning_rate},
    ]
    for group in groups:
        group["lr"] = group["base_lr"]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, fused=True)


class Engine:
    def __init__(self, model: StationaryFFNModel, config: Config):
        self.model, self.config = model, config
        compile_options = dict(fullgraph=True, dynamic=False, mode=config.compile_mode)
        self.step = torch.compile(model.step, **compile_options)
        objective = StationaryObjective(model)
        self.statistics = objective.statistics
        self.training_forward = torch.compile(objective.forward, **compile_options)
        self._validation_state: RingState | None = None

    def validation_state(self, batch: int, device) -> RingState:
        """The one ring the compiled validation step is recorded against."""
        if self._validation_state is None or self._validation_state.buffer.shape[1] != batch:
            self._validation_state = self.model.initial_state(batch, device)
        return self._validation_state

    def backward_page(self, page: dict, state: RingState) -> torch.Tensor:
        """Train observed transitions; return statistics summed over ticks.

        ``state`` is the external, detached ring. Each tick is its own graph;
        the tick's hidden is committed into the ring after its backward.
        """
        steps = page["inputs"].shape[0]
        statistics = torch.zeros(len(self.statistics), device=state.buffer.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for offset in range(steps):
                torch.compiler.cudagraph_mark_step_begin()
                loss, hidden, next_valid, stats = self.training_forward(
                    page["inputs"][offset], state.buffer, state.valid, state.ages,
                    page["targets"][offset], page["resets"][offset],
                )
                (loss / steps).backward()
                statistics.add_(stats)
                # The recorded forward and backward graphs read the ring in
                # place (static addresses), so the commit must follow backward.
                state.commit(hidden, next_valid)
                del loss, hidden, next_valid, stats
        return statistics


@torch.no_grad()
def evaluate(engine: Engine, page: dict, byte_luts: tuple) -> dict:
    """Exact causal codelength for the operational streaming policy.

    ``val_bpb_reset_latent`` reads an always-empty buffer: the context-free
    FFN, the same reference CE recursion measured with a zero carry.
    """
    model = engine.model
    model.eval()
    device = page["inputs"].device
    batch = page["inputs"].shape[1]
    # One persistent validation ring per engine: the compiled step is recorded
    # against its static addresses, so fresh rings would force re-records.
    state = engine.validation_state(batch, device)
    state.reset()
    # The always-empty arm reads the same ring through an all-true reset mask,
    # which masks every slot exactly as an empty ring would.
    always_reset = torch.ones(batch, device=device, dtype=torch.bool)
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    base_bytes, leading_space, boundary = byte_luts
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for offset in range(page["inputs"].shape[0]):
            observed, target, resets = page["inputs"][offset], page["targets"][offset], page["resets"][offset]
            torch.compiler.cudagraph_mark_step_begin()
            logits, hidden, next_valid = engine.step(observed, state.buffer, state.valid, state.ages, resets)
            totals[0].add_(F.cross_entropy(logits, target, reduction="sum"))
            state.commit(hidden, next_valid)
            del logits, hidden, next_valid
            torch.compiler.cudagraph_mark_step_begin()
            logits, _, _ = engine.step(observed, state.buffer, state.valid, state.ages, always_reset)
            totals[1].add_(F.cross_entropy(logits, target, reduction="sum"))
            del logits
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


def checkpoint(path: Path, *, engine: Engine, optimizer, state: RingState, stream_state,
               step: int, train_seconds: float, best_bpb: float, metadata: dict):
    value = {
        "architecture": ARCHITECTURE, "config": asdict(engine.config),
        "model_config": engine.model.config, "model": engine.model.state_dict(),
        "optimizer": optimizer.state_dict(), "ring": state.state_dict(),
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
        raise ValueError("checkpoint is not a stationary-buffer streaming FFN model")
    previous, current = dict(value["config"]), asdict(config)
    previous.pop("run_id")
    current.pop("run_id")
    if previous != current:
        raise ValueError("resume configuration differs from saved training contract")
    if value["metadata"]["sources"] != source_hashes():
        raise ValueError("source code changed since checkpoint; exact resume refused")
    return value


def contracts(config: Config) -> dict:
    """Human-readable statements of what the run differentiates and carries."""
    return {
        "prediction_contract": ("teacher-forced next-token CE from the final FFN hidden; the context is one attention read "
                                f"over the last {config.buffer_slots} detached post-final-norm hiddens of the document"),
        "objective": "ordinary next-token CE; the read block trains through the current tick's CE only",
        "target_contract": "ground-truth next token",
        "state_contract": (f"ring of the last {config.buffer_slots} detached hiddens per lane, each written once into the "
                           "oldest slot, masked before the document's BOS; rotary slot ages on keys; "
                           f"{config.read_heads} heads x {config.read_key_dim} key dims; entry {config.read_entry}: "
                           + ("a learned null slot and a zero-initialized value projection after mixing"
                              if config.read_entry == "value_map" else
                              f"no null slot, zero-initialized query, learned recency bias at slope "
                              f"{config.read_recency_slope}, the control's latent_norm on the mixed vector")),
        "page_boundary_contract": "every tick is its own graph; no temporal gradient, no lookahead, no bootstrap",
    }


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
    model = StationaryFFNModel(**config.model_config).to(device)
    engine = Engine(model, config)
    optimizer = build_optimizer(model, config)
    # Keep accumulated gradients outside the CUDA-graph pool across replays.
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    state = model.initial_state(config.document_batch, device)
    metadata = {
        "architecture": ARCHITECTURE, "config": asdict(config), "sources": source_hashes(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        "train_data": training_index.describe(), "val_data": validation_index.describe(),
        "train_fingerprint": training_index.fingerprint, "val_fingerprint": validation_index.fingerprint,
        "parameters": sum(p.numel() for p in model.parameters()),
        "read_parameters": sum(p.numel() for p in model.read_parameters()),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
        "evaluation_scope": "deterministic partial-document panel; not full challenge BPB or old nanoGPT validation window",
        **contracts(config),
        "optimizer": "fused AdamW beta=(0.9,0.95), one base rate, decay on matrices only, not nanoGPT baseline Muon",
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
        state.load_state_dict(saved["ring"])
        stream.load_state_dict(saved["stream"])
        torch.set_rng_state(saved["torch_rng_state"])
        torch.cuda.set_rng_state(saved["cuda_rng_state"])
        start_step, train_seconds, best_bpb = saved["step"], saved["train_seconds"], saved["best_bpb"]
        del saved
    atomic_json(output / "run_config.json", metadata)
    print(json.dumps({"parameters": metadata["parameters"], "read_parameters": metadata["read_parameters"],
                      "document_batch": config.document_batch,
                      "tokens_per_step": config.tokens_per_step, "state_updates_per_token": 1,
                      "state_mib": state.buffer.numel() * state.buffer.element_size() / 2**20,
                      "train_documents": training_index.document_count,
                      "compile_mode": config.compile_mode, "buffer_slots": config.buffer_slots}), flush=True)
    log = MetricLog(config)
    prefetch = PrefetchedDocuments(stream, config.stream_steps, 0)
    accumulation = torch.zeros(len(engine.statistics), device=device)
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
                    checkpoint(output / "best.pt", engine=engine, optimizer=optimizer, state=state,
                               stream_state=prefetch.committed_state, step=step,
                               train_seconds=train_seconds, best_bpb=best_bpb, metadata=metadata)
                if step == config.iterations:
                    break
            model.train()
            torch.cuda.synchronize()
            started = time.perf_counter()
            page, wait_seconds = prefetch.next(device)
            optimizer.zero_grad(set_to_none=False)
            multiplier = config.schedule_at(step)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * multiplier
            accumulation.add_(engine.backward_page(page, state))
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            train_seconds += elapsed
            interval_seconds += elapsed
            interval_wait += wait_seconds
            interval_steps += 1
            completed_step = step + 1
            if completed_step % config.log_every == 0 or completed_step == config.iterations:
                calls = interval_steps * config.stream_steps
                values = dict(zip(engine.statistics, (value / calls for value in accumulation.tolist())))
                if not all(math.isfinite(value) for value in values.values()):
                    raise FloatingPointError("non-finite streaming statistic")
                diagnostics = " ".join(f"{key}:{value:.10g}" for key, value in values.items() if key != "ce")
                token_count = interval_steps * config.tokens_per_step
                log.emit(f"step:{completed_step}/{config.iterations} train_loss:{values['ce']:.8f} "
                         f"train_time:{train_seconds * 1000:.3f}ms "
                         + (diagnostics + " " if diagnostics else "")
                         + f"tokens_per_second:{token_count / interval_seconds:.2f} "
                         f"step_avg_ms:{interval_seconds * 1000 / interval_steps:.3f} "
                         f"data_wait_ms:{interval_wait * 1000 / interval_steps:.3f} "
                         f"tokens_seen:{completed_step * config.tokens_per_step} "
                         f"state_updates:{completed_step * config.stream_steps} "
                         f"peak_vram_allocated_mib:{torch.cuda.max_memory_allocated() / 2**20:.2f} "
                         f"peak_vram_reserved_mib:{torch.cuda.max_memory_reserved() / 2**20:.2f}")
                accumulation.zero_()
                interval_steps, interval_seconds, interval_wait = 0, 0.0, 0.0
            if completed_step % config.checkpoint_every == 0 or completed_step == config.iterations:
                checkpoint(output / "checkpoint.pt", engine=engine, optimizer=optimizer, state=state,
                           stream_state=prefetch.committed_state, step=completed_step,
                           train_seconds=train_seconds, best_bpb=best_bpb, metadata=metadata)
        checkpoint(output / "checkpoint.pt", engine=engine, optimizer=optimizer, state=state,
                   stream_state=prefetch.committed_state, step=completed_step,
                   train_seconds=train_seconds, best_bpb=best_bpb, metadata=metadata)
        atomic_json(output / "result.json", {
            "name": config.run_id, "status": "completed", "returncode": 0,
            "objective": "ce", "buffer_slots": config.buffer_slots,
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
        raise ValueError("not a stationary-buffer streaming FFN checkpoint")
    config = Config(**saved["config"])
    device = setup_device(config)
    tokenizer = spm.SentencePieceProcessor(model_file=str(config.path(config.tokenizer_path)))
    if hashlib.sha256(config.path(config.tokenizer_path).read_bytes()).hexdigest() != saved["metadata"]["tokenizer_sha256"]:
        raise ValueError("checkpoint tokenizer changed")
    model = StationaryFFNModel(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    engine = Engine(model, config)
    # One ring per generate call: a one-shot CLI path, recorded once.
    state = model.initial_state(1, device)
    sampling_rng = torch.Generator(device=device).manual_seed(config.seed + 4)
    token = torch.tensor([tokenizer.bos_id()], device=device)
    no_reset = torch.zeros(1, device=device, dtype=torch.bool)
    prompt_tokens = tokenizer.encode(prompt, out_type=int)
    output = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for value in prompt_tokens:
            torch.compiler.cudagraph_mark_step_begin()
            _, hidden, next_valid = engine.step(token, state.buffer, state.valid, state.ages, no_reset)
            state.commit(hidden, next_valid)
            token.fill_(value)
        for _ in range(max_new_tokens):
            torch.compiler.cudagraph_mark_step_begin()
            logits, hidden, next_valid = engine.step(token, state.buffer, state.valid, state.ages, no_reset)
            target = torch.multinomial(logits.softmax(-1), 1, generator=sampling_rng).squeeze(1)
            value = int(target.item())
            if value in (tokenizer.bos_id(), tokenizer.eos_id()):
                break
            state.commit(hidden, next_valid)
            token.copy_(target)
            output.append(value)
    print(tokenizer.decode(prompt_tokens + output), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Stationary-buffer streaming FFN pretraining (run through mlq)")
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
