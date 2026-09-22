"""Compiled/graphed cached inference comparison; execute only through mlq.

Decode measures one next token after a fixed prefix. Cache restoration occurs
outside timing for BOTH models; these are not growing autoregressive streams.
Prefill is measured separately and populates real inference caches. Random
nonzero weights exercise numerical parity, not language-model quality.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from fla.models.utils import Cache
from scripts.benchmark_chunk_memory import PlainMini
from scripts.train_recurrent_slots import atomic_json
from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaGPT
from pretraining.nanogpt_mini.gated_delta_runtime import gdn2_dependency_provenance, runtime_dependency_versions


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parity(actual, expected, label, tolerance=.02):
    if not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()):
        raise FloatingPointError(label)
    if float(expected.float().std()) < .001:
        raise ValueError(f"{label}: zero readout makes parity vacuous")
    relative = float((actual.double() - expected.double()).norm() / expected.double().norm())
    if relative > tolerance:
        raise AssertionError(f"{label}: relative norm error {relative} > {tolerance}")
    return relative


def rotate_at(x, rotary, position):
    theta = position * rotary.angular_freq[None, None, None, :]
    a, b = x.float().chunk(2, -1)
    return torch.cat((a * theta.cos() + b * theta.sin(),
                      -a * theta.sin() + b * theta.cos()), -1).type_as(x)


def frozen_gdn_projections(module, hidden):
    projected = F.linear(hidden, module._inference_packed_weight)
    return projected.split((module.key_dim, module.key_dim, module.value_dim,
                            module.key_dim, module.value_dim, module.head_v_dim,
                            module.head_v_dim), dim=-1)


def prepare_deployment(model, architecture):
    """Persistent BF16 dense parameters and immutable inference-only packing."""
    model.requires_grad_(False)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.endswith(("A_log", "dt_bias")):
                parameter.data = parameter.data.bfloat16()
        for block in model.blocks:
            attention = block.attn
            if architecture == "gdn2":
                weights = (attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight,
                           attention.b_proj.weight, attention.w_proj.weight,
                           attention.f_proj[0].weight, attention.g_proj[0].weight)
                attention.register_buffer("_inference_packed_weight", torch.cat(weights))
                attention._project_inputs = types.MethodType(frozen_gdn_projections, attention)
            else:
                projections = (attention.q, attention.k, attention.v)
                attention.register_buffer("_inference_packed_weight", torch.cat([p.weight for p in projections]))
                attention.register_buffer("_inference_packed_bias", torch.cat([p.bias for p in projections]))


class MiniInference:
    """Actual six-layer mini KV caching with unchanged attention mathematics."""

    def __init__(self, model):
        self.model = model
        self.prefill = torch.compile(self._prefill, fullgraph=True, dynamic=False)
        self.decode = torch.compile(self._decode, fullgraph=True, dynamic=False)

    def _prefill(self, tokens):
        model = self.model
        x = model.norm1(model.embed(tokens))
        states = []
        for block in model.blocks:
            attention = block.attn
            normalized = block.norm1(x)
            shape = (*tokens.shape, attention.num_heads, attention.head_dim)
            q, k, v = (tensor.view(shape) for tensor in
                       F.linear(normalized, attention._inference_packed_weight,
                                attention._inference_packed_bias).split(512, dim=-1))
            q, k = F.rms_norm(q, (128,)), F.rms_norm(k, (128,))
            q, k = attention.rotary(q).transpose(1, 2), attention.rotary(k).transpose(1, 2)
            v = v.transpose(1, 2).contiguous()
            # RMSNorm/RoPE retain FP32 here, but autocast SDPA consumes BF16.
            # Persist that effective attention input instead of a FP32 cache.
            k = k.to(torch.bfloat16).contiguous()
            output = F.scaled_dot_product_attention(q, k, v, scale=.12, is_causal=True)
            x = x + attention.proj(output.transpose(1, 2).contiguous().view(*tokens.shape, 512))
            x = x + block.mlp(block.norm2(x))
            states.extend((k, v))
        return model.logits(model.norm2(x[:, -1])), tuple(states)

    def _decode(self, tokens, states):
        model = self.model
        x = model.norm1(model.embed(tokens))[:, None]
        for index, block in enumerate(model.blocks):
            attention = block.attn
            normalized = block.norm1(x)
            shape = (tokens.shape[0], 1, 4, 128)
            q, k, v = (tensor.view(shape) for tensor in
                       F.linear(normalized, attention._inference_packed_weight,
                                attention._inference_packed_bias).split(512, dim=-1))
            q, k = F.rms_norm(q, (128,)), F.rms_norm(k, (128,))
            previous_k, previous_v = states[2 * index:2 * index + 2]
            position = previous_k.shape[2] - 1
            q = rotate_at(q, attention.rotary, position).transpose(1, 2)
            k = rotate_at(k, attention.rotary, position).transpose(1, 2)
            previous_k[:, :, -1:].copy_(k)
            previous_v[:, :, -1:].copy_(v.transpose(1, 2))
            output = F.scaled_dot_product_attention(q, previous_k, previous_v, scale=.12, is_causal=False)
            x = x + attention.proj(output.transpose(1, 2).contiguous().view(tokens.shape[0], 1, 512))
            x = x + block.mlp(block.norm2(x))
        return model.logits(model.norm2(x[:, 0]))


class GDNInference:
    def __init__(self, model, prefix):
        self.model, self.prefix = model, prefix
        # Only the installed intentional kernel boundaries may break graphs.
        self.prefill = torch.compile(self._prefill, fullgraph=False, dynamic=False)
        self.decode = torch.compile(self._decode, fullgraph=False, dynamic=False)

    def _prefill(self, tokens, cache):
        hidden, cache = self.model.forward_hidden(tokens, state=cache, use_cache=True)
        states = tuple(tensor for index in range(len(self.model.blocks))
                       for tensor in (cache[index]["recurrent_state"], *cache[index]["conv_state"]))
        return self.model.logits(hidden[:, -1]), states

    def _decode(self, tokens, cache):
        hidden, _ = self.model.forward_hidden(tokens[:, None], state=cache, use_cache=True)
        return self.model.logits(hidden[:, 0])

    def cache(self, states):
        cache = Cache()
        for index in range(len(self.model.blocks)):
            start = index * 4
            cache.update(recurrent_state=states[start], conv_state=tuple(states[start + 1:start + 4]),
                         layer_idx=index, offset=self.prefix)
        return cache


def audit_breaks():
    reasons = dict(torch._dynamo.utils.counters["graph_break"])
    allowed = ("chunk_gdn2", "fused_recurrent_gdn2_fwd", "causal_conv1d_fwd",
               "causal_conv1d_update", "layer_norm_gated_fwd")
    for reason in reasons:
        if not ("disable" in reason and any(name in reason for name in allowed)):
            raise RuntimeError(f"Unexpected inference graph break: {reason}")
    return reasons


class InferenceGraph:
    def __init__(self, function, reset):
        self.reset = reset
        current = torch.cuda.current_stream()
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            for _ in range(5):
                reset()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    function()
            reset()
        current.wait_stream(self.stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                self.output = function()
        current.wait_stream(self.stream)

    def replay(self):
        self.reset()
        self.graph.replay()
        return self.output

    def measure(self, repeats, tokens):
        for _ in range(5):
            self.replay()
        torch.cuda.synchronize()
        gpu_ms, wall_ms = [], []
        for _ in range(repeats):
            self.reset()
            torch.cuda.synchronize()  # restoration is outside both timers
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            begin.record()
            self.graph.replay()
            end.record()
            end.synchronize()
            gpu_ms.append(begin.elapsed_time(end))
            wall_ms.append((time.perf_counter() - started) * 1000)
        return dict(gpu_latency_ms=gpu_ms, median_gpu_latency_ms=statistics.median(gpu_ms),
                    host_synchronized_latency_ms=wall_ms, median_host_latency_ms=statistics.median(wall_ms),
                    tokens_per_second=tokens * 1000 / statistics.median(gpu_ms), warmups=5, repeats=repeats)


def benchmark_case(architecture, batch, prefix, repeats, checkpoint_path=None):
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    torch.manual_seed(173)
    if architecture == "gdn2":
        model = GatedDeltaGPT(gdn_backend="fla", state_v_first=False,
                              disable_recompute=True, fused_projections=True).cuda().eval()
    else:
        model = PlainMini().cuda().eval()
    weights = "random_nonzero_output_projections"
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["model_config"] != model.config:
            raise ValueError(f"Checkpoint configuration differs for {architecture}")
        model.load_state_dict(checkpoint["model"], strict=True)
        weights = dict(checkpoint=str(checkpoint_path), sha256=sha256(checkpoint_path))
    else:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name == "proj.weight" or name.endswith(("mlp.proj.weight", "attn.o_proj.weight", "attn.proj.weight")):
                    p.normal_(std=.003)
    tokens = torch.randint(1024, (batch, prefix + 1), device="cuda", dtype=torch.int32)
    prefix_tokens, next_tokens = tokens[:, :-1].contiguous(), tokens[:, -1].contiguous()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        original_hidden, _ = model.forward_hidden(tokens)
        original_precision_logits = model.logits(original_hidden[:, -1]).clone()
        del original_hidden
    prepare_deployment(model, architecture)
    adapter = GDNInference(model, prefix) if architecture == "gdn2" else MiniInference(model)
    prefill_call = (lambda: adapter.prefill(prefix_tokens, Cache())) if architecture == "gdn2" else (lambda: adapter.prefill(prefix_tokens))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        prefill_logits, original_states = prefill_call()
        prefix_hidden, _ = model.forward_hidden(prefix_tokens)
        prefill_error = parity(prefill_logits, model.logits(prefix_hidden[:, -1]), "prefill")
        hidden, _ = model.forward_hidden(tokens)
        expected = model.logits(hidden[:, -1]).clone()
        deployment_error = parity(expected, original_precision_logits, "persistent BF16 versus original parameters")
    if architecture == "plain_mini":
        original_states = tuple(F.pad(tensor, (0, 0, 0, 1)) for tensor in original_states)
        if any(tensor.dtype != torch.bfloat16 for tensor in original_states):
            raise AssertionError("Mini deployment requires BF16 keys and values")
    reference_states = tuple(tensor.clone() for tensor in original_states)
    states = tuple(tensor.clone() for tensor in original_states)

    def reset():
        for state, reference in zip(states, reference_states):
            state.copy_(reference)

    decode_call = (lambda: adapter.decode(next_tokens, adapter.cache(states))) if architecture == "gdn2" else (lambda: adapter.decode(next_tokens, states))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        reset()
        decoded = decode_call().clone()
        decode_error = parity(decoded, expected, "cached versus full sequence")
        for state in states:
            state.zero_()
        zero_cache_logits = decode_call().clone()
        cache_effect = float((zero_cache_logits.double() - expected.double()).norm() / expected.double().norm())
        if cache_effect <= max(2 * decode_error, 1e-5):
            raise AssertionError("Cache effect is too small relative to parity error; use a meaningful trained checkpoint")
        reset()
    torch.cuda.reset_peak_memory_stats()
    prefill_graph = InferenceGraph(prefill_call, lambda: None)
    decode_graph = InferenceGraph(decode_call, reset)
    for _ in range(3):
        parity(decode_graph.replay(), expected, "captured reset-and-replay", tolerance=.02)
    captured_normal = decode_graph.replay().clone()
    for state in states:
        state.zero_()
    decode_graph.graph.replay()
    parity(decode_graph.output, zero_cache_logits, "captured cache mutation", tolerance=.005)
    reference_delta = zero_cache_logits.double() - decoded.double()
    captured_delta = decode_graph.output.double() - captured_normal.double()
    mutation_error = float((captured_delta - reference_delta).norm() / reference_delta.norm().clamp_min(1e-12))
    if not math.isfinite(mutation_error) or mutation_error > .1:
        raise AssertionError(f"Captured cache-effect mismatch: relative difference error {mutation_error}")
    prefill_output = prefill_graph.replay()[0]
    parity(prefill_output, prefill_logits, "captured prefill", tolerance=.005)
    breaks = audit_breaks()
    result = dict(architecture=architecture, batch_size=batch, prefix_tokens=prefix,
                  model_config=model.config, parameters=sum(p.numel() for p in model.parameters()),
                  weights=weights,
                  deployment_precision="persistent_BF16_parameters_except_FP32_A_log_and_dt_bias",
                  projection_packing="immutable_GDN_seven_inputs_or_mini_QKV_prepacked_once",
                  packed_buffer_bytes=sum(t.numel() * t.element_size() for n, t in model.named_buffers()
                                          if "_inference_packed" in n),
                  parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                  prefill_parity_relative_error=prefill_error,
                  deployment_conversion_relative_error=deployment_error,
                  decode_parity_relative_error=decode_error, graph_breaks=breaks,
                  zero_cache_logit_relative_change=cache_effect,
                  cache_effect_exceeds_twice_parity_error=True,
                  captured_cache_effect_relative_error=mutation_error,
                  prefill=prefill_graph.measure(repeats, batch * prefix),
                  cached_next_token=decode_graph.measure(repeats, batch),
                  cache_tensor_bytes=sum(t.numel() * t.element_size() for t in states),
                  cache_bytes_per_document=sum(t.numel() * t.element_size() for t in states) // batch,
                  cache_storage_contract=("BF16_keys_and_values_after_RoPE" if architecture == "plain_mini"
                                          else "FP32_recurrence_and_BF16_short_convolution"),
                  cache_tensors=[dict(shape=list(t.shape), dtype=str(t.dtype), bytes=t.numel() * t.element_size()) for t in states],
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    if architecture == "gdn2":
        result["recurrent_state_bytes_per_document"] = sum(t.numel() * t.element_size() for t in states[::4]) // batch
        result["convolution_state_bytes_per_document"] = sum(t.numel() * t.element_size() for i, t in enumerate(states) if i % 4) // batch
    torch.cuda.synchronize()
    decode_graph.graph.reset()
    prefill_graph.graph.reset()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--prefixes", type=int, nargs="+", choices=(128, 1024, 4096), default=[128, 1024, 4096])
    parser.add_argument("--batches", type=int, nargs="+", choices=(1, 32), default=[1, 32])
    parser.add_argument("--gdn-checkpoint", type=Path)
    parser.add_argument("--mini-checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 5:
        parser.error("at least five repeats required")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required; submit through mlq")
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    # Six static cache layer indices specialize separately for prefill and
    # decode. Allow that bounded set; never fall back on a cache-limit hit.
    torch._dynamo.config.recompile_limit = 32
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT / "scripts/benchmark_chunk_memory.py", ROOT / "scripts/train_recurrent_slots.py"]
    paths += [ROOT / "pretraining/nanogpt_mini" / name for name in
              ("gated_delta_model.py", "gated_delta_runtime.py", "nanogpt_mini_model.py")]
    paths += [path for path in (ROOT / "pretraining/gated_delta/vendor").rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    report = dict(status="running", benchmark="fixed_prefix_cached_next_token_inference",
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  dependency_versions=runtime_dependency_versions(), gdn2_dependency_provenance=gdn2_dependency_provenance(),
                  cache_restoration_in_timing=False, sampling_and_host_transfers_in_timing=False,
                  full_vocabulary_logits_in_timing=True, contexts4096_are_untrained_extrapolation=True,
                  limitations="Stationary next-token latency, not growing-stream throughput; random weights unless checkpoints supplied; no quality claim.",
                  source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths}, cases=[])
    for path in paths:
        destination = args.output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    atomic_json(args.output / "benchmark.json", report)
    try:
        for batch in args.batches:
            for prefix in args.prefixes:
                for architecture in ("plain_mini", "gdn2"):
                    checkpoint = args.gdn_checkpoint if architecture == "gdn2" else args.mini_checkpoint
                    result = benchmark_case(architecture, batch, prefix, args.repeats, checkpoint)
                    report["cases"].append(result)
                    atomic_json(args.output / "benchmark.json", report)
                    print(json.dumps(result), flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(args.output / "benchmark.json", report)


if __name__ == "__main__":
    main()
