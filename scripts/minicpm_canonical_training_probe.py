#!/usr/bin/env python3
"""Queue-only MiniCPM forward/state/backward qualification, not a training run.

Run with mlq submit --max-parallel-runs 1 --cwd REPO -- python THIS --output FILE.
The forward uses fixed-tile rounded-BF16 probabilities. Its backward deliberately
uses the ideal-softmax derivative computed by native BF16 CUDA FLASH SDPA, NOT
the derivative of the custom rounded-probability/reduction algorithm. Projection
casts and canonical RMSNorm likewise use finite-precision surrogate derivatives.
No CPU/model, SDPA math, attention-mask, or compilation fallback is permitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton

from postraining.fast_inference import build_fused_rollout_replica, synchronize_fused_lora_policy_
from postraining.invariant_attention import INVARIANT_ATTENTION
from postraining.minicpm_latent_rollout import _LatentStateDecoder
from postraining.minicpm_vapo import LoRAConfig, LoRALinear, MiniCPMVAPOPolicy
from postraining.train_minicpm_vapo import encode_math_prompt
from scripts.minicpm_canonical_attention_probe import canonical_attention, KERNEL_METADATA
from scripts.minicpm_canonical_norm_probe import install_canonical_norms_
from scripts.minicpm_canonical_projection_probe import (
    canonical_merged_weight, install_canonical_lora_, install_canonical_merged_,
    synchronize_canonical_lora_,
)


def flash_reference(query, key, value, scale, query_start=0):
    """[B,Q,H,D], explicit contiguous causal suffix, FLASH-only BF16 operands.

    Native SDPA uses upper-left causality, so a nonzero absolute query offset
    must NOT use is_causal=True. Slice each row's live prefix instead. Full
    sequences use one FLASH call; no dense mask or math fallback is constructed.
    Repeat KV only because native FLASH backward requires equal head counts.
    """
    if any(not t.is_cuda or t.dtype != torch.bfloat16 for t in (query, key, value)):
        raise ValueError("FLASH surrogate requires CUDA bf16 Q/K/V")
    if query_start < 0 or query_start + query.shape[1] > key.shape[1]:
        raise ValueError("query suffix must lie inside the explicit live KV prefix")
    q = query.transpose(1, 2)
    groups = query.shape[2] // key.shape[2]
    k = key.transpose(1, 2).repeat_interleave(groups, dim=1)
    v = value.transpose(1, 2).repeat_interleave(groups, dim=1)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        if query_start == 0:
            result = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        else:
            result = torch.cat([
                F.scaled_dot_product_attention(
                    q[:, :, index:index + 1], k[:, :, :query_start + index + 1],
                    v[:, :, :query_start + index + 1], is_causal=False, scale=scale,
                ) for index in range(query.shape[1])
            ], dim=2)
    return result.transpose(1, 2).contiguous()


class _TrainableAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, scale, query_start):
        ctx.save_for_backward(query, key, value)
        ctx.scale, ctx.query_start = scale, query_start
        positions = torch.arange(query_start, query_start + query.shape[1], device=query.device)
        lengths = torch.full((query.shape[0],), key.shape[1], device=query.device, dtype=torch.int32)
        return canonical_attention(query, key, value,
            query_positions=positions[None].expand(query.shape[0], -1), key_lengths=lengths, scale=scale)

    @staticmethod
    def backward(ctx, gradient):
        # Recompute from Q/K/V, never a saved output/hidden override. autograd.grad
        # is intentionally first-order: FLASH does not offer double backward.
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            operands = [tensor.detach().requires_grad_(True) for tensor in ctx.saved_tensors]
            reference = flash_reference(*operands, ctx.scale, ctx.query_start)
            gradients = torch.autograd.grad(reference, operands, gradient.to(torch.bfloat16),
                                            create_graph=False)
        return *(g if needed else None for g, needed in zip(gradients, ctx.needs_input_grad[:3])), None, None


def trainable_canonical_attention(query, key, value, *, scale, query_start=0):
    """One contiguous live sequence; offsets are absolute relative to sliced KV."""
    if query_start < 0 or query_start + query.shape[1] > key.shape[1]:
        raise ValueError("invalid absolute causal query offset")
    return _TrainableAttention.apply(query, key, value, float(scale), query_start)


class AttentionRegistry:
    """Explicit host layout for full/ragged/packed and GPU lengths for real cache.

    Padding is never inferred by copying a GPU mask to the host. Full ragged
    input uses right-padding and explicit lengths; decoder prefill uses explicit
    left-padding lengths. Packed boundaries reset the KV origin for each segment.
    Arbitrary additive masks/prefix-LM bidirectionality are unsupported and fail.
    """
    full_name = "parameter_golf_canonical_training_probe"

    def __init__(self, canonical):
        self.canonical = canonical
        self.phase = "full"
        self.lengths = None
        self.boundaries = None
        self.live_length = 0
        self.calls = Counter()
        self.dtypes = set()

    def install(self):
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        for name in (self.full_name, "parameter_golf_fa4_decode", INVARIANT_ATTENTION):
            ALL_ATTENTION_FUNCTIONS.register(name, self.attention)
            ALL_MASK_ATTENTION_FUNCTIONS.register(name, self.mask)

    def mask(self, **kwargs):
        supplied = kwargs.get("attention_mask")
        if supplied is not None:
            if supplied.ndim != 2 or supplied.dtype != torch.bool or self.lengths is None:
                raise ValueError("mask requires explicit bool ragged/prefill layout lengths")
            if supplied.shape[0] != len(self.lengths):
                raise ValueError("mask batch and explicit lengths differ")
            positions = torch.arange(supplied.shape[1], device=supplied.device)[None]
            lengths = torch.tensor(self.lengths, device=supplied.device)[:, None]
            expected = (positions >= supplied.shape[1] - lengths if self.phase == "prefill"
                        else positions < lengths)
            torch._assert_async((supplied == expected).all(),
                                "mask does not match the explicit contiguous layout")
        # Attention below applies the host layout and GPU cached lengths. No
        # arbitrary caller mask may be silently discarded.
        return supplied

    def segment(self, q, k, v, scale):
        if self.canonical:
            return trainable_canonical_attention(q, k, v, scale=scale)
        return flash_reference(q, k, v, scale)

    def attention(self, module, query, key, value, attention_mask,
                  dropout=0.0, scaling=None, **kwargs):
        if dropout:
            raise ValueError("dropout is outside this deterministic probe")
        q, k, v = (tensor.transpose(1, 2) for tensor in (query, key, value))
        if any(t.dtype != torch.bfloat16 or not t.is_cuda for t in (q, k, v)):
            raise ValueError("all attention operands must remain CUDA bf16")
        # Counters are eager evidence only, never a graph-side mutation.
        if not torch.compiler.is_compiling() and not torch.cuda.is_current_stream_capturing():
            self.calls[self.phase] += 1
            self.dtypes.add(str(q.dtype))
        scale = q.shape[-1] ** -0.5 if scaling is None else float(scaling)
        if getattr(module, "_rollout_cached_append", False):
            if q.shape[1] != 1 or attention_mask is not None:
                raise ValueError("cache append needs one query and explicit per-lane live lengths")
            if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
                raise ValueError("cached autograd is not supported; train using full/packed replay")
            lengths = module._rollout_sequence_lengths
            if self.canonical:
                return canonical_attention(q, k, v, query_positions=(lengths.long() - 1)[:, None],
                                           key_lengths=lengths, scale=scale), None
            if q.shape[0] != 1 or self.live_length < 1:
                raise ValueError("native cached control needs batch1 explicit live prefix")
            return flash_reference(q, k[:, :self.live_length], v[:, :self.live_length],
                                   scale, self.live_length - 1), None
        boundaries = getattr(module, "_packed_sequence_boundaries", None) or self.boundaries
        if boundaries is not None:
            if attention_mask is not None or q.shape[0] != 1 or q.shape[1] != k.shape[1]:
                raise ValueError("packed attention requires one unpadded full sequence")
            if boundaries[0] != 0 or boundaries[-1] != q.shape[1] or any(
                    a >= b for a, b in zip(boundaries, boundaries[1:])):
                raise ValueError("invalid packed sequence boundaries")
            return torch.cat([self.segment(q[:, a:b], k[:, a:b], v[:, a:b], scale)
                              for a, b in zip(boundaries, boundaries[1:])], dim=1), None
        if self.lengths is None:
            if attention_mask is not None or q.shape[1] != k.shape[1]:
                raise ValueError("full attention needs an explicit ragged layout or equal Q/K")
            return self.segment(q, k, v, scale), None
        if len(self.lengths) != q.shape[0] or q.shape[1] != k.shape[1]:
            raise ValueError("ragged layout must describe every full-attention row")
        outputs = []
        for row, length in enumerate(self.lengths):
            if not 0 < length <= q.shape[1]:
                raise ValueError("invalid live length")
            # Prefill returns original padded Q/K/V while separately compacting
            # its backing storage. Only subsequent cached calls see compact KV.
            offset = q.shape[1] - length if self.phase == "prefill" else 0
            out = self.segment(q[row:row + 1, offset:offset + length],
                               k[row:row + 1, offset:offset + length],
                               v[row:row + 1, offset:offset + length], scale)
            outputs.append(F.pad(out, (0, 0, 0, 0, offset, q.shape[1] - length - offset)))
        return torch.cat(outputs), None


def set_full(policy, control, boundaries=None):
    control.phase, control.lengths, control.boundaries = "full", None, boundaries
    policy.causal_lm.config._attn_implementation = control.full_name
    for layer in policy.causal_lm.model.layers:
        layer.self_attn._rollout_cached_append = False
        layer.self_attn._packed_sequence_boundaries = boundaries


def full_states(policy, control, prompt, raw):
    set_full(policy, control)
    # Adapt the actual raw record in this path's actual batch shape. Saving or
    # forcing rollout embeddings here would hide adapter numerical divergence.
    embedded = policy.thought_embeddings(raw)
    inputs = torch.cat((policy.token_embeddings(prompt), embedded))[None]
    positions = torch.arange(inputs.shape[1], device=inputs.device)[None]
    hidden = policy.replay_hidden(None, None, inputs_embeds=inputs, position_ids=positions)
    return hidden[0, prompt.numel() - 1:], embedded


def prepare_decoder(policy, control, prompt, steps, compile_decode=False):
    control.install()
    control.phase, control.lengths, control.boundaries = "prefill", (prompt.numel(),), None
    for layer in policy.causal_lm.model.layers:
        layer.self_attn._packed_sequence_boundaries = None
    decoder = _LatentStateDecoder(policy, 1, prompt.numel() + steps + 1, compile_decode)
    try:
        decoder.prefill([prompt])
        for layer in decoder.cache.layers:
            layer.indexed_decode = True
        hidden = decoder.admit([0], [0])
        control.phase, control.lengths = "cached", None
        return decoder, hidden
    except BaseException:
        decoder.release_cache()
        raise


@torch.no_grad()
def cached_states(policy, control, prompt, raw=None, *, steps=None, seed=1337):
    count = raw.shape[0] if raw is not None else steps
    decoder, hidden = prepare_decoder(policy, control, prompt, count)
    active = torch.ones(1, device=prompt.device, dtype=torch.bool)
    generator = torch.Generator(device=prompt.device).manual_seed(seed)
    states, actions, embeddings = [hidden[0].clone()], [], []
    before = control.calls["cached"]
    try:
        for index in range(count):
            action = raw[index:index + 1] if raw is not None else policy.transition.sample_latent(
                policy.transition.predict_mean(hidden), policy.transition.predict_log_sigma(hidden), generator)
            embedded = policy.thought_embeddings(action)
            actions.append(action[0].clone())
            embeddings.append(embedded[0].clone())
            control.live_length = prompt.numel() + index + 1
            hidden = decoder.advance(embedded, active)
            states.append(hidden[0].clone())
        if control.calls["cached"] - before != count * len(policy.causal_lm.model.layers):
            raise RuntimeError("registered attention was bypassed during real cache append")
        return torch.stack(states), torch.stack(actions), torch.stack(embeddings)
    finally:
        decoder.release_cache()


@contextmanager
def canonical_models(source, replica):
    originals = [(module, module.forward) for policy in (source, replica)
                 for module in policy.causal_lm.modules()]
    try:
        installed = {"source_lora": install_canonical_lora_(source.causal_lm),
                     "replica_merged": install_canonical_merged_(replica.causal_lm),
                     "norm_counts": [install_canonical_norms_(p.causal_lm) for p in (source, replica)]}
        yield installed
    finally:
        for module, forward in originals:
            module.forward = forward


@torch.no_grad()
def sync_replica(source, replica, canonical):
    count = (synchronize_canonical_lora_(replica.causal_lm, source.causal_lm) if canonical else
             synchronize_fused_lora_policy_(replica, source))
    for name in ("transition", "thinking_gate", "thought_adapter"):
        getattr(replica, name).load_state_dict(getattr(source, name).state_dict(), strict=True)
    return count


def normalized_adapter_forward(module, base, raw):
    result = type(module).forward(module, base, raw)
    return (F.rms_norm(result.float(), (result.shape[-1],)) *
            (module._probe_output_norm / math.sqrt(result.shape[-1]))).to(result.dtype)


def differences(left, right):
    delta = left.detach().float() - right.detach().float()
    finite = bool(torch.isfinite(delta).all())
    return {"finite": finite, "exact": bool(torch.equal(left, right)),
            "max_abs": delta.abs().max().item() if finite else None,
            "l2": delta.norm().item() if finite else None,
            "relative_l2": (delta.norm() / right.detach().float().norm().clamp_min(1e-30)).item() if finite else None}


def mean_comparison(policy, left, right):
    a, b = (policy.transition.predict_mean(t).float() for t in (left, right))
    result = differences(a, b)
    kl = 0.5 * ((a - b) / policy.transition.component_std).square().sum(-1)
    result.update({"mean_kl": kl.mean().item() if result["finite"] else None,
                   "max_mean_kl": kl.max().item() if result["finite"] else None,
                   "per_position_mean_kl": kl.tolist() if result["finite"] else None})
    return result


def parameter_contract(policy):
    named = dict(policy.named_parameters())
    trainable = {name: id(p) for name, p in named.items() if p.requires_grad}
    base_trainable = [name for name, p in policy.causal_lm.named_parameters()
                      if p.requires_grad and not name.endswith(("lora_a", "lora_b"))]
    actor_ids = {id(p) for p in policy.actor_parameters()}
    missing = [name for name, p in named.items() if name.startswith("causal_lm.")
               and name.endswith(("lora_a", "lora_b")) and id(p) not in actor_ids]
    if base_trainable or missing:
        raise RuntimeError(f"frozen-base/actor optimizer contract broken: {base_trainable}, {missing}")
    return trainable


def gradient_summary(named):
    rows = {}
    for name, parameter in named:
        grad = parameter.grad
        finite = grad is not None and bool(torch.isfinite(grad).all())
        rows[name] = {"present": grad is not None, "finite": finite,
                      "dtype": str(grad.dtype) if grad is not None else None,
                      "norm": grad.double().norm().item() if finite else None}
    return {"all_present_finite": bool(rows) and all(row["finite"] for row in rows.values()),
            "any_nonzero": any(row["norm"] is not None and row["norm"] > 0 for row in rows.values()),
            "parameters": rows}


def backward_probe(source, control, prompt, raw):
    source.zero_grad(set_to_none=True)
    source.train()
    states, embeddings = full_states(source, control, prompt, raw)
    mean = source.transition.predict_mean(states[:-1])
    raw_record = raw.detach()
    logprobs = source.transition.log_prob(raw_record, mean, source.transition.predict_log_sigma(states[:-1]))
    # Actual Gaussian likelihood, no ratio cap, recentering, or hidden override.
    loss = -logprobs.mean()
    embeddings.retain_grad()
    mean.retain_grad()
    loss.backward()
    result = {"loss": loss.detach().item(), "logprob_finite": bool(torch.isfinite(logprobs).all()),
              "raw_action_dtype": str(raw_record.dtype), "raw_action_requires_grad": raw_record.requires_grad,
              "hidden_dtype": str(states.dtype), "adapted_input_dtype": str(embeddings.dtype),
              "mean_dtype": str(mean.dtype),
              "mean_gradient_finite": mean.grad is not None and bool(torch.isfinite(mean.grad).all()),
              "embedding_gradient_finite": embeddings.grad is not None and bool(torch.isfinite(embeddings.grad).all()),
              "lora": gradient_summary((n, p) for n, p in source.causal_lm.named_parameters()
                                       if n.endswith(("lora_a", "lora_b"))),
              "gaussian_head": gradient_summary(source.transition.named_parameters()),
              "adapter": gradient_summary(source.thought_adapter.named_parameters()),
              "frozen_base_gradients": [n for n, p in source.causal_lm.named_parameters()
                                         if not p.requires_grad and p.grad is not None]}
    source.zero_grad(set_to_none=True)
    source.eval()
    return result


def attention_gradient_probe():
    result = []
    for q_length, k_length, offset in ((33, 33, 0), (257, 257, 0), (3, 113, 110)):
        operands = [torch.randn((1, length, heads, 128), device="cuda", dtype=torch.bfloat16,
                                requires_grad=True)
                    for length, heads in ((q_length, 16), (k_length, 2), (k_length, 2))]
        gradient = torch.randn_like(operands[0])
        output = trainable_canonical_attention(*operands, scale=128 ** -0.5, query_start=offset)
        got = torch.autograd.grad(output, operands, gradient)
        reference = flash_reference(*operands, 128 ** -0.5, offset)
        expected = torch.autograd.grad(reference, operands, gradient)
        result.append({"q": q_length, "k": k_length, "absolute_query_start": offset,
                       "operand_dtype": str(operands[0].dtype), "forward_vs_native_flash": differences(output, reference),
                       "surrogate_gradient_vs_native_flash": {
                           name: differences(a, b) for name, a, b in zip(("q", "k", "v"), got, expected)}})
    return result


def packed_ragged_probe(source, control, prompt, raw):
    # Real model packed replay: unequal trajectories with independent RoPE origins.
    first = torch.cat((source.token_embeddings(prompt), source.thought_embeddings(raw[:3])))
    second = torch.cat((source.token_embeddings(prompt), source.thought_embeddings(raw[:7])))
    lengths = (first.shape[0], second.shape[0])
    boundaries = (0, lengths[0], sum(lengths))
    set_full(source, control, boundaries)
    positions = torch.cat([torch.arange(n, device=prompt.device) for n in lengths])[None]
    packed = source.replay_hidden(None, None, inputs_embeds=torch.cat((first, second))[None],
                                  position_ids=positions,
                                  cu_seqlens=torch.tensor(boundaries, device=prompt.device, dtype=torch.int32),
                                  sequence_boundaries=boundaries, max_sequence_length=max(lengths))[0]
    independent = []
    for inputs in (first, second):
        set_full(source, control)
        independent.append(source.replay_hidden(None, None, inputs_embeds=inputs[None],
            position_ids=torch.arange(inputs.shape[0], device=prompt.device)[None])[0])
    set_full(source, control)
    control.lengths = lengths
    width = max(lengths)
    inputs = torch.stack([F.pad(t, (0, 0, 0, width - t.shape[0])) for t in (first, second)])
    mask = torch.arange(width, device=prompt.device)[None] < torch.tensor(lengths, device=prompt.device)[:, None]
    ragged = source.replay_hidden(None, mask, inputs_embeds=inputs,
                                  position_ids=torch.arange(width, device=prompt.device)[None].expand(2, -1))
    control.lengths = None
    return {"lengths": lengths, "boundaries": boundaries,
            "packed_vs_independent": differences(packed, torch.cat(independent)),
            "right_padded_ragged_vs_independent": differences(
                torch.cat([ragged[i, :n] for i, n in enumerate(lengths)]), torch.cat(independent))}


def qualify_replica(replica, control, prompt, raw, *, compile_decode, capture):
    """Attempt actual fixed-cache decode compilation/capture; never eager fallback."""
    result = {"compile_requested": compile_decode, "capture_requested": capture, "status": "running"}
    decoder = None
    try:
        decoder, _ = prepare_decoder(replica, control, prompt, 8, compile_decode)
        embedded = replica.thought_embeddings(raw[:1])
        active = torch.ones(1, device=prompt.device, dtype=torch.bool)
        # Replay every qualification at the same real cache state. This resets KV
        # metadata, not hidden outputs; every measured hidden is recomputed.
        def reset():
            decoder.lengths.fill_(prompt.numel())
            decoder.flash_lengths.fill_(prompt.numel() + 1)
        reset()
        expected = decoder._advance_hidden(embedded, active).clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                reset()
                output = decoder.advance(embedded, active)
        torch.cuda.current_stream().wait_stream(stream)
        result["compiled_or_eager_vs_eager"] = differences(output, expected)
        if capture:
            reset()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = decoder.advance(embedded, active)
            reset()
            graph.replay()
            torch.cuda.synchronize()
            result["captured_vs_eager"] = differences(output, expected)
            reset()
            graph.replay()
            torch.cuda.synchronize()
            result["second_replay_vs_eager"] = differences(output, expected)
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if decoder is not None:
            decoder.release_cache()
    return result


def save_report(path, report):
    # Preserve failure evidence even when the failure itself is nonfinite.
    nonfinite = []
    def json_value(value, location):
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite.append(location)
            return None
        if isinstance(value, dict):
            return {key: json_value(item, f"{location}.{key}") for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [json_value(item, f"{location}[{index}]") for index, item in enumerate(value)]
        return value
    payload = json_value(report, "$")
    payload["nonfinite_json_fields"] = nonfinite
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--thought-lengths", type=int, nargs="+", default=[32, 256])
    p.add_argument("--prompt-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--thought-sigma", type=float, default=1.0)
    p.add_argument("--lora-change", type=float, default=1e-4,
                   help="known nonzero fp32 B increment; must not be zero")
    p.add_argument("--normalize-adapter-output", action="store_true")
    p.add_argument("--adapter-norm", type=float, default=1.067)
    p.add_argument("--compile-replica", action="store_true")
    p.add_argument("--capture-replica", action="store_true")
    return p


def main():
    args = parser().parse_args()
    sidecar = args.output.with_suffix(".streams.pt")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("report or stream sidecar already exists")
    if min(args.thought_lengths) < 8 or not math.isfinite(args.lora_change) or args.lora_change == 0:
        raise ValueError("thought lengths >=8 and finite nonzero LoRA change required")
    if not math.isfinite(args.adapter_norm) or args.adapter_norm <= 0:
        raise ValueError("adapter norm must be finite and positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__).relative_to(root), *map(Path, (
        "scripts/minicpm_canonical_attention_probe.py", "scripts/minicpm_canonical_projection_probe.py",
        "scripts/minicpm_canonical_norm_probe.py", "postraining/minicpm_vapo.py",
        "postraining/minicpm_latent_rollout.py", "postraining/fast_inference.py", "postraining/latent_thought.py"))]
    report = {"schema": "minicpm-canonical-training-probe/v1", "status": "initializing",
              "argv": sys.argv[1:], "sources": {str(p): hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files},
              "surrogate_gradient": "BF16 native FLASH ideal-softmax derivative, NOT rounded-P canonical derivative; first order only",
              "kernel": KERNEL_METADATA, "cases": [], "qualification": {"status": "not_requested"},
              "raw_action_contract": "stored FP32 raw actions are detached likelihood targets; actor adapts them afresh; critic must independently adapt the same raw actions with its own parameters, never actor embeddings",
              "critic_execution": "not instantiated: critic input contract is preserved, not qualified by this actor probe",
              "normalization": {"enabled": args.normalize_adapter_output, "output_l2_target": args.adapter_norm},
              "scope": "diagnostic eager source backward and real forced-cache parity, no optimizer training or learning claim"}
    save_report(args.output, report)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        torch.manual_seed(args.seed)
        torch.set_float32_matmul_precision("high")
        report["runtime"] = {"torch": torch.__version__, "triton": triton.__version__,
                             "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability()}
        report["attention_gradient_reference"] = attention_gradient_probe()
        source, tokenizer = MiniCPMVAPOPolicy.from_pretrained(device=torch.device("cuda"),
            lora_config=LoRAConfig(initialization="nora"), gradient_checkpointing=False,
            latent_thinking=True, thought_sigma=args.thought_sigma)
        source.eval()
        if args.normalize_adapter_output:
            source.thought_adapter._probe_output_norm = args.adapter_norm
            source.thought_adapter.forward = MethodType(normalized_adapter_forward, source.thought_adapter)
        original_contract = parameter_contract(source)
        # Construct the optimizer before any installation; never step it here.
        optimizer = torch.optim.AdamW(source.actor_parameters(), lr=1e-5)
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        replica, fused = build_fused_rollout_replica(source)
        prompt = encode_math_prompt(tokenizer,
            {"prompt": [{"role": "user", "content": "Prove that the square root of 2 is irrational."}]},
            prompt_tokens=args.prompt_tokens, enable_thinking=True).to("cuda")
        report.update(model_id=source.model_id, revision=source.revision, prompt_tokens=prompt.numel(),
                      component_std=source.transition.component_std, vector_sigma=source.thought_sigma,
                      fused_projections=fused,
                      precision={"embedding_weight": str(source.causal_lm.get_input_embeddings().weight.dtype),
                                 "lora_master": str(next(m for m in source.causal_lm.modules() if isinstance(m, LoRALinear)).lora_a.dtype),
                                 "gaussian_head": str(source.transition.mean_head.weight.dtype)})
        native = AttentionRegistry(False)
        native.install()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, raw, _ = cached_states(replica, native, prompt, steps=max(args.thought_lengths), seed=args.seed)
        if raw.dtype != torch.float32:
            raise RuntimeError("Gaussian records must remain FP32")
        torch.save({"prompt_ids": prompt.cpu(), "raw_actions": raw.cpu(), "seed": args.seed,
                    "model_id": source.model_id, "revision": source.revision,
                    "contract": "raw actions only; no saved hidden or adapted-input replay override"}, sidecar)
        report["sidecar"] = {"path": str(sidecar), "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest()}
        for stage in ("zero_lora_b", "known_nonzero_lora_b"):
            if stage == "known_nonzero_lora_b":
                changed, rounded_changed, delta_square = 0, 0, 0.0
                with torch.no_grad():
                    for module in source.causal_lm.modules():
                        if isinstance(module, LoRALinear):
                            before = canonical_merged_weight(module)
                            module.lora_b.add_(args.lora_change)
                            after = canonical_merged_weight(module)
                            rounded_changed += int((before != after).sum())
                            changed += module.lora_b.numel()
                            delta_square += module.lora_b.numel() * args.lora_change ** 2
                report["known_change"] = {"operation": "all FP32 lora_b += constant", "constant": args.lora_change,
                    "changed_master_elements": changed, "master_delta_l2": math.sqrt(delta_square),
                    "changed_canonical_bf16_weight_elements": rounded_changed}
                if rounded_changed == 0:
                    raise RuntimeError("known LoRA perturbation disappeared entirely at canonical bf16 merge")
            for canonical in (False, True):
                control = AttentionRegistry(canonical)
                control.install()
                with canonical_models(source, replica) if canonical else nullcontext({}) as installed:
                    synchronized = sync_replica(source, replica, canonical)
                    if parameter_contract(source) != original_contract or optimizer_ids != {id(p) for p in source.actor_parameters()}:
                        raise RuntimeError("canonical installation replaced optimizer parameter identities/names")
                    for length in args.thought_lengths:
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            cached, _, cached_embeddings = cached_states(replica, control, prompt, raw[:length])
                            source_full, source_embeddings = full_states(source, control, prompt, raw[:length])
                            replica_full, replica_embeddings = full_states(replica, control, prompt, raw[:length])
                            case = {"stage": stage, "canonical": canonical, "thoughts": length,
                                    "synchronized_projections": synchronized, "installed": installed,
                                    "source_full_vs_replica_forced_cache": mean_comparison(source, source_full, cached),
                                    "replica_full_vs_replica_forced_cache": mean_comparison(source, replica_full, cached),
                                    "source_full_vs_replica_full": mean_comparison(source, source_full, replica_full),
                                    "source_full_adapter_vs_cached_adapter": differences(source_embeddings, cached_embeddings),
                                    "source_vs_replica_full_adapter": differences(source_embeddings, replica_embeddings),
                                    "actual_hidden_dtype": str(source_full.dtype), "actual_raw_dtype": str(raw.dtype),
                                    "actual_adapter_dtype": str(source_embeddings.dtype)}
                        if canonical:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                case["gaussian_backward"] = backward_probe(source, control, prompt, raw[:length])
                        case["attention_calls"] = dict(control.calls)
                        case["attention_operand_dtypes"] = sorted(control.dtypes)
                        report["cases"].append(case)
                        save_report(args.output, report)
                    if canonical and stage == "known_nonzero_lora_b":
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            report["packed_ragged"] = packed_ragged_probe(source, control, prompt, raw)
                            if args.compile_replica or args.capture_replica:
                                report["qualification"] = qualify_replica(replica, control, prompt, raw,
                                    compile_decode=args.compile_replica, capture=args.capture_replica)
        report["optimizer_contract_preserved"] = parameter_contract(source) == original_contract
        report["trainable_parameter_names"] = sorted(original_contract)
        backward_results = [c["gaussian_backward"] for c in report["cases"] if "gaussian_backward" in c]
        report["finite_backward_qualified"] = all(
            c["logprob_finite"] and c["mean_gradient_finite"] and c["embedding_gradient_finite"]
            and all(c[g]["all_present_finite"] and c[g]["any_nonzero"] for g in ("lora", "gaussian_head", "adapter"))
            and not c["frozen_base_gradients"] for c in backward_results)
        report["status"] = "completed" if report["finite_backward_qualified"] else "failed_gradient_qualification"
        save_report(args.output, report)
        print(json.dumps({"status": report["status"], "output": str(args.output),
                          "finite_backward_qualified": report["finite_backward_qualified"],
                          "compile_capture": report["qualification"]["status"]}), flush=True)
        return 0 if report["finite_backward_qualified"] and report["qualification"]["status"] != "failed" else 1
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        save_report(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
