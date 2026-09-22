#!/usr/bin/env python3
"""Queue-only bf16 projection/attention experiment; not a training implementation.

All variants share a frozen merged/fused pinned MiniCPM trunk, saved raw actions,
and saved bf16 stream embeddings. This isolates trunk replay arithmetic, not
adapter batching drift. FA4's private varlen-capable forward exposes tile_mn;
its public flash_attn_varlen_func does not. Unsupported controls never fall back.
No vocabulary projection, backward, or fp32 model fallback is used. Python
orchestration is uncompiled to inspect layer boundaries: timings are warm
diagnostic latency, not compiled production rollout throughput. Run through mlq.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
from importlib import import_module
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl

from postraining.fast_inference import build_fused_rollout_replica
from postraining.invariant_attention import INVARIANT_ATTENTION
from postraining.invariant_linear import InvariantLinear
from postraining.minicpm_latent_rollout import _LatentStateDecoder
from postraining.vapo.model.hf import MINICPM5_SPEC
from postraining.vapo.model.lora import LoRAConfig
from postraining.vapo.policy import VAPOPolicy
from postraining.train_minicpm_vapo import encode_math_prompt


# Fixed geometry, independent of batch/sequence. Row lane may change, but every
# lane has the same reduction. The probe measures, not assumes, lane invariance.
@triton.jit
def _flat_linear_kernel(
    X, W, Y,
    ROWS: tl.constexpr, WIDTH: tl.constexpr,
    K: tl.constexpr, N: tl.constexpr,
    SB: tl.constexpr, ST: tl.constexpr, SK: tl.constexpr,
):
    rows = tl.program_id(0) * 64 + tl.arange(0, 64)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    kk = tl.arange(0, 32)
    # Logical row -> (batch, token), preserving even noncontiguous input views.
    offsets = (rows // WIDTH) * SB + (rows % WIDTH) * ST
    acc = tl.full((64, 128), 0, tl.float32)
    for block in range(tl.cdiv(K, 32)):
        reduction = block * 32 + kk
        left = tl.load(
            X + offsets[:, None] + reduction[None, :] * SK,
            (rows[:, None] < ROWS) & (reduction[None, :] < K), other=0,
        )
        right = tl.load(
            W + cols[None, :] * K + reduction[:, None],
            (cols[None, :] < N) & (reduction[:, None] < K), other=0,
        )
        acc = tl.dot(left, right, acc, out_dtype=tl.float32)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < ROWS) & (cols[None, :] < N))


class CanonicalFlatLinear(nn.Linear):
    """Inference-only merged bf16 projection, sharing the original weight."""

    residual_fp32: bool = False

    @classmethod
    def from_linear(cls, module: nn.Linear) -> CanonicalFlatLinear:
        if module.bias is not None:
            raise ValueError("canonical trunk requires bias-free merged projections")
        if module.weight.dtype != torch.bfloat16 or not module.weight.is_contiguous():
            raise ValueError("canonical trunk requires contiguous bf16 weights")
        result = cls(module.in_features, module.out_features, bias=False,
                     device="meta", dtype=torch.bfloat16)
        result.weight = module.weight
        return result.eval()

    def forward(self, inputs: Tensor) -> Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("canonical experiment has no backward contract")
        if self.residual_fp32 and inputs.dtype == torch.float32:
            # Autocast cannot see a custom Triton launch. Residual precision is
            # independent of the always-bf16 tensorcore operands.
            inputs = inputs.to(torch.bfloat16)
        if inputs.ndim not in (2, 3) or inputs.dtype != torch.bfloat16 or not inputs.is_cuda:
            raise ValueError("expected CUDA bf16 [batch, width?, hidden]")
        if inputs.shape[-1] != self.in_features or inputs.device != self.weight.device:
            raise ValueError("projection dimensions/device differ")
        x = inputs[:, None, :] if inputs.ndim == 2 else inputs
        batch, width, hidden = x.shape
        if not batch or not width:
            raise ValueError("empty projection input")
        output = torch.empty((*x.shape[:-1], self.out_features),
                             device=x.device, dtype=torch.bfloat16)
        _flat_linear_kernel[(triton.cdiv(batch * width, 64),
                             triton.cdiv(self.out_features, 128))](
            x, self.weight, output, batch * width, width, hidden,
            self.out_features, *x.stride(), num_warps=4, num_stages=3,
            enable_fp_fusion=False,
        )
        return output[:, 0] if inputs.ndim == 2 else output


@contextmanager
def projection_variant(policy, kind, residual_fp32=False):
    originals = [(name, module) for name, module in policy.causal_lm.named_modules()
                 if isinstance(module, nn.Linear) and name != "lm_head"]
    changed = []
    hooks = []
    try:
        for name, original in originals:
            if kind == "cublas":
                continue
            replacement = (CanonicalFlatLinear.from_linear(original) if kind == "flat"
                           else InvariantLinear.from_linear(original))
            if kind == "flat":
                replacement.residual_fp32 = residual_fp32
            if kind == "lane":
                replacement.decoding = True  # Also canonicalize prefill/full replay.
                if residual_fp32:
                    hooks.append(replacement.register_forward_pre_hook(
                        lambda module, args: (args[0].to(torch.bfloat16),)
                    ))
            parent, _, child = name.rpartition(".")
            owner = policy.causal_lm.get_submodule(parent)
            setattr(owner, child, replacement)
            changed.append((owner, child, original))
        names = [name for name, _ in originals]
        layers = len(policy.causal_lm.model.layers)
        if sum(name.endswith(".fused") for name in names) != 2 * layers:
            raise RuntimeError("QKV/gate-up owner.fused projections were not all covered")
        yield names
    finally:
        for handle in hooks:
            handle.remove()
        for owner, child, original in changed:
            setattr(owner, child, original)


class UnsupportedControl(RuntimeError):
    pass


@contextmanager
def residual_stream(policy, enabled):
    """Only residual/input storage changes: model weights and GEMMs stay bf16."""
    hooks = []
    policy._canonical_residual_fp32 = enabled
    try:
        if enabled:
            for name, parameter in policy.causal_lm.named_parameters():
                if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
                    raise UnsupportedControl(f"residual control found non-bf16 trunk weight: {name}")
            hooks.append(policy.causal_lm.get_input_embeddings().register_forward_hook(
                lambda module, inputs, output: output.float()
            ))
            # RoPE follows residual dtype in HF, but Q/K/V and their cache remain bf16.
            hooks.append(policy.causal_lm.model.rotary_emb.register_forward_hook(
                lambda module, inputs, output: tuple(item.to(torch.bfloat16) for item in output)
            ))

            def assert_fp32(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.dtype != torch.float32:
                    raise UnsupportedControl(
                        f"residual fp32 was lost at {type(module).__name__}: {hidden.dtype}"
                    )

            for layer in (*policy.causal_lm.model.layers, policy.causal_lm.model.norm):
                hooks.append(layer.register_forward_hook(assert_fp32))
        yield
    finally:
        for handle in hooks:
            handle.remove()
        policy._canonical_residual_fp32 = False


class AttentionControl:
    """Own both decoder-forced registry names, including its prefill override.

    Workloads are batch-one, unpadded; each prompt is an independent sequence.
    This makes full causal masking and trimmed single-query cache SDPA explicit.
    """
    full_name = "parameter_golf_canonical_probe"

    def __init__(self, name, tile):
        self.name = name
        self.tile = tuple(tile)
        self.phase = "prefill"
        self.live_length = 0
        self.calls = Counter()
        self.forward = None
        if name == "fa4_unsplit":
            self.forward = import_module("flash_attn.cute.interface")._flash_attn_fwd
            required = {"tile_mn", "num_splits", "seqused_k", "pack_gqa"}
            if not required.issubset(inspect.signature(self.forward).parameters):
                raise UnsupportedControl("installed FA4 private forward lacks explicit common-tile/unsplit controls")
        if name == "optimized_splitkv" and torch.cuda.get_device_capability() != (12, 0):
            raise UnsupportedControl("current split-KV control is qualified only for SM120")

    def install(self):
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        for name in (self.full_name, "parameter_golf_fa4_decode", INVARIANT_ATTENTION):
            ALL_ATTENTION_FUNCTIONS.register(name, self.attention)
            ALL_MASK_ATTENTION_FUNCTIONS.register(name, self.no_mask)

    @staticmethod
    def no_mask(**kwargs):
        # No padding in this experiment. Causality is implemented in attention().
        return None

    def attention(self, module, query, key, value, attention_mask,
                  dropout=0.0, scaling=None, **kwargs):
        if query.shape[0] != 1 or attention_mask is not None or dropout:
            raise RuntimeError("probe attention supports unpadded batch-one inference only")
        if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
            raise RuntimeError("attention operands must stay bf16")
        cached = self.phase == "cached"
        if cached and (query.shape[2] != 1 or not module._rollout_cached_append):
            raise RuntimeError("cached control did not receive the real decoder continuation")
        scale = query.shape[-1] ** -0.5 if scaling is None else scaling
        if self.name == "canonical":
            from scripts.minicpm_canonical_attention_probe import canonical_attention
            q, k, v = (t.transpose(1, 2) for t in (query, key, value))
            lengths = (module._rollout_sequence_lengths if cached else
                       torch.full((1,), key.shape[2], device=key.device, dtype=torch.int32))
            positions = ((lengths.long()-1)[:, None] if cached else
                         torch.arange(query.shape[2], device=query.device)[None])
            self.calls[f"{self.phase}:canonical"] += 1
            return canonical_attention(q, k, v, query_positions=positions,
                                       key_lengths=lengths, scale=scale), None
        if self.name == "optimized_splitkv" and cached:
            from postraining.split_kv_attention import split_kv_attention
            q, k, v = (t.transpose(1, 2) for t in (query, key, value))
            if q.shape[1:] != (1, 16, 128) or k.shape[-2:] != (2, 128):
                raise UnsupportedControl("split-KV requires MiniCPM 16Q/2KV heads of width128")
            if not k.is_contiguous() or not v.is_contiguous():
                raise UnsupportedControl("split-KV requires sequence-major contiguous KV backing")
            self.calls["cached:splitkv4"] += 1
            return split_kv_attention(q, k, v, module._rollout_sequence_lengths, scale), None
        if self.name == "fa4_unsplit":
            q, k, v = (t.transpose(1, 2) for t in (query, key, value))
            lengths = (module._rollout_sequence_lengths if cached else
                       torch.full((1,), key.shape[2], device=key.device, dtype=torch.int32))
            self.calls[f"{self.phase}:fa4_unsplit"] += 1
            result = self.forward(
                q, k, v, seqused_k=lengths, max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1], softmax_scale=scale, causal=True,
                tile_mn=self.tile, num_splits=1, pack_gqa=True,
                disable_scheduler_metadata=True,
            )[0]
            return result.contiguous(), None
        # Explicit FLASH-only selection: no math/memory-efficient fallback.
        # SplitKV's full/prefill baseline is FLASH SDPA, as in the current decoder.
        if cached:
            key = key[:, :, :self.live_length]
            value = value[:, :, :self.live_length]
        groups = query.shape[1] // key.shape[1]
        if query.shape[1] != key.shape[1] * groups:
            raise UnsupportedControl("nonintegral GQA head grouping")
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        self.calls[f"{self.phase}:torch_flash"] += 1
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            result = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=not cached, scale=scale,
            )
        return result.transpose(1, 2).contiguous(), None

    def describe(self):
        return {
            "requested": self.name,
            "cached": {"optimized_splitkv": "existing split4 m16n32 fp32 partials",
                       "fa4_unsplit": "FA4 private varlen forward, num_splits=1",
                       "canonical": "fixed-order bf16 GQA online softmax, shared cached/full kernel",
                       "torch_flash": "FLASH-only SDPA; KV trimmed to live prefix, causal=False"}[self.name],
            "prefill_and_full": ({
                "fa4_unsplit": "FA4 private varlen forward, num_splits=1",
                "canonical": "fixed-order bf16 GQA online softmax, shared cached/full kernel",
            }.get(self.name, "FLASH-only SDPA, causal=True")),
            "tile_mn": list(self.tile) if self.name == "fa4_unsplit" else None,
            "calls": dict(self.calls),
        }


def synchronize():
    torch.cuda.synchronize()


def tensor_hash(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def handcheck():
    """Exact one-hot gather catches row aliasing, masked tails, and input strides."""
    weight = torch.zeros((67, 35), device="cuda", dtype=torch.bfloat16)
    columns = torch.arange(67, device="cuda")
    signs = torch.where(columns % 2 == 0, 1, -1).to(torch.bfloat16)
    weight[columns, columns % 35] = signs
    original = nn.Linear(35, 67, bias=False, device="meta", dtype=torch.bfloat16)
    original.weight = nn.Parameter(weight, requires_grad=False)
    linear = CanonicalFlatLinear.from_linear(original)
    storage = (torch.arange(2 * 37 * 70, device="cuda") % 127).to(torch.bfloat16).reshape(2, 37, 70)
    inputs = storage[..., ::2]
    cases = {"strided_3d_cross_tile": inputs,
             "contiguous_3d_cross_tile": inputs.contiguous(),
             "strided_2d_cross_tile": storage.reshape(74, 70)[:, ::2],
             "transposed_batch_token": inputs.transpose(0, 1)}
    result = {}
    for name, x in cases.items():
        actual = linear(x)
        expected = x[..., columns % 35] * signs
        equal = torch.equal(actual, expected)
        result[name] = {"shape": list(x.shape), "stride": list(x.stride()), "exact": equal}
        if not equal:
            raise RuntimeError(f"flat-row alias/stride/tail hand-check failed: {name}")
    # Dense reduction/lane check is measured separately; it is not a cuBLAS oracle.
    original = nn.Linear(35, 67, bias=False, device="cuda", dtype=torch.bfloat16).eval()
    original.requires_grad_(False)
    dense = (inputs / 127).contiguous()
    flat = CanonicalFlatLinear.from_linear(original)(dense)
    lane = InvariantLinear.from_linear(original)
    lane.decoding = True
    reference = lane(dense)
    result["dense_flat_vs_lane"] = {
        "exact": torch.equal(flat, reference),
        "max_abs": float((flat.float() - reference.float()).abs().max()),
    }
    return result


@contextmanager
def layer_trace(policy, mode, prompt_length, enabled):
    values = [[] for _ in policy.causal_lm.model.layers]
    handles = []
    if enabled:
        for index, layer in enumerate(policy.causal_lm.model.layers):
            def hook(module, inputs, output, index=index):
                hidden = output[0] if isinstance(output, tuple) else output
                selected = hidden[0, -1:] if mode == "cached" else hidden[0, prompt_length - 1:]
                values[index].append(selected.detach().clone())
            handles.append(layer.register_forward_hook(hook))
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


def cached_run(policy, control, prompt, embeddings, trace=False, collect=True):
    steps = embeddings.shape[0]
    if getattr(policy, "_canonical_residual_fp32", False):
        embeddings = embeddings.float()
    decoder = _LatentStateDecoder(policy, 1, prompt.numel() + steps + 1, False)
    control.install()
    control.phase = "prefill"
    active = torch.ones(1, device=prompt.device, dtype=torch.bool)
    states = []
    try:
        with layer_trace(policy, "cached", prompt.numel(), trace) as layers:
            synchronize()
            started = time.perf_counter()
            decoder.prefill([prompt])
            # Retain production's indexed cache append independent of model compile.
            for cache_layer in decoder.cache.layers:
                cache_layer.indexed_decode = True
            hidden = decoder.admit([0], [0])
            if collect:
                states.append(hidden[0].clone())
            synchronize()
            prefill_seconds = time.perf_counter() - started
            started = time.perf_counter()
            control.phase = "cached"
            before = sum(value for name, value in control.calls.items() if name.startswith("cached:"))
            for index in range(steps):
                control.live_length = prompt.numel() + index + 1
                hidden = decoder.advance(embeddings[index:index + 1], active)
                if collect:
                    states.append(hidden[0].clone())
            synchronize()
            decode_seconds = time.perf_counter() - started
            after = sum(value for name, value in control.calls.items() if name.startswith("cached:"))
            expected_calls = steps * len(policy.causal_lm.model.layers)
            if after - before != expected_calls:
                raise RuntimeError(f"chosen cached attention bypassed: {after - before} calls, expected {expected_calls}")
        return {
            "states": torch.stack(states) if collect else None,
            "layers": [torch.cat(row) for row in layers] if trace else None,
            "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
            "decode_ms_per_step": 1000 * decode_seconds / steps,
        }
    finally:
        decoder.release_cache()


def full_run(policy, control, prompt, embeddings, trace=False):
    control.install()
    control.phase = "full"
    policy.causal_lm.config._attn_implementation = control.full_name
    for layer in policy.causal_lm.model.layers:
        layer.self_attn._rollout_cached_append = False
    inputs = torch.cat((policy.token_embeddings(prompt), embeddings), dim=0)[None]
    if getattr(policy, "_canonical_residual_fp32", False):
        inputs = inputs.float()
    positions = torch.arange(inputs.shape[1], device=inputs.device)[None]
    with layer_trace(policy, "full", prompt.numel(), trace) as layers:
        synchronize()
        started = time.perf_counter()
        hidden = policy.replay_hidden(None, None, inputs_embeds=inputs, position_ids=positions)
        synchronize()
        elapsed = time.perf_counter() - started
    return {"states": hidden[0, prompt.numel() - 1:].clone(),
            "layers": [torch.cat(row) for row in layers] if trace else None,
            "seconds": elapsed}


def gaussian_comparison(policy, cached, full, prompt_length):
    # Gaussian head stays its specified fp32 policy arithmetic, not a fp32 trunk.
    means = [policy.transition.predict_mean(result["states"]).float()
             for result in (cached, full)]
    if not all(torch.isfinite(mean).all().item() for mean in means):
        raise RuntimeError("nonfinite Gaussian means")
    delta = means[0] - means[1]
    mean_kl = (0.5 * (delta / policy.transition.component_std).square().sum(-1)).cpu()
    l2 = delta.norm(dim=-1).cpu()
    mismatches = (delta != 0).any(dim=-1).nonzero().flatten().cpu().tolist()
    report = {
        "distribution": "isotropic fixed covariance; KL=0.5*sum((mu_cache-mu_full)^2/component_std^2)",
        "positions": "index0 = final prompt state; index t = state after t exact saved raw actions",
        "per_position_mean_kl": mean_kl.tolist(),
        "mean_kl_max": float(mean_kl.max()), "mean_kl_median": float(mean_kl.quantile(0.5)),
        "mean_kl_average": float(mean_kl.mean()),
        "per_position_mean_l2": l2.tolist(),
        "exact_mean_parity": not mismatches,
        "first_divergence": ({"continuation_position": mismatches[0],
                              "absolute_position": prompt_length - 1 + mismatches[0]}
                             if mismatches else None),
        "per_layer": [],
    }
    if cached["layers"] is not None:
        for layer_index, (a, b) in enumerate(zip(cached["layers"], full["layers"], strict=True)):
            difference = a.float() - b.float()
            if not torch.isfinite(difference).all().item():
                raise RuntimeError(f"nonfinite layer {layer_index} difference")
            per_position = difference.norm(dim=-1).cpu()
            mismatch = (difference != 0).any(-1).nonzero().flatten().cpu().tolist()
            report["per_layer"].append({
                "layer": layer_index, "l2_max": float(per_position.max()),
                "first_divergence_position": mismatch[0] if mismatch else None,
                "per_position_l2": per_position.tolist(),
            })
        candidates = [(row["first_divergence_position"], row["layer"])
                      for row in report["per_layer"] if row["first_divergence_position"] is not None]
        report["first_layer_divergence"] = (
            dict(zip(("continuation_position", "layer"), min(candidates), strict=True)) if candidates else None
        )
    else:
        report["first_layer_divergence"] = "not collected (--no-trace-layers)"
    return report


def make_stream(policy, control, prompt, steps, seed):
    """Sample ONCE from regular cuBLAS; all later variants teacher-force this draw."""
    decoder = _LatentStateDecoder(policy, 1, prompt.numel() + steps + 1, False)
    generator = torch.Generator(device=prompt.device).manual_seed(seed)
    active = torch.ones(1, device=prompt.device, dtype=torch.bool)
    raw, embeddings = [], []
    control.install()
    control.phase = "prefill"
    try:
        decoder.prefill([prompt])
        for layer in decoder.cache.layers:
            layer.indexed_decode = True
        hidden = decoder.admit([0], [0])
        control.phase = "cached"
        for index in range(steps):
            mean = policy.transition.predict_mean(hidden)
            action = policy.transition.sample_latent(
                mean, policy.transition.predict_log_sigma(hidden), generator=generator,
            )
            embedded = policy.thought_embeddings(action)
            if embedded.dtype != torch.bfloat16 or action.dtype != torch.float32:
                raise RuntimeError("stream must preserve fp32 raw actions and bf16 adapted inputs")
            raw.append(action[0].clone())
            embeddings.append(embedded[0].clone())
            control.live_length = prompt.numel() + index + 1
            hidden = decoder.advance(embedded, active)
        return torch.stack(raw), torch.stack(embeddings)
    finally:
        decoder.release_cache()


def save_report(output, report):
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--thought-lengths", nargs="+", type=int, default=[32, 256])
    result.add_argument("--projections", nargs="+", choices=["cublas", "flat", "lane"],
                        default=["cublas", "flat"])
    result.add_argument("--attention-controls", nargs="+",
                        choices=["optimized_splitkv", "fa4_unsplit", "torch_flash", "canonical"],
                        default=["optimized_splitkv", "fa4_unsplit", "torch_flash"])
    result.add_argument("--stream-attention", choices=["optimized_splitkv", "fa4_unsplit", "torch_flash", "canonical"],
                        default="optimized_splitkv")
    result.add_argument("--fa4-tile", nargs=2, type=int, default=[64, 64], metavar=("M", "N"))
    result.add_argument("--benchmark-lane", action="store_true",
                        help="include lane-wasting projection on the identical workloads")
    result.add_argument("--residual-fp32", action="store_true",
                        help="fp32 input/residual/final-norm stream, but bf16 weights/GEMMs; fixed sampled stream remains unchanged")
    result.add_argument("--canonical-norm", action="store_true")
    result.add_argument("--timing-repeats", type=int, default=1,
                        help="additional warm uninstrumented cached/full passes per completed variant")
    result.add_argument("--trace-layers", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--prompt-tokens", type=int, default=1024)
    result.add_argument("--long-prompt-repeats", type=int, default=16)
    result.add_argument("--seed", type=int, default=1337)
    result.add_argument("--thought-sigma", type=float, default=1.0)
    return result


def main():
    args = parser().parse_args()
    if args.output.exists() or args.output.with_suffix(".streams.pt").exists():
        raise FileExistsError("output or its stream artifact already exists")
    if min(args.thought_lengths) < 1 or args.timing_repeats < 1 or args.prompt_tokens < 128:
        raise ValueError("positive thought lengths/repeats and prompt-tokens >=128 required")
    if any(size not in (16, 32, 64, 128) for size in args.fa4_tile):
        raise ValueError("FA4 common tile dimensions must be 16,32,64,128; unsupported combinations report errors")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "minicpm-canonical-cached-replay/v1", "status": "initializing",
        "argv": sys.argv[1:], "model_id": MINICPM5_SPEC.model_id, "revision": MINICPM5_SPEC.revision,
        "precision": "bf16 trunk/stream; fixed64x128x32 GEMM fp32 accumulator; raw actions/head fp32",
        "residual_dtype": "fp32" if args.residual_fp32 else "bf16",
        "canonical_norm": args.canonical_norm,
        "timing_contract": "warm uncompiled Python orchestration; no trace/collection in cached timing; residual-fp32 adds dtype-only hooks; no backward/LM head",
        "scope": "batch1 independently for two unequal prompt lengths; not packed replay or ragged-batch qualification",
        "adapter_contract": "identical individually-adapted saved bf16 inputs; adapter batch-shape drift intentionally excluded",
        "sources": {}, "workloads": [], "variants": [],
    }
    for name in ("scripts/diagnose_minicpm_canonical.py", "postraining/invariant_linear.py",
                 "postraining/invariant_attention.py", "postraining/fast_inference.py",
                 "postraining/minicpm_latent_rollout.py", "postraining/vapo/policy.py",
                 "postraining/latent_thought.py", "postraining/split_kv_attention.py",
                 "scripts/minicpm_canonical_attention_probe.py",
                 "scripts/minicpm_canonical_norm_probe.py"):
        report["sources"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    save_report(args.output, report)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU model fallback")
        torch.manual_seed(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        report["runtime"] = {
            "torch": torch.__version__, "triton": triton.__version__,
            "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
            "cublas_allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        }
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            report["handcheck"] = handcheck()
            save_report(args.output, report)
            source, tokenizer = VAPOPolicy.from_family("minicpm5", 
                model_id=MINICPM5_SPEC.model_id, revision=MINICPM5_SPEC.revision,
                device=torch.device("cuda"), lora_config=LoRAConfig(initialization="nora"),
                latent_thinking=True, thought_sigma=args.thought_sigma,
                gradient_checkpointing=False,
            )
            source.eval()
            policy, groups = build_fused_rollout_replica(source)
            del source
            report["fused_projection_groups"] = list(groups)
            report["component_std"] = policy.transition.component_std
            text = "Prove that the square root of 2 is irrational."
            long_text = ("Use a rigorous elementary argument. State all assumptions and justify each divisibility step. " * args.long_prompt_repeats) + text
            prompts = [encode_math_prompt(
                tokenizer, {"prompt": [{"role": "user", "content": content}]},
                prompt_tokens=args.prompt_tokens, enable_thinking=True,
            ).to("cuda") for content in (text, long_text)]
            if prompts[0].numel() == prompts[1].numel():
                raise RuntimeError("prompt encoder did not produce two different prompt lengths")
            streams = []
            stream_artifact = args.output.with_suffix(".streams.pt")
            for index, prompt in enumerate(prompts):
                control = AttentionControl(args.stream_attention, args.fa4_tile)
                raw, embedded = make_stream(policy, control, prompt, max(args.thought_lengths), args.seed + index)
                streams.append((prompt, raw, embedded))
                report["workloads"].append({
                    "prompt_index": index, "prompt_tokens": prompt.numel(),
                    "thought_lengths": args.thought_lengths, "prompt_ids": prompt.cpu().tolist(),
                    "raw_sha256": tensor_hash(raw), "embeddings_sha256": tensor_hash(embedded),
                    "sampling": "once from frozen merged cuBLAS, same raw stream prefix for every variant/length",
                    "sampling_attention": control.describe(),
                })
                torch.save({"revision": MINICPM5_SPEC.revision, "seed": args.seed,
                            "streams": [{"prompt": p.cpu(), "raw": r.cpu(), "embeddings": e.cpu()}
                                        for p, r, e in streams]}, stream_artifact)
                report["stream_artifact"] = str(stream_artifact)
                save_report(args.output, report)
            if args.canonical_norm:
                from scripts.minicpm_canonical_norm_probe import install_canonical_norms_
                install_canonical_norms_(policy.causal_lm)
            projections = list(dict.fromkeys(args.projections + (["lane"] if args.benchmark_lane else [])))
            report["status"] = "running"
            for projection in projections:
                with residual_stream(policy, args.residual_fp32), projection_variant(
                    policy, projection, args.residual_fp32
                ) as names:
                    report["patched_projection_names"] = names
                    for attention in dict.fromkeys(args.attention_controls):
                        for prompt_index, (prompt, raw, embedded) in enumerate(streams):
                            for steps in sorted(set(args.thought_lengths)):
                                case = {"projection": projection, "attention": attention,
                                        "prompt_index": prompt_index, "prompt_tokens": prompt.numel(),
                                        "thought_steps": steps, "residual_fp32": args.residual_fp32,
                                        "status": "running"}
                                started = time.perf_counter()
                                try:
                                    control = AttentionControl(attention, args.fa4_tile)
                                    inputs = embedded[:steps]
                                    # First evidence pass also warms every visited cache length/kernel shape.
                                    cached = cached_run(policy, control, prompt, inputs, trace=args.trace_layers)
                                    full = full_run(policy, control, prompt, inputs, trace=args.trace_layers)
                                    case["comparison"] = gaussian_comparison(policy, cached, full, prompt.numel())
                                    case["cold_instrumented_seconds"] = {
                                        "prefill": cached["prefill_seconds"], "cached": cached["decode_seconds"],
                                        "full": full["seconds"],
                                    }
                                    del cached, full
                                    timing = []
                                    for _ in range(args.timing_repeats):
                                        cached_time = cached_run(policy, control, prompt, inputs, collect=False)
                                        full_time = full_run(policy, control, prompt, inputs)
                                        timing.append({"prefill_seconds": cached_time["prefill_seconds"],
                                                       "decode_seconds": cached_time["decode_seconds"],
                                                       "decode_ms_per_step": cached_time["decode_ms_per_step"],
                                                       "full_seconds": full_time["seconds"]})
                                        del full_time
                                    case["warm_latency"] = timing
                                    case["actual_attention"] = control.describe()
                                    case["status"] = "completed"  # Numerical parity is a separate measured field.
                                except Exception as exc:
                                    case["status"] = "unsupported" if isinstance(exc, UnsupportedControl) else "error"
                                    case["error"] = f"{type(exc).__name__}: {exc}"
                                    case["traceback"] = traceback.format_exc()
                                case["wall_seconds"] = time.perf_counter() - started
                                report["variants"].append(case)
                                save_report(args.output, report)
                                print(json.dumps({key: case[key] for key in (
                                    "projection", "attention", "prompt_index", "thought_steps", "status", "wall_seconds"
                                )}), flush=True)
            failures = [case for case in report["variants"] if case["status"] != "completed"]
            report["status"] = "completed_with_errors" if failures else "completed"
            save_report(args.output, report)
            if failures:
                return 2
            return 0
    except Exception as exc:
        report["status"] = "fatal_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        save_report(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
