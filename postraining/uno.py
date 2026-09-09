"""Gated diffusion LoRA and single-pass Uno distillation for native Llama.

Reference: ifm-ai/uno@46fbdb66f026bae9c68a1e5a3f97a17c7805c778,
training/{modeling,losses,lora}.py; released modeling_sdar.py at
s-sahoo/uno-qwen3-8B@8819e09ac901e7290d8d89d62c98b9f756c602fe.
The equivalent [clean, noise] layout uses duplicated logical RoPE positions.
Unlike the release's observed-token noise bound, corruption covers the entire
vocabulary. The objective called TV upstream is probability L1 (twice TV).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
import torch.nn.functional as F

UNO_SCHEMA = "minicpm_uno/v1"
UNO_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class UnoConfig:
    rank: int = 48
    alpha: float = 3072.0
    targets: tuple[str, ...] = UNO_TARGETS

    def __post_init__(self) -> None:
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise ValueError("Uno rank must be a positive integer")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("Uno alpha must be finite and positive")
        if tuple(self.targets) != UNO_TARGETS:
            raise ValueError("Uno requires all seven projections in canonical order")


def uno_model_config(causal_lm: nn.Module) -> dict[str, Any]:
    config: Any = causal_lm.config
    if config.model_type != "llama":
        raise ValueError("Uno requires the native Llama model architecture")
    fields = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    )
    result: dict[str, Any] = {name: int(getattr(config, name)) for name in fields}
    result["head_dim"] = int(
        getattr(config, "head_dim", None)
        or result["hidden_size"] // result["num_attention_heads"]
    )
    result["model_type"] = config.model_type
    return result


class _UnoProjection(nn.Module):
    def __init__(
        self,
        input_width: int,
        output_width: int,
        config: UnoConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.scaling = config.alpha / config.rank
        self.lora_a = nn.Parameter(
            torch.empty(config.rank, input_width, dtype=torch.float32, device=device)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(output_width, config.rank, dtype=torch.float32, device=device)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dtype != self.lora_a.dtype and not torch.is_autocast_enabled(
            inputs.device.type
        ):
            a = self.lora_a.to(inputs.dtype)
            b = self.lora_b.to(inputs.dtype)
        else:
            a, b = self.lora_a, self.lora_b
        return F.linear(F.linear(inputs, a), b) * self.scaling


class UnoAdapterBank(nn.Module):
    """Independent fp32 masters, including when the owning model changes dtype."""

    def __init__(self, causal_lm: nn.Module, config: UnoConfig) -> None:
        super().__init__()
        self.config = config
        dims = uno_model_config(causal_lm)
        hidden, intermediate = dims["hidden_size"], dims["intermediate_size"]
        q_width = dims["num_attention_heads"] * dims["head_dim"]
        kv_width = dims["num_key_value_heads"] * dims["head_dim"]
        shapes = {
            "q_proj": (hidden, q_width),
            "k_proj": (hidden, kv_width),
            "v_proj": (hidden, kv_width),
            "o_proj": (q_width, hidden),
            "gate_proj": (hidden, intermediate),
            "up_proj": (hidden, intermediate),
            "down_proj": (intermediate, hidden),
        }
        device = next(causal_lm.parameters()).device
        self.projections = nn.ModuleDict()
        self.names: tuple[str, ...] = tuple(
            f"model.layers.{layer}.{'self_attn' if target in UNO_TARGETS[:4] else 'mlp'}.{target}"
            for layer in range(dims["num_hidden_layers"])
            for target in config.targets
        )
        for name in self.names:
            self.projections[self._key(name)] = _UnoProjection(
                *shapes[name.rsplit(".", 1)[1]], config, device
            )

    @staticmethod
    def _key(name: str) -> str:
        return name.replace(".", "__")

    def _apply(self, fn, recurse: bool = True):
        # Device migration is allowed; model.bfloat16() must not quantize masters.
        def master_fn(tensor):
            if tensor.is_floating_point():
                probe = fn(torch.empty(0, device=tensor.device, dtype=tensor.dtype))
                return tensor.to(device=probe.device, dtype=torch.float32)
            return fn(tensor)

        return super()._apply(master_fn, recurse=recurse)


class UnoAdapterRouter:
    def __init__(self, causal_lm: nn.Module, bank: UnoAdapterBank) -> None:
        if hasattr(causal_lm, "uno_adapter"):
            raise ValueError("a Uno bank is already attached to this model")
        modules = [(name, causal_lm.get_submodule(name)) for name in bank.names]
        self.causal_lm = causal_lm
        self.bank = bank
        self.gate: Tensor | None = None
        self._handles = []
        causal_lm.add_module("uno_adapter", bank)
        for name, module in modules:
            projection = bank.projections[bank._key(name)]

            def hook(_module, inputs, output, projection=projection):
                gate = self.gate
                if gate is None:
                    return output
                return output + projection(inputs[0]).to(output.dtype) * gate.to(
                    output.dtype
                )

            self._handles.append(module.register_forward_hook(hook))

    def set_gate(self, gate: Tensor | None) -> None:
        if gate is not None and (gate.ndim != 3 or gate.shape[-1] != 1):
            raise ValueError("Uno gate must broadcast as [batch, sequence, 1]")
        self.gate = gate

    @contextmanager
    def _with_gate(self, gate: Tensor | None):
        previous = self.gate
        self.gate = gate
        try:
            yield
        finally:
            self.gate = previous

    def checkpoint_contexts(self):
        # Capture at checkpoint creation, not at backward: recomputation must use
        # the original routing even if a later forward changed the live gate.
        gate = self.gate
        return self._with_gate(gate), self._with_gate(gate)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.gate = None


def attach_uno_adapters(causal_lm: nn.Module, bank: UnoAdapterBank) -> UnoAdapterRouter:
    return UnoAdapterRouter(causal_lm, bank)


def enable_uno_checkpointing(causal_lm: nn.Module, router: UnoAdapterRouter) -> None:
    cast(Any, causal_lm).gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "context_fn": router.checkpoint_contexts,
        }
    )


def uno_mask_mod(sequence_length: int, block_size: int):
    """Clean is token-causal; noise sees earlier clean blocks and own causal block."""
    if sequence_length < 1 or block_size < 1:
        raise ValueError("sequence length and block size must be positive")

    def mask(batch, head, query, key):
        del batch, head
        noisy_q, noisy_k = query >= sequence_length, key >= sequence_length
        q, k = query % sequence_length, key % sequence_length
        clean = (~noisy_q) & (~noisy_k) & (q >= k)
        context = noisy_q & (~noisy_k) & (q // block_size > k // block_size)
        noise = noisy_q & noisy_k & (q // block_size == k // block_size) & (q >= k)
        return (
            (clean | context | noise)
            & (query < 2 * sequence_length)
            & (key < 2 * sequence_length)
        )

    return mask


def make_uno_block_mask(sequence_length: int, block_size: int, device: torch.device):
    from torch.nn.attention.flex_attention import create_block_mask

    compiled_create = torch.compile(create_block_mask)
    return compiled_create(
        uno_mask_mod(sequence_length, block_size),
        B=None,
        H=None,
        Q_LEN=2 * sequence_length,
        KV_LEN=2 * sequence_length,
        device=str(device),
    )


def paired_uno_inputs(
    clean_ids: Tensor, vocab_size: int, *, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    if clean_ids.ndim != 2 or clean_ids.shape[1] < 1 or vocab_size < 1:
        raise ValueError("Uno needs a nonempty [batch, sequence] token tensor")
    noise = torch.randint(
        vocab_size, clean_ids.shape, device=clean_ids.device, generator=generator
    )
    positions = (
        torch.arange(clean_ids.shape[1], device=clean_ids.device)
        .repeat(2)[None, :]
        .expand(clean_ids.shape[0], -1)
    )
    gate = torch.cat(
        (torch.zeros_like(clean_ids), torch.ones_like(clean_ids)), dim=1
    ).unsqueeze(-1)
    return torch.cat((clean_ids, noise), dim=1), positions, gate


class _ChunkedHeadL1(torch.autograd.Function):
    """Save hidden states only; recompute each bounded full-vocabulary head chunk."""

    @staticmethod
    def forward(
        ctx,
        student: Tensor,
        teacher: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        chunk_size: int,
        token_weights: Tensor,
    ):
        ctx.save_for_backward(student, teacher, weight, token_weights)
        ctx.bias, ctx.chunk_size = bias, chunk_size
        ctx.autocast = torch.is_autocast_enabled(student.device.type)
        ctx.autocast_dtype = torch.get_autocast_dtype(student.device.type)
        total = student.new_zeros((), dtype=torch.float32)
        denominator = token_weights.sum()
        ctx.denominator = denominator
        for start in range(0, student.shape[0], chunk_size):
            end = start + chunk_size
            p = F.linear(teacher[start:end], weight, bias).float().softmax(-1)
            q = F.linear(student[start:end], weight, bias).float().softmax(-1)
            total.add_(((q - p).abs().sum(-1) * token_weights[start:end]).sum())
        return total / denominator

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor):
        if len(grad_outputs) != 1:
            raise RuntimeError("Uno L1 backward expects one gradient")
        grad_output = grad_outputs[0]
        student, teacher, weight, token_weights = ctx.saved_tensors
        gradient = torch.empty_like(student)
        with torch.autocast(
            student.device.type, enabled=ctx.autocast, dtype=ctx.autocast_dtype
        ):
            for start in range(0, student.shape[0], ctx.chunk_size):
                end = start + ctx.chunk_size
                p = F.linear(teacher[start:end], weight, ctx.bias).float().softmax(-1)
                q = F.linear(student[start:end], weight, ctx.bias).float().softmax(-1)
                signs = (q - p).sign()
                dlogits = q * (signs - (signs * q).sum(-1, keepdim=True))
                dlogits.mul_(
                    (token_weights[start:end] * grad_output / ctx.denominator)[:, None]
                )
                # Match bf16 head forward/backward rounding under autocast.
                compute_dtype = ctx.autocast_dtype if ctx.autocast else student.dtype
                gradient[start:end] = dlogits.to(compute_dtype) @ weight.to(
                    compute_dtype
                )
        return gradient, None, None, None, None, None


def chunked_head_l1(
    student: Tensor,
    teacher: Tensor,
    lm_head: nn.Module,
    *,
    chunk_size: int = 32,
    token_weights: Tensor | None = None,
) -> Tensor:
    if student.shape != teacher.shape or chunk_size < 1:
        raise ValueError(
            "aligned student/teacher shapes and positive chunk size required"
        )
    if any(p.requires_grad for p in lm_head.parameters()):
        raise ValueError("Uno distillation requires a frozen output head")
    student = student.reshape(-1, student.shape[-1])
    teacher = teacher.detach().reshape_as(student)
    if token_weights is None:
        token_weights = torch.ones(
            student.shape[0], device=student.device, dtype=torch.float32
        )
    else:
        token_weights = token_weights.reshape(-1).to(
            device=student.device, dtype=torch.float32
        )
    return _ChunkedHeadL1.apply(
        student,
        teacher,
        lm_head.weight,
        getattr(lm_head, "bias", None),
        chunk_size,
        token_weights,
    )


def uno_distillation_loss(
    causal_lm: nn.Module,
    router: UnoAdapterRouter,
    clean_ids: Tensor,
    block_mask: Any,
    *,
    chunk_size: int = 32,
    generator: torch.Generator | None = None,
    token_weights: Tensor | None = None,
) -> Tensor:
    model: Any = causal_lm
    ids, positions, gate = paired_uno_inputs(
        clean_ids, model.config.vocab_size, generator=generator
    )
    with router._with_gate(gate):
        hidden = model.model(
            input_ids=ids,
            position_ids=positions,
            attention_mask=block_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
    length = clean_ids.shape[1]
    # SAME output index: both are next-token distributions. Do not shift the
    # teacher or train to reconstruct the current corrupted input token.
    return chunked_head_l1(
        hidden[:, length:],
        hidden[:, :length],
        model.lm_head,
        chunk_size=chunk_size,
        token_weights=token_weights,
    )


def uno_checkpoint_payload(
    bank: UnoAdapterBank,
    causal_lm: nn.Module,
    *,
    model_id: str,
    revision: str,
    trained_tokens: int,
    step: int,
    teacher_sha256: str,
    training: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": UNO_SCHEMA,
        "model_id": model_id,
        "revision": revision,
        "model_config": uno_model_config(causal_lm),
        "uno_config": asdict(bank.config),
        "adapter": {
            name: value.detach().cpu() for name, value in bank.state_dict().items()
        },
        "trained_tokens": trained_tokens,
        "step": step,
        "teacher_sha256": teacher_sha256,
        "training": training,
    }


def load_uno_adapter(
    path: str | Path, causal_lm: nn.Module, *, model_id: str, revision: str
) -> tuple[UnoAdapterBank, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != UNO_SCHEMA:
        raise ValueError("unsupported Uno checkpoint schema")
    if payload.get("model_id") != model_id or payload.get("revision") != revision:
        raise ValueError("Uno checkpoint model identity differs")
    if payload.get("model_config") != uno_model_config(causal_lm):
        raise ValueError("Uno checkpoint model dimensions differ")
    for field in ("trained_tokens", "step"):
        if type(payload.get(field)) is not int or payload[field] < 1:
            raise ValueError(f"Uno checkpoint must contain positive {field}")
    digest = payload.get("teacher_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Uno checkpoint lacks a valid teacher identity")
    if not isinstance(payload.get("training"), dict) or not payload["training"].get(
        "corpus_sha256"
    ):
        raise ValueError("Uno checkpoint lacks training corpus identity")
    config = UnoConfig(**payload["uno_config"])
    # Loading an inference adapter must not perturb actor sampling RNG.
    with torch.random.fork_rng(
        devices=[]
        if next(causal_lm.parameters()).device.type == "cpu"
        else [next(causal_lm.parameters()).device]
    ):
        bank = UnoAdapterBank(causal_lm, config)
    state = payload.get("adapter")
    expected = bank.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Uno checkpoint adapter keys differ")
    for name, tensor in state.items():
        if (
            not isinstance(tensor, Tensor)
            or tensor.shape != expected[name].shape
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"invalid Uno adapter tensor: {name}")
    bank.load_state_dict(state, strict=True)
    return bank, payload
