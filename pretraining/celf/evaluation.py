"""Fixed-anchor NELBO, teacher-forced block sampling, and cached generation.

The latent variable is the anchored latent ``w`` at ``decode_time`` on the flow
path. Its posterior is ``N((1 - t_d) z(x), t_d^2 I)`` by construction, and its
prior given history is the flow marginal at ``t_d``, so a negative ELBO exists
without any learned variance. It is a Monte Carlo upper-bound estimate of the
byte rate: one posterior sample, Hutchinson divergence, finite Heun steps.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from .model import CelfModel

_LOG_2PI = math.log(2 * math.pi)
_LOG_2 = math.log(2)

Velocity = Callable[[Tensor, Tensor, Tensor], Tensor]


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _standard_normal_log_prob(value: Tensor) -> Tensor:
    return -0.5 * (value.float().square() + _LOG_2PI).flatten(1).sum(1)


def _check_states(history: Tensor, initial: Tensor) -> None:
    if initial.ndim != 3 or min(initial.shape) < 1 or history.shape != initial.shape:
        raise ValueError("history and state must share nonempty [B, L, D] shapes")
    if history.device != initial.device:
        raise ValueError("history and state must share a device")


def integrate_sample(
    velocity: Velocity, history: Tensor, initial: Tensor, *, stop_time: float, steps: int
) -> Tensor:
    """Heun from noise time 1 down to ``stop_time``; no gradients, no density."""
    _positive_integer("steps", steps)
    _check_states(history, initial)
    if not 0.0 < stop_time < 1.0:
        raise ValueError("stop_time must lie strictly inside (0, 1)")
    state = initial.detach().float()
    times_shape = initial.shape[:2]
    dt = (1.0 - stop_time) / steps

    def evaluate(value: Tensor, t: float) -> Tensor:
        times = torch.full(times_shape, t, device=value.device, dtype=torch.float32)
        return velocity(history, value, times).float()

    with torch.no_grad():
        for index in range(steps):
            t = 1.0 - index * dt
            first = evaluate(state, t)
            predictor = state - dt * first
            second = evaluate(predictor, t - dt)
            state = state - (0.5 * dt) * (first + second)
    return state


def integrate_log_density(
    velocity: Velocity,
    history: Tensor,
    initial: Tensor,
    *,
    start_time: float,
    steps: int,
    probe_vectors: Tensor,
) -> dict:
    """Integrate the flow from ``start_time`` to 1 and accumulate the divergence.

    Returns per-example ``log_prob`` = log N(state at 1) + integral of div v,
    the flow marginal density of ``initial`` at ``start_time`` given ``history``.
    History and probes stay fixed at every Heun stage. State, traces, and
    reductions use FP32; no graph is retained across evaluations.
    """
    _positive_integer("steps", steps)
    if torch.is_inference_mode_enabled():
        raise RuntimeError("density divergence requires input gradients, not inference_mode")
    _check_states(history, initial)
    if not 0.0 < start_time < 1.0:
        raise ValueError("start_time must lie strictly inside (0, 1)")
    if probe_vectors.ndim != 4 or probe_vectors.shape[1:] != initial.shape:
        raise ValueError("probe_vectors must have shape [P, B, L, D]")
    _positive_integer("probe count", probe_vectors.shape[0])
    if probe_vectors.device != initial.device:
        raise ValueError("probes must share the state device")
    state = initial.detach().float()
    fixed_history = history.detach()
    probes = probe_vectors.detach().float()
    integrated_trace = torch.zeros(initial.shape[0], device=initial.device, dtype=torch.float32)
    dt = (1.0 - start_time) / steps

    def evaluate(value: Tensor, t: float) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            differentiable = value.detach().requires_grad_(True)
            times = torch.full(
                value.shape[:2], t, device=value.device, dtype=torch.float32
            )
            result = velocity(fixed_history, differentiable, times).float()
            if result.shape != value.shape:
                raise ValueError("velocity must preserve the [B, L, D] state shape")
            if not result.requires_grad:
                raise RuntimeError(
                    "velocity is not differentiable in its state; no density can be reported"
                )
            trace = torch.zeros_like(integrated_trace)
            for index, probe in enumerate(probes):
                derivative = torch.autograd.grad(
                    result,
                    differentiable,
                    grad_outputs=probe,
                    retain_graph=index + 1 < probes.shape[0],
                    create_graph=False,
                )[0]
                trace = trace + (derivative.float() * probe).flatten(1).sum(1)
            trace = trace / probes.shape[0]
            return result.detach(), trace.detach()

    for index in range(steps):
        t = start_time + index * dt
        first_velocity, first_trace = evaluate(state, t)
        predictor = (state + dt * first_velocity).detach()
        second_velocity, second_trace = evaluate(predictor, t + dt)
        state = (state + (0.5 * dt) * (first_velocity + second_velocity)).detach()
        integrated_trace = (
            integrated_trace + (0.5 * dt) * (first_trace + second_trace)
        ).detach()
    return {
        "state": state,
        "divergence_integral": integrated_trace,
        "log_prob": _standard_normal_log_prob(state) + integrated_trace,
        "nfe": 2 * steps,
    }


def _model_device(model: CelfModel) -> torch.device:
    if not model.compiled:
        raise ValueError("call model.compile_components() before evaluation")
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("CELF neural evaluation requires CUDA; no CPU fallback")
    if any(value.device != device for value in (*model.parameters(), *model.buffers())):
        raise ValueError("all model tensors must share one CUDA device")
    if torch.is_autocast_enabled("cuda"):
        raise ValueError("CELF uses explicit BF16 neural execution, not autocast")
    return device


@contextmanager
def evaluation_mode(model: CelfModel) -> Iterator[None]:
    """Freeze weights, not inputs; restore heterogeneous module modes."""
    parameter_flags = [(p, p.requires_grad) for p in model.parameters()]
    module_flags = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        for parameter, _ in parameter_flags:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in parameter_flags:
            parameter.requires_grad_(requires_grad)
        for module, training in module_flags:
            module.training = training


def _check_ids(model: CelfModel, ids: Tensor, device: torch.device) -> Tensor:
    config = model.config
    if ids.device != device or ids.ndim != 2 or ids.shape[0] < 1:
        raise ValueError("ids must be a nonempty [B, T] batch on the model CUDA device")
    if ids.shape[1] != config.seq_bytes:
        raise ValueError("evaluation requires complete configured byte contexts")
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("ids must be integer byte values")
    if bool(((ids < 0) | (ids >= config.vocab_size)).any()):
        raise ValueError("byte values must lie in 0..255")
    return ids.view(ids.shape[0], config.seq_patches, config.patch_size)


def _byte_nll(logits: Tensor, patches: Tensor) -> Tensor:
    return F.cross_entropy(
        logits.float().flatten(0, 2), patches.long().flatten(), reduction="none"
    ).view_as(patches)


def estimate_nelbo(
    model: CelfModel, ids: Tensor, *, steps: int, seed: int, probes: int = 1
) -> dict:
    """Fixed-anchor negative ELBO in nats and bits per literal byte.

    ``log q`` is the closed-form anchored posterior density at its own sample;
    ``log p`` is the conditional flow marginal at ``decode_time`` with the
    posterior history sample held fixed for every block at once.
    """
    _positive_integer("steps", steps)
    _positive_integer("probes", probes)
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("density divergence requires input gradients, not inference_mode")
    device = _model_device(model)
    patches = _check_ids(model, ids, device)
    config = model.config
    generator = torch.Generator(device=device).manual_seed(seed)
    with evaluation_mode(model):
        with torch.no_grad():
            z = model.encoder(patches, torch.zeros(patches.shape[:2], dtype=torch.bool, device=device))
            anchor_noise = torch.randn(z.shape, device=device, dtype=torch.float32, generator=generator)
            w = model.anchor(z, anchor_noise)
            posterior_logprob = (
                -0.5
                * (_LOG_2PI + 2.0 * math.log(config.decode_time) + anchor_noise.square())
            ).sum()
            reconstruction_nll = _byte_nll(model.decoder(w), patches).sum()
            probe_vectors = torch.randn(
                (probes, *w.shape), device=device, dtype=torch.float32, generator=generator
            )
        integrated = integrate_log_density(
            model.prior, w, w, start_time=config.decode_time, steps=steps, probe_vectors=probe_vectors
        )
        prior_logprob = integrated["log_prob"].sum()
        negative_elbo = reconstruction_nll - prior_logprob + posterior_logprob
        values = (
            torch.stack((reconstruction_nll, prior_logprob, posterior_logprob, negative_elbo))
            .detach()
            .cpu()
            .tolist()
        )
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("non-finite negative-ELBO estimate; no rate can be reported")
    reconstruction, prior, posterior, total = values
    byte_count = ids.numel()
    return {
        "bytes": byte_count,
        "patches": patches.shape[0] * patches.shape[1],
        "batch_size": ids.shape[0],
        "reconstruction_nll_nats": reconstruction,
        "prior_logprob_nats": prior,
        "posterior_logprob_nats": posterior,
        "negative_elbo_estimate_nats": total,
        "reconstruction_bpb": reconstruction / (byte_count * _LOG_2),
        "rate_bpb": (posterior - prior) / (byte_count * _LOG_2),
        "negative_elbo_estimate_bpb": total / (byte_count * _LOG_2),
        "steps": steps,
        "probes": probes,
        "seed": seed,
        "solver": "heun",
        "time_interval": [config.decode_time, 1.0],
        "nfe": integrated["nfe"],
        "exact_marginal_bpb": False,
    }


def sample_blocks(model: CelfModel, ids: Tensor, *, steps: int, seed: int) -> dict:
    """Teacher-forced generation of every block from its true anchored history.

    Reports the byte accuracy of greedy decoding at the sampled anchored latents
    and the decoder NLL of the true bytes there. Neither is a bound; both are
    generation-quality diagnostics at the training conditioning distribution.
    """
    _positive_integer("steps", steps)
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    device = _model_device(model)
    patches = _check_ids(model, ids, device)
    config = model.config
    generator = torch.Generator(device=device).manual_seed(seed)
    with evaluation_mode(model), torch.no_grad():
        z = model.encoder(patches, torch.zeros(patches.shape[:2], dtype=torch.bool, device=device))
        w = model.anchor(z, torch.randn(z.shape, device=device, generator=generator))
        initial = torch.randn(z.shape, device=device, generator=generator)
        sampled = integrate_sample(
            model.prior, w, initial, stop_time=config.decode_time, steps=steps
        )
        logits = model.decoder(sampled)
        nll = _byte_nll(logits, patches)
        correct = (logits.argmax(-1) == patches).sum()
        posterior_logits = model.decoder(w)
        posterior_nll = _byte_nll(posterior_logits, patches)
        values = torch.stack((nll.sum(), correct, posterior_nll.sum())).cpu().tolist()
    sampled_nll, correct_bytes, posterior_nll_sum = values
    byte_count = ids.numel()
    return {
        "bytes": byte_count,
        "sampled_nll_nats": sampled_nll,
        "sampled_nll_per_byte": sampled_nll / byte_count,
        "sampled_byte_accuracy": correct_bytes / byte_count,
        "posterior_nll_per_byte": posterior_nll_sum / byte_count,
        "steps": steps,
        "seed": seed,
    }


def generate_bytes(
    model: CelfModel,
    prompt: bytes,
    *,
    new_bytes: int,
    steps: int,
    seed: int,
    temperature: float = 1.0,
) -> dict:
    """Sample a byte suffix block by block with cached anchored history.

    Prompt and suffix lengths must be whole blocks. The prompt's anchored
    latents are one posterior sample. Each new block integrates from noise to
    ``decode_time`` under the cached history, is appended, and every block is
    decoded jointly at the end. Prompt bytes are never resampled.
    """
    config = model.config
    if not isinstance(prompt, (bytes, bytearray)):
        raise TypeError("prompt must be bytes")
    if type(new_bytes) is not int or new_bytes < 0:
        raise ValueError("new_bytes must be a nonnegative integer")
    _positive_integer("steps", steps)
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if len(prompt) < config.block_bytes or len(prompt) % config.block_bytes or new_bytes % config.block_bytes:
        raise ValueError("prompt and suffix must be nonempty whole blocks")
    if len(prompt) + new_bytes > config.seq_bytes:
        raise ValueError("prompt plus suffix exceeds the configured context")
    device = _model_device(model)
    generator = torch.Generator(device=device).manual_seed(seed)
    started = time.perf_counter()
    prompt_ids = torch.tensor(list(prompt), dtype=torch.long, device=device)
    patches = prompt_ids.view(1, -1, config.patch_size)
    new_blocks = new_bytes // config.block_bytes
    with evaluation_mode(model), torch.no_grad():
        z = model.encoder(patches, torch.zeros(patches.shape[:2], dtype=torch.bool, device=device))
        w = model.anchor(z, torch.randn(z.shape, device=device, generator=generator))
        cache = None
        for start in range(0, w.shape[1], config.block_patches):
            cache = model.prior.append_history(w[:, start : start + config.block_patches], cache)
        generated = []
        for block_index in range(new_blocks):
            state = torch.randn(
                (1, config.block_patches, config.latent_dim), device=device, generator=generator
            )
            dt = (1.0 - config.decode_time) / steps
            for index in range(steps):
                t = 1.0 - index * dt
                times = torch.full((1, config.block_patches), t, device=device)
                first = model.prior.block_velocity(state, times, cache).float()
                predictor = state - dt * first
                second = model.prior.block_velocity(predictor, times - dt, cache).float()
                state = state - (0.5 * dt) * (first + second)
            generated.append(state)
            if block_index + 1 < new_blocks:
                cache = model.prior.append_history(state, cache)
        full = torch.cat([w, *generated], dim=1) if generated else w
        logits = model.decoder(full)[:, w.shape[1] :]
        if new_blocks:
            if temperature == 0:
                sampled = logits.argmax(-1)
            else:
                probabilities = torch.softmax(logits.float() / temperature, dim=-1)
                sampled = torch.multinomial(
                    probabilities.flatten(0, 2), 1, generator=generator
                ).view(probabilities.shape[:-1])
            suffix = bytes(sampled.flatten().to(torch.uint8).cpu().tolist())
        else:
            suffix = b""
    return {
        "prompt_bytes": len(prompt),
        "generated_bytes": len(suffix),
        "generated_hex": suffix.hex(),
        "generated_text": suffix.decode("utf-8", errors="replace"),
        "steps": steps,
        "seed": seed,
        "temperature": temperature,
        "elapsed_seconds": time.perf_counter() - started,
    }
