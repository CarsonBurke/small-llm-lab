"""Numerical latent-density evaluation and cached token sampling for Cola.

The reported rate is a Monte Carlo negative-ELBO *estimate*, not an exact
marginal token likelihood or a certified bound after ODE discretization. Density
conditioning consists of sampled clean predecessor latents held fixed throughout
the solve. Time is normalized to [0, 1]; the model owns its embedding scale.
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
    from pretraining.cola.model import ColaModel

_LOG_2PI = math.log(2 * math.pi)
_LOG_2 = math.log(2)


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _standard_normal_log_prob(value: Tensor) -> Tensor:
    return -0.5 * (value.float().square() + _LOG_2PI).flatten(1).sum(1)


def integrate_log_density(
    velocity: Callable[[Tensor, Tensor, Tensor], Tensor],
    clean: Tensor,
    initial: Tensor,
    *,
    steps: int,
    probe_vectors: Tensor,
) -> dict:
    """Integrate a conditional flow and its density from clean time 0 to noise 1.

    ``velocity(clean, state, times)`` consumes [B,T,D] coordinates and [B,T]
    normalized times. ``probe_vectors`` has shape [P,B,T,D]; callers supply
    independent standard Gaussian probes for the Hutchinson estimator. Probes
    and detached clean conditioning remain fixed at *every* Heun stage. This
    model-independent helper also accepts CPU analytic flows for mathematical
    tests; neural callers must enforce their compiled CUDA execution contract.

    Returns per-example ``log_prob`` = log N(state_at_1) + integral div(v) ds,
    ``divergence_integral``, the terminal ``state``, and ``nfe``. State, traces,
    and reductions use FP32. No graph is retained across solver evaluations.
    Finite-step Heun error and finite-probe error are not uncertainty bounds.
    """
    _positive_integer("steps", steps)
    if torch.is_inference_mode_enabled():
        raise RuntimeError(
            "density divergence requires input gradients, not inference_mode"
        )
    if initial.ndim != 3 or min(initial.shape) < 1 or clean.shape != initial.shape:
        raise ValueError("clean and initial must share nonempty [B,T,D] shapes")
    if probe_vectors.ndim != 4 or probe_vectors.shape[1:] != initial.shape:
        raise ValueError("probe_vectors must have shape [P,B,T,D]")
    _positive_integer("probe count", probe_vectors.shape[0])
    if clean.device != initial.device or probe_vectors.device != initial.device:
        raise ValueError("conditioning, state, and probes must share a device")
    state = initial.detach().float()
    history = clean.detach().float()
    probes = probe_vectors.detach().float()
    integrated_trace = torch.zeros(
        initial.shape[0], device=initial.device, dtype=torch.float32
    )
    dt = 1.0 / steps

    def evaluate(value: Tensor, normalized_time: float) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            differentiable = value.detach().requires_grad_(True)
            times = torch.full(
                value.shape[:2],
                normalized_time,
                device=value.device,
                dtype=torch.float32,
            )
            result = velocity(history, differentiable, times).float()
            if result.shape != value.shape:
                raise ValueError("velocity must preserve the [B,T,D] state shape")
            trace = torch.zeros_like(integrated_trace)
            if result.requires_grad:
                for index, probe in enumerate(probes):
                    derivative = torch.autograd.grad(
                        result,
                        differentiable,
                        grad_outputs=probe,
                        retain_graph=index + 1 < probes.shape[0],
                        create_graph=False,
                        allow_unused=True,
                    )[0]
                    if derivative is not None:
                        trace = trace + (derivative.float() * probe).flatten(1).sum(1)
                trace = trace / probes.shape[0]
            return result.detach(), trace.detach()

    for index in range(steps):
        first_velocity, first_trace = evaluate(state, index / steps)
        predictor = (state + dt * first_velocity).detach()
        second_velocity, second_trace = evaluate(predictor, (index + 1) / steps)
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


def _model_device(model: ColaModel) -> torch.device:
    if not model.compiled:
        raise ValueError("call model.compile_components() before evaluation")
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("Cola neural evaluation requires CUDA; no CPU fallback")
    if any(value.device != device for value in (*model.parameters(), *model.buffers())):
        raise ValueError("all model tensors must share one CUDA device")
    if torch.is_autocast_enabled("cuda"):
        raise ValueError("Cola uses explicit BF16 neural execution, not autocast")
    return device


@contextmanager
def _evaluation_mode(model: ColaModel) -> Iterator[None]:
    """Freeze weights, not inputs, and restore even heterogeneous module modes."""
    parameter_flags = [
        (parameter, parameter.requires_grad) for parameter in model.parameters()
    ]
    module_flags = [(module, module.training) for module in model.modules()]
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


def estimate_negative_elbo(
    model: ColaModel,
    ids: Tensor,
    *,
    steps: int,
    seed: int,
    probes: int = 1,
    source_bytes: int,
) -> dict:
    """Estimate aligned token reconstruction + conditional-prior KL in bits/byte.

    A private device generator draws one posterior sample, then the fixed
    Gaussian probes; it never reseeds or consumes the training generator. The
    density and aligned reconstruction are evaluated at that same posterior
    sample, never at the posterior mode. ``prior_nats_per_byte``
    means *negative* log p / bytes, whereas ``posterior_logprob_nats_per_byte``
    is signed log q / bytes. These are continuous latent densities, not discrete
    code lengths, and either contribution may be negative. Totals are sums over
    the entire batch (no padding/ignored-token convention). Each row contains
    exactly the configured number of positions, including when that configuration
    represents a shorter final context. ``source_bytes`` is the exact raw-byte
    count represented by the entire batch, supplied by the caller's data codec;
    token counts are never used as a substitute.

    The posterior seed and the probe count jointly identify the random draws.
    Heun solves all block conditionals together, with the sampled clean history
    fixed; it does not replace predecessors with their noisy ODE coordinates.
    This routine enables first-order input gradients and rejects inference_mode.
    """
    _positive_integer("steps", steps)
    _positive_integer("probes", probes)
    _positive_integer("source_bytes", source_bytes)
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if torch.is_inference_mode_enabled():
        raise RuntimeError(
            "density divergence requires input gradients, not inference_mode"
        )
    device = _model_device(model)
    if ids.device != device or ids.ndim != 2 or ids.shape[0] < 1:
        raise ValueError("ids must be a nonempty [B,T] batch on the model CUDA device")
    if ids.shape[1] != model.config.seq_len:
        raise ValueError("density evaluation requires a complete configured sequence")
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("ids must be integer token indices")
    if bool(((ids < 0) | (ids >= model.config.vocab_size)).any()):
        raise ValueError("evaluation targets must be within the configured vocabulary")
    generator = torch.Generator(device=device).manual_seed(seed)
    with _evaluation_mode(model):
        with torch.no_grad():
            mean, logvar = model.vae.encode(ids)
            mean, logvar = mean.float(), logvar.float()
            noise = torch.randn(
                mean.shape, device=device, dtype=torch.float32, generator=generator
            )
            sample = mean + (0.5 * logvar).exp() * noise
            posterior_logprob = (
                -0.5
                * (_LOG_2PI + logvar + (sample - mean).square() * (-logvar).exp()).sum()
            )
            logits = model.vae.decode(sample)
            reconstruction_nll = F.cross_entropy(
                logits.float().reshape(-1, model.config.vocab_size),
                ids.long().reshape(-1),
                reduction="sum",
            )
            probe_vectors = torch.randn(
                (probes, *sample.shape),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
        integrated = integrate_log_density(
            model.prior, sample, sample, steps=steps, probe_vectors=probe_vectors
        )
        prior_logprob = integrated["log_prob"].sum()
        negative_elbo = reconstruction_nll - prior_logprob + posterior_logprob
        values = (
            torch.stack(
                (reconstruction_nll, prior_logprob, posterior_logprob, negative_elbo)
            )
            .detach()
            .cpu()
            .tolist()
        )
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError(
            "non-finite negative-ELBO estimate; no rate can be reported"
        )
    reconstruction, prior, posterior, total = values
    positions = ids.numel()
    return {
        "bytes": source_bytes,
        "positions": positions,
        "batch_size": ids.shape[0],
        "reconstruction_nll_nats": reconstruction,
        "prior_logprob_nats": prior,
        "prior_nll_nats": -prior,
        "posterior_logprob_nats": posterior,
        "negative_elbo_estimate_nats": total,
        "reconstruction_bits_per_byte": reconstruction / (source_bytes * _LOG_2),
        "prior_nats_per_byte": -prior / source_bytes,
        "posterior_logprob_nats_per_byte": posterior / source_bytes,
        "negative_elbo_estimate_bpb": total / (source_bytes * _LOG_2),
        "steps": steps,
        "probes": probes,
        "seed": seed,
        "posterior_seed": seed,
        "probe_distribution": "standard_gaussian_fixed_per_solve",
        "solver": "heun",
        "time_interval": [0.0, 1.0],
        "nfe": integrated["nfe"],
        "prior_calls": integrated["nfe"],
        "prior_positions": integrated["nfe"] * 2 * positions,
        "rate_label": "negative_elbo_estimate_bpb",
        "exact_marginal_bpb": False,
    }


def generate_tokens(
    model: ColaModel,
    prompt_ids: list[int],
    *,
    new_tokens: int,
    steps: int = 16,
    seed: int = 1337,
    temperature: float = 1.0,
) -> dict:
    """Sample token suffixes using cached prior blocks and a latent-causal VAE.

    Heun integrates dz/ds=v from s=1 to 0. Complete prompt blocks use posterior
    modes in the clean cache. In a partial prompt block, known latent coordinates
    follow a fixed-noise linear bridge at both Heun stages and are pinned exactly
    to their posterior modes at time zero. This repaint-style conditioning is a
    sampling heuristic, not an exact conditional draw. Prompt IDs are never
    decoded/resampled/replaced. Every decoder call sees all preceding latents.

    The configured vocabulary is sampled categorically, without CFG (scale 1).
    Zero temperature is greedy. Returned IDs are exact; text conversion and
    raw-byte accounting belong to the caller's data codec.
    ``prior_positions`` counts new query/token positions actually processed by
    append/velocity calls, including fixed and surplus positions in full blocks;
    reused cached keys are not recomputed positions. The final unused clean cache
    append is omitted. The decoder is deliberately uncached for short samples.
    Requests exceeding the configured context are rejected rather than truncated.
    """
    if not isinstance(prompt_ids, list) or any(
        type(identity) is not int for identity in prompt_ids
    ):
        raise TypeError("prompt_ids must be a list of integer token indices")
    if any(
        identity < 0 or identity >= model.config.vocab_size for identity in prompt_ids
    ):
        raise ValueError("prompt_ids must be within the configured vocabulary")
    if type(new_tokens) is not int or new_tokens < 0:
        raise ValueError("new_tokens must be a nonnegative integer")
    _positive_integer("steps", steps)
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be finite and nonnegative")
    if len(prompt_ids) + new_tokens > model.config.seq_len:
        raise ValueError("prompt plus suffix exceeds the configured latent context")
    generated: list[int] = []
    velocity_calls = append_calls = prior_positions = decoder_positions = 0
    encoder_positions = 0
    elapsed = prefill_seconds = decode_seconds = 0.0
    if new_tokens:
        device = _model_device(model)
        generator = torch.Generator(device=device).manual_seed(seed)
        block_size = model.config.block_size
        latent_dim = model.config.latent_dim
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with _evaluation_mode(model), torch.no_grad():
            cache = None
            history = torch.empty(
                (1, 0, latent_dim), device=device, dtype=torch.float32
            )
            if prompt_ids:
                prompt_tensor = torch.tensor(
                    prompt_ids, device=device, dtype=torch.long
                )[None]
                prompt_latents, _ = model.vae.encode(prompt_tensor)
                prompt_latents = prompt_latents.float()
                encoder_positions = len(prompt_ids)
            else:
                prompt_latents = history
            complete_prefix = len(prompt_ids) // block_size * block_size
            for start in range(0, complete_prefix, block_size):
                cache = model.prior.append_clean(
                    prompt_latents[:, start : start + block_size], cache
                )
                append_calls += 1
                prior_positions += block_size
            history = prompt_latents[:, :complete_prefix]
            known = prompt_latents[:, complete_prefix:]
            torch.cuda.synchronize(device)
            decode_started = time.perf_counter()
            prefill_seconds = decode_started - started
            while len(generated) < new_tokens:
                state = torch.randn(
                    (1, block_size, latent_dim),
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
                fixed_count = known.shape[1]
                fixed_noise = state[:, :fixed_count].clone()

                def paint(
                    value: Tensor,
                    normalized_time: float,
                    fixed_count: int = fixed_count,
                    known: Tensor = known,
                    fixed_noise: Tensor = fixed_noise,
                ) -> Tensor:
                    if fixed_count:
                        value[:, :fixed_count] = (
                            1 - normalized_time
                        ) * known + normalized_time * fixed_noise
                    return value

                state = paint(state, 1.0)
                for index in range(steps):
                    current_time = 1.0 - index / steps
                    next_time = 1.0 - (index + 1) / steps
                    current_times = torch.full(
                        (1, block_size),
                        current_time,
                        device=device,
                        dtype=torch.float32,
                    )
                    first = model.prior.block_velocity(
                        state, current_times, cache
                    ).float()
                    velocity_calls += 1
                    prior_positions += block_size
                    predictor = paint(state - first / steps, next_time)
                    next_times = torch.full(
                        (1, block_size), next_time, device=device, dtype=torch.float32
                    )
                    second = model.prior.block_velocity(
                        predictor, next_times, cache
                    ).float()
                    velocity_calls += 1
                    prior_positions += block_size
                    state = paint(
                        state - (0.5 / steps) * (first + second), next_time
                    ).detach()
                take = min(block_size - fixed_count, new_tokens - len(generated))
                context = torch.cat((history, state[:, : fixed_count + take]), dim=1)
                logits = model.vae.decode(context)[
                    :, history.shape[1] + fixed_count :
                ].float()
                decoder_positions += context.shape[1]
                if temperature == 0:
                    identities = logits.argmax(dim=-1)
                else:
                    probabilities = (logits / temperature).softmax(dim=-1)
                    identities = torch.multinomial(
                        probabilities.reshape(-1, model.config.vocab_size),
                        1,
                        generator=generator,
                    ).reshape(1, take)
                generated.extend(identities.flatten().cpu().tolist())
                if len(generated) < new_tokens:
                    cache = model.prior.append_clean(state, cache)
                    append_calls += 1
                    prior_positions += block_size
                    history = context
                    known = known[:, :0]
            torch.cuda.synchronize(device)
            finished = time.perf_counter()
            decode_seconds = finished - decode_started
            elapsed = finished - started
    output = prompt_ids + generated
    return {
        "prompt_ids": prompt_ids.copy(),
        "generated_ids": generated,
        "output_ids": output,
        "prompt_tokens": len(prompt_ids),
        "generated_count": len(generated),
        "total_tokens": len(output),
        "requested_new_tokens": new_tokens,
        "prior_calls": velocity_calls + append_calls,
        "prior_velocity_calls": velocity_calls,
        "prior_append_calls": append_calls,
        "prior_positions": prior_positions,
        "encoder_positions": encoder_positions,
        "decoder_positions": decoder_positions,
        "nfe": velocity_calls,
        "steps": steps,
        "seed": seed,
        "temperature": float(temperature),
        "cfg_scale": 1.0,
        "solver": "heun",
        "elapsed_seconds": elapsed,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "tokens_per_second": len(generated) / elapsed if elapsed > 0 else 0.0,
        "timing_includes_prompt_and_compilation": True,
        "partial_prefix_conditioning": "fixed_noise_linear_bridge_repaint",
        "exact_conditional_sampling": False,
    }
