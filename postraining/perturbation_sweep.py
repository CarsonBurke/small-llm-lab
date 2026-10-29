"""Antithetic parameter-perturbation sweep of the gated-delta memory.

Step 1 of ``docs/memory_write_critic_plan_20261020.md``'s revised order:
before any write critic is built, measure with no critic whether moving the
memory writer (or, separately, the reader's output projection) changes the
expected return detectably, and whether the replay's truncated token-path
gradient predicts that change.

Common random numbers make this cheap. A pool's token draw is counter-keyed
Gumbel-max on (pool seed, request, sample, slot), and the pool seed is the
sampler cursor, so rewinding the cursor and collecting again reproduces the
same prompts and the same noise. With the parameters moved to theta +/- delta
d, a coupled trajectory diverges only where the move flips a near-tie token,
and G(+) - G(-) over the same (prompt, sample) is a low-variance estimate of
2 delta times the directional derivative D = dJ/d epsilon of the return J
along d. The unperturbed re-collect is the null control: it must reproduce
every token and reward.

Directions live in an RMS-relative metric: each target tensor's coordinates
are measured in units of that tensor's own RMS, and every direction is scaled
to metric norm sqrt(n), n the target's element count, so ``delta`` is the
average per-element move relative to its tensor's RMS whatever the kind:

- ``random``: Gaussian, seeded by (seed, target, pool, index).
- ``radial``: the parameters themselves, theta -> (1 +/- delta) theta.
- ``gradient``: the steepest ascent direction of the token-path gradient in
  that metric, P g with P = rms^2 per tensor, where g is the mean of the
  trainer's actor-gradient estimates on the PREVIOUS pools, so its noise is
  independent of the pool it is tested on.

Only weight matrices are targets. The gate biases are excluded: the decay
bias is log(tau - 1), about 5 in RMS, and an RMS-relative move of it would
change memory timescales by orders of magnitude rather than perturb them.

A parameter move shifts every write of every trajectory along one
direction, so it bounds nothing about exploration in write space, which is
what a DDPG-style write critic would learn from. ``run_write_noise_sweep``
measures that directly: the same coupled collects, but each perturbed
collect explores the writes at a signed noise scale (``WriteNoisePlan``),
every sample of a prompt sharing one counter-keyed noise pattern while its
tokens stay independently seeded. For a trajectory, D = (G(+s) - G(-s)) / 2
is the odd part of the return's response to its noise -- the first-order
part a DDPG-style critic's action gradient follows. Its second moment
bounds the variance of the expected odd response from above (it includes
the continuation noise after a flipped token), and the mean product over
distinct samples of one prompt -- which share the noise but not the
continuation noise -- estimates the prompt-level part of it without bias.
The even part C = (G(+s) + G(-s)) / 2 - G(0) gets the same within-prompt
estimator; D never sees it, so neither bound covers a response to the
noise's magnitude.

Everything here except the collection callbacks is CPU-deterministic and
covered by ``tests/test_perturbation_sweep.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from postraining.latent_rollout import (
    PAD_SLOT,
    LatentRolloutBatch,
    WriteNoisePlan,
    emitted_token_rows,
)

PERTURBATION_SWEEP_SCHEMA = "memory_antithetic_crn_parameter_sweep_rms_metric/v1"
WRITE_NOISE_SWEEP_SCHEMA = "memory_antithetic_crn_write_noise_sweep_shared_per_prompt/v1"
PERTURBATION_TARGETS: dict[str, tuple[str, ...]] = {
    "writer": (
        "combiner.writer.key.weight",
        "combiner.writer.value.weight",
        "combiner.writer.write_gate.weight",
        "combiner.writer.decay_gate.weight",
    ),
    "reader_out": ("combiner.reader.out.weight",),
}
DIRECTION_KINDS = ("gradient", "radial", "random")


def target_parameters(
    named_parameters: Mapping[str, Tensor], target: str
) -> dict[str, Tensor]:
    """The target's tensors by name, in the registry's fixed order."""
    if target not in PERTURBATION_TARGETS:
        raise ValueError(f"unknown perturbation target {target!r}")
    missing = [name for name in PERTURBATION_TARGETS[target] if name not in named_parameters]
    if missing:
        raise ValueError(f"perturbation target {target!r} lacks {missing}; is this a memory policy?")
    return {name: named_parameters[name] for name in PERTURBATION_TARGETS[target]}


def tensor_rms(parameters: Mapping[str, Tensor]) -> dict[str, float]:
    """Per-tensor fp64 RMS, reduced on CPU so an offline analysis rebuilds
    every direction bit for bit; a zero tensor has no RMS-relative metric."""
    rms = {
        name: float(value.detach().to("cpu", torch.float64).square().mean().sqrt())
        for name, value in parameters.items()
    }
    zero = [name for name, value in rms.items() if not value > 0.0]
    if zero:
        raise ValueError(f"zero-RMS tensors have no relative metric: {zero}")
    return rms


def metric_norm(direction: Mapping[str, Tensor], rms: Mapping[str, float]) -> float:
    """sqrt(sum over tensors of |d_t / rms_t|^2)."""
    return math.sqrt(sum(
        float(direction[name].double().square().sum()) / rms[name] ** 2 for name in rms
    ))


def _scaled_to_metric(direction: dict[str, Tensor], rms: Mapping[str, float]) -> dict[str, Tensor]:
    elements = sum(direction[name].numel() for name in rms)
    norm = metric_norm(direction, rms)
    if not norm > 0.0:
        raise ValueError("a direction must be nonzero")
    scale = math.sqrt(elements) / norm
    return {name: (direction[name].double() * scale).float() for name in rms}


def direction_seed(seed: int, target: str, pool: int, index: int) -> int:
    """A 63-bit generator seed bound to (seed, target, pool, index) alone."""
    key = f"{PERTURBATION_SWEEP_SCHEMA}/{seed}/{target}/{pool}/{index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") >> 1


def random_direction(
    parameters: Mapping[str, Tensor], seed: int, target: str, pool: int, index: int
) -> dict[str, Tensor]:
    """Isotropic Gaussian in the RMS metric, fp32 on CPU, metric norm sqrt(n)."""
    rms = tensor_rms(parameters)
    generator = torch.Generator(device="cpu").manual_seed(direction_seed(seed, target, pool, index))
    raw = {
        name: torch.randn(value.shape, generator=generator, dtype=torch.float64) * rms[name]
        for name, value in parameters.items()
    }
    return _scaled_to_metric(raw, rms)


def radial_direction(parameters: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """theta itself: metric norm sqrt(n) by construction."""
    rms = tensor_rms(parameters)
    return _scaled_to_metric(
        {name: value.detach().to("cpu", torch.float64) for name, value in parameters.items()}, rms
    )


def gradient_direction(
    parameters: Mapping[str, Tensor], gradient: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """The metric's steepest ascent direction of ``gradient``: P g, P = rms^2."""
    rms = tensor_rms(parameters)
    return _scaled_to_metric(
        {name: gradient[name].to("cpu", torch.float64) * rms[name] ** 2 for name in rms}, rms
    )


def direction_digest(direction: Mapping[str, Tensor]) -> str:
    """sha256 over the fp32 bytes in the target's order: the analysis
    regenerates each direction and must reproduce this."""
    digest = hashlib.sha256()
    for name, value in direction.items():
        digest.update(name.encode())
        digest.update(value.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


def inner_product(gradient: Mapping[str, Tensor], direction: Mapping[str, Tensor]) -> float:
    """fp64 g . d over the direction's tensors."""
    return float(sum(
        (gradient[name].double().cpu() * value.double().cpu()).sum() for name, value in direction.items()
    ))


@contextmanager
def perturbed(
    parameters: Mapping[str, Tensor], direction: Mapping[str, Tensor], step: float
) -> Iterator[None]:
    """Move the live parameters to theta + step * d in place; restore them
    exactly (a copy back, not a subtraction) on exit, however it exits.

    In place keeps every address that compiled or graph-captured code reads,
    and the decode arena refreshes its rounded weight copies from these
    masters at the start of every rollout.
    """
    if set(direction) != set(parameters):
        raise ValueError("a direction must cover exactly the target's tensors")
    with torch.no_grad():
        saved = {name: value.detach().clone() for name, value in parameters.items()}
        try:
            for name, value in parameters.items():
                value.add_(direction[name].to(device=value.device, dtype=value.dtype), alpha=step)
            yield
        finally:
            for name, value in parameters.items():
                value.copy_(saved[name])


def _prompt_digest(batch: LatentRolloutBatch) -> str:
    prompt = batch.token_ids[0, : batch.prompt_length][batch.kind[0, : batch.prompt_length] != PAD_SLOT]
    return hashlib.sha256(prompt.to("cpu", torch.int64).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class PoolTrajectories:
    """A pool's trajectories, prompt-major: group g's sample s is row
    g * samples + s. ``prompts`` holds one token digest per group."""

    prompts: tuple[str, ...]
    samples: int
    tokens: tuple[tuple[int, ...], ...]
    rewards: tuple[float, ...]

    @classmethod
    def from_groups(cls, groups: Sequence[LatentRolloutBatch]) -> PoolTrajectories:
        if not groups:
            raise ValueError("a pool has at least one prompt group")
        samples = groups[0].kind.size(0)
        prompts, tokens, rewards = [], [], []
        for group in groups:
            if group.kind.size(0) != samples:
                raise ValueError("every prompt group has the same sample count")
            prompts.append(_prompt_digest(group))
            tokens.extend(tuple(row) for row in emitted_token_rows(group))
            rewards.extend(float(value) for value in group.reward_scalar.cpu().tolist())
        return cls(tuple(prompts), samples, tuple(tokens), tuple(rewards))

    def compare(self, other: PoolTrajectories) -> dict[str, list]:
        """Per-trajectory coupling against this (base) pool: whether the
        token sequence changed, and the first emitted index where it did
        (-1 when identical). Refuses a pool over different prompts."""
        if other.prompts != self.prompts or other.samples != self.samples:
            raise ValueError("a coupled collect must reproduce the base pool's prompts")
        changed, divergence = [], []
        for base, tokens in zip(self.tokens, other.tokens, strict=True):
            if base == tokens:
                changed.append(False)
                divergence.append(-1)
                continue
            changed.append(True)
            shared = min(len(base), len(tokens))
            index = next((i for i in range(shared) if base[i] != tokens[i]), shared)
            divergence.append(index)
        return {"changed": changed, "first_divergence": divergence}


@dataclass(frozen=True)
class SweepConfig:
    """``gradient_warmup_pools`` pools first only collect and estimate the
    token-path gradient; their mean fixes the one gradient direction every
    later (test) pool is perturbed along, so it is independent of all of
    them and the test pools' D estimates pool directly."""

    targets: tuple[str, ...]
    scales: tuple[float, ...]
    random_directions: int
    gradient_direction: bool
    radial_direction: bool
    seed: int
    gradient_warmup_pools: int = 0

    def __post_init__(self) -> None:
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("perturbation targets must be distinct and nonempty")
        for target in self.targets:
            if target not in PERTURBATION_TARGETS:
                raise ValueError(f"unknown perturbation target {target!r}")
        if not self.scales or any(not (math.isfinite(s) and s > 0.0) for s in self.scales):
            raise ValueError("perturbation scales must be finite and positive")
        if len(set(self.scales)) != len(self.scales):
            raise ValueError("perturbation scales must be distinct")
        if self.random_directions < 0:
            raise ValueError("the random direction count must be nonnegative")
        if not (self.random_directions or self.gradient_direction or self.radial_direction):
            raise ValueError("a perturbation sweep needs at least one direction")
        if self.gradient_direction != (self.gradient_warmup_pools > 0):
            raise ValueError("a gradient direction needs gradient warmup pools, and only it does")

    def test_collects(self) -> int:
        """Coupled collects of one test pool, the null re-collect included."""
        kinds = int(self.gradient_direction) + int(self.radial_direction) + self.random_directions
        return 1 + len(self.targets) * kinds * len(self.scales) * 2


def mean_gradient(gradients: Sequence[Mapping[str, Tensor]], names) -> dict[str, Tensor]:
    """fp64 elementwise mean of per-pool gradient estimates."""
    return {name: torch.stack([g[name].double() for g in gradients]).mean(0) for name in names}


def pool_directions(
    config: SweepConfig,
    target: str,
    parameters: Mapping[str, Tensor],
    pool: int,
    warmup_gradient: Mapping[str, Tensor] | None,
) -> list[tuple[str, int, dict[str, Tensor], list[int]]]:
    """(kind, index, direction, gradient source pools) to test on ``pool``."""
    directions = []
    if config.gradient_direction:
        if warmup_gradient is None:
            raise ValueError("the gradient direction needs the warmup pools' mean gradient")
        directions.append(("gradient", 0, gradient_direction(parameters, warmup_gradient),
                           list(range(config.gradient_warmup_pools))))
    if config.radial_direction:
        directions.append(("radial", 0, radial_direction(parameters), []))
    for index in range(config.random_directions):
        directions.append(("random", index, random_direction(
            parameters, config.seed, target, pool, index), []))
    return directions


def parameter_digest(targets: Mapping[str, Mapping[str, Tensor]]) -> str:
    """sha256 over every target tensor's live bytes: equal before and after
    a pool's perturbations, or the restore failed."""
    return direction_digest({
        f"{target}/{name}": value for target, parameters in targets.items()
        for name, value in parameters.items()
    })


class SweepWriter:
    """The sweep's canonical artifact under ``root``: ``manifest.json``, one
    ``collects.jsonl`` record per collect (append-only, flushed per record),
    and each pool's token-path gradient estimate in ``gradients/`` (fp64,
    the precision the gradient directions are built in)."""

    def __init__(self, root: Path, manifest: dict):
        self.root = Path(root)
        if self.root.exists():
            raise ValueError(f"{self.root} exists; a sweep artifact is written once")
        self.root.mkdir(parents=True)
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        self.collects = self.root / "collects.jsonl"

    def record(self, **values) -> None:
        with self.collects.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, sort_keys=True) + "\n")

    def save_gradient(self, pool: int, gradient: Mapping[str, Tensor]) -> Path:
        path = self.root / "gradients" / f"pool_{pool:04d}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save({name: value.detach().to("cpu", torch.float64) for name, value in gradient.items()}, path)
        return path


def run_sweep(
    config: SweepConfig,
    targets: Mapping[str, Mapping[str, Tensor]],
    pools: int,
    cursor: Callable[[], int],
    collect_at: Callable[[int], list[LatentRolloutBatch]],
    token_gradient: Callable[[list[LatentRolloutBatch]], tuple[dict[str, Tensor], dict]] | None,
    writer: SweepWriter,
    out_of_time: Callable[[float], bool],
    log: Callable[[dict], None],
) -> int:
    """Run up to ``pools`` pools (warmup pools included); return how many
    completed.

    A warmup pool is one base collect and its token-path gradient estimate.
    A test pool is the base collect, its gradient estimate (when a gradient
    direction is requested: the same-pool prediction g . d), the null
    re-collect, then for every target, direction, scale and sign a coupled
    collect at the same cursor. ``collect_at(cursor)`` rewinds the sampler
    and collects; ``token_gradient(groups)`` returns the actor's estimate of
    grad J at every target tensor and its telemetry. Before each pool,
    ``out_of_time(seconds)`` is asked with that pool's predicted duration
    (the slowest steady-state collect and gradient so far, times the pool's
    count), so the sweep stops at a pool boundary.
    """
    if config.gradient_direction and token_gradient is None:
        raise ValueError("a gradient direction needs the token-path gradient")
    warmup: list[dict[str, Tensor]] = []
    warmup_gradient = None
    names = [name for parameters in targets.values() for name in parameters]
    collect_seconds: list[float] = []
    gradient_seconds = 0.0
    completed = 0
    for pool in range(pools):
        is_warmup = pool < config.gradient_warmup_pools
        # The first collect compiles; it predicts nothing.
        per_collect = max(collect_seconds[1:], default=collect_seconds[0] if collect_seconds else 0.0)
        estimate = (1 + (0 if is_warmup else config.test_collects())) * per_collect + gradient_seconds
        if out_of_time(estimate):
            log({"type": "perturbation_stop", "pool": pool, "reason": "time",
                 "estimate_seconds": estimate})
            break
        started = time.perf_counter()
        start = cursor()
        before = parameter_digest(targets)
        record = lambda **values: writer.record(pool=pool, cursor=start, **values)

        collect_started = time.perf_counter()
        groups = collect_at(start)
        base = PoolTrajectories.from_groups(groups)
        collect_seconds.append(time.perf_counter() - collect_started)
        record(kind="base", warmup=is_warmup, prompts=list(base.prompts), samples=base.samples,
               rewards=list(base.rewards), lengths=[len(row) for row in base.tokens],
               seconds=collect_seconds[-1])
        gradient = None
        if config.gradient_direction:
            gradient_started = time.perf_counter()
            gradient, stats = token_gradient(groups)
            gradient_seconds = max(gradient_seconds, time.perf_counter() - gradient_started)
            writer.save_gradient(pool, gradient)
            record(kind="token_gradient", warmup=is_warmup, **stats)
        del groups
        summary = {"type": "perturbation_pool", "pool": pool, "cursor": start, "warmup": is_warmup,
                   "base_reward_mean": sum(base.rewards) / len(base.rewards)}
        if is_warmup:
            warmup.append(gradient)
            if len(warmup) == config.gradient_warmup_pools:
                warmup_gradient = mean_gradient(warmup, names)
        else:
            def coupled(**identity) -> dict:
                collect_started = time.perf_counter()
                other = PoolTrajectories.from_groups(collect_at(start))
                collect_seconds.append(time.perf_counter() - collect_started)
                comparison = base.compare(other)
                record(**identity, rewards=list(other.rewards), **comparison,
                       seconds=collect_seconds[-1])
                return comparison

            summary["null_changed"] = sum(coupled(kind="null")["changed"])
            for target in config.targets:
                parameters = targets[target]
                for kind, index, direction, sources in pool_directions(
                    config, target, parameters, pool, warmup_gradient
                ):
                    identity = dict(
                        kind=kind, target=target, index=index,
                        direction_sha256=direction_digest(direction),
                        gradient_source_pools=sources,
                        # The token path's g . d from this pool's own
                        # estimate, independent of d (built from other pools).
                        same_pool_token_prediction=(
                            None if gradient is None else inner_product(gradient, direction)
                        ),
                    )
                    for scale in config.scales:
                        for sign in (1, -1):
                            with perturbed(parameters, direction, sign * scale):
                                comparison = coupled(**identity, scale=scale, sign=sign)
                            changed = comparison["changed"]
                            summary[f"{target}/{kind}{index}/{scale:g}/{sign:+d}/changed"] = (
                                sum(changed) / len(changed)
                            )
        if parameter_digest(targets) != before:
            raise RuntimeError(f"pool {pool} did not restore the target parameters exactly")
        summary["seconds"] = time.perf_counter() - started
        log(summary)
        completed += 1
    return completed


@dataclass(frozen=True)
class WriteNoiseSweepConfig:
    """Per pool: a noiseless base collect, its null re-collect, then a
    +scale and a -scale collect at every scale, each sharing one noise
    pattern per prompt."""

    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.scales or any(not (math.isfinite(s) and s > 0.0) for s in self.scales):
            raise ValueError("write noise scales must be finite and positive")
        if len(set(self.scales)) != len(self.scales):
            raise ValueError("write noise scales must be distinct")

    def collects(self) -> int:
        return 2 + 2 * len(self.scales)


def run_write_noise_sweep(
    config: WriteNoiseSweepConfig,
    pools: int,
    cursor: Callable[[], int],
    collect_at: Callable[[int, WriteNoisePlan], list[LatentRolloutBatch]],
    writer: SweepWriter,
    out_of_time: Callable[[float], bool],
    log: Callable[[dict], None],
) -> int:
    """Run up to ``pools`` write-noise pools; return how many completed.

    ``collect_at(cursor, plan)`` rewinds the sampler and collects under
    ``plan``. The base and null collects explore at scale zero through the
    same rollout path, so the null proves the coupling: every token must
    reproduce, checked before any explored collect. Rewards need not (code
    tests run under wall-clock timeouts); their shift is logged. Pools stop at a boundary when ``out_of_time``
    says the next one (the slowest steady-state collect so far, times the
    pool's count) would not fit.
    """
    collect_seconds: list[float] = []
    completed = 0
    for pool in range(pools):
        per_collect = max(collect_seconds[1:], default=collect_seconds[0] if collect_seconds else 0.0)
        estimate = config.collects() * per_collect
        if out_of_time(estimate):
            log({"type": "write_noise_stop", "pool": pool, "reason": "time",
                 "estimate_seconds": estimate})
            break
        started = time.perf_counter()
        start = cursor()
        record = lambda **values: writer.record(pool=pool, cursor=start, **values)

        def collected(scale: float) -> tuple[PoolTrajectories, float]:
            collect_started = time.perf_counter()
            groups = collect_at(start, WriteNoisePlan(scale, shared_per_prompt=True))
            trajectories = PoolTrajectories.from_groups(groups)
            collect_seconds.append(time.perf_counter() - collect_started)
            return trajectories, collect_seconds[-1]

        base, seconds = collected(0.0)
        record(kind="base", prompts=list(base.prompts), samples=base.samples,
               rewards=list(base.rewards), lengths=[len(row) for row in base.tokens],
               seconds=seconds)
        summary = {"type": "write_noise_pool", "pool": pool, "cursor": start,
                   "base_reward_mean": sum(base.rewards) / len(base.rewards)}
        for kind, scale, sign in [("null", 0.0, 0)] + [
            ("write_noise", scale, sign) for scale in config.scales for sign in (1, -1)
        ]:
            other, seconds = collected(sign * scale)
            comparison = base.compare(other)
            identity = {} if kind == "null" else {"scale": scale, "sign": sign}
            record(kind=kind, **identity, rewards=list(other.rewards), **comparison,
                   seconds=seconds)
            changed = comparison["changed"]
            key = "null" if kind == "null" else f"{scale:g}/{sign:+d}"
            summary[f"{key}/changed"] = sum(changed) / len(changed)
            summary[f"{key}/reward_mean"] = sum(other.rewards) / len(other.rewards)
            if kind == "null":
                if any(changed):
                    # Before any explored collect: the failed pool stays
                    # incomplete, so the analysis never reads it.
                    raise RuntimeError(f"pool {pool}'s null re-collect did not reproduce the base")
                # Identical tokens can still score differently (wall-clock
                # code-test timeouts): verifier noise inside every D.
                summary["null/reward_shift"] = max(
                    abs(a - b) for a, b in zip(base.rewards, other.rewards))
        summary["seconds"] = time.perf_counter() - started
        log(summary)
        completed += 1
    return completed
