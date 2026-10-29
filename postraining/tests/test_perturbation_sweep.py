"""CPU contracts of the antithetic memory perturbation sweep."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from postraining.latent_rollout import PAD_SLOT, TOKEN_SLOT
from postraining.perturbation_sweep import (
    PERTURBATION_TARGETS,
    PoolTrajectories,
    SweepConfig,
    SweepWriter,
    WriteNoiseSweepConfig,
    direction_digest,
    direction_seed,
    gradient_direction,
    inner_product,
    metric_norm,
    perturbed,
    radial_direction,
    random_direction,
    run_sweep,
    run_write_noise_sweep,
    target_parameters,
    tensor_rms,
)
from postraining.vapo.config import (
    build_arg_parser,
    validate_perturbation_sweep,
    validate_write_noise_sweep,
)


def writer_parameters(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shapes = {
        "combiner.writer.key.weight": (8, 6),
        "combiner.writer.value.weight": (8, 6),
        "combiner.writer.write_gate.weight": (2, 6),
        "combiner.writer.decay_gate.weight": (2, 6),
    }
    scales = {"key": 0.4, "value": 0.4, "write_gate": 0.004, "decay_gate": 0.003}
    return {
        name: torch.nn.Parameter(
            torch.randn(shape, generator=generator) * scales[name.split(".")[2]]
        )
        for name, shape in shapes.items()
    }


def elements(parameters) -> int:
    return sum(value.numel() for value in parameters.values())


def test_target_parameters_take_registry_names_and_refuse_missing():
    named = {**writer_parameters(), "combiner.reader.out.weight": torch.ones(3, 8)}
    assert list(target_parameters(named, "writer")) == list(PERTURBATION_TARGETS["writer"])
    assert list(target_parameters(named, "reader_out")) == ["combiner.reader.out.weight"]
    with pytest.raises(ValueError, match="lacks"):
        target_parameters({"combiner.reader.out.weight": torch.ones(1)}, "writer")
    with pytest.raises(ValueError, match="unknown"):
        target_parameters(named, "reader")


def test_zero_tensor_has_no_relative_metric():
    with pytest.raises(ValueError, match="zero-RMS"):
        tensor_rms({"w": torch.zeros(3)})


def test_perturbed_moves_by_step_and_restores_bitwise_even_on_error():
    parameters = writer_parameters()
    original = {name: value.detach().clone() for name, value in parameters.items()}
    direction = random_direction(parameters, 7, "writer", 0, 0)
    with perturbed(parameters, direction, -0.25):
        for name, value in parameters.items():
            torch.testing.assert_close(value.detach(), original[name] - 0.25 * direction[name])
    for name, value in parameters.items():
        assert torch.equal(value.detach(), original[name])
    with pytest.raises(RuntimeError, match="boom"):
        with perturbed(parameters, direction, 1e3):
            raise RuntimeError("boom")
    for name, value in parameters.items():
        assert torch.equal(value.detach(), original[name])
    with pytest.raises(ValueError, match="exactly"):
        with perturbed(parameters, {"combiner.writer.key.weight": direction["combiner.writer.key.weight"]}, 1.0):
            pass


def test_direction_seed_is_deterministic_and_keyed_on_every_field():
    seeds = {
        direction_seed(seed, target, pool, index)
        for seed in (1, 2) for target in ("writer", "reader_out")
        for pool in (0, 1) for index in (0, 1)
    }
    assert len(seeds) == 16
    assert direction_seed(1337, "writer", 3, 1) == direction_seed(1337, "writer", 3, 1)
    assert all(0 <= seed < 2**63 for seed in seeds)


def test_random_direction_reproduces_and_has_metric_norm_sqrt_n():
    parameters = writer_parameters()
    rms = tensor_rms(parameters)
    first = random_direction(parameters, 1337, "writer", 2, 1)
    again = random_direction(parameters, 1337, "writer", 2, 1)
    assert direction_digest(first) == direction_digest(again)
    assert direction_digest(first) != direction_digest(random_direction(parameters, 1337, "writer", 2, 0))
    assert direction_digest(first) != direction_digest(random_direction(parameters, 1337, "writer", 3, 1))
    assert metric_norm(first, rms) == pytest.approx(math.sqrt(elements(parameters)), rel=1e-6)
    # RMS-relative: each tensor's coordinates are on its own tensor's scale.
    gate = first["combiner.writer.write_gate.weight"].abs().mean()
    key = first["combiner.writer.key.weight"].abs().mean()
    assert 10 < float(key / gate) < 1000


def test_radial_direction_is_theta():
    parameters = writer_parameters()
    radial = radial_direction(parameters)
    for name, value in parameters.items():
        torch.testing.assert_close(radial[name], value.detach(), rtol=1e-6, atol=0)


def test_gradient_direction_is_the_metric_steepest_ascent():
    parameters = writer_parameters()
    rms = tensor_rms(parameters)
    generator = torch.Generator().manual_seed(3)
    gradient = {name: torch.randn(value.shape, generator=generator) for name, value in parameters.items()}
    direction = gradient_direction(parameters, gradient)
    n = elements(parameters)
    assert metric_norm(direction, rms) == pytest.approx(math.sqrt(n), rel=1e-6)
    # g . d at metric norm sqrt(n) is maximal: sqrt(n) * |P^1/2 g|.
    best = math.sqrt(n) * math.sqrt(sum(
        float((gradient[name].double() * rms[name]).square().sum()) for name in rms))
    assert inner_product(gradient, direction) == pytest.approx(best, rel=1e-5)
    for index in range(20):
        other = random_direction(parameters, 11, "writer", 0, index)
        assert inner_product(gradient, other) < best
    # Proportional to rms^2 g per tensor.
    ratios = [
        float((direction[name].double() / (gradient[name].double() * rms[name] ** 2)).mean())
        for name in rms
    ]
    assert max(ratios) == pytest.approx(min(ratios), rel=1e-5)


def fake_group(prompt: list[int], completions: list[list[int]], rewards: list[float]):
    width = max(len(row) for row in completions)
    rows = len(completions)
    prompt_length = len(prompt) + 1  # one left pad
    kind = torch.full((rows, prompt_length + width), PAD_SLOT)
    tokens = torch.zeros((rows, prompt_length + width), dtype=torch.long)
    kind[:, 1:prompt_length] = TOKEN_SLOT
    tokens[:, 1:prompt_length] = torch.tensor(prompt)
    for row, completion in enumerate(completions):
        kind[row, prompt_length:prompt_length + len(completion)] = TOKEN_SLOT
        tokens[row, prompt_length:prompt_length + len(completion)] = torch.tensor(completion)
    return SimpleNamespace(kind=kind, token_ids=tokens, prompt_length=prompt_length,
                           reward_scalar=torch.tensor(rewards))


def test_pool_trajectories_compare_marks_changed_rows_and_first_divergence():
    base = PoolTrajectories.from_groups([
        fake_group([5, 6], [[1, 2, 3], [1, 2]], [1.0, 0.0]),
        fake_group([7], [[4], [4, 4]], [0.5, 0.5]),
    ])
    assert base.samples == 2 and base.rewards == (1.0, 0.0, 0.5, 0.5)
    other = PoolTrajectories.from_groups([
        fake_group([5, 6], [[1, 9, 3], [1, 2]], [0.0, 0.0]),
        fake_group([7], [[4], [4, 4, 4]], [0.5, 1.0]),
    ])
    comparison = base.compare(other)
    assert comparison == {"changed": [True, False, False, True], "first_divergence": [1, -1, -1, 2]}
    assert base.compare(base) == {"changed": [False] * 4, "first_divergence": [-1] * 4}
    shifted = PoolTrajectories.from_groups([
        fake_group([5, 8], [[1, 2, 3], [1, 2]], [1.0, 0.0]),
        fake_group([7], [[4], [4, 4]], [0.5, 0.5]),
    ])
    with pytest.raises(ValueError, match="prompts"):
        base.compare(shifted)


def test_sweep_config_rejects_empty_or_invalid():
    with pytest.raises(ValueError):
        SweepConfig((), (0.5,), 1, False, False, 0)
    with pytest.raises(ValueError):
        SweepConfig(("writer",), (0.0,), 1, False, False, 0)
    with pytest.raises(ValueError):
        SweepConfig(("writer",), (0.5,), 0, False, False, 0)
    with pytest.raises(ValueError):
        SweepConfig(("writer", "writer"), (0.5,), 1, False, False, 0)
    with pytest.raises(ValueError, match="warmup"):
        SweepConfig(("writer",), (0.5,), 0, True, False, 0)
    with pytest.raises(ValueError, match="warmup"):
        SweepConfig(("writer",), (0.5,), 1, False, False, 0, gradient_warmup_pools=2)


def toy_policy(parameters, sampler, samples=4, prompts=2):
    """Each sample's reward is 1 when w . x_s exceeds a per-(cursor, sample)
    threshold: the analogue of a coupled near-tie flip."""
    collected_at = []

    def collect_at(cursor):
        collected_at.append(cursor)
        sampler.cursor = cursor + prompts
        weight = parameters["combiner.reader.out.weight"].detach()
        groups = []
        for prompt in range(prompts):
            generator = torch.Generator().manual_seed(1000 * cursor + prompt)
            features = torch.randn(samples, 4, generator=generator)
            thresholds = torch.randn(samples, generator=generator)
            wins = (features @ weight > thresholds).float()
            groups.append(fake_group([cursor + prompt], [[int(w) + 1] for w in wins.tolist()], wins.tolist()))
        return groups

    return collect_at, collected_at


def test_run_sweep_couples_restores_and_fixes_the_gradient_direction_after_warmup(tmp_path):
    parameters = {"combiner.reader.out.weight": torch.nn.Parameter(torch.full((4,), 0.5))}
    original = parameters["combiner.reader.out.weight"].detach().clone()
    sampler = SimpleNamespace(cursor=0)
    collect_at, collected_at = toy_policy(parameters, sampler)
    gradient_calls = []

    def token_gradient(groups):
        gradient_calls.append(len(groups))
        # Always taken at the unperturbed parameters.
        assert torch.equal(parameters["combiner.reader.out.weight"].detach(), original)
        return {"combiner.reader.out.weight": torch.tensor([1.0, 0.0, 0.0, float(len(gradient_calls))])}, {
            "trajectories": 8}

    writer = SweepWriter(tmp_path / "sweep", {"schema": "test"})
    config = SweepConfig(("reader_out",), (0.5, 0.1), 2, True, True, 1337, gradient_warmup_pools=2)
    assert config.test_collects() == 1 + 4 * 2 * 2
    logged = []
    completed = run_sweep(
        config, {"reader_out": parameters}, 4, cursor=lambda: sampler.cursor,
        collect_at=collect_at, token_gradient=token_gradient, writer=writer,
        out_of_time=lambda estimate: False, log=logged.append,
    )
    assert completed == 4
    assert torch.equal(parameters["combiner.reader.out.weight"].detach(), original)
    records = [json.loads(line) for line in (tmp_path / "sweep" / "collects.jsonl").read_text().splitlines()]
    per_pool = {}
    for record in records:
        per_pool.setdefault(record["pool"], []).append(record)
    assert [len(per_pool[pool]) for pool in range(4)] == [2, 2, 3 + 16, 3 + 16]
    assert collected_at == [0, 2] + [4] * 18 + [6] * 18
    assert sampler.cursor == 8 and gradient_calls == [2, 2, 2, 2]
    for pool in (0, 1):
        assert [r["kind"] for r in per_pool[pool]] == ["base", "token_gradient"]
        assert per_pool[pool][0]["warmup"]
    for pool in (2, 3):
        kinds = [record["kind"] for record in per_pool[pool]]
        assert kinds[:3] == ["base", "token_gradient", "null"]
        null = per_pool[pool][2]
        assert not any(null["changed"])
        assert null["rewards"] == per_pool[pool][0]["rewards"]
    gradient_records = [r for r in records if r["kind"] == "gradient"]
    # One fixed direction from the warmup pools, tested on every test pool.
    assert {r["pool"] for r in gradient_records} == {2, 3}
    assert all(r["gradient_source_pools"] == [0, 1] for r in gradient_records)
    assert len({r["direction_sha256"] for r in gradient_records}) == 1
    # Its same-pool prediction uses that pool's own gradient: (1, 0, 0, k) . d.
    predictions = {r["pool"]: r["same_pool_token_prediction"] for r in gradient_records}
    assert predictions[3] > predictions[2]
    random_digests = {(r["pool"], r["index"]): r["direction_sha256"] for r in records if r["kind"] == "random"}
    assert len(set(random_digests.values())) == 4
    assert len(list((tmp_path / "sweep" / "gradients").iterdir())) == 4
    assert [entry["pool"] for entry in logged] == [0, 1, 2, 3]


def test_run_sweep_refuses_a_pool_that_does_not_restore(tmp_path):
    parameters = {"combiner.reader.out.weight": torch.nn.Parameter(torch.full((4,), 0.5))}
    sampler = SimpleNamespace(cursor=0)
    collect_at, _ = toy_policy(parameters, sampler)

    def leaky_collect(cursor):
        groups = collect_at(cursor)
        with torch.no_grad():
            parameters["combiner.reader.out.weight"].add_(1e-3)
        return groups

    with pytest.raises(RuntimeError, match="restore"):
        run_sweep(
            SweepConfig(("reader_out",), (0.5,), 1, False, False, 0), {"reader_out": parameters}, 1,
            cursor=lambda: sampler.cursor, collect_at=leaky_collect, token_gradient=None,
            writer=SweepWriter(tmp_path / "sweep", {}), out_of_time=lambda estimate: False,
            log=lambda values: None,
        )


def test_run_sweep_stops_at_a_pool_boundary_when_out_of_time(tmp_path):
    parameters = {"combiner.reader.out.weight": torch.nn.Parameter(torch.ones(2))}
    sampler = SimpleNamespace(cursor=0)

    def collect_at(cursor):
        sampler.cursor = cursor + 1
        return [fake_group([1], [[1]], [1.0])]

    calls = []

    def out_of_time(estimate):
        calls.append(estimate)
        return len(calls) > 2

    logged = []
    completed = run_sweep(
        SweepConfig(("reader_out",), (0.5,), 0, False, True, 0), {"reader_out": parameters}, 10,
        cursor=lambda: sampler.cursor, collect_at=collect_at, token_gradient=None,
        writer=SweepWriter(tmp_path / "sweep", {}), out_of_time=out_of_time, log=logged.append,
    )
    assert completed == 2
    # Nothing is known before the first pool; then 1 + 1 + 2 collects at the
    # slowest collect seen.
    assert calls[0] == 0.0 and calls[1] > 0.0
    assert logged[-1]["type"] == "perturbation_stop"


def test_sweep_writer_refuses_an_existing_artifact(tmp_path):
    SweepWriter(tmp_path / "sweep", {})
    with pytest.raises(ValueError, match="exists"):
        SweepWriter(tmp_path / "sweep", {})


BASE_ARGV = ["--checkpoint", "c.pt", "--output", "out"]


def sweep_args(*extra):
    parser = build_arg_parser()
    return parser, parser.parse_args(BASE_ARGV + list(extra))


VALID = [
    "--rollout-only", "--reasoning-mode", "memory", "--perturbation-sweep", "writer", "reader_out",
    "--perturbation-scales", "0.5", "--perturbation-random-directions", "2",
]


GRADIENT = ["--perturbation-gradient-direction", "--mean-actor-critic", "--mac-estimator",
            "sampled_baseline", "--curriculum-init", "snap.pt"]


def test_flag_validation_accepts_a_complete_sweep():
    parser, args = sweep_args(*VALID, "--perturbation-radial-direction")
    validate_perturbation_sweep(parser, args)
    parser, args = sweep_args(
        *VALID, "--perturbation-gradient-direction", "--mean-actor-critic",
        "--mac-estimator", "sampled_baseline", "--curriculum-init", "snap.pt",
        "--perturbation-gradient-warmup-pools", "4", "--rollout-only-repeats", "5")
    validate_perturbation_sweep(parser, args)


@pytest.mark.parametrize("argv", [
    ["--perturbation-scales", "0.5"],
    [arg for arg in VALID if arg != "--rollout-only"],
    ["--rollout-only", "--reasoning-mode", "cot", *VALID[3:]],
    [*VALID, "--gate-transcripts"],
    [*VALID[:6], *VALID[8:]],  # no scales
    [*VALID[:7], "-0.5", *VALID[8:]],
    [*VALID[:7], "0.5", "0.5", *VALID[8:]],
    [*VALID[:-1], "0"],  # no direction at all
    [*VALID, "--perturbation-gradient-direction"],  # no MAC sampled baseline
    [*VALID, "--perturbation-gradient-direction", "--mean-actor-critic", "--mac-estimator",
     "sampled_baseline"],  # no snapshot critic
    [*VALID, "--perturbation-gradient-direction", "--mean-actor-critic", "--mac-estimator",
     "all_action", "--curriculum-init", "snap.pt"],
    [*VALID, *GRADIENT, "--rollout-only-repeats", "5"],  # no warmup
    [*VALID, *GRADIENT, "--perturbation-gradient-warmup-pools", "5", "--rollout-only-repeats", "5"],
    [*VALID, "--perturbation-gradient-warmup-pools", "2", "--rollout-only-repeats", "5"],
])
def test_flag_validation_rejects_incomplete_sweeps(argv):
    parser, args = sweep_args(*argv)
    with pytest.raises(SystemExit):
        validate_perturbation_sweep(parser, args)


# --- write-space sweep ------------------------------------------------------------


def toy_noise_policy(sampler, samples=4, prompts=2):
    """A sample wins when x_s + scale * eps_prompt beats its threshold: one
    noise draw per prompt, independent draws per sample."""
    plans = []

    def collect_at(cursor, plan):
        plans.append((cursor, plan))
        sampler.cursor = cursor + prompts
        groups = []
        for prompt in range(prompts):
            generator = torch.Generator().manual_seed(1000 * cursor + prompt)
            features = torch.randn(samples, generator=generator)
            thresholds = torch.randn(samples, generator=generator)
            noise = torch.randn((), generator=generator)
            wins = (features + plan.scale * noise > thresholds).float()
            groups.append(fake_group([cursor + prompt], [[int(w) + 1] for w in wins.tolist()], wins.tolist()))
        return groups

    return collect_at, plans


def test_run_write_noise_sweep_collects_base_null_and_antithetic_pairs(tmp_path):
    sampler = SimpleNamespace(cursor=0)
    collect_at, plans = toy_noise_policy(sampler)
    config = WriteNoiseSweepConfig((0.5, 2.0))
    assert config.collects() == 6
    logged = []
    completed = run_write_noise_sweep(
        config, 3, cursor=lambda: sampler.cursor, collect_at=collect_at,
        writer=SweepWriter(tmp_path / "sweep", {"schema": "test"}),
        out_of_time=lambda estimate: False, log=logged.append,
    )
    assert completed == 3 and sampler.cursor == 6
    assert not (tmp_path / "sweep" / "gradients").exists()
    assert [cursor for cursor, _ in plans] == [0] * 6 + [2] * 6 + [4] * 6
    assert [plan.scale for _, plan in plans[:6]] == [0.0, 0.0, 0.5, -0.5, 2.0, -2.0]
    assert all(plan.shared_per_prompt for _, plan in plans)
    records = [json.loads(line) for line in (tmp_path / "sweep" / "collects.jsonl").read_text().splitlines()]
    pool = [record for record in records if record["pool"] == 1]
    assert [record["kind"] for record in pool] == ["base", "null"] + ["write_noise"] * 4
    assert [(r.get("scale"), r.get("sign")) for r in pool[1:]] == [
        (None, None), (0.5, 1), (0.5, -1), (2.0, 1), (2.0, -1)]
    assert pool[1]["rewards"] == pool[0]["rewards"] and not any(pool[1]["changed"])
    assert any(any(record["changed"]) for record in records if record["kind"] == "write_noise")
    summary = logged[-1]
    assert summary["type"] == "write_noise_pool" and summary["null/changed"] == 0
    assert summary["null/reward_shift"] == 0.0
    assert {"0.5/+1/changed", "0.5/-1/reward_mean", "2/+1/changed"} <= set(summary)


def test_run_write_noise_sweep_refuses_a_null_that_does_not_reproduce(tmp_path):
    sampler = SimpleNamespace(cursor=0)
    calls = []

    def collect_at(cursor, plan):
        calls.append(plan)
        sampler.cursor = cursor + 1
        return [fake_group([1], [[1 + len(calls)]], [1.0])]

    with pytest.raises(RuntimeError, match="null"):
        run_write_noise_sweep(
            WriteNoiseSweepConfig((1.0,)), 1, cursor=lambda: sampler.cursor,
            collect_at=collect_at, writer=SweepWriter(tmp_path / "sweep", {}),
            out_of_time=lambda estimate: False, log=lambda values: None,
        )
    # Refused before any explored collect: the pool stays incomplete.
    assert [plan.scale for plan in calls] == [0.0, 0.0]


def test_run_write_noise_sweep_stops_at_a_pool_boundary_when_out_of_time(tmp_path):
    sampler = SimpleNamespace(cursor=0)
    collect_at, _ = toy_noise_policy(sampler)
    calls, logged = [], []

    def out_of_time(estimate):
        calls.append(estimate)
        return len(calls) > 2

    completed = run_write_noise_sweep(
        WriteNoiseSweepConfig((1.0,)), 10, cursor=lambda: sampler.cursor,
        collect_at=collect_at, writer=SweepWriter(tmp_path / "sweep", {}),
        out_of_time=out_of_time, log=logged.append,
    )
    assert completed == 2
    assert calls[0] == 0.0 and calls[1] > 0.0
    assert logged[-1]["type"] == "write_noise_stop"


def test_write_noise_sweep_config_rejects_invalid_scales():
    for scales in ((), (0.0,), (-1.0,), (float("inf"),), (0.5, 0.5)):
        with pytest.raises(ValueError):
            WriteNoiseSweepConfig(scales)


NOISE_VALID = ["--rollout-only", "--reasoning-mode", "memory", "--write-noise-sweep", "0.3", "1"]


def test_write_noise_flag_validation():
    parser, args = sweep_args(*NOISE_VALID)
    validate_write_noise_sweep(parser, args)
    assert args.write_noise_sweep == [0.3, 1.0]
    for argv in (
        NOISE_VALID[1:],
        ["--rollout-only", "--reasoning-mode", "cot", *NOISE_VALID[3:]],
        [*NOISE_VALID, "--gate-transcripts"],
        [*NOISE_VALID[:4], "0.3", "0.3"],
        [*NOISE_VALID[:4], "-1"],
        [*NOISE_VALID, "--rollout-scheduler", "continuous_refill"],
        [*NOISE_VALID, "--samples-per-prompt", "1"],
        [*NOISE_VALID, "--prompts-per-rollout", "1"],
        [*NOISE_VALID, "--perturbation-sweep", "writer", "--perturbation-scales", "0.5",
         "--perturbation-random-directions", "1"],
    ):
        parser, args = sweep_args(*argv)
        with pytest.raises(SystemExit):
            validate_write_noise_sweep(parser, args)
