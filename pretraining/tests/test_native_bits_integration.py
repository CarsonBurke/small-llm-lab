"""CPU metadata, packet-integrity and metric contracts; no neural workloads."""

import argparse
import math
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from pretraining.nanogpt_mini import nanogpt_mini_native_bits_train as training
from pretraining.nanogpt_mini.native_bits_data import sha256_file
from pretraining.nanogpt_mini.native_bits_wire import pack_packet, unpack_packet
from scripts import native_bits as cli


def checkpoint_payload(model_kind):
    # Native deliberately predates mixture_components/model_kind checkpoint fields.
    model_configs = {
        "native": {"vocab_size": 3, "code_bits": 32, "latent_bits": 8},
        "bitflow": {"vocab_size": 3, "code_bits": 2},
        "softmax": {"vocab_size": 3},
        "adaptive": asdict(training.AdaptiveConfig(vocab_size=3, code_bits=2)),
        "dynamics": asdict(training.DynamicsConfig(vocab_size=3, code_bits=2)),
        "dynamics_bounded": asdict(training.DynamicsConfig(vocab_size=3, code_bits=2)),
        "embedder": asdict(training.EmbedderConfig(vocab_size=3, gate_mode="fixed")),
    }
    return {
        "architecture": training.ARCHITECTURES[model_kind],
        "checkpoint_version": 1,
        "model_config": model_configs[model_kind],
        "alphabet": ["a", "é", "🙂"],
        "model": {},
        "provenance": {},
        "optimizer": [],
        "rng": {},
        "step": 0,
        "train_config": {"data_path": "unused"},
        "data_cursor": 0,
        "training_ms": 0.0,
    }


@pytest.mark.parametrize(
    "model_kind,widths",
    [
        ("native", (8, 2)),
        ("bitflow", (2, 0)),
        ("softmax", (2, 0)),
        ("adaptive", (2, 0)),
        ("dynamics", (2, 0)),
        ("dynamics_bounded", (2, 0)),
        ("embedder", (2, 0)),
    ],
)
def test_empty_cli_roundtrip_uses_architecture_widths_without_cuda(
    tmp_path, monkeypatch, model_kind, widths
):
    def forbid_cuda():
        pytest.fail("metadata-only transport must not construct a CUDA model")

    monkeypatch.setattr(training, "cuda_device", forbid_cuda)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(checkpoint_payload(model_kind), checkpoint)
    source = tmp_path / "source.txt"
    source.write_bytes(b"")
    packet_path = tmp_path / "source.nb01"
    restored = tmp_path / "restored.txt"
    cli.encode(
        argparse.Namespace(checkpoint=checkpoint, input=source, output=packet_path)
    )
    packet = unpack_packet(packet_path.read_bytes())
    assert (packet["latent_bits"], packet["identity_bits"]) == widths
    assert packet["checkpoint_sha256"] == sha256_file(checkpoint)
    assert packet["source_sha256"] == sha256_file(source)
    cli.decode(
        argparse.Namespace(checkpoint=checkpoint, input=packet_path, output=restored)
    )
    assert restored.read_bytes() == source.read_bytes()


def test_historical_native_checkpoint_compares_equal_to_independent_prior_config(
    tmp_path,
):
    path = tmp_path / "historical.pt"
    torch.save(checkpoint_payload("native"), path)
    checkpoint = training.load_checkpoint(path)
    historical = training.checkpoint_model_config(checkpoint)
    independent = training.NativeBitsConfig(vocab_size=3, mixture_components=1)
    assert asdict(historical) == asdict(independent)
    assert training.TrainConfig(**checkpoint["train_config"]).model_kind == "native"


def test_historical_bitflow_resume_cannot_guess_the_backward_rule(
    tmp_path, monkeypatch
):
    checkpoint = checkpoint_payload("bitflow")
    path = tmp_path / "historical.pt"
    torch.save(checkpoint, path)
    monkeypatch.setenv("NATIVE_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("NATIVE_RESUME", str(path))
    monkeypatch.setenv("MODEL_KIND", "bitflow")
    monkeypatch.setenv("MASK_SURROGATE", "sigmoid")
    monkeypatch.setattr(
        training,
        "PreparedData",
        lambda _: SimpleNamespace(
            validation=SimpleNamespace(size=65536),
            alphabet=checkpoint["alphabet"],
            metadata={},
        ),
    )
    monkeypatch.setattr(
        training,
        "cuda_device",
        lambda: pytest.fail("ambiguous resume must fail before any CUDA workload"),
    )
    with pytest.raises(ValueError, match="lacks mask_surrogate"):
        training.train()


@pytest.mark.parametrize(
    "options,message",
    [
        (["--model", "native", "--density-head", "mixture"], "only valid with"),
        (["--model", "softmax", "--prefix-width", "128"], "only valid with"),
        (["--model", "bitflow", "--prefix-width", "128"], "requires --density-head"),
        (["--model", "bitflow", "--density-head", "prefix"], "requires --fixed-codec"),
        (
            [
                "--model",
                "bitflow",
                "--fixed-codec",
                "--density-head",
                "prefix",
                "--mixture-components",
                "8",
            ],
            "requires --mixture-components 1",
        ),
        (
            [
                "--model",
                "bitflow",
                "--fixed-codec",
                "--density-head",
                "prefix",
                "--prefix-width",
                "0",
            ],
            "must be positive",
        ),
        (["--model", "adaptive", "--frozen-codes"], "only valid with"),
        (["--model", "adaptive", "--fixed-codec"], "only valid with"),
        (["--model", "adaptive", "--latent-bits", "8"], "only valid with"),
        (["--model", "adaptive", "--mixture-components", "1"], "not used by"),
        (["--model", "adaptive", "--mask-surrogate", "identity"], "only valid with"),
        (["--model", "adaptive", "--density-head", "prefix"], "only valid with"),
        (["--model", "adaptive", "--flow-layers", "4"], "not used by"),
        (["--model", "adaptive", "--codec-dim", "64"], "not used by"),
        (["--model", "adaptive", "--codec-weight", "1"], "not used by"),
        (["--model", "adaptive", "--code-lr", "0.004"], "not used by"),
        (["--model", "native", "--local-layers", "2"], "only valid with"),
        (["--model", "bitflow", "--local-dim", "128"], "only valid with"),
        (["--model", "softmax", "--refresh-cost", "0.02"], "only valid with"),
        (["--model", "dynamics", "--frozen-codes"], "only valid with"),
        (["--model", "dynamics", "--fixed-codec"], "only valid with"),
        (["--model", "dynamics", "--latent-bits", "8"], "only valid with"),
        (["--model", "dynamics", "--mixture-components", "1"], "not used by"),
        (["--model", "dynamics", "--mask-surrogate", "identity"], "only valid with"),
        (["--model", "dynamics", "--density-head", "prefix"], "only valid with"),
        (["--model", "dynamics", "--flow-layers", "4"], "not used by"),
        (["--model", "dynamics", "--codec-dim", "64"], "not used by"),
        (["--model", "dynamics", "--codec-weight", "1"], "not used by"),
        (["--model", "dynamics", "--code-lr", "0.004"], "not used by"),
        (["--model", "dynamics", "--local-layers", "2"], "only valid with"),
        (["--model", "dynamics", "--local-dim", "128"], "only valid with"),
        (["--model", "native", "--dynamics-width", "128"], "only valid with"),
        (["--model", "adaptive", "--rollout-horizon", "2"], "only valid with"),
        (["--model", "bitflow", "--latent-weight", "1"], "only valid with"),
        (["--model", "softmax", "--rollout-weight", "1"], "only valid with"),
        (["--model", "adaptive", "--gate-weight", "1"], "only valid with"),
        (["--model", "softmax", "--gate-mode", "fixed"], "only valid with"),
        (["--model", "embedder", "--code-bits", "12"], "require a bit model"),
    ],
)
def test_cli_rejects_unsupported_density_options_before_launch(
    tmp_path, monkeypatch, options, message
):
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "native_bits",
            "train",
            "--data",
            str(tmp_path),
            "--name",
            "invalid",
            *options,
        ],
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("invalid options must not launch ablation"),
    )
    with pytest.raises(ValueError, match=message):
        cli.main()


@pytest.mark.parametrize(
    "invalid",
    [
        {"learn_codec": True},
        {"mixture_components": 8},
        {"prefix_width": 0},
    ],
)
def test_checkpoint_rejects_unsupported_prefix_density_metadata(tmp_path, invalid):
    checkpoint = checkpoint_payload("bitflow")
    checkpoint["model_config"].update(
        density_head="prefix",
        prefix_width=128,
        mixture_components=1,
        learn_codec=False,
    )
    checkpoint["model_config"].update(invalid)
    path = tmp_path / "invalid-prefix.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        training.load_checkpoint(path)


@pytest.mark.parametrize("previous_head", ["historical-mixture", "prefix"])
def test_resume_rejects_changed_density_head_or_prefix_width_before_cuda(
    tmp_path, monkeypatch, previous_head
):
    checkpoint = checkpoint_payload("bitflow")
    path = tmp_path / "previous.pt"
    for name, value in {
        "NATIVE_DATA_PATH": str(tmp_path),
        "NATIVE_RESUME": str(path),
        "MODEL_KIND": "bitflow",
        "DENSITY_HEAD": "prefix",
        "PREFIX_WIDTH": "128",
        "LEARN_CODEC": "0",
        "MIXTURE_COMPONENTS": "1",
        "MASK_SURROGATE": "sigmoid",
    }.items():
        monkeypatch.setenv(name, value)
    checkpoint["model_config"] = asdict(
        training.model_config_from_env(len(checkpoint["alphabet"]), "bitflow")
    )
    if previous_head == "historical-mixture":
        del checkpoint["model_config"]["density_head"]
        del checkpoint["model_config"]["prefix_width"]
    else:
        checkpoint["model_config"]["prefix_width"] = 64
    checkpoint["provenance"]["data"] = {}
    torch.save(checkpoint, path)
    monkeypatch.setattr(
        training,
        "PreparedData",
        lambda _: SimpleNamespace(
            validation=SimpleNamespace(size=65536),
            alphabet=checkpoint["alphabet"],
            metadata={},
        ),
    )
    monkeypatch.setattr(
        training,
        "cuda_device",
        lambda: pytest.fail("incompatible density must fail before any CUDA workload"),
    )
    with pytest.raises(
        ValueError, match="resume model, alphabet or data provenance mismatch"
    ):
        training.train()


@pytest.mark.parametrize("model_kind", ["bitflow", "dynamics"])
@pytest.mark.parametrize("damage", ["checkpoint", "widths", "source"])
def test_decode_rejects_wrong_checkpoint_widths_or_source_before_publishing(
    tmp_path, monkeypatch, damage, model_kind
):
    def forbid_cuda():
        pytest.fail("invalid packet metadata must fail before any CUDA workload")

    monkeypatch.setattr(training, "cuda_device", forbid_cuda)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = checkpoint_payload(model_kind)
    torch.save(checkpoint, checkpoint_path)
    source = tmp_path / "empty.txt"
    source.write_bytes(b"")
    packet_path = tmp_path / "source.nb01"
    packet_path.write_bytes(
        pack_packet(
            [],
            [],
            sha256_file(checkpoint_path),
            "00" * 32 if damage == "source" else sha256_file(source),
            latent_bits=3 if damage == "widths" else 2,
            identity_bits=0,
        )
    )
    if damage == "checkpoint":
        checkpoint["step"] = 1
        torch.save(checkpoint, checkpoint_path)
    output = tmp_path / "restored.txt"
    with pytest.raises(ValueError):
        cli.decode(
            argparse.Namespace(
                checkpoint=checkpoint_path, input=packet_path, output=output
            )
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "model_kind", ["native", "bitflow", "softmax", "adaptive", "dynamics"]
)
def test_checkpoint_rejects_alphabet_that_does_not_match_model(tmp_path, model_kind):
    checkpoint = checkpoint_payload(model_kind)
    checkpoint["alphabet"] = ["a", "é"]
    path = tmp_path / "bad.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        training.load_checkpoint(path)


def test_dynamic_statistics_weight_short_tail_by_characters_and_rate_by_utf8_bytes():
    protocol = SimpleNamespace(
        sum_stat_names=("rate_nats",), mean_stat_names=("symbol_accuracy",)
    )
    full = training.stats_vector(
        protocol,
        torch.tensor(12.0),
        {"rate_nats": torch.tensor(12.0), "symbol_accuracy": torch.tensor(0.5)},
        4,
    )
    tail = training.stats_vector(
        protocol,
        torch.tensor(3.0),
        {"rate_nats": torch.tensor(3.0), "symbol_accuracy": torch.tensor(1.0)},
        1,
    )
    metrics = training.unpack_stats(protocol, full + tail, count=5, byte_count=9)
    assert metrics["loss"] == 3.0
    assert metrics["rate_nats_per_character"] == 3.0
    assert metrics["bpb"] == pytest.approx(15 / math.log(2) / 9)
    assert metrics["symbol_accuracy"] == pytest.approx(0.6)
    assert "codec_loss" not in metrics
    assert "latent_bpb" not in metrics
    assert "residual_bpb" not in metrics


@pytest.mark.parametrize(
    "invalid",
    [
        {"refresh_cost": -0.01},
        {"refresh_cost": float("nan")},
        {"local_dim": 64},
        {"code_bits": 1},
        {"local_layers": 0},
    ],
)
def test_checkpoint_rejects_invalid_adaptive_metadata_before_cuda(
    tmp_path, monkeypatch, invalid
):
    checkpoint = checkpoint_payload("adaptive")
    checkpoint["model_config"].update(invalid)
    path = tmp_path / "invalid-adaptive.pt"
    torch.save(checkpoint, path)
    monkeypatch.setattr(
        training,
        "cuda_device",
        lambda: pytest.fail("invalid metadata must not construct a CUDA model"),
    )
    with pytest.raises(ValueError):
        training.load_model(training.load_checkpoint(path))


def test_checkpoint_does_not_guess_missing_adaptive_routing_setting(tmp_path):
    checkpoint = checkpoint_payload("adaptive")
    del checkpoint["model_config"]["refresh_cost"]
    path = tmp_path / "missing-routing.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="incomplete adaptive model_config"):
        training.load_checkpoint(path)


@pytest.mark.parametrize(
    "changed",
    [
        {"refresh_cost": 0.03},
        {"local_layers": 3},
        {"local_dim": 256},
        {"prefix_width": 64},
    ],
)
def test_resume_rejects_changed_adaptive_routing_config_before_cuda(
    tmp_path, monkeypatch, changed
):
    checkpoint = checkpoint_payload("adaptive")
    checkpoint["model_config"].update(changed)
    checkpoint["provenance"]["data"] = {}
    path = tmp_path / "previous.pt"
    torch.save(checkpoint, path)
    for name, value in {
        "NATIVE_DATA_PATH": str(tmp_path),
        "NATIVE_RESUME": str(path),
        "MODEL_KIND": "adaptive",
        "CODE_BITS": "2",
        "NUM_LAYERS": "6",
        "MODEL_DIM": "512",
        "LOCAL_LAYERS": "2",
        "LOCAL_DIM": "128",
        "PREFIX_WIDTH": "128",
        "REFRESH_COST": "0.02",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        training,
        "PreparedData",
        lambda _: SimpleNamespace(
            validation=SimpleNamespace(size=65536),
            alphabet=checkpoint["alphabet"],
            metadata={},
        ),
    )
    monkeypatch.setattr(
        training,
        "cuda_device",
        lambda: pytest.fail("changed routing settings must fail before CUDA"),
    )
    with pytest.raises(
        ValueError, match="resume model, alphabet or data provenance mismatch"
    ):
        training.train()


@pytest.mark.parametrize(
    "model_kind,options",
    [
        ("adaptive", ["--local-dim", "64"]),
        ("adaptive", ["--refresh-cost", "nan"]),
        ("adaptive", ["--code-bits", "1"]),
        ("dynamics", ["--dynamics-width", "0"]),
        ("dynamics", ["--rollout-horizon", "0"]),
        ("dynamics", ["--latent-weight", "-1"]),
        ("dynamics", ["--rollout-weight", "nan"]),
        ("dynamics", ["--gate-weight", "inf"]),
        ("dynamics", ["--refresh-cost", "0"]),
        ("dynamics", ["--model-dim", "64"]),
        ("dynamics", ["--code-bits", "1"]),
    ],
)
def test_cli_rejects_invalid_routed_dimensions_before_launch(
    tmp_path, monkeypatch, model_kind, options
):
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "native_bits",
            "train",
            "--data",
            str(tmp_path),
            "--name",
            "invalid",
            "--model",
            model_kind,
            *options,
        ],
    )
    monkeypatch.setattr(
        cli,
        "PreparedData",
        lambda _: SimpleNamespace(
            validation=SimpleNamespace(size=65536), alphabet=["a", "b", "c"]
        ),
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "invalid dimensions must not launch ablation"
        ),
    )
    with pytest.raises(ValueError):
        cli.main()


@pytest.mark.parametrize(
    "model_kind,setting",
    [
        ("adaptive", "DENSITY_HEAD"),
        ("adaptive", "FLOW_LAYERS"),
        ("bitflow", "LOCAL_LAYERS"),
        ("dynamics", "DENSITY_HEAD"),
        ("dynamics", "CODEC_DIM"),
        ("dynamics", "FLOW_LAYERS"),
        ("dynamics", "LATENT_BITS"),
        ("dynamics", "MIXTURE_COMPONENTS"),
        ("dynamics", "LEARN_CODES"),
        ("dynamics", "LEARN_CODEC"),
        ("dynamics", "MASK_SURROGATE"),
        ("dynamics", "CODEC_WEIGHT"),
        ("dynamics", "CODE_LR"),
        ("dynamics", "LOCAL_LAYERS"),
        ("dynamics", "LOCAL_DIM"),
        ("adaptive", "DYNAMICS_WIDTH"),
        ("bitflow", "ROLLOUT_HORIZON"),
        ("native", "LATENT_WEIGHT"),
        ("softmax", "ROLLOUT_WEIGHT"),
        ("adaptive", "GATE_WEIGHT"),
        ("embedder", "CODE_BITS"),
        ("softmax", "GATE_MODE"),
    ],
)
def test_environment_rejects_irrelevant_routed_options(
    monkeypatch, model_kind, setting
):
    monkeypatch.setenv(setting, "1")
    with pytest.raises(ValueError, match="environment settings"):
        training.model_config_from_env(3, model_kind)


@pytest.mark.parametrize(
    "model_kind", ["softmax", "adaptive", "dynamics", "dynamics_bounded", "embedder"]
)
@pytest.mark.parametrize(
    "case",
    ["characters", "temperature", "seed", "checkpoint", "prompt", "overwrite"],
)
def test_generate_rejects_invalid_requests_without_cuda_or_publishing(
    tmp_path, monkeypatch, case, model_kind
):
    checkpoint = checkpoint_payload("native" if case == "checkpoint" else model_kind)
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    output = tmp_path / "generation.json"
    if case == "overwrite":
        output.write_text("keep existing output", encoding="utf-8")
    monkeypatch.setattr(
        training,
        "cuda_device",
        lambda: pytest.fail("invalid generation request must fail before CUDA"),
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "native_bits",
            "generate",
            "--checkpoint",
            str(path),
            "--prompt",
            "unknown" if case == "prompt" else "a",
            "--characters",
            "-1" if case == "characters" else "8",
            "--temperature",
            "nan" if case == "temperature" else "0",
            "--seed",
            str(2**32) if case == "seed" else "1337",
            "--output",
            str(output),
        ],
    )
    with pytest.raises((ValueError, FileExistsError)):
        cli.main()
    if case == "overwrite":
        assert output.read_text(encoding="utf-8") == "keep existing output"
    else:
        assert not output.exists()


def test_auxiliary_losses_and_compute_counters_are_not_coding_rate():
    protocol = SimpleNamespace(
        sum_stat_names=(
            "rate_nats",
            "compute_nats",
            "baseline_loss",
            "emitted_events",
            "padded_global_positions",
        ),
        mean_stat_names=("refresh_rate",),
    )
    values = training.stats_vector(
        protocol,
        torch.tensor(9.0),
        {
            "rate_nats": torch.tensor(8.0),
            "compute_nats": torch.tensor(1.0),
            "baseline_loss": torch.tensor(12.0),
            "emitted_events": torch.tensor(1.0),
            "padded_global_positions": torch.tensor(2.0),
            "refresh_rate": torch.tensor(0.25),
        },
        4,
    )
    metrics = training.unpack_stats(protocol, values, count=4, byte_count=8)
    assert metrics["loss"] == 2.0
    assert metrics["objective"] == 2.25
    assert metrics["bpb"] == pytest.approx(1 / math.log(2))
    assert metrics["compute_nats_per_character"] == 0.25
    assert metrics["baseline_loss_per_character"] == 3.0
    assert metrics["emitted_events_per_character"] == 0.25
    assert metrics["padded_global_positions_per_character"] == 0.5
    assert {name for name in metrics if name.endswith("bpb")} == {"bpb"}
    assert "compute_bpb" not in metrics


@pytest.mark.parametrize(
    "model_kind,policy_field",
    [("dynamics", "refresh_cost"), ("embedder", "gate_mode")],
)
def test_checkpoint_does_not_guess_missing_causal_policy(
    tmp_path, model_kind, policy_field
):
    checkpoint = checkpoint_payload(model_kind)
    del checkpoint["model_config"][policy_field]
    path = tmp_path / "incomplete-policy.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match=f"incomplete {model_kind} model_config"):
        training.load_checkpoint(path)


@pytest.mark.parametrize(
    "invalid",
    [
        {"dynamics_width": 0},
        {"rollout_horizon": 1.5},
        {"num_layers": True},
        {"latent_weight": -1.0},
        {"rollout_weight": float("nan")},
        {"gate_weight": float("inf")},
        {"refresh_cost": 0.0},
        {"code_bits": 33},
        {"model_dim": 64},
        {"refresh_interval": 4},
    ],
)
def test_dynamics_checkpoint_rejects_invalid_semantics_before_cuda(
    tmp_path, monkeypatch, invalid
):
    checkpoint = checkpoint_payload("dynamics")
    checkpoint["model_config"].update(invalid)
    path = tmp_path / "invalid-dynamics.pt"
    torch.save(checkpoint, path)
    monkeypatch.setattr(
        training,
        "cuda_device",
        lambda: pytest.fail("invalid dynamics metadata must fail before CUDA"),
    )
    with pytest.raises(ValueError):
        training.load_model(training.load_checkpoint(path))


def test_resume_rejects_changed_dynamics_refresh_policy_before_cuda(
    tmp_path, monkeypatch
):
    checkpoint = checkpoint_payload("dynamics")
    checkpoint["model_config"]["refresh_cost"] = 0.03
    checkpoint["provenance"]["data"] = {}
    path = tmp_path / "previous.pt"
    torch.save(checkpoint, path)
    for name, value in {
        "NATIVE_DATA_PATH": str(tmp_path),
        "NATIVE_RESUME": str(path),
        "MODEL_KIND": "dynamics",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        training,
        "PreparedData",
        lambda _: SimpleNamespace(
            validation=SimpleNamespace(size=65536),
            alphabet=checkpoint["alphabet"],
            metadata={},
        ),
    )
    monkeypatch.setattr(
        training,
        "cuda_device",
        lambda: pytest.fail("changed dynamics semantics must fail before CUDA"),
    )
    with pytest.raises(
        ValueError, match="resume model, alphabet or data provenance mismatch"
    ):
        training.train()


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "dynamics_generation.py",
    ],
)
def test_dynamics_resume_rejects_unverified_runtime_and_teacher_sources(
    tmp_path, monkeypatch, damage
):
    checkpoint = checkpoint_payload("dynamics")
    checkpoint["train_config"] = asdict(
        training.TrainConfig(data_path=str(tmp_path), model_kind="dynamics")
    )
    checkpoint["provenance"]["data"] = {}
    if damage != "missing":
        hashes = training.source_hashes("dynamics")
        dependency = f"pretraining/nanogpt_mini/{damage}"
        hashes[dependency] = f"changed:{hashes[dependency]}"
        checkpoint["provenance"]["source_hashes"] = hashes
    path = tmp_path / "previous.pt"
    torch.save(checkpoint, path)
    for name, value in {
        "NATIVE_DATA_PATH": str(tmp_path),
        "NATIVE_RESUME": str(path),
        "MODEL_KIND": "dynamics",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        training,
        "PreparedData",
        lambda _: SimpleNamespace(
            validation=SimpleNamespace(size=65536),
            alphabet=checkpoint["alphabet"],
            metadata={},
        ),
    )
    monkeypatch.setattr(
        training,
        "cuda_device",
        lambda: pytest.fail("unverified dynamics resume must fail before CUDA"),
    )
    with pytest.raises(ValueError, match="source provenance mismatch"):
        training.train()


def test_dynamics_auxiliary_objective_never_enters_coding_rate():
    # Use the real statistics schema without constructing or running a model.
    values = training.stats_vector(
        training.DynamicsGPT,
        torch.tensor(20.0),
        {
            "rate_nats": torch.tensor(8.0),
            "latent_loss": torch.tensor(0.5),
            "rollout_kl": torch.tensor(1.5),
            "gate_loss": torch.tensor(1.0),
            "refresh_rate": torch.tensor(0.25),
            "backbone_positions_per_character": torch.tensor(0.75),
            "evaluation_backbone_positions_per_character": torch.tensor(1.0),
        },
        4,
    )
    metrics = training.unpack_stats(training.DynamicsGPT, values, count=4, byte_count=8)
    assert metrics["loss"] == 2.0
    assert metrics["objective"] == 5.0
    assert metrics["bpb"] == pytest.approx(1 / math.log(2))
    assert metrics["latent_loss"] == 0.5
    assert metrics["rollout_kl"] == 1.5
    assert metrics["gate_loss"] == 1.0
    assert metrics["refresh_rate"] == 0.25
    assert metrics["backbone_positions_per_character"] == 0.75
    assert metrics["evaluation_backbone_positions_per_character"] == 1.0
    assert {name for name in metrics if name.endswith("bpb")} == {"bpb"}


@pytest.mark.parametrize(
    "model_kind,setting",
    [
        ("dynamics", "LOCAL_DIM"),
        ("dynamics", "MIXTURE_COMPONENTS"),
        ("dynamics", "CODEC_WEIGHT"),
        ("adaptive", "ROLLOUT_HORIZON"),
        ("native", "GATE_WEIGHT"),
    ],
)
def test_cli_rejects_inherited_irrelevant_environment_before_launch(
    tmp_path, monkeypatch, model_kind, setting
):
    monkeypatch.setenv(setting, "1")
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "native_bits",
            "train",
            "--data",
            str(tmp_path),
            "--name",
            "invalid",
            "--model",
            model_kind,
        ],
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "irrelevant inherited settings must not launch ablation"
        ),
    )
    with pytest.raises(ValueError, match="environment settings"):
        cli.main()
