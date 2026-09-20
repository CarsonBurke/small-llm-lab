"""CPU-only configuration, initialization-parity, and checkpoint contracts; no model execution."""

from dataclasses import asdict, replace

import pytest
import torch

from pretraining.future_credit_stream.model import StreamingFFNModel
from pretraining.stationary_stream import training
from pretraining.stationary_stream.config import ARCHITECTURE, Config
from pretraining.stationary_stream.model import BufferRead, StationaryFFNModel


def test_default_configuration_is_valid():
    config = Config()
    config.validate()
    assert config.buffer_slots == 10 and config.read_heads == 4 and config.read_key_dim == 32
    assert config.read_entry == "latent_norm" and config.read_recency_slope == 1.0
    assert set(config.model_config) == {"vocab_size", "model_dim", "num_layers", "mlp_hidden", "buffer_slots",
                                        "read_heads", "read_key_dim", "read_entry", "read_recency_slope"}


def test_read_entry_must_be_a_known_path(monkeypatch):
    with pytest.raises(ValueError):
        replace(Config(), read_entry="gate").validate()
    monkeypatch.setenv("STATIONARY_STREAM_READ_ENTRY", "value_map")
    assert Config.from_env().read_entry == "value_map"
    for slope in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            replace(Config(), read_recency_slope=slope).validate()
        with pytest.raises(ValueError):
            BufferRead(8, 4, 2, 8, "latent_norm", slope)
        with pytest.raises(ValueError):
            BufferRead(8, 4, 2, 8, "value_map", slope)
    replace(Config(), read_recency_slope=0.0).validate()
    assert torch.all(BufferRead(8, 4, 2, 8, "latent_norm", 0.0).age_bias == 0)
    # The slope only shapes the latent_norm bias: a value_map run must not record one.
    with pytest.raises(ValueError):
        replace(Config(), read_entry="value_map", read_recency_slope=4.0).validate()
    replace(Config(), read_entry="value_map").validate()
    monkeypatch.setenv("STATIONARY_STREAM_READ_RECENCY_SLOPE", "4")
    with pytest.raises(ValueError):  # the environment still selects value_map
        Config.from_env()
    monkeypatch.delenv("STATIONARY_STREAM_READ_ENTRY")
    assert Config.from_env().read_recency_slope == 4.0


@pytest.mark.parametrize("field", ["buffer_slots", "read_heads", "read_key_dim"])
def test_read_shape_fields_must_be_positive(field):
    with pytest.raises(ValueError):
        replace(Config(), **{field: 0}).validate()


def test_heads_must_divide_model_dim():
    with pytest.raises(ValueError):
        replace(Config(), read_heads=3).validate()
    replace(Config(), read_heads=8).validate()


def test_key_dim_must_be_a_multiple_of_four():
    with pytest.raises(ValueError):
        replace(Config(), read_key_dim=30).validate()
    replace(Config(), read_key_dim=16).validate()


def test_environment_overrides_read_fields(monkeypatch):
    monkeypatch.setenv("STATIONARY_STREAM_BUFFER_SLOTS", "32")
    monkeypatch.setenv("STATIONARY_STREAM_READ_HEADS", "8")
    monkeypatch.setenv("RUN_ID", "stat_k32")
    config = Config.from_env()
    assert config.buffer_slots == 32 and config.read_heads == 8 and config.run_id == "stat_k32"


def test_shared_training_fields_match_the_ce_recursion_line():
    """The sibling line's CE arm is the control: every shared field must agree by default."""
    from pretraining.future_credit_stream.config import Config as RecursionConfig
    ours, theirs = asdict(Config()), asdict(RecursionConfig())
    shared = set(ours) & set(theirs) - {"run_id"}
    assert {"data_path", "seed", "learning_rate", "document_batch", "stream_steps", "iterations"} <= shared
    assert all(ours[key] == theirs[key] for key in shared), [key for key in shared if ours[key] != theirs[key]]


@pytest.mark.parametrize("entry", ["value_map", "latent_norm"])
def test_backbone_initialization_is_bitwise_the_ce_recursion_control(entry):
    torch.manual_seed(1337)
    control = StreamingFFNModel(vocab_size=32, model_dim=128, num_layers=3, mlp_hidden=256, use_writer=False)
    torch.manual_seed(1337)
    model = StationaryFFNModel(vocab_size=32, model_dim=128, num_layers=3, mlp_hidden=256,
                               buffer_slots=5, read_heads=2, read_key_dim=8, read_entry=entry)
    control_parameters = dict(control.named_parameters())
    parameters = dict(model.named_parameters())
    common = {name for name in parameters if not name.startswith("read.")}
    # value_map lacks the control's latent_norm; latent_norm has the control's whole backbone.
    assert set(control_parameters) - common == ({"latent_norm.gains"} if entry == "value_map" else set())
    for name in common:
        assert torch.equal(parameters[name], control_parameters[name]), name
    read = sum(p.numel() for p in model.read_parameters())
    projections = 2 * (128 * 16 + 16)
    assert read == projections + ((128 * 128 + 128) + 2 if entry == "value_map" else 2 * 5)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    if entry == "latent_norm":
        assert torch.all(model.read.query.weight == 0) and torch.all(model.read.query.bias == 0)
        assert torch.equal(model.read.age_bias, -torch.arange(5.0)[None].repeat(2, 1))
        steep = StationaryFFNModel(vocab_size=32, model_dim=128, num_layers=3, mlp_hidden=256, buffer_slots=5,
                                   read_heads=2, read_key_dim=8, read_entry=entry, read_recency_slope=4.0)
        assert torch.equal(steep.read.age_bias, -4 * torch.arange(5.0)[None].repeat(2, 1))
        assert steep.config["read_recency_slope"] == 4.0
    # Rotary ages are derived constants, not checkpoint state.
    assert "read.age_cos" not in model.state_dict() and model.read.age_cos.shape == (5, 4)


def test_resume_rejects_changed_sources(tmp_path, monkeypatch):
    # Use an isolated source tree: never modify the actual upstream baseline.
    module = tmp_path / "pretraining/stationary_stream/training.py"
    for relative in (
        "pretraining/stationary_stream/training.py",
        "pretraining/stationary_stream/model.py",
        "pretraining/stationary_stream/objective.py",
        "pretraining/stationary_stream/config.py",
        "pretraining/future_credit_stream/data.py",
        "pretraining/future_credit_stream/training.py",
        "pretraining/nanogpt_mini/nanogpt_mini_model.py",
        "pretraining/nextlat.py",
        "train_gpt.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original source\n")
    monkeypatch.setattr(training, "__file__", str(module))
    monkeypatch.setattr(training, "REPO_ROOT", tmp_path)
    config = Config()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"architecture": ARCHITECTURE, "config": asdict(config),
                "metadata": {"sources": training.source_hashes()}}, checkpoint)
    training.load_checkpoint(checkpoint, config)

    (tmp_path / "pretraining/future_credit_stream/data.py").write_text("changed page contract\n")
    with pytest.raises(ValueError):
        training.load_checkpoint(checkpoint, config)
    (tmp_path / "pretraining/future_credit_stream/data.py").write_text("original source\n")
    with pytest.raises(ValueError):
        training.load_checkpoint(checkpoint, replace(config, buffer_slots=3))
