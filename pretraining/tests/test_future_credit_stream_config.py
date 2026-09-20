"""CPU-only configuration contract checks; no model execution."""

from dataclasses import replace

import pytest

from pretraining.future_credit_stream.config import OBJECTIVES, Config


def test_default_configuration_is_valid_and_uses_the_writer():
    config = Config()
    config.validate()
    assert config.objective == "future_bag"
    assert config.model_config["use_writer"]
    assert config.page_horizon == config.horizon == 32
    assert set(config.model_config) == {"vocab_size", "model_dim", "num_layers", "mlp_hidden", "use_writer"}


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_every_objective_maps_to_a_consistent_model_and_page(objective):
    config = replace(Config(), objective=objective)
    config.validate()
    uses_writer = objective == "future_bag"
    assert config.uses_writer is uses_writer
    assert config.model_config["use_writer"] is uses_writer
    # Only the writer objective needs future targets in its pages.
    assert config.page_horizon == (config.horizon if uses_writer else 0)


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError):
        replace(Config(), objective="usefulness").validate()


@pytest.mark.parametrize("objective", ["ce", "tbptt"])
def test_backbone_future_weight_requires_the_writer(objective):
    with pytest.raises(ValueError):
        replace(Config(), objective=objective, backbone_future_weight=0.5).validate()
    replace(Config(), objective="future_bag", backbone_future_weight=0.5).validate()


@pytest.mark.parametrize("weight", [-0.1, 1.5, float("nan")])
def test_backbone_future_weight_range(weight):
    with pytest.raises(ValueError):
        replace(Config(), backbone_future_weight=weight).validate()


@pytest.mark.parametrize("discount", [-0.1, 1.5, float("inf")])
def test_discount_range(discount):
    with pytest.raises(ValueError):
        replace(Config(), discount=discount).validate()


def test_horizon_must_be_positive():
    with pytest.raises(ValueError):
        replace(Config(), horizon=0).validate()
    replace(Config(), horizon=1, discount=0.0).validate()


def test_schedule_multiplier_drives_every_group():
    config = replace(Config(), iterations=200, warmup_steps=10)
    assert config.schedule_at(0) == pytest.approx(0.1)
    assert config.schedule_at(9) == pytest.approx(1.0)
    assert config.schedule_at(199) == pytest.approx(0.1)
    assert config.learning_rate_at(9) == pytest.approx(config.learning_rate)


def test_environment_overrides_new_fields(monkeypatch):
    monkeypatch.setenv("FUTURE_CREDIT_STREAM_OBJECTIVE", "future_bag")
    monkeypatch.setenv("FUTURE_CREDIT_STREAM_HORIZON", "16")
    monkeypatch.setenv("FUTURE_CREDIT_STREAM_DISCOUNT", "0.5")
    monkeypatch.setenv("RUN_ID", "ffn_bag_half")
    config = Config.from_env()
    assert config.objective == "future_bag"
    assert config.horizon == 16 and config.page_horizon == 16
    assert config.discount == 0.5
    assert config.run_id == "ffn_bag_half"
