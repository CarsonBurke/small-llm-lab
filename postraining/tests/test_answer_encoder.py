import copy
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from postraining.answer_encoder.data import (
    SHARD_HEADER_BYTES,
    SHARD_MAGIC,
    SHARD_VERSION,
    MaskedPredictionBatch,
    MixedCorpus,
    MaskedPredictionBatchSampler,
    SpanMaskConfig,
    TokenExample,
    TokenShard,
    ViewBatchSampler,
    ViewConfig,
    make_example_views,
    make_span_mask,
    make_view,
)
from postraining.answer_encoder.inference import (
    FrozenAnswerScorer,
    FrozenPatchSetScorer,
    load_encoder_checkpoint,
    load_target_cache,
    maximum_weight_set_similarity,
    save_target_cache,
)
from postraining.answer_encoder.model import (
    ANSWER_ENCODER_SCHEMA,
    AnswerEncoderConfig,
    AnswerSimilarityReward,
    GlobalLatentObjective,
    GlobalLatentPredictor,
    LeJEPAObjective,
    LeJEPAObjectiveConfig,
    SIGReg,
    TextAnswerEncoder,
)
from postraining.answer_encoder.train import (
    _diagnose_batches,
    _encode_view_batch,
    _encode_global_latent_batch,
    _flatten_views,
    _learning_rate,
    _load_module_training_state,
    _module_state_payload,
    _resolved_projection_dimension,
    _split_answer_examples,
    _validate_resume_checkpoint_args,
    _validate_training_geometry,
    build_parser,
    evaluate_objective,
    evaluate_global_latent_objective,
)


def tiny_config() -> AnswerEncoderConfig:
    return AnswerEncoderConfig(
        vocab_size=64,
        max_tokens=12,
        model_dim=16,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2,
        projection_hidden_dim=32,
        projection_dim=8,
        dropout=0.0,
        pad_token_id=62,
        mask_token_id=63,
    )


def tiny_predictor() -> GlobalLatentPredictor:
    return GlobalLatentPredictor(
        tiny_config().model_dim,
        32,
        num_heads=tiny_config().num_heads,
        num_layers=2,
    )


def test_encoder_and_lejepa_objective_have_finite_gradients():
    torch.manual_seed(0)
    model = TextAnswerEncoder(tiny_config())
    token_ids = torch.tensor(
        [
            [1, 2, 3, 62],
            [1, 3, 2, 62],
            [4, 5, 6, 7],
            [4, 6, 5, 7],
        ]
    )
    mask = token_ids != 62
    _, projection = model(token_ids, mask)
    views = projection.reshape(2, 2, -1)
    objective = LeJEPAObjective(
        LeJEPAObjectiveConfig(sigreg_weight=0.02, sigreg_slices=8)
    )
    loss, diagnostics = objective(views)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_sequence_hidden_api_preserves_existing_cls_output():
    model = TextAnswerEncoder(tiny_config()).eval()
    token_ids = torch.tensor([[1, 2, 3, 62], [4, 5, 62, 62]])
    attention = token_ids != 62
    hidden = model.encode_hidden(token_ids, attention)
    assert hidden.shape == (2, 5, tiny_config().model_dim)
    assert torch.equal(hidden[:, 0], model.encode_backbone(token_ids, attention))


def test_compacted_context_keeps_original_token_positions():
    model = TextAnswerEncoder(tiny_config()).eval()
    token_ids = torch.tensor([[7, 8]])
    attention = torch.ones_like(token_ids, dtype=torch.bool)
    compact = model.encode_hidden(
        token_ids[:, 1:],
        attention[:, 1:],
        position_ids=torch.tensor([[1]]),
    )
    reindexed = model.encode_hidden(
        token_ids[:, 1:],
        attention[:, 1:],
        position_ids=torch.tensor([[0]]),
    )
    default = model.encode_hidden(token_ids[:, 1:], attention[:, 1:])

    # Contextualization differs after deletion, but position 1 is still encoded
    # in the same absolute coordinate system rather than silently becoming 0.
    assert not torch.equal(compact[:, 1], reindexed[:, 1])
    assert torch.equal(default, reindexed)


def test_global_predictor_uses_variable_cardinality_memory():
    torch.manual_seed(9)
    predictor = tiny_predictor().eval()
    context = torch.randn(2, 3, tiny_config().model_dim)
    context_attention = torch.tensor([[True, True, False], [True, True, True]])

    global_prediction = predictor(context, context_attention)
    changed_context = context.clone()
    changed_context[:, 1, 0] += 1.0
    changed_global = predictor(changed_context, context_attention)

    assert global_prediction.shape == (2, tiny_config().model_dim)
    assert not torch.equal(global_prediction, changed_global)


def test_predictor_layers_are_initialized_independently():
    predictor = tiny_predictor()
    first, second = predictor.decoder.layers
    assert not torch.equal(
        first.self_attn.in_proj_weight,
        second.self_attn.in_proj_weight,
    )
    assert not torch.equal(
        first.multihead_attn.in_proj_weight,
        second.multihead_attn.in_proj_weight,
    )


def test_predictor_resumes_exactly_but_is_omitted_from_final_checkpoint():
    torch.manual_seed(14)
    model = TextAnswerEncoder(tiny_config())
    predictor = tiny_predictor()
    training_payload = copy.deepcopy(
        _module_state_payload(model, predictor, training=True)
    )
    expected = {
        key: value.clone() for key, value in predictor.state_dict().items()
    }

    with torch.no_grad():
        for parameter in predictor.parameters():
            parameter.add_(1.0)
    _load_module_training_state(training_payload, model, predictor)

    assert all(
        torch.equal(predictor.state_dict()[key], value)
        for key, value in expected.items()
    )
    final_payload = _module_state_payload(model, predictor, training=False)
    assert set(final_payload) == {"model"}


def test_global_latent_objective_preserves_two_view_lejepa_mse_scale():
    objective = GlobalLatentObjective(
        LeJEPAObjectiveConfig(sigreg_weight=0.0, sigreg_slices=4)
    )
    predicted_global = torch.zeros(2, 3)
    target_global = torch.full((2, 3), 2.0)
    loss, diagnostics = objective(predicted_global, target_global)

    assert diagnostics["global_prediction_mse"] == pytest.approx(4.0)
    assert diagnostics["invariance_loss"] == pytest.approx(1.0)
    assert loss == pytest.approx(1.0)
    assert not any("patch" in key for key in diagnostics)


def test_global_latent_reports_reference_view_center_cosine():
    objective = GlobalLatentObjective(
        LeJEPAObjectiveConfig(sigreg_weight=0.0, sigreg_slices=4)
    )
    first = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    second = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    _, diagnostics = objective(first, second)

    assert diagnostics["global_prediction_cosine"] == pytest.approx(0.0)
    assert diagnostics["view_center_cosine"] == pytest.approx(2.0**-0.5)


def test_global_latent_trains_target_context_projector_and_predictor():
    torch.manual_seed(4)
    model = TextAnswerEncoder(tiny_config()).train()
    predictor = tiny_predictor().train()
    target_ids = torch.tensor([[10, 12], [11, 13]])
    attention = torch.ones_like(target_ids, dtype=torch.bool)
    batch = MaskedPredictionBatch(
        target_ids=target_ids,
        target_attention_mask=attention,
        prediction_mask=torch.tensor([[True, False], [True, False]]),
        context_ids=torch.tensor([[12], [13]]),
        context_attention_mask=torch.ones(2, 1, dtype=torch.bool),
        context_position_ids=torch.ones(2, 1, dtype=torch.long),
    )
    outputs = _encode_global_latent_batch(
        model,
        predictor,
        batch,
    )
    objective = GlobalLatentObjective(
        LeJEPAObjectiveConfig(sigreg_weight=0.02, sigreg_slices=8)
    )
    loss, diagnostics = objective(
        outputs["predicted_global"], outputs["target_global"]
    )
    loss.backward()

    assert set(outputs) == {
        "predicted_global",
        "target_global",
        "target_global_backbone",
        "masked_fraction",
        "visible_fraction",
        "visible_tokens_per_sample",
    }
    assert not any("patch" in key for key in diagnostics)
    assert model.token_embedding.weight.grad[10].abs().sum() > 0
    assert model.token_embedding.weight.grad[11].abs().sum() > 0
    assert model.token_embedding.weight.grad[12].abs().sum() > 0
    assert model.token_embedding.weight.grad[13].abs().sum() > 0
    mask_gradient = model.token_embedding.weight.grad[tiny_config().mask_token_id]
    assert mask_gradient.abs().sum() == 0
    assert any(parameter.grad is not None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.projector.parameters())
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in predictor.parameters()
    )


def test_global_latent_outputs_only_one_cls_per_sample():
    model = TextAnswerEncoder(tiny_config()).eval()
    predictor = tiny_predictor().eval()
    batch = MaskedPredictionBatch(
        target_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        target_attention_mask=torch.ones(2, 3, dtype=torch.bool),
        prediction_mask=torch.tensor(
            [[False, True, False], [True, False, True]]
        ),
        context_ids=torch.tensor([[1, 3], [5, 62]]),
        context_attention_mask=torch.tensor([[True, True], [True, False]]),
        context_position_ids=torch.tensor([[0, 2], [1, 0]]),
    )

    outputs = _encode_global_latent_batch(model, predictor, batch)

    assert outputs["predicted_global"].shape == (2, tiny_config().projection_dim)
    assert outputs["target_global"].shape == (2, tiny_config().projection_dim)
    assert outputs["visible_tokens_per_sample"] == pytest.approx(1.5)


def test_fully_deleted_single_token_context_has_finite_backward():
    model = TextAnswerEncoder(tiny_config()).train()
    predictor = tiny_predictor().train()
    sampler = MaskedPredictionBatchSampler(
        _SingleTokenCorpus(),
        SpanMaskConfig(probability=0.3, mean_span_length=3.0),
        max_tokens=6,
        min_source_tokens=1,
        pad_token_id=62,
        seed=5,
    )
    outputs = _encode_global_latent_batch(model, predictor, sampler.batch(2))
    objective = GlobalLatentObjective(
        LeJEPAObjectiveConfig(sigreg_weight=0.02, sigreg_slices=8)
    )
    loss, _ = objective(outputs["predicted_global"], outputs["target_global"])
    loss.backward()

    assert torch.isfinite(loss)
    assert outputs["visible_tokens_per_sample"] == 0
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in predictor.parameters()
    )


def test_prediction_only_gradient_reaches_visible_context_not_target_value():
    config = AnswerEncoderConfig(
        **{**tiny_config().to_dict(), "projector_normalization": "layer"}
    )
    model = TextAnswerEncoder(config).train()
    predictor = tiny_predictor().train()
    batch = MaskedPredictionBatch(
        target_ids=torch.tensor([[10, 12], [11, 13]]),
        target_attention_mask=torch.ones(2, 2, dtype=torch.bool),
        prediction_mask=torch.tensor([[True, False], [True, False]]),
        context_ids=torch.tensor([[12], [13]]),
        context_attention_mask=torch.ones(2, 1, dtype=torch.bool),
        context_position_ids=torch.ones(2, 1, dtype=torch.long),
    )
    outputs = _encode_global_latent_batch(model, predictor, batch)
    prediction_only = outputs["predicted_global"].square().mean()
    prediction_only.backward()

    assert model.token_embedding.weight.grad[10].abs().sum() == 0
    assert model.token_embedding.weight.grad[11].abs().sum() == 0
    assert model.token_embedding.weight.grad[12].abs().sum() > 0
    assert model.token_embedding.weight.grad[13].abs().sum() > 0


def test_variable_cardinality_predictions_ignore_padding_and_batch_neighbors():
    model = TextAnswerEncoder(tiny_config()).eval()
    predictor = tiny_predictor().eval()
    alone = MaskedPredictionBatch(
        target_ids=torch.tensor([[1, 2]]),
        target_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        prediction_mask=torch.tensor([[False, True]]),
        context_ids=torch.tensor([[1]]),
        context_attention_mask=torch.ones(1, 1, dtype=torch.bool),
        context_position_ids=torch.zeros(1, 1, dtype=torch.long),
    )
    mixed = MaskedPredictionBatch(
        target_ids=torch.tensor([[1, 2, 62, 62], [3, 4, 5, 6]]),
        target_attention_mask=torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        ),
        prediction_mask=torch.tensor(
            [[False, True, False, False], [True, False, True, False]]
        ),
        context_ids=torch.tensor([[1, 62], [4, 6]]),
        context_attention_mask=torch.tensor([[True, False], [True, True]]),
        context_position_ids=torch.tensor([[0, 0], [1, 3]]),
    )

    alone_outputs = _encode_global_latent_batch(model, predictor, alone)
    mixed_outputs = _encode_global_latent_batch(model, predictor, mixed)

    for key in ("predicted_global", "target_global", "target_global_backbone"):
        assert torch.allclose(
            alone_outputs[key][0], mixed_outputs[key][0], atol=1e-6
        )


def test_sigreg_matches_lejepa_minimal_reference_formula():
    config = LeJEPAObjectiveConfig(sigreg_knots=17, sigreg_slices=32)
    values = torch.randn(4, 16, 8, generator=torch.Generator().manual_seed(17))
    sigreg = SIGReg(config)

    torch.manual_seed(23)
    directions = torch.randn(values.size(-1), config.sigreg_slices)
    directions /= directions.norm(p=2, dim=0)
    angles = (values @ directions).unsqueeze(-1) * sigreg.t
    error = (
        (angles.cos().mean(-3) - sigreg.gaussian_cf).square()
        + angles.sin().mean(-3).square()
    )
    expected = ((error @ sigreg.weights) * values.size(-2)).mean()

    torch.manual_seed(23)
    assert sigreg(values) == pytest.approx(float(expected), abs=1e-7)


def test_reference_projector_matches_torchvision_mlp_defaults():
    config = tiny_config()
    config = AnswerEncoderConfig(**{**config.to_dict(), "reference_projector": True})
    model = TextAnswerEncoder(config)
    linears = [module for module in model.projector.modules() if isinstance(module, torch.nn.Linear)]
    activations = [module for module in model.projector.modules() if isinstance(module, torch.nn.ReLU)]
    dropouts = [module for module in model.projector.modules() if isinstance(module, torch.nn.Dropout)]
    assert len(linears) == 3
    assert all(module.bias is not None for module in linears)
    assert len(activations) == 2
    assert len(dropouts) == 3
    assert all(module.p == 0.0 for module in dropouts)


def test_deployable_projector_uses_per_example_normalization():
    config = tiny_config()
    config = AnswerEncoderConfig(
        **{
            **config.to_dict(),
            "reference_projector": True,
            "projector_normalization": "layer",
        }
    )
    model = TextAnswerEncoder(config)
    assert len([module for module in model.projector.modules() if isinstance(module, torch.nn.LayerNorm)]) == 2
    assert not any(
        isinstance(module, torch.nn.BatchNorm1d) for module in model.projector.modules()
    )


def test_reference_learning_rate_matches_update_timing_and_nonzero_floor():
    rates = [_learning_rate(step, 2_000, 100, 2e-3, 1e-3) for step in range(2_000)]
    assert rates[0] == pytest.approx(2e-5)
    assert rates[99] == pytest.approx(1.9802e-3)
    assert rates[100] == pytest.approx(2e-3)
    assert rates[-1] > 1e-3
    assert rates[-1] == pytest.approx(
        1e-3 + 1e-3 * 0.5 * (1.0 + math.cos(math.pi * 1_899 / 1_900))
    )


def test_resume_checkpoint_is_bound_to_objective_space():
    manifest = {"name": "run", "steps": 2_000, "objective_space": "backbone"}
    checkpoint = {"name": "run", "steps": 1_000, "objective_space": "projection"}
    with pytest.raises(ValueError, match="objective_space"):
        _validate_resume_checkpoint_args(checkpoint, manifest)


def test_legacy_resume_checkpoint_defaults_to_projection_objective():
    manifest = {"name": "run", "steps": 2_000, "objective_space": "projection"}
    checkpoint = {"name": "run", "steps": 1_000}
    _validate_resume_checkpoint_args(checkpoint, manifest)


def test_view_flattening_preserves_sample_major_reference_order():
    token_ids = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    attention = torch.ones_like(token_ids, dtype=torch.bool)
    flat_ids, flat_attention = _flatten_views(token_ids, attention)
    assert torch.equal(flat_ids, token_ids.reshape(6, 4))
    assert torch.equal(flat_attention, attention.reshape(6, 4))


def test_direct_backbone_objective_bypasses_projector_gradients():
    model = TextAnswerEncoder(tiny_config()).train()
    token_ids = torch.tensor(
        [
            [[1, 2, 3, 62], [1, 2, 4, 62]],
            [[5, 6, 7, 62], [5, 6, 8, 62]],
        ]
    )
    attention = token_ids != 62
    embeddings = _encode_view_batch(
        model,
        token_ids,
        attention,
        objective_space="backbone",
    )
    embeddings.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.backbone.parameters())
    assert all(parameter.grad is None for parameter in model.projector.parameters())


def test_answer_split_is_disjoint_and_deterministic():
    examples = [TokenExample(tokens=(token,)) for token in range(1, 21)]
    first_train, first_val = _split_answer_examples(examples, 0.2, seed=7)
    second_train, second_val = _split_answer_examples(examples, 0.2, seed=7)
    key = lambda rows: [tuple(map(int, row.tokens.tolist())) for row in rows]
    assert key(first_train) == key(second_train)
    assert key(first_val) == key(second_val)
    assert set(key(first_train)).isdisjoint(key(first_val))


def test_cls_pooling_preserves_token_order_information():
    model = TextAnswerEncoder(tiny_config()).eval()
    token_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [4, 5, 6, 1, 2, 3],
        ]
    )
    attention = torch.ones_like(token_ids, dtype=torch.bool)
    for space in ("backbone", "projection"):
        embeddings = model.encode(token_ids, attention, space=space)
        assert not torch.allclose(embeddings[0], embeddings[1])


def test_transformer_layers_are_independently_initialized():
    model = TextAnswerEncoder(tiny_config())
    weights = [layer.self_attn.in_proj_weight for layer in model.backbone.layers]
    assert all(not torch.equal(weights[0], weight) for weight in weights[1:])


def test_answer_reward_uses_temperature_scaled_cosine_kernel():
    reward = AnswerSimilarityReward(torch.tensor([1.0, 0.0]))
    answers = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert torch.allclose(
        reward(answers),
        torch.tensor([1.0, math.exp(-10.0), math.exp(-20.0)]),
    )


def test_answer_reward_rejects_invalid_temperature():
    for temperature in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="temperature"):
            AnswerSimilarityReward(
                torch.tensor([1.0, 0.0]), temperature=temperature
            )


def test_answer_reward_temperature_controls_sharpness():
    target = torch.tensor([1.0, 0.0])
    orthogonal = torch.tensor([[0.0, 1.0]])
    sharp = AnswerSimilarityReward(target, temperature=0.1)(orthogonal)
    broad = AnswerSimilarityReward(target, temperature=0.5)(orthogonal)
    assert sharp.item() == pytest.approx(math.exp(-10.0))
    assert broad.item() == pytest.approx(math.exp(-2.0))
    assert sharp < broad


def test_views_can_shuffle_blocks_and_always_keep_a_token():
    example = TokenExample(
        tokens=(1, 2, 3, 4, 5),
        blocks=((1, 2), (4, 5)),
    )
    config = ViewConfig(
        global_views=2,
        local_views=2,
        global_scale_min=1.0,
        global_scale_max=1.0,
        local_scale_min=0.01,
        local_scale_max=0.01,
        mask_probability=0.0,
        block_shuffle_probability=1.0,
    )
    views = make_example_views(
        example,
        config,
        max_view_tokens=8,
        block_separator_tokens=(3,),
        mask_token_id=63,
        rng=random.Random(4),
    )

    assert len(views) == 4
    assert all(view for view in views)
    assert all(len(view) == 1 for view in views[2:])
    assert all(sorted(view) == [1, 2, 3, 4, 5] for view in views[:2])


def test_default_training_recipe_uses_standard_multicrop_views_and_projected_reward():
    args = build_parser().parse_args(["train", "--name", "test"])
    assert args.training_objective == "multicrop"
    assert args.global_views == 2
    assert args.local_views == 6
    assert (args.global_scale_min, args.global_scale_max) == (0.3, 1.0)
    assert (args.local_scale_min, args.local_scale_max) == (0.05, 0.3)
    assert args.mask_probability == 0.1
    assert args.objective_space == "projection"
    assert args.reward_space == "projection"
    assert args.projector_normalization == "layer"
    assert args.checkpoint_interval_seconds == 480.0
    assert _resolved_projection_dimension(args) == 16


def test_global_latent_recipe_uses_256d_cls_and_training_only_predictor():
    args = build_parser().parse_args(
        ["train", "--name", "test", "--training-objective", "global-latent"]
    )
    assert args.span_mask_probability == pytest.approx(0.3)
    assert args.mean_mask_span_length == pytest.approx(3.0)
    assert args.predictor_hidden_dim == 1024
    assert args.predictor_layers == 2
    assert args.objective_space == "projection"
    assert args.reward_space == "projection"
    assert _resolved_projection_dimension(args) == 256

    overridden = build_parser().parse_args(
        [
            "train",
            "--name",
            "test",
            "--training-objective",
            "global-latent",
            "--projection-dim",
            "64",
        ]
    )
    assert _resolved_projection_dimension(overridden) == 64


def test_global_latent_requires_projected_objective_and_deployed_reward():
    args = build_parser().parse_args(
        ["train", "--name", "test", "--training-objective", "global-latent"]
    )
    _validate_training_geometry(args)

    args.reward_space = "backbone"
    with pytest.raises(ValueError, match="deployment operates in projection"):
        _validate_training_geometry(args)
    args.reward_space = "projection"
    args.objective_space = "backbone"
    with pytest.raises(ValueError, match="prediction operates in projection"):
        _validate_training_geometry(args)


def test_default_multicrop_views_are_genuinely_distinct():
    tokens = tuple(range(1, 257))
    views = make_example_views(
        TokenExample(tokens=tokens),
        ViewConfig(),
        max_view_tokens=256,
        block_separator_tokens=(257,),
        mask_token_id=258,
        rng=random.Random(7),
    )
    global_lengths = [len(view) for view in views[:2]]
    local_lengths = [len(view) for view in views[2:]]
    assert all(77 <= length <= 256 for length in global_lengths)
    assert all(13 <= length <= 77 for length in local_lengths)
    assert len({tuple(view) for view in views}) == 8


def test_masking_never_erases_the_entire_answer():
    view = make_view(
        TokenExample(tokens=(10,)),
        scale=(1.0, 1.0),
        max_view_tokens=8,
        mask_probability=1.0,
        block_shuffle_probability=0.0,
        block_separator_tokens=(3,),
        mask_token_id=63,
        rng=random.Random(0),
    )
    assert view == [10]


def test_span_mask_has_exact_target_count_and_contiguous_content():
    mask = make_span_mask(
        10,
        SpanMaskConfig(probability=0.3, mean_span_length=3.0),
        rng=random.Random(5),
    )
    assert len(mask) == 10
    assert sum(mask) == 3


class _SingleTokenCorpus:
    def sample(self, rng, min_tokens, max_tokens):
        del rng, min_tokens, max_tokens
        return TokenExample(tokens=(10,))


def test_masked_prediction_sampler_fully_masks_single_token_target():
    sampler = MaskedPredictionBatchSampler(
        _SingleTokenCorpus(),
        SpanMaskConfig(probability=0.3, mean_span_length=3.0),
        max_tokens=6,
        min_source_tokens=1,
        pad_token_id=62,
        seed=3,
    )
    batch = sampler.batch(2)
    assert torch.equal(batch.target_ids, torch.tensor([[10], [10]]))
    assert torch.equal(batch.context_ids, torch.tensor([[62], [62]]))
    assert not bool(batch.context_attention_mask.any())
    assert bool(batch.target_attention_mask.all())
    assert bool(batch.prediction_mask.all())


class _VariableLengthCorpus:
    def __init__(self):
        self.index = 0

    def sample(self, rng, min_tokens, max_tokens):
        del rng, min_tokens, max_tokens
        rows = ((1, 2, 3, 4), (5, 6))
        row = rows[self.index % len(rows)]
        self.index += 1
        return TokenExample(tokens=row)


def test_masked_prediction_sampler_preserves_alignment_and_excludes_padding():
    sampler = MaskedPredictionBatchSampler(
        _VariableLengthCorpus(),
        SpanMaskConfig(probability=0.5, mean_span_length=2.0),
        max_tokens=6,
        min_source_tokens=1,
        pad_token_id=62,
        seed=7,
    )
    batch = sampler.batch(2)
    assert not bool(
        (batch.prediction_mask & ~batch.target_attention_mask).any()
    )
    assert bool((batch.target_ids[~batch.target_attention_mask] == 62).all())
    assert bool(batch.prediction_mask.any(dim=1).all())
    for sample in range(2):
        target = batch.target_ids[sample][batch.target_attention_mask[sample]]
        missing = batch.prediction_mask[sample][batch.target_attention_mask[sample]]
        expected_positions = (~missing).nonzero(as_tuple=False).squeeze(1)
        visible_count = int(batch.context_attention_mask[sample].sum())
        assert torch.equal(
            batch.context_ids[sample, :visible_count], target[~missing]
        )
        assert torch.equal(
            batch.context_position_ids[sample, :visible_count], expected_positions
        )
    assert batch.context_ids.shape[1] < batch.target_ids.shape[1]


def test_masked_prediction_sampler_state_roundtrip_reproduces_next_batch():
    answers = [TokenExample(tokens=(token, token + 1, token + 2)) for token in range(10, 20)]

    def make_sampler(seed):
        return MaskedPredictionBatchSampler(
            MixedCorpus(_OneExampleCorpus(), answers, answer_probability=1.0),
            SpanMaskConfig(probability=0.5, mean_span_length=2.0),
            max_tokens=6,
            min_source_tokens=1,
            pad_token_id=62,
            seed=seed,
        )

    original = make_sampler(11)
    original.batch(3)
    state = original.state_dict()
    expected = original.batch(4)
    restored = make_sampler(999)
    restored.load_state_dict(state)
    actual = restored.batch(4)
    assert all(
        torch.equal(getattr(expected, field), getattr(actual, field))
        for field in expected.__dataclass_fields__
    )


def test_token_shard_is_memory_mapped_and_sampled(tmp_path):
    path = tmp_path / "tokens.bin"
    header = np.zeros(256, dtype="<i4")
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = 10
    with path.open("wb") as stream:
        stream.write(header.tobytes())
        stream.write(np.arange(10, dtype="<u2").tobytes())

    shard = TokenShard(path)
    sampled = shard.sample(random.Random(0), min_tokens=4, max_tokens=4)
    assert path.stat().st_size == SHARD_HEADER_BYTES + 20
    assert len(sampled.tokens) == 4
    assert all(0 <= int(token) < 10 for token in sampled.tokens)


class _OneExampleCorpus:
    def sample(self, rng, min_tokens, max_tokens):
        del rng, min_tokens, max_tokens
        return TokenExample(tokens=(1, 2, 3))


def _tiny_sampler() -> ViewBatchSampler:
    return ViewBatchSampler(
        _OneExampleCorpus(),
        ViewConfig(
            global_views=2,
            local_views=0,
            global_scale_min=1.0,
            global_scale_max=1.0,
            mask_probability=0.0,
        ),
        max_view_tokens=6,
        min_source_tokens=1,
        block_separator_tokens=(3,),
        pad_token_id=62,
        mask_token_id=63,
    )


def test_global_latent_evaluation_reports_full_deployed_cls_rank():
    sampler = MaskedPredictionBatchSampler(
        _OneExampleCorpus(),
        SpanMaskConfig(probability=0.5, mean_span_length=2.0),
        max_tokens=6,
        min_source_tokens=1,
        pad_token_id=62,
        seed=2,
    )
    config = AnswerEncoderConfig(
        **{**tiny_config().to_dict(), "projection_dim": 256}
    )
    model = TextAnswerEncoder(config)
    predictor = tiny_predictor()
    objective = GlobalLatentObjective(
        LeJEPAObjectiveConfig(sigreg_slices=8)
    )
    metrics = evaluate_global_latent_objective(
        model,
        predictor,
        objective,
        sampler,
        batches=2,
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert metrics["rank_sample_count"] == 8
    assert metrics["objective_effective_rank"] == metrics["projection_effective_rank"]
    assert metrics["projection_dimension"] == 256
    assert metrics["objective_dimension"] == 256
    assert not any("patch" in key for key in metrics)


def test_batch_stat_evaluation_preserves_bn_state_and_reports_rank_population():
    model = TextAnswerEncoder(tiny_config()).eval()
    objective = LeJEPAObjective(LeJEPAObjectiveConfig(sigreg_slices=8))
    batch_norms = [
        module
        for module in model.projector.modules()
        if isinstance(module, torch.nn.BatchNorm1d)
    ]
    before = [
        (module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone())
        for module in batch_norms
    ]

    metrics = evaluate_objective(
        model,
        objective,
        _tiny_sampler(),
        batches=2,
        batch_size=4,
        device=torch.device("cpu"),
    )

    assert metrics["rank_sample_count"] == 8
    assert "projection_effective_rank" in metrics
    assert "backbone_effective_rank" in metrics
    assert metrics["objective_effective_rank"] == metrics["projection_effective_rank"]
    for module, (mean, variance, count) in zip(batch_norms, before, strict=True):
        assert not module.training
        assert module.track_running_stats
        assert torch.equal(module.running_mean, mean)
        assert torch.equal(module.running_var, variance)
        assert torch.equal(module.num_batches_tracked, count)


def test_direct_backbone_objective_reports_deployed_representation_rank():
    model = TextAnswerEncoder(tiny_config()).eval()
    objective = LeJEPAObjective(LeJEPAObjectiveConfig(sigreg_slices=8))
    metrics = evaluate_objective(
        model,
        objective,
        _tiny_sampler(),
        batches=2,
        batch_size=4,
        device=torch.device("cpu"),
        objective_space="backbone",
    )
    assert metrics["objective_effective_rank"] == metrics["backbone_effective_rank"]


def test_batch_stat_diagnostic_ignores_bad_running_statistics_without_mutation():
    model = TextAnswerEncoder(tiny_config()).eval()
    objective = LeJEPAObjective(LeJEPAObjectiveConfig(sigreg_slices=8))
    batch_norms = [
        module
        for module in model.projector.modules()
        if isinstance(module, torch.nn.BatchNorm1d)
    ]
    for module in batch_norms:
        module.running_mean.fill_(100.0)
        module.running_var.fill_(0.01)
    before = [
        (module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone())
        for module in batch_norms
    ]
    rows = torch.tensor(
        [
            [1, 2, 3, 62, 62, 62],
            [4, 5, 6, 62, 62, 62],
            [7, 8, 9, 10, 62, 62],
            [11, 12, 13, 14, 15, 62],
        ]
    )
    token_ids = rows[:, None, :].expand(-1, 2, -1).clone()
    attention = token_ids != 62
    batches = [(token_ids, attention)]

    stored = _diagnose_batches(
        model,
        objective,
        batches,
        device=torch.device("cpu"),
        projector_batch_statistics=False,
    )
    current = _diagnose_batches(
        model,
        objective,
        batches,
        device=torch.device("cpu"),
        projector_batch_statistics=True,
    )

    assert stored["loss"] != pytest.approx(current["loss"])
    for module, (mean, variance, count) in zip(batch_norms, before, strict=True):
        assert not module.training
        assert module.track_running_stats
        assert torch.equal(module.running_mean, mean)
        assert torch.equal(module.running_var, variance)
        assert torch.equal(module.num_batches_tracked, count)


def test_mixed_corpus_visits_every_answer_before_repeating():
    answers = [TokenExample(tokens=(token,)) for token in (10, 11, 12)]
    corpus = MixedCorpus(_OneExampleCorpus(), answers, answer_probability=1.0)
    rng = random.Random(9)
    first_epoch = [int(corpus.sample(rng, 1, 3).tokens[0]) for _ in answers]
    assert sorted(first_epoch) == [10, 11, 12]


def test_view_sampler_state_roundtrip_reproduces_exact_next_batch():
    answers = [TokenExample(tokens=(token, token + 1, token + 2)) for token in range(10, 20)]
    config = ViewConfig(global_views=2, local_views=0, mask_probability=0.3)

    def make_sampler(seed):
        return ViewBatchSampler(
            MixedCorpus(_OneExampleCorpus(), answers, answer_probability=1.0),
            config,
            max_view_tokens=6,
            min_source_tokens=1,
            block_separator_tokens=(3,),
            pad_token_id=62,
            mask_token_id=63,
            seed=seed,
        )

    original = make_sampler(11)
    original.batch(3)
    state = original.state_dict()
    expected = original.batch(4)
    restored = make_sampler(999)
    restored.load_state_dict(state)
    actual = restored.batch(4)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual, strict=True))


def test_view_batch_sampler_pads_variable_views():
    sampler = ViewBatchSampler(
        _OneExampleCorpus(),
        ViewConfig(global_views=2, local_views=1),
        max_view_tokens=6,
        min_source_tokens=1,
        block_separator_tokens=(3,),
        pad_token_id=62,
        mask_token_id=63,
    )
    token_ids, attention_mask = sampler.batch(2)
    assert token_ids.shape == (2, 3, 3)
    assert attention_mask.shape == token_ids.shape
    assert bool(attention_mask.any(dim=-1).all())
    assert bool((token_ids[~attention_mask] == 62).all())


def test_frozen_scorer_preencodes_target():
    model = TextAnswerEncoder(tiny_config()).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scorer = FrozenAnswerScorer(
        model,
        lambda text: [ord(character) % 62 for character in text],
        device=torch.device("cpu"),
    )
    target = scorer.preencode_target("ten")
    scores = scorer.score(["ten", "nine"], target)
    assert target.shape == (1, tiny_config().model_dim)
    assert scores.shape == (2,)
    assert scores[0].item() == pytest.approx(1.0, abs=1e-6)


def test_patch_set_matching_is_reorder_invariant_and_localizes_a_bug():
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reordered = target.flip(0)
    bugged = torch.tensor([[1.0, 0.0], [0.0, -1.0]])

    exact = maximum_weight_set_similarity(
        reordered, target, penalize_unmatched=True
    )
    bug = maximum_weight_set_similarity(
        bugged, target, penalize_unmatched=True
    )

    assert exact == pytest.approx(1.0)
    assert bug == pytest.approx(0.0)
    assert exact > bug


def test_patch_set_matching_reports_tokenizer_cardinality_separately():
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidate = target[:1]

    cardinality = maximum_weight_set_similarity(
        candidate, target, penalize_unmatched=True
    )
    matched_only = maximum_weight_set_similarity(
        candidate, target, penalize_unmatched=False
    )

    assert cardinality == pytest.approx(0.0)
    assert matched_only == pytest.approx(1.0)
    assert maximum_weight_set_similarity(
        target, candidate, penalize_unmatched=True
    ) == pytest.approx(cardinality)
    assert maximum_weight_set_similarity(
        target, candidate, penalize_unmatched=False
    ) == pytest.approx(matched_only)


def test_frozen_patch_set_scorer_returns_exact_self_match():
    config = AnswerEncoderConfig(
        **{**tiny_config().to_dict(), "projector_normalization": "layer"}
    )
    model = TextAnswerEncoder(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scorer = FrozenPatchSetScorer(
        model,
        lambda text: [ord(character) % 62 for character in text],
        device=torch.device("cpu"),
        space="projection",
    )

    target = scorer.preencode_target("ten")
    scores = scorer.score(["ten", "nine"], target)

    assert target.shape == (3, config.projection_dim)
    assert scores.shape == (2,)
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    assert scores[1] < scores[0]


def test_frozen_scorer_keeps_suffix_beyond_training_length():
    model = TextAnswerEncoder(tiny_config()).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scorer = FrozenAnswerScorer(
        model,
        lambda text: [ord(character) % 62 for character in text],
        device=torch.device("cpu"),
    )
    shared_prefix = "a" * tiny_config().max_tokens
    embeddings = scorer.encode([shared_prefix + "correct", shared_prefix + "wrong"])
    assert not torch.allclose(embeddings[0], embeddings[1])


def test_behavioral_gate_authorizes_only_its_embedding_space(tmp_path):
    model = TextAnswerEncoder(tiny_config())
    checkpoint = tmp_path / "encoder.pt"
    torch.save(
        {
            "schema": ANSWER_ENCODER_SCHEMA,
            "model": model.state_dict(),
            "encoder_config": tiny_config().to_dict(),
            "behavioral_gate_passed": True,
            "behavioral_gate_space": "backbone",
        },
        checkpoint,
    )

    loaded, _ = load_encoder_checkpoint(
        checkpoint,
        torch.device("cpu"),
        require_behavioral_gate=True,
        required_embedding_space="backbone",
    )
    assert not loaded.training
    with pytest.raises(ValueError, match="does not authorize"):
        load_encoder_checkpoint(
            checkpoint,
            torch.device("cpu"),
            require_behavioral_gate=True,
            required_embedding_space="projection",
        )


def test_target_cache_is_bound_to_checkpoint_hash(tmp_path):
    checkpoint = tmp_path / "encoder.pt"
    checkpoint.write_text(json.dumps({"schema": ANSWER_ENCODER_SCHEMA}))
    cache = tmp_path / "targets.pt"
    save_target_cache(
        cache,
        checkpoint=checkpoint,
        space="projection",
        targets={"problem-1": torch.tensor([[1.0, 0.0]])},
    )
    loaded = load_target_cache(
        cache,
        checkpoint=checkpoint,
        expected_space="projection",
    )
    assert torch.equal(loaded["problem-1"], torch.tensor([[1.0, 0.0]]))

    checkpoint.write_text("changed")
    with pytest.raises(ValueError, match="different encoder checkpoint"):
        load_target_cache(
            cache,
            checkpoint=checkpoint,
            expected_space="projection",
        )


def test_patch_set_probe_entrypoint_imports_from_outside_repository(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/probe_answer_patch_sets.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "order-invariant patch-set readouts" in completed.stdout
