from __future__ import annotations

import math

import torch
from torch import nn

from scripts.ablation import MetricsWriter, parse_log_line
from pretraining.train_bolmo import (
    ExampleStream,
    PerHeadMuon,
    SOURCE_LR_REPORT,
    SOURCE_MUON_ALGORITHM,
    SourceMuon,
    SourceOptimizerRecipe,
    StageOptimizer,
    batch_totals,
    cosine_warmup_decay,
    global_per_head_projections,
    microbatch_objective,
    paper_stage_schedule,
    source_recipe_optimizers,
    split_parameter_groups,
    TRAINING_RECIPES,
    validate_paper_data_manifest,
    validate_checkpoint_tokenizer,
    validate_stage1_resume_checkpoint,
    validate_training_contract,
)
from pretraining.bolmo import BolmoArchitecture, BolmoLoss, BolmoModel
from pretraining.nanogpt_mini.nanogpt_mini_kda_model import Block, RMSNorm


def test_stopped_run_is_chronological_prefix_of_paper_schedule() -> None:
    planned_stage1, planned_stage2, stage1, stage2 = paper_stage_schedule(2000, 1000)
    assert (planned_stage1, planned_stage2) == (667, 1333)
    assert (stage1, stage2) == (667, 333)


def test_full_run_preserves_paper_stage_and_token_ratios() -> None:
    planned_stage1, planned_stage2, stage1, stage2 = paper_stage_schedule(2000, 2000)
    assert (planned_stage1, planned_stage2) == (stage1, stage2)
    # Table 8 doubles the Stage-2 batch. Integer rounding is within one update
    # of the paper's 1:4 Stage-1:Stage-2 source-token ratio.
    assert abs((stage2 * 256) / (stage1 * 128) - 4.0) < 0.01


def test_stage1_only_stop_does_not_prematurely_enter_stage2() -> None:
    assert paper_stage_schedule(2000, 500) == (667, 1333, 500, 0)


def test_paper_manifest_rejects_old_bos_plus_full_sequence_geometry() -> None:
    manifest = {
        "examples": {
            "source_sequence_length": 2048,
            "training_real_source_tokens": 2048,
            "training_stored_source_width": 2049,
            "validation_stored_source_width": 2050,
        },
        "splits": {
            "train": {
                "context_source_tokens": 0,
                "skip_last_source_token": False,
            },
            "validation": {
                "context_source_tokens": 1,
                "skip_last_source_token": False,
            },
        },
    }
    try:
        validate_paper_data_manifest(manifest)
    except ValueError as error:
        assert "non-paper row geometry" in str(error)
    else:
        raise AssertionError("old BOS-plus-full-sequence geometry was accepted")


def test_example_stream_skip_preserves_the_next_row() -> None:
    stream = ExampleStream.__new__(ExampleStream)
    stream.repeat = False
    stream.epoch = 0
    stream._iterator = iter(
        {"source_ids": torch.tensor([value])} for value in range(5)
    )
    assert stream.skip(3) == 3
    assert stream.take(1)[0]["source_ids"].item() == 3


def test_stage1_resume_checkpoint_is_bound_to_schedule_and_data() -> None:
    class FakeModel:
        def __init__(self, *, chunk_size: int = 128, encoder_layers: int = 1):
            self.chunk_size = chunk_size
            self.encoder_layers = encoder_layers

        def export_config(self) -> dict:
            return {
                "architecture": {
                    "backend": "triton",
                    "chunk_size": self.chunk_size,
                    "autocast_kernel_dtype": "bfloat16",
                    "encoder_layers": self.encoder_layers,
                }
            }

    checkpoint = {
        "architecture": "bolmo_kda_nope_v1",
        "training_recipe": TRAINING_RECIPES["paper"],
        "source_checkpoint_sha256": "source",
        "data_manifest_sha256": "data",
        "completed_steps": 667,
        "planned_steps": 2000,
        "stage1_planned_steps": 667,
        "stage1_completed_steps": 667,
        "stage2_completed_steps": 0,
        "stage1_boundary_accuracy": 0.992,
        "stage1_boundary_precision": 0.993,
        "stage1_boundary_recall": 0.991,
        "stage1_oracle_byte_bpb": 1.4,
        "stage1_predicted_oracle_bpb_gap": 0.02,
        "stage1_train_diagnostics": {
            "boundary": 0.01,
            "ce": 1.0,
            "encoder_stitch": 0.1,
            "encoder_stitch_cosine": 0.01,
            "decoder_distill": 0.1,
        },
        "model_config": {
            "architecture": {
                "backend": "triton",
                "chunk_size": 128,
                "autocast_kernel_dtype": "float32",
                "encoder_layers": 1,
            }
        },
        "model": {"weight": torch.tensor(1.0)},
    }
    validate_stage1_resume_checkpoint(
        checkpoint,
        model=FakeModel(chunk_size=64),  # type: ignore[arg-type]
        source_sha256="source",
        data_manifest_sha256="data",
        training_recipe=TRAINING_RECIPES["paper"],
        planned_steps=2000,
        planned_stage1=667,
        minimum_boundary_accuracy=0.99,
    )
    checkpoint["data_manifest_sha256"] = "wrong"
    try:
        validate_stage1_resume_checkpoint(
            checkpoint,
            model=FakeModel(),  # type: ignore[arg-type]
            source_sha256="source",
            data_manifest_sha256="data",
            training_recipe=TRAINING_RECIPES["paper"],
            planned_steps=2000,
            planned_stage1=667,
            minimum_boundary_accuracy=0.99,
        )
    except ValueError as error:
        assert "data_manifest_sha256" in str(error)
    else:
        raise AssertionError("a mismatched Stage-1 resume checkpoint was accepted")

    checkpoint["data_manifest_sha256"] = "data"
    try:
        validate_stage1_resume_checkpoint(
            checkpoint,
            model=FakeModel(encoder_layers=2),  # type: ignore[arg-type]
            source_sha256="source",
            data_manifest_sha256="data",
            training_recipe=TRAINING_RECIPES["paper"],
            planned_steps=2000,
            planned_stage1=667,
            minimum_boundary_accuracy=0.99,
        )
    except ValueError as error:
        assert "model_config" in str(error)
    else:
        raise AssertionError("an architecture-changing resume was accepted")

    checkpoint["stage1_boundary_accuracy"] = 0.98
    try:
        validate_stage1_resume_checkpoint(
            checkpoint,
            model=FakeModel(),  # type: ignore[arg-type]
            source_sha256="source",
            data_manifest_sha256="data",
            training_recipe=TRAINING_RECIPES["paper"],
            planned_steps=2000,
            planned_stage1=667,
            minimum_boundary_accuracy=0.99,
        )
    except ValueError as error:
        assert "boundary gate" in str(error)
    else:
        raise AssertionError("a Stage-1 checkpoint below the boundary gate was accepted")

    # The explicit experimental override waives only the refusal.
    validate_stage1_resume_checkpoint(
        checkpoint,
        model=FakeModel(),  # type: ignore[arg-type]
        source_sha256="source",
        data_manifest_sha256="data",
        training_recipe=TRAINING_RECIPES["paper"],
        planned_steps=2000,
        planned_stage1=667,
        minimum_boundary_accuracy=0.99,
        allow_gate_failure=True,
    )

    # A missing gate result is still fatal, override or not: the override
    # preserves a failing measurement, it does not excuse an absent one.
    del checkpoint["stage1_boundary_accuracy"]
    try:
        validate_stage1_resume_checkpoint(
            checkpoint,
            model=FakeModel(),  # type: ignore[arg-type]
            source_sha256="source",
            data_manifest_sha256="data",
            training_recipe=TRAINING_RECIPES["paper"],
            planned_steps=2000,
            planned_stage1=667,
            minimum_boundary_accuracy=0.99,
            allow_gate_failure=True,
        )
    except ValueError as error:
        assert "no boundary gate result" in str(error)
    else:
        raise AssertionError("an ungated Stage-1 checkpoint was accepted")

    # Unrelated contract mismatches stay fatal under the override.
    checkpoint["stage1_boundary_accuracy"] = 0.98
    checkpoint["source_checkpoint_sha256"] = "wrong"
    try:
        validate_stage1_resume_checkpoint(
            checkpoint,
            model=FakeModel(),  # type: ignore[arg-type]
            source_sha256="source",
            data_manifest_sha256="data",
            training_recipe=TRAINING_RECIPES["paper"],
            planned_steps=2000,
            planned_stage1=667,
            minimum_boundary_accuracy=0.99,
            allow_gate_failure=True,
        )
    except ValueError as error:
        assert "source_checkpoint_sha256" in str(error)
    else:
        raise AssertionError("the override waived an unrelated lineage mismatch")


def _counting_row(source_ids: list[int]) -> dict[str, torch.Tensor]:
    source_ids_tensor = torch.tensor(source_ids)
    return {
        "source_ids": source_ids_tensor,
        "source_valid_mask": torch.ones_like(source_ids_tensor, dtype=torch.bool),
        "valid_mask": torch.ones(len(source_ids) + 2, dtype=torch.bool),
    }


def test_batch_totals_count_eos_coalesced_stitch_patches() -> None:
    rows = [_counting_row([0, 7, 0, 8]), _counting_row([0, 9, 10])]
    totals = batch_totals(rows)
    assert totals["patches"] == 7
    assert totals["stitch_patches"] == 6
    assert totals["target_patches"] == 5


def test_batch_totals_preserve_initial_patch_before_immediate_eot() -> None:
    totals = batch_totals([_counting_row([0, 0, 7])])
    assert totals["patches"] == 3
    assert totals["stitch_patches"] == 3


def test_stage1_microbatch_stitch_weighting_is_grouping_invariant() -> None:
    rows = [_counting_row([0, 7, 0, 8]), _counting_row([0, 9, 10])]
    totals = batch_totals(rows)
    first_counts = batch_totals(rows[:1])
    second_counts = batch_totals(rows[1:])

    def loss(stitch: float) -> BolmoLoss:
        zero = torch.tensor(0.0)
        return BolmoLoss(
            total=zero,
            ce=zero,
            boundary=zero,
            encoder_stitch=torch.tensor(stitch),
            decoder_distill=zero,
        )

    grouped = (
        microbatch_objective(loss(2.0), first_counts, totals, stage=1)
        + microbatch_objective(loss(5.0), second_counts, totals, stage=1)
    )
    # The first row has three aligned patches and the second has three.
    torch.testing.assert_close(grouped, torch.tensor(3.5))


def test_only_embedding_tables_are_exempt_from_weight_decay() -> None:
    module = nn.ModuleDict(
        {
            "embedding": nn.Embedding(4, 3),
            "projection": nn.Linear(3, 3),
            "norm": nn.LayerNorm(3),
        }
    )
    groups = split_parameter_groups(
        list(module.named_parameters()), weight_decay=0.1, lr=1e-3
    )
    by_decay = {group["weight_decay"]: set(map(id, group["params"])) for group in groups}
    assert id(module["embedding"].weight) in by_decay[0.0]
    assert id(module["projection"].bias) in by_decay[0.1]
    assert id(module["norm"].weight) in by_decay[0.1]


def test_bolmo_metrics_are_parsed_and_namespaced_for_tensorboard() -> None:
    entry = parse_log_line(
        "step:10/2000 train_loss:2.5 stage_id:1 bolmo_boundary:0.2 "
        "train_time:1000ms"
    )
    assert entry is not None
    assert entry["bolmo_boundary"] == 0.2
    assert MetricsWriter._extra_scalar_tag("bolmo_boundary") == "bolmo/boundary"
    assert (
        MetricsWriter._extra_scalar_tag("predicted_bytes_per_patch")
        == "bolmo/predicted_bytes_per_patch"
    )
    assert (
        MetricsWriter._extra_scalar_tag("stage_source_tokens_per_second")
        == "perf/stage_source_tokens_per_second"
    )
    assert (
        MetricsWriter._extra_scalar_tag("train_device_time_ms")
        == "perf/train_device_time_ms"
    )
    assert MetricsWriter._extra_scalar_tag("eval_seconds") == "perf/eval_seconds"
    assert MetricsWriter._extra_scalar_tag("val_joint_bpb") == "val/joint_bpb"
    assert (
        MetricsWriter._extra_scalar_tag("val_boundary_accuracy")
        == "val/boundary_accuracy"
    )


def test_dataset_tokenizer_must_match_checkpoint_identity() -> None:
    checkpoint = {
        "model_config": {
            "tokenizer_provenance": {
                "kind": "toast_tst",
                "vocab_size": 50_257,
                "eot_id": 0,
                "spec_sha256": "spec",
                "ngrams_sha256": "ngrams",
            }
        }
    }
    source_meta = {
        "kind": "toast_tst",
        "logical_vocab_size": 50_257,
        "eot_id": 0,
        "spec_file_sha256": "spec",
        "ngrams_sha256": "ngrams",
        "tst_group_size": 1,
        "tst_compound": True,
    }
    validate_checkpoint_tokenizer(checkpoint, source_meta)
    source_meta["ngrams_sha256"] = "other"
    try:
        validate_checkpoint_tokenizer(checkpoint, source_meta)
    except ValueError as error:
        assert "dataset tokenizer differs" in str(error)
    else:
        raise AssertionError("a tokenizer mismatch was accepted")


def _source_reference_multiplier(
    step: int, train_steps: int, warmup_fraction: float
) -> float:
    """``nanogpt_mini_gpt2vocab_kda_3to1_pm_train.set_hparams``, cosine branch."""

    progress = step / train_steps
    assert 0 <= progress < 1
    if progress < warmup_fraction:
        return (step + 1) / max(1, round(train_steps * warmup_fraction))
    post_warmup = (progress - warmup_fraction) / (1 - warmup_fraction)
    return 0.5 * (1 + math.cos(math.pi * post_warmup))


def test_cosine_schedule_reproduces_the_source_trainer() -> None:
    planned = 1600
    for step in range(1, planned + 1):
        assert cosine_warmup_decay(step, planned, 0.01) == _source_reference_multiplier(
            step - 1, planned, 0.01
        )
    # Unlike the paper's linear decay the cosine body never reaches zero within
    # its horizon, so the last update is still a real update.
    assert cosine_warmup_decay(planned, planned, 0.01) > 0.0
    for outside in (0, planned + 1):
        try:
            cosine_warmup_decay(outside, planned, 0.01)
        except ValueError as error:
            assert "outside" in str(error)
        else:
            raise AssertionError("a stage step outside the horizon was accepted")


def _recipe_model() -> BolmoModel:
    """A Bolmo whose trunk carries both mixer types the source recipe covers."""

    torch.manual_seed(0)
    blocks = nn.Sequential(
        *(
            Block(
                128,
                use_kda=use_kda,
                mlp_hidden=32,
                delta_num_heads=2,
                delta_full_rank_gate=False,
                delta_mlp_on_delta=False,
                dense_position_encoding="none",
            )
            for use_kda in (True, False)
        )
    )
    return BolmoModel(
        global_blocks=blocks,
        global_norm=RMSNorm(128),
        source_embedding=torch.randn(24, 128),
        source_model_config={"model_dim": 128},
        architecture=BolmoArchitecture(
            model_dim=128,
            local_heads=4,
            local_ffn_hidden=32,
            backend="native",
            chunk_size=4,
            num_special_tokens=1,
        ),
    )


def test_source_recipe_partitions_every_trainable_parameter_exactly_once() -> None:
    model = _recipe_model()
    model.freeze_global(False)
    optimizers = source_recipe_optimizers(
        model, stage=2, recipe=SourceOptimizerRecipe()
    )
    grouped = [
        parameter
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_source_recipe_routes_parameters_to_the_source_learning_rates() -> None:
    model = _recipe_model()
    model.freeze_global(False)
    recipe = SourceOptimizerRecipe()
    optimizers = source_recipe_optimizers(model, stage=2, recipe=recipe)
    lr_of = {}
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                lr_of[id(parameter)] = (group["lr"], group["tag"])
    named = dict(model.named_parameters())
    assert lr_of[id(named["local_encoder.expanded_embedding.weight"])] == (
        recipe.embed_lr,
        "embed",
    )
    assert lr_of[id(named["local_decoder.lm_head.weight"])] == (recipe.proj_lr, "proj")
    assert lr_of[id(named["global_blocks.0.attn.A_log"])] == (
        recipe.scalar_lr,
        "delta_decay",
    )
    # Rank three, so rank alone would have sent it to Muon.
    assert lr_of[id(named["global_blocks.0.attn.q_conv1d.weight"])] == (
        recipe.delta_conv_lr,
        "delta_conv",
    )
    assert lr_of[id(named["global_blocks.0.norm1.gains"])] == (
        recipe.scalar_lr,
        "scalar",
    )
    assert lr_of[id(named["global_blocks.0.attn.o_proj.weight"])] == (
        recipe.muon_lr,
        "muon_global",
    )
    assert lr_of[id(named["local_decoder.blocks.0.ffn.down.weight"])] == (
        recipe.muon_lr,
        "muon_local",
    )
    per_head = {
        id(parameter): heads
        for parameter, heads in global_per_head_projections(model)
    }
    assert per_head[id(named["global_blocks.0.attn.q_proj.weight"])] == 2
    assert per_head[id(named["global_blocks.1.attn.v.weight"])] == 1


def test_source_recipe_stage1_refuses_an_unfrozen_trunk() -> None:
    model = _recipe_model()
    model.freeze_global(True)
    stage1 = source_recipe_optimizers(model, stage=1, recipe=SourceOptimizerRecipe())
    tags = {group["tag"] for optimizer in stage1 for group in optimizer.param_groups}
    assert "muon_global" not in tags and "delta_conv" not in tags
    model.freeze_global(False)
    try:
        source_recipe_optimizers(model, stage=1, recipe=SourceOptimizerRecipe())
    except ValueError as error:
        assert "must be" in str(error) and "frozen" in str(error)
    else:
        raise AssertionError("Stage 1 accepted a trainable trunk")


def test_unrecognized_trunk_attention_is_rejected_rather_than_downgraded() -> None:
    model = _recipe_model()

    class Foreign(nn.Module):
        pass

    model.global_blocks[0].attn = Foreign()
    try:
        global_per_head_projections(model)
    except ValueError as error:
        assert "per-head Muon recipe does not cover" in str(error)
    else:
        raise AssertionError("an unknown mixer silently skipped per-head Muon")


def test_stage_optimizer_scales_peak_lrs_and_warms_muon_momentum() -> None:
    recipe = SourceOptimizerRecipe()
    matrix = torch.nn.Parameter(torch.zeros(4, 4))
    scalar = torch.nn.Parameter(torch.zeros(4))
    adam = torch.optim.AdamW([{"params": [scalar], "lr": recipe.scalar_lr}])
    for group in adam.param_groups:
        group["tag"] = "scalar"
    muon = SourceMuon([matrix], lr=recipe.muon_lr, mu=recipe.muon_momentum_warmup_start)
    for group in muon.param_groups:
        group["tag"] = "muon_local"
    optimizer = StageOptimizer(
        [adam, muon],
        planned_steps=1600,
        schedule=lambda step, planned: cosine_warmup_decay(
            step, planned, recipe.warmup_fraction
        ),
        report=SOURCE_LR_REPORT,
        momentum_warmup_steps=recipe.muon_momentum_warmup_steps,
        momentum_start=recipe.muon_momentum_warmup_start,
        momentum_end=recipe.muon_momentum,
    )
    multiplier = optimizer.apply_schedule(1)
    report = optimizer.learning_rate_report()
    assert report["lr_scalar"] == recipe.scalar_lr * multiplier
    assert report["lr_local"] == recipe.muon_lr * multiplier
    # No trunk group in this stage; the report must say zero, not borrow a
    # learning rate from another group.
    assert report["lr_global"] == 0.0
    assert report["muon_mu"] == recipe.muon_momentum_warmup_start
    optimizer.apply_schedule(recipe.muon_momentum_warmup_steps + 1)
    assert optimizer.learning_rate_report()["muon_mu"] == recipe.muon_momentum
    halfway = optimizer.momentum_for(recipe.muon_momentum_warmup_steps // 2 + 1)
    assert recipe.muon_momentum_warmup_start < halfway < recipe.muon_momentum


def test_cosine_warmup_never_exceeds_the_peak_on_a_rounded_down_horizon() -> None:
    # 1333 * 0.01 rounds down to 13, so the source's unrounded branch test
    # would spend stage step 14 at 1.077x the peak learning rate.
    planned = 1333
    multipliers = [cosine_warmup_decay(step, planned, 0.01) for step in range(1, 40)]
    assert max(multipliers) == 1.0
    assert multipliers[12] == 1.0 and multipliers[13] == 1.0
    # Horizons that round exactly are unchanged, so parity with the source run
    # on the horizons it actually used is preserved.
    for exact in (1600, 2000):
        assert max(
            cosine_warmup_decay(step, exact, 0.01) for step in range(1, 40)
        ) == 1.0


def test_paper_contract_does_not_gain_a_key_that_would_break_old_resumes() -> None:
    # The recipe identifier already names the arm, so adding a second key would
    # only make every pre-existing Stage-1 checkpoint un-resumable.
    contract = {"recipe": TRAINING_RECIPES["paper"], "seed": 1337}
    rng_state = {"python": (), "torch_cpu": torch.tensor(0), "torch_cuda": []}
    validate_training_contract(
        {"training_contract": dict(contract), "rng_state": rng_state},
        contract,
    )
    try:
        validate_training_contract(
            {
                "training_contract": contract | {"optimizer_recipe": "paper"},
                "rng_state": rng_state,
            },
            contract,
        )
    except ValueError as error:
        assert "training contract mismatch" in str(error)
    else:
        raise AssertionError("an extra contract key was silently accepted")


def test_source_recipe_binds_its_structural_choices_into_the_contract() -> None:
    contract = SourceOptimizerRecipe().as_contract()
    assert contract["per_head_muon"] is True
    assert contract["schedule"] == "cosine"
    assert contract["momentum_warmup_scope"] == "per_stage"
    assert contract["schedule_scope"] == "per_stage"
    assert contract["muon_algorithm"] == SOURCE_MUON_ALGORITHM
    for unsupported in (
        {"schedule": "linear"},
        {"momentum_warmup_scope": "per_run", "schedule_scope": "per_run"},
        # The momentum ramp has to follow the learning-rate horizon; restarting
        # it inside an unbroken decay is neither of the two supported shapes.
        {"schedule_scope": "continuous"},
        {"momentum_warmup_scope": "continuous"},
    ):
        try:
            SourceOptimizerRecipe(**unsupported)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{unsupported} was accepted")


def test_disabling_per_head_muon_moves_the_trunk_qkv_into_whole_matrix_muon() -> None:
    model = _recipe_model()
    model.freeze_global(False)
    recipe = SourceOptimizerRecipe(per_head_muon=False)
    optimizers = source_recipe_optimizers(model, stage=2, recipe=recipe)
    assert not any(isinstance(optimizer, PerHeadMuon) for optimizer in optimizers)
    tagged = {
        id(parameter): group["tag"]
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    named = dict(model.named_parameters())
    assert tagged[id(named["global_blocks.0.attn.q_proj.weight"])] == "muon_global"


def test_an_unsupported_trunk_mixer_is_refused_before_stage_one_spends_gpu_time() -> None:
    model = _recipe_model()

    class Foreign(nn.Module):
        pass

    model.global_blocks[0].attn = Foreign()
    model.freeze_global(True)
    try:
        source_recipe_optimizers(model, stage=1, recipe=SourceOptimizerRecipe())
    except ValueError as error:
        assert "per-head Muon recipe does not cover" in str(error)
    else:
        raise AssertionError("Stage 1 started with a trunk Stage 2 cannot optimize")


def test_continuous_scope_carries_one_decay_across_the_stage_boundary() -> None:
    # The two shapes, on the launched plan: 1,000 updates split 333/667.
    planned, stage1 = 1000, 333
    recipe = SourceOptimizerRecipe(
        schedule_scope="continuous", momentum_warmup_scope="continuous"
    )
    assert recipe.continuous_schedule
    assert not SourceOptimizerRecipe().continuous_schedule

    # Per stage: Stage 1 anneals to ~0 by its own step 333, then Stage 2 warms
    # back up to the full peak over its own 1% and anneals again. Two anneals
    # separated by a restart.
    stage2 = planned - stage1
    assert cosine_warmup_decay(stage1, stage1) < 1e-4
    assert cosine_warmup_decay(round(stage2 * 0.01), stage2) == 1.0
    assert cosine_warmup_decay(stage2, stage2) < 1e-4

    # Continuous: one horizon, so the boundary is invisible to the schedule.
    # The trunk unfreezes at 0.760 of peak rather than 1.0 — closer to the
    # 0.509 the source had reached when we forked its checkpoint.
    boundary = cosine_warmup_decay(stage1 + 1, planned)
    assert 0.75 < boundary < 0.77
    assert cosine_warmup_decay(stage1, planned) > boundary
    assert cosine_warmup_decay(planned, planned) < 1e-4
    # Monotone the whole way down after warmup, which is what "no restart"
    # has to mean and what a per-stage horizon violates at the boundary.
    multipliers = [
        cosine_warmup_decay(step, planned) for step in range(11, planned + 1)
    ]
    assert all(
        later <= earlier for earlier, later in zip(multipliers, multipliers[1:])
    )
