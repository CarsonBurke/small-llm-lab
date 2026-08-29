from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import pytest
import torch

import pretraining.byte_diffusion.inference as inference_module
import pretraining.eval_byte_diffusion_gsm8k as gsm_module
from pretraining.byte_diffusion.config import ByteDiffusionConfig
from pretraining.byte_diffusion.inference import (
    EntropyPatchedCanvasGenerator,
    _variable_prefix_metadata,
    denoise_entropy_blt_reference,
    denoise_entropy_blt_reference_batched,
    entropy_next_byte_starts_patch,
    prepare_entropy_blt_end,
)
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.patching import (
    CausalEntropyPatcher,
    EntropyPatchConfig,
    HashedNgramEntropyConfig,
    HashedNgramEntropyModel,
)
from pretraining.byte_diffusion.sampling import BatchedCanvasSample
from pretraining.byte_diffusion.variable_patching import (
    ENTROPY_DATASET_SCHEMA,
    PATCHING_POLICY_SCHEMA,
)
from pretraining.eval_byte_diffusion_gsm8k import (
    entropy_blt_generate_bytes,
    load_entropy_patcher_for_checkpoint,
    validate_serving_contract,
)


def _patcher(
    *, threshold: float = 9.0, max_patch_size: int = 4
) -> CausalEntropyPatcher:
    model = HashedNgramEntropyModel(
        HashedNgramEntropyConfig(vocab_size=261, table_size=8)
    )
    return CausalEntropyPatcher(
        model,
        EntropyPatchConfig(
            mode="cumulative",
            threshold=threshold,
            max_patch_size=max_patch_size,
        ),
    )


def _arbitrary_boundary(patcher: CausalEntropyPatcher) -> int:
    ids = torch.zeros(24, dtype=torch.long)
    plan = patcher.patch(ids.numpy(), torch.zeros(24, dtype=torch.long).numpy())
    return next(
        int(start)
        for start in plan.layout.patch_starts
        if start > 1 and start % 4
    )


def test_variable_blt_accepts_authenticated_arbitrary_start_and_rejects_nonboundary() -> None:
    torch.manual_seed(3)
    patcher = _patcher()
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    boundary = _arbitrary_boundary(patcher)
    prefix = torch.zeros(boundary, dtype=torch.long)

    logits = denoise_entropy_blt_reference(
        model,
        patcher,
        prefix,
        torch.full((4,), model.config.vocab.mask_id, dtype=torch.long),
    )
    assert logits.shape == (4, model.config.vocab.output_size)

    probe = torch.zeros(24, dtype=torch.long)
    authenticated = set(
        map(
            int,
            patcher.patch(
                probe.numpy(), torch.zeros_like(probe).numpy()
            ).layout.patch_starts,
        )
    )
    nonboundary = next(
        index for index in range(boundary + 1, 20) if index not in authenticated
    )
    clean = torch.zeros(nonboundary + 1, dtype=torch.long)
    metadata, _ = _variable_prefix_metadata(patcher, clean)
    starts = torch.tensor([[nonboundary]], dtype=torch.long)
    with pytest.raises(ValueError, match="authenticated physical patch"):
        model.forward_blt_d_branches(
            clean[None],
            metadata.pop("valid"),
            torch.full((1, 1, 4), model.config.vocab.mask_id, dtype=torch.long),
            torch.ones((1, 1, 4), dtype=torch.bool),
            starts,
            document_ids=metadata.pop("document_ids"),
            branch_condition_indices=torch.zeros((1, 1), dtype=torch.long),
            max_patch_size=4,
            **metadata,
        )


@pytest.mark.parametrize("block_length", [4, 8, 16])
def test_entropy_blt_block_horizon_is_independent_of_patch_topology(
    block_length: int,
) -> None:
    patcher = _patcher(max_patch_size=5)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    boundary = _arbitrary_boundary(patcher)
    _, layout = _variable_prefix_metadata(
        patcher, torch.zeros(boundary + 1, dtype=torch.long)
    )
    origin = layout.physical_patch_start_columns.eq(boundary).nonzero().item()
    expected_condition = int(
        layout.physical_patch_prior_condition_indices[origin]
    )
    observed: dict[str, object] = {}
    original = model.forward_blt_d_branches

    def capture(*args, **kwargs):
        observed["noisy_shape"] = args[2].shape
        observed["branch_valid_shape"] = args[3].shape
        observed["start"] = args[4].item()
        observed["condition"] = kwargs["branch_condition_indices"].item()
        observed["max_patch_size"] = kwargs["max_patch_size"]
        return original(*args, **kwargs)

    model.forward_blt_d_branches = capture  # type: ignore[method-assign]

    logits = denoise_entropy_blt_reference(
        model,
        patcher,
        torch.zeros(boundary, dtype=torch.long),
        torch.full(
            (block_length,), model.config.vocab.mask_id, dtype=torch.long
        ),
    )

    assert logits.shape == (block_length, model.config.vocab.output_size)
    assert observed == {
        "noisy_shape": (1, 1, block_length),
        "branch_valid_shape": (1, 1, block_length),
        "start": boundary,
        "condition": expected_condition,
        "max_patch_size": patcher.config.max_patch_size,
    }


@pytest.mark.parametrize("block_length", [4, 8, 16])
def test_entropy_generator_diffuses_immediately_and_commits_the_whole_block(
    monkeypatch: pytest.MonkeyPatch,
    block_length: int,
) -> None:
    patcher = _patcher()
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    probe = torch.zeros(24, dtype=torch.long)
    starts = set(
        map(
            int,
            patcher.patch(
                probe.numpy(), torch.zeros_like(probe).numpy()
            ).layout.patch_starts,
        )
    )
    prefix_length = next(index for index in range(2, 20) if index not in starts)
    prefix = torch.zeros(prefix_length, dtype=torch.long)
    assert not entropy_next_byte_starts_patch(patcher, prefix)
    seen_prefixes: list[torch.Tensor] = []

    def denoise(
        _model: ByteDiffusionModel,
        _patcher: CausalEntropyPatcher,
        committed: tuple[torch.Tensor, ...],
        noisy: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        assert len(committed) == 1
        seen_prefixes.append(committed[0].clone())
        assert not committed[0].eq(model.config.vocab.mask_id).any()
        assert noisy.eq(model.config.vocab.mask_id).all()
        assert noisy.shape == (1, block_length)
        return torch.zeros((1, block_length, model.config.vocab.output_size))

    candidate = torch.arange(65, 65 + block_length, dtype=torch.long)[None]

    def sample(
        initial: torch.Tensor,
        denoiser,
        **_: object,
    ) -> BatchedCanvasSample:
        denoiser(initial)
        return BatchedCanvasSample(
            ids=candidate.clone(),
            active=torch.ones_like(candidate, dtype=torch.bool),
            useful_nfe=torch.ones(1, dtype=torch.long),
            executed_nfe=1,
        )

    monkeypatch.setattr(
        inference_module, "denoise_entropy_blt_reference_batched", denoise
    )
    monkeypatch.setattr(
        inference_module, "sample_absorbing_canvas_batched", sample
    )
    generator = EntropyPatchedCanvasGenerator(
        model, patcher, block_length=block_length, seed=7
    )
    generator.prefill(prefix)
    generated = generator.generate_blt(steps=4, stochastic=False)

    combined = torch.cat((prefix, candidate[0]))
    assert not generated.alignment_ids.numel()
    torch.testing.assert_close(generated.committed_canvas_ids, candidate[0])
    assert not generated.overflow_canvas_ids.numel()
    torch.testing.assert_close(generator.ids, combined)
    assert generator.rejected_bytes == 0
    assert len(seen_prefixes) == 1
    torch.testing.assert_close(seen_prefixes[0], prefix)


def test_entropy_end_origin_uses_final_open_latent_without_clean_outputs() -> None:
    torch.manual_seed(19)
    patcher = _patcher()
    model = ByteDiffusionModel(
        ByteDiffusionConfig.tiny(
            decoder_conditioning="split_cross_attention",
            decoder_prefix_window=None,
            decoder_branch_attention="shared_flex",
        )
    ).eval()
    probe = torch.zeros(24, dtype=torch.long)
    starts = set(
        map(
            int,
            patcher.patch(
                probe.numpy(), torch.zeros_like(probe).numpy()
            ).layout.patch_starts,
        )
    )
    prefix_length = next(index for index in range(2, 20) if index not in starts)
    prefix = torch.zeros(prefix_length, dtype=torch.long)
    _, prefix_layout = _variable_prefix_metadata(patcher, prefix)
    expected_final_latent = int(
        prefix_layout.physical_to_global_patch_indices[-1]
    )
    observed: dict[str, object] = {}
    original = model.forward_blt_d_branches

    def capture(*args, **kwargs):
        observed["clean_ids"] = args[0].clone()
        observed["start"] = int(args[4].item())
        observed["condition"] = int(kwargs["branch_condition_indices"].item())
        observed["return_clean_logits"] = kwargs["return_clean_logits"]
        observed["return_clean_patch_states"] = kwargs["return_clean_patch_states"]
        output = original(*args, **kwargs)
        observed["clean_logits_shape"] = tuple(output.clean_logits.shape)
        observed["clean_patch_shape"] = tuple(output.clean_patch_states.shape)
        return output

    model.forward_blt_d_branches = capture  # type: ignore[method-assign]
    logits = denoise_entropy_blt_reference(
        model,
        patcher,
        prefix,
        torch.full((4,), model.config.vocab.mask_id, dtype=torch.long),
    )

    clean_ids = observed["clean_ids"]
    assert isinstance(clean_ids, torch.Tensor)
    torch.testing.assert_close(clean_ids[0, :prefix_length], prefix)
    assert int(clean_ids[0, prefix_length]) == 0
    assert observed["start"] == prefix_length
    assert observed["condition"] == expected_final_latent
    assert observed["return_clean_logits"] is False
    assert observed["return_clean_patch_states"] is False
    assert observed["clean_logits_shape"] == (1, 0, model.config.vocab.output_size)
    assert observed["clean_patch_shape"][:2] == (1, 0)
    assert logits.shape == (4, model.config.vocab.output_size)


def test_entropy_batched_end_origins_match_independent_rows() -> None:
    torch.manual_seed(23)
    patcher = _patcher()
    model = ByteDiffusionModel(
        ByteDiffusionConfig.tiny(
            decoder_conditioning="split_cross_attention",
            decoder_prefix_window=None,
            decoder_branch_attention="shared_flex",
        )
    ).eval()
    prefixes = (
        torch.tensor([65, 66, 67], dtype=torch.long),
        torch.tensor([70, 71, 72, 73, 74, 75], dtype=torch.long),
    )
    noisy = torch.tensor(
        [
            [model.config.vocab.mask_id, 80, model.config.vocab.mask_id, 81],
            [82, model.config.vocab.mask_id, 83, model.config.vocab.mask_id],
        ],
        dtype=torch.long,
    )

    batched = denoise_entropy_blt_reference_batched(
        model, patcher, prefixes, noisy
    )
    plan = prepare_entropy_blt_end(model, patcher, prefixes, noisy.shape[1])
    planned = denoise_entropy_blt_reference_batched(
        model, patcher, prefixes, noisy, plan=plan
    )
    independent = torch.stack(
        [
            denoise_entropy_blt_reference(model, patcher, prefix, canvas)
            for prefix, canvas in zip(prefixes, noisy, strict=True)
        ]
    )
    torch.testing.assert_close(batched, independent, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(planned, batched, rtol=0, atol=0)


def test_entropy_eval_batches_nonboundary_prefixes_and_traces_every_nfe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patcher = _patcher()
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    probe = torch.zeros(24, dtype=torch.long)
    starts = set(
        map(
            int,
            patcher.patch(
                probe.numpy(), torch.zeros_like(probe).numpy()
            ).layout.patch_starts,
        )
    )
    prefix_length = next(index for index in range(2, 20) if index not in starts)

    observed_prefixes: list[tuple[torch.Tensor, ...]] = []

    def denoise(
        _model: ByteDiffusionModel,
        _patcher: CausalEntropyPatcher,
        committed_rows: tuple[torch.Tensor, ...],
        noisy: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        observed_prefixes.append(tuple(row.clone() for row in committed_rows))
        logits = torch.zeros(
            (*noisy.shape, model.config.vocab.output_size), dtype=torch.float32
        )
        for row in range(noisy.shape[0]):
            unresolved = noisy[row].eq(model.config.vocab.mask_id).nonzero().flatten()
            if unresolved.numel():
                logits[row, unresolved[0], 65] = 20.0
        return logits

    monkeypatch.setattr(gsm_module, "denoise_entropy_blt_reference_batched", denoise)
    prompts = [bytes(prefix_length), bytes(prefix_length + 1)]
    traces = [
        {"alignment_steps": [], "diffusion_blocks": []} for _ in prompts
    ]
    results = entropy_blt_generate_bytes(
        model,
        patcher,
        prompts,
        max_new_bytes=3,
        max_native_actions=10,
        context_bytes=32,
        block_length=4,
        stops=(),
        diffusion_steps=4,
        unmasking_strategy="confidence",
        confidence_threshold=0.7,
        entropy_budget=1.0,
        seed=13,
        stochastic=False,
        device=torch.device("cpu"),
        trace_records=traces,
    )

    assert observed_prefixes
    assert len(observed_prefixes[0]) == 2
    for expected, observed in zip(prompts, observed_prefixes[0], strict=True):
        torch.testing.assert_close(observed, torch.tensor(list(expected)))
    assert [result.raw for result in results] == [b"AAA", b"AAA"]
    assert all(result.termination == "byte_cap" for result in results)
    assert all(result.native_actions == 4 for result in results)
    assert all(result.model_forwards == 4 for result in results)
    for trace in traces:
        assert trace["alignment_steps"] == []
        blocks = trace["diffusion_blocks"]
        assert isinstance(blocks, list) and len(blocks) == 1
        assert len(blocks[0]["steps"]) == 4
        assert all(step["denoiser_executed"] for step in blocks[0]["steps"])
        assert blocks[0]["logical_active_nfe"] == 4
        assert blocks[0]["cohort_physical_nfe"] == 4
        assert blocks[0]["semantic_actions"] == [65, 65, 65]
        assert blocks[0]["committed_ids"] == [65, 65, 65]
        assert blocks[0]["overflow_ids"] == [65]
        assert blocks[0]["final_retained_output_bytes"] == 3
        assert trace["final_retained_output_bytes"] == 3


def test_entropy_trace_separates_logical_nfe_and_retained_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patcher = _patcher()
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()

    def denoise(
        _model: ByteDiffusionModel,
        _patcher: CausalEntropyPatcher,
        committed_rows: tuple[torch.Tensor, ...],
        noisy: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        assert len(committed_rows) == 2
        logits = torch.zeros(
            (*noisy.shape, model.config.vocab.output_size), dtype=torch.float32
        )
        unresolved_eot = noisy[0].eq(model.config.vocab.mask_id).nonzero().flatten()
        if unresolved_eot.numel():
            logits[0, unresolved_eot[0], model.config.vocab.eot_id] = 20.0
        unresolved_text = noisy[1].eq(model.config.vocab.mask_id).nonzero().flatten()
        if unresolved_text.numel():
            position = int(unresolved_text[0])
            logits[1, position, 65 + position] = 20.0
        return logits

    monkeypatch.setattr(gsm_module, "denoise_entropy_blt_reference_batched", denoise)
    traces = [
        {"alignment_steps": [], "diffusion_blocks": []},
        {"alignment_steps": [], "diffusion_blocks": []},
    ]
    results = entropy_blt_generate_bytes(
        model,
        patcher,
        [b"prompt one", b"prompt two"],
        max_new_bytes=4,
        max_native_actions=8,
        context_bytes=32,
        block_length=4,
        stops=("BC",),
        diffusion_steps=4,
        unmasking_strategy="confidence",
        confidence_threshold=0.7,
        entropy_budget=1.0,
        seed=29,
        stochastic=False,
        device=torch.device("cpu"),
        trace_records=traces,
    )

    assert results[0].termination == "eot" and results[0].raw == b""
    assert results[0].native_actions == 1
    assert results[1].termination == "text_stop" and results[1].raw == b"A"
    assert results[1].native_actions == 4
    assert all(result.model_forwards == 4 for result in results)

    eot_block = traces[0]["diffusion_blocks"][0]
    assert len(eot_block["steps"]) == 1
    assert eot_block["logical_active_nfe"] == 1
    assert eot_block["cohort_physical_nfe"] == 4
    assert eot_block["semantic_actions"] == [model.config.vocab.eot_id]
    assert eot_block["committed_ids"] == []
    assert eot_block["final_retained_output_bytes"] == 0

    text_block = traces[1]["diffusion_blocks"][0]
    assert len(text_block["steps"]) == 4
    assert text_block["logical_active_nfe"] == 4
    assert text_block["cohort_physical_nfe"] == 4
    assert text_block["semantic_actions"] == [65, 66, 67]
    assert text_block["committed_ids"] == [65]
    assert text_block["overflow_ids"] == [68]
    assert text_block["final_retained_output_bytes"] == 1


def test_entropy_eval_does_not_align_past_the_context_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patcher = _patcher()
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    probe = torch.zeros(24, dtype=torch.long)
    starts = set(
        map(
            int,
            patcher.patch(
                probe.numpy(), torch.zeros_like(probe).numpy()
            ).layout.patch_starts,
        )
    )
    prefix_length = next(index for index in range(2, 20) if index not in starts)

    def reject_ar(*_: object) -> torch.Tensor:
        raise AssertionError("context-capped alignment must not execute")

    monkeypatch.setattr(inference_module, "entropy_ar_next_logits", reject_ar)
    result = entropy_blt_generate_bytes(
        model,
        patcher,
        [bytes(prefix_length)],
        max_new_bytes=1,
        max_native_actions=10,
        context_bytes=prefix_length,
        block_length=4,
        stops=(),
        diffusion_steps=4,
        unmasking_strategy="confidence",
        confidence_threshold=0.7,
        entropy_budget=1.0,
        seed=17,
        stochastic=False,
        device=torch.device("cpu"),
    )[0]

    assert result.raw == b""
    assert result.termination == "context_cap"
    assert result.native_actions == 0


def test_entropy_patcher_is_loaded_only_through_checkpoint_pinned_manifest(
    tmp_path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    patcher = _patcher()
    artifact = patcher.to_bytes()
    artifact_path = dataset / "entropy-patcher.bdpatch"
    artifact_path.write_bytes(artifact)
    manifest = {
        "schema": ENTROPY_DATASET_SCHEMA,
        "packing": {"patch_stride": None},
        "patching": {
            "schema": PATCHING_POLICY_SCHEMA,
            "name": "causal_entropy_v1",
            "patcher_artifact": {
                "path": artifact_path.name,
                "sha256": hashlib.sha256(artifact).hexdigest(),
                "bytes": len(artifact),
            },
            "entropy_model_config": asdict(patcher.model.config),
            "boundary_config": asdict(patcher.config),
            "max_patch_size": patcher.config.max_patch_size,
        },
    }
    payload_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["payload_sha256"] = payload_hash
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    checkpoint = {
        "run_contract": {
            "patching_policy": "causal_entropy_v1",
            "corruption": {"canvas_length": 4},
        },
        "dataset_provenance": {"payload_sha256": payload_hash},
    }

    loaded = load_entropy_patcher_for_checkpoint(checkpoint, dataset)
    assert loaded is not None and loaded.sha256 == patcher.sha256
    for block_length in (4, 8, 16):
        checkpoint["run_contract"]["corruption"]["canvas_length"] = block_length
        validate_serving_contract(
            checkpoint,
            decode_mode="blt",
            block_length=block_length,
            entropy_patcher=loaded,
        )

    checkpoint["dataset_provenance"]["payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint-pinned"):
        load_entropy_patcher_for_checkpoint(checkpoint, dataset)
