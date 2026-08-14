from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import pytest
import torch

import pretraining.byte_diffusion.inference as inference_module
from pretraining.byte_diffusion.config import ByteDiffusionConfig
from pretraining.byte_diffusion.inference import (
    EntropyPatchedCanvasGenerator,
    _variable_prefix_metadata,
    denoise_entropy_blt_reference,
    entropy_next_byte_starts_patch,
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
def test_entropy_generator_ar_aligns_then_commits_the_whole_block(
    monkeypatch: pytest.MonkeyPatch,
    block_length: int,
) -> None:
    patcher = _patcher()
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    boundary = _arbitrary_boundary(patcher)
    aligned_prefix = torch.zeros(boundary, dtype=torch.long)
    assert entropy_next_byte_starts_patch(patcher, aligned_prefix)
    seen_prefixes: list[torch.Tensor] = []

    def denoise(
        _model: ByteDiffusionModel,
        _patcher: CausalEntropyPatcher,
        committed: torch.Tensor,
        noisy: torch.Tensor,
    ) -> torch.Tensor:
        seen_prefixes.append(committed.clone())
        assert not committed.eq(model.config.vocab.mask_id).any()
        assert noisy.eq(model.config.vocab.mask_id).all()
        assert noisy.shape == (block_length,)
        return torch.zeros((block_length, model.config.vocab.output_size))

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
        inference_module, "denoise_entropy_blt_reference", denoise
    )
    monkeypatch.setattr(
        inference_module, "sample_absorbing_canvas_batched", sample
    )
    generator = EntropyPatchedCanvasGenerator(
        model, patcher, block_length=block_length, seed=7
    )
    generator.prefill(aligned_prefix)
    generated = generator.generate_blt(steps=4, stochastic=False)

    combined = torch.cat((aligned_prefix, candidate[0]))
    starts = patcher.patch(
        combined.numpy(), torch.zeros(combined.numel(), dtype=torch.long).numpy()
    ).layout.patch_starts
    assert sum(start > boundary for start in starts) >= block_length // 4
    torch.testing.assert_close(generated.committed_canvas_ids, candidate[0])
    assert not generated.overflow_canvas_ids.numel()
    torch.testing.assert_close(generator.ids, combined)
    assert generator.rejected_bytes == 0
    assert len(seen_prefixes) == 1
    torch.testing.assert_close(seen_prefixes[0], aligned_prefix)


def test_entropy_generator_uses_clean_ar_until_the_next_causal_boundary(
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
    prefix = torch.zeros(prefix_length, dtype=torch.long)
    observed: list[torch.Tensor] = []

    def ar_logits(
        _model: ByteDiffusionModel,
        _patcher: CausalEntropyPatcher,
        committed: torch.Tensor,
    ) -> torch.Tensor:
        observed.append(committed.clone())
        assert not committed.eq(model.config.vocab.mask_id).any()
        logits = torch.full((model.config.vocab.output_size,), -20.0)
        logits[65] = 20.0
        return logits

    monkeypatch.setattr(inference_module, "entropy_ar_next_logits", ar_logits)
    generator = EntropyPatchedCanvasGenerator(
        model, patcher, block_length=4, seed=11
    )
    generator.prefill(prefix)
    aligned = generator.align_prefix_ar(stochastic=False)

    assert 1 <= aligned.numel() < patcher.config.max_patch_size
    assert aligned.tolist() == [65] * aligned.numel()
    assert entropy_next_byte_starts_patch(patcher, generator.ids)
    torch.testing.assert_close(observed[0], prefix)
    assert all(not item.eq(model.config.vocab.mask_id).any() for item in observed)


def test_entropy_eval_stops_alignment_at_the_literal_byte_cap(
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

    def ar_logits(*_: object) -> torch.Tensor:
        logits = torch.full((model.config.vocab.output_size,), -20.0)
        logits[65] = 20.0
        return logits

    monkeypatch.setattr(inference_module, "entropy_ar_next_logits", ar_logits)
    result = entropy_blt_generate_bytes(
        model,
        patcher,
        [bytes(prefix_length)],
        max_new_bytes=1,
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
    )[0]

    assert result.raw == b"A"
    assert result.termination == "byte_cap"
    assert result.native_actions == 1


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
