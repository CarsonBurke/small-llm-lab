from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pretraining.byte_diffusion.config import AtomicVocabulary, ByteDiffusionConfig
from pretraining.byte_diffusion.duo_model import DuoModel
from pretraining.byte_diffusion.export import (
    MAGIC,
    MAX_ENTROPY_PATCHER_RAW_BYTES,
    build_artifact,
    duo_serving_contract,
    load_embedded_entropy_patcher,
    parse_artifact,
)
from pretraining.byte_diffusion.patching import (
    CausalEntropyPatcher,
    EntropyPatchConfig,
    HashedNgramEntropyConfig,
    HashedNgramEntropyModel,
)
from pretraining.eval_byte_duo_gsm8k import (
    EVALUATOR_SCHEMA,
    evaluator_source_provenance,
    load_duo_artifact,
)
from pretraining.byte_diffusion.data import AtomicIdManifest
from pretraining.byte_diffusion.serving_duo import duo_generate_bytes_batched
from pretraining.gsm8k_contract import (
    PROMPT_FORMATS,
    build_prompt,
    extract_gold,
    extract_prediction,
)
from scripts.export_byte_duo import (
    cached_inference_smoke,
    checkpoint_entropy_patcher_artifact,
    prepare_validation_model,
    validation_compile_contract,
)


def _patcher_bytes() -> bytes:
    model = HashedNgramEntropyModel(
        HashedNgramEntropyConfig(vocab_size=261, table_size=8),
        np.arange(8 * 261, dtype=np.uint32).reshape(8, 261),
    )
    return CausalEntropyPatcher(
        model,
        EntropyPatchConfig(threshold=3.0, max_patch_size=8),
    ).to_bytes()


def _entropy_config() -> ByteDiffusionConfig:
    return ByteDiffusionConfig.tiny(
        duo_mutable_topology="full_resolution_decoder",
        duo_random_phase_training=True,
        duo_noisy_ngrams=False,
        duo_clean_patching="causal_entropy_v1",
    )


def _rewrite_metadata(artifact: bytes, mutate) -> bytes:
    metadata_size = struct.unpack_from("<Q", artifact, len(MAGIC))[0]
    start = len(MAGIC) + 8
    stop = start + metadata_size
    metadata = json.loads(artifact[start:stop])
    mutate(metadata)
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return MAGIC + struct.pack("<Q", len(encoded)) + encoded + artifact[stop:]


def test_duo_artifact_roundtrips_schedule_policy_and_compressed_patcher() -> None:
    raw = _patcher_bytes()
    artifact = build_artifact(
        DuoModel(_entropy_config(), schedule_eps=0.017),
        _entropy_config(),
        entropy_patcher=raw,
    )
    metadata, _ = parse_artifact(artifact)
    serving = duo_serving_contract(metadata)
    patcher = metadata["entropy_patcher"]

    assert serving == {
        "schema": "byte_duo_serving/v1",
        "schedule_eps": 0.017,
        "patching_policy": "causal_entropy_v1",
        "entropy_patcher_sha256": hashlib.sha256(raw).hexdigest(),
        "entropy_patcher_max_patch_size": 8,
    }
    assert patcher["encoding"] == "zlib"
    assert patcher["raw_bytes"] == len(raw)
    assert patcher["compressed_bytes"] < patcher["raw_bytes"]
    assert patcher["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    restored = load_embedded_entropy_patcher(artifact)
    assert restored is not None
    assert restored.to_bytes() == raw


def test_duo_export_recovers_only_checkpoint_pinned_dataset_patcher(
    tmp_path: Path,
) -> None:
    raw = _patcher_bytes()
    patcher = CausalEntropyPatcher.from_bytes(raw)
    patcher_path = tmp_path / "patcher.bdpatch"
    patcher_path.write_bytes(raw)
    manifest = {
        "patching": {
            "name": "causal_entropy_v1",
            "patcher_artifact": {
                "path": patcher_path.name,
                "sha256": patcher.sha256,
            },
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    payload = {
        "model_config": _entropy_config().to_dict(),
        "training": {
            "dataset_patching": {
                "name": "causal_entropy_v1",
                "max_patch_size": patcher.config.max_patch_size,
                "patcher_sha256": patcher.sha256,
            }
        },
    }

    assert checkpoint_entropy_patcher_artifact(payload, tmp_path) == raw
    payload["training"]["dataset_patching"]["patcher_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="authenticated provenance"):
        checkpoint_entropy_patcher_artifact(payload, tmp_path)


def test_artifact_only_loader_reconstructs_schedule_policy_and_weights(
    tmp_path: Path,
) -> None:
    config = ByteDiffusionConfig.tiny()
    source = DuoModel(config, schedule_eps=0.023)
    artifact = build_artifact(
        source,
        config,
        atomic_manifest=AtomicIdManifest.reference(),
    )
    path = tmp_path / "duo.bdg"
    path.write_bytes(artifact)

    restored, metadata, patcher = load_duo_artifact(path, torch.device("cpu"))

    assert restored.schedule_eps == 0.023
    assert restored.config == config
    assert patcher is None
    assert duo_serving_contract(metadata)["patching_policy"] == "fixed_stride_v1"
    for name, parameter in restored.named_parameters():
        assert torch.isfinite(parameter).all(), name


def test_artifact_only_loader_rejects_model_manifest_eot_disagreement(
    tmp_path: Path,
) -> None:
    config = ByteDiffusionConfig.tiny(
        vocab=AtomicVocabulary(eot_id=260), duo_diffusion_atoms=261
    )
    artifact = build_artifact(
        DuoModel(config),
        config,
        atomic_manifest=AtomicIdManifest.reference(),
    )
    path = tmp_path / "bad-vocab.bdg"
    path.write_bytes(artifact)

    with pytest.raises(ValueError, match="config and atomic manifest disagree"):
        load_duo_artifact(path, torch.device("cpu"))


def test_export_smoke_exercises_authenticated_entropy_topology() -> None:
    patcher = CausalEntropyPatcher.from_bytes(_patcher_bytes())
    result = cached_inference_smoke(DuoModel(_entropy_config()).eval(), patcher)

    assert result["finite"] is True
    assert result["patching_policy"] == "causal_entropy_v1"
    assert result["entropy_patcher_sha256"] == patcher.sha256
    assert result["branch_start"] == result["clean_atoms"]
    assert result["origin_phase"] == 1
    assert result["full_vs_cached_max_abs_error"] < 1e-5


@pytest.mark.parametrize(
    ("config", "dynamic_shapes", "reason"),
    (
        (
            ByteDiffusionConfig.tiny(),
            False,
            "fixed_stride_patch_counts",
        ),
        (
            _entropy_config(),
            True,
            "ragged_causal_entropy_patch_counts",
        ),
    ),
)
def test_validation_compile_contract_tracks_patch_count_geometry(
    config: ByteDiffusionConfig,
    dynamic_shapes: bool,
    reason: str,
) -> None:
    contract = validation_compile_contract(config, torch.device("cuda"))

    assert contract == {
        "compiled": True,
        "dynamic_shapes": dynamic_shapes,
        "patching_policy": config.duo_clean_patching,
        "shape_reason": reason,
    }


def test_entropy_validation_compiles_with_dynamic_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class FakeModel:
        config = _entropy_config()

        def forward(self):
            return None

        def to(self, device: torch.device):
            observed["device"] = device
            return self

        def eval(self):
            observed["eval"] = True
            return self

    def fake_compile(function, **kwargs):
        observed["function"] = function
        observed.update(kwargs)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = FakeModel()

    assert prepare_validation_model(model, torch.device("cuda")) is model  # type: ignore[arg-type]
    assert observed["device"] == torch.device("cuda")
    assert observed["eval"] is True
    assert observed["dynamic"] is True
    assert observed["fullgraph"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda metadata: metadata["entropy_patcher"].__setitem__(
                "raw_bytes", MAX_ENTROPY_PATCHER_RAW_BYTES + 1
            ),
            "raw size exceeds",
        ),
        (
            lambda metadata: metadata["entropy_patcher"].__setitem__(
                "compressed_bytes",
                metadata["entropy_patcher"]["compressed_bytes"] + 1,
            ),
            "compressed size mismatch",
        ),
        (
            lambda metadata: metadata["duo_serving"].__setitem__(
                "schedule_eps", 0.0
            ),
            "schedule_eps",
        ),
        (
            lambda metadata: metadata["duo_serving"].__setitem__(
                "entropy_patcher_sha256", "0" * 64
            ),
            "patcher sha256 mismatch",
        ),
    ),
)
def test_duo_artifact_rejects_corrupt_serving_metadata(mutation, message: str) -> None:
    artifact = build_artifact(
        DuoModel(_entropy_config()),
        _entropy_config(),
        entropy_patcher=_patcher_bytes(),
    )
    with pytest.raises(ValueError, match=message):
        parse_artifact(_rewrite_metadata(artifact, mutation))


def test_duo_artifact_rejects_trailing_compressed_stream_data() -> None:
    artifact = build_artifact(
        DuoModel(_entropy_config()),
        _entropy_config(),
        entropy_patcher=_patcher_bytes(),
    )

    def add_claimed_trailer(metadata: dict[str, object]) -> None:
        patcher = metadata["entropy_patcher"]
        patcher["compressed_bytes"] += 1
        patcher["payload_bytes"] += 1

    trailed = _rewrite_metadata(artifact, add_claimed_trailer) + b"x"
    with pytest.raises(ValueError, match="trailing compressed data"):
        parse_artifact(trailed)


def test_duo_only_evaluator_closure_is_genuine_and_excludes_other_serving_stacks() -> None:
    provenance = evaluator_source_provenance()
    files = tuple(str(path) for path in provenance["files"])
    forbidden = (
        "eval_byte_diffusion_gsm8k.py",
        "eval_fewshot_gsm8k.py",
        "idlm",
        "postraining/",
        "train_gpt.py",
        "train_byte_diffusion_gemma.py",
    )

    assert provenance["bytes"] == sum((Path.cwd() / path).stat().st_size for path in files)
    assert int(provenance["bytes"]) < 1_000_000
    assert not [path for path in files if any(token in path for token in forbidden)]


def test_shared_prompt_parser_and_versioned_evaluator_contracts() -> None:
    import pretraining.eval_byte_diffusion_gsm8k as generic

    fmt = PROMPT_FORMATS["harness"]
    exemplars = [
        {
            "question": "What is 2 + 3?",
            "answer": "Add them. #### 5",
        }
    ]
    prompt = build_prompt(
        exemplars,
        "What is 3 + 4?",
        fmt,
        strip_calculator=True,
    )

    assert generic.EVALUATOR_SCHEMA == "byte_diffusion_gsm8k/v10"
    assert EVALUATOR_SCHEMA == "byte_diffusion_gsm8k/v9"
    assert generic.build_prompt(exemplars, "What is 3 + 4?", fmt, strip_calculator=True) == prompt
    assert generic.extract_gold("work #### 1,024.0") == extract_gold("work #### 1,024.0")
    assert generic.extract_prediction("work #### 7", fmt) == extract_prediction("work #### 7", fmt)
    assert asdict(fmt) == {
        "name": "harness",
        "delimiter": "####",
        "stops": ("\nQuestion:", "\nAnswer:", "\n\n"),
    }


def test_duo_only_generation_fails_closed_without_policy_matched_patcher() -> None:
    model = DuoModel(_entropy_config()).eval()

    with pytest.raises(ValueError, match="authenticated patcher"):
        duo_generate_bytes_batched(
            model,
            [b"Q"],
            max_new_bytes=4,
            max_new_atoms=4,
            max_native_actions=3,
            context_bytes=16,
            stops=(),
            canvas_length=4,
            diffusion_steps=1,
            seed=1,
            device=torch.device("cpu"),
        )


def test_duo_only_generation_uses_artifact_schedule_not_sampler_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pretraining.byte_diffusion.serving_duo as serving

    observed: dict[str, float] = {}

    def fake_generate(model, ids, valid, **kwargs):
        observed["schedule_eps"] = kwargs["schedule"].eps
        batch = ids.shape[0]
        return SimpleNamespace(
            ids=ids,
            generated_valid=torch.zeros_like(valid),
            text_stopped=torch.zeros(batch, dtype=torch.bool),
            byte_capped=torch.zeros(batch, dtype=torch.bool),
            denoising_actions=torch.full((batch,), 2, dtype=torch.long),
            clean_bank_actions=torch.ones(batch, dtype=torch.long),
            model_forwards=3,
            trajectories=None,
            trajectory_rows=None,
            trajectory_starts=None,
            trajectory_active=None,
            lengths=valid.sum(1),
        )

    monkeypatch.setattr(serving, "generate_duo_continuation", fake_generate)
    model = DuoModel(ByteDiffusionConfig.tiny(), schedule_eps=0.017).eval()
    serving.duo_generate_bytes_batched(
        model,
        [b"Q"],
        max_new_bytes=4,
        max_new_atoms=4,
        max_native_actions=3,
        context_bytes=16,
        stops=(),
        canvas_length=4,
        diffusion_steps=1,
        seed=1,
        device=torch.device("cpu"),
    )

    assert observed["schedule_eps"] == 0.017
