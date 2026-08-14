from types import MethodType, SimpleNamespace

import pytest
import torch

import pretraining.eval_byte_diffusion_gsm8k as gsm8k_eval
from pretraining.byte_diffusion.config import ByteDiffusionConfig
from pretraining.byte_diffusion.data import AtomicIdManifest
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.duo_model import DuoModel
from pretraining.byte_diffusion.diffusion_gemma_model import DiffusionGemmaModel
from pretraining.byte_diffusion.idlm_model import IDLMModel, IDLMModelConfig
from pretraining.byte_diffusion.sampling import BatchedCanvasSample
from pretraining.byte_diffusion.training import CHECKPOINT_SCHEMA
from pretraining.eval_byte_diffusion_gsm8k import (
    _decode_atomic_continuation,
    _duo_required_canvases,
    _duo_warmup_prompt_groups,
    blt_generate_bytes_batched,
    diffusion_gemma_generate_bytes_batched,
    duo_generate_bytes_batched,
    idlm_generate_bytes_batched,
    validate_serving_contract,
)
from scripts.eval_byte_diffusion import validate_evaluation_provenance


def _trainer(*, train_hash: str = "train-a", validation_hash: str = "val-a"):
    return SimpleNamespace(
        train_cursor=SimpleNamespace(dataset_sha256=train_hash),
        validation_sha256=validation_hash,
        source_provenance={"schema": "source/v1", "sha256": "source-a"},
    )


def test_duo_decode_rejects_control_inside_utf8_code_point() -> None:
    manifest = AtomicIdManifest.reference()
    atoms = (0xF0, 257, 0x9F, 0x99, 0x82)

    raw, text, invalid, saw_eot = _decode_atomic_continuation(atoms, manifest)

    assert raw == b"\xf0"
    assert text is None
    assert invalid
    assert not saw_eot


def test_duo_action_budget_covers_partial_first_patch() -> None:
    assert _duo_required_canvases(512, 512, 4) == 2
    assert _duo_required_canvases(509, 512, 4) == 1
    assert _duo_required_canvases(64, 512, 4, 32) == 2
    assert _duo_required_canvases(1020, 512, 4, 510) == 3
    assert _duo_required_canvases(1022, 512, 4, 511) == 3


@pytest.mark.parametrize(
    ("prompt_count", "batch_size", "expected_sizes"),
    [
        (7, 32, (7,)),
        (32, 32, (32,)),
        (33, 32, (32, 1)),
        (1_319, 32, (32, 7)),
    ],
)
def test_duo_warmup_covers_full_and_tail_batch_geometries(
    prompt_count: int, batch_size: int, expected_sizes: tuple[int, ...]
) -> None:
    prompts = tuple(str(index).encode() for index in range(prompt_count))

    groups = _duo_warmup_prompt_groups(prompts, batch_size)

    assert tuple(map(len, groups)) == expected_sizes
    assert groups[0] == prompts[:batch_size]
    assert groups[-1] == prompts[-expected_sizes[-1] :]


def _payload() -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "dataset_sha256": "train-a",
        "validation_sha256": "val-a",
        "dataset_provenance": {
            "payload_sha256": "payload-a",
            "source_manifests": [{"sha256": "source-a"}],
        },
        "source_provenance": {"schema": "source/v1", "sha256": "source-a"},
    }


def test_proxy_evaluation_requires_exact_train_validation_and_payload() -> None:
    payload = _payload()
    provenance = payload["dataset_provenance"]
    validate_evaluation_provenance(
        payload, _trainer(), provenance, full_validation=False
    )
    with pytest.raises(ValueError, match="proxy validation"):
        validate_evaluation_provenance(
            payload,
            _trainer(validation_hash="val-b"),
            provenance,
            full_validation=False,
        )
    with pytest.raises(ValueError, match="training split"):
        validate_evaluation_provenance(
            payload,
            _trainer(train_hash="train-b"),
            provenance,
            full_validation=False,
        )
    with pytest.raises(ValueError, match="payload/source"):
        validate_evaluation_provenance(
            payload,
            _trainer(),
            {"payload_sha256": "payload-b", "source_manifests": []},
            full_validation=False,
        )
    payload["source_provenance"]["sha256"] = "source-b"
    with pytest.raises(ValueError, match="pinned training source"):
        validate_evaluation_provenance(
            payload, _trainer(), provenance, full_validation=False
        )


def test_full_evaluation_uses_bound_source_identity_not_proxy_hash() -> None:
    payload = _payload()
    validate_evaluation_provenance(
        payload,
        _trainer(validation_hash="full-validation"),
        payload["dataset_provenance"],
        full_validation=True,
    )
    payload["schema"] = "byte_diffusion_training/obsolete"
    with pytest.raises(ValueError, match="schema"):
        validate_evaluation_provenance(
            payload,
            _trainer(validation_hash="full-validation"),
            payload["dataset_provenance"],
            full_validation=True,
        )


def test_batched_blt_generation_handles_mixed_prompt_alignment_on_cpu() -> None:
    torch.manual_seed(7)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    generated = blt_generate_bytes_batched(
        model,
        [b"abcd", b"seven!!", b"twelve bytes"],
        max_new_bytes=8,
        max_native_actions=20,
        context_bytes=64,
        stops=("\n\n",),
        block_length=4,
        diffusion_steps=4,
        unmasking_strategy="confidence",
        confidence_threshold=0.7,
        entropy_budget=1.0,
        seed=31,
        stochastic=False,
        device=torch.device("cpu"),
    )
    assert len(generated) == 3
    assert len({item.model_forwards for item in generated}) == 1
    assert all(item.native_actions <= 20 for item in generated)
    assert all(len(item.raw) <= 8 for item in generated)


def test_idlm_gsm_adapter_uses_exact_fused_native_sampler() -> None:
    model = IDLMModel(IDLMModelConfig.tiny(block_size=4)).eval()
    generated = idlm_generate_bytes_batched(
        model,
        [b"abcd", b"seven"],
        max_new_bytes=2,
        max_native_actions=32,
        context_bytes=32,
        stops=("\n\n",),
        seed=37,
        device=torch.device("cpu"),
    )
    assert len(generated) == 2
    assert all(item.generated_atoms <= 2 for item in generated)
    assert all(item.model_forwards <= 32 for item in generated)


def test_diffusion_gemma_gsm_adapter_only_noises_fresh_canvas_suffix() -> None:
    model = DiffusionGemmaModel(ByteDiffusionConfig.tiny()).eval()
    traces = [{"alignment_steps": [], "diffusion_blocks": []}]
    generated = diffusion_gemma_generate_bytes_batched(
        model,
        [b"abc"],
        max_new_bytes=1,
        max_new_atoms=1,
        max_native_actions=4,
        context_bytes=16,
        stops=("\n\n",),
        canvas_length=4,
        diffusion_steps=1,
        seed=41,
        device=torch.device("cpu"),
        trace_records=traces,
    )
    assert len(generated) == 1
    assert generated[0].generated_atoms == 1
    block = traces[0]["diffusion_blocks"][0]
    assert block["prompt_atoms_noised"] is False
    assert len(block["steps"]) == 1


def test_batched_duo_generation_keeps_clean_prompts_and_traces_every_nfe() -> None:
    model = DuoModel(ByteDiffusionConfig.tiny()).eval()
    traces = [
        {"alignment_steps": [], "diffusion_blocks": []},
        {"alignment_steps": [], "diffusion_blocks": []},
    ]
    generated = duo_generate_bytes_batched(
        model,
        [b"abc", b"hello"],
        max_new_bytes=4,
        max_new_atoms=4,
        max_native_actions=4,
        context_bytes=32,
        stops=("\n\n",),
        canvas_length=8,
        diffusion_steps=1,
        seed=37,
        device=torch.device("cpu"),
        trace_records=traces,
    )
    assert len(generated) == 2
    assert all(item.native_actions == 3 for item in generated)
    assert all(item.model_forwards == 3 for item in generated)
    assert all(len(item.raw) <= 4 for item in generated)
    assert all(len(trace["diffusion_blocks"]) == 1 for trace in traces)
    for trace in traces:
        block = trace["diffusion_blocks"][0]
        # One grid transition plus the exact final noise-removal posterior.
        assert len(block["steps"]) == 2
        assert block["prompt_atoms_noised"] is False
        assert block["schedule_eps"] == model.schedule_eps
        assert block["sampling_terminal_eps"] == 1e-5
        assert block["posterior_precision"] == "float32"
        assert block["steps"][-1]["transition_kind"] == "exact_residual_noise_cleanup"
        assert block["steps"][-1]["time_s"] == 0.0
        assert block["steps"][-1]["alpha_s"] == 1.0
        assert block["termination_after_block"] != "continue"
        assert block["next_canvas_transition"] is None
        assert block["committed_ids"] == tuple(
            block["steps"][-1]["output_ids"][position]
            for position in block["commit_positions"]
        )
        assert block["committed_byte_values"] == tuple(
            value for value in block["committed_ids"] if value < 256
        )
        assert block["generated_bytes_before"] == 0
        assert block["generated_bytes_after"] == len(
            block["committed_byte_values"]
        )
        assert trace["returned_byte_count_after_stop_trim"] == len(
            trace["returned_byte_values_after_stop_trim"]
        )
    assert traces[0]["diffusion_blocks"][0]["revisable_positions"] == [3, 4, 5, 6]
    assert traces[1]["diffusion_blocks"][0]["revisable_positions"] == [1, 2, 3, 4]


def test_batched_duo_generation_rejects_unaligned_explicit_work_width() -> None:
    model = DuoModel(ByteDiffusionConfig.tiny()).eval()
    with pytest.raises(ValueError, match="patch aligned"):
        duo_generate_bytes_batched(
            model,
            [b"abc"],
            max_new_bytes=1,
            max_new_atoms=1,
            max_native_actions=4,
            context_bytes=16,
            stops=("\n\n",),
            canvas_length=8,
            diffusion_steps=1,
            seed=37,
            device=torch.device("cpu"),
            work_width=5,
        )


def _force_alignment_byte(model: ByteDiffusionModel, byte_id: int = 65) -> None:
    def forward_ar_varlen(
        self: ByteDiffusionModel,
        ids: torch.Tensor,
        valid: torch.Tensor,
        **_: object,
    ) -> SimpleNamespace:
        logits = torch.full(
            (int(valid.sum()), self.config.vocab.output_size),
            -20.0,
            device=ids.device,
        )
        logits[:, byte_id] = 20.0
        return SimpleNamespace(logits=logits)

    model.forward_ar_varlen = MethodType(forward_ar_varlen, model)


def _fixed_byte_canvas(
    initial_ids: torch.Tensor,
    _denoise: object,
    *,
    steps: int,
    row_active: torch.Tensor,
    **_: object,
) -> BatchedCanvasSample:
    ids = torch.where(
        row_active[:, None],
        torch.full_like(initial_ids, 65),
        initial_ids,
    )
    return BatchedCanvasSample(
        ids=ids,
        active=torch.ones_like(ids, dtype=torch.bool) & row_active[:, None],
        useful_nfe=row_active.to(torch.int64) * steps,
        executed_nfe=steps,
    )


def test_batched_blt_action_cap_retires_only_the_exhausted_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    _force_alignment_byte(model)
    monkeypatch.setattr(
        gsm8k_eval, "sample_absorbing_canvas_batched", _fixed_byte_canvas
    )
    generated = blt_generate_bytes_batched(
        model,
        [b"abcd", b"abcde"],
        max_new_bytes=4,
        max_native_actions=6,
        context_bytes=32,
        stops=("\n\n",),
        block_length=4,
        diffusion_steps=4,
        unmasking_strategy="confidence",
        confidence_threshold=0.7,
        entropy_budget=1.0,
        seed=31,
        stochastic=False,
        device=torch.device("cpu"),
    )

    assert generated[0].raw == b"AAAA"
    assert generated[0].termination == "byte_cap"
    assert generated[0].native_actions == 6
    assert generated[1].raw == b"AAA"
    assert generated[1].termination == "native_action_cap"
    # Three alignment forwards plus the cache prefill are all charged.
    assert generated[1].native_actions == 4


def test_batched_blt_context_cap_retires_only_the_overflowing_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    monkeypatch.setattr(
        gsm8k_eval, "sample_absorbing_canvas_batched", _fixed_byte_canvas
    )
    generated = blt_generate_bytes_batched(
        model,
        [b"abcdefgh", b"abcd"],
        max_new_bytes=8,
        max_native_actions=11,
        context_bytes=12,
        stops=("\n\n",),
        block_length=4,
        diffusion_steps=4,
        unmasking_strategy="confidence",
        confidence_threshold=0.7,
        entropy_budget=1.0,
        seed=31,
        stochastic=False,
        device=torch.device("cpu"),
    )

    assert generated[0].raw == b"AAAA"
    assert generated[0].termination == "context_cap"
    assert generated[0].native_actions == 6
    assert generated[1].raw == b"AAAAAAAA"
    assert generated[1].termination == "byte_cap"
    assert generated[1].native_actions == 11


def test_batched_blt_action_cap_charges_prefill_when_no_canvas_fits() -> None:
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    generated = blt_generate_bytes_batched(
        model,
        [b"abcd"],
        max_new_bytes=4,
        max_native_actions=1,
        context_bytes=32,
        stops=("\n\n",),
        block_length=4,
        diffusion_steps=4,
        unmasking_strategy="confidence",
        confidence_threshold=0.7,
        entropy_budget=1.0,
        seed=31,
        stochastic=False,
        device=torch.device("cpu"),
    )

    assert generated[0].raw == b""
    assert generated[0].termination == "native_action_cap"
    assert generated[0].native_actions == 1
    assert generated[0].model_forwards == 1


@pytest.mark.parametrize(
    ("max_new_bytes", "stops", "expected_raw", "expected_termination", "actions"),
    [
        (1, ("\n\n",), b"A", "byte_cap", 1),
        (4, ("AA",), b"", "text_stop", 2),
    ],
)
def test_batched_blt_alignment_obeys_semantic_stops(
    max_new_bytes: int,
    stops: tuple[str, ...],
    expected_raw: bytes,
    expected_termination: str,
    actions: int,
) -> None:
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    _force_alignment_byte(model)
    generated = blt_generate_bytes_batched(
        model,
        [b"abcde"],
        max_new_bytes=max_new_bytes,
        max_native_actions=10,
        context_bytes=32,
        stops=stops,
        block_length=4,
        diffusion_steps=4,
        unmasking_strategy="confidence",
        confidence_threshold=0.7,
        entropy_budget=1.0,
        seed=31,
        stochastic=False,
        device=torch.device("cpu"),
    )

    assert generated[0].raw == expected_raw
    assert generated[0].termination == expected_termination
    assert generated[0].native_actions == actions
    assert generated[0].model_forwards == actions


def test_serving_contract_rejects_untrained_blt_geometry_and_entropy_hack() -> None:
    payload = {
        "run_contract": {
            "patching_policy": "fixed_stride_v1",
            "corruption": {"canvas_length": 16},
        }
    }
    validate_serving_contract(payload, decode_mode="blt", block_length=16)
    with pytest.raises(ValueError, match="trained canvas"):
        validate_serving_contract(payload, decode_mode="blt", block_length=8)
    payload["run_contract"]["patching_policy"] = "causal_entropy_v1"
    with pytest.raises(ValueError, match="runtime entropy patcher"):
        validate_serving_contract(payload, decode_mode="blt", block_length=16)
