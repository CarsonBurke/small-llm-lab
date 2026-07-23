"""Pinned-EMIT (cot/none) reasoning-mode invariants across the VAPO stack."""

from __future__ import annotations

import os

import pytest
import sentencepiece as spm
import torch

from fresh_lejepa_train import FreshHyperparameters

from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    TOKEN_SLOT,
    LatentRolloutBatch,
    assemble_stream_latents,
    pack_rollout_groups_for_replay,
    refresh_old_statistics,
    rollout_continuations,
    select_trajectory_rows,
    trim_stream,
)
from postraining.latent_thought import (
    EMIT,
    PINNED_EMIT_ROLLOUT_POLICY_SCHEMAS,
    RENDERER_FEATURES_SCHEMA,
    ROLLOUT_POLICY_SCHEMA,
    THOUGHT_DISTRIBUTION_SCHEMA,
    THOUGHT_INPUT_SCHEMA,
    THOUGHT_MEAN_SCHEMA,
    LatentThoughtModel,
    rollout_policy_schema_for_mode,
    validate_renderer_checkpoint,
)
from postraining.core import GPT2BPETokenizer, load_posttraining_tokenizer
from postraining.nano_backbone import NanoGPTBackbone
from postraining.train_latent_vapo import (
    answer_prefix_token_ids,
    build_optimizers,
    evaluate_aime_latent,
    renderer_parameters,
    rollout_diagnostics,
    score_math_rollout,
    update_minibatch,
)
from postraining.value_model import SeparateCritic

from postraining.tests.test_latent_rollout import _wrapper


def _pinned_rollout(wrapper, **overrides):
    prompt_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    kwargs = dict(
        max_new_tokens=4,
        max_stream_steps=4,
        temperature=1.0,
        top_p=1.0,
        pin_emit=True,
    )
    kwargs.update(overrides)
    return rollout_continuations(wrapper, prompt_ids, **kwargs)


def test_pin_emit_rollout_invariants():
    wrapper = _wrapper()
    torch.manual_seed(11)
    batch = _pinned_rollout(wrapper)
    assert not bool((batch.kind == THOUGHT_SLOT).any())
    # Zero-width thought storage even with replay_storage=True (the default):
    # the pinned policy has no thought action to replay.
    assert batch.thoughts.size(-1) == 0
    assert batch.old_thought_logprobs.size(-1) == 0
    assert float(batch.gate_mask.sum()) == 0.0
    assert torch.equal(batch.emit_mask, batch.action_mask)
    action = batch.action_mask.bool()
    assert bool((batch.gate_actions[action] == EMIT).all())
    assert float(batch.old_gate_logprobs.abs().sum()) == 0.0


def test_pin_emit_rejects_forced_think():
    wrapper = _wrapper()
    with pytest.raises(ValueError, match="pin_emit"):
        _pinned_rollout(
            wrapper, force_initial_think=torch.tensor([True, False])
        )


def test_pin_emit_nano_backbone_rollout():
    torch.manual_seed(7)
    backbone = NanoGPTBackbone(
        vocab_size=64, num_layers=2, model_dim=256
    ).float().eval()
    wrapper = LatentThoughtModel(backbone).eval()
    with torch.no_grad():
        batch = _pinned_rollout(wrapper)
    assert not bool((batch.kind == THOUGHT_SLOT).any())
    assert batch.thoughts.size(-1) == 0
    assert float(batch.action_mask.sum()) >= 2.0


def test_pin_emit_stream_latents_skip_adapter():
    wrapper = _wrapper()
    torch.manual_seed(13)
    batch = trim_stream(_pinned_rollout(wrapper))
    inputs = assemble_stream_latents(wrapper, batch)
    reference = wrapper.embed_tokens(batch.token_ids) * (
        batch.kind != PAD_SLOT
    )[..., None].float()
    torch.testing.assert_close(inputs, reference)


def test_pin_emit_refresh_and_age0_update_clip_zero():
    wrapper = _wrapper()
    critic = _fresh_critic(wrapper)
    torch.manual_seed(17)
    batch = trim_stream(_pinned_rollout(wrapper))
    batch.reward_scalar.uniform_(0.0, 1.0)
    positions = (
        batch.action_mask.size(1) - 1 - batch.action_mask.flip(1).argmax(1)
    ).long()
    batch.rewards.zero_()
    batch.rewards[torch.arange(batch.rewards.size(0)), positions] = (
        batch.reward_scalar
    )
    refresh_old_statistics(wrapper, critic, batch)
    assert batch.old_thought_logprobs.size(-1) == 0
    optimizers = build_optimizers(wrapper, critic, 1e-4, fused=False)
    metrics = update_minibatch(wrapper, critic, batch, optimizers)
    assert metrics["policy_clip_fraction"] == 0.0
    assert metrics["gate_action_count"] == 0.0
    assert metrics["thought_action_count"] == 0.0
    # Every reported scalar stays finite despite the empty gate/thought sets.
    non_finite = {
        key: value
        for key, value in metrics.items()
        if not torch.isfinite(torch.tensor(float(value)))
    }
    assert not non_finite


def _fresh_critic(wrapper):
    return SeparateCritic(
        type(wrapper.backbone)(**wrapper.backbone.init_kwargs)
        if hasattr(wrapper.backbone, "init_kwargs")
        else _wrapper(seed=5).backbone,
        num_bins=17,
        prior_value=0.1,
    )


def test_update_minibatch_rejects_unrefreshed_batches_in_both_modes():
    wrapper = _wrapper()
    critic = _fresh_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, 1e-4, fused=False)
    for pin_emit in (True, False):
        torch.manual_seed(23)
        batch = trim_stream(_pinned_rollout(wrapper, pin_emit=pin_emit))
        assert batch.statistics_refreshed is False
        with pytest.raises(RuntimeError, match="requires refresh"):
            update_minibatch(wrapper, critic, batch, optimizers)


def test_statistics_refreshed_flag_survives_batch_helpers():
    wrapper = _wrapper()
    critic = _fresh_critic(wrapper)
    torch.manual_seed(29)
    batch = trim_stream(_pinned_rollout(wrapper))
    refresh_old_statistics(wrapper, critic, batch)
    assert batch.statistics_refreshed is True
    rows = torch.tensor([0, 1])
    selected = select_trajectory_rows(batch, rows, batch.stream_length)
    assert selected.statistics_refreshed is True
    assert trim_stream(selected).statistics_refreshed is True
    packed = pack_rollout_groups_for_replay([selected, selected])
    assert packed.statistics_refreshed is True
    torch.manual_seed(31)
    unrefreshed = trim_stream(_pinned_rollout(wrapper))
    mixed = pack_rollout_groups_for_replay([selected, unrefreshed])
    assert mixed.statistics_refreshed is False


class _AnswerTokenizer:
    """id 1 -> "\nAnswer:", id 2 -> " 42", id 5 -> EOS."""

    fragments = {0: "", 1: "\nAnswer:", 2: " 42", 5: ""}

    def decode(self, ids):
        return "".join(self.fragments[int(i)] for i in ids)


def test_score_math_rollout_prepends_answer_prefix():
    stream = 4
    kind = torch.tensor([[TOKEN_SLOT, TOKEN_SLOT, TOKEN_SLOT, TOKEN_SLOT]])
    token_ids = torch.tensor([[0, 0, 2, 5]])
    zeros = torch.zeros(1, stream)
    batch = LatentRolloutBatch(
        kind=kind,
        token_ids=token_ids,
        thoughts=torch.zeros(1, stream, 0),
        gate_actions=torch.full((1, stream), EMIT, dtype=torch.long),
        action_mask=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        gate_mask=zeros.clone(),
        emit_mask=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        old_gate_logprobs=zeros.clone(),
        old_token_logprobs=zeros.clone(),
        old_thought_logprobs=torch.zeros(1, stream, 0),
        old_values=zeros.clone(),
        rewards=zeros.clone(),
        reward_scalar=torch.zeros(1),
        prompt_length=2,
    )
    tokenizer = _AnswerTokenizer()
    # Without the prefix the continuation " 42" alone has no Answer: field.
    score_math_rollout(batch, "42", tokenizer, stop_ids=(5,))
    assert float(batch.reward_scalar) == 0.0
    score_math_rollout(
        batch, "42", tokenizer, stop_ids=(5,), solution_prefix_ids=(1,)
    )
    assert float(batch.reward_scalar) == 1.0


def test_rollout_policy_schema_for_mode():
    assert rollout_policy_schema_for_mode("latent") == ROLLOUT_POLICY_SCHEMA
    assert (
        rollout_policy_schema_for_mode("cot")
        == PINNED_EMIT_ROLLOUT_POLICY_SCHEMAS["cot"]
    )
    assert rollout_policy_schema_for_mode("cot") != (
        rollout_policy_schema_for_mode("none")
    )
    with pytest.raises(ValueError, match="unknown reasoning mode"):
        rollout_policy_schema_for_mode("verbose")


def test_validate_renderer_checkpoint_rejects_mode_mismatch():
    payload = {
        "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
        "rollout_policy_schema": rollout_policy_schema_for_mode("cot"),
        "thought_input_schema": THOUGHT_INPUT_SCHEMA,
        "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
        "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
    }
    validate_renderer_checkpoint(
        payload,
        "ckpt.pt",
        expected_rollout_policy_schema=rollout_policy_schema_for_mode("cot"),
    )
    with pytest.raises(ValueError, match="rollout"):
        validate_renderer_checkpoint(payload, "ckpt.pt")
    with pytest.raises(ValueError, match="rollout"):
        validate_renderer_checkpoint(
            payload,
            "ckpt.pt",
            expected_rollout_policy_schema=rollout_policy_schema_for_mode(
                "none"
            ),
        )


def test_rollout_diagnostics_pinned_reports_no_forced_or_think():
    wrapper = _wrapper()
    torch.manual_seed(19)
    batch = trim_stream(
        _pinned_rollout(
            wrapper,
            max_new_tokens=3,
            max_stream_steps=3,
        )
    )
    metrics = rollout_diagnostics(batch, samples_per_prompt=2)
    assert metrics["think_fraction"] == 0.0
    assert metrics["forced_initial_trajectory_fraction"] == 0.0
    assert metrics["forced_initial_thinks_per_trajectory"] == 0.0
    assert metrics["thoughts_per_trajectory"] == 0.0


def test_evaluate_latent_math_pin_emit_with_prompt_suffix(monkeypatch):
    wrapper = _wrapper()

    class _Tokenizer:
        def eos_id(self) -> int:
            return 5

        def bos_id(self) -> int:
            return -1

        def encode(self, text: str) -> list[int]:
            return list(range(1, len(text) + 1))

        def decode(self, ids: list[int]) -> str:
            return "Answer: 42"

    rows = [
        {
            "prompt": [{"content": text}],
            "reward_model": {"ground_truth": "42"},
        }
        for text in ("a", "abc")
    ]
    import postraining.latent_eval as evaluator

    seen = []
    original = evaluator.rollout_continuations

    def _spy(wrapper_arg, prompt_ids, *args, **kwargs):
        seen.append(
            (
                prompt_ids.clone(),
                kwargs["pin_emit"],
                kwargs["force_initial_think"].clone(),
            )
        )
        return original(wrapper_arg, prompt_ids, *args, **kwargs)

    monkeypatch.setattr(evaluator, "rollout_continuations", _spy)
    metrics = evaluate_aime_latent(
        wrapper, _Tokenizer(), rows, samples=2, max_new_tokens=2,
        max_stream_steps=2, chunk=2, seed=5, device=torch.device("cpu"),
        prompt_tokens=8, batch_trajectories=4,
        pin_emit=True, prompt_suffix_ids=(9, 9),
    )
    assert metrics["think_fraction"] == 0.0
    assert metrics["forced_initial_fraction"] == 0.0
    assert metrics["pin_emit"] is True
    assert len(seen) == 1
    prompt_ids, pin_emit, forced = seen[0]
    assert pin_emit is True
    assert not bool(forced.any())
    # The teacher-forced suffix terminates every truncated prompt.
    assert prompt_ids[:, -2:].tolist() == [[9, 9], [9, 9]]


@pytest.mark.skipif(
    not os.path.exists(FreshHyperparameters.tokenizer_path),
    reason="sp1024 tokenizer model not present",
)
def test_answer_prefix_token_ids_match_document_tokenization():
    tokenizer = spm.SentencePieceProcessor(
        model_file=FreshHyperparameters.tokenizer_path
    )
    prefix = answer_prefix_token_ids(tokenizer)
    # The naive standalone encoding drops the newline and splits the word;
    # the derived ids must instead match the in-document tokenization.
    assert prefix != tuple(tokenizer.encode("\nAnswer:"))
    question = "What is 7 times 8?"
    document = tokenizer.encode(question + "\nAnswer: 56")
    question_ids = tokenizer.encode(question)
    assert document[: len(question_ids)] == question_ids
    assert tuple(document[len(question_ids):][: len(prefix)]) == prefix
    # The score path decodes prefix + emitted; the parser needs "Answer:".
    assert "Answer:" in tokenizer.decode(list(prefix))


def test_load_posttraining_tokenizer_selects_gpt2_for_gpt2vocab_archs():
    pytest.importorskip("transformers")
    tokenizer = load_posttraining_tokenizer(
        "nanogpt_mini_gpt2vocab_v1", FreshHyperparameters.tokenizer_path
    )
    assert isinstance(tokenizer, GPT2BPETokenizer)
    # The single <|endoftext|> token reports as both EOS and BOS; the
    # trainer's dict.fromkeys dedupe must collapse it to one stop id.
    assert tokenizer.eos_id() == 50256
    assert tokenizer.bos_id() == 50256
    stop_ids = tuple(
        dict.fromkeys(
            t for t in (tokenizer.eos_id(), tokenizer.bos_id()) if t >= 0
        )
    )
    assert stop_ids == (50256,)
    text = "Solve 2+2.\nAnswer: 4"
    ids = tokenizer.encode(text)
    assert all(0 <= token < 50257 for token in ids)
    assert tokenizer.decode(ids) == text
    # The score path decodes emitted ids that may include the stop token.
    assert tokenizer.decode(ids + [50256]) == text


def test_answer_prefix_derivation_is_boundary_stable_under_gpt2():
    pytest.importorskip("transformers")
    tokenizer = GPT2BPETokenizer()
    prefix = answer_prefix_token_ids(tokenizer)
    # Byte-level BPE pre-splits on the regex boundary, so the in-context
    # derivation must agree with the standalone encoding (unlike sp1024).
    assert list(prefix) == tokenizer.encode("\nAnswer:")
    assert "Answer:" in tokenizer.decode(list(prefix))


@pytest.mark.skipif(
    not os.path.exists(FreshHyperparameters.tokenizer_path),
    reason="sp1024 tokenizer model not present",
)
def test_load_posttraining_tokenizer_keeps_sentencepiece_elsewhere():
    for architecture in ("nanogpt_mini_v1", "nanogpt_mini_tieddot_v1", "fresh"):
        tokenizer = load_posttraining_tokenizer(
            architecture, FreshHyperparameters.tokenizer_path
        )
        assert isinstance(tokenizer, spm.SentencePieceProcessor)


def test_nano_load_records_train_context_tokens(tmp_path):
    from postraining.model_io import load_model

    backbone = NanoGPTBackbone(vocab_size=64, num_layers=2, model_dim=256)
    config = dict(vocab_size=64, num_layers=2, model_dim=256, mlp_hidden=1024)
    base = {
        "model": {
            k: v.to(torch.bfloat16) if "embed" in k else v
            for k, v in backbone.state_dict().items()
        },
        "model_config": config,
        "architecture": "nanogpt_mini_v1",
    }
    legacy = tmp_path / "legacy.pt"
    torch.save(base, legacy)
    assert load_model(legacy, torch.device("cpu")).train_context_tokens == 1024
    longctx = tmp_path / "longctx.pt"
    torch.save({**base, "train_seq_len": 4096}, longctx)
    assert load_model(longctx, torch.device("cpu")).train_context_tokens == 4096


def test_build_optimizers_nano_backbone_six_group_layout():
    torch.manual_seed(37)
    backbone = NanoGPTBackbone(
        vocab_size=64, num_layers=2, model_dim=256
    ).float()
    wrapper = LatentThoughtModel(backbone)
    critic = SeparateCritic(
        NanoGPTBackbone(vocab_size=64, num_layers=2, model_dim=256).float(),
        num_bins=17,
        prior_value=0.1,
    )
    optimizers = build_optimizers(wrapper, critic, 1e-4, fused=False)
    groups = optimizers["actor"].param_groups
    assert len(groups) == 6
    renderer = renderer_parameters(backbone)
    assert [id(p) for p in groups[3]["params"]] == [id(p) for p in renderer]
    trunk_ids = {id(p) for p in groups[0]["params"]}
    renderer_ids = {id(p) for p in renderer}
    assert not trunk_ids & renderer_ids
    # Trunk + renderer must exactly cover the backbone's parameters.
    assert trunk_ids | renderer_ids == {id(p) for p in backbone.parameters()}
    assert all(len(group["params"]) > 0 for group in groups[:4])
