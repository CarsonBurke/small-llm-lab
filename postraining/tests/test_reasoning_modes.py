"""Pinned-EMIT (cot/none) reasoning-mode invariants across the VAPO stack."""

from __future__ import annotations

import os

import pytest
import sentencepiece as spm
import torch

from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters

from postraining.latent_rollout import (
    PAD_SLOT,
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
    PINNED_EMIT_ROLLOUT_POLICY_SCHEMAS,
    RENDERER_FEATURES_SCHEMA,
    ROLLOUT_POLICY_SCHEMA,
    THOUGHT_INPUT_SCHEMA,
    LatentThoughtModel,
    rollout_policy_schema_for_mode,
    validate_renderer_checkpoint,
)
from postraining.core import GPT2BPETokenizer, load_posttraining_tokenizer
from postraining.reasoning_modes import (
    checkpoint_training_rollout_budget,
    mode_rollout_budget,
    training_rollout_budget,
)
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


def test_reasoning_budgets_cover_latent_cot_and_answer_only_modes():
    # Thinking rides inside the carried belief rather than occupying stream
    # slots, so the stream cap always equals the emitted-token cap.
    assert mode_rollout_budget("latent", 700, answer_tokens=24) == (700, 700)
    assert mode_rollout_budget("cot", 700, answer_tokens=24) == (700, 700)
    assert mode_rollout_budget("none", 700, answer_tokens=24) == (24, 24)
    with pytest.raises(ValueError, match="unknown reasoning mode"):
        mode_rollout_budget("verbose", 700, answer_tokens=24)

    assert training_rollout_budget(
        "latent", 1024, answer_tokens=24
    ) == (1024, 1024)
    assert training_rollout_budget(
        "none", 1024, answer_tokens=24
    ) == (24, 24)


@pytest.mark.parametrize(
    ("saved", "expected"),
    [
        (
            {
                "resolved_train_max_new_tokens": 700,
                "resolved_train_max_stream_steps": 700,
            },
            (700, 700),
        ),
        (
            {"reasoning_mode": "latent", "continuation_tokens": 1024},
            (1024, 1024),
        ),
        (
            {"reasoning_mode": "cot", "continuation_tokens": 700},
            (700, 700),
        ),
        (
            {
                "reasoning_mode": "none",
                "continuation_tokens": 700,
                "answer_tokens": 24,
            },
            (24, 24),
        ),
    ],
)
def test_checkpoint_rollout_budget_reconstructs_saved_runs(saved, expected):
    assert checkpoint_training_rollout_budget(saved) == expected


def test_pin_emit_rollout_invariants():
    wrapper = _wrapper()
    torch.manual_seed(11)
    batch = _pinned_rollout(wrapper)
    assert bool((batch.kind != PAD_SLOT).any())
    # Zero-width hidden storage even with replay_storage=True (the default):
    # the pinned policy never carries a belief, so there is nothing to store.
    assert batch.hiddens.size(-1) == 0
    assert float(batch.action_mask.sum()) > 0.0


def test_pin_emit_nano_backbone_rollout():
    torch.manual_seed(7)
    backbone = NanoGPTBackbone(
        vocab_size=64, num_layers=2, model_dim=256
    ).float().eval()
    wrapper = LatentThoughtModel(backbone).eval()
    with torch.no_grad():
        batch = _pinned_rollout(wrapper)
    assert batch.hiddens.size(-1) == 0
    assert float(batch.action_mask.sum()) >= 2.0


def test_pin_emit_stream_latents_skip_combiner():
    wrapper = _wrapper()
    with torch.no_grad():
        # Even a live combiner must not touch a zero-width-hidden batch.
        wrapper.combiner.carry.weight.normal_(std=0.05)
        wrapper.combiner.type_bias.normal_(std=0.1)
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
    assert batch.hiddens.size(-1) == 0
    optimizers = build_optimizers(wrapper, critic, 1e-4, fused=False)
    metrics = update_minibatch(wrapper, critic, batch, optimizers)
    assert metrics["policy_clip_fraction"] == 0.0
    # Every reported scalar stays finite for the token-only batch.
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
        batch = trim_stream(
            _pinned_rollout(wrapper, pin_emit=pin_emit)
        )
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
        hiddens=torch.zeros(1, stream, 0),
        action_mask=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        old_token_logprobs=zeros.clone(),
        old_token_log_odds=zeros.clone(),
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


def test_rollout_diagnostics_pinned_reports_token_actions():
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
    assert metrics["actions_per_trajectory"] == 3.0
    assert metrics["trajectories"] == 2


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
    assert metrics["pin_emit"] is True
    assert len(seen) == 1
    prompt_ids, pin_emit = seen[0]
    assert pin_emit is True
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


def test_explicit_gpt2_provenance_keeps_gpt2_dispatch():
    pytest.importorskip("transformers")
    tokenizer = load_posttraining_tokenizer(
        "nanogpt_mini_gpt2vocab_v1",
        FreshHyperparameters.tokenizer_path,
        tokenizer_provenance={
            "kind": "gpt2",
            "name": "gpt2",
            "vocab_size": 50_257,
            "eot_id": 50_256,
            "directory": None,
            "spec_sha256": None,
            "ngrams_sha256": None,
        },
    )
    assert isinstance(tokenizer, GPT2BPETokenizer)


def test_checkpoint_provenance_overrides_misleading_gpt2_architecture(
    monkeypatch,
):
    decoded_runs = []

    class Spec:
        specials = (
            "<|endoftext|>",
            "<think>",
            "</think>",
            "<answer>",
            "</answer>",
        )

    class Tokenizer:
        spec = Spec()
        eot_id = 0

        def encode(self, text):
            assert text == "<think>x</think>"
            return [1, 5, 2]

        def decode(self, ids):
            ids = [int(token) for token in ids]
            decoded_runs.append(ids)
            return "".join({5: "x", 6: "y"}.get(token, "") for token in ids)

    monkeypatch.setattr(
        "pretraining.byte_accounting.load_bound_tokenizer",
        lambda provenance: Tokenizer(),
    )
    tokenizer = load_posttraining_tokenizer(
        "nanogpt_mini_gpt2vocab_kda_kdkd_mixers_v3",
        FreshHyperparameters.tokenizer_path,
        think_tokens=True,
        answer_tokens=True,
        tokenizer_provenance={"kind": "toast_tst"},
    )
    assert tokenizer.encode("<think>x</think>") == [1, 5, 2]
    assert tokenizer.decode([0, 1, 5, 2, 3, 6, 4]) == "xy"
    assert decoded_runs == [[5], [6]]
    assert (tokenizer.bos_id(), tokenizer.eos_id()) == (0, 0)
    assert (
        tokenizer.think_open_id,
        tokenizer.think_close_id,
        tokenizer.answer_open_id,
        tokenizer.answer_close_id,
    ) == (1, 2, 3, 4)


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


def test_build_optimizers_nano_backbone_three_group_layout():
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
    assert len(groups) == 3
    renderer = renderer_parameters(backbone)
    assert [id(p) for p in groups[2]["params"]] == [id(p) for p in renderer]
    trunk_ids = {id(p) for p in groups[0]["params"]}
    renderer_ids = {id(p) for p in renderer}
    assert not trunk_ids & renderer_ids
    # Trunk + renderer must exactly cover the backbone's parameters.
    assert trunk_ids | renderer_ids == {id(p) for p in backbone.parameters()}
    combiner_ids = {id(p) for p in groups[1]["params"]}
    assert combiner_ids == {id(p) for p in wrapper.combiner.parameters()}
    assert all(len(group["params"]) > 0 for group in groups)
