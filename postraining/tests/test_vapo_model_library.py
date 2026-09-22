"""Contracts of the model-agnostic VAPO library.

These are the guarantees that make one trainer serve several architectures:
the registry is checkpoint-bound, every adapter answers the same questions,
a family without adapters trains its trunk outright, and a nano rollout
produces exactly the currency the Hugging Face engine produces.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from postraining.nano_backbone import NanoGPTBackbone, NanoTiedDotBackbone
from postraining.vapo.model import Capability, get_family, list_families
from postraining.vapo.model.hf import HFCausalTrunk, HFModelSpec, MINICPM5_SPEC
from postraining.vapo.model.nano import NanoTrunk
from postraining.vapo.model.protocols import TrunkGeometry
from postraining.vapo.model.registry import register_family
from postraining.vapo.policy import VAPOCritic, VAPOPolicy
from postraining.vapo.rollout.nano_engine import NanoRolloutEngine
from postraining.vapo.rollout.results import ContinuousTrainingGeneration


VOCAB = 23
# The nano attention head dimension is fixed at 128, so a width below
# that gives zero heads and a degenerate trunk.
WIDTH = 256


def _nano(vocab: int = VOCAB, tied: bool = False) -> NanoTrunk:
    model_class = NanoTiedDotBackbone if tied else NanoGPTBackbone
    torch.manual_seed(7)
    backbone = model_class(vocab_size=vocab, num_layers=2, model_dim=WIDTH, mlp_hidden=512)
    backbone.model_config = {
        "vocab_size": vocab, "num_layers": 2, "model_dim": WIDTH, "mlp_hidden": 512,
    }
    backbone.architecture = "nanogpt_mini_test"
    return NanoTrunk(backbone, pad_token_id=0)


class _HFBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, WIDTH)
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.mlp = nn.Module()
        self.layers = nn.ModuleList([layer])

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        inputs = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(last_hidden_state=inputs.cumsum(dim=1))


class _HFCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="llama", vocab_size=VOCAB, hidden_size=WIDTH, pad_token_id=0,
            num_key_value_heads=2, head_dim=WIDTH // 2,
        )
        self.model = _HFBody()
        self.lm_head = nn.Linear(WIDTH, VOCAB, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens


TEST_SPEC = HFModelSpec(
    key="test", model_id="test/fixture", revision="0" * 40,
    vocab_size=VOCAB, requires_chat_template=False,
)


# -- registry --------------------------------------------------------------


def test_registry_lists_shipped_families_and_refuses_ambiguity() -> None:
    assert {"minicpm5", "nano"} <= set(list_families())
    assert get_family("minicpm5").key == "minicpm5"
    with pytest.raises(KeyError, match="unknown model family"):
        get_family("gpt5")
    with pytest.raises(ValueError, match="already registered"):
        register_family("minicpm5", lambda: get_family("minicpm5"))
    with pytest.raises(ValueError, match="lowercase"):
        register_family("MiniCPM5", lambda: get_family("minicpm5"))


def test_minicpm_identity_stays_pinned() -> None:
    """The family key is a checkpoint invariant; so is what it resolves to."""
    assert MINICPM5_SPEC.model_id == "openbmb/MiniCPM5-1B"
    assert MINICPM5_SPEC.revision == "87179e5c1f455ef22e6223592d2d61351b525bfc"
    assert MINICPM5_SPEC.vocab_size == 130_560


def test_family_override_is_recorded_rather_than_silent() -> None:
    family = get_family("minicpm5")
    overridden = family.resolved_spec(revision="a" * 40)
    assert overridden.revision == "a" * 40
    assert overridden.model_id == MINICPM5_SPEC.model_id
    assert family.resolved_spec() is MINICPM5_SPEC


# -- geometry --------------------------------------------------------------


def test_cache_geometry_refuses_incoherent_or_absent_attention_state() -> None:
    geometry = TrunkGeometry(
        hidden_size=8, vocab_size=9, num_layers=2,
        num_key_value_heads=2, head_dim=4, pad_token_id=0,
    )
    assert geometry.kv_cache_bytes(batch_size=3, cache_length=5) == 3 * 5 * 2 * 2 * 2 * 4 * 2
    recurrent = TrunkGeometry(
        hidden_size=8, vocab_size=9, num_layers=2,
        num_key_value_heads=0, head_dim=0, pad_token_id=0,
    )
    with pytest.raises(ValueError, match="no attention KV cache"):
        recurrent.kv_cache_bytes(batch_size=1, cache_length=2)
    with pytest.raises(ValueError, match="travel together"):
        TrunkGeometry(
            hidden_size=8, vocab_size=9, num_layers=2,
            num_key_value_heads=2, head_dim=0, pad_token_id=0,
        )
    with pytest.raises(ValueError, match="inside the vocabulary"):
        TrunkGeometry(
            hidden_size=8, vocab_size=9, num_layers=2,
            num_key_value_heads=2, head_dim=4, pad_token_id=9,
        )


def test_both_adapters_answer_the_same_questions() -> None:
    nano = _nano()
    hf = HFCausalTrunk(_HFCausalLM(), TEST_SPEC)
    for trunk in (nano, hf):
        assert trunk.hidden_size == WIDTH
        assert trunk.vocab_size == VOCAB
        assert trunk.geometry.num_layers == len(trunk.layers())
        assert trunk.identity()["family"] == trunk.family
        assert trunk.device == next(trunk.module.parameters()).device
    assert nano.geometry.num_key_value_heads > 0
    assert Capability.PAGED_KV_CACHE in nano.capabilities
    assert Capability.LORA_ADAPTERS not in nano.capabilities
    assert Capability.STATIC_KV_CACHE in hf.capabilities


def test_capability_requirements_name_what_is_missing() -> None:
    nano = _nano()
    with pytest.raises(RuntimeError, match="lora_adapters"):
        nano.require(Capability.LORA_ADAPTERS)
    with pytest.raises(RuntimeError, match="does not support gradient checkpointing"):
        nano.set_gradient_checkpointing(1)
    assert nano.set_gradient_checkpointing(0) == 0


# -- forward paths ---------------------------------------------------------


def test_nano_replay_reproduces_the_pretraining_forward() -> None:
    """Replay through the adapter must equal the backbone's own readout."""
    trunk = _nano()
    ids = torch.randint(0, VOCAB, (2, 5))
    hidden = trunk.hidden_states(input_ids=ids)
    torch.testing.assert_close(
        trunk.readout.logits(hidden), trunk.module.policy_logits(ids)
    )


def test_nano_replay_refuses_contracts_it_cannot_honour() -> None:
    trunk = _nano()
    ids = torch.randint(0, VOCAB, (1, 4))
    with pytest.raises(RuntimeError, match="packed variable-length"):
        trunk.hidden_states(input_ids=ids, cu_seqlens=torch.tensor([0, 4]))
    with pytest.raises(RuntimeError, match="neither an attention mask nor positions"):
        trunk.hidden_states(input_ids=ids, position_ids=torch.arange(4)[None])
    with pytest.raises(ValueError, match="exactly one"):
        trunk.hidden_states()


def test_nano_decode_step_matches_a_full_teacher_forced_pass() -> None:
    """A cached step must equal replay at the same position, or rollout
    log-probabilities and replay log-probabilities disagree from step one."""
    trunk = _nano()
    ids = torch.tensor([[3, 9, 4, 1]])
    caches = trunk.new_kv_cache(
        batch_size=1, cache_length=6, device=torch.device("cpu"), dtype=torch.float32
    )
    for keys, values in caches:
        keys.zero_()
        values.zero_()
    valid = torch.zeros(1, 6, dtype=torch.bool)
    valid[0, :3] = True
    trunk.module.prefill_belief(
        trunk.embed_tokens(ids[:, :3]), caches, valid[:, :3]
    )
    stepped = trunk.cached_hidden_states(
        input_ids=ids[:, 3:4],
        past_key_values=caches,
        cache_position=torch.tensor(3),
        attention_mask=valid[:, :4].clone().index_fill_(1, torch.tensor([3]), True),
    )
    dense = trunk.hidden_states(input_ids=ids)
    torch.testing.assert_close(stepped[:, -1], dense[:, -1], atol=2e-5, rtol=2e-5)


def test_tied_readout_scores_targets_like_its_dense_logits() -> None:
    trunk = _nano(tied=True)
    features = torch.randn(7, WIDTH)
    targets = torch.randint(0, VOCAB, (7,))
    expected = torch.log_softmax(trunk.readout.logits(features), dim=-1).gather(
        1, targets[:, None]
    ).squeeze(1)
    for chunk in (1, 3, 16):
        torch.testing.assert_close(
            trunk.readout.target_logprobs(features, targets, chunk_tokens=chunk),
            expected,
        )
    assert not trunk.readout.frozen


# -- policy and critic -----------------------------------------------------


def test_adapterless_family_trains_its_whole_trunk() -> None:
    trunk = _nano()
    policy = VAPOPolicy(trunk, None)
    trainable = {id(parameter) for parameter in policy.actor_parameters()}
    assert trainable >= {id(parameter) for parameter in trunk.parameters()}
    assert policy.lora_modules == ()


def test_lora_family_refuses_an_unadapted_actor() -> None:
    trunk = HFCausalTrunk(_HFCausalLM(), TEST_SPEC)
    with pytest.raises(ValueError, match="trains through LoRA adapters"):
        VAPOPolicy(trunk, None)


def test_checkpoints_carry_trunk_lineage() -> None:
    policy = VAPOPolicy(_nano(), None)
    critic = VAPOCritic(_nano(), None, critic_width=4)
    for payload, side in ((policy.checkpoint_payload(), policy), (critic.checkpoint_payload(), critic)):
        assert payload["trunk"] == side.trunk.identity()
        assert payload["trunk"]["architecture"] == "nanogpt_mini_test"
        assert payload["lora_config"] is None


# -- rollout ---------------------------------------------------------------


def _engine(trunk: NanoTrunk, **options) -> NanoRolloutEngine:
    return NanoRolloutEngine(
        trunk, batch_size=3, cache_length=12, stop_ids=(1,),
        temperature=0.0, cache_dtype=torch.float32, **options,
    )


def test_nano_engine_returns_the_shared_rollout_currency() -> None:
    trunk = _nano()
    engine = _engine(trunk)
    prompts = [torch.tensor([5, 6, 7]), torch.tensor([8, 2])]
    result = engine.generate_prompt_pool(prompts, max_new_tokens=4)
    assert isinstance(result, ContinuousTrainingGeneration)
    assert len(result.responses) == len(prompts) == len(result.logprobs)
    for response, logprobs in zip(result.responses, result.logprobs):
        assert response.dtype == torch.int32 and response.device.type == "cpu"
        assert 1 <= response.numel() <= 4
        assert logprobs.numel() == response.numel()
        assert bool(torch.isfinite(logprobs).all())
        # A stop token only ever terminates a row; it never appears early.
        assert not bool((response[:-1] == 1).any())
    assert result.useful_tokens == sum(int(r.numel()) for r in result.responses)
    assert result.decode_steps >= 1


def test_greedy_rollout_logprobs_match_a_teacher_forced_rescore() -> None:
    """The recorded behavior log-probability is the model's own, so the
    first replay step sees an importance ratio of exactly one."""
    trunk = _nano()
    engine = _engine(trunk)
    prompt = torch.tensor([4, 11, 6])
    result = engine.generate_prompt_pool([prompt], max_new_tokens=5)
    response = result.responses[0].long()
    stream = torch.cat((prompt, response))[None]
    logits = trunk.readout.logits(trunk.hidden_states(input_ids=stream))
    scored = torch.log_softmax(logits[0, prompt.numel() - 1 : -1].float(), dim=-1).gather(
        1, response[:, None]
    ).squeeze(1)
    torch.testing.assert_close(result.logprobs[0], scored, atol=2e-5, rtol=2e-5)


def test_engine_refuses_geometry_it_cannot_serve() -> None:
    trunk = _nano()
    engine = _engine(trunk)
    with pytest.raises(ValueError, match="exceed the rollout cache"):
        engine.generate_prompt_pool([torch.tensor([1, 2, 3])], max_new_tokens=11)
    with pytest.raises(ValueError, match="physical rows"):
        engine.generate_prompt_pool([torch.tensor([1])] * 4, max_new_tokens=2)
    with pytest.raises(ValueError, match="cannot be empty"):
        engine.generate_prompt_pool([], max_new_tokens=2)
    with pytest.raises(TypeError, match="does not implement"):
        engine.generate_prompt_pool(
            [torch.tensor([1, 2])], max_new_tokens=2, prefill_batch_prompts=4
        )


def test_releasing_the_cache_frees_it_and_regeneration_restores_it() -> None:
    trunk = _nano()
    engine = _engine(trunk)
    engine.prepare_generation()
    assert engine._caches is not None
    engine.release_cache()
    assert engine._caches is None
    result = engine.generate_prompt_pool([torch.tensor([3, 3])], max_new_tokens=2)
    assert engine._caches is not None
    assert result.responses[0].numel() >= 1
