from __future__ import annotations
from collections import Counter
import copy

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from postraining.core import generalized_advantage_estimate, length_adaptive_lambda
from postraining.fast_inference import (
    CapturedTrainingRolloutEngine,
    ContinuousTrainingGeneration,
    _CompactStaticLayer,
    _FrozenParameterStash,
    _completion_poll_chunk,
    _take_refill_rows,
    W8A16Linear,
    fuse_llama_projections_,
    synchronize_fused_lora_policy_,
    selected_token_logprobs,
    top_k_top_p_sample,
)
from postraining.minicpm_vapo import (
    LoRAConfig,
    LoRALinear,
    MiniCPMVAPOPolicy,
    NextLatAuxiliaryHead,
    StaticCachePool,
    TrajectoryRecord,
    _nucleus_membership,
    _packed_replay_attention,
    _packed_replay_attention_fa4,
    _ReplaySiLU,
    chunked_frozen_head_logprobs,
    collate_replay_microbatch,
    dense_top_p_probabilities,
    enable_replay_mlp_compilation,
    exact_top_p_sample,
    maximal_coupling_verify,
    inject_lora,
    merge_lora_for_inference,
    plan_replay_microbatches,
    prepare_text_only_transformers_runtime,
    share_frozen_parameters_,
    replay_storage_bytes,
    use_packed_replay_attention,
)
from postraining.runtime.profiling import DeviceSampler
from postraining.nextlat_speculative import (
    NextLatSpeculativeEngine,
    RaggedStaticCacheLayer,
    ragged_causal_mask,
    RaggedStaticCache,
)
from postraining.train_minicpm_vapo import (
    RolloutEngine,
    _validate_args,
    _ChunkedNextLatKL,
    _accumulate_balanced_parameter_gradients_,
    _combine_balanced_hidden_gradients_,
    _accumulate_rescaled_parameter_gradients_,
    _clip_finite_grad_norm_,
    _approximate_kl_terms,
    _scale_auxiliary_loss,
    _build_group_records,
    _nextlat_training_loss,
    _LIVE_TAGS,
    _ROLLOUT_EFFICIENCY_TAGS,
    _ROLLOUT_PERFORMANCE_TAGS,
    _ROLLOUT_QUALITY_TAGS,
    _ROLLOUT_SAMPLING_TAGS,
    _TRAIN_TAGS,
    _optimizer_minibatches,
    _nextlat_shard_samples,
    collect_rollouts,
    configure_replay_checkpointing,
    build_parser,
    device_phase_metrics,
    file_sha256,
    static_kv_cache_bytes,
    tensorboard_scalars,
    tensorboard_rollout_samples,
    reassert_optimizer_learning_rates,
    validate_resume_configuration,
    validate_resume_dataset,
    validate_resume_rollout_arithmetic,
)
from postraining.validate_minicpm_vapo import prompt_ids
from scripts.benchmark_minicpm_nextlat import required_warmup_steps


class _TinyLlamaBlock(nn.Module):
    def __init__(self, width: int = 6) -> None:
        super().__init__()
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            setattr(self, name, nn.Linear(width, width, bias=False))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = inputs
        for module in self.children():
            result = module(result)
        return result


def test_lora_injection_is_initially_exact_and_freezes_base() -> None:
    torch.manual_seed(1)
    model = _TinyLlamaBlock()
    inputs = torch.randn(3, 6)
    expected = model(inputs)
    names = inject_lora(model, LoRAConfig(rank=2, alpha=4))
    actual = model(inputs)
    assert len(names) == 7
    assert torch.equal(actual, expected)
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert len(trainable) == 14
    assert all(name.endswith(("lora_a", "lora_b")) for name in trainable)




def test_nora_init_normalizes_lora_a_columns_once() -> None:
    torch.manual_seed(7)
    base = nn.Linear(6, 5, bias=False)
    standard = LoRALinear(
        copy.deepcopy(base),
        LoRAConfig(rank=2, alpha=4, initialization="standard"),
    )
    torch.manual_seed(7)
    base = nn.Linear(6, 5, bias=False)
    nora = LoRALinear(
        copy.deepcopy(base),
        LoRAConfig(rank=2, alpha=4, initialization="nora"),
    )

    standard_norms = torch.linalg.vector_norm(standard.lora_a, dim=0)
    nora_norms = torch.linalg.vector_norm(nora.lora_a, dim=0)
    torch.testing.assert_close(nora_norms, torch.ones_like(nora_norms))
    torch.testing.assert_close(
        nora.lora_a,
        standard.lora_a / standard_norms.unsqueeze(0),
    )
    assert torch.count_nonzero(nora.lora_b) == 0


def test_lora_config_rejects_unknown_initialization() -> None:
    with pytest.raises(ValueError, match="initialization"):
        LoRAConfig(initialization="unknown")  # type: ignore[arg-type]



def test_actor_and_critic_share_only_frozen_parameter_storage() -> None:
    torch.manual_seed(3)
    actor = _TinyLlamaBlock()
    critic = copy.deepcopy(actor)
    inject_lora(actor, LoRAConfig(rank=2, alpha=4))
    inject_lora(critic, LoRAConfig(rank=2, alpha=4))
    shared = share_frozen_parameters_(critic, actor)
    assert len(shared) == 7
    actor_parameters = dict(actor.named_parameters())
    critic_parameters = dict(critic.named_parameters())
    for name in shared:
        assert (
            actor_parameters[name].untyped_storage().data_ptr()
            == critic_parameters[name].untyped_storage().data_ptr()
        )
    for name in actor_parameters:
        if name.endswith(("lora_a", "lora_b")):
            assert (
                actor_parameters[name].untyped_storage().data_ptr()
                != critic_parameters[name].untyped_storage().data_ptr()
            )




def test_reference_nextlat_head_matches_one_billion_parameter_configuration() -> None:
    head = NextLatAuxiliaryHead(1536, projection_factor=1.6)
    assert sum(parameter.numel() for parameter in head.parameters()) == 46_074_880
    assert isinstance(head.norm, nn.LayerNorm)
    assert head.norm.bias is None

def test_benchmark_reads_serialized_warmup_requirement() -> None:
    assert required_warmup_steps(
        {"args": {"value_warmup_steps": 10}}
    ) == 10
    with pytest.raises(ValueError, match="training arguments"):
        required_warmup_steps({})
    with pytest.raises(ValueError, match="warmup requirement"):
        required_warmup_steps({"args": {"value_warmup_steps": 0}})


def test_lora_merge_preserves_outputs_and_removes_runtime_adapters() -> None:
    torch.manual_seed(7)
    model = _TinyLlamaBlock()
    inputs = torch.randn(3, 6)
    injected = inject_lora(model, LoRAConfig(rank=2, alpha=4))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora_b"):
                parameter.normal_(std=0.01)
    expected = model(inputs)
    merged = merge_lora_for_inference(model)
    actual = model(inputs)
    assert merged == injected
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert not any(
        name.endswith(("lora_a", "lora_b"))
        for name, _ in model.named_parameters()
    )


def test_w8a16_linear_uses_rowwise_scales() -> None:
    torch.manual_seed(11)
    linear = nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
    quantized = W8A16Linear(linear)
    reconstructed = quantized.reconstructed_weight()
    assert quantized.weight_scale.shape == (32, 1)
    assert reconstructed.shape == linear.weight.shape
    assert torch.cosine_similarity(
        linear.weight.float().flatten(),
        reconstructed.flatten(),
        dim=0,
    ) > 0.999


def test_llama_projection_fusion_preserves_eager_outputs() -> None:
    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(6, 6, bias=False)
            self.k_proj = nn.Linear(6, 2, bias=False)
            self.v_proj = nn.Linear(6, 2, bias=False)

        def forward(self, inputs):
            return (
                self.q_proj(inputs),
                self.k_proj(inputs),
                self.v_proj(inputs),
            )

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(6, 8, bias=False)
            self.up_proj = nn.Linear(6, 8, bias=False)

        def forward(self, inputs):
            return torch.nn.functional.silu(
                self.gate_proj(inputs)
            ) * self.up_proj(inputs)

    layer = nn.Module()
    layer.self_attn = Attention()
    layer.mlp = MLP()
    model = nn.Module()
    model.layers = nn.ModuleList([layer])
    causal_lm = nn.Module()
    causal_lm.model = model
    inputs = torch.randn(2, 6)
    expected_attention = layer.self_attn(inputs)
    expected_mlp = layer.mlp(inputs)
    assert len(fuse_llama_projections_(causal_lm)) == 2
    for _ in range(2):
        actual_attention = layer.self_attn(inputs)
        actual_mlp = layer.mlp(inputs)
        for actual, expected in zip(
            actual_attention, expected_attention, strict=True
        ):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual_mlp, expected_mlp, atol=1e-6, rtol=1e-6)


def test_fused_rollout_replica_synchronizes_live_actor_lora() -> None:
    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(6, 6, bias=False)
            self.k_proj = nn.Linear(6, 6, bias=False)
            self.v_proj = nn.Linear(6, 6, bias=False)
            self.o_proj = nn.Linear(6, 6, bias=False)

        def forward(self, inputs):
            q, k, v = self.q_proj(inputs), self.k_proj(inputs), self.v_proj(inputs)
            return self.o_proj(q + k + v)

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(6, 8, bias=False)
            self.up_proj = nn.Linear(6, 8, bias=False)
            self.down_proj = nn.Linear(8, 6, bias=False)

        def forward(self, inputs):
            return self.down_proj(
                torch.nn.functional.silu(self.gate_proj(inputs))
                * self.up_proj(inputs)
            )

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attention()
            self.mlp = MLP()

        def forward(self, inputs):
            return self.mlp(self.self_attn(inputs))

    torch.manual_seed(17)
    source_lm = nn.Module()
    source_lm.model = nn.Module()
    source_lm.model.layers = nn.ModuleList([Layer()])
    destination_lm = copy.deepcopy(source_lm)
    inject_lora(source_lm, LoRAConfig(rank=2, alpha=4))
    with torch.no_grad():
        for name, parameter in source_lm.named_parameters():
            if name.endswith("lora_b"):
                parameter.normal_(std=0.03)
    fuse_llama_projections_(destination_lm)
    source = SimpleNamespace(causal_lm=source_lm)
    destination = SimpleNamespace(causal_lm=destination_lm)

    assert synchronize_fused_lora_policy_(destination, source) == 7
    inputs = torch.randn(4, 6)
    torch.testing.assert_close(
        destination_lm.model.layers[0](inputs),
        source_lm.model.layers[0](inputs),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("indexed_decode", [False, True])
def test_compact_static_cache_packs_prefill_and_appends_per_row(
    indexed_decode: bool,
) -> None:
    sequence_lengths = torch.tensor([3, 2])
    prefill_mask = torch.tensor(
        [[False, True, True, True], [False, False, True, True]]
    )
    layer = _CompactStaticLayer(
        8, sequence_lengths, prefill_mask, indexed_decode=indexed_decode
    )
    keys = torch.arange(2 * 2 * 4 * 3, dtype=torch.float32).reshape(2, 2, 4, 3)
    values = keys + 100

    returned_keys, returned_values = layer.update(keys, values)
    assert returned_keys is keys
    assert returned_values is values
    torch.testing.assert_close(
        layer.key_backing[0, :3], keys[0, :, 1:].transpose(0, 1)
    )
    torch.testing.assert_close(
        layer.key_backing[1, :2], keys[1, :, 2:].transpose(0, 1)
    )

    next_keys = torch.full((2, 2, 1, 3), 999.0)
    next_values = next_keys + 100
    layer.prefilling = False
    cached_keys, cached_values = layer.update(next_keys, next_values)
    torch.testing.assert_close(
        cached_keys[torch.arange(2), :, sequence_lengths],
        next_keys[:, :, 0],
    )
    torch.testing.assert_close(
        cached_values[torch.arange(2), :, sequence_lengths],
        next_values[:, :, 0],
    )


def test_compact_static_cache_reset_invalidates_without_rewriting_storage() -> None:
    sequence_lengths = torch.tensor([2])
    prefill_mask = torch.tensor([[True, True]])
    layer = _CompactStaticLayer(6, sequence_lengths, prefill_mask)
    keys = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    layer.update(keys, keys)
    stale_tail = torch.full_like(layer.key_backing[:, 2:], 17.0)
    layer.key_backing[:, 2:].copy_(stale_tail)
    key_pointer = layer.key_backing.data_ptr()

    layer.reset()

    assert layer.key_backing.data_ptr() == key_pointer
    assert int(layer.cumulative_length) == 0
    torch.testing.assert_close(layer.key_backing[:, 2:], stale_tail)


def test_frozen_parameter_stash_moves_only_immutable_storage() -> None:
    module = nn.Sequential(
        nn.Linear(3, 4, bias=False),
        nn.Linear(4, 2, bias=False),
    )
    first = cast(nn.Linear, module[0])
    second = cast(nn.Linear, module[1])
    first.weight.requires_grad_(False)
    frozen = first.weight
    trainable = second.weight
    expected = frozen.detach().clone()
    stash = _FrozenParameterStash(module)

    stash.offload()
    assert not stash.resident
    assert stash.bytes == frozen.numel() * frozen.element_size()
    assert stash.parameters[0] is frozen
    assert trainable.requires_grad

    stash.restore()
    assert stash.resident
    torch.testing.assert_close(frozen, expected)


def test_rollout_capacity_reserves_workspace_after_weight_offload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = cast(Any, object.__new__(CapturedTrainingRolloutEngine))
    engine._runtime_device = torch.device("cuda")
    engine.estimated_cache_bytes = 19 << 30
    engine.offloaded_source_bytes = 2 << 30
    engine.cache = SimpleNamespace(layers=[])
    monkeypatch.setattr(
        "postraining.fast_inference._cuda_allocatable_bytes",
        lambda device: 20 << 30,
    )

    engine._validate_cache_capacity()

    monkeypatch.setattr(
        "postraining.fast_inference._cuda_allocatable_bytes",
        lambda device: (20 << 30) - 1,
    )
    with pytest.raises(MemoryError, match="KV plus 1.00 GiB workspace"):
        engine._validate_cache_capacity()

def test_compact_static_cache_safely_replays_completed_max_length_rows() -> None:
    sequence_lengths = torch.tensor([4])
    prefill_mask = torch.ones((1, 2), dtype=torch.bool)
    layer = _CompactStaticLayer(4, sequence_lengths, prefill_mask)
    keys = torch.zeros((1, 2, 2, 3))
    layer.update(keys, keys)
    final = torch.full((1, 2, 1, 3), 7.0)

    layer.prefilling = False
    layer.update(final, final)

    torch.testing.assert_close(layer.key_backing[0, -1], final[0, :, 0])
    assert int(layer.cumulative_length) == 4





def test_first_captured_schedule_replays_every_persistent_decode() -> None:
    class Graph:
        replays = 0

        def replay(self) -> None:
            self.replays += 1

    graph = Graph()
    engine = SimpleNamespace(
        _graph_logits=torch.zeros(1),
        _graph_values=torch.zeros(1),
        _compile_decode=True,
        _decode_graph=None,
        _report_progress=lambda *args, **kwargs: None,
    )
    engine._capture_static_decode_schedule = lambda: setattr(
        engine, "_decode_graph", graph
    )

    CapturedTrainingRolloutEngine._run_decode_schedule(
        cast(CapturedTrainingRolloutEngine, engine),
        torch.ones(1),
        torch.ones(1),
        5,
        started=0.0,
        progress_callback=None,
    )

    assert graph.replays == 3


def test_statistics_free_captured_schedule_uses_continuous_graph() -> None:
    class Graph:
        replays = 0

        def replay(self) -> None:
            self.replays += 1

    graph = Graph()
    engine = SimpleNamespace(
        _graph_logits=torch.zeros(1),
        _graph_values=torch.zeros(1),
        _compile_decode=True,
        _continuous_decode_graph=None,
        _report_progress=lambda *args, **kwargs: None,
    )
    engine._capture_continuous_decode_schedule = lambda: setattr(
        engine, "_continuous_decode_graph", graph
    )

    CapturedTrainingRolloutEngine._run_decode_schedule(
        cast(CapturedTrainingRolloutEngine, engine),
        torch.ones(1),
        None,
        5,
        started=0.0,
        progress_callback=None,
    )

    assert graph.replays == 3

def test_continuous_refill_fills_every_available_lane() -> None:
    free = [1, 3, 4, 7, 8, 9, 10]
    selected, rows = _take_refill_rows(
        free,
        pending_row=2,
        total_rows=8,
    )
    assert selected == [1, 3, 4, 7, 8, 9]
    assert rows == 6

    selected, rows = _take_refill_rows(
        free,
        pending_row=7,
        total_rows=8,
    )
    assert selected == [1]
    assert rows == 1

def test_continuous_polling_batches_sync_without_crossing_token_limit() -> None:
    assert (
        _completion_poll_chunk(
            [0, 93, 99],
            [1, 2],
            max_new_tokens=100,
            poll_steps=16,
        )
        == 1
    )
    assert (
        _completion_poll_chunk(
            [0, 32, 48],
            [1, 2],
            max_new_tokens=100,
            poll_steps=16,
        )
        == 16
    )
    assert (
        _completion_poll_chunk(
            [94, 90],
            [0, 1],
            max_new_tokens=100,
            poll_steps=16,
            maximum_tokens_per_step=5,
        )
        == 2
    )

def test_prompt_major_prefix_slice_is_contiguous_across_layers() -> None:
    bank = torch.empty(3, 11, 4, 2, 8)
    assert bank[1, :7].is_contiguous()


def test_captured_rollout_release_keeps_replica_resident_and_evicts_only_cache() -> (
    None
):
    import gc
    import weakref

    attention = SimpleNamespace(
        _rollout_sequence_lengths=torch.ones(1),
    )
    policy = SimpleNamespace(
        causal_lm=SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
        ),
    )
    engine = cast(Any, object.__new__(CapturedTrainingRolloutEngine))
    engine.policy = policy
    engine._rollout_resident = True
    engine._compile_decode = True
    engine._decode_graph = object()
    engine._continuous_decode_graph = object()
    engine._num_hidden_layers = 1
    engine.cache_length = 8
    engine.sequence_lengths = torch.zeros(2, dtype=torch.long)
    engine.attention_mask = torch.zeros(2, 8, dtype=torch.bool)
    engine.cache = engine._new_cache()
    engine.cache.layers[0].lazy_initialization(
        torch.empty(2, 1, 1, 2), torch.empty(2, 1, 1, 2)
    )
    retained_cache = engine.cache
    allocations = [
        weakref.ref(tensor)
        for layer in retained_cache.layers
        for tensor in (layer.key_backing, layer.value_backing)
    ]
    engine._source_stash = SimpleNamespace(restore=lambda: None)

    automatic_gc = gc.isenabled()
    gc.disable()
    try:
        # Simulate unreachable Dynamo bookkeeping retaining a layer separately.
        cycle = {"layer": retained_cache.layers[0]}
        cycle["self"] = cycle
        del cycle
        engine.release_cache()
        assert all(reference() is None for reference in allocations)
    finally:
        if automatic_gc:
            gc.enable()
        gc.collect()


def test_capacity_guard_counts_only_unallocated_kv(monkeypatch) -> None:
    import postraining.fast_inference as runtime

    engine = cast(Any, object.__new__(CapturedTrainingRolloutEngine))
    engine._runtime_device = torch.device("cuda")
    engine.offloaded_source_bytes = 0
    engine.estimated_cache_bytes = 32
    layer = SimpleNamespace(
        is_initialized=False,
        key_backing=torch.empty(4),
        value_backing=torch.empty(4),
    )
    engine.cache = SimpleNamespace(layers=[layer])
    monkeypatch.setattr(
        runtime, "_cuda_allocatable_bytes",
        lambda device: runtime.ROLLOUT_WORKSPACE_RESERVE_BYTES + 16,
    )
    with pytest.raises(MemoryError):
        engine._validate_cache_capacity()
    layer.is_initialized = True
    engine._validate_cache_capacity()
    monkeypatch.setattr(
        runtime, "_cuda_allocatable_bytes",
        lambda device: runtime.ROLLOUT_WORKSPACE_RESERVE_BYTES - 1,
    )
    with pytest.raises(MemoryError):
        engine._validate_cache_capacity()


def test_continuous_generation_reports_productive_utilization() -> None:
    generation = ContinuousTrainingGeneration(
        responses=(torch.arange(3), torch.arange(5)),
        logprobs=(torch.zeros(3), torch.zeros(5)),
        prefill_seconds=0.1,
        decode_seconds=0.2,
        decode_steps=2,
        useful_tokens=8,
        capacity_row_steps=16,
        admission_events=1,
        minimum_active_rows_with_backlog=8,
    )
    assert generation.productive_utilization == 0.5


def test_captured_rollouts_reuse_unique_prompt_prefills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = cast(Any, object.__new__(CapturedTrainingRolloutEngine))
    engine.prompts_per_rollout = 2
    engine.samples_per_prompt = 1
    engine.stop_ids = (9,)
    engine.top_k = 2
    engine.top_p = 0.9
    engine.policy = SimpleNamespace(
        causal_lm=SimpleNamespace(config=SimpleNamespace(vocab_size=10))
    )
    called: list[int] = []

    def generate_prompt_pool(prompt_ids, **kwargs):
        del kwargs
        called.append(len(prompt_ids))
        return ContinuousTrainingGeneration(
            responses=(torch.tensor([9]), torch.tensor([9])),
            logprobs=(torch.zeros(1), torch.zeros(1)),
            prefill_seconds=0.1,
            decode_seconds=0.2,
            decode_steps=1,
            useful_tokens=2,
            capacity_row_steps=2,
            admission_events=1,
            minimum_active_rows_with_backlog=2,
        )

    engine.generate_prompt_pool = generate_prompt_pool
    engine.generate_prompts = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("captured rollout must not repeat each prompt prefill")
    )
    monkeypatch.setattr(
        "postraining.train_minicpm_vapo.encode_math_prompt",
        lambda *args, **kwargs: torch.tensor([1]),
    )
    monkeypatch.setattr(
        "postraining.train_minicpm_vapo._build_group_records",
        lambda *args, **kwargs: ([SimpleNamespace()], 1),
    )

    result = collect_rollouts(
        engine,
        object(),
        [{}, {}],
        prompt_tokens=8,
        max_new_tokens=4,
        enable_thinking=True,
    )

    assert called == [2]
    assert len(result.records) == 2


def test_fast_top_k_top_p_sampler_respects_nucleus_boundary() -> None:
    logits = torch.tensor(
        [[20.0, 0.0, -20.0, -40.0], [-40.0, -20.0, 0.0, 20.0]]
    )
    sampled = top_k_top_p_sample(
        logits,
        temperature=1.0,
        top_k=2,
        top_p=0.95,
    )
    assert sampled.tolist() == [0, 3]


def test_chunked_frozen_head_logprobs_match_dense_values_and_gradients() -> None:
    torch.manual_seed(2)
    hidden = torch.randn(11, 5, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    weight = torch.randn(17, 5)
    targets = torch.randint(0, 17, (11,))
    coefficients = torch.randn(11)

    actual = chunked_frozen_head_logprobs(
        hidden, targets, weight, chunk_tokens=3
    )
    expected = torch.log_softmax(reference_hidden @ weight.T, dim=-1).gather(
        1, targets[:, None]
    ).squeeze(1)
    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert hidden.grad is not None
    assert reference_hidden.grad is not None
    assert torch.allclose(hidden.grad, reference_hidden.grad, atol=2e-6, rtol=2e-6)


def test_exact_top_p_sampler_prices_selected_token_under_full_policy() -> None:
    logits = torch.tensor(
        [[8.0, 7.0, 6.0, -9.0, -10.0], [4.0, 3.0, 2.0, 1.0, -8.0]]
    )
    sampled, logprobs, stats = exact_top_p_sample(
        logits,
        temperature=0.9,
        top_p=0.8,
        generator=torch.Generator().manual_seed(3),
    )
    expected = torch.log_softmax(logits, dim=-1).gather(
        1, sampled[:, None]
    ).squeeze(1)
    assert stats.scanned_vocabulary == 5
    assert stats.nucleus_mass_lower_bound == pytest.approx(0.8)
    assert torch.allclose(logprobs, expected)


def test_nucleus_membership_uses_deterministic_token_id_tie_breaking() -> None:
    logits = torch.tensor(
        [[3.0, 2.0, 1.0], [1.0, 1.0, 0.0]]
    )
    probabilities = logits.softmax(dim=-1)
    assert _nucleus_membership(
        logits[:1].expand(3, -1),
        probabilities[:1].expand(3, -1),
        torch.tensor([0, 1, 2]),
        0.8,
    ).tolist() == [True, True, False]
    assert _nucleus_membership(
        logits[1:].expand(2, -1),
        probabilities[1:].expand(2, -1),
        torch.tensor([0, 1]),
        0.4,
    ).tolist() == [True, False]


def test_exact_top_p_rejection_sampler_never_leaves_the_nucleus() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]]).expand(4_096, -1)
    sampled, _, _ = exact_top_p_sample(
        logits,
        temperature=1.0,
        top_p=0.8,
        generator=torch.Generator().manual_seed(7),
    )
    assert set(sampled.tolist()) == {0, 1}
    expected_zero_fraction = torch.softmax(logits[0, :2], dim=0)[0]
    observed_zero_fraction = (sampled == 0).float().mean()
    assert observed_zero_fraction == pytest.approx(
        float(expected_zero_fraction), abs=0.03
    )

def test_dense_top_p_matches_deterministic_nucleus() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    probabilities = dense_top_p_probabilities(
        logits, temperature=1.0, top_p=0.8
    )
    expected = torch.softmax(logits[0, :2], dim=0)
    torch.testing.assert_close(probabilities[0, :2], expected)
    assert probabilities[0, 2:].count_nonzero() == 0
    tied = dense_top_p_probabilities(
        torch.tensor([[1.0, 1.0, 0.0]]),
        temperature=1.0,
        top_p=0.4,
    )
    assert tied.nonzero(as_tuple=False).tolist() == [[0, 0]]

def test_adaptive_dense_top_p_matches_full_stable_sort() -> None:
    torch.manual_seed(11)
    logits = torch.randn(8, 1_024)
    for top_p in (0.1, 0.5, 0.95):
        actual = dense_top_p_probabilities(
            logits, temperature=0.9, top_p=top_p
        )
        scaled = logits.float() / 0.9
        sorted_logits, sorted_ids = torch.sort(
            scaled, dim=-1, descending=True, stable=True
        )
        sorted_probabilities = sorted_logits.softmax(dim=-1)
        preceding = (
            sorted_probabilities.cumsum(dim=-1) - sorted_probabilities
        )
        sorted_probabilities.masked_fill_(preceding >= top_p, 0.0)
        sorted_probabilities /= sorted_probabilities.sum(
            dim=-1, keepdim=True
        )
        expected = torch.zeros_like(sorted_probabilities)
        expected.scatter_(1, sorted_ids, sorted_probabilities)
        torch.testing.assert_close(actual, expected)


def test_maximal_coupling_accepts_identical_distributions() -> None:
    probabilities = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    proposals = torch.tensor([0, 1])
    committed, accepted = maximal_coupling_verify(
        probabilities,
        probabilities,
        proposals,
        torch.tensor([True, True]),
        generator=torch.Generator().manual_seed(3),
    )
    assert accepted.tolist() == [True, True]
    assert torch.equal(committed, proposals)
    committed, accepted = maximal_coupling_verify(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([1]),
        torch.tensor([True]),
        generator=torch.Generator().manual_seed(4),
    )
    assert accepted.tolist() == [False]
    assert committed.tolist() == [0]


def test_nextlat_auxiliary_trains_source_hidden_and_predictor() -> None:
    class Policy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.nextlat_head = NextLatAuxiliaryHead(128, projection_factor=0.5)
            self.embedding = nn.Embedding(7, 128)
            for parameter in self.embedding.parameters():
                parameter.requires_grad_(False)

        @property
        def lm_head_weight(self):
            return self.embedding.weight

        def token_embeddings(self, token_ids):
            return self.embedding(token_ids)

    policy = Policy()
    hidden = torch.randn(1, 5, 128, requires_grad=True)
    batch = SimpleNamespace(
        input_ids=torch.tensor([[0, 1, 2, 3, 4]]),
        response_state_mask=torch.ones((1, 5), dtype=torch.bool),
        sequence_ids=torch.zeros((1, 5), dtype=torch.long),
    )
    result = _nextlat_training_loss(
        cast(MiniCPMVAPOPolicy, policy),
        hidden,
        batch,
        max_samples=3,
        horizon=2,
        mse_coefficient=1.0,
        kl_coefficient=1.0,
        kl_chunk_tokens=2,
    )
    result.loss.backward()
    assert result.samples == 3
    assert result.transitions == 6
    assert torch.isfinite(result.loss)
    assert result.categorical_kl >= 0
    assert hidden.grad is not None
    assert float(hidden.grad.abs().sum()) > 0
    assert any(
        parameter.grad is not None
        for parameter in policy.nextlat_head.parameters()
    )
    assert all(parameter.grad is None for parameter in policy.embedding.parameters())


def test_chunked_nextlat_kl_matches_dense_value_and_gradient() -> None:
    torch.manual_seed(17)
    predicted = torch.randn(5, 8, requires_grad=True)
    target = torch.randn(5, 8)
    weight = torch.randn(11, 8)
    chunked = _ChunkedNextLatKL.apply(predicted, target, weight, 2)
    dense_teacher = torch.nn.functional.log_softmax(target @ weight.T, dim=-1)
    dense_student = torch.nn.functional.log_softmax(predicted @ weight.T, dim=-1)
    dense = (
        dense_teacher.exp() * (dense_teacher - dense_student)
    ).sum(dim=-1).mean()
    dense_gradient = torch.autograd.grad(dense, predicted, retain_graph=True)[0]
    chunked_gradient = torch.autograd.grad(chunked, predicted)[0]
    torch.testing.assert_close(chunked, dense)
    torch.testing.assert_close(chunked_gradient, dense_gradient)


def test_selected_rollout_logprobs_match_full_policy_distribution() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0]])
    tokens = torch.tensor([2, 1])
    expected = logits.log_softmax(dim=-1).gather(1, tokens[:, None]).squeeze(1)
    torch.testing.assert_close(selected_token_logprobs(logits, tokens), expected)


def test_ragged_cache_writes_independent_row_positions() -> None:
    layer = RaggedStaticCacheLayer(
        batch_size=2,
        num_heads=1,
        max_cache_len=5,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    layer.write_positions = torch.tensor([[1], [3]])
    keys = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])
    values = keys + 10
    cached_keys, cached_values = layer.update(keys, values)
    torch.testing.assert_close(cached_keys[0, 0, 1], keys[0, 0, 0])
    torch.testing.assert_close(cached_keys[1, 0, 3], keys[1, 0, 0])
    torch.testing.assert_close(cached_values[0, 0, 1], values[0, 0, 0])
    assert cached_keys[0, 0, 3].count_nonzero() == 0


def test_ragged_causal_mask_respects_each_row_cursor() -> None:
    valid = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    mask = ragged_causal_mask(valid, torch.tensor([[1], [2]]))
    assert mask[:, 0, 0].tolist() == [
        [True, True, False, False],
        [True, True, True, False],
    ]

def test_ragged_cache_runs_through_real_tiny_llama() -> None:
    prepare_text_only_transformers_runtime()
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    model = LlamaForCausalLM(config).model.eval()
    cache = RaggedStaticCache(
        config,
        batch_size=2,
        max_cache_len=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    prefill_positions = torch.tensor([[0, 1], [0, 1]])
    prefill_valid = torch.tensor(
        [[True, True, False, False], [True, False, False, False]]
    )
    cache.set_write_positions(prefill_positions)
    model(
        input_ids=torch.tensor([[1, 2], [3, 0]]),
        attention_mask=ragged_causal_mask(
            prefill_valid, prefill_positions
        ),
        position_ids=prefill_positions,
        past_key_values=cache,
        use_cache=True,
    )
    decode_positions = torch.tensor([[2], [1]])
    decode_valid = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    cache.set_write_positions(decode_positions)
    output = model(
        input_ids=torch.tensor([[4], [5]]),
        attention_mask=ragged_causal_mask(
            decode_valid, decode_positions
        ),
        position_ids=decode_positions,
        past_key_values=cache,
        use_cache=True,
    )
    assert output.last_hidden_state.shape == (2, 1, 16)
    assert cache.layers[0].keys[0, :, 2].count_nonzero() > 0
    assert cache.layers[0].keys[1, :, 1].count_nonzero() > 0

def test_speculative_engine_keeps_independent_row_lengths() -> None:
    class Config:
        num_hidden_layers = 1
        num_key_value_heads = 1
        head_dim = 1
        pad_token_id = 0
        vocab_size = 4

    class CausalLM:
        config = Config()

    class ExactPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.causal_lm = CausalLM()
            self.cache_positions: list[torch.Tensor] = []

        def cached_hidden(self, input_ids, **kwargs):
            self.cache_positions.append(kwargs["cache_position"].clone())
            return torch.nn.functional.one_hot(
                input_ids, num_classes=4
            ).float()

        def logits(self, hidden):
            current = hidden.argmax(dim=-1)
            target = (current + 1).remainder(4)
            logits = torch.full((*current.shape, 4), -100.0)
            return logits.scatter_(-1, target[..., None], 100.0)

        def rollout_values(self, hidden):
            return torch.zeros(hidden.shape[:-1], dtype=torch.bfloat16)

        def nextlat_hidden(self, _, token):
            return torch.nn.functional.one_hot(
                token, num_classes=4
            ).float()

    exact_policy = ExactPolicy()
    engine = NextLatSpeculativeEngine(
        cast(MiniCPMVAPOPolicy, exact_policy),
        stop_ids=(3,),
        prompts_per_rollout=2,
        samples_per_prompt=2,
        cache_length=8,
        draft_length=2,
        temperature=1.0,
        top_p=1.0,
        compile_decode=False,
    )
    responses, _, _, _, _, stats = engine.generate_prompts(
        [torch.tensor([0]), torch.tensor([1])],
        max_new_tokens=4,
    )
    assert responses.tolist() == [
        [1, 2, 3],
        [1, 2, 3],
        [2, 3, 3],
        [2, 3, 3],
    ]
    assert stats.accepted_tokens == stats.proposed_tokens
    assert stats.target_decode_calls == 1


    assert stats.speculative_row_cycles == 4
    assert stats.proposed_by_position == (4, 4)
    assert stats.accepted_by_position == (4, 4)
    assert [positions.tolist() for positions in exact_policy.cache_positions] == [
        [0],
        [1, 2],
    ]

    class MixedStopPolicy(ExactPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.nextlat_inputs: list[torch.Tensor] = []

        def nextlat_hidden(self, hidden, token):
            self.nextlat_inputs.append(hidden.clone())
            return super().nextlat_hidden(hidden, token)

    mixed_stop_policy = MixedStopPolicy()
    mixed_stop_engine = NextLatSpeculativeEngine(
        cast(MiniCPMVAPOPolicy, mixed_stop_policy),
        stop_ids=(3,),
        prompts_per_rollout=2,
        samples_per_prompt=1,
        cache_length=8,
        draft_length=2,
        temperature=1.0,
        top_p=1.0,
        compile_decode=False,
    )
    mixed_responses, _, _, _, _, _ = mixed_stop_engine.generate_prompts(
        [torch.tensor([2]), torch.tensor([3])],
        max_new_tokens=4,
    )
    assert mixed_responses.tolist() == [[3, 3, 3, 3], [0, 1, 2, 3]]
    assert mixed_stop_policy.nextlat_inputs[2][0].tolist() == [
        0.0,
        0.0,
        1.0,
        0.0,
    ]

    long_engine = NextLatSpeculativeEngine(
        cast(MiniCPMVAPOPolicy, ExactPolicy()),
        stop_ids=(-1,),
        prompts_per_rollout=1,
        samples_per_prompt=1,
        cache_length=8,
        draft_length=2,
        temperature=1.0,
        top_p=1.0,
        compile_decode=False,
    )
    long_responses, _, _, _, _, long_stats = long_engine.generate_prompts(
        [torch.tensor([0])],
        max_new_tokens=4,
    )
    assert long_responses.tolist() == [[1, 2, 3, 0]]
    assert long_stats.target_decode_calls == 2
    assert long_stats.target_decode_positions == 4
    with pytest.raises(ValueError, match="exceeds the NextLat cache"):
        long_engine.generate_prompts(
            [torch.tensor([0])],
            max_new_tokens=8,
        )

    class WrongPolicy(ExactPolicy):
        def nextlat_hidden(self, _, token):
            wrong = (token + 1).remainder(4)
            return torch.nn.functional.one_hot(
                wrong, num_classes=4
            ).float()

    rejecting_engine = NextLatSpeculativeEngine(
        cast(MiniCPMVAPOPolicy, WrongPolicy()),
        stop_ids=(-1,),
        prompts_per_rollout=1,
        samples_per_prompt=1,
        cache_length=8,
        draft_length=2,
        temperature=1.0,
        top_p=1.0,
        compile_decode=False,
    )
    rejected_responses, _, _, _, _, rejected_stats = (
        rejecting_engine.generate_prompts(
            [torch.tensor([0])],
            max_new_tokens=4,
        )
    )
    assert rejected_responses.tolist() == [[1, 2, 3, 0]]
    assert rejected_stats.target_decode_calls == 3
    assert rejected_stats.accepted_tokens == 1

    class RowDivergentPolicy(ExactPolicy):
        def nextlat_hidden(self, _, token):
            predicted = token.clone()
            predicted[::2] = (predicted[::2] + 1).remainder(4)
            return torch.nn.functional.one_hot(
                predicted, num_classes=4
            ).float()

    divergent_engine = NextLatSpeculativeEngine(
        cast(MiniCPMVAPOPolicy, RowDivergentPolicy()),
        stop_ids=(-1,),
        prompts_per_rollout=2,
        samples_per_prompt=1,
        cache_length=8,
        draft_length=2,
        temperature=1.0,
        top_p=1.0,
        compile_decode=False,
    )
    divergent_responses, _, _, _, _, _ = divergent_engine.generate_prompts(
        [torch.tensor([0]), torch.tensor([1])],
        max_new_tokens=4,
    )
    assert divergent_responses.tolist() == [
        [1, 2, 3, 0],
        [2, 3, 0, 1],
    ]


def test_multi_prompt_preparation_forms_one_left_padded_batch() -> None:
    class Config:
        pad_token_id = 9

    class CausalLM:
        config = Config()

    class Policy:
        causal_lm = CausalLM()

    engine = object.__new__(RolloutEngine)
    engine.policy = cast(MiniCPMVAPOPolicy, Policy())
    engine.prompts_per_rollout = 2
    engine.samples_per_prompt = 2
    engine.batch_size = 4
    engine.cache_length = 8
    engine.generated_buffer = torch.empty(4, 8)
    engine.attention_mask_buffer = torch.zeros(4, 8, dtype=torch.bool)
    engine.prompt_lengths_buffer = torch.empty(4, dtype=torch.long)
    prompt_batch, position_ids, width = engine._prepare_prompts(
        [torch.tensor([1, 2, 3]), torch.tensor([4])]
    )
    assert width == 3
    assert prompt_batch.tolist() == [
        [1, 2, 3],
        [1, 2, 3],
        [9, 9, 4],
        [9, 9, 4],
    ]
    assert position_ids.tolist() == [
        [0, 1, 2],
        [0, 1, 2],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert engine.attention_mask_buffer[:, :3].tolist() == [
        [True, True, True],
        [True, True, True],
        [False, False, True],
        [False, False, True],
    ]
    assert engine.prompt_lengths_buffer.tolist() == [3, 3, 1, 1]

def test_ppo_approximate_kl_is_pointwise_non_negative() -> None:
    log_ratio = torch.tensor([-2.0, -0.1, 0.0, 0.2, 3.0])
    terms = _approximate_kl_terms(log_ratio)
    assert torch.all(terms >= 0)
    assert terms[2] == 0

def test_auxiliary_loss_is_downscaled_without_amplifying_small_losses() -> None:
    large = torch.tensor(20.0, requires_grad=True)
    balanced_large = _scale_auxiliary_loss(large, torch.tensor(0.25))
    assert balanced_large.item() == pytest.approx(0.25)
    balanced_large.backward()
    assert large.grad.item() == pytest.approx(0.0125)

    small = torch.tensor(0.1, requires_grad=True)
    balanced_small = _scale_auxiliary_loss(small, torch.tensor(0.25))
    assert balanced_small.item() == pytest.approx(0.1)
    balanced_small.backward()
    assert small.grad.item() == pytest.approx(1.0)

def test_auxiliary_parameter_gradient_is_capped_to_primary_norm() -> None:
    primary = [torch.tensor([3.0, 4.0])]
    large_auxiliary_probe = [torch.tensor([0.0, 1e-19])]
    capped: list[torch.Tensor | None] = [None]
    _accumulate_balanced_parameter_gradients_(
        capped,
        primary,
        large_auxiliary_probe,
        probe_scale=1e-20,
    )
    assert capped[0] is not None
    torch.testing.assert_close(capped[0], torch.tensor([3.0, 9.0]))

    small_auxiliary_probe = [torch.tensor([0.0, 2e-20])]
    uncapped: list[torch.Tensor | None] = [None]
    _accumulate_balanced_parameter_gradients_(
        uncapped,
        primary,
        small_auxiliary_probe,
        probe_scale=1e-20,
    )
    assert uncapped[0] is not None
    torch.testing.assert_close(uncapped[0], torch.tensor([3.0, 6.0]))




def test_hidden_gradient_balancing_caps_auxiliary_before_lora_backward() -> None:
    torch.testing.assert_close(
        _combine_balanced_hidden_gradients_(
            torch.tensor([3.0, 4.0]),
            torch.tensor([0.0, 10.0]),
        ),
        torch.tensor([3.0, 9.0]),
    )
    base = nn.Linear(4, 4, bias=False)
    layer = LoRALinear(base, LoRAConfig(rank=2, alpha=2.0))
    layer.lora_b.data.fill_(0.1)
    hidden = layer(torch.ones(3, 4))
    primary_hidden = hidden.detach().requires_grad_()
    auxiliary_hidden = hidden.detach().requires_grad_()
    (primary_hidden.square().sum() * 1e-3).backward()
    (auxiliary_hidden.sum() * 1e-2).backward()
    assert primary_hidden.grad is not None
    assert auxiliary_hidden.grad is not None

    hidden.backward(
        _combine_balanced_hidden_gradients_(
            primary_hidden.grad,
            auxiliary_hidden.grad,
        )
    )

    assert layer.lora_a.grad is not None
    assert layer.lora_b.grad is not None
    assert torch.isfinite(layer.lora_a.grad).all()
    assert torch.isfinite(layer.lora_b.grad).all()
    assert torch.linalg.vector_norm(layer.lora_a.grad) > 0
    assert torch.linalg.vector_norm(layer.lora_b.grad) > 0


def test_scaled_primary_gradient_is_restored_before_balancing() -> None:
    accumulator: list[torch.Tensor | None] = [None]
    _accumulate_balanced_parameter_gradients_(
        accumulator,
        [torch.tensor([3e-8, 4e-8])],
        [torch.tensor([0.0, 1e-19])],
        primary_scale=1e-8,
        probe_scale=1e-20,
    )
    assert accumulator[0] is not None
    torch.testing.assert_close(accumulator[0], torch.tensor([3.0, 9.0]))


def test_scaled_storage_is_restored_only_during_final_clip() -> None:
    accumulator: list[torch.Tensor | None] = [None]
    _accumulate_balanced_parameter_gradients_(
        accumulator,
        [torch.tensor([3e-8, 4e-8])],
        [torch.tensor([0.0, 1e-19])],
        primary_scale=1e-8,
        probe_scale=1e-20,
        storage_scale=1e-8,
    )
    assert accumulator[0] is not None
    torch.testing.assert_close(accumulator[0], torch.tensor([3e-8, 9e-8]))
    parameter = nn.Parameter(torch.zeros(2))
    parameter.grad = accumulator[0]
    true_norm = _clip_finite_grad_norm_(
        [parameter],
        1.0,
        label="test",
        gradient_scale=1e-8,
    )
    assert true_norm.item() == pytest.approx(90.0**0.5)
    assert torch.linalg.vector_norm(parameter.grad).item() == pytest.approx(1.0)

def test_auxiliary_head_probe_gradient_is_restored() -> None:
    accumulator: list[torch.Tensor | None] = [None]
    _accumulate_rescaled_parameter_gradients_(
        accumulator,
        [torch.tensor([4e-20])],
        probe_scale=1e-20,
    )
    assert accumulator[0] is not None
    torch.testing.assert_close(accumulator[0], torch.tensor([4.0]))

def test_gradient_clip_accumulates_large_finite_gradients_in_fp64() -> None:
    parameter = nn.Parameter(torch.zeros(4))
    parameter.grad = torch.full_like(parameter, 1e20)
    total_norm = _clip_finite_grad_norm_([parameter], 1.0, label="test")
    assert total_norm.item() == pytest.approx(2e20)
    assert torch.linalg.vector_norm(parameter.grad).item() == pytest.approx(1.0)






def _record(length: int, prompt_length: int, correct: bool = False) -> TrajectoryRecord:
    response = length - prompt_length
    return TrajectoryRecord(
        token_ids=torch.arange(length, dtype=torch.int32),
        prompt_length=prompt_length,
        old_logprobs=torch.linspace(-2, -1, response, dtype=torch.float32),
        advantages=torch.zeros(response, dtype=torch.float32),
        correct=correct,
        text="answer",
    )


def test_replay_plan_and_collation_preserve_variable_length_actions() -> None:
    records = [_record(9, 4, True), _record(6, 3), _record(8, 6)]
    plan = plan_replay_microbatches(
        records, [2, 0, 1], token_budget=16, max_trajectories=2
    )
    assert sorted(index for batch in plan for index in batch) == [0, 1, 2]
    assert all(
        sum(records[index].input_length for index in batch) <= 16
        for batch in plan
    )

    batch = collate_replay_microbatch(
        records,
        [0, 1],
        pad_token_id=99,
        device=torch.device("cpu"),
    )
    assert batch.input_ids.shape == (1, 13)
    assert batch.attention_mask is None
    assert batch.cu_seqlens.tolist() == [0, 8, 13]
    assert batch.sequence_boundaries == (0, 8, 13)
    assert batch.position_ids.tolist() == [
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4]
    ]
    assert batch.sequence_ids.tolist() == [[0] * 8 + [1] * 5]
    assert batch.action_count == records[0].response_length + records[1].response_length
    assert batch.targets.tolist() == [4, 5, 6, 7, 8, 3, 4, 5]
    assert batch.action_positions.tolist() == [3, 4, 5, 6, 7, 10, 11, 12]
    assert batch.response_state_mask.tolist() == [
        [False, False, False, True, True, True, True, True, False, False, True, True, True]
    ]
    assert replay_storage_bytes(records) == sum(record.storage_bytes for record in records)


def test_replay_plan_accepts_optimizer_minibatch_subset() -> None:
    records = [_record(20, 4), _record(6, 3), _record(8, 6)]
    plan = plan_replay_microbatches(
        records, [2, 1], token_budget=14, max_trajectories=2
    )
    assert plan == [(2, 1)]


def test_compiled_replay_copy_uses_replica_weights_and_accepts_checkpoint() -> None:
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.eye(4))
            self.act_fn = nn.SiLU()

        def forward(self, inputs):
            return self.act_fn(inputs @ self.weight)

    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    layer.mlp = MLP()
    model.model.layers = nn.ModuleList([layer])
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    expected = layer.mlp(inputs).detach()
    checkpoint = copy.deepcopy(model.state_dict())
    enable_replay_mlp_compilation(model)
    replica = copy.deepcopy(model)
    with torch.no_grad():
        layer.mlp.weight.fill_(float("nan"))
    replica.load_state_dict(checkpoint, strict=True)
    torch.testing.assert_close(replica.model.layers[0].mlp(inputs), expected)


def test_replay_silu_preserves_native_derivative_across_retained_backward() -> None:
    inputs = torch.linspace(-20, 20, 257, dtype=torch.bfloat16).requires_grad_()
    actual = _ReplaySiLU()(inputs)
    reference = torch.nn.functional.silu(inputs)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for gradient in (torch.ones_like(inputs), inputs.detach() * 0.125):
        actual_gradient = torch.autograd.grad(
            actual, inputs, gradient, retain_graph=True
        )[0]
        reference_gradient = torch.autograd.grad(
            reference, inputs, gradient, retain_graph=True
        )[0]
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("boundaries", [(0, 5), (0, 3, 5)])
def test_segmented_sdpa_packed_replay_has_finite_backward(boundaries) -> None:
    module = SimpleNamespace(
        _packed_sequence_boundaries=boundaries,
        _packed_cu_seqlens=torch.tensor(boundaries, dtype=torch.int32),
    )
    query = torch.randn(1, 4, 5, 8, requires_grad=True)
    key = torch.randn(1, 2, 5, 8, requires_grad=True)
    value = torch.randn(1, 2, 5, 8, requires_grad=True)
    output, weights = _packed_replay_attention(
        cast(nn.Module, module),
        query,
        key,
        value,
        None,
        scaling=0.125,
    )
    expected = torch.cat(
        [
            torch.nn.functional.scaled_dot_product_attention(
                query[:, :, start:stop],
                key[:, :, start:stop],
                value[:, :, start:stop],
                is_causal=True,
                scale=0.125,
                enable_gqa=True,
            )
            for start, stop in zip(boundaries, boundaries[1:])
        ],
        dim=2,
    ).transpose(1, 2)
    torch.testing.assert_close(output, expected)
    assert weights is None
    upstream = torch.randn_like(output)
    actual_gradients = torch.autograd.grad(output, (query, key, value), upstream)
    expected_gradients = torch.autograd.grad(expected, (query, key, value), upstream)
    for actual, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, reference)


def test_packed_replay_attention_preserves_transformers_output_layout() -> None:
    calls: dict[str, object] = {}

    def flash(query, key, value, **kwargs):
        calls.update(kwargs)
        assert query.shape == (5, 4, 8)
        assert key.shape == value.shape == (5, 2, 8)
        return query

    module = SimpleNamespace(
        _packed_flash_varlen=flash,
        _packed_cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.int32),
        _packed_max_sequence_length=3,
    )
    query = torch.randn(1, 4, 5, 8)
    key = torch.randn(1, 2, 5, 8)
    value = torch.randn(1, 2, 5, 8)
    output, weights = _packed_replay_attention_fa4(
        cast(nn.Module, module),
        query,
        key,
        value,
        None,
        scaling=0.125,
    )
    torch.testing.assert_close(output, query.transpose(1, 2))
    assert weights is None
    assert calls["cu_seqlens_q"] is module._packed_cu_seqlens
    assert calls["max_seqlen_q"] == 3
    assert calls["causal"] is True
    assert calls["pack_gqa"] is True

def test_packed_replay_attention_can_yield_actor_to_slow_rollout() -> None:
    model = nn.Module()
    model.config = SimpleNamespace(_attn_implementation="packed")
    model._packed_replay_attention_implementation = "packed"
    use_packed_replay_attention(model, enabled=False)
    assert model.config._attn_implementation == "sdpa"
    use_packed_replay_attention(model, enabled=True)
    assert model.config._attn_implementation == "packed"





def test_replay_plan_rejects_trajectory_above_memory_budget() -> None:
    records = [_record(20, 4)]
    with pytest.raises(ValueError, match="exceeds token budget"):
        plan_replay_microbatches(
            records, [0], token_budget=12, max_trajectories=1
        )


def test_rollout_precomputes_gae_without_replay_time_scan() -> None:
    values = torch.tensor([0.1, -0.2, 0.3, 0.4])
    record = TrajectoryRecord.from_device(
        token_ids=torch.arange(7),
        prompt_length=3,
        old_logprobs=torch.full((4,), -2.0),
        old_values=values,
        correct=True,
        text="answer",
    )
    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    expected, _ = generalized_advantage_estimate(
        rewards,
        values[None],
        torch.ones_like(rewards),
        length_adaptive_lambda(torch.tensor([4.0])),
    )
    assert torch.allclose(record.advantages, expected[0], atol=1e-6)


def test_replay_record_rejects_statistic_length_mismatch() -> None:
    with pytest.raises(ValueError, match="log-probability"):
        TrajectoryRecord(
            token_ids=torch.arange(5, dtype=torch.int32),
            prompt_length=3,
            old_logprobs=torch.zeros(1, dtype=torch.float32),
            advantages=torch.zeros(2, dtype=torch.float32),
            correct=False,
            text="",
        )


def test_static_cache_pool_resets_between_prompt_groups() -> None:
    class Cache:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    created: list[Cache] = []

    def factory() -> Cache:
        cache = Cache()
        created.append(cache)
        return cache

    pool = StaticCachePool(factory, batch_size=4)
    first = pool.acquire(4)
    second = pool.acquire(4)
    assert first is second
    assert len(created) == 1
    assert second.reset_calls == 1
    assert pool.reset_count == 1
    pool.clear()
    third = pool.acquire(4)
    assert third is not first
    assert len(created) == 2
    assert pool.reset_count == 1
    with pytest.raises(ValueError, match="batch size"):
        pool.acquire(2)


def test_replay_checkpointing_selects_uniform_layer_subset() -> None:
    causal_lm = nn.Module()
    causal_lm.model = nn.Module()
    causal_lm.model.layers = nn.ModuleList([nn.Module() for _ in range(6)])
    for layer in causal_lm.model.layers:
        layer.gradient_checkpointing = False

    assert configure_replay_checkpointing(causal_lm, 2) == 3
    assert [
        layer.gradient_checkpointing for layer in causal_lm.model.layers
    ] == [True, False, True, False, True, False]
    assert configure_replay_checkpointing(causal_lm, 0) == 0
    assert not any(
        layer.gradient_checkpointing for layer in causal_lm.model.layers
    )
    with pytest.raises(ValueError, match="cannot be negative"):
        configure_replay_checkpointing(causal_lm, -1)


def test_training_config_bounds_parallel_rollout_context() -> None:
    args = build_parser().parse_args([])
    _validate_args(args)
    args.max_new_tokens = 18_000
    with pytest.raises(ValueError, match="replay token budget"):
        _validate_args(args)
    args = build_parser().parse_args(["--optimizer-minibatches", "1"])
    _validate_args(args)
    args = build_parser().parse_args(["--optimizer-minibatches", "0"])
    with pytest.raises(ValueError, match="positive integer"):
        _validate_args(args)
    args = build_parser().parse_args(
        ["--rollout-physical-batch-size", "65"]
    )
    with pytest.raises(ValueError, match="cannot exceed trajectories"):
        _validate_args(args)
    args = build_parser().parse_args(
        ["--no-fast-rollout", "--rollout-physical-batch-size", "48"]
    )
    with pytest.raises(ValueError, match="require fast rollout"):
        _validate_args(args)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--temperature", "nan"),
        ("--temperature", "inf"),
        ("--clip-high", "nan"),
        ("--actor-lr", "inf"),
    ],
)
def test_training_config_rejects_nonfinite_floats(option: str, value: str) -> None:
    args = build_parser().parse_args([option, value])
    with pytest.raises(ValueError, match="finite float options"):
        _validate_args(args)


def test_resume_dataset_fingerprint_rejects_changed_bytes(tmp_path) -> None:
    data = tmp_path / "dapo.parquet"
    data.write_bytes(b"first")
    resume = {"data_sha256": file_sha256(data)}
    validate_resume_dataset(resume, file_sha256(data))
    data.write_bytes(b"second")
    with pytest.raises(ValueError, match="different dataset bytes"):
        validate_resume_dataset(resume, file_sha256(data))




def test_resume_treats_missing_lora_initialization_as_legacy_standard() -> None:
    prior_args = vars(build_parser().parse_args([]))
    prior_args.pop("lora_initialization")
    resume = {"args": prior_args, "pending_records": None}

    standard_args = build_parser().parse_args(["--lora-initialization", "standard"])
    validate_resume_configuration(resume, standard_args)

    nora_args = build_parser().parse_args(["--lora-initialization", "nora"])
    with pytest.raises(ValueError, match="lora_initialization"):
        validate_resume_configuration(resume, nora_args)


def test_resume_changes_runtime_gate_and_rollout_limit_between_batches() -> None:
    prior_args = build_parser().parse_args([])
    resumed_args = build_parser().parse_args(
        [
            "--max-new-tokens",
            "12000",
            "--min-rollout-tokens-per-second",
            "2800",
            "--logit-chunk-tokens",
            "256",
            "--nextlat-kl-chunk-tokens",
            "32",
            "--nextlat-trunk-balance",
            "hidden",
            "--optimizer-minibatches",
            "1",
        ]
    )
    resume = {"args": vars(prior_args), "pending_records": None}
    resumed_args.actor_lr = 3e-6
    resumed_args.critic_lr = 6e-6
    validate_resume_configuration(resume, resumed_args)

    resume["pending_records"] = [object()]
    with pytest.raises(ValueError, match="max_new_tokens"):
        validate_resume_configuration(resume, resumed_args)

    same_limit_args = build_parser().parse_args(
        ["--optimizer-minibatches", "1"]
    )
    with pytest.raises(ValueError, match="optimizer_minibatches"):
        validate_resume_configuration(resume, same_limit_args)


def test_pending_legacy_ar_resume_rejects_arithmetic_promotion() -> None:
    from postraining.invariant_linear import LEGACY_ARITHMETIC, OPTIMIZED_ARITHMETIC

    resume = {"args": {}, "pending_records": [object()]}
    validate_resume_rollout_arithmetic(resume, LEGACY_ARITHMETIC)
    with pytest.raises(ValueError, match="numerical target"):
        validate_resume_rollout_arithmetic(resume, OPTIMIZED_ARITHMETIC)
    resume["pending_records"] = None
    validate_resume_rollout_arithmetic(resume, OPTIMIZED_ARITHMETIC)


def test_pending_optimized_ar_resume_rejects_arithmetic_demotion() -> None:
    from postraining.invariant_linear import LEGACY_ARITHMETIC, OPTIMIZED_ARITHMETIC

    resume = {
        "args": {"rollout_arithmetic": OPTIMIZED_ARITHMETIC},
        "pending_records": [object()],
    }
    validate_resume_rollout_arithmetic(resume, OPTIMIZED_ARITHMETIC)
    with pytest.raises(ValueError, match="numerical target"):
        validate_resume_rollout_arithmetic(resume, LEGACY_ARITHMETIC)


def test_resume_reasserts_cli_learning_rates_on_every_optimizer_group() -> None:
    actor_parameters = [nn.Parameter(torch.zeros(())) for _ in range(2)]
    critic_parameters = [nn.Parameter(torch.zeros(())) for _ in range(2)]
    actor_optimizer = torch.optim.AdamW(
        [{"params": [parameter]} for parameter in actor_parameters],
        lr=4e-4,
    )
    critic_optimizer = torch.optim.AdamW(
        [{"params": [parameter]} for parameter in critic_parameters],
        lr=4e-4,
    )
    reassert_optimizer_learning_rates(
        actor_optimizer,
        critic_optimizer,
        actor_lr=1e-6,
        critic_lr=2e-6,
    )
    assert {group["lr"] for group in actor_optimizer.param_groups} == {1e-6}
    assert {group["lr"] for group in critic_optimizer.param_groups} == {2e-6}




def test_static_kv_cache_estimate_matches_minicpm_layout() -> None:
    class Config:
        num_hidden_layers = 24
        num_key_value_heads = 2
        head_dim = 128

    expected = 64 * 5_120 * 24 * 2 * 2 * 128 * 2
    assert static_kv_cache_bytes(
        Config(),
        batch_size=64,
        cache_length=5_120,
    ) == expected


def test_device_phase_metrics_keep_only_numeric_samples() -> None:
    class Sampler:
        def window(self, windows, power_floor):
            assert windows == [(1.0, 3.0)]
            assert power_floor == 400.0
            return {
                "power_draw_watts_mean": 512.5,
                "utilization_gpu_percent_readings": 7,
                "withheld": ["clocks_sm_mhz"],
            }

    assert device_phase_metrics(
        cast(DeviceSampler, Sampler()),
        started=1.0,
        ended=3.0,
        power_floor=400.0,
        prefix="rollout_",
    ) == {
        "rollout_device_power_draw_watts_mean": 512.5,
        "rollout_device_utilization_gpu_percent_readings": 7,
    }


def test_group_scoring_trims_stop_tokens_before_compacting_replay() -> None:
    class Tokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens
            return r"Answer: \boxed{42}"

    records, token_count = _build_group_records(
        Tokenizer(),
        {"reward_model": {"ground_truth": "42", "style": "minerva"}},
        torch.tensor([10, 11]),
        torch.tensor([[20, 99, 21], [22, 23, 24]]),
        torch.full((2, 3), -1.5),
        torch.zeros((2, 3)),
        samples_per_prompt=2,
        stop_ids=(99,),
    )
    assert token_count == 5
    assert [record.response_length for record in records] == [2, 3]
    assert [record.token_ids.tolist() for record in records] == [
        [10, 11, 20, 99],
        [10, 11, 22, 23, 24],
    ]
    assert all(record.correct for record in records)


def test_parity_prompt_ids_accept_transformers_batch_encoding() -> None:
    class Batch:
        def __getitem__(self, name):
            assert name == "input_ids"
            return torch.tensor([[1, 2, 3]])

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return Batch()

    assert prompt_ids(
        Tokenizer(), "question", torch.device("cpu")
    ).tolist() == [[1, 2, 3]]


def test_nextlat_shard_samples_are_dense_and_capacity_weighted() -> None:
    torch.manual_seed(5)
    capacities = [3, 17, 0, 80]
    samples = _nextlat_shard_samples(capacities, 64)
    assert sum(samples) == 64
    assert sum(bool(count) for count in samples) == 1
    assert samples[2] == 0
    assert _nextlat_shard_samples([0, 0], 64) == (0, 0)


def test_optimizer_minibatches_are_four_real_disjoint_updates() -> None:
    torch.manual_seed(11)
    minibatches = _optimizer_minibatches(64, 4)
    assert len(minibatches) == 4
    assert [len(indices) for indices in minibatches] == [16, 16, 16, 16]
    assert sorted(index for indices in minibatches for index in indices) == list(
        range(64)
    )


def test_optimizer_minibatches_support_one_full_rollout_update() -> None:
    torch.manual_seed(11)
    minibatches = _optimizer_minibatches(64, 1)
    assert len(minibatches) == 1
    assert sorted(minibatches[0]) == list(range(64))


def test_future_tensorboard_categories_have_at_most_twelve_charts() -> None:
    destinations = [
        *_ROLLOUT_QUALITY_TAGS.values(),
        *_ROLLOUT_PERFORMANCE_TAGS.values(),
        *_ROLLOUT_SAMPLING_TAGS.values(),
        *_ROLLOUT_EFFICIENCY_TAGS.values(),
        *_TRAIN_TAGS.values(),
        *_LIVE_TAGS.values(),
    ]
    category_counts = Counter(tag.split("/", 1)[0] for tag in set(destinations))
    assert max(category_counts.values()) <= 12


def test_tensorboard_scalars_flush_live_cards(tmp_path) -> None:
    writer = SummaryWriter(tmp_path)
    tensorboard_scalars(
        writer,
        "rollout_live",
        {"decode_steps": 256, "non_numeric": "ignored"},
        256,
    )
    writer.close()

    events = EventAccumulator(str(tmp_path))
    events.Reload()
    assert "system_live/decode_steps" in events.Tags()["scalars"]
    point = events.Scalars("system_live/decode_steps")[0]
    assert point.step == 256
    assert point.value == 256


def test_tensorboard_rollout_samples_publish_rewarded_training_text() -> None:
    class Writer:
        def __init__(self) -> None:
            self.text: dict[str, tuple[str, int]] = {}
            self.flushes = 0

        def add_text(self, tag: str, text: str, step: int) -> None:
            self.text[tag] = (text, step)

        def flush(self) -> None:
            self.flushes += 1

    writer = Writer()
    rows = [
        {
            "prompt": [{"content": "What is 2 + 2?"}],
            "reward_model": {"ground_truth": "4"},
        }
    ]
    tensorboard_rollout_samples(
        cast(SummaryWriter, writer),
        "rollout_samples",
        rows,
        [_record(8, 3, True), _record(7, 3, False)],
        samples_per_prompt=2,
        step=12,
    )

    assert set(writer.text) == {
        "samples/rollout_correct",
        "samples/rollout_incorrect",
    }
    correct, step = writer.text["samples/rollout_correct"]
    assert step == 12
    assert "Reward: 1" in correct
    assert "What is 2 + 2?" in correct
    assert "Ground truth:\n4" in correct
    assert "Model response:\nanswer" in correct
    assert writer.flushes == 0


def test_uno_rollout_requires_trained_artifact_and_captured_fast_path(tmp_path) -> None:
    args = build_parser().parse_args(["--uno-rollout"])
    with pytest.raises(ValueError, match="pretrained"):
        _validate_args(args)
    checkpoint = tmp_path / "uno.pt"
    checkpoint.write_bytes(b"identity checked by loader")
    args.uno_checkpoint = str(checkpoint)
    _validate_args(args)
    args.compile_rollout = False
    with pytest.raises(ValueError, match="CUDA-graph"):
        _validate_args(args)
    args.compile_rollout = True
    args.fast_rollout = False
    with pytest.raises(ValueError, match="CUDA-graph"):
        _validate_args(args)
    args.fast_rollout = True
    args.uno_rollout = False
    with pytest.raises(ValueError, match="requires --uno-rollout"):
        _validate_args(args)


def test_resume_can_enable_uno_only_at_completed_rollout_boundary(tmp_path) -> None:
    prior = vars(build_parser().parse_args([]))
    for name in ("uno_rollout", "uno_checkpoint", "uno_block_size"):
        prior.pop(name)
    artifact = tmp_path / "uno.pt"
    artifact.write_bytes(b"trained adapter identity")
    args = build_parser().parse_args(
        ["--uno-rollout", "--uno-checkpoint", str(artifact)]
    )
    resume = {"args": prior, "pending_records": None}
    validate_resume_configuration(resume, args)
    resume["pending_records"] = [object()]
    with pytest.raises(ValueError, match="uno_rollout"):
        validate_resume_configuration(resume, args)


def test_resume_pins_uno_artifact_bytes_not_only_its_path(tmp_path) -> None:
    artifact = tmp_path / "uno.pt"
    artifact.write_bytes(b"first trained adapter")
    args = build_parser().parse_args(
        ["--uno-rollout", "--uno-checkpoint", str(artifact)]
    )
    prior = {**vars(args), "uno_checkpoint_sha256": file_sha256(artifact)}
    resume = {"args": prior, "pending_records": None}
    validate_resume_configuration(resume, args)
    relocated = tmp_path / "relocated.pt"
    relocated.write_bytes(artifact.read_bytes())
    args.uno_checkpoint = str(relocated)
    validate_resume_configuration(resume, args)
    relocated.write_bytes(b"different trained adapter")
    with pytest.raises(ValueError, match="adapter bytes differ"):
        validate_resume_configuration(resume, args)


def test_pending_uno_resume_pins_numerical_target(tmp_path) -> None:
    from postraining.invariant_linear import INVARIANT_ARITHMETIC

    artifact = tmp_path / "uno.pt"
    artifact.write_bytes(b"trained adapter")
    args = build_parser().parse_args(
        ["--uno-rollout", "--uno-checkpoint", str(artifact)]
    )
    prior = {**vars(args), "uno_checkpoint_sha256": file_sha256(artifact)}
    resume = {"args": prior, "pending_records": [object()]}
    with pytest.raises(ValueError, match="numerical target"):
        validate_resume_configuration(resume, args)
    prior["uno_arithmetic"] = INVARIANT_ARITHMETIC
    validate_resume_configuration(resume, args)
    prior["uno_arithmetic"] = "different-arithmetic"
    resume["pending_records"] = None
    validate_resume_configuration(resume, args)
