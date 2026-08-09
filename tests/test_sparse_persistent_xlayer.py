from __future__ import annotations

import copy
import multiprocessing
import tempfile
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xlayer.sparse import sparse_persistent_xlayer_train_gpt as persistent
from scripts.ablation import MetricsWriter, parse_log_line
from train_gpt import dequantize_state_dict_int8, quantize_state_dict_int8


def _encoded(source: torch.Tensor, lag: torch.Tensor, seqlen: int) -> torch.Tensor:
    return source.to(torch.int64) * seqlen + lag


def _ddp_rewire_worker(rank: int, world_size: int, init_file: str, queue) -> None:
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=f"file://{init_file}"
    )
    try:
        module = persistent.PersistentXLayerAttention(32, 4, 2, 10_000.0, 1.5)
        module.k_budget = 8
        module.configure_graph(0, 32)
        before = _encoded(module.wire_source_layer, module.wire_lag, 32)
        module.utility_den.fill_(1.0)
        module.utility_num[:, :4].zero_()
        module.utility_num[:, 4:].fill_(2.0)
        persistent._MODULES = [module]
        persistent._REWIRING_ACTIVE = True
        persistent._TRAIN_SEQ_LEN = 32
        persistent._GRAPH_STATS = None
        persistent.rewire_all_graphs()
        after = _encoded(module.wire_source_layer, module.wire_lag, 32)
        gathered = [torch.empty_like(after) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, after)
        if rank == 0:
            queue.put((
                all(torch.equal(gathered[0], replica) for replica in gathered[1:]),
                bool(torch.all(after[:, :4] != before[:, :4])),
                bool(torch.equal(after[:, 4:], before[:, 4:])),
            ))
    finally:
        torch.distributed.destroy_process_group()


def test_xlayer_initialization_has_exact_prior_quota_and_no_duplicates() -> None:
    source, lag = persistent.initial_templates(3, 2, 128, 256)
    encoded = _encoded(source, lag, 256)
    assert source.shape == (2, 128)
    assert torch.all((source < 3).sum(dim=1) == 32)
    assert torch.all(torch.tensor([row.unique().numel() for row in encoded]) == 128)
    assert torch.all((source >= 0) & (source <= 3))
    assert torch.all((lag >= 0) & (lag < 256))
    for head in range(source.size(0)):
        prior_counts = torch.bincount(source[head][source[head] < 3].to(torch.int64), minlength=3)
        assert int(prior_counts.max() - prior_counts.min()) <= 1


def test_same_layer_initialization_never_uses_prior_layers() -> None:
    source, lag = persistent.initial_templates(3, 2, 128, 256, same_layer=True)
    assert torch.all(source == 3)
    assert torch.all(torch.tensor([row.unique().numel() for row in lag]) == 128)


def test_graph_indices_apply_relative_lags_and_causality() -> None:
    source = torch.tensor([[0, 2, 1]], dtype=torch.int16)
    lag = torch.tensor([[0, 1, 3]], dtype=torch.int32)
    idx, valid = persistent.graph_indices(source, lag, bsz=2, seqlen=4)
    assert idx.shape == valid.shape == (2, 1, 4, 3)
    assert idx[0, 0, 3].tolist() == [3, 10, 4]
    assert valid[0, 0].tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, False],
        [True, True, True],
    ]
    assert torch.equal(idx[0], idx[1])
    assert idx.stride(0) == 0
    assert valid.stride(0) == 0


def test_utility_denominator_retains_invalid_wire_occurrences() -> None:
    # Row 0 has one valid wire -> uniform opportunity 1/2.  Row 1 has two -> 1/3.
    valid = torch.tensor([[[[True, False], [True, True]]]])
    p = torch.tensor([[[[0.5, 0.0], [0.2, 0.3]]]])
    pn = torch.tensor([[[0.5, 0.5]]])
    num, den, support, opportunities, null_sum, rows = persistent.utility_components(p, pn, valid)
    assert torch.allclose(num, torch.tensor([[0.7, 0.3]]))
    assert torch.allclose(den, torch.tensor([[5.0 / 6.0]]))
    assert torch.equal(support, torch.tensor([[2.0, 1.0]]))
    assert torch.equal(opportunities, torch.tensor([[2.0, 1.0]]))
    assert torch.equal(null_sum, torch.tensor([1.0]))
    assert torch.equal(rows, torch.tensor([2.0]))
    assert torch.allclose(num / den[:, None], torch.tensor([[0.84, 0.36]]))


def test_every_below_threshold_slot_changes_to_absent_unique_template() -> None:
    source, lag = persistent.initial_templates(0, 2, 4, 16)
    old = _encoded(source, lag, 16)
    result = persistent.eager_rewire_templates(
        source,
        lag,
        torch.zeros(2, 4),
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
        layer=0,
        seqlen=16,
    )
    new_source, new_lag, cursor, epoch, traversed, rewired = result
    new = _encoded(new_source, new_lag, 16)
    assert rewired.all()
    assert torch.all(new != old)
    for head in range(2):
        assert set(new[head].tolist()).isdisjoint(old[head].tolist())
        assert new[head].unique().numel() == 4
    assert torch.all(traversed >= 4)
    assert torch.all((cursor > 0) | (epoch > 0))


def test_good_slots_are_preserved_and_all_slots_are_eligible() -> None:
    source, lag = persistent.initial_templates(1, 1, 4, 16)
    old = _encoded(source, lag, 16)
    utility = torch.tensor([[1.0, 1.1, 0.999, 0.0]])
    result = persistent.eager_rewire_templates(
        source,
        lag,
        utility,
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        layer=1,
        seqlen=16,
    )
    new = _encoded(result[0], result[1], 16)
    assert torch.equal(new[0, :2], old[0, :2])
    assert torch.all(new[0, 2:] != old[0, 2:])
    assert result[-1].tolist() == [[False, False, True, True]]


def test_traversal_is_a_complete_deterministic_permutation() -> None:
    for epoch in range(3):
        values = [persistent._permuted_candidate(30, 4, 2, epoch, i) for i in range(30)]
        assert len(set(values)) == 30
        assert sorted(values) == list(range(30))
        assert values == [persistent._permuted_candidate(30, 4, 2, epoch, i) for i in range(30)]


def test_rewire_accepts_only_installed_wire_utilities() -> None:
    source, lag = persistent.initial_templates(0, 1, 4, 16)
    with pytest.raises(ValueError, match="installed"):
        persistent.eager_rewire_templates(
            source,
            lag,
            torch.zeros(1, 16),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            layer=0,
            seqlen=16,
        )


def test_graph_buffers_serialize_but_utility_accumulators_do_not() -> None:
    module = persistent.PersistentXLayerAttention(32, 4, 2, 10_000.0, 1.5)
    module.k_budget = 8
    module.configure_graph(2, 32)
    module.bfloat16()
    assert module.utility_num.dtype == torch.float32
    assert module.utility_den.dtype == torch.float32
    module.wire_cursor.add_(3)
    module.utility_num.add_(7)
    state = copy.deepcopy(module.state_dict())
    assert "wire_source_layer" in state
    assert "wire_lag" in state
    assert "wire_cursor" in state
    assert "utility_num" not in state

    restored = persistent.PersistentXLayerAttention(32, 4, 2, 10_000.0, 1.5)
    restored.k_budget = 8
    restored.configure_graph(2, 32)
    restored.load_state_dict(state, strict=True)
    assert torch.equal(restored.wire_source_layer, module.wire_source_layer)
    assert torch.equal(restored.wire_lag, module.wire_lag)
    assert torch.equal(restored.wire_cursor, module.wire_cursor)

    quantized, _ = quantize_state_dict_int8(state)
    roundtrip = dequantize_state_dict_int8(quantized)
    assert torch.equal(roundtrip["wire_source_layer"], state["wire_source_layer"])
    assert torch.equal(roundtrip["wire_lag"], state["wire_lag"])
    assert torch.equal(roundtrip["wire_cursor"], state["wire_cursor"])


def test_graph_stats_parse_and_route_to_graph_namespace() -> None:
    entry = parse_log_line(
        "graph_stats graph_rewire_l0:0.75 graph_support_l0:0.25 graph_utility_l0:1.5"
    )
    assert entry == {
        "type": "graph_stats",
        "graph_rewire_l0": 0.75,
        "graph_support_l0": 0.25,
        "graph_utility_l0": 1.5,
    }
    assert MetricsWriter._extra_scalar_tag("graph_rewire_l0") == "graph/rewire_l0"
    assert MetricsWriter._extra_scalar_tag("graph_support_l0") == "graph/support_l0"
    assert MetricsWriter._extra_scalar_tag("nextlat_kl") == "nextlat/kl"
    assert MetricsWriter._extra_scalar_tag("train_ce") == "train/ce"


def test_muon_wrapper_updates_graph_once_per_optimizer_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def optimizer_step(_self) -> str:
        calls.append("optimizer")
        return "result"

    def graph_step() -> None:
        calls.append("graph")

    monkeypatch.setattr(persistent, "rewire_all_graphs", graph_step)
    wrapped = persistent._wrap_muon_step(optimizer_step)
    assert wrapped(object()) == "result"
    assert calls == ["optimizer", "graph"]


def test_eight_microbatches_feed_one_graph_update(monkeypatch: pytest.MonkeyPatch) -> None:
    module = persistent.PersistentXLayerAttention(32, 4, 2, 10_000.0, 1.5)
    module.k_budget = 8
    module.configure_graph(0, 32)
    valid = torch.ones(1, 4, 2, 8, dtype=torch.bool)
    p = torch.zeros(1, 4, 2, 8)
    pn = torch.ones(1, 4, 2)
    for _ in range(8):
        num, den, support, opportunities, null_sum, rows = persistent.utility_components(p, pn, valid)
        module.utility_num.add_(num)
        module.utility_den.add_(den)
        module.support_num.add_(support)
        module.valid_num.add_(opportunities)
        module.null_sum.add_(null_sum)
        module.row_count.add_(rows)
    assert torch.allclose(module.utility_den, torch.full((4,), 16.0 / 9.0))
    assert module.wire_traversed.sum() == 0

    monkeypatch.setattr(persistent, "_MODULES", [module])
    monkeypatch.setattr(persistent, "_REWIRING_ACTIVE", True)
    monkeypatch.setattr(persistent, "_TRAIN_SEQ_LEN", 32)
    monkeypatch.setattr(persistent, "_GRAPH_STATS", None)
    persistent.rewire_all_graphs()
    traversed_after_step = module.wire_traversed.clone()
    assert torch.all(traversed_after_step >= module.k_budget)
    assert module.utility_den.sum() == 0

    # No new microbatch evidence means no second topology update.
    persistent.rewire_all_graphs()
    assert torch.equal(module.wire_traversed, traversed_after_step)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed required")
def test_ddp_rewire_broadcasts_identical_graphs() -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as rendezvous:
        init_file = rendezvous.name
    processes = [
        context.Process(target=_ddp_rewire_worker, args=(rank, 2, init_file, queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert queue.get(timeout=5) == (True, True, True)


def test_graph_buffer_value_change_does_not_recompile() -> None:
    class BufferReader(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("wire", torch.tensor([1.0]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.wire

    from torch._dynamo.testing import CompileCounter

    counter = CompileCounter()
    module = BufferReader()
    compiled = torch.compile(module, backend=counter, fullgraph=True)
    assert compiled(torch.tensor([2.0])).item() == 3.0
    module.wire.copy_(torch.tensor([4.0]))
    assert compiled(torch.tensor([2.0])).item() == 6.0
    assert counter.frame_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/Triton required")
def test_actual_graph_and_custom_op_value_change_does_not_recompile() -> None:
    from xlayer.sparse.sparse_xlayer_kernel import xlayer_entmax_attention_stats
    from torch._dynamo.testing import CompileCounter

    class SparseGraphReader(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            source, lag = persistent.initial_templates(0, 2, 8, 16)
            self.register_buffer("source", source.cuda())
            self.register_buffer("lag", lag.cuda())

        def forward(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, null: torch.Tensor
        ) -> torch.Tensor:
            idx, valid = persistent.graph_indices(self.source, self.lag, q.size(0), q.size(2))
            return xlayer_entmax_attention_stats(q, [k], [v], idx, valid, null, 0.25)[0]

    counter = CompileCounter()
    module = SparseGraphReader()
    compiled = torch.compile(module, backend=counter, fullgraph=True)
    q = torch.randn(2, 2, 16, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 1, 16, 16, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 1, 16, 16, device="cuda", dtype=torch.bfloat16)
    null = torch.zeros(2, device="cuda")
    first = compiled(q, k, v, null)
    module.lag[0, 0].copy_((module.lag[0, 0] + 1) % 16)
    second = compiled(q, k, v, null)
    assert counter.frame_count == 1
    assert not torch.equal(first, second)


def test_warmup_suppression_clears_evidence_without_mutating_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    module = persistent.PersistentXLayerAttention(32, 4, 2, 10_000.0, 1.5)
    module.k_budget = 8
    module.configure_graph(0, 32)
    module.utility_den.fill_(1)
    module.utility_num.zero_()
    before = tuple(t.clone() for t in (
        module.wire_source_layer, module.wire_lag, module.wire_cursor,
        module.wire_epoch, module.wire_traversed,
    ))
    monkeypatch.setattr(persistent, "_MODULES", [module])
    monkeypatch.setattr(persistent, "_REWIRING_ACTIVE", False)
    persistent.rewire_all_graphs()
    after = (
        module.wire_source_layer, module.wire_lag, module.wire_cursor,
        module.wire_epoch, module.wire_traversed,
    )
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
    assert module.utility_num.sum() == 0
    assert module.utility_den.sum() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/Triton required")
def test_eval_forward_keeps_graph_and_utility_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(persistent, "_TRAIN_SEQ_LEN", 16)
    module = persistent.PersistentXLayerAttention(32, 4, 2, 10_000.0, 1.5)
    module.k_budget = 8
    module.configure_graph(0, 16)
    module = module.cuda().bfloat16().eval()
    before = tuple(t.clone() for t in (
        module.wire_source_layer,
        module.wire_lag,
        module.wire_cursor,
        module.wire_epoch,
        module.wire_traversed,
        module.utility_num,
        module.utility_den,
    ))
    persistent._STATE["i"] = 0
    persistent._STATE["kv"] = None
    with torch.inference_mode():
        output = module(torch.randn(1, 16, 32, device="cuda", dtype=torch.bfloat16))
    assert output.shape == (1, 16, 32)
    after = (
        module.wire_source_layer,
        module.wire_lag,
        module.wire_cursor,
        module.wire_epoch,
        module.wire_traversed,
        module.utility_num,
        module.utility_den,
    )
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/Triton required")
def test_xlayer_list_wrapper_forward_backward_matches_flat_bank() -> None:
    from xlayer.sparse.sparse_entmax_kernel import sparse_entmax_attention_stats
    from xlayer.sparse.sparse_xlayer_kernel import xlayer_entmax_attention_stats

    torch.manual_seed(7)
    device = torch.device("cuda")
    bsz, heads, kv_heads, seqlen, dim, kb = 1, 2, 1, 16, 16, 8
    q1 = torch.randn(bsz, heads, seqlen, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    ks1 = [
        torch.randn(bsz, kv_heads, seqlen, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        for _ in range(2)
    ]
    vs1 = [
        torch.randn(bsz, kv_heads, seqlen, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        for _ in range(2)
    ]
    q2 = q1.detach().clone().requires_grad_(True)
    k2 = torch.cat([k.detach() for k in ks1], dim=2).requires_grad_(True)
    v2 = torch.cat([v.detach() for v in vs1], dim=2).requires_grad_(True)
    source, lag = persistent.initial_templates(1, heads, kb, seqlen)
    idx, valid = persistent.graph_indices(source.to(device), lag.to(device), bsz, seqlen)
    null1 = torch.zeros(heads, device=device, requires_grad=True)
    null2 = null1.detach().clone().requires_grad_(True)

    out1 = xlayer_entmax_attention_stats(q1, ks1, vs1, idx, valid, null1, dim**-0.5)
    out2 = sparse_entmax_attention_stats(q2, k2, v2, idx, valid, null2, dim**-0.5)
    for lhs, rhs in zip(out1, out2, strict=True):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
    out1[0].float().sum().backward()
    out2[0].float().sum().backward()
    torch.testing.assert_close(q1.grad, q2.grad, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([k.grad for k in ks1], dim=2), k2.grad, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([v.grad for v in vs1], dim=2), v2.grad, rtol=0, atol=0)
    torch.testing.assert_close(null1.grad, null2.grad, rtol=0, atol=0)
