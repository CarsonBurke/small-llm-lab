"""Shared-pool dispatch, routing, initialization and gate contracts; no CUDA execution."""
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini.gated_delta_pool import (
    BALANCE_COEFFICIENT, POOL_PASSES, POOL_READS, Z_COEFFICIENT, SharedPoolGatedDeltaGPT, balance_loss,
    gather_events, pool_layout, pool_row_chunks, resolve_pool_options, scatter_events, z_loss,
)

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = dict(gdn_backend="fla", disable_recompute=True, custom_ops=True, fused_projections=True)


def test_packed_layout_places_each_bank_in_its_own_chunk_aligned_segment_bank_major():
    route = torch.tensor([[0, 2, 0, 1, 0, 2, 2, 1], [1, 1, 1, 1, 0, 0, 0, 0]])
    banks, chunk = 3, 2
    assert pool_row_chunks(8, banks, chunk) == 7
    layout = pool_layout(route, banks, chunk)
    assert layout.chunks == 14 and layout.rows == 28
    # Segments in bank-major order (bank, sequence): bank 0 holds sequence 0's events 0, 2, 4 (chunks 0-1)
    # and sequence 1's events 4-7 (chunks 2-3); bank 1 holds sequence 0's events 3, 7 (chunk 4) and
    # sequence 1's events 0-3 (chunks 5-6); bank 2 holds sequence 0's events 1, 5, 6 (chunks 7-8) and
    # nothing of sequence 1 (an empty segment); the filler takes chunks 9-13.
    assert layout.slots.tolist() == [[0, 14, 1, 8, 2, 15, 16, 9], [10, 11, 12, 13, 4, 5, 6, 7]]
    assert layout.cu_seqlens.tolist() == [0, 4, 8, 10, 14, 18, 18, 28]
    assert layout.cu_seqlens.dtype == layout.chunk_indices.dtype == layout.tile_bank.dtype == torch.int32
    assert layout.bank_tiles.dtype == torch.int32
    assert layout.chunk_indices.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [3, 0], [3, 1], [4, 0], [4, 1],
                                             [6, 0], [6, 1], [6, 2], [6, 3], [6, 4]]
    assert layout.tile_bank.tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
    assert layout.bank_tiles.tolist() == [0, 4, 7, 9]
    with pytest.raises(ValueError):
        pool_layout(route.to(torch.int32), banks, chunk)
    with pytest.raises(ValueError):
        pool_row_chunks(7, banks, chunk)


@pytest.mark.parametrize("seed", range(4))
def test_packed_layout_invariants_hold_for_random_routes(seed):
    torch.manual_seed(seed)
    batch, events, banks, chunk = 3, 64, 5, 8
    route = torch.randint(banks, (batch, events))
    if seed == 0:
        route[0] = 2  # one bank takes a whole sequence: the filler still gets its chunks
    layout = pool_layout(route, banks, chunk)
    chunks, rows = layout.chunks, layout.rows
    assert chunks == batch * (events // chunk + banks) and rows == chunks * chunk
    slots = layout.slots
    assert slots.min() >= 0 and slots.max() < rows and slots.flatten().unique().numel() == slots.numel()
    cu = layout.cu_seqlens.tolist()
    assert cu == sorted(cu) and cu[0] == 0 and cu[-1] == rows and len(cu) == batch * banks + 2
    assert all(offset % chunk == 0 for offset in cu), "segments are chunk-aligned"
    assert cu[-1] - cu[-2] >= batch * chunk, "at least one filler chunk per sequence"
    for bank in range(banks):
        for b in range(batch):
            segment = bank * batch + b
            members = (route[b] == bank).nonzero().flatten()
            assert cu[segment + 1] - cu[segment] == chunk * ((members.numel() + chunk - 1) // chunk)
            bank_slots = slots[b, members]
            assert bank_slots.tolist() == list(range(cu[segment], cu[segment] + members.numel())), "event order, packed"
    indices = layout.chunk_indices.tolist()
    assert len(indices) == chunks
    filler = cu[-2] // chunk
    for c, (segment, within) in enumerate(indices):
        assert cu[segment] + within * chunk == c * chunk, "chunk lies at its offset within its segment"
        assert cu[segment + 1] > cu[segment] + within * chunk, "chunk lies inside a nonempty segment"
        expected_bank = segment // batch if c < filler else banks - 1
        assert layout.tile_bank[c] == expected_bank
    tiles = layout.bank_tiles.tolist()
    assert tiles[0] == 0 and tiles[-1] == filler and tiles == sorted(tiles) and len(tiles) == banks + 1
    for bank in range(banks):
        assert (layout.tile_bank[tiles[bank]:tiles[bank + 1]] == bank).all(), "a bank's tiles are contiguous"


def test_scatter_gather_round_trip_and_padding_rows_stay_zero():
    torch.manual_seed(5)
    batch, events, heads, dim, banks, chunk = 2, 8, 2, 3, 3, 2
    route = torch.tensor([[0, 2, 0, 1, 0, 2, 2, 1], [1, 1, 1, 1, 0, 0, 0, 0]])
    layout = pool_layout(route, banks, chunk)
    slots, rows = layout.slots, layout.rows
    field = torch.randn(batch, events, heads, dim, requires_grad=True)
    pool = scatter_events([field], slots.unsqueeze(-1), rows)
    assert pool.shape == (rows, heads * dim) and pool.is_contiguous()
    # Every event sits at its row; padding rows are zero (no-op events).
    filled = torch.zeros(rows, dtype=torch.bool)
    for b in range(batch):
        for e in range(events):
            torch.testing.assert_close(pool[slots[b, e]], field[b, e].flatten())
            filled[slots[b, e]] = True
    assert filled.sum() == batch * events and (pool[~filled] == 0).all()
    read, = gather_events(pool.view(rows, heads, dim), slots.unsqueeze(-1))
    assert read.shape == (batch, events, heads, dim)
    torch.testing.assert_close(read, field)
    upstream = torch.randn_like(read)
    (read * upstream).sum().backward()
    torch.testing.assert_close(field.grad, upstream)
    # One field scattered to two slot columns (a query read from two banks) receives both gradients.
    field.grad = None
    twice = scatter_events([field, field], slots.unsqueeze(-1).expand(batch, events, 2).contiguous(), rows)
    twice.sum().backward()
    torch.testing.assert_close(field.grad, torch.full_like(field, 2.0))


def test_layerwise_dispatch_matches_a_dense_reference_and_traces_whole():
    """Layout, per-layer scatter and gather trace as one graph with eager-equal gradients."""
    torch.manual_seed(9)
    batch, time, layers, heads, dim, banks, chunk = 2, 4, 3, 2, 3, 2, 4
    route = torch.randint(banks, (batch, time * layers))
    fields = [torch.randn(batch, time, heads, dim, requires_grad=True) for _ in range(layers)]

    def dispatch(route, fields):
        layout = pool_layout(route, banks, chunk)
        layer_slots = layout.slots.view(batch, time, layers)
        pool = scatter_events(fields, layer_slots, layout.rows)
        return layout, pool, gather_events(pool.view(layout.rows, heads, dim), layer_slots)

    layout, pool, reads = dispatch(route, fields)
    reference = torch.zeros(layout.rows, heads * dim)
    for b in range(batch):
        for t_ in range(time):
            for layer in range(layers):
                reference[layout.slots[b, t_ * layers + layer]] = fields[layer][b, t_].flatten()
    torch.testing.assert_close(pool, reference)
    for layer, read in enumerate(reads):
        torch.testing.assert_close(read, fields[layer].detach())
    upstream = [torch.randn_like(read) for read in reads]
    sum((read * up).sum() for read, up in zip(reads, upstream)).backward()
    for layer, field in enumerate(fields):
        torch.testing.assert_close(field.grad, upstream[layer])
    eager_grads = [field.grad.clone() for field in fields]
    for field in fields:
        field.grad = None
    compiled = torch.compile(dispatch, backend="aot_eager", fullgraph=True, dynamic=False)
    layout_c, pool_c, reads_c = compiled(route, fields)
    torch.testing.assert_close(pool_c, reference)
    for name in ("slots", "cu_seqlens", "chunk_indices", "tile_bank", "bank_tiles"):
        assert torch.equal(getattr(layout_c, name), getattr(layout, name)), name
    sum((read * up).sum() for read, up in zip(reads_c, upstream)).backward()
    for field, grad in zip(fields, eager_grads):
        torch.testing.assert_close(field.grad, grad)


def test_balance_loss_is_one_when_uniform_and_grows_with_concentration():
    banks = 4
    reads = torch.stack((torch.arange(64).remainder(banks), (torch.arange(64) + 1).remainder(banks)), -1).view(2, 32, 2)
    probs = torch.full((2, 32, banks), 1 / banks)
    assert balance_loss(probs, reads, banks).item() == pytest.approx(1.0)
    peaked = F.one_hot(reads[..., 0], banks).float() * .7 + .1
    # f_j = 1/4 (every bank is selected equally often) and P_j = 0.7/4 + 0.1: 4 * sum_j f_j P_j = 1.1.
    assert balance_loss(peaked, reads, banks).item() == pytest.approx(1.1, rel=1e-6)
    # Selections and probability mass concentrating on banks 0 and 1: f = (1/2, 1/2, 0, 0), P = (0.45, 0.45, ...).
    collapsed = torch.zeros_like(reads)
    collapsed[..., 1] = 1
    concentrated = torch.zeros(2, 32, banks)
    concentrated[..., :2] = .45
    concentrated[..., 2:] = .05
    assert balance_loss(concentrated, collapsed, banks).item() == pytest.approx(4 * (.5 * .45 + .5 * .45), rel=1e-6)
    live = peaked.clone().requires_grad_()
    balance_loss(live, reads, banks).backward()
    assert live.grad is not None and live.grad.abs().sum() > 0
    logits = torch.zeros(2, 32, banks)
    assert z_loss(logits).item() == pytest.approx(math.log(banks) ** 2)
    assert z_loss(logits + 3).item() == pytest.approx((math.log(banks) + 3) ** 2, rel=1e-6)


@pytest.mark.parametrize("bad", [
    dict(custom_ops=False, gdn_backend="fla", disable_recompute=True, fused_projections=True),
    dict(fused_projections=False, gdn_backend="fla", disable_recompute=True, custom_ops=True),
    dict(gate_in_kernel=True, **PRODUCTION),
    dict(allow_neg_eigval=True, **PRODUCTION),
    dict(pool_banks=1, **PRODUCTION), dict(pool_banks=True, **PRODUCTION),
    dict(pool_writer="top2", **PRODUCTION),
    dict(pool_writer="layer", pool_banks=3, **PRODUCTION),   # one bank per layer, and the model has two
    dict(pool_heads=0, **PRODUCTION), dict(pool_heads=True, **PRODUCTION),
    dict(pool_heads=1, **PRODUCTION),                          # latent width 32 is not a multiple of 64
])
def test_pool_rejects_non_production_or_unchunked_configurations(bad):
    with pytest.raises(ValueError):
        SharedPoolGatedDeltaGPT(vocab_size=32, num_layers=2, model_dim=64, head_dim=32, **bad)
    # The retired static capacity is an unknown option, like any other key the private model lacks.
    with pytest.raises(TypeError):
        SharedPoolGatedDeltaGPT(vocab_size=32, num_layers=2, model_dim=64, head_dim=32, pool_capacity=64, **PRODUCTION)


def test_pool_option_resolution_records_the_fixed_policy():
    routed = resolve_pool_options(None, "routed", 2, num_layers=6, head_dim=128)
    assert routed == dict(shared_pool=True, pool_banks=12, pool_writer="routed", pool_heads=2, pool_reads=POOL_READS,
                          pool_passes=POOL_PASSES, pool_balance_coefficient=BALANCE_COEFFICIENT,
                          pool_z_coefficient=Z_COEFFICIENT)
    assert resolve_pool_options(None, "layer", 2, num_layers=6, head_dim=128)["pool_banks"] == 6
    assert resolve_pool_options(8, "routed", 4, num_layers=6, head_dim=128)["pool_banks"] == 8
    with pytest.raises(ValueError):
        resolve_pool_options(12, "layer", 2, num_layers=6, head_dim=128)


def small(**overrides):
    torch.manual_seed(11)
    options = dict(vocab_size=32, num_layers=2, model_dim=64, head_dim=32, pool_banks=3, **PRODUCTION)
    return SharedPoolGatedDeltaGPT(**dict(options, **overrides))


def pool_parameter_names(blocks, banks):
    per_block = ("router.weight", "pool_gate_proj.0.weight", "pool_gate_proj.1.weight", "pool_gate_proj.1.bias",
                 "pool_norm.weight", "pool_o_proj.weight")
    shared = [f"pool.{n}_proj.weight" for n in "qkvbw"] + ["pool.f_proj.0.weight", "pool.f_proj.1.weight"]
    shared += [f"pool.{n}_conv1d.weight" for n in "qkv"] + ["pool.A_log", "pool.dt_bias"]
    shared += [f"adapters.{n}" for n in ("query", "key", "value", "output")]
    return {f"blocks.{i}.{m}" for i in range(blocks) for m in per_block} | set(shared)


@pytest.mark.parametrize("writer", ["routed", "layer"])
def test_pool_configuration_initialization_and_parameter_ownership(writer):
    from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaGPT
    banks = 3 if writer == "routed" else 2
    net = small(pool_writer=writer, pool_banks=3 if writer == "routed" else None)
    assert net.config["shared_pool"] and net.config["pool_banks"] == banks and net.config["pool_writer"] == writer
    assert net.config["pool_heads"] == 2 and net.config["pool_reads"] == POOL_READS == 2
    assert net.config["pool_passes"] == POOL_PASSES == 2 and net.config["pool_balance_coefficient"] == BALANCE_COEFFICIENT
    assert net.config["pool_z_coefficient"] == Z_COEFFICIENT
    assert net.events_per_layer == (2 if writer == "routed" else 3)
    assert net.read_events == ([0, 1] if writer == "routed" else [1, 2])
    private = {k: v for k, v in net.config.items() if not k.startswith("pool") and k != "shared_pool"}
    plain = GatedDeltaGPT(**private)
    assert plain.config == private
    latent = 64
    for block in net.blocks:
        assert block.pool_writer == writer
        assert block.pool_o_proj.weight.abs().sum() == 0 and block.pool_o_proj.weight.shape == (64, latent)
        assert (block.pool_norm.weight == 1).all() and block.pool_norm.weight.shape == (32,)
        assert block.pool_gate_proj[1].bias.abs().sum() == 0
        assert 0 < block.router.weight.abs().max() <= 2 ** -2.5 * math.sqrt(6 / (64 + banks))
        assert 0 < block.pool_gate_proj[0].weight.abs().max() <= 2 ** -2.5 * math.sqrt(6 / (64 + 32))
    pool = net.pool
    for name in "qkvbw":
        weight = getattr(pool, f"{name}_proj").weight
        assert weight.shape == (latent, 64) and 0 < weight.abs().max() <= 2 ** -2.5 * math.sqrt(6 / (64 + latent))
    assert pool.q_conv1d.weight.shape == (latent, 1, 4) and pool.q_conv1d.weight.abs().sum() > 0
    assert pool.A_log.shape == (2,) and (pool.A_log >= 0).all() and (pool.A_log <= math.log(16)).all()
    assert pool.dt_bias.shape == (latent,) and torch.isfinite(pool.dt_bias).all()
    assert pool.A_log._no_weight_decay and pool.dt_bias._no_weight_decay
    adapters = net.adapters
    eye = torch.eye(latent).expand(banks, latent, latent)
    for name in ("key", "value"):
        matrix = getattr(adapters, name)
        assert matrix.shape == (banks, latent, latent)
        torch.testing.assert_close(matrix @ matrix.mT, eye, atol=1e-5, rtol=0)
    assert torch.equal(adapters.query, adapters.key) and torch.equal(adapters.output, adapters.value.mT)
    assert not torch.equal(adapters.key[0], adapters.key[1]), "banks start as different bases"
    names = {name for name, _ in net.named_parameters()}
    plain_names = {name for name, _ in plain.named_parameters()}
    assert plain_names < names
    assert names - plain_names == pool_parameter_names(2, banks)
    # The private model's state dict loads into the pool model unchanged (the GPU parity test uses this).
    missing, unexpected = net.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and set(missing) == names - plain_names


def test_pool_private_parameters_are_seed_matched_to_the_standalone_model():
    """Pool modules are built after the backbone, so the control's initialization is reproduced exactly."""
    from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaGPT
    net = small()
    private = {k: v for k, v in net.config.items() if not k.startswith("pool") and k != "shared_pool"}
    torch.manual_seed(11)
    plain = GatedDeltaGPT(**private)
    pool_state = net.state_dict()
    for name, value in plain.state_dict().items():
        assert torch.equal(pool_state[name], value), name


def test_pool_config_rebuilds_the_pool_model_and_never_the_private_one():
    from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaGPT, build_gated_delta_model
    net = small()
    rebuilt = SharedPoolGatedDeltaGPT(**net.config)
    assert rebuilt.config == net.config
    assert rebuilt.load_state_dict(net.state_dict(), strict=True)
    dispatched = build_gated_delta_model(net.config)
    assert isinstance(dispatched, SharedPoolGatedDeltaGPT) and dispatched.config == net.config
    # The class bodies stay intact around the builder: every execution entry point remains a method.
    for cls in (GatedDeltaGPT, SharedPoolGatedDeltaGPT):
        assert all(callable(getattr(cls, m)) for m in ("forward_hidden", "logits", "step", "new_cache"))
    assert callable(SharedPoolGatedDeltaGPT.forward_passes)
    private = {k: v for k, v in net.config.items() if not k.startswith("pool") and k != "shared_pool"}
    assert isinstance(build_gated_delta_model(private), GatedDeltaGPT)
    assert not build_gated_delta_model(private).config.get("shared_pool")
    with pytest.raises(TypeError):
        GatedDeltaGPT(**net.config)
    for key, value in (("pool_passes", 3), ("pool_balance_coefficient", 0.02), ("shared_pool", False),
                       ("pool_reads", 3), ("pool_z_coefficient", 0.1)):
        with pytest.raises(ValueError):
            SharedPoolGatedDeltaGPT(**dict(net.config, **{key: value}))


def test_pool_optimizer_groups_route_routers_to_slow_adamw_and_matrices_to_muon():
    from pretraining.nanogpt_mini.gated_delta_runtime import build_gated_delta_optimizers
    from scripts.train_recurrent_slots import Muon
    net = small()
    adam, muon = build_gated_delta_optimizers(net)
    assert isinstance(muon, Muon)
    by_id = {id(p): name for name, p in net.named_parameters()}
    groups = {(round(g["lr"], 6), g.get("weight_decay", 0.001)): [by_id[id(p)] for p in g["params"]]
              for g in adam.param_groups}
    assert set(groups[(0.001, 0.0)]) == {"blocks.0.router.weight", "blocks.1.router.weight"}
    assert {"pool.q_conv1d.weight", "pool.v_conv1d.weight", "blocks.0.attn.q_conv1d.weight"} <= set(groups[(0.002, 0.001)])
    assert {"pool.A_log", "pool.dt_bias", "blocks.0.attn.A_log"} <= set(groups[(0.002, 0.0)])
    assert {"blocks.0.pool_norm.weight", "blocks.0.pool_gate_proj.1.bias", "blocks.0.attn.o_norm.weight"} <= set(groups[(0.015, 0.001)])
    matrices = {by_id[id(p)] for g in muon.param_groups for p in g["params"]}
    assert {"pool.q_proj.weight", "pool.f_proj.1.weight", "blocks.0.pool_o_proj.weight", "blocks.0.pool_gate_proj.0.weight",
            "adapters.query", "adapters.key", "adapters.value", "adapters.output", "blocks.0.attn.o_proj.weight"} <= matrices
    from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaGPT
    plain = GatedDeltaGPT(vocab_size=32, num_layers=1, model_dim=64, head_dim=32, **PRODUCTION)
    plain_adam, _ = build_gated_delta_optimizers(plain)
    assert all(round(g["lr"], 6) != 0.001 for g in plain_adam.param_groups)


def test_muon_update_is_batched_over_stacked_adapters():
    """Each stacked matrix is orthogonalized on its own: slices neither mix nor depend on each other."""
    from scripts.train_recurrent_slots import muon_update
    torch.manual_seed(3)
    gradient = torch.randn(3, 8, 8)
    stacked = muon_update(gradient, torch.zeros_like(gradient))
    perturbed = gradient.clone()
    perturbed[1:] *= -2
    assert torch.equal(muon_update(perturbed, torch.zeros_like(gradient))[0], stacked[0])
    eye = torch.eye(8).expand(3, 8, 8)
    torch.testing.assert_close((stacked @ stacked.mT).float(), eye, atol=.1, rtol=0)  # bf16 Newton-Schulz
    single = torch.stack([muon_update(gradient[i], torch.zeros(8, 8)) for i in range(3)])
    torch.testing.assert_close(stacked.float(), single.float(), atol=.05, rtol=0)


def test_pool_loss_wrapper_selection_and_cpu_rejection():
    from pretraining.nanogpt_mini.gated_delta_runtime import (
        CompiledGatedDeltaLoss, CompiledSharedPoolLoss, compiled_gated_delta_loss)
    net = small()
    assert type(compiled_gated_delta_loss(net, 64)) is CompiledSharedPoolLoss
    assert issubclass(CompiledSharedPoolLoss, CompiledGatedDeltaLoss)
    with pytest.raises(ValueError, match="CUDA"):
        net.forward_passes(torch.zeros(1, 64, dtype=torch.int32))
    with pytest.raises(NotImplementedError):
        net.forward_hidden(torch.zeros(1, 64, dtype=torch.int32), use_cache=True)


@pytest.mark.parametrize("writer,banks,expected", [("routed", 12, 30_027_034), ("layer", 6, 28_435_738)])
def test_production_pool_parameter_count(writer, banks, expected):
    torch.manual_seed(0)
    net = SharedPoolGatedDeltaGPT(pool_writer=writer, **PRODUCTION)
    private = 24_708_888
    latent = 256
    per_layer = 512 * banks + (512 * 128 + 128 * latent + latent) + 128 + latent * 512
    shared = 5 * 512 * latent + (512 * 128 + 128 * latent) + 3 * latent * 4 + 2 + latent
    adapters = 4 * banks * latent * latent
    assert sum(p.numel() for p in net.parameters()) == private + 6 * per_layer + shared + adapters == expected


def test_gate_decision_compares_published_precision_at_update_400():
    from scripts.train_recurrent_slots import GATE_STEP, MINIMUM_IMPROVEMENT, gate_decision
    assert GATE_STEP == 400 and float(MINIMUM_IMPROVEMENT) == 0.005
    assert gate_decision(380, 1.0, 2.0) is None
    passed = gate_decision(400, 1.50324, 1.5082213)
    assert passed["passed"] and passed["candidate_bpb"] == 1.5032 and passed["reference_bpb"] == 1.5082
    assert passed["maximum_candidate_bpb"] == 1.5032 and passed["improvement_bpb"] == pytest.approx(0.005)
    assert not gate_decision(400, 1.50326, 1.5082213)["passed"]
    with pytest.raises(ValueError):
        gate_decision(400, float("nan"), 1.5)


def test_gate_reference_requires_a_matched_completed_run_with_one_update_400_validation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.train_recurrent_slots as trainer
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    run = tmp_path / "ablation_results" / "control"
    run.mkdir(parents=True)
    args = SimpleNamespace(steps=1000, val_every=20, seed=1337, seq_len=1024)
    data_path, tokenizer_path = tmp_path / "data/datasets/fineweb10B_sp1024", tmp_path / "data/tokenizers/fineweb_1024_bpe.model"
    base_result = dict(status="completed", steps=1000, completed_steps=1000, val_every=20, final_val_bpb=1.3586)
    base_config = dict(steps=1000, val_every=20, data_path="data/datasets/fineweb10B_sp1024",
                       tokenizer="data/tokenizers/fineweb_1024_bpe.model", seed=1337, seq_len=1024,
                       batch_tokens=524288, val_tokens=1048576)
    base_entries = [dict(type="val", step=step, val_bpb=2.0 - step * 0.001) for step in range(0, 1001, 20)]
    base_entries.insert(5, dict(type="train", step=400, train_loss=2.5))

    def write(result, config, entries):
        (run / "result.json").write_text(json.dumps(result))
        (run / "config.json").write_text(json.dumps(config))
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    write(base_result, base_config, base_entries)
    reference = trainer.load_gate_reference("control", args, data_path, tokenizer_path)
    assert reference["val_bpb"] == pytest.approx(1.6) and reference["step"] == 400
    assert len(reference["metrics_sha256"]) == 64 == len(reference["config_sha256"])
    assert reference["matched_fields"] == sorted(("completed", "steps", "val_every", "data_path", "tokenizer",
                                                  "seed", "seq_len", "batch_tokens", "val_tokens"))
    mutations = dict(status=dict(result=dict(status="pruned")), incomplete=dict(result=dict(completed_steps=400)),
                     val_every=dict(config=dict(val_every=40)), seed=dict(config=dict(seed=1)),
                     data=dict(config=dict(data_path="data/datasets/other")),
                     tokenizer=dict(config=dict(tokenizer="data/tokenizers/other.model")),
                     seq_len=dict(config=dict(seq_len=2048)), steps=dict(config=dict(steps=400)),
                     duplicate=dict(entries=base_entries + [dict(type="val", step=400, val_bpb=1.0)]),
                     missing=dict(entries=[e for e in base_entries if not (e["type"] == "val" and e["step"] == 400)]))
    for name, mutation in mutations.items():
        write(dict(base_result, **mutation.get("result", {})), dict(base_config, **mutation.get("config", {})),
              mutation.get("entries", base_entries))
        with pytest.raises(ValueError):
            trainer.load_gate_reference("control", args, data_path, tokenizer_path)
