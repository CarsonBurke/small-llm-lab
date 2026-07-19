from __future__ import annotations

import torch

from fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
from postraining.eval_open_loop import open_loop_depth_metrics
from postraining.latent_thought import LatentThoughtModel
from postraining.model_io import _pope_construction

KWARGS = dict(
    vocab_size=32, num_layers=3, model_dim=32, num_heads=4, num_kv_heads=2,
    mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.01,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
)


def _wrapper(seed: int = 3) -> LatentThoughtModel:
    torch.manual_seed(seed)
    with _pope_construction():
        backbone = FreshLeJEPASharedRMSV1PoPE(**KWARGS).eval()
    # Fresh models zero-init every output projection, which makes logits a
    # constant and the trunk blind to its own history — the equivalence and
    # reproducibility assertions below would then compare constants and pass
    # vacuously.  De-zero so cache/position dynamics are actually exercised.
    with torch.no_grad():
        for module in backbone.modules():
            if (
                isinstance(module, torch.nn.Linear)
                and float(module.weight.abs().max()) == 0.0
            ):
                torch.nn.init.normal_(module.weight, std=0.2)
    return LatentThoughtModel(backbone).eval()


def _luts(vocab: int = 32):
    return (
        torch.ones(vocab, dtype=torch.int16),
        torch.zeros(vocab, dtype=torch.bool),
        torch.zeros(vocab, dtype=torch.bool),
    )


def _batch(batch: int = 2, length: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    trajectory = torch.randint(0, 32, (batch, length + 1))
    return trajectory[:, :-1], trajectory[:, 1:]


def test_all_grounded_matches_teacher_forced_eval_loss():
    torch.manual_seed(17)
    wrapper = _wrapper()
    x, y = _batch(batch=2, length=12)
    with torch.no_grad():
        reference = wrapper(x, y)
        metrics = open_loop_depth_metrics(wrapper, x, y, 4, 0, *_luts())
    assert [m["depth"] for m in metrics] == [0]
    assert metrics[0]["tokens"] == x.numel()
    torch.testing.assert_close(
        torch.tensor(metrics[0]["loss"]), reference, rtol=2e-4, atol=2e-4
    )


def test_depth_buckets_follow_the_ground_imagine_schedule():
    wrapper = _wrapper()
    x, y = _batch(batch=2, length=8)
    metrics = open_loop_depth_metrics(wrapper, x, y, 2, 2, *_luts())
    by_depth = {m["depth"]: m["tokens"] for m in metrics}
    # Positions 0,1,4,5 grounded (depth 0); 2,6 depth 1; 3,7 depth 2.
    assert by_depth == {0: 8, 1: 4, 2: 4}
    # Unit byte LUT with no leading-space corrections: bpb == bits per token.
    for entry in metrics:
        torch.testing.assert_close(
            torch.tensor(entry["bpb"]),
            torch.tensor(entry["loss"] / torch.log(torch.tensor(2.0)).item()),
        )


def test_byte_accounting_matches_hand_computation():
    import math

    wrapper = _wrapper()
    x, y = _batch(batch=2, length=6)
    base = torch.full((32,), 2, dtype=torch.int16)
    space = torch.ones(32, dtype=torch.bool)
    boundary = torch.arange(32) % 2 == 0
    metrics = open_loop_depth_metrics(wrapper, x, y, 3, 0, base, space, boundary)
    entry = metrics[0]
    # eval_val convention: bytes(target) = base[tgt] + (space[tgt] & ~boundary[prev])
    # with prev the TRUE input token at the scored position.
    expected_bytes = float((2 + (~boundary[x]).to(torch.int16)).sum())
    expected_bpb = entry["loss"] * entry["tokens"] / math.log(2.0) / expected_bytes
    torch.testing.assert_close(
        torch.tensor(entry["bpb"]), torch.tensor(expected_bpb)
    )


def test_negative_imagine_is_rejected():
    wrapper = _wrapper()
    x, y = _batch()
    try:
        open_loop_depth_metrics(wrapper, x, y, 2, -1, *_luts())
    except ValueError:
        return
    raise AssertionError("negative imagine must raise")


def test_sampled_imagination_is_finite_and_reproducible():
    wrapper = _wrapper()
    x, y = _batch(batch=2, length=8)
    first = open_loop_depth_metrics(
        wrapper, x, y, 2, 2, *_luts(), mode="sample",
        generator=torch.Generator().manual_seed(5),
    )
    second = open_loop_depth_metrics(
        wrapper, x, y, 2, 2, *_luts(), mode="sample",
        generator=torch.Generator().manual_seed(5),
    )
    assert all(torch.isfinite(torch.tensor(m["loss"])) for m in first)
    assert [m["loss"] for m in first] == [m["loss"] for m in second]
    # Sampled imagination must actually diverge from the mean path at d>0.
    mean_mode = open_loop_depth_metrics(wrapper, x, y, 2, 2, *_luts())
    assert first[1]["loss"] != mean_mode[1]["loss"]
