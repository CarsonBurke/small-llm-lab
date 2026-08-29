"""CPU tests for the DiffusionBlocks machinery in
``pretraining/nanogpt_mini/nanogpt_mini_dblock.py``.

Covers the deterministic contracts: the equal-mass sigma partition, block
routing, EDM preconditioning, the layer partition rule ("every maximal run of
KDA mixers plus the dense layer that closes it is one block"), the two-stream
layout helpers, and — against the pure-PyTorch KDA oracle — the read-only
interleaved-slot construction that keeps the recurrent clean-state trajectory
bit-identical to a clean-only pass.
"""

import math
from statistics import NormalDist

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_dblock import (
    READ_ONLY_LOGIT,
    TimestepEmbedder,
    _probit,
    block_sigma_range,
    dblock_mask_mod,
    dblock_positions,
    deinterleave_streams,
    edm_coefficients,
    equal_mass_sigma_boundaries,
    inference_sigma_schedule,
    interleave_streams,
    lognormal_cdf,
    noisy_conv_preactivation,
    normalized_embedding_table,
    partition_layers_by_dense,
    sample_block_sigmas,
    sigma_to_block,
)
from pretraining.nanogpt_mini.nanogpt_mini_kda_model import (
    kda_recurrent_step,
    reference_kda_recurrence,
)

P_MEAN, P_STD = -1.2, 1.2
SIGMA_MIN, SIGMA_MAX = 0.002, 80.0


def default_boundaries(num_blocks: int = 6) -> list[float]:
    return equal_mass_sigma_boundaries(
        num_blocks,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        p_mean=P_MEAN,
        p_std=P_STD,
    )


########################################
#        Noise-level partition         #
########################################


def test_probit_matches_normaldist():
    normal = NormalDist()
    masses = torch.linspace(1e-6, 1 - 1e-6, 20011, dtype=torch.float64)
    approx = _probit(masses)
    exact = torch.tensor(
        [normal.inv_cdf(float(m)) for m in masses], dtype=torch.float64
    )
    assert torch.allclose(approx, exact, atol=5e-8, rtol=1e-6)


def test_equal_mass_boundaries_endpoints_and_masses():
    for num_blocks in (2, 3, 6):
        boundaries = default_boundaries(num_blocks)
        assert len(boundaries) == num_blocks + 1
        assert boundaries[0] == SIGMA_MIN
        assert boundaries[-1] == SIGMA_MAX
        assert boundaries == sorted(boundaries)
        cdfs = [lognormal_cdf(s, P_MEAN, P_STD) for s in boundaries]
        masses = [b - a for a, b in zip(cdfs, cdfs[1:])]
        for mass in masses:
            assert mass == pytest.approx(masses[0], rel=1e-6)


def test_block_sigma_range_layer_ordering():
    boundaries = default_boundaries(6)
    # Layer-order block 0 owns the highest noise band.
    low0, high0 = block_sigma_range(boundaries, 0, gamma=0.0)
    assert (low0, high0) == (boundaries[5], boundaries[6])
    low_last, high_last = block_sigma_range(boundaries, 5, gamma=0.0)
    assert (low_last, high_last) == (boundaries[0], boundaries[1])
    # gamma=0 ranges tile the boundary list exactly, in reverse.
    for block_index in range(6):
        low, high = block_sigma_range(boundaries, block_index, gamma=0.0)
        assert low == boundaries[5 - block_index]
        assert high == boundaries[6 - block_index]


def test_block_sigma_range_gamma_extension_and_clipping():
    boundaries = default_boundaries(6)
    gamma = 0.1
    for block_index in range(6):
        low0, high0 = block_sigma_range(boundaries, block_index, gamma=0.0)
        low, high = block_sigma_range(boundaries, block_index, gamma=gamma)
        log_range = math.log(high0) - math.log(low0)
        assert low == pytest.approx(
            max(math.exp(math.log(low0) - gamma * log_range), SIGMA_MIN)
        )
        assert high == pytest.approx(
            min(math.exp(math.log(high0) + gamma * log_range), SIGMA_MAX)
        )
        assert low >= SIGMA_MIN and high <= SIGMA_MAX
    # The extreme blocks hit the global clip on their outer edge.
    assert block_sigma_range(boundaries, 0, gamma=gamma)[1] == SIGMA_MAX
    assert block_sigma_range(boundaries, 5, gamma=gamma)[0] == SIGMA_MIN


def test_sigma_to_block_routing():
    boundaries = default_boundaries(6)
    assert sigma_to_block(SIGMA_MIN, boundaries) == 5
    assert sigma_to_block(SIGMA_MAX, boundaries) == 0
    for block_index in range(6):
        low, high = block_sigma_range(boundaries, block_index, gamma=0.0)
        midpoint = math.exp((math.log(low) + math.log(high)) / 2)
        assert sigma_to_block(midpoint, boundaries) == block_index
    # Just below/above an interior boundary routes to adjacent blocks.
    for interior in range(1, 6):
        sigma = boundaries[interior]
        assert sigma_to_block(sigma * 0.999, boundaries) == 6 - interior
        assert sigma_to_block(sigma * 1.001, boundaries) == 5 - interior


def test_inference_schedule_descends_and_covers_blocks_once():
    for num_blocks in (4, 6):
        boundaries = default_boundaries(num_blocks)
        schedule = inference_sigma_schedule(
            num_blocks,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
            p_mean=P_MEAN,
            p_std=P_STD,
        )
        assert len(schedule) == num_blocks
        assert schedule[0] == SIGMA_MAX
        assert schedule[-1] == SIGMA_MIN
        assert schedule == sorted(schedule, reverse=True)
        routed = [sigma_to_block(sigma, boundaries) for sigma in schedule]
        assert routed == list(range(num_blocks))


def test_sample_block_sigmas_stays_in_band():
    boundaries = default_boundaries(6)
    generator = torch.Generator()
    generator.manual_seed(7)
    for block_index in range(6):
        low, high = block_sigma_range(boundaries, block_index, gamma=0.1)
        sigmas = sample_block_sigmas(
            low, high, 4096, P_MEAN, P_STD,
            torch.device("cpu"), generator=generator,
        )
        assert sigmas.dtype == torch.float32
        assert float(sigmas.min()) >= low * (1 - 1e-5)
        assert float(sigmas.max()) <= high * (1 + 1e-5)
        # The truncated lognormal restricted to the band has a uniform CDF
        # image: check the transformed sample mean is near 1/2.
        cdf_low = lognormal_cdf(low, P_MEAN, P_STD)
        cdf_high = lognormal_cdf(high, P_MEAN, P_STD)
        images = torch.tensor(
            [
                (lognormal_cdf(float(s), P_MEAN, P_STD) - cdf_low)
                / (cdf_high - cdf_low)
                for s in sigmas[:1024]
            ]
        )
        assert abs(float(images.mean()) - 0.5) < 0.03


########################################
#         EDM preconditioning          #
########################################


def test_edm_coefficients_identities():
    sigma_data = 0.5
    sigma = torch.tensor([0.002, 0.05, 0.5, 4.0, 80.0])
    coeffs = edm_coefficients(sigma, sigma_data)
    variance = sigma.square() + sigma_data**2
    assert torch.allclose(coeffs["c_skip"], sigma_data**2 / variance)
    assert torch.allclose(coeffs["c_out"], sigma * sigma_data / variance.sqrt())
    # Preconditioning identities from Karras et al. (2022).
    assert torch.allclose(coeffs["c_in"].square() * variance, torch.ones_like(sigma))
    assert torch.allclose(
        coeffs["c_skip"].square() + coeffs["c_out"].square() / variance
        * sigma_data**-2 * variance,
        coeffs["c_skip"].square() + coeffs["c_out"].square() / sigma_data**2,
    )
    # The training weight is exactly 1 / c_out^2.
    assert torch.allclose(coeffs["weight"], coeffs["c_out"].square().reciprocal())
    assert torch.allclose(coeffs["c_noise"], 0.25 * sigma.log())


def test_normalized_embedding_table_unit_rows():
    weight = torch.randn(97, 32, dtype=torch.bfloat16)
    weight[3] = 0  # degenerate row must not produce NaN
    table = normalized_embedding_table(weight)
    assert table.dtype == torch.float32
    norms = table.square().sum(-1).sqrt()
    keep = torch.ones(97, dtype=torch.bool)
    keep[3] = False
    assert torch.allclose(norms[keep], torch.ones(96), atol=1e-3)
    assert table.isfinite().all()


########################################
#          Layer partitioning          #
########################################


def test_partition_default_3to1_schedule():
    delta = frozenset(i for i in range(24) if (i + 1) % 4)
    groups = partition_layers_by_dense(delta, 24)
    assert groups == [
        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],
        [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23],
    ]


def test_partition_merges_groups_evenly():
    delta = frozenset(i for i in range(24) if (i + 1) % 4)
    merged = partition_layers_by_dense(delta, 24, num_blocks=3)
    assert merged == [
        list(range(0, 8)), list(range(8, 16)), list(range(16, 24)),
    ]
    assert partition_layers_by_dense(delta, 24, num_blocks=6) == \
        partition_layers_by_dense(delta, 24)


def test_partition_rejects_trailing_kda_and_bad_counts():
    with pytest.raises(ValueError, match="last layer must be dense"):
        partition_layers_by_dense(frozenset({6, 7}), 8)
    delta = frozenset(i for i in range(24) if (i + 1) % 4)
    with pytest.raises(ValueError, match="evenly divide"):
        partition_layers_by_dense(delta, 24, num_blocks=4)
    with pytest.raises(ValueError, match="outside the layer range"):
        partition_layers_by_dense(frozenset({30}), 24)


def test_partition_all_dense_gives_singleton_groups():
    groups = partition_layers_by_dense(frozenset(), 4)
    assert groups == [[0], [1], [2], [3]]


########################################
#          Two-stream helpers          #
########################################


def test_interleave_roundtrip():
    clean = torch.randn(2, 5, 3, 7)
    noisy = torch.randn(2, 5, 3, 7)
    mixed = interleave_streams(clean, noisy)
    assert mixed.shape == (2, 10, 3, 7)
    assert torch.equal(mixed[:, 0], clean[:, 0])
    assert torch.equal(mixed[:, 1], noisy[:, 0])
    assert torch.equal(mixed[:, 8], clean[:, 4])
    back_clean, back_noisy = deinterleave_streams(mixed)
    assert torch.equal(back_clean, clean)
    assert torch.equal(back_noisy, noisy)


def test_dblock_positions():
    positions = dblock_positions(4, torch.device("cpu"))
    assert positions.tolist() == [0, 1, 2, 3, 1, 2, 3, 4]
    assert positions.dtype == torch.float32


def test_dblock_mask_mod_matches_bruteforce():
    seq_len = 9
    mask_mod = dblock_mask_mod(seq_len)
    total = 2 * seq_len
    for q_idx in range(total):
        for kv_idx in range(total):
            got = bool(
                mask_mod(
                    torch.tensor(0),
                    torch.tensor(0),
                    torch.tensor(q_idx),
                    torch.tensor(kv_idx),
                )
            )
            if q_idx < seq_len:
                # Clean query: ordinary causal attention over clean keys.
                expected = kv_idx <= q_idx
            else:
                # Noisy query for target position q_pos: clean keys up to and
                # including q_pos, plus itself, never another noisy token.
                q_pos = q_idx - seq_len
                expected = (kv_idx < seq_len and kv_idx <= q_pos) or (
                    kv_idx == q_idx
                )
            assert got == expected, (q_idx, kv_idx)


def test_noisy_conv_preactivation_matches_window_substitution():
    torch.manual_seed(0)
    B, T, D, W = 2, 11, 6, 4
    clean = torch.randn(B, T, D)
    noisy = torch.randn(B, T, D)
    weight = torch.randn(D, 1, W)
    got = noisy_conv_preactivation(clean, noisy, weight)
    assert got.shape == (B, T, D)
    # Brute force: at position i, run the plain causal conv over the sequence
    # [c_0, ..., c_i, n_i] and take the newest output — the preactivation the
    # original network would compute were the next input the noisy embedding.
    for i in range(T):
        substituted = torch.cat(
            (clean[:, : i + 1], noisy[:, i : i + 1]), dim=1
        )
        full = torch.nn.functional.conv1d(
            substituted.transpose(1, 2),
            weight,
            groups=D,
            padding=W - 1,
        )[..., : i + 2].transpose(1, 2)
        assert torch.allclose(got[:, i], full[:, -1], atol=1e-5), i


########################################
#     Read-only KDA slot semantics     #
########################################


def _random_kda_inputs(B, T, H, Dk, Dv, generator):
    def draw(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    return {
        "q": draw(B, T, H, Dk),
        "k": draw(B, T, H, Dk),
        "v": draw(B, T, H, Dv),
        "decay_logits": draw(B, T, H, Dk),
        "beta_logits": draw(B, T, H),
        "A_log": draw(H).abs() * 0.1,
        "dt_bias": draw(H * Dk) * 0.1 - 3.0,
    }


def test_readonly_interleaved_slots_preserve_clean_trajectory():
    """The trainer's masked interleave [c_0, n_0, c_1, n_1, ...] must leave
    the clean recurrence untouched and give each noisy slot exactly the state
    read after its inclusive clean prefix."""
    generator = torch.Generator()
    generator.manual_seed(42)
    B, T, H, Dk, Dv = 2, 7, 3, 8, 8
    ins = _random_kda_inputs(B, T, H, Dk, Dv, generator)
    noisy = _random_kda_inputs(B, T, H, Dk, Dv, generator)

    def interleave(clean_t, noisy_t):
        return interleave_streams(clean_t, noisy_t)

    q_int = interleave(ins["q"], noisy["q"])
    k_int = interleave(ins["k"], noisy["k"])
    v_int = interleave(ins["v"], noisy["v"])
    g_int = interleave(
        ins["decay_logits"],
        torch.full_like(ins["decay_logits"], READ_ONLY_LOGIT),
    )
    beta_int = interleave(
        ins["beta_logits"],
        torch.full_like(ins["beta_logits"], READ_ONLY_LOGIT),
    )

    out_int, final_int = reference_kda_recurrence(
        q_int, k_int, v_int, g_int, beta_int, ins["A_log"], ins["dt_bias"]
    )
    out_clean, final_clean = reference_kda_recurrence(
        ins["q"], ins["k"], ins["v"], ins["decay_logits"],
        ins["beta_logits"], ins["A_log"], ins["dt_bias"],
    )

    clean_slots, noisy_slots = deinterleave_streams(out_int)
    # (a) Clean outputs and the final state are bit-identical to the
    # clean-only pass: the read-only slots neither decay nor write.
    assert torch.equal(clean_slots, out_clean)
    assert torch.equal(final_int, final_clean)

    # (b) Every noisy slot reads exactly the state after its inclusive clean
    # prefix c_0..c_i. Replay the clean recurrence and issue a pure read
    # (zero log-decay, zero beta) with the noisy projections at each step.
    state = torch.zeros(B, H, Dv, Dk, dtype=torch.float32)
    zero_gate = torch.zeros(B, H, Dk)
    zero_beta = torch.zeros(B, H)
    from pretraining.nanogpt_mini.nanogpt_mini_kda_model import kda_decay_gate

    gate = kda_decay_gate(ins["decay_logits"], ins["A_log"], ins["dt_bias"])
    beta = torch.sigmoid(ins["beta_logits"].float())
    for t in range(T):
        kda_recurrent_step(
            ins["q"][:, t], ins["k"][:, t], ins["v"][:, t],
            gate[:, t], beta[:, t], state,
        )
        probe = state.clone()
        expected = kda_recurrent_step(
            noisy["q"][:, t], noisy["k"][:, t], noisy["v"][:, t],
            zero_gate, zero_beta, probe,
        )
        assert torch.equal(probe, state), t  # the read must not write
        assert torch.allclose(noisy_slots[:, t].float(), expected, atol=1e-5), t


def test_read_only_logit_is_exact_under_gate_math():
    """sigmoid(READ_ONLY_LOGIT) must underflow to exactly zero in fp32 so the
    masked slots are read-only to the bit, not approximately."""
    assert torch.sigmoid(torch.tensor(READ_ONLY_LOGIT)).item() == 0.0
    from pretraining.nanogpt_mini.nanogpt_mini_kda_model import kda_decay_gate

    logits = torch.full((1, 1, 2, 4), READ_ONLY_LOGIT)
    A_log = torch.tensor([0.5, -0.3])
    dt_bias = torch.full((8,), -2.0)
    gate = kda_decay_gate(logits, A_log, dt_bias)
    assert torch.equal(gate, torch.zeros_like(gate))
    assert (gate.exp() == 1).all()
    # bf16 representable (the trainer interleaves bf16 activations).
    assert torch.tensor(READ_ONLY_LOGIT).bfloat16().isfinite()


########################################
#          Noise conditioning          #
########################################


def test_timestep_embedder_shapes_and_determinism():
    embedder = TimestepEmbedder(64, freq_dim=32)
    c_noise = 0.25 * torch.tensor([0.002, 1.0, 80.0]).log()
    out = embedder(c_noise)
    assert out.shape == (3, 64)
    assert out.isfinite().all()
    assert torch.equal(out, embedder(c_noise))
    # Distinct noise levels produce distinct conditioning vectors.
    assert not torch.allclose(out[0], out[2])
