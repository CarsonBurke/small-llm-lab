"""Full MiniCPM numerical regressions, opt-in and exclusively run through mlq.

RUN_UNO_CUDA_VALIDATION=1 enables real bf16 FlexAttention backward and captured
FA4 rollout. The temporary one-update adapter is a numerical fixture, never a
learning-quality or throughput acceptance result.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_UNO_CUDA_VALIDATION") != "1",
    reason="real model workload: enable explicitly inside an exclusive mlq job",
)


def _report(name, **values):
    print(json.dumps({"check": name, **values}, sort_keys=True), flush=True)


def test_invariant_projection_handles_strides_partial_tiles_and_ar_head_shape():
    """Noncontiguous inputs and masked edge tiles must keep the same arithmetic."""
    from postraining.invariant_linear import InvariantLinear, compile_invariant

    torch.manual_seed(149)
    projection = InvariantLinear(
        192, 136, bias=False, device="cuda", dtype=torch.bfloat16
    )
    projection.decoding = True
    inputs = torch.randn(67, 4, 384, device="cuda", dtype=torch.bfloat16)[..., ::2]
    forward = compile_invariant(projection)
    with torch.inference_mode():
        actual = forward(inputs)
        serial = torch.stack(
            [forward(inputs[:, index].contiguous()) for index in range(4)], dim=1
        )
        expected = F.linear(inputs.float(), projection.weight.float())
    torch.testing.assert_close(actual, serial, rtol=0, atol=0)
    torch.testing.assert_close(actual.float(), expected, rtol=0.01, atol=0.005)
    _report(
        "invariant_projection",
        maximum_serial_error=(actual - serial).abs().max().item(),
    )


@pytest.mark.parametrize("optimized_decode", [False, True])
def test_native_fa4_ragged_suffix_matches_full_precision_causal_reference(
    optimized_decode,
):
    from postraining.invariant_attention import invariant_fa4

    torch.manual_seed(149)
    device = torch.device("cuda")
    # Actual MiniCPM GQA geometry; poisoned future KV must never enter attention.
    keys = torch.randn(4, 8192, 2, 128, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    lengths = torch.tensor([8192, 513, 65, 16], device=device, dtype=torch.int32)
    for row, length in enumerate(lengths.tolist()):
        keys[row, length:] = float("nan")
        values[row, length:] = float("nan")
    for width in (1, 2, 4, 8, 16):
        query = torch.randn(4, width, 16, 128, device=device, dtype=torch.bfloat16)
        arguments = (
            query,
            keys,
            values,
            lengths,
            width,
            8192,
            128**-0.5,
            optimized_decode,
            None,
            None,
        )
        actual = invariant_fa4(*arguments)
        if width in (1, 4):
            torch.library.opcheck(
                invariant_fa4,
                arguments,
                test_utils=("test_schema", "test_faketensor"),
            )
        references = []
        for row, length in enumerate(lengths.tolist()):
            mask = torch.arange(length, device=device)[None] <= (
                length - width + torch.arange(width, device=device)[:, None]
            )
            reference = F.scaled_dot_product_attention(
                query[row : row + 1].transpose(1, 2).float(),
                keys[row : row + 1, :length].transpose(1, 2).float(),
                values[row : row + 1, :length].transpose(1, 2).float(),
                attn_mask=mask,
                enable_gqa=True,
            )
            references.append(reference.transpose(1, 2))
        expected = torch.cat(references)
        torch.testing.assert_close(actual.float(), expected, atol=0.025, rtol=0.025)
        _report(
            "fa4_suffix",
            width=width,
            maximum_absolute_error=(actual.float() - expected).abs().max().item(),
        )


@pytest.mark.parametrize(("batch", "capacity"), [(1, 257), (4, 257), (4, 8192)])
def test_split_fa4_graph_tracks_retirement_refill_and_empty_partitions(batch, capacity):
    from postraining.invariant_attention import invariant_fa4
    from postraining.split_kv_plan import SPLITS, plan_split_kv, split_kv_metadata

    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 split-KV qualification")
    torch.manual_seed(163)
    # Also exercise noncontiguous head/dimension strides through graph replay.
    if batch == 4 and capacity == 257:
        query = torch.randn(batch, 1, 128, 32, device="cuda", dtype=torch.bfloat16)
        query = query.transpose(2, 3)[:, :, ::2, :]
    else:
        query = torch.randn(batch, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(batch, capacity, 2, 128, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    lengths = torch.tensor(
        [capacity, 65, 2, 1][:batch], device="cuda", dtype=torch.int32
    )
    arguments = (query, keys, values, lengths, 1, capacity, 128**-0.5, True, None, None)
    # The rollout engine plans partitions once per step and shares the plan
    # across layers; that path must replay bitwise-identically to per-launch planning.
    metadata = split_kv_metadata(batch, capacity, "cuda")
    plan_offsets = torch.zeros(batch * SPLITS + 1, device="cuda", dtype=torch.int32)
    plan_live = torch.zeros(batch * SPLITS, device="cuda", dtype=torch.int32)

    def shared_plan_launch():
        offsets, live = plan_split_kv(lengths, *metadata)
        plan_offsets.copy_(offsets)
        plan_live.copy_(live)
        return invariant_fa4(*arguments[:-2], plan_offsets, plan_live)

    for _ in range(3):
        invariant_fa4(*arguments)
        shared_plan_launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = invariant_fa4(*arguments)
        captured_shared = shared_plan_launch()

    # Reuse the captured addresses across shrinking lengths, then refill.
    # NaNs outside each live prefix catch both stale offsets and empty-split reads.
    for live_lengths in ([capacity, 65, 2, 1], [1, 64, 1, 1], [129, 1, capacity, 65]):
        live_lengths = live_lengths[:batch]
        lengths.copy_(torch.tensor(live_lengths, device="cuda", dtype=torch.int32))
        query.normal_()
        keys.normal_()
        values.normal_()
        for row, length in enumerate(live_lengths):
            keys[row, length:] = float("nan")
            values[row, length:] = float("nan")
        graph.replay()
        torch.testing.assert_close(captured_shared, captured, atol=0, rtol=0)
        references = []
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for row, length in enumerate(live_lengths):
                references.append(
                    F.scaled_dot_product_attention(
                        query[row : row + 1].transpose(1, 2).float(),
                        keys[row : row + 1, :length].transpose(1, 2).float(),
                        values[row : row + 1, :length].transpose(1, 2).float(),
                        enable_gqa=True,
                    ).transpose(1, 2)
                )
        expected = torch.cat(references)
        torch.testing.assert_close(captured.float(), expected, atol=0.015, rtol=0.015)
        # A single live token must reproduce V exactly, despite three empty splits.
        for row, length in enumerate(live_lengths):
            if length == 1:
                torch.testing.assert_close(
                    captured[row, 0],
                    values[row, 0].repeat_interleave(8, dim=0),
                    atol=0,
                    rtol=0,
                )
        _report(
            "split_fa4_graph",
            capacity=capacity,
            live_lengths=live_lengths,
            maximum_absolute_error=(captured.float() - expected).abs().max().item(),
        )


def test_split_fa4_retains_small_residual_across_cancelling_partitions():
    from postraining.invariant_attention import invariant_fa4

    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 split-KV qualification")
    query = torch.zeros(1, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.zeros(1, 128, 2, 128, device="cuda", dtype=torch.bfloat16)
    values = torch.zeros_like(keys)
    values[:, :16] = 1.0
    values[:, 16:32] = 1.0078125
    values[:, 32:64] = -1.0
    lengths = torch.tensor([128], device="cuda", dtype=torch.int32)
    actual = invariant_fa4(
        query, keys, values, lengths, 1, 128, 128**-0.5, True, None, None
    )
    # Uniform attention: (16 + 16*1.0078125 - 32) / 128 == 2**-10.
    # Rounding the first normalized partial to BF16 would erase this residual.
    torch.testing.assert_close(actual, torch.full_like(actual, 2**-10), atol=0, rtol=0)


@pytest.fixture(scope="module")
def numerical_adapter(tmp_path_factory):
    from postraining.hf_runtime import prepare_text_only_transformers_runtime
    from postraining.vapo.model.hf import MINICPM5_SPEC
    from postraining.uno import (
        UnoAdapterBank,
        UnoConfig,
        attach_uno_adapters,
        chunked_head_l1,
        enable_uno_checkpointing,
        make_uno_block_mask,
        paired_uno_inputs,
        uno_checkpoint_payload,
        uno_distillation_loss,
    )

    prepare_text_only_transformers_runtime()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(151)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        MINICPM5_SPEC.model_id, revision=MINICPM5_SPEC.revision
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            MINICPM5_SPEC.model_id,
            revision=MINICPM5_SPEC.revision,
            dtype=torch.bfloat16,
            attn_implementation="flex_attention",
            low_cpu_mem_usage=True,
        )
        .to(device)
        .requires_grad_(False)
        .eval()
    )
    bank = UnoAdapterBank(model, UnoConfig())
    router = attach_uno_adapters(model, bank)
    encoded = tokenizer.encode(
        "Compute 37 times 46. 37 times 40 is 1480; 37 times 6 is 222. The answer is 1702.\n"
    )
    clean = torch.tensor((encoded * (2048 // len(encoded) + 1))[:2048], device=device)[
        None
    ]
    mask = make_uno_block_mask(2048, 4, device)
    ids, positions, gate = paired_uno_inputs(clean, model.config.vocab_size)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        router.set_gate(None)
        reference = model.model(input_ids=clean, use_cache=False).last_hidden_state
        # Match physical GEMM/attention shapes while keeping the reference
        # strictly ordinary causal AR. Future tokens cannot affect its prefix.
        matched_reference = model.model(
            input_ids=ids, use_cache=False
        ).last_hidden_state[:, :2048]
        router.set_gate(gate)
        paired = model.model(
            input_ids=ids, position_ids=positions, attention_mask=mask, use_cache=False
        ).last_hidden_state
        clean_error = (
            paired[:, :2048].float() - matched_reference.float()
        ).norm() / matched_reference.float().norm()
        shape_error = (
            matched_reference.float() - reference.float()
        ).norm() / reference.float().norm()
        sample_positions = torch.linspace(0, 2047, 128, device=device).long()
        p_native = model.lm_head(reference[:, sample_positions]).float().softmax(-1)
        p_paired = model.lm_head(paired[:, sample_positions]).float().softmax(-1)
        shape_tv = (p_native - p_paired).abs().sum(-1) / 2
        _report(
            "teacher_geometry",
            matched_shape_relative_l2=clean_error.item(),
            ordinary_ar_shape_relative_l2=shape_error.item(),
            ordinary_ar_shape_mean_tv=shape_tv.mean().item(),
            ordinary_ar_shape_maximum_tv=shape_tv.max().item(),
        )
        torch.testing.assert_close(paired[:, :2048], matched_reference, rtol=0, atol=0)
        del matched_reference, p_native, p_paired, shape_tv
        for projection in bank.projections.values():
            projection.lora_b.normal_(std=0.0002)
        # Only this fixture mutates trainable masters inside one autocast scope.
        # Production optimizer updates happen after the scope exits.
        torch.clear_autocast_cache()
        changed = model.model(
            input_ids=ids, position_ids=positions, attention_mask=mask, use_cache=False
        ).last_hidden_state
        # Same mask and kernels: diffusion weights cannot affect clean teacher positions.
        torch.testing.assert_close(changed[:, :2048], paired[:, :2048], rtol=0, atol=0)
        assert (changed[:, 2048:] - paired[:, 2048:]).abs().max().item() > 0
        for projection in bank.projections.values():
            projection.lora_b.zero_()
    router.set_gate(None)

    # Real 130,560-way head, uneven position chunks, bf16 forward/backward.
    student = paired[:, 2048:2055].detach().clone().requires_grad_(True)
    teacher = reference[:, :7].detach()
    reference_student = student.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual_loss = chunked_head_l1(student, teacher, model.lm_head, chunk_size=3)
        # Match each bf16 GEMM shape; a one-row tail can choose a different
        # cuBLAS kernel from an unchunked head. Autograd remains the independent
        # reference for the custom analytical softmax/L1 backward.
        p = torch.cat(
            [
                model.lm_head(teacher[:, start : start + 3]).float().softmax(-1)
                for start in range(0, 7, 3)
            ],
            dim=1,
        )
        q = torch.cat(
            [
                model.lm_head(reference_student[:, start : start + 3])
                .float()
                .softmax(-1)
                for start in range(0, 7, 3)
            ],
            dim=1,
        )
        reference_loss = (q - p).abs().sum(-1).mean()
    actual_loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(actual_loss, reference_loss, atol=2e-4, rtol=2e-3)
    torch.testing.assert_close(
        student.grad, reference_student.grad, atol=3e-5, rtol=0.03
    )
    del (
        paired,
        changed,
        reference,
        ids,
        positions,
        gate,
        student,
        reference_student,
        teacher,
        p,
        q,
    )
    gc.collect()
    torch.cuda.empty_cache()

    enable_uno_checkpointing(model, router)
    model.train()
    optimizer = torch.optim.AdamW(
        bank.parameters(), lr=1e-5, weight_decay=0.0, fused=True
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = uno_distillation_loss(model, router, clean, mask, chunk_size=32)
    # Router is deliberately disabled before checkpoint recomputation.
    assert router.gate is None
    loss.backward()
    assert torch.isfinite(loss)
    adapter_ids = {id(parameter) for parameter in bank.parameters()}
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
        if id(parameter) not in adapter_ids
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in bank.parameters()
    )
    assert (
        sum(
            projection.lora_b.grad.abs().sum().item()
            for projection in bank.projections.values()
        )
        > 0
    )
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    corpus_digest = hashlib.sha256(clean.cpu().numpy().tobytes()).hexdigest()
    teacher_digest = hashlib.sha256(
        json.dumps(
            {"model_id": MINICPM5_SPEC.model_id, "revision": MINICPM5_SPEC.revision},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    payload = uno_checkpoint_payload(
        bank,
        model,
        model_id=MINICPM5_SPEC.model_id,
        revision=MINICPM5_SPEC.revision,
        trained_tokens=2048,
        step=1,
        teacher_sha256=teacher_digest,
        training={
            "corpus_sha256": corpus_digest,
            "purpose": "numerical_regression_fixture_not_learning_evidence",
        },
    )
    path = tmp_path_factory.mktemp("uno-numerical-fixture") / "adapter.pt"
    torch.save(payload, path)
    _report(
        "full_model_distillation_backward",
        clean_relative_l2=clean_error.item(),
        l1=loss.item(),
        compile_inclusive_update_seconds=elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        trained_fixture_tokens=2048,
    )
    router.close()
    del (
        payload,
        optimizer,
        loss,
        actual_loss,
        reference_loss,
        model,
        bank,
        router,
        mask,
        clean,
    )
    gc.collect()
    torch.cuda.empty_cache()
    yield path
    path.unlink(missing_ok=True)


def _reference_block(policy, prompts, suffixes):
    result = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for prompt, suffix in zip(prompts, suffixes, strict=True):
            tokens = torch.cat((prompt, suffix.cpu())).to("cuda")[None]
            hidden = policy.replay_hidden(tokens, None)
            result.append(policy.logits(hidden[:, -suffix.numel() :]).float())
    return torch.cat(result)


def _native_token_forward(engine, current):
    hidden = engine.policy.cached_hidden(
        current,
        past_key_values=engine.cache,
        cache_position=engine._block_offsets[:1],
        position_ids=engine.sequence_lengths[:, None],
    )
    return engine.policy.logits(hidden)


def _serial_suffix(engine, tokens, cursor, *, padded):
    """Current fused actor, same clean KV, serially committed real tokens."""
    engine.sequence_lengths.copy_(cursor)
    outputs = []
    width = engine.uno_block_size if padded else 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(tokens.size(1)):
            engine.flash_sequence_lengths.copy_((engine.sequence_lengths + width).int())
            current = tokens[:, index : index + 1].contiguous()
            if padded:
                current = torch.cat((current, torch.zeros_like(tokens[:, 1:])), dim=1)
                logits = engine._verify_forward(current)
            else:
                logits = engine._validation_serial_forward(current)
            outputs.append(logits[:, 0].float().clone())
            engine.sequence_lengths.add_(1)
    return torch.stack(outputs, dim=1)


def _compare_serial_suffix(engine, tokens):
    cursor = engine.sequence_lengths.clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = engine._verify_forward(tokens).float().clone()
    rows = torch.arange(engine.batch_size, device="cuda")[:, None]
    positions = cursor[:, None] + torch.arange(tokens.size(1), device="cuda")[None]
    committed = [
        (
            layer.key_backing[rows, positions].clone(),
            layer.value_backing[rows, positions].clone(),
        )
        for layer in engine.cache.layers
    ]
    padded = _serial_suffix(engine, tokens, cursor, padded=True)
    serial = _serial_suffix(engine, tokens, cursor, padded=False)
    same_shape_tv = (actual.softmax(-1) - padded.softmax(-1)).abs().sum(-1) / 2
    native_tv = (actual.softmax(-1) - serial.softmax(-1)).abs().sum(-1) / 2
    _report(
        "block_vs_serial_actor",
        padded_serial_maximum_tv=same_shape_tv.max().item(),
        invariant_serial_maximum_tv=native_tv.max().item(),
        invariant_serial_mean_tv=native_tv.mean().item(),
    )
    torch.testing.assert_close(actual, serial, rtol=0, atol=0)
    torch.testing.assert_close(actual, padded, rtol=0, atol=0)
    block_ids, block_p = engine._support(actual.bfloat16())
    for index in range(tokens.size(1)):
        serial_ids, serial_p = engine._support(serial[:, index].contiguous().bfloat16())
        torch.testing.assert_close(block_ids[:, index], serial_ids, rtol=0, atol=0)
        torch.testing.assert_close(block_p[:, index], serial_p, rtol=0, atol=0)
    for layer, (keys, values) in zip(engine.cache.layers, committed):
        torch.testing.assert_close(
            layer.key_backing[rows, positions], keys, rtol=0, atol=0
        )
        torch.testing.assert_close(
            layer.value_backing[rows, positions], values, rtol=0, atol=0
        )
    return actual


def _check_committed_cycle(engine, policy, prompts):
    with torch.inference_mode():
        engine.prepare_generation()
        bank = engine._build_prompt_prefix_bank(
            prompts,
            prefill_batch_prompts=4,
            collect_values=False,
            storage_device=engine.generated.device,
        )
        engine._ensure_continuous_cache(bank)
        rows_per_prompt = engine.batch_size // len(prompts)
        for index in range(len(prompts)):
            engine._admit_prompt_rows(
                bank,
                index,
                range(index * rows_per_prompt, (index + 1) * rows_per_prompt),
                max_new_tokens=33,
            )
        probe = torch.cat(
            (
                engine._pending[:, None],
                torch.zeros((engine.batch_size, 3), dtype=torch.long, device="cuda"),
            ),
            dim=1,
        )
        engine.flash_sequence_lengths.copy_((engine.sequence_lengths + 4).int())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            engine.uno_router.set_gate(None)
            clean_first = engine._verify_forward(probe)[:, 0].float().clone()
            engine.uno_router.set_gate(engine._draft_gate)
            draft_first = engine._draft_forward(probe)[:, 0].float().clone()
        first_tv = (clean_first.softmax(-1) - draft_first.softmax(-1)).abs().sum(-1) / 2
        _report("gated_first_position", maximum_tv=first_tv.max().item())
        torch.testing.assert_close(draft_first, clean_first, rtol=0, atol=0)
        engine._continuous_split_decode_step()
        counts = engine.output_position.clone()
        responses = engine.generated[:, :5].clone()
        committed_prompts = []
        for index, prompt in enumerate(prompts):
            slot = index * rows_per_prompt
            length = int(engine.output_position[slot])
            assert 1 <= length <= 5
            assert int(engine.sequence_lengths[slot]) == prompt.numel() + length - 1
            committed_prompts.append(
                torch.cat((prompt, engine.generated[slot, : length - 1].cpu()))
            )
        suffixes = torch.cat(
            (
                engine._pending[:, None],
                torch.randint(
                    policy.causal_lm.config.vocab_size,
                    (engine.batch_size, 3),
                    device="cuda",
                ),
            ),
            dim=1,
        )
        engine.flash_sequence_lengths.copy_((engine.sequence_lengths + 4).int())
        engine.uno_router.set_gate(None)
        actual_all = _compare_serial_suffix(engine, suffixes)
        # Rebuild from the untouched original prompt bank, using ONLY serial
        # clean actor forwards for the newly committed tokens. No draft or
        # verification scratch from the exercised cycle can survive this reset.
        for index, prompt in enumerate(prompts):
            slots = range(index * rows_per_prompt, (index + 1) * rows_per_prompt)
            for layer_index, layer in enumerate(engine.cache.layers):
                for backing, original in (
                    (layer.key_backing, bank.layer_keys),
                    (layer.value_backing, bank.layer_values),
                ):
                    expected_prefix = original[index, : prompt.numel() - 1, layer_index]
                    torch.testing.assert_close(
                        backing[list(slots), : prompt.numel() - 1],
                        expected_prefix[None].expand(rows_per_prompt, -1, -1, -1),
                        rtol=0,
                        atol=0,
                    )
            engine._admit_prompt_rows(bank, index, slots, max_new_tokens=33)
        original_pending = engine._pending.clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for offset in range(int(counts.max())):
                token = original_pending if offset == 0 else responses[:, offset - 1]
                padded = torch.cat(
                    (token[:, None], torch.zeros_like(suffixes[:, 1:])), dim=1
                )
                engine.flash_sequence_lengths.copy_((engine.sequence_lengths + 4).int())
                engine._verify_forward(padded)
                engine.sequence_lengths.add_((offset < counts).long())
            engine.flash_sequence_lengths.copy_((engine.sequence_lengths + 4).int())
            rebuilt = engine._verify_forward(suffixes).float()
        rebuilt_tv = (actual_all.softmax(-1) - rebuilt.softmax(-1)).abs().sum(-1) / 2
        _report("independent_clean_kv_rebuild", maximum_tv=rebuilt_tv.max().item())
        assert rebuilt_tv.max().item() < 1e-4
        actual = actual_all[::rows_per_prompt]
        engine.release_cache()
        expected = _reference_block(
            policy, committed_prompts, suffixes[::rows_per_prompt]
        )
        tv = (actual.softmax(-1) - expected.softmax(-1)).abs().sum(-1) / 2
        _report("post_cycle_committed_cache", maximum_tv=tv.max().item())


def _check_clean_verifier(engine, policy, prompts, suffixes):
    with torch.inference_mode():
        engine.prepare_generation()
        bank = engine._build_prompt_prefix_bank(
            prompts,
            prefill_batch_prompts=4,
            collect_values=False,
            storage_device=engine.generated.device,
        )
        engine._ensure_continuous_cache(bank)
        rows_per_prompt = engine.batch_size // len(prompts)
        for index in range(len(prompts)):
            engine._admit_prompt_rows(
                bank,
                index,
                range(index * rows_per_prompt, (index + 1) * rows_per_prompt),
                max_new_tokens=33,
            )
        engine.flash_sequence_lengths.copy_(
            (engine.sequence_lengths + engine.uno_block_size).int()
        )
        pending = torch.cat(
            (
                engine._pending[:, None],
                torch.zeros(
                    (engine.batch_size, engine.uno_block_size - 1),
                    dtype=torch.long,
                    device="cuda",
                ),
            ),
            dim=1,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            first = (
                engine._verify_forward(pending)[:, 0].float().clone()[::rows_per_prompt]
            )
        engine.sequence_lengths.add_(1)
        engine.flash_sequence_lengths.copy_(
            (engine.sequence_lengths + engine.uno_block_size).int()
        )
        engine.uno_router.set_gate(None)
        tokens = suffixes.repeat_interleave(rows_per_prompt, dim=0)
        actual = _compare_serial_suffix(engine, tokens)[::rows_per_prompt]
        engine.release_cache()
        expected = _reference_block(policy, prompts, suffixes)
        relative_l2 = (actual - expected).norm() / expected.norm()
        tv = (actual.softmax(-1) - expected.softmax(-1)).abs().sum(-1) / 2
        _report(
            "clean_verifier",
            relative_l2=relative_l2.item(),
            maximum_tv=tv.max().item(),
            mean_tv=tv.mean().item(),
        )
        combined = torch.cat((first[:, None], actual), dim=1)
        invariant = _actual_ar_reference(
            policy,
            prompts,
            suffixes,
            engine.stop_ids,
            invariant=True,
        )
        torch.testing.assert_close(combined, invariant, rtol=0, atol=0)
        _report("actual_invariant_ar_exact", maximum_logit_error=0.0)
        from postraining.uno_speculative import sparse_sampling_support

        p_ids, p = sparse_sampling_support(
            combined, temperature=0.9, top_k=20, top_p=0.95
        )
        q_ids, q = sparse_sampling_support(
            invariant, temperature=0.9, top_k=20, top_p=0.95
        )
        torch.testing.assert_close(p_ids, q_ids, rtol=0, atol=0)
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        legacy = _actual_ar_reference(
            policy, prompts, suffixes, engine.stop_ids, invariant=False
        )
        legacy_tv = (combined.softmax(-1) - legacy.softmax(-1)).abs().sum(-1) / 2
        _report(
            "actual_ar_target",
            invariant_maximum_tv=0.0,
            legacy_maximum_tv=legacy_tv.max().item(),
            legacy_mean_tv=legacy_tv.mean().item(),
        )
        return combined


def _actual_ar_reference(
    policy,
    prompts,
    suffixes,
    stop_ids,
    *,
    invariant,
):
    from postraining.fast_inference import CapturedTrainingRolloutEngine

    reference = CapturedTrainingRolloutEngine(
        policy,
        stop_ids=stop_ids,
        prompts_per_rollout=4,
        samples_per_prompt=32,
        physical_batch_size=64,
        cache_length=8192 + 33,
        temperature=0.9,
        top_k=20,
        top_p=0.95,
        compile_decode=True,
        invariant_decode=invariant,
        optimized_decode=False,
    )
    reference.prepare_generation()
    bank = reference._build_prompt_prefix_bank(
        prompts,
        prefill_batch_prompts=4,
        collect_values=False,
        storage_device=reference.generated.device,
    )
    reference._ensure_continuous_cache(bank)
    for index in range(4):
        reference._admit_prompt_rows(
            bank, index, range(index * 16, (index + 1) * 16), max_new_tokens=33
        )
    tokens = suffixes.repeat_interleave(16, dim=0)
    outputs = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if invariant:
            for index in range(5):
                reference.flash_sequence_lengths.copy_(
                    (reference.sequence_lengths + 1).int()
                )
                outputs.append(reference._predict_pending().float()[::16].clone())
                if index < 4:
                    reference._commit_pending(tokens[:, index].contiguous())
        else:
            outputs.append(reference._graph_logits.float()[::16].clone())
            for index in range(4):
                outputs.append(
                    reference.decode_without_statistics(
                        tokens[:, index].contiguous(), reference.cache
                    )
                    .float()[::16]
                    .clone()
                )
    result = torch.stack(outputs, dim=1)
    if invariant:
        for limit in (1, 6):
            rollout = reference.generate_prompt_pool(
                prompts, max_new_tokens=limit, completion_poll_steps=2
            )
            assert len(rollout.responses) == 128
            assert all(1 <= response.numel() <= limit for response in rollout.responses)
            assert (
                sum(response.numel() for response in rollout.responses)
                == rollout.useful_tokens
            )
        _report("invariant_ar_refill", useful_tokens=rollout.useful_tokens)
        # The fixed-batch API supports both logits-first statistics and
        # pending-first statistics-free schedules. Neither may consume the
        # final prompt token twice at that boundary.
        reference.prompts_per_rollout = 2
        prompt_lengths = torch.tensor(
            [prompt.numel() for prompt in prompts[:2]], device="cuda"
        ).repeat_interleave(32)
        for statistics in (False, True):
            reference.generate_prompts(
                prompts[:2], max_new_tokens=1, collect_statistics=statistics
            )
            torch.testing.assert_close(
                reference.sequence_lengths,
                prompt_lengths + int(statistics),
                rtol=0,
                atol=0,
            )
    reference.release_cache()
    del reference, bank
    gc.collect()
    return result


def test_captured_full_model_refill_capacity_and_actor_resynchronization(
    numerical_adapter,
):
    from postraining.vapo.policy import VAPOPolicy
    from postraining.vapo.model.lora import LoRAConfig
    from postraining.uno_speculative import UnoTrainingRolloutEngine
    from postraining.invariant_linear import compile_invariant

    torch.manual_seed(157)
    policy, tokenizer = VAPOPolicy.from_family("minicpm5", 
        device=torch.device("cuda"),
        lora_config=LoRAConfig(initialization="standard"),
        gradient_checkpointing=False,
    )
    policy.eval()
    text = tokenizer.encode(
        "Compute 37 times 46. Explain the arithmetic and give the answer.\n"
    )
    prompts = [
        torch.tensor((text * (length // len(text) + 1))[:length])
        for length in (1, 65, 513, 8192)
    ]
    engine = UnoTrainingRolloutEngine(
        policy,
        uno_checkpoint=str(numerical_adapter),
        uno_block_size=4,
        stop_ids=[tokenizer.eos_token_id],
        prompts_per_rollout=4,
        samples_per_prompt=32,
        physical_batch_size=64,
        cache_length=8192 + 33,
        temperature=0.9,
        top_k=20,
        top_p=0.95,
        compile_decode=True,
    )
    engine._validation_serial_forward = compile_invariant(
        lambda current, engine=engine: _native_token_forward(engine, current),
    )
    adapter_before = {
        name: value.cpu().clone()
        for name, value in engine.policy.causal_lm.uno_adapter.state_dict().items()
    }
    for limit in (1, 6, 33):
        result = engine.generate_prompt_pool(
            prompts, max_new_tokens=limit, completion_poll_steps=2
        )
        assert len(result.responses) == 128
        assert result.admission_events >= 2
        assert (
            sum(response.numel() for response in result.responses)
            == result.useful_tokens
        )
        for response in result.responses:
            assert 1 <= response.numel() <= limit
            stops = (response == tokenizer.eos_token_id).nonzero().flatten()
            assert not stops.numel() or stops.tolist() == [response.numel() - 1]
            assert response.numel() == limit or stops.numel() == 1
        assert 1 <= engine.last_uno_metrics["uno_tau"] <= 5
        _report(
            "captured_rollout",
            response_limit=limit,
            useful_tokens=result.useful_tokens,
            admissions=result.admission_events,
            **engine.last_uno_metrics,
        )
    import weakref

    old_backings = [
        weakref.ref(tensor)
        for layer in engine.cache.layers
        for tensor in (layer.key_backing, layer.value_backing, layer.keys, layer.values)
    ]
    _check_committed_cycle(engine, policy, prompts)
    assert all(reference() is None for reference in old_backings)
    _report(
        "cache_payload_released_at_phase_boundary",
        allocated=torch.cuda.memory_allocated(),
    )
    suffixes = torch.randint(policy.causal_lm.config.vocab_size, (4, 4), device="cuda")
    before = _check_clean_verifier(engine, policy, prompts, suffixes)
    with torch.no_grad():
        for name, parameter in policy.causal_lm.named_parameters():
            if name.endswith("lora_b"):
                parameter.normal_(std=0.005)
    after = _check_clean_verifier(engine, policy, prompts, suffixes)
    assert (before - after).abs().max().item() > 0.1
    result = engine.generate_prompt_pool(
        prompts, max_new_tokens=6, completion_poll_steps=2
    )
    assert len(result.responses) == 128
    for name, value in engine.policy.causal_lm.uno_adapter.state_dict().items():
        torch.testing.assert_close(value.cpu(), adapter_before[name], rtol=0, atol=0)
    engine.release_cache()
    actor_path = numerical_adapter.parent / "actor.pt"
    torch.save(
        {
            "policy": {
                "schema": "minicpm5_vapo_adapter/v6",
                "actor": policy.checkpoint_payload(),
            }
        },
        actor_path,
    )
    del engine, before, after
    gc.collect()
    torch.cuda.empty_cache()
    from postraining.fast_inference import CapturedTrainingRolloutEngine

    ar_engine = CapturedTrainingRolloutEngine(
        policy,
        stop_ids=[tokenizer.eos_token_id],
        prompts_per_rollout=4,
        samples_per_prompt=32,
        physical_batch_size=64,
        cache_length=8192 + 33,
        temperature=0.9,
        top_k=20,
        top_p=0.95,
        compile_decode=True,
    )
    ar_result = ar_engine.generate_prompt_pool(
        prompts, max_new_tokens=6, completion_poll_steps=2
    )
    assert len(ar_result.responses) == 128
    assert all(1 <= response.numel() <= 6 for response in ar_result.responses)
    assert (
        sum(response.numel() for response in ar_result.responses)
        == ar_result.useful_tokens
    )
    _report(
        "default_ar_after_uno",
        useful_tokens=ar_result.useful_tokens,
        admissions=ar_result.admission_events,
    )
    ar_engine.release_cache()
    del ar_engine
    gc.collect()
    torch.cuda.empty_cache()
    # A nonzero actor checkpoint must distill successfully, not fail merging
    # CPU-initialized adapter masters into the already-CUDA backbone.
    from scripts.train_minicpm_uno import file_sha256, load_teacher

    teacher, digest = load_teacher(
        SimpleNamespace(
            model=policy.model_id,
            revision=policy.revision,
            teacher_checkpoint=actor_path,
        ),
        torch.device("cuda"),
    )
    assert digest == file_sha256(actor_path)
    from postraining.vapo.model.lora import merge_lora_for_inference

    # The teacher's target is the merged actor, not the differently rounded
    # unmerged training representation.
    merge_lora_for_inference(policy.causal_lm)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = torch.cat((prompts[1].to("cuda"), suffixes[1]))[None]
        # Match real distillation geometry rather than the unsupported short
        # Flex decoding specialization; both references see identical tokens.
        tokens = tokens.repeat(1, (2048 + tokens.size(1) - 1) // tokens.size(1))[
            :, :2048
        ]
        start = prompts[1].numel()
        expected = policy.logits(
            policy.replay_hidden(tokens, None)[:, start : start + 4]
        ).float()
        hidden = teacher.model(input_ids=tokens, use_cache=False).last_hidden_state
        actual = teacher.lm_head(hidden[:, start : start + 4]).float()
    tv = (actual.softmax(-1) - expected.softmax(-1)).abs().sum(-1) / 2
    assert tv.max().item() < 0.03
    _report("nonzero_actor_teacher_merge", maximum_tv=tv.max().item())
    actor_path.unlink()
    del teacher, policy, hidden, actual, expected
    gc.collect()
    torch.cuda.empty_cache()
