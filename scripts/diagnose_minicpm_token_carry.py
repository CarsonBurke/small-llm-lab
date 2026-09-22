#!/usr/bin/env python3
"""Queue a bounded real-MiniCPM stored-carry correctness diagnostic, not a quality run."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORKER_ENV = "PARAMETER_GOLF_TOKEN_CARRY_DIAGNOSTIC"


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worker(args):
    if not args.queue_token or os.environ.get(WORKER_ENV) != args.queue_token:
        raise RuntimeError("invoke without --worker to submit through mlq")
    import faulthandler
    import torch
    from postraining.fast_inference import CapturedTrainingRolloutEngine
    from postraining.vapo.policy import (
        VAPOCritic,
        VAPOPolicy,
        TrajectoryRecord,
        collate_replay_microbatch,
    )
    from postraining.vapo.model.hf import (
        enable_packed_replay_attention,
        enable_replay_mlp_compilation,
    )
    from postraining.vapo.model.lora import (
        LoRAConfig,
        adapter_state_dict,
        load_adapter_state_dict,
    )
    from postraining.train_minicpm_vapo import (
        _replay_hidden, _action_logprobs, _stop_ids, refresh_behavior_statistics,
        update_step, save_checkpoint, build_parser,
    )

    started = time.monotonic()
    report = {
        "schema": "minicpm-token-carry-diagnostic/v3", "status": "initializing",
        "scope": "Correctness only: two short prompts, eight sampled tokens per row, one optimizer transition. No quality inference.",
        "sampling": {"temperature": 0.9, "top_p": 0.95, "top_k": 20},
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "postraining/vapo/policy.py", "postraining/token_carry.py",
                "postraining/fast_inference.py", "postraining/train_minicpm_vapo.py",
                "scripts/diagnose_minicpm_token_carry.py",
            )
        },
        "checks": {},
    }
    save(args.output, report)
    faulthandler.dump_traceback_later(120, repeat=True)

    def phase(name):
        report["status"] = name
        report["wall_seconds"] = time.monotonic() - started
        save(args.output, report)
        print(json.dumps({"phase": name, "seconds": report["wall_seconds"]}), flush=True)

    def seed():
        torch.manual_seed(1337)
        torch.cuda.manual_seed_all(1337)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        seed()
        torch.set_float32_matmul_precision("high")
        device = torch.device("cuda")
        config = LoRAConfig()
        policy, tokenizer = VAPOPolicy.from_family("minicpm5", 
            device=device, lora_config=config, gradient_checkpointing=False,
            token_carry=True,
        )
        policy.eval()
        prompts = [
            torch.tensor(tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=True,
                add_generation_prompt=True, enable_thinking=True, return_dict=True,
            )["input_ids"], dtype=torch.long)
            for text in ("What is 2 + 3?", "What is three times seven? Explain briefly.")
        ]
        report["prompt_ids"] = [prompt.tolist() for prompt in prompts]
        stop_ids = _stop_ids(policy, tokenizer)
        def make_engine(source):
            return CapturedTrainingRolloutEngine(
                source, stop_ids=stop_ids, prompts_per_rollout=2, samples_per_prompt=1,
                cache_length=max(map(len, prompts)) + 8,
                temperature=0.9, top_k=20, top_p=0.95, compile_decode=True,
            )
        phase("native_identity_control")
        native = copy.deepcopy(policy)
        native.token_carry = False
        native_engine = make_engine(native)
        seed()
        native_generation = native_engine.generate_prompts(prompts, max_new_tokens=8)
        native_engine.release_cache()
        del native_engine, native
        gc.collect()
        torch.cuda.empty_cache()
        phase("carry_identity_control")
        engine = make_engine(policy)
        seed()
        identity_generation = engine.generate_prompts(prompts, max_new_tokens=8)
        torch.testing.assert_close(identity_generation[0], native_generation[0], rtol=0, atol=0)
        torch.testing.assert_close(identity_generation[1], native_generation[1], rtol=0, atol=0)
        report["checks"]["identity_matches_native_tokens_and_logprobs"] = True
        report["checks"]["combiner_parameters"] = sum(p.numel() for p in policy.token_combiner.parameters())
        phase("nonzero_carry_and_continuous_refill")
        with torch.no_grad():
            policy.token_combiner.carry.weight.copy_(torch.eye(policy.causal_lm.config.hidden_size, device=device) * 0.01)
        seed()
        carried = engine.generate_prompts(prompts, max_new_tokens=8)
        carried_hiddens = engine.last_carry_hiddens
        assert carried_hiddens is not None and not carried_hiddens.requires_grad
        assert engine._decode_graph is not None
        # Warm the statistics-free graph before matching RNG streams. Four logical
        # requests over two physical lanes exercise admission and carry reset.
        pool_prompts = prompts + prompts[::-1]
        engine.generate_prompt_pool(pool_prompts, max_new_tokens=8, prefill_batch_prompts=2)
        seed()
        pool_a = engine.generate_prompt_pool(pool_prompts, max_new_tokens=8, prefill_batch_prompts=2)
        seed()
        pool_b = engine.generate_prompt_pool(pool_prompts, max_new_tokens=8, prefill_batch_prompts=2)
        for first, second in zip(pool_a.responses, pool_b.responses, strict=True):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert pool_a.carry_hiddens is not None and pool_b.carry_hiddens is not None
        for first, second, response in zip(
            pool_a.carry_hiddens, pool_b.carry_hiddens, pool_b.responses, strict=True,
        ):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            assert first.shape == (response.numel(), policy.causal_lm.config.hidden_size)
            assert first.device.type == "cpu" and first.dtype == torch.bfloat16
        assert pool_b.admission_events >= 2 and engine._continuous_decode_graph is not None
        report["checks"]["continuous_refill_repeatable"] = True
        report["checks"]["admission_events"] = pool_b.admission_events
        engine.release_cache()
        records = []
        for row, prompt in enumerate(prompts):
            response = carried[0][row]
            stopping = [i + 1 for i, token in enumerate(response.tolist()) if token in stop_ids]
            length = stopping[0] if stopping else len(response)
            response = response[:length]
            records.append(TrajectoryRecord.from_device(
                token_ids=torch.cat((prompt, response)), prompt_length=len(prompt),
                old_logprobs=carried[1][row, :length], old_values=torch.zeros(length),
                correct=False, text=tokenizer.decode(response.tolist()),
                carry_hiddens=carried_hiddens[row, :length],
            ))
        report["responses"] = [record.text for record in records]
        enable_packed_replay_attention(policy.causal_lm)
        enable_replay_mlp_compilation(policy.causal_lm)
        batch = collate_replay_microbatch(records, [0, 1], pad_token_id=1, device=device)
        frozen_carries = batch.carry_hiddens.clone()
        assert not batch.carry_hiddens.requires_grad
        report["checks"]["stored_carries_survive_rollout_release"] = True

        def logprobs(side):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = _replay_hidden(side, batch)
                actions = hidden[batch.action_batch_indices, batch.action_positions]
                return _action_logprobs(side, actions, batch, chunk_tokens=16)

        phase("parallel_stored_carry_replay")
        with torch.no_grad():
            behavior = logprobs(policy)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            differentiable = logprobs(policy)
        rollout_logprobs = torch.cat([record.old_logprobs for record in records]).to(device)
        rollout_error = (behavior - rollout_logprobs).abs().max().item()
        packed_error = (differentiable.detach() - behavior).abs().max().item()
        report["checks"]["rollout_replay_max_abs_logprob"] = rollout_error
        report["checks"]["no_grad_grad_replay_max_abs_logprob"] = packed_error
        # Existing fused rollout versus unfused BF16 replay are not bit-identical.
        # This explicit diagnostic bound is not a claim of exact sampler parity.
        assert rollout_error < 0.25, rollout_error
        assert packed_error < 0.02, packed_error
        (-differentiable.mean()).backward()
        gradients = {}
        for name, parameter in policy.token_combiner.named_parameters():
            assert parameter.grad is not None, name
            gradients[name] = parameter.grad.float().norm().item()
        assert all(0 < value < float("inf") for value in gradients.values()), gradients
        report["checks"]["combiner_gradient_norms"] = gradients
        policy.zero_grad(set_to_none=True)
        del differentiable
        with torch.no_grad():
            policy.token_combiner.carry.weight.mul_(2)
            changed = logprobs(policy)
        torch.testing.assert_close(batch.carry_hiddens, frozen_carries, rtol=0, atol=0)
        report["checks"]["optimizer_inputs_remain_fixed_behavior_data"] = True
        change = (changed - behavior).abs().max().item()
        assert change > 1e-4, change
        report["checks"]["current_weight_replay_logprob_change"] = change
        phase("independent_critic_and_optimizer_transition")
        critic = VAPOCritic.from_family("minicpm5", 
            device=device, lora_config=config, gradient_checkpointing=False,
            shared_frozen_source=policy.causal_lm, token_carry=True,
        )
        load_adapter_state_dict(critic.causal_lm, adapter_state_dict(policy.causal_lm))
        critic.token_combiner.load_state_dict(policy.token_combiner.state_dict())
        enable_packed_replay_attention(critic.causal_lm)
        enable_replay_mlp_compilation(critic.causal_lm)
        assert policy.token_combiner.carry.weight.data_ptr() != critic.token_combiner.carry.weight.data_ptr()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            critic_hidden = _replay_hidden(critic, batch)
        critic_hidden[..., 0].float().sum().backward()
        assert critic.token_combiner.carry.weight.grad is not None
        assert critic.token_combiner.carry.weight.grad.norm().item() > 0
        assert policy.token_combiner.carry.weight.grad is None
        critic.zero_grad(set_to_none=True)
        del critic_hidden
        report["checks"]["critic_owns_combiner_and_cannot_backpropagate_to_actor"] = True
        refreshed, _ = refresh_behavior_statistics(
            policy, critic, records, replay_token_budget=1024,
            replay_max_trajectories=2, logit_chunk_tokens=16,
        )
        actor_optimizer = torch.optim.AdamW(list(policy.actor_parameters()), lr=1e-6, fused=True, weight_decay=0)
        critic_optimizer = torch.optim.AdamW(list(critic.backbone_parameters()) + list(critic.value_head.parameters()), lr=1e-5, fused=True, weight_decay=0)
        before = policy.token_combiner.carry.weight.detach().clone()
        update = update_step(
            policy, critic, refreshed, actor_optimizer, critic_optimizer,
            optimizer_minibatches=1, replay_token_budget=1024, replay_max_trajectories=2,
            logit_chunk_tokens=16, clip_low=0.2, clip_high=0.28, value_coefficient=1,
            nextlat_horizon=2, nextlat_samples=1, nextlat_mse_coefficient=1,
            nextlat_kl_coefficient=1, nextlat_kl_chunk_tokens=16,
            train_nextlat=False, grad_clip_norm=1,
        )
        movement = (policy.token_combiner.carry.weight.detach() - before).abs().max().item()
        assert movement > 0, movement
        report["checks"]["actor_optimizer_carry_max_change"] = movement
        report["optimizer_transition"] = update
        assert update["clip_fraction"] == 0, update
        assert update["ratio_abs_log_max"] < 0.02, update
        phase("checkpoint_roundtrip")
        checkpoint_path = args.output.with_suffix(".pt")
        training_args = build_parser().parse_args([
            "--token-carry", "--answer-reserve-tokens", "0", "--no-train-nextlat",
        ])
        save_checkpoint(
            checkpoint_path, policy, critic, actor_optimizer, critic_optimizer,
            step=1, cursor=2, warmup_step=0, pending_records=refreshed,
            pending_epoch=0, args=training_args, data_sha256="diagnostic-no-dataset",
            corpus_identity="diagnostic-only-not-resumable",
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert state["policy"]["schema"] == "minicpm5_vapo_token_carry/v4"
        for original, loaded in zip(refreshed, state["pending_records"], strict=True):
            torch.testing.assert_close(original.carry_hiddens, loaded.carry_hiddens, rtol=0, atol=0)
        with torch.no_grad():
            expected = logprobs(policy)
            policy.token_combiner.carry.weight.zero_()
        policy.load_token_carry_state_dict(state["policy"]["actor"])
        critic.load_token_carry_state_dict(state["policy"]["critic"])
        with torch.no_grad():
            restored = logprobs(policy)
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)
        report["checks"]["checkpoint_restores_predictions"] = True
        report["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        # Diagnostic state is not a trained artifact; retain metrics, not synthetic checkpoints.
        checkpoint_path.unlink()
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        save(args.output, report)
        faulthandler.cancel_dump_traceback_later()
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--queue-token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.worker:
        return worker(args)
    token = secrets.token_hex(16)
    return subprocess.run([
        "mlq", "submit", "--name", "minicpm-token-carry-correctness",
        "--max-parallel-runs", "1", "--time-limit", "5m", "--max-attempts", "1",
        "--cwd", str(ROOT), "--env", f"{WORKER_ENV}={token}", "--",
        sys.executable, str(Path(__file__).resolve()), "--worker", "--queue-token", token,
        "--output", str(args.output),
    ], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
