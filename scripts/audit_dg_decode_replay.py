#!/usr/bin/env python3
"""Record actual native decode probabilities and compare unchanged replay.

Run CUDA workloads through mlq. The frozen trainer collects one normal pool;
no backward passes, optimizer steps, or native checkpoints are permitted.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import fields
import importlib.util
import inspect
import hashlib
import json
import os
from pathlib import Path
import sys
import textwrap


class AuditFinished(Exception):
    pass


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def instrument_arena(graph, torch):
    """Add probability records to the native compiled tick, without files.

    Sampling calls and their logits are unchanged. Native production normally
    returns zero likelihoods, so capturing the unmodified batch is insufficient.
    """
    original_tick = graph._decode_tick
    original_init = graph.PinnedDecodeArena.__init__
    original_args = graph.PinnedDecodeArena._tick_args
    original_rollout = graph.PinnedDecodeArena.rollout
    tick = ast.parse(textwrap.dedent(inspect.getsource(original_tick))).body[0]
    tick.args.args.append(ast.arg(arg="audit_logprobs"))
    additions = ast.parse("""
audit_lp = (logits.float() / temperature).log_softmax(-1).gather(1, pending[:, None]).squeeze(1)
audit_logprobs.index_put_((live_rows, next_slot), audit_lp)
""").body
    for index, node in enumerate(tick.body):
        if isinstance(node, ast.Expr) and "pending.copy_(" in ast.unparse(node):
            tick.body[index + 1:index + 1] = additions
            break
    else:
        raise RuntimeError("native tick sampling structure changed")
    tick_module = ast.fix_missing_locations(ast.Module(body=[tick], type_ignores=[]))
    namespace = dict(vars(graph))
    exec(compile(tick_module, str(Path(graph.__file__)), "exec"), namespace)
    graph._decode_tick = namespace[tick.name]

    rollout = ast.parse(textwrap.dedent(inspect.getsource(original_rollout))).body[0]
    inserted = 0
    class Recorder(ast.NodeTransformer):
        def visit_Expr(self, node):
            nonlocal inserted
            text = ast.unparse(node)
            if text.startswith("self.pending[:batch].copy_("):
                inserted += 1
                return [node, *ast.parse("""
self.audit_logprobs[:batch, width - 1].copy_(
    (repeat(prefilled.logits).float() / self.temperature).log_softmax(-1)
    .gather(1, self.pending[:batch, None]).squeeze(1))
""").body]
            return node

        def visit_keyword(self, node):
            if node.arg == "old_token_logprobs":
                node.value = ast.parse("own(self.audit_logprobs[stream])", mode="eval").body
            return self.generic_visit(node)

    rollout = Recorder().visit(rollout)
    if inserted != 1:
        raise RuntimeError("native first-token sampling structure changed")
    # Clear records before each pool chunk. Capture warmup may touch dummy
    # positions; every consumed slot must subsequently be overwritten.
    rollout.body.insert(1, ast.parse("self.audit_logprobs.fill_(float('nan'))").body[0])
    rollout_module = ast.fix_missing_locations(ast.Module(body=[rollout], type_ignores=[]))
    namespace = dict(vars(graph))
    exec(compile(rollout_module, str(Path(graph.__file__)), "exec"), namespace)

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.temperature != 1.0 or self.top_p != 1.0:
            raise ValueError("this probability audit requires the actual untruncated unit-temperature training sampler")
        self.audit_logprobs = torch.full_like(self.action_mask, float("nan"))

    graph.PinnedDecodeArena.__init__ = init
    graph.PinnedDecodeArena._tick_args = lambda self, rows: (*original_args(self, rows), self.audit_logprobs)
    graph.PinnedDecodeArena.rollout = namespace[rollout.name]
    return {
        "sampling_calls_unchanged": True,
        "first_token_recorded": True,
        "compiled_tick_logprob_recorded": True,
        "record_semantics": "log_softmax of actual sampler logits before any parallel replay",
        "graph_caveat": "Added log_softmax recording can alter compiler fusion; production bitwise identity has not been validated.",
    }


def summarize(torch, actual, replay):
    difference = replay.double() - actual.double()
    quantiles = torch.tensor([0., .01, .5, .99, 1.], device=difference.device, dtype=torch.float64)
    return {
        "count": difference.numel(),
        "logprob_difference_mean": difference.mean().item(),
        "logprob_difference_rms": difference.square().mean().sqrt().item(),
        "logprob_difference_max_abs": difference.abs().max().item(),
        "logprob_difference_quantiles": torch.quantile(difference, quantiles).cpu().tolist(),
        "absolute_probability_difference_mean": (replay.exp() - actual.exp()).abs().mean().item(),
        "sampled_k3_decode_to_replay": (torch.expm1(difference) - difference).mean().item(),
        "decode_probability_mean": actual.exp().mean().item(),
        "replay_probability_mean": replay.exp().mean().item(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-rows", type=int, default=8)
    parser.add_argument("--module-after-checkpoint", type=Path)
    parser.add_argument("--module-prefix-file", type=Path,
                        help="Reuse exact captured histories for module comparisons only; native probability/Q audit remains on this checkpoint's own pool")
    cli = parser.parse_args()
    if cli.output.exists():
        raise FileExistsError("audit destination must be new")
    orchestration = load_script("dg_audit_orchestration", Path(__file__).with_name("run_dg_qtarget_study.py"))
    gate = load_script("dg_audit_gate_helpers", Path(__file__).with_name("diagnose_q_dg_gate.py"))
    study = json.loads(cli.study.read_text())
    orchestration.verify_source(study)
    trainer = orchestration.frozen_trainer(study)
    import torch
    import postraining.graph_decode as graph
    import postraining.latent_rollout as rollout
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU model execution")
    manifest_path = cli.checkpoint.parent / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = cli.checkpoint.parent.parent / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    before_sha = orchestration.sha256(cli.checkpoint)
    cli.output.mkdir(parents=True)
    # Native resume carries critic initialization from the manifest beside
    # its checkpoint. Snapshots live one directory below that manifest;
    # supply input aliases in the NEW audit directory, never modifying or
    # copying large source checkpoints into their original run. A hard link
    # keeps pathlib resolution beside this copied manifest.
    resume_input = cli.output.resolve() / "_resume_input"
    resume_input.mkdir()
    native_checkpoint = resume_input / cli.checkpoint.name
    os.link(cli.checkpoint.resolve(), native_checkpoint)
    (resume_input / "manifest.json").write_bytes(manifest_path.read_bytes())
    instrumentation = instrument_arena(graph, torch)
    disabled = gate.NativeWritesDisabled()
    result = None

    def probe(wrapper, critic, batch, optimizers, **kwargs):
        nonlocal result
        if kwargs.get("value_only") or not kwargs.get("fused_behavior_statistics"):
            raise ValueError("audit requires the original age-zero token-Q update input")
        if kwargs.get("actor_step", True) or kwargs.get("critic_step", True):
            raise RuntimeError("audit intercepted a potentially updating call")
        mask = batch.emit_mask.bool()
        decode = batch.old_token_logprobs.detach().clone()
        if not torch.isfinite(decode[mask]).all():
            raise RuntimeError("native sampler record does not cover every consumed slot")
        trainer.refresh_old_statistics(
            wrapper, critic, batch,
            max_trajectories=kwargs["replay_max_trajectories"],
            attention_budget=kwargs["replay_attention_budget"],
            bucket_multiple=kwargs["replay_bucket"],
            slot_budget=kwargs["replay_slot_budget"], replay_plan=kwargs["replay_plan"],
        )
        replay = batch.old_token_logprobs.detach()
        if not torch.isfinite(replay[mask]).all():
            raise RuntimeError("nonfinite replay record")
        taken = trainer.next_slot_tokens(batch.token_ids)
        groups = {
            "all": mask,
            "stop_50256": mask & (taken == 50256),
            "fence_50257_50260": mask & (taken >= 50257) & (taken <= 50260),
            "decode_probability_gt_999": mask & (decode.exp() > .999),
            "decode_probability_lt_01": mask & (decode.exp() < .01),
        }
        result = {
            "schema": "native_decode_parallel_replay_audit/v1",
            "checkpoint": str(cli.checkpoint.resolve()),
            "source_root": study["source_root"],
            "checkpoint_sha256_before": before_sha,
            "instrumentation": instrumentation,
            "persistent_training_updates": 0,
            "interpretation": "Unchanged actor, identical sampled prefixes and stored behavior beliefs; sampled KL estimator is not full-vocabulary KL.",
            "groups": {name: summarize(torch, decode[member], replay[member]) for name, member in groups.items() if member.any()},
        }
        result["phase"] = "decode_replay_complete"
        (cli.output / "partial_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        # Retain exact encoded prefixes and native carry records for bounded
        # follow-on action and module-substitution audits.
        chosen_rows = torch.linspace(0, batch.kind.size(0) - 1, min(cli.save_rows, batch.kind.size(0)), device=batch.kind.device).long().unique()
        selected = rollout.select_trajectory_rows(batch, chosen_rows, batch.kind.size(1)).to(torch.device("cpu"))
        selected_decode = decode[chosen_rows].cpu()
        saved = {field.name: getattr(selected, field.name) for field in fields(selected) if field.name != "replay_layout_token"}
        torch.save({"batch": saved, "decode_token_logprobs": selected_decode,
                    "original_row_indices": chosen_rows.cpu()}, cli.output / "captured_prefixes.pt")
        result["captured_prefixes"] = "captured_prefixes.pt"
        # All following analyses operate on bounded captured histories. Q
        # checks precede actor weight substitutions, so actor and critic are
        # still the paired checkpoint, including the borrowed writer.
        bounded = rollout.select_trajectory_rows(batch, chosen_rows, batch.kind.size(1))
        inputs_and_beliefs = trainer.replay_head_inputs(wrapper, bounded)
        beliefs, inputs = inputs_and_beliefs
        latents = critic.action_latents()
        # Q is centred under the policy that sampled the history.
        q_taken, directions = critic.token_values(
            bounded, rollout.policy_action_offsets_at_actions(wrapper, bounded, latents, 512))
        checks = []
        selections = []
        for row in range(min(2, bounded.kind.size(0))):
            active = bounded.emit_mask[row].nonzero().flatten().cpu().tolist()
            first = active[0]
            # First 32 thought decisions and their boundary, plus actually
            # observed structural decisions; bound the sequential prefix.
            slots = sorted(set(active[:35] + [slot for slot in active
                           if int(trainer.next_slot_tokens(bounded.token_ids)[row, slot]) >= 50256]))
            slots = [slot for slot in slots if slot <= first + 128][:48]
            selections.append((row, first, slots))
            slot_tensor = torch.tensor(slots, device=batch.kind.device)
            features = wrapper.renderer_features(inputs[row, slot_tensor], beliefs[row, slot_tensor])
            logits = wrapper.backbone.logits_from_features(features).float()
            probabilities = logits.softmax(-1)
            u = directions[row, slot_tensor].detach().float()
            targets = trainer.next_slot_tokens(bounded.token_ids)[row, slot_tensor]
            states = q_taken[row, slot_tensor].detach().float() - (u * latents[targets]).sum(-1)
            previous_precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")
            mean_q = states + (u * (probabilities @ latents)).sum(-1)
            candidates = torch.tensor([50256, 50258, 50259, 50260], device=batch.kind.device)
            candidate_q = states[:, None] + u @ latents[candidates].T
            torch.set_float32_matmul_precision(previous_precision)
            for ordinal, slot in enumerate(slots):
                # The verifier grades emitted response tokens, not prompt
                # text or padding. A query at action slot first has not yet
                # emitted anything; slot first+1 contains its first token.
                prefix = bounded.token_ids[row, first + 1:slot + 1].cpu().tolist()
                opens = [index for index, token in enumerate(prefix) if token == 50257]
                closes = [index for index, token in enumerate(prefix) if token == 50258]
                thinking = bool(opens) and (not closes or opens[-1] > closes[-1])
                think_count = len(prefix) - opens[-1] - 1 if thinking else None
                for column, token in enumerate(candidates.cpu().tolist()):
                    impossible_close = token == 50258 and thinking and think_count < source_manifest["args"]["think_min_tokens"]
                    actual_failed_eos = token == 50256 and int(targets[ordinal]) == token and float(bounded.reward_scalar[row]) == 0
                    impossible_eos = token == 50256 and 50260 not in prefix
                    checks.append({"captured_row": row, "original_row": int(chosen_rows[row]), "action_slot": slot,
                                   "taken_token": int(targets[ordinal]), "candidate_token": token,
                                   "probability": float(probabilities[ordinal, token]),
                                   "q": float(candidate_q[ordinal, column]),
                                   "policy_mean_q": float(mean_q[ordinal]),
                                   "advantage": float(candidate_q[ordinal, column] - mean_q[ordinal]),
                                   "think_count": think_count,
                                   "known_zero_terminal_return": impossible_close or actual_failed_eos or impossible_eos,
                                   "oracle_reason": "think_close_below_floor" if impossible_close else "actual_failed_terminal_eos" if actual_failed_eos else "eos_before_answer_close" if impossible_eos else None})
        result["local_q_checks"] = checks
        result["local_q_scope"] = "Paired unchanged actor/critic, stored native belief history; stable highest-precision policy mean readout. Positive AQ on a known zero-return action contradicts true nonnegative-return advantage, but does not measure gradient impact. Contract oracle grades generated response tokens only."
        result["phase"] = "local_q_checks_complete"
        (cli.output / "partial_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if cli.module_after_checkpoint:
            component = load_script("dg_audit_components", Path(__file__).with_name("audit_dg_module_changes.py"))
            before = torch.load(cli.checkpoint, map_location="cpu", mmap=True, weights_only=False)
            after = torch.load(cli.module_after_checkpoint, map_location="cpu", mmap=True, weights_only=False)
            if after["actor_objective_schema"] != before["actor_objective_schema"]:
                raise ValueError("module comparison requires matching actor schema")
            # Fresh sequential memory at each module case, not cached
            # behavior beliefs. Native BF16 step kernels remain compiled.
            compiled_step = torch.compile(wrapper.step_core, fullgraph=True, dynamic=False)
            evaluators = []
            fidelity = {}
            module_history = bounded
            module_original_rows = chosen_rows
            module_selections = selections
            if cli.module_prefix_file:
                captured = torch.load(cli.module_prefix_file, map_location="cpu", mmap=True, weights_only=False)
                module_history = rollout.LatentRolloutBatch(**captured["batch"]).to(batch.kind.device)
                module_original_rows = captured["original_row_indices"]
                module_selections = []
                history_targets = trainer.next_slot_tokens(module_history.token_ids)
                for row in range(min(2, module_history.kind.size(0))):
                    active = module_history.emit_mask[row].nonzero().flatten().cpu().tolist()
                    first = active[0]
                    slots = sorted(set(active[:35] + [slot for slot in active if int(history_targets[row, slot]) >= 50256]))
                    slots = [slot for slot in slots if slot <= first + 128][:48]
                    module_selections.append((row, first, slots))
                result["module_history_source"] = str(cli.module_prefix_file.resolve())
                result["module_history_source_sha256"] = orchestration.sha256(cli.module_prefix_file)
                result["module_history_semantics"] = "Exact external token histories, newly recomputed prefix beliefs and memory under each actor; external stored hiddens are never consumed by module evaluators."
                previous_result = json.loads((cli.module_prefix_file.parent / "result.json").read_text())
                if len(previous_result["module_positions"]) != len(module_selections):
                    raise ValueError("external prefix comparison changed selected row count")
                for (row, first, slots), previous in zip(module_selections, previous_result["module_positions"], strict=True):
                    name = f"captured_row_{row}_fresh_sequential"
                    expected_targets = previous_result["module_substitution"]["evaluators"][name]["cases"]["all"]["_positions"]["target_token_ids"]
                    actual_targets = module_history.token_ids[row, torch.tensor(slots, device=module_history.token_ids.device) + 1].cpu().tolist()
                    if previous["captured_row"] != row or previous["action_slots"] != slots or expected_targets != actual_targets:
                        raise ValueError("external prefix comparison changed positions or target tokens")
                result["external_module_position_identity_verified"] = True
            for row, first, slots in module_selections:
                positions = torch.tensor([[0, slot + 1] for slot in slots], dtype=torch.long)
                row_tokens = module_history.token_ids[row:row + 1]
                targets = row_tokens[0, positions[:, 1].to(row_tokens.device)]
                evaluator = component.forced_prefix_evaluator(
                    name=f"captured_row_{row}_fresh_sequential",
                    token_ids=row_tokens, prompt_length=first + 1,
                    prompt_valid=module_history.kind[row:row + 1, :first + 1] != rollout.PAD_SLOT,
                    positions=positions,
                    groups={"stop": targets == 50256,
                            "fence": (targets >= 50257) & (targets <= 50260)},
                    step_core=compiled_step,
                )
                evaluators.append(evaluator)
                if not cli.module_prefix_file:
                    forced_logits = evaluator.logits(wrapper).float()
                    forced_chosen = forced_logits.log_softmax(-1).gather(1, evaluator.targets[:, None]).squeeze(1)
                    recorded_chosen = decode[chosen_rows[row], torch.tensor(slots, device=decode.device)]
                    fidelity[evaluator.name] = summarize(torch, recorded_chosen, forced_chosen)
                    fidelity[evaluator.name]["scope"] = "Fresh row-one native BF16 teacher forcing versus recorded production arena; different batch/cache shapes and compiled kernels, measured rather than assumed identical."
                    del forced_logits
            result["forced_prefix_native_sampler_fidelity"] = fidelity if fidelity else {"not_applicable": "External histories have no recorded sampler probabilities under the currently loaded actor; baseline repeat measures evaluator reproducibility."}
            result["phase"] = "forced_prefix_fidelity_complete"
            (cli.output / "partial_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            result["module_substitution"] = component.audit_module_changes(
                wrapper, before["model"], after["model"], evaluators,
                restore_state=before["model"], before_label=str(before["step"]), after_label=str(after["step"]),
            )
            result["module_positions"] = [{"captured_row": row, "original_row": int(module_original_rows[row]),
                                           "prompt_length": first + 1, "action_slots": slots,
                                           "targets": module_history.token_ids[row, torch.tensor(slots, device=module_history.token_ids.device) + 1].cpu().tolist(),
                                           "token_history_sha256": hashlib.sha256(module_history.token_ids[row].cpu().numpy().tobytes()).hexdigest()}
                                          for row, first, slots in module_selections]
            result["module_after_checkpoint"] = str(cli.module_after_checkpoint.resolve())
        delta = (replay - decode).abs().masked_fill(~mask, -1)
        worst = delta.flatten().topk(min(100, int(mask.sum()))).indices
        positions = []
        for index in worst.cpu().tolist():
            row, slot = divmod(index, delta.size(1))
            positions.append({"row": row, "slot": slot, "token_id": int(taken[row, slot]),
                              "decode_logprob": float(decode[row, slot]), "replay_logprob": float(replay[row, slot]),
                              "reward": float(batch.reward_scalar[row]),
                              "source_id": int(batch.source_id[row]) if batch.source_id is not None else None})
        result["worst_positions"] = positions
        raise AuditFinished()

    argv = gate.driver_argv(study["argv"], native_checkpoint, (cli.output / "_driver").resolve())
    parsed = trainer.build_arg_parser().parse_args(orchestration.trainer_arguments(argv))
    # Native main resolves these architecture/provenance-derived defaults
    # before its resume contract. Our earlier preflight runs before that
    # resolution, so supply the recorded exact values, without relaxing any
    # resume check or altering topology.
    for name in ("memory_heads", "memory_head_dim", "think_min_tokens"):
        if getattr(parsed, name) is None:
            recorded = source_manifest["args"][name]
            if recorded is not None:
                argv.extend(["--" + name.replace("_", "-"), str(recorded)])
                setattr(parsed, name, recorded)
    orchestration.resolve_runtime_budget(trainer, parsed, source_manifest)
    trainer.validate_resume_arg_contract(source_manifest["args"], parsed)
    replacements = {"update_minibatch": probe, "step_optimizers": disabled.step,
                    "save_checkpoint": disabled.checkpoint, "JsonlLogger": disabled.logger,
                    "SummaryWriter": disabled.tensorboard}
    try:
        sys.argv = [str(trainer.__file__), *orchestration.trainer_arguments(argv)]
        with gate.instrument(trainer, replacements):
            trainer.main()
        raise RuntimeError("native driver returned before audit interception")
    except AuditFinished:
        result["native_writes_suppressed"] = vars(disabled)
        result["phase"] = "complete"
        result["checkpoint_sha256_after"] = orchestration.sha256(cli.checkpoint)
        if result["checkpoint_sha256_after"] != before_sha or disabled.suppressed_steps:
            raise RuntimeError("audit attempted a model update or changed checkpoint")
        orchestration.verify_source(study)
        (cli.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"audit": str(cli.output), "all": result["groups"]["all"]}), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
