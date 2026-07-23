"""Shared batched evaluation for the latent THINK/EMIT math policy."""

from __future__ import annotations

import math
import random
import warnings

import torch

from postraining.benchmark_report import (
    CAPTURE_PROBLEMS,
    CAPTURE_SAMPLES_PER_PROBLEM,
)
from postraining.core import answer_style, encode_prompt, verify_answer
from postraining.latent_rollout import (
    THOUGHT_SLOT,
    TOKEN_SLOT,
    emitted_token_and_kind_rows,
    emitted_token_rows,
    half_forced_group_members,
    rollout_continuations,
    trim_stream,
)
from postraining.latent_thought import THINK, LatentThoughtModel
from postraining.train_vapo import prompt_text


COMPILED_EVAL_TAIL_BATCH = 16


def verify_terminated_answer(
    emitted: list[int],
    truth: str,
    tokenizer,
    stop_ids: tuple[int, ...],
    style: str = "minerva",
    prefix_ids: tuple[int, ...] = (),
) -> tuple[bool, str]:
    """Verify a response only when it emitted BOS/EOS itself.

    ``prefix_ids`` are teacher-forced solution tokens that live in the prompt
    (the none-mode ``Answer:`` prefix): the emitted continuation alone never
    contains them, so they rejoin the decode before parsing.
    """
    stop_set = set(stop_ids)
    cut = next((i for i, token in enumerate(emitted) if token in stop_set), None)
    if cut is None:
        return False, "[UNTERMINATED]"
    return verify_answer(
        tokenizer.decode(list(prefix_ids) + emitted[: cut + 1]), truth, style
    )


@torch.no_grad()
def evaluate_latent_math(
    wrapper: LatentThoughtModel,
    tokenizer,
    rows: list[dict],
    samples: int,
    max_new_tokens: int,
    max_stream_steps: int,
    chunk: int,
    seed: int,
    device: torch.device,
    prompt_tokens: int,
    batch_trajectories: int = 128,
    compiled_step_core=None,
    captured_attempts: list[dict[str, object]] | None = None,
    answer_style_override: str | None = None,
    capture_problem_count: int = CAPTURE_PROBLEMS,
    capture_samples_per_problem: int = CAPTURE_SAMPLES_PER_PROBLEM,
    temperature: float = 1.0,
    top_p: float = 0.7,
    compact_finished: bool = False,
    compiled_tail_batch: int | None = COMPILED_EVAL_TAIL_BATCH,
    pin_emit: bool = False,
    prompt_suffix_ids: tuple[int, ...] = (),
) -> dict[str, object]:
    """Batched verifier evaluation through the latent policy itself.

    ``pin_emit`` evaluates a pinned-EMIT (cot/none reasoning mode) policy:
    no gate or thought is ever sampled and the 50/50 forced-THINK split is
    inert (every row is unforced). ``prompt_suffix_ids`` are teacher-forced
    onto the END of every truncated prompt (the none-mode ``Answer:`` prefix)
    and rejoin the decoded solution before verification.

    Generation runs the gate-conditioned rollout, so the evaluated policy is
    exactly the trained one — including its latent thinking. RNG state is
    saved and restored so evaluation never perturbs training reproducibility.
    AIME callers use temperature 1.0 / top-p 0.7 per the VAPO protocol;
    standalone inspection may pass other explicit sampling settings.

    Prompts keep their TAIL ``prompt_tokens`` exactly like the DAPO training
    path: the question and answer-format instruction sit at the end, and the
    truncation keeps prompt + stream budget inside the training context
    (PoPE extrapolates poorly beyond it). Unequal prompts are left-padded and
    multiple problem groups share each GPU rollout; ``batch_trajectories``
    bounds that rollout's row count independently of avg@k's sample chunk.

    ``compiled_step_core`` is a dynamic-shape compiled copy of the eager
    one-token model step. It is installed only for this evaluation and uses
    position tensors with narrow attention, so compilation does not force
    the much slower full-cache masked-attention path used by static CUDA
    graphs.

    When ``captured_attempts`` is provided, it is filled with the configured
    leading samples and rows in ORIGINAL dataset order. Length bucketing
    therefore changes compute order but never output attribution. The
    automatic benchmark keeps the default 4x4 panel; standalone inspection
    can capture an entire batched sample without a second model pass.

    Grading follows each row's ``answer_style`` unless
    ``answer_style_override`` forces one (AIME passes ``"aime"``: integer in
    [0, 999]).  Rows tagged with an ``extra_info.module`` additionally get a
    per-module accuracy breakdown so aggregate movement can be attributed.
    """
    if samples < 2 or samples % 2:
        raise ValueError("samples must be even for the 50/50 forced split")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if prompt_suffix_ids and len(prompt_suffix_ids) >= prompt_tokens:
        raise ValueError(
            "prompt_suffix_ids must leave room for at least one prompt token"
        )
    if batch_trajectories < 1:
        raise ValueError("batch_trajectories must be positive")
    if compiled_tail_batch is not None and compiled_tail_batch < 1:
        raise ValueError("compiled_tail_batch must be positive or None")
    if captured_attempts is not None:
        if capture_problem_count < 1 or len(rows) < capture_problem_count:
            raise ValueError(
                "answer capture requires a positive problem count no larger "
                "than the evaluated row set"
            )
        if (
            capture_samples_per_problem < 1
            or samples < capture_samples_per_problem
        ):
            raise ValueError(
                "answer capture requires a positive sample count no larger "
                "than samples per problem"
            )
        captured_attempts.clear()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    python_state = random.getstate()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    stop_ids = tuple(
        t for t in (tokenizer.eos_id(), tokenizer.bos_id()) if t >= 0
    )
    if not stop_ids:
        raise RuntimeError(
            "posttraining requires a valid BOS or EOS token for explicit "
            "trajectory termination"
        )
    correct = 0
    total = 0
    forced_correct = 0
    forced_total = 0
    unforced_correct = 0
    unforced_total = 0
    prompt_correct = [0] * len(rows)
    module_correct: dict[str, int] = {}
    module_total: dict[str, int] = {}
    think_actions = torch.zeros((), dtype=torch.float32, device=device)
    actions = torch.zeros((), dtype=torch.float32, device=device)
    emitted_counts: list[int] = []
    stream_action_counts: list[int] = []
    recurrent_steps_per_rollout: list[int] = []
    terminated_total = 0
    if compiled_step_core is not None and getattr(
        compiled_step_core, "_latent_eval_disabled", False
    ):
        compiled_step_core = None
    original_step_core = wrapper.step_core
    if compiled_step_core is not None:
        wrapper.step_core = compiled_step_core
    compile_error = None
    try:
        encoded_rows = [
            (
                encode_prompt(
                    tokenizer,
                    prompt_text(row),
                    prompt_tokens - len(prompt_suffix_ids),
                )
                + list(prompt_suffix_ids),
                row["reward_model"]["ground_truth"],
                original_index,
                prompt_text(row),
                str((row.get("extra_info") or {}).get("index", original_index)),
                answer_style_override or answer_style(row),
                (row.get("extra_info") or {}).get("module"),
            )
            for original_index, row in enumerate(rows)
        ]
        # Stable length bucketing minimizes left-padding and cache work while
        # preserving a deterministic evaluation order for a fixed dataset.
        encoded_rows.sort(key=lambda item: len(item[0]))
        force_members = (
            torch.zeros(samples, dtype=torch.bool, device=device)
            if pin_emit
            else half_forced_group_members(1, samples, device)
        )
        member_start = 0
        while member_start < samples:
            width = min(chunk, batch_trajectories, samples - member_start)
            groups_per_rollout = max(1, batch_trajectories // width)
            for row_start in range(0, len(encoded_rows), groups_per_rollout):
                row_chunk = encoded_rows[
                    row_start : row_start + groups_per_rollout
                ]
                prompt_width = max(len(item[0]) for item in row_chunk)
                rollout_width = len(row_chunk) * width
                prompt_ids = torch.zeros(
                    (len(row_chunk), prompt_width),
                    dtype=torch.long,
                    device=device,
                )
                prompt_lengths = torch.empty(
                    len(row_chunk), dtype=torch.long, device=device
                )
                for group, (prompt, *_) in enumerate(row_chunk):
                    prompt_tensor = torch.tensor(
                        prompt, dtype=torch.long, device=device
                    )
                    prompt_ids[group, -len(prompt) :] = prompt_tensor
                    prompt_lengths[group] = len(prompt)
                force_chunk = force_members[
                    member_start : member_start + width
                ].repeat(len(row_chunk))
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    batch = trim_stream(
                        rollout_continuations(
                            wrapper, prompt_ids,
                            max_new_tokens, max_stream_steps, temperature, top_p,
                            stop_ids=stop_ids or None,
                            prompt_lengths=prompt_lengths,
                            force_initial_think=force_chunk,
                            tensor_positions=compiled_step_core is not None,
                            replay_storage=False,
                            record_likelihoods=False,
                            cache_dtype=(
                                torch.bfloat16 if device.type == "cuda" else None
                            ),
                            # Keep the full batch compiled until the survivors
                            # fit one fixed B16 tail. Padding that tail with
                            # inert finished rows bounds Inductor to one extra
                            # specialization instead of arbitrary live counts.
                            compact_finished=(
                                compact_finished
                                or (
                                    compiled_step_core is not None
                                    and compiled_tail_batch is not None
                                )
                            ),
                            finished_batch_size=(
                                compiled_tail_batch
                                if compiled_step_core is not None
                                else None
                            ),
                            prompt_repeats=width,
                            pin_emit=pin_emit,
                        )
                    )
                recurrent_steps_per_rollout.append(
                    batch.stream_length - batch.prompt_length
                )
                stream_action_counts.extend(
                    int(count)
                    for count in batch.action_mask.sum(-1).cpu().tolist()
                )
                think_actions += (
                    (
                        (batch.gate_actions == THINK).float()
                        * batch.gate_mask
                    ).sum()
                )
                actions += batch.gate_mask.sum()
                capture_kinds: dict[int, list[int]] = {}
                capture_members: list[int] = []
                if (
                    captured_attempts is not None
                    and member_start < capture_samples_per_problem
                ):
                    capture_members = [
                        flat_member
                        for flat_member in range(rollout_width)
                        if row_chunk[flat_member // width][2]
                        < capture_problem_count
                        and member_start + flat_member % width
                        < capture_samples_per_problem
                    ]
                if capture_members:
                    # One packed transfer per rollout supplies both scoring
                    # tokens and traces for every captured trajectory.
                    emitted_rows, kind_rows = emitted_token_and_kind_rows(batch)
                    capture_kinds = {
                        member: kind_rows[member] for member in capture_members
                    }
                else:
                    emitted_rows = emitted_token_rows(batch)
                for flat_member, emitted in enumerate(emitted_rows):
                    emitted_counts.append(len(emitted))
                    terminated_total += int(
                        any(token in stop_ids for token in emitted)
                    )
                    group = flat_member // width
                    truth = row_chunk[group][1]
                    style = row_chunk[group][5]
                    module = row_chunk[group][6]
                    is_correct, _ = verify_terminated_answer(
                        emitted, truth, tokenizer, stop_ids, style,
                        prefix_ids=prompt_suffix_ids,
                    )
                    correct += int(is_correct)
                    total += 1
                    if module:
                        module_correct[module] = (
                            module_correct.get(module, 0) + int(is_correct)
                        )
                        module_total[module] = module_total.get(module, 0) + 1
                    member = member_start + flat_member % width
                    forced = member % 2 == 0 and not pin_emit
                    forced_correct += int(is_correct and forced)
                    forced_total += int(forced)
                    unforced_correct += int(is_correct and not forced)
                    unforced_total += int(not forced)
                    original_index = row_chunk[group][2]
                    prompt_correct[original_index] += int(is_correct)
                    if (
                        captured_attempts is not None
                        and original_index < capture_problem_count
                        and member < capture_samples_per_problem
                    ):
                        stop_cut = next(
                            (
                                index
                                for index, token in enumerate(emitted)
                                if token in stop_ids
                            ),
                            None,
                        )
                        emitted_text = tokenizer.decode(
                            list(prompt_suffix_ids) + emitted
                        )
                        _, parsed_answer = verify_answer(emitted_text, truth, style)
                        action_trace = "".join(
                            "T" if kind == THOUGHT_SLOT else "E"
                            for kind in capture_kinds[flat_member]
                            if kind in (THOUGHT_SLOT, TOKEN_SLOT)
                        )
                        runs: list[int] = []
                        current_run = 0
                        for action in action_trace:
                            if action == "T":
                                current_run += 1
                            elif current_run:
                                runs.append(current_run)
                                current_run = 0
                        if current_run:
                            runs.append(current_run)
                        total_thoughts = action_trace.count("T")
                        captured_attempts.append(
                            {
                                "problem_index": original_index,
                                "dataset_index": row_chunk[group][4],
                                "sample_index": member,
                                "prompt": row_chunk[group][3],
                                "ground_truth": str(truth),
                                "answer_style": style,
                                "emitted_token_ids": emitted,
                                "emitted_text": emitted_text,
                                "parsed_answer": parsed_answer,
                                "correct": bool(is_correct),
                                "terminated": stop_cut is not None,
                                "termination_token_id": (
                                    emitted[stop_cut] if stop_cut is not None else None
                                ),
                                "emitted_token_count": len(emitted),
                                "forced_initial_think": forced,
                                "forced_thought_count": int(forced),
                                "optional_thought_count": (
                                    total_thoughts - int(forced)
                                ),
                                "total_thought_count": total_thoughts,
                                "think_run_lengths": runs,
                                "action_trace": action_trace,
                            }
                        )
            member_start += width
    except (
        torch._dynamo.exc.BackendCompilerFailed,
        torch._dynamo.exc.TorchRuntimeError,
        torch._dynamo.exc.Unsupported,
    ) as error:
        if compiled_step_core is None:
            raise
        compile_error = error
    finally:
        wrapper.step_core = original_step_core
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state)
        random.setstate(python_state)
    if compile_error is not None:
        setattr(compiled_step_core, "_latent_eval_disabled", True)
        if captured_attempts is not None:
            captured_attempts.clear()
        warnings.warn(
            "dynamic compiled evaluation step failed; rerunning this evaluation "
            f"eager without changing training RNG: {compile_error}",
            RuntimeWarning,
            stacklevel=2,
        )
        metrics = evaluate_latent_math(
            wrapper,
            tokenizer,
            rows,
            samples,
            max_new_tokens,
            max_stream_steps,
            chunk,
            seed,
            device,
            prompt_tokens,
            batch_trajectories,
            compiled_step_core=None,
            captured_attempts=captured_attempts,
            answer_style_override=answer_style_override,
            capture_problem_count=capture_problem_count,
            capture_samples_per_problem=capture_samples_per_problem,
            temperature=temperature,
            top_p=top_p,
            compact_finished=compact_finished,
            compiled_tail_batch=compiled_tail_batch,
            pin_emit=pin_emit,
            prompt_suffix_ids=prompt_suffix_ids,
        )
        metrics["compile_fallback"] = True
        return metrics
    if captured_attempts is not None:
        captured_attempts.sort(
            key=lambda attempt: (
                int(attempt["problem_index"]),
                int(attempt["sample_index"]),
            )
        )

    def summarize(values: list[int], prefix: str) -> dict[str, float | int]:
        if not values:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_p95": 0,
                f"{prefix}_max": 0,
            }
        ordered = sorted(values)
        p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        return {
            f"{prefix}_mean": sum(ordered) / len(ordered),
            f"{prefix}_p95": ordered[p95_index],
            f"{prefix}_max": ordered[-1],
        }

    metrics: dict[str, object] = {
        "accuracy": correct / max(total, 1),
        "samples": total,
        "prompt_groups": len(prompt_correct),
        "prompt_any_correct_fraction": sum(
            count > 0 for count in prompt_correct
        ) / max(len(prompt_correct), 1),
        "prompt_mixed_reward_fraction": sum(
            0 < count < samples for count in prompt_correct
        ) / max(len(prompt_correct), 1),
        "prompt_all_correct_fraction": sum(
            count == samples for count in prompt_correct
        ) / max(len(prompt_correct), 1),
        "prompt_zero_correct_fraction": sum(
            count == 0 for count in prompt_correct
        ) / max(len(prompt_correct), 1),
        "within_group_reward_std": sum(
            math.sqrt((count / samples) * (1.0 - count / samples))
            for count in prompt_correct
        ) / max(len(prompt_correct), 1),
        "think_fraction": float(think_actions / actions.clamp_min(1.0)),
        "forced_initial_accuracy": forced_correct / max(forced_total, 1),
        "unforced_initial_accuracy": unforced_correct / max(unforced_total, 1),
        "forced_initial_fraction": forced_total / max(total, 1),
        "ended_fraction": terminated_total / max(total, 1),
        **summarize(emitted_counts, "emitted_tokens"),
        **summarize(stream_action_counts, "stream_actions"),
        **summarize(recurrent_steps_per_rollout, "recurrent_steps_per_rollout"),
        "compiled": compiled_step_core is not None,
        "compile_fallback": False,
        "pin_emit": pin_emit,
        "finished_compaction": (
            f"compiled_tail_b{compiled_tail_batch}"
            if compiled_step_core is not None and compiled_tail_batch is not None
            else ("exact" if compact_finished else "none")
        ),
        "sampling_schema": (
            "global_rng_compacted_tail/v1"
            if (
                compact_finished
                or (
                    compiled_step_core is not None
                    and compiled_tail_batch is not None
                )
            )
            else "global_rng_fixed_batch/v1"
        ),
    }
    if module_total:
        metrics["module_accuracy"] = {
            module: module_correct.get(module, 0) / count
            for module, count in sorted(module_total.items())
        }
    return metrics
