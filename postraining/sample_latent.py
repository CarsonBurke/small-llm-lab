"""Sample the latent THINK/EMIT policy on demand and inspect its outputs.

Generates through the same gate-conditioned rollout the trainer and the AIME
eval use, so what you see is exactly the trained policy — including where it
chose to think.  Works against the pretraining checkpoint alone (untrained
gate) or with a latent-VAPO checkpoint layered on top.

    # An AIME problem by index, 4 samples (the base --checkpoint is resolved
    # from the run's manifest.json when omitted):
    python3 -m postraining.sample_latent \
        --wrapper-checkpoint postraining/runs/<name>/latent_vapo_checkpoint.pt \
        --aime-row 0 --samples 4

    # Arbitrary text against the raw pretraining checkpoint:
    python3 -m postraining.sample_latent \
        --checkpoint ablation_results/<run>/pretraining_checkpoint.pt \
        --prompt "The sky"
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import torch

import train_gpt as baseline  # noqa: F401  (import order: patches must load first)
from fresh_lejepa_train import FreshHyperparameters
from postraining.core import (
    POSTTRAIN_PROMPT_TOKENS,
    POSTTRAIN_REWARD_SCHEMA,
    POSTTRAIN_RESPONSE_TOKENS,
    POSTTRAIN_STREAM_TOKENS,
    answer_style,
    encode_prompt,
    load_posttraining_tokenizer,
    load_unique_math_rows,
    validate_posttraining_context_budget,
)
from postraining.latent_rollout import (
    THOUGHT_SLOT,
    TOKEN_SLOT,
    continuation_reward,
    emitted_token_and_kind_rows,
    half_forced_group_members,
    rollout_continuations,
    trim_stream,
)
from postraining.latent_eval import evaluate_latent_math, verify_terminated_answer
from postraining.latent_thought import (
    LatentThoughtModel,
    migrate_legacy_wrapper_checkpoint,
    validate_renderer_checkpoint,
    wrapper_init_kwargs_from_checkpoint,
)
from postraining.model_io import load_model
from postraining.train_vapo import prompt_text


def gate_trace_from_kinds(kinds: list[int]) -> str:
    """Compact action trace from an already-copied continuation kind row."""
    symbols = {TOKEN_SLOT: "E", THOUGHT_SLOT: "t"}
    return "".join(symbols.get(kind, "") for kind in kinds)


def think_run_lengths_from_trace(trace: str) -> list[int]:
    """Lengths of contiguous latent-thought runs in an E/t action trace."""
    runs: list[int] = []
    current = 0
    for action in trace:
        if action == "t":
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def decode_trace_with_think_markers(
    tokenizer,
    emitted: list[int],
    trace: str,
    stop_ids: tuple[int, ...] = (),
) -> str:
    """CPU-only marked decode from an action trace and emitted token ids."""
    parts: list[str] = []
    segment: list[int] = []
    emitted_index = 0
    run = 0
    at_start = True

    def flush_segment() -> None:
        nonlocal segment, at_start
        if not segment:
            return
        text = tokenizer.decode(segment)
        if not at_start and tokenizer.id_to_piece(segment[0]).startswith("▁"):
            text = " " + text
        parts.append(text)
        segment = []
        at_start = False

    for action in trace.upper():
        if action == "T":
            flush_segment()
            run += 1
            continue
        if action != "E":
            continue
        if run:
            parts.append(f"{run}🪙")
            at_start = False
            run = 0
        if emitted_index >= len(emitted):
            raise ValueError("action trace contains more EMITs than token ids")
        token = emitted[emitted_index]
        emitted_index += 1
        if token in stop_ids:
            flush_segment()
            parts.append(tokenizer.id_to_piece(token))
            break
        segment.append(token)
    flush_segment()
    if run:
        parts.append(f"{run}🪙")
    if emitted_index != len(emitted):
        raise ValueError("emitted token ids outnumber action-trace EMITs")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=None,
        help="base pretraining checkpoint; defaults to the one recorded in "
        "the wrapper checkpoint's run manifest",
    )
    parser.add_argument("--wrapper-checkpoint", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--aime-row", type=int, default=None)
    parser.add_argument("--aime-data", default="postraining/data/aime-2024.parquet")
    parser.add_argument(
        "--fineweb", type=int, default=None, metavar="N",
        help="sample N Tier-0 training pairs: real FineWeb prompt + reference "
        "continuation, scored with the actual continuation reward",
    )
    parser.add_argument(
        "--math-rows", type=int, default=None, metavar="N",
        help="sample N seeded-random problems from --math-data with "
        "--samples rollouts each, verifier-scored",
    )
    parser.add_argument(
        "--math-data", default="postraining/data/deepmind-interpolate-rl.parquet"
    )
    parser.add_argument(
        "--json-out", default=None,
        help="also write the sampled records as JSON (math-rows mode only)",
    )
    parser.add_argument(
        "--prompt-tokens", type=int, default=POSTTRAIN_PROMPT_TOKENS
    )
    parser.add_argument("--fineweb-prompt-tokens", type=int, default=256)
    parser.add_argument("--continuation-tokens", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument(
        "--max-new-tokens", type=int, default=POSTTRAIN_RESPONSE_TOKENS
    )
    # Total generated-slot budget (thinks + emits); 0 = 4x the emit cap.
    parser.add_argument(
        "--max-stream-steps", type=int, default=POSTTRAIN_STREAM_TOKENS
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument(
        "--answer-style", choices=("auto", "aime", "exact", "minerva"),
        default="auto",
        help="math-row verifier override; use 'aime' for the standard integer "
        "in [0, 999] scorer",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-batch-trajectories", type=int, default=128)
    parser.add_argument(
        "--eval-compile", action=argparse.BooleanOptionalAction, default=True,
        help="dynamically compile the narrow-cache one-token evaluation step",
    )
    parser.add_argument(
        "--print-samples", action="store_true",
        help="print every math-row response; JSON output remains complete "
        "without this expensive console dump",
    )
    parser.add_argument(
        "--emit-only", action="store_true",
        help="pin optional gate decisions to EMIT; exactly half of each "
        "group still receives the forced initial latent thought",
    )
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        parser.error("--samples must be even for the 50/50 forced split")
    if args.eval_batch_trajectories < 1:
        parser.error("--eval-batch-trajectories must be positive")
    stream_steps = args.max_stream_steps or 4 * args.max_new_tokens
    validate_posttraining_context_budget(args.prompt_tokens, stream_steps)
    modes = sum(
        value is not None
        for value in (args.prompt, args.aime_row, args.fineweb, args.math_rows)
    )
    if modes != 1:
        parser.error(
            "exactly one of --prompt, --aime-row, --fineweb, or --math-rows "
            "is required"
        )
    if args.checkpoint is None:
        if args.wrapper_checkpoint is None:
            parser.error("--checkpoint is required without --wrapper-checkpoint")
        manifest = Path(args.wrapper_checkpoint).parent / "manifest.json"
        if not manifest.exists():
            parser.error(f"cannot resolve the base checkpoint: {manifest} not found")
        args.checkpoint = json.loads(manifest.read_text())["base"]["checkpoint"]
        print(f"base checkpoint (from manifest): {args.checkpoint}")

    device = torch.device("cuda")
    backbone = load_model(args.checkpoint, device)
    backbone.eval()
    wrapper_step = None
    if args.wrapper_checkpoint:
        payload = torch.load(
            args.wrapper_checkpoint, map_location="cpu", weights_only=False
        )
        wrapper = LatentThoughtModel(
            backbone, **wrapper_init_kwargs_from_checkpoint(payload)
        ).to(device)
        migrate_legacy_wrapper_checkpoint(payload, wrapper)
        validate_renderer_checkpoint(
            payload,
            args.wrapper_checkpoint,
            expected_thought_input_schema=wrapper.thought_input_schema,
        )
        wrapper.load_state_dict(payload["model"], strict=True)
        wrapper_step = payload.get("step")
        print(f"policy: {args.wrapper_checkpoint} (step {payload.get('step')})")
    else:
        wrapper = LatentThoughtModel(backbone).to(device)
        print("policy: untrained heads over the pretraining checkpoint")
    wrapper.eval()
    if args.emit_only:
        # Zero gate weight + saturated bias: EMIT with probability ~1.
        with torch.no_grad():
            wrapper.gate.head.weight.zero_()
            wrapper.gate.head.bias.fill_(30.0)
        print("gate pinned to EMIT except for the forced half of each group")

    tokenizer = load_posttraining_tokenizer(
        backbone.architecture, FreshHyperparameters.tokenizer_path
    )

    if args.fineweb is not None:
        from postraining.train_latent_vapo import sample_prompt_batch

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        loader = baseline.DistributedTokenLoader(
            FreshHyperparameters.train_files, 0, 1, device
        )
        prompt_ids, reference_ids = sample_prompt_batch(
            loader, args.fineweb_prompt_tokens, args.continuation_tokens,
            args.fineweb, args.samples, FreshHyperparameters.train_seq_len,
        )
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16
        ):
            batch = trim_stream(
                rollout_continuations(
                    wrapper, prompt_ids, args.continuation_tokens,
                    args.max_stream_steps or 4 * args.continuation_tokens,
                    args.temperature, args.top_p,
                    force_initial_think=half_forced_group_members(
                        args.fineweb, args.samples, device
                    ),
                    replay_storage=False,
                    record_likelihoods=False,
                    cache_dtype=torch.bfloat16,
                )
            )
        emitted_rows, kind_rows = emitted_token_and_kind_rows(batch)
        references = reference_ids.to(device="cpu").tolist()
        prompt_tails = prompt_ids[:, -48:].to(device="cpu").tolist()
        for index, emitted in enumerate(emitted_rows):
            generated = tokenizer.decode(emitted)
            reference = tokenizer.decode(references[index])
            reward = continuation_reward(generated, reference)
            trace = gate_trace_from_kinds(kind_rows[index])
            if index % args.samples == 0:
                prompt_tail = tokenizer.decode(prompt_tails[index])
                print(f"=== prompt {index // args.samples}  (…{prompt_tail!r})")
                print(f"reference: {reference!r}")
            marked = decode_trace_with_think_markers(
                tokenizer, emitted, trace,
            )
            print(f"--- sample {index % args.samples}  reward: {reward:.3f}  "
                  f"(thinks: {trace.count('t')})")
            print(f"generated: {marked!r}")
            print()
        return

    if args.math_rows is not None:
        import random
        rows = load_unique_math_rows(args.math_data)
        picked = random.Random(args.seed).sample(range(len(rows)), args.math_rows)
        stop_ids = tuple(
            t for t in (tokenizer.eos_id(), tokenizer.bos_id()) if t >= 0
        )
        selected_rows = []
        for row_index in picked:
            row = rows[row_index]
            selected_rows.append(
                {
                    **row,
                    "extra_info": {
                        **(row.get("extra_info") or {}),
                        "index": row_index,
                    },
                }
            )

        compiled_step_core = None
        if args.eval_compile:
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit, 64
            )
            compiled_step_core = torch.compile(
                wrapper.step_core,
                mode="max-autotune-no-cudagraphs",
                fullgraph=True,
                dynamic=True,
            )
        attempts: list[dict[str, object]] = []
        metrics = evaluate_latent_math(
            wrapper,
            tokenizer,
            selected_rows,
            samples=args.samples,
            max_new_tokens=args.max_new_tokens,
            max_stream_steps=stream_steps,
            chunk=args.samples,
            seed=args.seed,
            device=device,
            prompt_tokens=args.prompt_tokens,
            batch_trajectories=args.eval_batch_trajectories,
            compiled_step_core=compiled_step_core,
            captured_attempts=attempts,
            answer_style_override=(
                None if args.answer_style == "auto" else args.answer_style
            ),
            capture_problem_count=len(selected_rows),
            capture_samples_per_problem=args.samples,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        records = []
        for problem_index, row in enumerate(selected_rows):
            row_attempts = attempts[
                problem_index * args.samples : (problem_index + 1) * args.samples
            ]
            samples = []
            for attempt in row_attempts:
                trace = str(attempt["action_trace"]).replace("T", "t")
                emitted = [int(token) for token in attempt["emitted_token_ids"]]
                samples.append(
                    {
                        "text": decode_trace_with_think_markers(
                            tokenizer, emitted, trace, stop_ids=stop_ids
                        ),
                        "emitted_text": attempt["emitted_text"],
                        "trace": trace,
                        "thinks": int(attempt["total_thought_count"]),
                        "emits": int(attempt["emitted_token_count"]),
                        "correct": bool(attempt["correct"]),
                        "prediction": attempt["parsed_answer"],
                        "terminated": bool(attempt["terminated"]),
                        "forced_initial_think": bool(
                            attempt["forced_initial_think"]
                        ),
                        "optional_thought_count": int(
                            attempt["optional_thought_count"]
                        ),
                        "think_run_lengths": list(attempt["think_run_lengths"]),
                    }
                )
            records.append(
                {
                    "row_index": picked[problem_index],
                    "problem": prompt_text(row),
                    "ground_truth": row["reward_model"]["ground_truth"],
                    "answer_style": (
                        answer_style(row)
                        if args.answer_style == "auto"
                        else args.answer_style
                    ),
                    "samples": samples,
                }
            )
            if args.print_samples:
                correct = sum(sample["correct"] for sample in samples)
                print(
                    f"=== row {picked[problem_index]}: "
                    f"{correct}/{len(samples)} correct "
                    f"(truth: {records[-1]['ground_truth']})"
                )
                print(textwrap.shorten(records[-1]["problem"], 200))
                for sample_index, sample in enumerate(samples):
                    print(
                        f"--- sample {sample_index}  "
                        f"{'CORRECT' if sample['correct'] else 'wrong'} "
                        f"(extracted: {sample['prediction']}, "
                        f"thinks: {sample['thinks']})"
                    )
                    print(sample["text"])
                print()

        terminated = sum(
            sample["terminated"]
            for record in records
            for sample in record["samples"]
        ) / max(len(records) * args.samples, 1)
        print(
            f"sampled {len(records)} problems x {args.samples}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"terminated={terminated:.4f}"
        )
        if args.json_out:
            payload = {
                "wrapper_checkpoint": args.wrapper_checkpoint,
                "wrapper_step": wrapper_step,
                "reward_schema": POSTTRAIN_REWARD_SCHEMA,
                "math_data": args.math_data,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "samples_per_problem": args.samples,
                "max_new_tokens": args.max_new_tokens,
                "max_stream_steps": stream_steps,
                "eval_batch_trajectories": args.eval_batch_trajectories,
                "eval_compile_requested": bool(args.eval_compile),
                "eval_compiled": bool(metrics["compiled"]),
                "eval_compile_fallback": bool(metrics["compile_fallback"]),
                "seed": args.seed,
                "metrics": metrics,
                "records": records,
            }
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(json.dumps(payload, indent=1))
            print(f"wrote {args.json_out}")
        return

    truth = None
    if args.aime_row is not None:
        rows = load_unique_math_rows(args.aime_data)
        row = rows[args.aime_row]
        text = prompt_text(row)
        truth = row["reward_model"]["ground_truth"]
        print(f"AIME row {args.aime_row} (ground truth: {truth})")
    else:
        text = args.prompt
    print(f"prompt ({len(text)} chars): {textwrap.shorten(text, 200)}\n")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    stop_ids = tuple(t for t in (tokenizer.eos_id(), tokenizer.bos_id()) if t >= 0)
    prompt_ids = torch.tensor(
        encode_prompt(tokenizer, text, args.prompt_tokens),
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16
    ):
        batch = trim_stream(
            rollout_continuations(
                wrapper,
                prompt_ids[None].expand(args.samples, -1),
                args.max_new_tokens,
                stream_steps,
                args.temperature,
                args.top_p,
                stop_ids=stop_ids or None,
                force_initial_think=half_forced_group_members(
                    1, args.samples, device
                ),
                replay_storage=False,
                record_likelihoods=False,
                cache_dtype=torch.bfloat16,
            )
        )

    emitted_rows, kind_rows = emitted_token_and_kind_rows(batch)
    for index, emitted in enumerate(emitted_rows):
        cut = next((i for i, t in enumerate(emitted) if t in stop_ids), None)
        if cut is not None:
            emitted = emitted[: cut + 1]
        trace = gate_trace_from_kinds(kind_rows[index])
        thinks = trace.count("t")
        marked = decode_trace_with_think_markers(
            tokenizer, emitted, trace, stop_ids=stop_ids,
        )
        print(f"--- sample {index}  (thinks: {thinks}, emits: {trace.count('E')})")
        print(f"trace: {trace}")
        if truth is not None:
            is_correct, prediction = verify_terminated_answer(
                emitted, truth, tokenizer, stop_ids, "aime"
            )
            print(f"verdict: {'CORRECT' if is_correct else 'wrong'} (extracted: {prediction})")
        print(marked)
        print()


if __name__ == "__main__":
    main()
