"""Sample the forced-initial latent-thought policy on demand.

The sampler uses the trainer's exact raw Gaussian, one-way gate, and stored
thought-slot rollout. Wrapper checkpoints reconstruct sigma, gate bias, and
combiner geometry strictly.

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
from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters
from postraining.core import (
    POSTTRAIN_PROMPT_TOKENS,
    POSTTRAIN_REWARD_SCHEMA,
    POSTTRAIN_RESPONSE_TOKENS,
    answer_style,
    encode_prompt,
    load_posttraining_tokenizer,
    load_unique_math_rows,
    validate_posttraining_context_budget,
)
from postraining.latent_rollout import (
    continuation_reward,
    emitted_token_rows,
    rollout_continuations,
    trim_stream,
)
from postraining.latent_eval import evaluate_latent_math, verify_terminated_answer
from postraining.latent_thought import (
    LatentThoughtModel,
    combiner_init_kwargs_from_checkpoint,
    rollout_policy_schema_for_mode,
    validate_renderer_checkpoint,
)
from postraining.math_prompt import require_answer_fence_prompt_schema
from postraining.model_io import load_model
from postraining.train_latent_vapo import rewrite_prompts_for_answer_fence
from postraining.train_vapo import prompt_text


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
    parser.add_argument("--max-stream-steps", type=int, default=None)
    parser.add_argument("--thought-sigma", type=float, default=1.0)
    parser.add_argument(
        "--init-stop-thinking-probability", type=float, default=0.9
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
        help="token-only policy: bypass gate, noise, and thought slots",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.eval_batch_trajectories < 1:
        parser.error("--eval-batch-trajectories must be positive")
    stream_steps = (
        args.max_stream_steps
        if args.max_stream_steps is not None
        else args.max_new_tokens if args.emit_only else 4 * args.max_new_tokens
    )
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
    base_payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    backbone = load_model(args.checkpoint, device, payload=base_payload)
    sft_provenance = (
        (base_payload.get("sft") or {})
        if isinstance(base_payload, dict)
        else {}
    )
    sft_args = sft_provenance.get("args") or {}
    del base_payload
    backbone.eval()
    wrapper_step = None
    if args.wrapper_checkpoint:
        payload = torch.load(
            args.wrapper_checkpoint, map_location="cpu", weights_only=False
        )
        wrapper = LatentThoughtModel(
            backbone, **combiner_init_kwargs_from_checkpoint(payload)
        ).to(device)
        saved_mode = str(
            payload.get("args", {}).get("reasoning_mode", "latent")
        )
        validate_renderer_checkpoint(
            payload,
            args.wrapper_checkpoint,
            expected_rollout_policy_schema=rollout_policy_schema_for_mode(
                saved_mode
            ),
        )
        wrapper.load_state_dict(payload["model"], strict=True)
        wrapper_step = payload.get("step")
        print(f"policy: {args.wrapper_checkpoint} (step {payload.get('step')})")
        if saved_mode != "latent" and not args.emit_only:
            parser.error(
                f"checkpoint was trained in reasoning mode {saved_mode!r}; "
                "sample it with --emit-only"
            )
    else:
        wrapper = LatentThoughtModel(
            backbone,
            thought_sigma=args.thought_sigma,
            init_stop_thinking_probability=(
                args.init_stop_thinking_probability
            ),
        ).to(device)
        print("policy: fresh forced-initial Gaussian policy")
    wrapper.eval()
    if args.emit_only:
        print("token-only policy: gate, noise, and thought slots bypassed")

    # Fence flags come from the RL run's saved args when a wrapper
    # checkpoint is loaded, else from the base checkpoint's SFT
    # provenance: a fence-trained policy sampled through a fence-less
    # tokenizer silently decodes its fence ids away and grades near-zero.
    run_args = (
        payload.get("args", {}) if args.wrapper_checkpoint else sft_args
    )
    run_answer_fence = bool(run_args.get("answer_fence"))
    require_answer_fence_prompt_schema(
        payload if args.wrapper_checkpoint else sft_provenance,
        answer_fence=run_answer_fence,
        source=args.wrapper_checkpoint or args.checkpoint,
    )
    tokenizer = load_posttraining_tokenizer(
        backbone.architecture,
        FreshHyperparameters.tokenizer_path,
        think_tokens=bool(run_args.get("think_tokens")),
        answer_tokens=run_answer_fence,
        tokenizer_provenance=backbone.model_config.get("tokenizer_provenance"),
    )
    answer_fence_ids = (
        (tokenizer.answer_open_id, tokenizer.answer_close_id)
        if run_answer_fence
        else None
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
                    args.continuation_tokens,
                    args.temperature, args.top_p,
                    replay_storage=False,
                    record_likelihoods=False,
                    cache_dtype=torch.bfloat16,
                    pin_emit=args.emit_only,
                )
            )
        emitted_rows = emitted_token_rows(batch)
        references = reference_ids.to(device="cpu").tolist()
        prompt_tails = prompt_ids[:, -48:].to(device="cpu").tolist()
        for index, emitted in enumerate(emitted_rows):
            generated = tokenizer.decode(emitted)
            reference = tokenizer.decode(references[index])
            reward = continuation_reward(generated, reference)
            if index % args.samples == 0:
                prompt_tail = tokenizer.decode(prompt_tails[index])
                print(f"=== prompt {index // args.samples}  (…{prompt_tail!r})")
                print(f"reference: {reference!r}")
            print(f"--- sample {index % args.samples}  reward: {reward:.3f}")
            print(f"generated: {generated!r}")
            print()
        return

    if args.math_rows is not None:
        import random
        rows = load_unique_math_rows(args.math_data)
        picked = random.Random(args.seed).sample(range(len(rows)), args.math_rows)
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
        if run_answer_fence:
            # A fence-trained policy sampled under the plain-text
            # Answer: instruction is off-distribution; frame the task
            # exactly as the run did.
            selected_rows = rewrite_prompts_for_answer_fence(selected_rows)

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
            pin_emit=args.emit_only,
            answer_fence_ids=answer_fence_ids,
        )

        records = []
        for problem_index, row in enumerate(selected_rows):
            row_attempts = attempts[
                problem_index * args.samples : (problem_index + 1) * args.samples
            ]
            samples = []
            for attempt in row_attempts:
                samples.append(
                    {
                        "text": attempt["emitted_text"],
                        "segments": attempt.get("emitted_segments", []),
                        "emits": int(attempt["emitted_token_count"]),
                        "correct": bool(attempt["correct"]),
                        "prediction": attempt["parsed_answer"],
                        "terminated": bool(attempt["terminated"]),
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
                        f"emits: {sample['emits']})"
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
            f"policy_accuracy={metrics['policy_accuracy']:.4f}, "
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
        if run_answer_fence:
            row = rewrite_prompts_for_answer_fence([row])[0]
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
                replay_storage=False,
                record_likelihoods=False,
                cache_dtype=torch.bfloat16,
                pin_emit=args.emit_only,
            )
        )

    for index, emitted in enumerate(emitted_token_rows(batch)):
        cut = next((i for i, t in enumerate(emitted) if t in stop_ids), None)
        if cut is not None:
            emitted = emitted[: cut + 1]
        print(f"--- sample {index}  (emits: {len(emitted)})")
        if truth is not None:
            is_correct, prediction = verify_terminated_answer(
                emitted, truth, tokenizer, stop_ids, "aime",
                answer_fence_ids=answer_fence_ids,
            )
            print(f"verdict: {'CORRECT' if is_correct else 'wrong'} (extracted: {prediction})")
        print(tokenizer.decode(emitted))
        print()


if __name__ == "__main__":
    main()
