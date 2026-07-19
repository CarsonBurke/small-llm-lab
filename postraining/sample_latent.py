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
    POSTTRAIN_RESPONSE_TOKENS,
    POSTTRAIN_STREAM_TOKENS,
    encode_prompt,
    load_unique_math_rows,
    validate_posttraining_context_budget,
    verify_answer,
)
from postraining.latent_rollout import (
    THOUGHT_SLOT,
    TOKEN_SLOT,
    continuation_reward,
    emitted_token_rows,
    rollout_continuations,
    trim_stream,
)
from postraining.latent_thought import (
    LatentThoughtModel,
    validate_renderer_checkpoint,
)
from postraining.model_io import load_model
from postraining.train_vapo import prompt_text


def gate_trace(kind_row: torch.Tensor, prompt_length: int) -> str:
    """Compact per-position trace after the prompt: E=emit, t=think."""
    symbols = {TOKEN_SLOT: "E", THOUGHT_SLOT: "t"}
    return "".join(
        symbols.get(int(slot), "") for slot in kind_row[prompt_length:]
    )


def decode_with_think_markers(
    tokenizer,
    kind_row: torch.Tensor,
    token_row: torch.Tensor,
    prompt_length: int,
    stop_ids: tuple[int, ...] = (),
) -> str:
    """Continuation text with an inline ``{n}🪙`` marker per THINK run.

    Emitted tokens are decoded in contiguous segments; SentencePiece strips a
    segment-leading space, so it is restored from the first piece's ``▁``
    whenever the segment is not the very start of the continuation.
    """
    parts: list[str] = []
    segment: list[int] = []
    run = 0
    at_start = True

    def flush_segment() -> None:
        nonlocal segment, at_start
        if segment:
            text = tokenizer.decode(segment)
            if not at_start and tokenizer.id_to_piece(segment[0]).startswith("▁"):
                text = " " + text
            parts.append(text)
            segment = []
            at_start = False

    for slot, token in zip(
        kind_row[prompt_length:].tolist(), token_row[prompt_length:].tolist()
    ):
        if slot == THOUGHT_SLOT:
            flush_segment()
            run += 1
        elif slot == TOKEN_SLOT:
            if run:
                parts.append(f"{run}🪙")
                at_start = False
                run = 0
            if token in stop_ids:
                # Control pieces (BOS/EOS) decode to nothing — show them.
                flush_segment()
                parts.append(tokenizer.id_to_piece(token))
                break
            segment.append(token)
    flush_segment()
    if run:
        parts.append(f"{run}🪙")
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
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--emit-only", action="store_true",
        help="pin the gate to EMIT so sampling uses only the belief renderer "
        "(no latent thinking)",
    )
    args = parser.parse_args()
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
    wrapper = LatentThoughtModel(backbone).to(device)
    if args.wrapper_checkpoint:
        payload = torch.load(
            args.wrapper_checkpoint, map_location="cpu", weights_only=False
        )
        validate_renderer_checkpoint(payload, args.wrapper_checkpoint)
        wrapper.load_state_dict(payload["model"], strict=True)
        print(f"policy: {args.wrapper_checkpoint} (step {payload.get('step')})")
    else:
        print("policy: untrained heads over the pretraining checkpoint")
    wrapper.eval()
    if args.emit_only:
        # Zero gate weight + saturated bias: EMIT with probability ~1.
        with torch.no_grad():
            wrapper.gate.head.weight.zero_()
            wrapper.gate.head.bias.fill_(30.0)
        print("gate pinned to EMIT: sampling the belief renderer directly")

    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor(model_file=FreshHyperparameters.tokenizer_path)

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
        with torch.no_grad():
            batch = trim_stream(
                rollout_continuations(
                    wrapper, prompt_ids, args.continuation_tokens,
                    args.max_stream_steps or 4 * args.continuation_tokens,
                    args.temperature, args.top_p,
                )
            )
        for index, emitted in enumerate(emitted_token_rows(batch)):
            generated = tokenizer.decode(emitted)
            reference = tokenizer.decode(reference_ids[index].tolist())
            reward = continuation_reward(generated, reference)
            trace = gate_trace(batch.kind[index], batch.prompt_length)
            if index % args.samples == 0:
                prompt_tail = tokenizer.decode(prompt_ids[index][-48:].tolist())
                print(f"=== prompt {index // args.samples}  (…{prompt_tail!r})")
                print(f"reference: {reference!r}")
            marked = decode_with_think_markers(
                tokenizer, batch.kind[index], batch.token_ids[index],
                batch.prompt_length,
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
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        stop_ids = tuple(
            t for t in (tokenizer.eos_id(), tokenizer.bos_id()) if t >= 0
        )
        records = []
        for row_index in picked:
            row = rows[row_index]
            text = prompt_text(row)
            truth = row["reward_model"]["ground_truth"]
            prompt_ids = torch.tensor(
                encode_prompt(tokenizer, text, args.prompt_tokens),
                dtype=torch.long,
                device=device,
            )
            with torch.no_grad():
                batch = trim_stream(
                    rollout_continuations(
                        wrapper,
                        prompt_ids[None].expand(args.samples, -1),
                        args.max_new_tokens,
                        stream_steps,
                        args.temperature,
                        args.top_p,
                        stop_ids=stop_ids or None,
                    )
                )
            samples = []
            for index, emitted in enumerate(emitted_token_rows(batch)):
                cut = next(
                    (i for i, t in enumerate(emitted) if t in stop_ids), None
                )
                if cut is not None:
                    emitted = emitted[: cut + 1]
                decoded = tokenizer.decode(emitted)
                is_correct, prediction = verify_answer(decoded, truth)
                trace = gate_trace(batch.kind[index], batch.prompt_length)
                samples.append(
                    {
                        "text": decode_with_think_markers(
                            tokenizer, batch.kind[index], batch.token_ids[index],
                            batch.prompt_length, stop_ids=stop_ids,
                        ),
                        "trace": trace,
                        "thinks": trace.count("t"),
                        "emits": trace.count("E"),
                        "correct": is_correct,
                        "prediction": prediction,
                    }
                )
            correct = sum(sample["correct"] for sample in samples)
            print(
                f"=== row {row_index}: {correct}/{len(samples)} correct "
                f"(truth: {truth})"
            )
            print(textwrap.shorten(text, 200))
            for index, sample in enumerate(samples):
                print(
                    f"--- sample {index}  "
                    f"{'CORRECT' if sample['correct'] else 'wrong'} "
                    f"(extracted: {sample['prediction']}, "
                    f"thinks: {sample['thinks']})"
                )
                print(sample["text"])
            print()
            records.append(
                {
                    "row_index": row_index,
                    "problem": text,
                    "ground_truth": truth,
                    "samples": samples,
                }
            )
        if args.json_out:
            payload = {
                "wrapper_checkpoint": args.wrapper_checkpoint,
                "math_data": args.math_data,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "samples_per_problem": args.samples,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
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
    with torch.no_grad():
        batch = trim_stream(
            rollout_continuations(
                wrapper,
                prompt_ids[None].expand(args.samples, -1),
                args.max_new_tokens,
                stream_steps,
                args.temperature,
                args.top_p,
                stop_ids=stop_ids or None,
            )
        )

    for index, emitted in enumerate(emitted_token_rows(batch)):
        cut = next((i for i, t in enumerate(emitted) if t in stop_ids), None)
        if cut is not None:
            emitted = emitted[: cut + 1]
        trace = gate_trace(batch.kind[index], batch.prompt_length)
        thinks = trace.count("t")
        decoded = tokenizer.decode(emitted)
        marked = decode_with_think_markers(
            tokenizer, batch.kind[index], batch.token_ids[index],
            batch.prompt_length, stop_ids=stop_ids,
        )
        print(f"--- sample {index}  (thinks: {thinks}, emits: {trace.count('E')})")
        print(f"trace: {trace}")
        if truth is not None:
            is_correct, prediction = verify_answer(decoded, truth)
            print(f"verdict: {'CORRECT' if is_correct else 'wrong'} (extracted: {prediction})")
        print(marked)
        print()


if __name__ == "__main__":
    main()
