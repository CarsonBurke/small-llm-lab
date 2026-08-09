"""Score a checkpoint on the held-out arithmetic probe, per family and digit.

The frozen math gates report one accuracy per source, which is enough to
notice that a model is bad at mathematics and not enough to say why. This
runner answers the narrower question the corpus work is actually trying to
move: can the model add two five-digit numbers, divide with a remainder,
compare decimals, take a percentage. Each is reported separately, and each is
reported per digit count, because "can add" and "can add five-digit numbers"
routinely disagree.

Generation goes through the same production rollout stack the SFT sampling
gate uses, at greedy-equivalent settings -- arithmetic has one right answer,
so sampling temperature would measure the decoder rather than the model.
Grading is `arithmetic_probe.graded`, exact match after numeric
canonicalization, run over the answer the evaluator already parsed out of the
fenced span.

Scope: this runs post-training checkpoints, which are the ones that carry the
registered think/answer tokens and the canonical prompt schema. A
pretraining-only checkpoint has neither, and comparing the corpus ablation
arms to each other is a job for the per-domain bits-per-byte the trainer
already emits -- that metric is tokenizer-independent, which the probe's
answer-span parse is not.

This executes a model on the GPU and must be submitted through mlq.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from postraining.arithmetic_probe import PROBE_SCHEMA, read_panel, score
from postraining.core import load_posttraining_tokenizer
from postraining.latent_eval import evaluate_latent_math
from postraining.latent_thought import LatentThoughtModel
from postraining.math_prompt import (
    ANSWER_FENCE_SUFFIX,
    require_answer_fence_prompt_schema,
)
from postraining.model_io import load_model
from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters

PROBE_RUN_SCHEMA = "arithmetic_probe_run/v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def carries_trained_combiner(payload: dict) -> bool:
    """Whether a checkpoint holds a hidden-carry combiner this probe would hide.

    Decided on the parameters, because they are the property being refused.
    `args["reasoning_mode"]` is the weakest available signal: the latent VAPO
    trainer reads its own mode with a `latent` default, so a payload can carry
    a trained combiner and record no mode at all. The top-level
    `reasoning_mode` is kept as a second line for a payload whose parameters
    are stored some other way.
    """
    parameters = payload.get("model") or {}
    return any(
        key.startswith("combiner.") for key in parameters
    ) or payload.get("reasoning_mode") == "latent"


def contract_predictions(
    captured: list[dict[str, object]], parsed: list[str]
) -> list[str]:
    """Blank out the answers that did not come from a valid completion.

    An unterminated or unfenced attempt still yields a `parsed_answer`: the
    evaluator falls back to scraping the last number out of the tail. That is
    the right leniency for a training-time signal and the wrong one for a
    capability measurement -- a model that rambles past its budget and happens
    to end on the right digits has not answered the question.
    """
    return [
        prediction
        if attempt["terminated"] and attempt["structural_format_ok"]
        else ""
        for attempt, prediction in zip(captured, parsed, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--panel",
        default="data/math_drills/v1/probe.jsonl",
        help="held-out panel written by scripts/build_math_drills.py",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--batch-trajectories", type=int, default=128)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 selects the argmax path; arithmetic has one right answer, so "
        "the default measures the model rather than the decoder",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if min(args.max_new_tokens, args.prompt_tokens, args.batch_trajectories) < 1:
        parser.error("token budgets and batch sizes must be positive")
    if not 0.0 <= args.temperature:
        parser.error("--temperature must be nonnegative")

    output = Path("postraining/runs") / args.name
    if output.exists():
        parser.error(f"refusing to overwrite existing probe output {output}")
    # `output` is created later, immediately after generation, so a run that
    # dies before the model produces anything leaves no directory to explain.

    panel_path = Path(args.panel)
    panel, panel_provenance = read_panel(panel_path)
    if not panel_provenance.get("disjoint_from_training"):
        parser.error(
            f"{panel_path} does not record disjointness from the drill "
            "training stream; it is not a held-out panel"
        )

    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # `LatentThoughtModel(backbone)` below builds a zero-init combiner and the
    # rollout runs with `pin_emit=True`, which is the SFT gate's arrangement
    # and the right one for a checkpoint that has no trained carry. A latent
    # VAPO checkpoint does have one, and measuring it with the carry pinned
    # off would report the wrong policy under the right name -- so refuse it
    # rather than silently produce a number.
    #
    # This has to run before the prompt-schema check below: a latent VAPO
    # checkpoint has no top-level `sft` key, so that check would reject it
    # first and name the wrong reason.
    if carries_trained_combiner(payload):
        parser.error(
            f"{checkpoint_path} carries a trained hidden-carry combiner; this "
            "probe pins the carry off and would measure a different policy "
            "than the one that was trained"
        )
    # The probe is only a measurement of the policy if the prompt it sees is
    # the one the policy was trained under. Reported through `parser.error`
    # like every other rejection here, rather than as a bare traceback.
    try:
        require_answer_fence_prompt_schema(
            payload.get("sft") or {},
            answer_fence=True,
            source="arithmetic-probe checkpoint",
        )
    except ValueError as error:
        parser.error(str(error))
    device = torch.device("cuda")
    backbone = load_model(checkpoint_path, device, payload=payload)
    tokenizer = load_posttraining_tokenizer(
        payload["architecture"],
        FreshHyperparameters.tokenizer_path,
        think_tokens=True,
        answer_tokens=True,
        tokenizer_provenance=payload["model_config"].get(
            "tokenizer_provenance"
        ),
    )
    wrapper = LatentThoughtModel(backbone).to(device)
    wrapper.eval()

    rows = [
        {
            "prompt": [{"content": item.problem + ANSWER_FENCE_SUFFIX}],
            # Every probe answer is a literal string, so the rule verifier is
            # the right style; the probe's own grader decides correctness
            # below, and this only shapes the parse.
            "reward_model": {"ground_truth": item.answer, "style": "rule"},
        }
        for item in panel
    ]
    think_fence_ids = (
        (tokenizer.think_open_id, tokenizer.think_close_id)
        if getattr(tokenizer, "think_open_id", None) is not None
        else None
    )
    answer_fence_ids = (
        (tokenizer.answer_open_id, tokenizer.answer_close_id)
        if getattr(tokenizer, "answer_open_id", None) is not None
        else None
    )
    captured: list[dict[str, object]] = []
    metrics = evaluate_latent_math(
        wrapper,
        tokenizer,
        rows,
        samples=1,
        max_new_tokens=args.max_new_tokens,
        max_stream_steps=args.max_new_tokens,
        chunk=1,
        seed=args.seed,
        device=device,
        prompt_tokens=args.prompt_tokens,
        batch_trajectories=args.batch_trajectories,
        captured_attempts=captured,
        # Capture every item: a probe that scored a sample of itself would be
        # a different measurement with the same name.
        capture_problem_count=len(rows),
        capture_samples_per_problem=1,
        temperature=args.temperature,
        top_p=args.top_p,
        pin_emit=True,
        think_fence_ids=think_fence_ids,
        answer_fence_ids=answer_fence_ids,
    )
    # The transcripts land before the scoring guards below, not after. Both
    # guards fire on a capture that came back wrong, and the capture is
    # exactly the evidence needed to work out why -- throwing it away to
    # preserve "an output directory means a scored run" would trade the
    # diagnosis for the tidiness. `result.json` is still written only on a
    # complete run, so a partial directory is unmistakable.
    output.mkdir(parents=True)
    (output / "transcripts.json").write_text(json.dumps(captured, indent=2) + "\n")

    if len(captured) != len(panel):
        raise RuntimeError(
            f"captured {len(captured)} attempts for {len(panel)} panel items; "
            "the probe cannot be scored from a partial capture"
        )
    # `evaluate_latent_math` buckets rows by length for compute and sorts the
    # capture back into dataset order, so position is attribution. Scoring
    # pairs attempts with panel items positionally, and a silent misalignment
    # would report a per-family accuracy that is a permutation of the truth --
    # wrong in a way no aggregate would reveal.
    misaligned = [
        index
        for index, attempt in enumerate(captured)
        if int(attempt["problem_index"]) != index
    ]
    if misaligned:
        raise RuntimeError(
            f"capture is not in panel order at {len(misaligned)} positions, "
            f"first {misaligned[0]}; the probe cannot attribute its scores"
        )

    parsed = [str(attempt.get("parsed_answer") or "") for attempt in captured]
    # The headline score requires the completion contract; `lenient_score`
    # keeps the scraped number, so the gap between the two is on the page
    # rather than assumed either way.
    result = score(panel, contract_predictions(captured, parsed))
    lenient = score(panel, parsed)
    unterminated = sum(1 for attempt in captured if not attempt["terminated"])
    malformed = sum(
        1 for attempt in captured if not attempt["structural_format_ok"]
    )

    report = {
        "schema": PROBE_RUN_SCHEMA,
        "probe_schema": PROBE_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "panel": str(panel_path),
        "panel_sha256": file_sha256(panel_path),
        "panel_provenance": panel_provenance,
        "score": result,
        "lenient_score": lenient,
        # Reported beside accuracy rather than folded into it: a model that
        # never terminates and a model that terminates on the wrong number
        # are both at zero accuracy and need different fixes.
        "unterminated": unterminated,
        "malformed_format": malformed,
        "generation_metrics": metrics,
        "args": vars(args),
    }
    temporary = output / f"result.json.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "result.json")

    print(f"overall {result['accuracy']:.3f} over {result['items']:,} items")
    for family, values in result["by_family"].items():
        digits = result["by_family_digits"][family]
        spread = " ".join(
            f"{size}d={row['accuracy']:.2f}" for size, row in digits.items()
        )
        print(f"  {family:16s} {values['accuracy']:.3f}  {spread}")
    print(
        f"ungradable {result['ungradable_predictions']:,}  "
        f"unterminated {unterminated:,}  malformed {malformed:,}"
    )
    print(
        f"lenient {lenient['accuracy']:.3f} (scraped tail answers credited); "
        f"the gap is what the completion contract costs"
    )


if __name__ == "__main__":
    main()
