"""Can a group's own siblings assign token credit? A prefix-trie audit.

A rollout group is one prompt with several samples from the same policy.
Samples that share an exact token prefix are in the same state there, so
the other siblings through a state are free Monte-Carlo continuations from
it. The leave-one-out sibling baseline

    b(s_t) = mean reward of the OTHER group members whose prefix equals s_t

is independent of the action taken at s_t (siblings are sampled
independently), so ``R - b(s_t)`` is an unbiased policy-gradient
coefficient. Its useful property: at a state where every sharer took the
same next token, the sharers' coefficients sum to exactly zero, so shared
tokens carry no net update; at a state where they split, the update is the
contrast between the branches' mean rewards. Where no sibling shares the
state, a learned critic must stand in.

This measures, on a frozen policy's bench capture (``--bench-capture-*``):

- coverage: the share of trajectory tokens whose state has a sibling;
- for failed trajectories in mixed groups, located by the drill step
  checker (``analyze_drill_step_errors``), whether the trie blames the
  culprit: is the first wrong token a branch token (a sibling sharing its
  state took another token), and where does the trajectory leave all its
  siblings relative to that token;
- the share of a group-mean baseline's net (state, action) coefficient
  mass that sits on shared non-branching tokens, which this baseline
  removes exactly.

    .venv/bin/python scripts/analyze_sibling_prefix_credit.py \\
        --bench-data postraining/data/drill_rl_study_heldout_20261014_v1.parquet \\
        --answers <run>/bench_answers/step_000000.json --output <out>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_drill_step_errors import canonical_solutions, step_lines

# GPT-2's <|endoftext|> and the registered fence tokens above it.
SPECIAL_IDS = frozenset({50256}) | frozenset(range(50257, 50304))


def _bpe_char_to_byte() -> dict[str, int]:
    """Inverse of GPT-2's byte-to-unicode table (its tokens spell bytes this way)."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1),
                 *range(ord("®"), ord("ÿ") + 1)]
    chars, extra = {}, 0
    for byte in range(256):
        if byte in printable:
            chars[chr(byte)] = byte
        else:
            chars[chr(256 + extra)] = byte
            extra += 1
    return chars


BPE_CHAR_TO_BYTE = _bpe_char_to_byte()


def token_char_offsets(token_ids: list[int], text: str, tokenizer) -> list[int | None]:
    """Character offset in ``text`` where each emitted token starts.

    Offsets come from the tokens' exact bytes, so a character split across
    tokens (byte-level BPE) belongs to the token that completes it. Special
    tokens are not part of the text and map to None. The bytes must decode
    to ``text`` exactly, else the mapping is refused.
    """
    byte_of = BPE_CHAR_TO_BYTE
    offsets: list[int | None] = []
    emitted = bytearray()
    for token, piece in zip(token_ids, tokenizer.convert_ids_to_tokens(token_ids)):
        if token in SPECIAL_IDS:
            offsets.append(None)
            continue
        offsets.append(len(emitted.decode("utf-8", errors="ignore")))
        emitted.extend(byte_of[char] for char in piece)
    if emitted.decode("utf-8", errors="replace") != text:
        raise ValueError("emitted token ids do not decode to the emitted text")
    return offsets


def first_error_char(text: str, methods: list[list[str]]) -> int | None:
    """Offset of the first wrong character, against the method followed furthest.

    None when the working matches a method in full (a correct working, or a
    wrong final assembly, which this does not localise).
    """
    lines = text.split("\n")
    starts, position = [], 0
    for line in lines:
        starts.append(position)
        position += len(line) + 1
    numeric = [
        (start + len(line) - len(line.lstrip()), line.strip())
        for start, line in zip(starts, lines)
        if step_lines(line)
    ]
    best: tuple[int, int | None] = (-1, None)
    for method in methods:
        for index, want in enumerate(method):
            if index >= len(numeric):
                break  # truncated working: not a localised slip
            start, got = numeric[index]
            if got != want:
                column = next(
                    (c for c, (g, w) in enumerate(zip(got, want)) if g != w),
                    min(len(got), len(want)),
                )
                if index > best[0]:
                    best = (index, start + column)
                break
        else:
            if len(numeric) >= len(method):
                return None
    return best[1]


def analyze(bench_data: Path, answers: Path, tokenizer) -> dict:
    solutions = canonical_solutions(bench_data)
    groups: dict[str, list[dict]] = defaultdict(list)
    for attempt in json.loads(answers.read_text())["attempts"]:
        groups[attempt["dataset_index"]].append(attempt)
    per_module: dict[str, Counter] = defaultdict(Counter)
    departure_minus_error: dict[str, list[int]] = defaultdict(list)
    for index, group in groups.items():
        module = "_".join(index.split("/")[1:3])
        counts = per_module[module]
        sequences = [tuple(attempt["emitted_token_ids"]) for attempt in group]
        rewards = [float(bool(attempt["correct"])) for attempt in group]
        # Trie as prefix -> {next token: [members]}; states are prefixes.
        children: dict[tuple, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for member, sequence in enumerate(sequences):
            for t, token in enumerate(sequence):
                children[sequence[:t]][token].append(member)
        group_mean = sum(rewards) / len(rewards)
        for state, branches in children.items():
            members = [m for ms in branches.values() for m in ms]
            if len(members) < 2:
                continue
            for ms in branches.values():
                # Net coefficient on (state, action) under a group-mean baseline.
                mass = abs(sum(rewards[m] - group_mean for m in ms))
                counts["groupmean_mass_shared"] += mass
                if len(branches) == 1:
                    counts["groupmean_mass_shared_nonbranching"] += mass
        mixed = 0 < sum(rewards) < len(rewards)
        counts["groups"] += 1
        counts["mixed_groups"] += mixed
        for member, (attempt, sequence) in enumerate(zip(group, sequences)):
            sharers = [len([m for ms in children[sequence[:t]].values() for m in ms])
                       for t in range(len(sequence))]
            counts["tokens"] += len(sequence)
            counts["tokens_with_sibling"] += sum(n >= 2 for n in sharers)
            if not mixed or attempt["correct"]:
                continue
            counts["failed_in_mixed"] += 1
            error_char = first_error_char(attempt["emitted_text"], solutions[attempt["prompt"]])
            if error_char is None:
                counts["failed_unlocalised"] += 1
                continue
            offsets = token_char_offsets(list(sequence), attempt["emitted_text"], tokenizer)
            # The token whose decoded span contains the first wrong character.
            error_token = max(
                t for t, start in enumerate(offsets) if start is not None and start <= error_char
            )
            departure = next((t for t, n in enumerate(sharers) if n < 2), len(sequence))
            departure_minus_error[module].append(departure - error_token)
            counts["failed_localised"] += 1
            state_branches = children[sequence[:error_token]]
            counts["error_state_shared"] += sharers[error_token] >= 2
            counts["error_is_branch_token"] += len(state_branches) > 1
            # LOO baseline at the culprit: the siblings that shared its state.
            others = [m for ms in state_branches.values() for m in ms if m != member]
            if others:
                counts["error_coefficient_sum"] += 0.0 - sum(rewards[m] for m in others) / len(others)
    report = {}
    for module, counts in sorted(per_module.items()):
        localised = counts["failed_localised"]
        gaps = sorted(departure_minus_error[module])
        for key in ("failed_localised", "failed_in_mixed"):
            counts.setdefault(key, 0)
        report[module] = {
            **counts,
            "sibling_coverage": counts["tokens_with_sibling"] / counts["tokens"],
            "error_state_shared_rate": counts["error_state_shared"] / localised if localised else None,
            "error_is_branch_rate": counts["error_is_branch_token"] / localised if localised else None,
            "mean_error_coefficient": (
                counts["error_coefficient_sum"] / counts["error_state_shared"]
                if counts["error_state_shared"] else None
            ),
            "departure_minus_error_quartiles": (
                [gaps[int(q * (len(gaps) - 1))] for q in (0.25, 0.5, 0.75)] if gaps else None
            ),
            "groupmean_nonbranching_share": (
                counts["groupmean_mass_shared_nonbranching"] / counts["groupmean_mass_shared"]
                if counts["groupmean_mass_shared"] else None
            ),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    from transformers import GPT2TokenizerFast

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bench-data", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.bench_data, args.answers, GPT2TokenizerFast.from_pretrained("gpt2"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    fmt = lambda x: "    -" if x is None else f"{x:5.2f}"
    print(f"{'module':18s} mixed cover  shared branch  coef  nonbr  fails  dep-err q")
    for module, row in report.items():
        print(f"{module:18s} {row['mixed_groups']:3d}/{row['groups']:<3d} "
              f"{fmt(row['sibling_coverage'])} {fmt(row['error_state_shared_rate'])} "
              f"{fmt(row['error_is_branch_rate'])} {fmt(row['mean_error_coefficient'])} "
              f"{fmt(row['groupmean_nonbranching_share'])} "
              f"{row['failed_localised']:3d}/{row['failed_in_mixed']:<3d} "
              f"{row['departure_minus_error_quartiles']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
