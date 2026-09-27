"""Counterfactual grading of saved responses; no model execution or new tokens.

Run through mlq. Relaxation selects a single answer without looking at the
gold: first closed answer span, then a final Answer line, then last boxed
answer. It does not search all numbers or answer options for a lucky match.
"""

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re

from postraining.core import GPT2BPETokenizer, answer_style, extract_final_answer, structural_format_ok, verify_answer
from postraining.prepare_sft_traces import last_boxed


def unwrap_answer(value):
    value = value.strip()
    for _ in range(8):
        previous = value
        if value.startswith("$") and value.endswith("$"):
            value = value.strip("$").strip()
        elif value.startswith(r"\(") and value.endswith(r"\)"):
            value = value[2:-2].strip()
        elif value.startswith(r"\[") and value.endswith(r"\]"):
            value = value[2:-2].strip()
        else:
            match = re.fullmatch(r"\\(?:boxed|text|mathrm|mathbf)\{([^{}]*)\}", value)
            if match:
                value = match.group(1).strip()
            elif value.startswith(r"\boxed{") and value.endswith("}"):
                # Balanced whole-field boxes, including fractional content.
                depth = 0
                closes_at_end = True
                for index, char in enumerate(value[6:], 6):
                    depth += (char == "{") - (char == "}")
                    if depth == 0 and index != len(value) - 1:
                        closes_at_end = False
                if closes_at_end and depth == 0:
                    value = value[7:-1].strip()
        if value == previous:
            break
    if re.fullmatch(r"[A-J][.)]", value):
        value = value[0]
    return value


def selected_answer(row, tokenizer):
    tokens = row["token_ids"]
    try:
        start = tokens.index(50259)
        end = tokens.index(50260, start + 1)
        if end > start + 1:
            return tokenizer.decode(tokens[start + 1:end]), "closed_answer_span"
    except ValueError:
        pass
    field = extract_final_answer(row["text"], window=None)
    if field is not None:
        return field, "explicit_answer_line"
    field = last_boxed(row["text"])
    return (field, "last_boxed") if field is not None else (None, "no_answer")


@lru_cache(maxsize=65536)
def grade(field, truth, style):
    return bool(verify_answer("Answer: " + field, truth, style, window=None)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    sources = {}
    examples = []
    for source in ("deepmind_easy", "ultradata_math", "dapo", "science_mc", "ultradata_knowledge", "ultradata_code_l3"):
        counts = Counter()
        lengths = Counter()
        methods = Counter()
        path = Path(f"postraining/runs/kda8_readiness_{source}_20260926/gate_transcripts.jsonl")
        for line in path.open():
            row = json.loads(line)
            counts["samples"] += 1
            strict = row["reward"] == 1
            counts["strict_correct"] += strict
            bucket = "truncated" if not row["terminated"] else (
                "terminated_bad_format" if not row["structural_format_ok"] else "accepted_format")
            counts[bucket] += 1
            tokens = row["token_ids"]
            if bucket == "truncated":
                lengths[len(tokens)] += 1
            if bucket == "terminated_bad_format":
                counts["bad_format_not_starting_think"] += not tokens or tokens[0] != 50257
                counts["bad_format_fixed_by_think_min_1"] += structural_format_ok(tokens, (50257, 50258), (50259, 50260), 1)
            if source == "ultradata_code_l3":
                continue
            field, method = selected_answer(row, tokenizer)
            methods[method] += 1
            if field is None:
                counts["no_selected_answer"] += 1
                continue
            truth = row["reward_model"]["ground_truth"]
            if len(field) > 512 or len(truth) > 512:
                counts["long_answer_ungraded"] += 1
                continue
            style = answer_style(row)
            raw_correct = grade(field, truth, style)
            normalized = unwrap_answer(field)
            relaxed_correct = grade(normalized, truth, style)
            counts["selected_answer_correct_raw"] += raw_correct
            counts["selected_answer_correct_unwrapped"] += relaxed_correct
            if not strict and raw_correct:
                counts["raw_recovered_" + bucket] += 1
            if not strict and relaxed_correct:
                counts["relaxed_recovered_" + bucket] += 1
                if len(examples) < 1000:
                    examples.append({"source": source, "id": row["extra_info"]["index"],
                                     "repeat": row["repeat"], "sample": row["sample"], "bucket": bucket,
                                     "method": method, "field": field, "unwrapped": normalized,
                                     "truth": truth, "raw_correct": raw_correct})
            if strict and not raw_correct:
                counts["strict_selected_disagreement"] += 1
        sources[source] = {"counts": dict(counts), "truncated_token_lengths": dict(lengths), "selection_methods": dict(methods)}
        print(json.dumps({"source": source, **dict(counts)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump({"schema": "saved_response_format_attribution/v1", "sources": sources, "recovered_examples": examples,
                   "limitations": "No new generation: cannot estimate success after extending the token budget. First closed spans/last boxes can be provisional answers. Recovery is a diagnostic, not an endorsed reward-parser change. Fields over512characters left ungraded."}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
