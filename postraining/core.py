"""Pure VAPO math, verification, data, and rollout helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor


POSTTRAIN_REWARD_SCHEMA = (
    "terminated_final_answer_exact1_numeric_log1p_max01/v3"
)


# Latent-policy inference budget. The backbone was pretrained on 1024-token
# sequences, while PoPE can execute beyond that window. Posttraining keeps a
# full pretrained window available to the prompt and gives generation four
# stream slots (THINK or EMIT) per permitted emitted answer token.
POSTTRAIN_CONTEXT_TOKENS = 5 * 1024
POSTTRAIN_PROMPT_TOKENS = 1024
POSTTRAIN_RESPONSE_TOKENS = 1024
POSTTRAIN_STREAM_TOKENS = POSTTRAIN_CONTEXT_TOKENS - POSTTRAIN_PROMPT_TOKENS


def validate_posttraining_context_budget(
    prompt_tokens: int,
    stream_tokens: int,
    context_tokens: int = POSTTRAIN_CONTEXT_TOKENS,
) -> None:
    """Reject configured prompt/stream caps that exceed the RL context."""
    if prompt_tokens < 1 or stream_tokens < 1:
        raise ValueError("prompt and stream token budgets must be positive")
    if prompt_tokens + stream_tokens > context_tokens:
        raise ValueError(
            f"prompt ({prompt_tokens}) + stream ({stream_tokens}) exceeds "
            f"the posttraining context ({context_tokens})"
        )


SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""),
    (" ", ""), ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "points", "feet", "minutes", "digits", "cents", "degrees", "cm",
    "gm", "pounds", "meters", "meals", "edges", "students", "multiples", "sue",
    "childrentickets", "\\text{\ns}", "\\text{\n}",
    "\\ldots", "\\dots", "\\text{s}", "\\text{.}", "\\text{}^2",
    "\\text{}^3", "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}",
    r"\;", r",\!", "{,}", '"',
]


def normalize_final_answer(answer: str) -> str:
    """DAPO/Minerva-compatible normalization used for train and AIME rewards."""
    answer = answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        answer = answer.replace(before, after)
    for expression in REMOVED_EXPRESSIONS:
        answer = answer.replace(expression, "")
    answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"$\3$", answer)
    answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", answer)
    answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", answer)
    answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", answer)
    answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", answer)
    answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", answer)
    answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", answer)
    answer = answer.replace("$", "")
    if answer.replace(",", "").isdigit():
        answer = answer.replace(",", "")
    return answer.strip()


# ``reward_model.style`` -> grading style.  The DAPO and AIME parquets tag
# "rule-lighteval/MATH_v2" (Minerva normalization IS lighteval-MATH's grader);
# the DeepMind mathematics_dataset conversions tag plain "rule" and are graded
# the dataset's official way — exact match on the canonical answer string.
_ANSWER_STYLE_BY_RULE = {
    "rule": "exact",
    "rule-lighteval/MATH_v2": "minerva",
}


def answer_style(row: dict) -> str:
    """Grading style for one data row, resolved from its reward_model.style."""
    rule = (row.get("reward_model") or {}).get("style")
    return _ANSWER_STYLE_BY_RULE.get(rule, "minerva")


def extract_final_answer(solution: str, window: int | None = 300) -> str | None:
    """The last ``Answer: ...`` line, per the shared DAPO prompt contract."""
    text = solution if window is None else solution[-window:]
    matches = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", text)
    return matches[-1] if matches else None


def verify_answer(
    solution: str,
    ground_truth: str,
    style: str = "minerva",
    window: int | None = 300,
) -> tuple[bool, str]:
    """``window`` bounds the tail searched for the ``Answer:`` field (the
    DAPO contract's 300-char default). Callers that constructed
    ``solution`` themselves — e.g. the fenced-answer reframe, which IS the
    answer field — pass None so a long value cannot push its own prefix
    out of the search window and silently grade [INVALID]."""
    if style == "minerva":
        extracted = extract_final_answer(solution, window=window)
        prediction = normalize_final_answer(
            "[INVALID]" if extracted is None else extracted
        )
        # DAPO's ground truths are numeric.  Require its raw final Answer:
        # field to contain one number before applying Minerva's canonical
        # formatting, otherwise comma-separated candidate lists can collapse
        # into an apparently correct integer (for example ``3, 4`` -> ``34``).
        # Preserve Minerva behavior for genuinely nonnumeric datasets.
        if (
            parse_numeric_answer(ground_truth) is not None
            and (
                extracted is None
                or parse_numeric_answer(extracted) is None
            )
        ):
            return False, prediction
        return prediction == normalize_final_answer(ground_truth), prediction
    if style == "exact":
        # Official mathematics_dataset grading: the canonical answer string,
        # matched exactly (modulo surrounding whitespace).  Ground truths are
        # already canonical, so Minerva's rewrites could only widen the match
        # — every widening collapses a multi-candidate or hedged answer line
        # onto a single graded token — and the whole emission is searched
        # because a fixed tail window turns a verbose-but-correct final line
        # into [INVALID].
        extracted = extract_final_answer(solution, window=None)
        prediction = "[INVALID]" if extracted is None else extracted.strip()
        return prediction == ground_truth.strip(), prediction
    if style == "aime":
        # AIME answers are integers in [0, 999]; standard graders reject any
        # response whose final answer does not parse as one.
        extracted = extract_final_answer(solution, window=window)
        prediction = "[INVALID]" if extracted is None else extracted.strip()
        cleaned = prediction.rstrip(".").strip("$ ")
        try:
            value = int(cleaned)
            truth_value = int(ground_truth.strip())
        except ValueError:
            # A non-integer ground truth means the row is not actually AIME;
            # score it wrong rather than crash the eval mid-run.
            return False, prediction
        return 0 <= value <= 999 and value == truth_value, prediction
    raise ValueError(f"unknown answer style {style!r}")


def parse_numeric_answer(answer: str) -> Fraction | None:
    """Parse one raw final-answer field as an exact finite number.

    This deliberately accepts decimals, scientific notation, and simple
    fractions while rejecting candidate lists or surrounding prose. The
    caller must first extract the final ``Answer:`` line; parsing the whole
    response would let repeated candidates manufacture reward.
    """
    text = answer.strip()
    if not text or len(text) > 128:
        return None
    if text.startswith("$") and text.endswith("$") and text.count("$") == 2:
        text = text[1:-1].strip()
    boxed = re.fullmatch(r"\\boxed\{(.+)\}", text)
    if boxed:
        text = boxed.group(1).strip()
    latex_fraction = re.fullmatch(
        r"\\?frac\{([+-]?(?:\d+(?:\.\d*)?|\.\d+))\}"
        r"\{([+-]?(?:\d+(?:\.\d*)?|\.\d+))\}",
        text,
    )
    number = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    try:
        if latex_fraction:
            denominator = Fraction(latex_fraction.group(2))
            return Fraction(latex_fraction.group(1)) / denominator
        plain_fraction = re.fullmatch(rf"({number})/({number})", text)
        if plain_fraction:
            numerator = Fraction(plain_fraction.group(1).replace(",", ""))
            denominator = Fraction(plain_fraction.group(2).replace(",", ""))
            return numerator / denominator
        if not re.fullmatch(number, text):
            return None
        return Fraction(text.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def nearby_numeric_reward(
    prediction: str,
    ground_truth: str,
    maximum: float = 0.1,
) -> float:
    """Bounded partial credit for one wrong but numerically parseable answer.

    Exact correctness remains the verifier's responsibility. This function is
    called only on its wrong branch, where distance may still be zero (for
    example ``5.0`` versus a strict canonical ``5``). ``log1p`` keeps floats
    near the target bounded by ``maximum`` and decays asymptotically to zero.
    """
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("maximum nearby reward must be finite and nonnegative")
    predicted_value = parse_numeric_answer(prediction)
    truth_value = parse_numeric_answer(ground_truth)
    if predicted_value is None or truth_value is None or maximum == 0.0:
        return 0.0
    distance = abs(predicted_value - truth_value)
    try:
        distance_float = float(distance)
    except OverflowError:
        return 0.0
    if not math.isfinite(distance_float):
        return 0.0
    return maximum / (1.0 + math.log1p(distance_float))


def module_answer_baselines(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Per-module modal-answer share of a row set's ground truths.

    This is the accuracy of a policy that answers every prompt in a module
    with the module's single most common ground truth — the strongest
    zero-reasoning constant strategy the verifier cannot distinguish from
    solving.  Module accuracy is evidence of solving only where it clears
    this baseline, so it is computed from the data at load time and logged
    next to the accuracies it calibrates.
    """
    by_module: dict[str, list[str]] = {}
    for row in rows:
        module = (row.get("extra_info") or {}).get("module")
        if module:
            by_module.setdefault(str(module), []).append(
                row["reward_model"]["ground_truth"]
            )
    baselines: dict[str, dict[str, float]] = {}
    for module, truths in sorted(by_module.items()):
        counts: dict[str, int] = {}
        for truth in truths:
            counts[truth] = counts.get(truth, 0) + 1
        baselines[module] = {
            "rows": float(len(truths)),
            "modal_share": max(counts.values()) / len(truths),
        }
    return baselines


def modal_answer_baseline(rows: list[dict]) -> dict[str, str | float]:
    """Strongest constant-answer baseline under each row's grading style."""
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        style = answer_style(row)
        truth = str(row["reward_model"]["ground_truth"])
        canonical = truth.strip() if style == "exact" else normalize_final_answer(truth)
        key = (style, canonical)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return {"answer": "", "style": "", "accuracy": 0.0}
    (style, answer), count = max(
        counts.items(), key=lambda item: (item[1], item[0])
    )
    return {
        "answer": answer,
        "style": style,
        "accuracy": count / len(rows),
    }


MATH_CORPUS_SCHEMA = "exact_prompt_reviewed_contract/v1"
_MATH_TARGET_REVIEWS_PATH = Path(__file__).with_name("data") / "math_target_reviews.json"


def _corpus_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _math_prompt_json(prompt: object) -> str:
    if not isinstance(prompt, list) or not prompt or any(
        not isinstance(message, dict)
        or ("role" in message and not isinstance(message["role"], str))
        or not isinstance(message.get("content"), str)
        for message in prompt
    ):
        raise ValueError("math corpus requires nonempty ordered text chat messages")
    return _corpus_json(prompt)


def _math_corpus_policy() -> tuple[dict[str, dict], str]:
    """Validate reviews before any row can be relabeled, including file copies."""
    registry = json.loads(_MATH_TARGET_REVIEWS_PATH.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("schema") != "math_target_reviews/v1":
        raise ValueError("incompatible math target review registry")
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ValueError("math target review registry requires an entries list")
    reviews: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid math target review entry")
        prompt = _math_prompt_json(entry.get("prompt"))
        fingerprint = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if entry.get("prompt_sha256") != fingerprint or fingerprint in reviews:
            raise ValueError("math target review fingerprint mismatch or duplicate")
        expected = entry.get("expected_targets")
        if (
            entry.get("action") not in {"correct", "quarantine"}
            or not isinstance(expected, list)
            or not expected
            or any(not isinstance(target, str) for target in expected)
            or not entry.get("reason")
            or not entry.get("evidence")
            or (
                entry["action"] == "correct"
                and not isinstance(entry.get("corrected_target"), str)
            )
        ):
            raise ValueError(f"invalid math target review contract: {fingerprint}")
        reviews[fingerprint] = entry
    policy = {"schema": MATH_CORPUS_SCHEMA, "reviews": registry}
    return reviews, hashlib.sha256(_corpus_json(policy).encode("utf-8")).hexdigest()


def math_corpus_policy_sha256() -> str:
    """Bind corpus semantics to the loader version and validated review registry."""
    return _math_corpus_policy()[1]


def _math_reward_contract(row: dict) -> dict:
    # Keep styles and all test metadata strict: normalization of a label or a
    # code test here could merge tasks with different reward semantics.
    reward = row.get("reward_model")
    if not isinstance(reward, dict) or not isinstance(reward.get("ground_truth"), str):
        raise ValueError("math corpus row requires a string reward_model.ground_truth")
    return {
        "reward_model": reward,
        "verification_info": row.get("verification_info"),
    }


def math_corpus_identity(rows: list[dict]) -> str:
    """Hash ordered effective prompts/contracts, not incidental source indices."""
    digest = hashlib.sha256()
    digest.update(_corpus_json({
        "policy_sha256": math_corpus_policy_sha256(),
        "rows": len(rows),
    }).encode("utf-8"))
    for row in rows:
        digest.update(b"\n")
        digest.update(_math_prompt_json(row["prompt"]).encode("utf-8"))
        digest.update(b"\n")
        digest.update(_corpus_json(_math_reward_contract(row)).encode("utf-8"))
    return digest.hexdigest()


def load_unique_math_rows(path: str | Path, *, audit: dict | None = None) -> list[dict]:
    """Load one row per exact full chat prompt, quarantining ambiguous rewards.

    Source IDs only accelerate physical-repeat recognition; equality of the
    entire prompt and raw grading contract is still checked on every cache hit.
    Thus a reused ID cannot hide different content, labels, styles, or tests.
    """
    reviews, policy_sha256 = _math_corpus_policy()
    available = set(pq.read_schema(path).names)
    required = {"prompt", "reward_model", "extra_info"}
    if not required <= available:
        raise ValueError(f"{path} lacks verifier columns {sorted(required - available)}")
    columns = [
        name
        for name in (
            "prompt",
            "reward_model",
            "extra_info",
            "verification_info",
            "data_source",
            "ability",
        )
        if name in available
    ]
    groups: dict[str, dict] = {}
    source_cache: dict[tuple, list[dict]] = {}
    physical_rows = 0
    input_rows = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        for row in batch.to_pylist():
            physical_rows += 1
            info = row.get("extra_info") or {}
            source_index = info.get("index")
            prompt = _math_prompt_json(row["prompt"]) if source_index is None else None
            # Most DAPO rows repeat 100 times. Compare to a retained source
            # variant instead of reserializing/hashing these long prompts.
            source_key = (
                row.get("data_source"), type(source_index).__name__,
                prompt if source_index is None else str(source_index),
            )
            variants = source_cache.get(source_key)
            if variants is None:
                variants = []
                source_cache[source_key] = variants
            repeated = None
            for variant in variants:
                raw = variant["raw"]
                if (
                    raw["prompt"] == row["prompt"]
                    and raw["reward_model"] == row["reward_model"]
                    and raw.get("verification_info") == row.get("verification_info")
                ):
                    repeated = variant
                    break
            if repeated is not None:
                repeated["physical_rows"] += 1
                groups[repeated["prompt_key"]]["physical_rows"] += 1
                continue
            input_rows += 1
            if prompt is None:
                prompt = _math_prompt_json(row["prompt"])
            raw_contract = _math_reward_contract(row)
            group = groups.get(prompt)
            if group is None:
                fingerprint = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                group = {
                    "prompt_sha256": fingerprint,
                    "row": row,
                    "contracts": {},
                    "variants": [],
                    "physical_rows": 0,
                    "quarantine_reasons": set(),
                }
                groups[prompt] = group
            fingerprint = group["prompt_sha256"]
            review = reviews.get(fingerprint)
            receipt = None
            effective_row = row
            if review is not None:
                # Match full messages too: a digest alone is not permission to
                # apply a reviewed correction to different prompt content.
                if row["prompt"] != review["prompt"]:
                    raise ValueError(f"math target review prompt mismatch: {fingerprint}")
                target = raw_contract["reward_model"]["ground_truth"]
                allowed = target in review["expected_targets"] or (
                    review["action"] == "correct" and target == review["corrected_target"]
                )
                receipt = {
                    "schema": "math_target_reviews/v1",
                    "policy_sha256": policy_sha256,
                    "prompt_sha256": fingerprint,
                    "action": review["action"],
                    "original_target": target,
                    "reason": review["reason"],
                    "evidence": review["evidence"],
                    "scope": "terminal_answer_only",
                    "source_solution_reviewed": False,
                }
                if not allowed:
                    group["quarantine_reasons"].add("unexpected_review_target")
                    receipt["status"] = "unexpected_target"
                elif review["action"] == "quarantine":
                    group["quarantine_reasons"].add("reviewed_quarantine")
                    receipt["status"] = "quarantined"
                else:
                    corrected = review["corrected_target"]
                    receipt["corrected_target"] = corrected
                    receipt["status"] = "already_correct" if target == corrected else "corrected"
                    effective_row = {
                        **row,
                        "reward_model": {**row["reward_model"], "ground_truth": corrected},
                        "extra_info": {**info, "math_target_review": receipt},
                    }
            effective_contract = _math_reward_contract(effective_row)
            contract = _corpus_json(effective_contract)
            group["contracts"].setdefault(contract, effective_contract)
            if not group["variants"]:
                group["row"] = effective_row
            variant = {
                "raw": row,
                "prompt_key": prompt,
                "source": {"index": source_index, "data_source": row.get("data_source")},
                "physical_rows": 1,
                "review": receipt,
            }
            variants.append(variant)
            group["variants"].append(variant)
            group["physical_rows"] += 1

    result = []
    duplicates = []
    conflicts = []
    quarantines = []
    corrections = []
    for group in groups.values():
        if len(group["contracts"]) > 1:
            group["quarantine_reasons"].add("conflicting_effective_contracts")
        reasons = sorted(group["quarantine_reasons"])
        if not reasons:
            result.append(group["row"])
        if audit is None:
            continue
        detail = {
            "prompt_sha256": group["prompt_sha256"],
            "input_rows": len(group["variants"]),
            "physical_rows": group["physical_rows"],
            "sources": [variant["source"] for variant in group["variants"]],
            "effective_contracts": list(group["contracts"].values()),
        }
        if len(group["variants"]) > 1:
            duplicates.append(detail)
        if len(group["contracts"]) > 1:
            conflicts.append(detail)
        if reasons:
            quarantines.append({
                **detail,
                "reasons": reasons,
                "reviews": [
                    variant["review"] for variant in group["variants"]
                    if variant["review"] is not None
                ],
            })
        for variant in group["variants"]:
            receipt = variant["review"]
            if receipt is not None and receipt["status"] in {"corrected", "already_correct"}:
                corrections.append({
                    **receipt,
                    "source": variant["source"],
                    "physical_rows": variant["physical_rows"],
                    "retained": not reasons,
                })
    if audit is not None:
        collisions = [
            {
                "source": variants[0]["source"],
                "prompt_sha256": list(dict.fromkeys(
                    groups[variant["prompt_key"]]["prompt_sha256"] for variant in variants
                )),
            }
            for variants in source_cache.values()
            if len({variant["prompt_key"] for variant in variants}) > 1
        ]
        audit.clear()
        audit.update({
            "schema": MATH_CORPUS_SCHEMA,
            "policy_sha256": policy_sha256,
            "physical_rows": physical_rows,
            "input_rows": input_rows,
            "input_source_ids": len(source_cache),
            "unique_prompt_rows": len(groups),
            "output_rows": len(result),
            "physical_repeat_rows": physical_rows - input_rows,
            "duplicate_prompt_rows": input_rows - len(groups),
            "duplicate_prompt_groups": duplicates,
            "conflicting_prompt_groups": conflicts,
            "quarantined_prompt_groups": quarantines,
            "corrections": corrections,
            "source_id_collisions": collisions,
        })
    if not result:
        raise ValueError(f"{path} has no effective math corpus rows after review/quarantine")
    return result


def deterministic_math_subset(rows: list[dict], max_rows: int) -> list[dict]:
    """Select a fixed, order-preserving hash sample of unique math rows.

    DAPO is ordered and physically repeats each logical prompt, so taking a
    prefix is not a representative evaluation subset.  Ranking stable row
    identities by SHA-256 provides a reproducible sample without mutable RNG
    state; restoring dataset order afterward keeps answer-report attribution
    intuitive and evaluation batching deterministic.

    ``max_rows == 0`` means the full dataset.
    """
    if max_rows < 0:
        raise ValueError("max_rows must be nonnegative")
    if max_rows == 0 or max_rows >= len(rows):
        return list(rows)

    ranked: list[tuple[bytes, int]] = []
    for position, row in enumerate(rows):
        info = row.get("extra_info") or {}
        identity = str(info.get("index", row["prompt"][0]["content"]))
        digest = hashlib.sha256(identity.encode("utf-8")).digest()
        ranked.append((digest, position))
    selected = {
        position
        for _, position in sorted(ranked, key=lambda item: (item[0], item[1]))[
            :max_rows
        ]
    }
    return [row for position, row in enumerate(rows) if position in selected]


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
# The pretraining checkpoints pad the GPT-2 vocab (50257) to 50304 for
# kernel efficiency, so rows 50257..50303 exist in both the embedding and
# the readout but were never reachable — free real estate for new special
# tokens with no re-pretraining and no architecture change.
PRETRAIN_PADDED_VOCAB = 50304


class GPT2BPETokenizer:
    """GPT-2 byte-level BPE behind the SentencePiece surface this stack uses.

    The gpt2vocab nano variant pretrains on modded-nanogpt's GPT-2 shards,
    where the single ``<|endoftext|>`` token (50256) is both the leading
    document-boundary cue and the only stop signal — so it plays the roles
    SentencePiece splits between BOS and EOS. ``decode`` skips special
    tokens so a terminal ``<|endoftext|>`` never leaks into answer parsing.

    ``think_tokens=True`` registers ``<think>``/``</think>`` as dedicated
    special tokens in the padded-vocab slack (ids 50257/50258). Two distinct
    tokens rather than one parity fence: the close token's logit IS the
    stop-thinking policy — directly measurable, biasable, and rewardable —
    and a distinct pair cannot desync the think/output parse the way a
    dropped toggle token would. ``decode`` skips them like every special
    token, so answer parsing is unaffected.

    ``answer_tokens=True`` additionally registers ``<answer>``/``</answer>``
    (ids 50259/50260, the R1-Zero template's other half). With the answer a
    token-delimited span rather than a decoded-text ``Answer:`` field, both
    the reward gate and the verifier's extraction become purely structural
    on token ids — the case/spacing/position bypass class of a regex over
    decoded text cannot exist. Registration order is fixed (think pair
    first) so ids are stable whether or not either flag is set.
    """

    EOT_ID = 50256

    def __init__(self, think_tokens: bool = False, answer_tokens: bool = False):
        from transformers import GPT2TokenizerFast

        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        # Prompts are encoded in full and tail-truncated afterwards; the
        # 1024-token warning threshold is pretraining trivia here.
        self._tokenizer.model_max_length = 1 << 30
        self.think_open_id: int | None = None
        self.think_close_id: int | None = None
        self.answer_open_id: int | None = None
        self.answer_close_id: int | None = None
        if answer_tokens and not think_tokens:
            # The answer fence only exists to close a think span; ids also
            # depend on the think pair registering first.
            raise ValueError("answer_tokens requires think_tokens")
        if think_tokens:
            specials = [THINK_OPEN, THINK_CLOSE]
            if answer_tokens:
                specials += [ANSWER_OPEN, ANSWER_CLOSE]
            self._tokenizer.add_special_tokens(
                {"additional_special_tokens": specials}
            )
            ids = [
                int(self._tokenizer.convert_tokens_to_ids(token))
                for token in specials
            ]
            if max(ids) >= PRETRAIN_PADDED_VOCAB:
                raise ValueError(
                    "special tokens fell outside the padded pretraining "
                    f"vocab: {ids} vs {PRETRAIN_PADDED_VOCAB}"
                )
            self.think_open_id, self.think_close_id = ids[0], ids[1]
            if answer_tokens:
                self.answer_open_id, self.answer_close_id = ids[2], ids[3]

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        # The fast tokenizer's batch call is ``encode`` per text (GPT-2 adds
        # no BOS/EOS) with the Rust implementation parallel across texts.
        return self._tokenizer(list(texts))["input_ids"]

    def decode(self, ids) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=True)

    def eos_id(self) -> int:
        return self.EOT_ID

    def bos_id(self) -> int:
        return self.EOT_ID

    def id_to_piece(self, token_id: int) -> str:
        # GPT-2 pieces mark leading spaces with "Ġ", never SentencePiece's
        # "▁", so piece-based display heuristics degrade to no-ops.
        return self._tokenizer.convert_ids_to_tokens(int(token_id))


class ToastTSTPosttrainingTokenizer:
    """Post-training adapter for a checkpoint-bound ToaST+TST vocabulary."""

    def __init__(
        self,
        provenance: dict,
        think_tokens: bool = False,
        answer_tokens: bool = False,
    ) -> None:
        from pretraining.byte_accounting import load_bound_tokenizer

        if answer_tokens and not think_tokens:
            raise ValueError("answer_tokens requires think_tokens")
        self._tokenizer = load_bound_tokenizer(provenance)
        specials = tuple(self._tokenizer.spec.specials)
        self._special_ids = frozenset(range(len(specials)))

        def special_id(piece: str) -> int:
            try:
                return specials.index(piece)
            except ValueError as error:
                raise ValueError(
                    f"checkpoint tokenizer does not reserve {piece!r}"
                ) from error

        self._eot_id = special_id("<|endoftext|>")
        self.think_open_id = special_id(THINK_OPEN) if think_tokens else None
        self.think_close_id = special_id(THINK_CLOSE) if think_tokens else None
        self.answer_open_id = special_id(ANSWER_OPEN) if answer_tokens else None
        self.answer_close_id = special_id(ANSWER_CLOSE) if answer_tokens else None

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, ids) -> str:
        # Match GPT2BPETokenizer(skip_special_tokens=True): structural fences
        # and document separators guide generation but are not response text.
        # TST's numeric decoding is context-dependent, so a skipped special is
        # also a hard run boundary: filtering all specials first could merge a
        # number before </think> with one after <answer> and change its value.
        pieces: list[str] = []
        run: list[int] = []
        for value in ids:
            token = int(value)
            if token in self._special_ids:
                if run:
                    pieces.append(self._tokenizer.decode(run))
                    run = []
            else:
                run.append(token)
        if run:
            pieces.append(self._tokenizer.decode(run))
        return "".join(pieces)

    def eos_id(self) -> int:
        return self._eot_id

    def bos_id(self) -> int:
        return self._eot_id

    def id_to_piece(self, token_id: int) -> str:
        token_id = int(token_id)
        if token_id in self._special_ids:
            return self._tokenizer.spec.specials[token_id]
        return self._tokenizer.decode([token_id])


def load_posttraining_tokenizer(
    architecture: str,
    sp_model_path: str,
    think_tokens: bool = False,
    answer_tokens: bool = False,
    tokenizer_provenance: dict | None = None,
):
    """The tokenizer family the checkpoint's pretraining data was built with."""
    tokenizer_kind = (
        tokenizer_provenance.get("kind")
        if tokenizer_provenance is not None
        else None
    )
    if tokenizer_kind == "toast_tst":
        return ToastTSTPosttrainingTokenizer(
            tokenizer_provenance,
            think_tokens=think_tokens,
            answer_tokens=answer_tokens,
        )
    if tokenizer_kind not in {None, "gpt2"}:
        raise ValueError(f"unsupported checkpoint tokenizer kind {tokenizer_kind!r}")
    if "gpt2vocab" in architecture:
        return GPT2BPETokenizer(
            think_tokens=think_tokens, answer_tokens=answer_tokens
        )
    if think_tokens or answer_tokens:
        raise ValueError(
            "think/answer tokens live in the GPT-2 padded-vocab slack; "
            f"architecture {architecture!r} has no such rows"
        )
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=sp_model_path)


def single_fence_span(
    tokens, fence_ids: tuple[int, int]
) -> tuple[int, int] | None:
    """``(open_index, close_index)`` of exactly one well-ordered fence pair.

    None means the fence structure is broken — missing, duplicated, or
    reversed fences — so no span exists. Structural gates and extraction
    both key off this single definition; anything looser reopens a
    duplicated-fence ambiguity about which span is "the" span.
    """
    open_id, close_id = fence_ids
    if isinstance(tokens, np.ndarray):
        # The same scan, vectorized: iterating an array element by element
        # boxes every token, and SFT validates a whole corpus through here.
        opens = np.flatnonzero(tokens == open_id).tolist()
        closes = np.flatnonzero(tokens == close_id).tolist()
    else:
        opens = [index for index, token in enumerate(tokens) if token == open_id]
        closes = [
            index for index, token in enumerate(tokens) if token == close_id
        ]
    if len(opens) != 1 or len(closes) != 1 or closes[0] < opens[0]:
        return None
    return opens[0], closes[0]


def structural_format_ok(
    tokens,
    think_fence_ids: tuple[int, int],
    answer_fence_ids: tuple[int, int],
    min_think_tokens: int = 1,
) -> bool:
    """Require one anchored, ordered think/answer pair before terminal EOS."""
    if len(tokens) < 2:
        return False
    think_span = single_fence_span(tokens, think_fence_ids)
    if think_span is None:
        return False
    think_open, think_close = think_span
    if think_open != 0:
        return False
    if think_close - think_open - 1 < max(min_think_tokens, 1):
        return False
    answer_span = single_fence_span(tokens, answer_fence_ids)
    if answer_span is None:
        return False
    answer_open, answer_close = answer_span
    return (
        think_close < answer_open
        and answer_close - answer_open >= 2
        and answer_close == len(tokens) - 2
    )


def fenced_answer_text(
    tokens, tokenizer, answer_fence_ids: tuple[int, int]
) -> str | None:
    """Decoded inner text of the single non-empty ``<answer>`` span.

    None when the fence structure is broken or the span is empty. The
    returned text is the ONLY thing the verifier grades for a fenced
    policy — everything outside the span is structurally ungraded, which
    is what deletes the decoded-text position-check bypass class.
    """
    span = single_fence_span(tokens, answer_fence_ids)
    if span is None:
        return None
    open_index, close_index = span
    if close_index - open_index < 2:
        return None
    return tokenizer.decode(list(tokens[open_index + 1:close_index]))


def special_token_roles(tokenizer) -> dict[int, str]:
    """Special token ids the display layer keeps visible, keyed by role.

    Duck-typed over both tokenizer families: fence ids are optional
    attributes (only ``GPT2BPETokenizer`` registers them), BOS/EOS come
    from the SentencePiece-surface methods every family exposes. A shared
    BOS/EOS id (GPT-2's ``<|endoftext|>``) reports as "eos" — terminal is
    the role a token stream renderer cares about.
    """
    roles: dict[int, str] = {}
    eos = int(tokenizer.eos_id())
    if eos >= 0:
        roles[eos] = "eos"
    bos = int(tokenizer.bos_id())
    if bos >= 0:
        roles.setdefault(bos, "bos")
    for attribute, role in (
        ("think_open_id", "think_open"),
        ("think_close_id", "think_close"),
        ("answer_open_id", "answer_open"),
        ("answer_close_id", "answer_close"),
    ):
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None:
            roles[int(token_id)] = role
    return roles


def emitted_display_segments(
    tokens, tokenizer, kind: str = "text"
) -> list[dict[str, object]]:
    """Lossless display decomposition of a token stream.

    ``decode`` strips special tokens so grading never sees fence markup,
    but that same stripping blinds human-facing reports to the exact
    structure the reward gates on. This splits on special token IDS —
    never by matching decoded text, which model output could fake — into
    plain-text runs (``kind`` as given, e.g. "prefix" for teacher-forced
    prompt-suffix tokens) and ``"special"`` markers carrying the token's
    role and printable piece, so a renderer can show both faithfully.
    """
    roles = special_token_roles(tokenizer)
    segments: list[dict[str, object]] = []
    run: list[int] = []

    def flush() -> None:
        if run:
            segments.append({"kind": kind, "text": tokenizer.decode(list(run))})
            run.clear()

    for token in tokens:
        token = int(token)
        if token in roles:
            flush()
            segments.append(
                {
                    "kind": "special",
                    "role": roles[token],
                    "text": tokenizer.id_to_piece(token),
                    "token_id": token,
                    # Special markers keep their provenance too, so a
                    # teacher-forced fence or BOS in the prompt suffix can
                    # never render as model-emitted.
                    "source": kind,
                }
            )
        else:
            run.append(token)
    flush()
    return segments


def encode_many(tokenizer, texts: list[str]) -> list[list[int]]:
    """``[tokenizer.encode(text) for text in texts]``, batched when possible.

    Tokenizers with a native batched encoder expose it as ``encode_batch``;
    the GPT-2 one runs the Rust tokenizer across every core instead of one
    Python call per text, which is the difference between minutes and hours
    of CPU on a million-document corpus. Results are identical to per-text
    ``encode``.
    """
    encode_batch = getattr(tokenizer, "encode_batch", None)
    if encode_batch is None:
        return [list(tokenizer.encode(text)) for text in texts]
    return encode_batch(texts)


def encode_prompt(tokenizer, text: str, max_tokens: int | None = None) -> list[int]:
    """Encode a prompt the way pretraining framed documents: BOS-first.

    Every pretraining document was sharded as ``[BOS] tokens`` — no EOS is
    ever appended (``APPEND_EOS`` is off in the shard writer) — so BOS is the
    model's only learned document-boundary cue.  Prompting without it frames
    the problem as a mid-document continuation.  Truncation (``max_tokens``)
    keeps the BOS plus the LAST ``max_tokens - 1`` content tokens.
    """
    return frame_prompt(tokenizer, tokenizer.encode(text), max_tokens)


def frame_prompt(
    tokenizer, ids, max_tokens: int | None = None
) -> list[int]:
    """``encode_prompt``'s BOS framing and truncation over encoded ids."""
    ids = list(ids)
    bos = tokenizer.bos_id()
    if bos < 0:
        return ids if max_tokens is None else ids[-max_tokens:]
    if max_tokens is not None:
        # ids[-0:] would be the whole list, not the empty tail.
        ids = ids[-(max_tokens - 1):] if max_tokens > 1 else []
    return [bos] + ids


def length_adaptive_lambda(lengths: Tensor, alpha: float = 0.05) -> Tensor:
    """VAPO's length-adaptive GAE lambda with a floored credit horizon.

    VAPO sets the credit horizon 1/(1 - lambda) = alpha*l, calibrated for
    thousand-token responses; below l = 1/alpha the raw formula prescribes a
    sub-one-step horizon (lambda <= 0 — TD(0)), which severs direct reward
    credit exactly when trajectories are short enough for full-length credit
    to be cheap.  The horizon is therefore floored at min(l, 1/alpha): short
    responses get whole-trajectory credit, mid lengths get the fixed
    lambda = 1 - alpha baseline VAPO generalized (0.95 at alpha = 0.05), and
    long responses recover VAPO's alpha*l exactly.
    """
    lengths = lengths.to(torch.float32).clamp_min(1)
    horizon = torch.maximum(
        alpha * lengths, torch.minimum(lengths, torch.full_like(lengths, 1.0 / alpha))
    )
    return (1.0 - 1.0 / horizon).clamp(0.0, 1.0)


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    lambdas: Tensor,
    gamma: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Masked per-token GAE; lambdas is one scalar per trajectory."""
    advantage = torch.zeros_like(values)
    running = torch.zeros(values.size(0), device=values.device, dtype=values.dtype)
    for t in range(values.size(1) - 1, -1, -1):
        next_value = values[:, t + 1] if t + 1 < values.size(1) else torch.zeros_like(running)
        next_valid = mask[:, t + 1] if t + 1 < mask.size(1) else torch.zeros_like(mask[:, t])
        delta = rewards[:, t] + gamma * next_value * next_valid - values[:, t]
        running = delta + gamma * lambdas * running * next_valid
        running = running * mask[:, t]
        advantage[:, t] = running
    return advantage, advantage + values


def generalized_advantage_and_return_targets(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    lambdas: Tensor,
    gamma: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Compute lambda-GAE and lambda-one return targets in one reverse pass.

    The two recurrences share the same temporal-difference residual. Keeping
    both running accumulators in one loop halves eager launch overhead while
    remaining exactly row-separable, so callers can compute the complete
    optimizer minibatch once and slice the results for replay shards.
    """
    # Everything that does not carry the recurrence is computed once for the
    # whole stream instead of per column: the residual, the shifted validity,
    # and the loop-invariant gamma*lambdas product. The remaining loop body
    # is bit-identical -- these are the same elementwise expressions in the
    # same association order, just evaluated for every t at once -- and drops
    # the eager launch count per column from 15 to 8, which is what this
    # dispatch-bound reverse scan actually costs.
    length = values.size(1)
    if length == 0:
        # ``torch.stack`` cannot rebuild an empty stream, which the old
        # preallocated writes handled implicitly.
        return torch.zeros_like(values), values.clone()
    zero_column = torch.zeros_like(values[:, :1])
    next_values = torch.cat((values[:, 1:], zero_column), dim=1)
    next_valids = torch.cat(
        (mask[:, 1:], torch.zeros_like(mask[:, :1])), dim=1
    )
    # The trainer uses undiscounted episodic returns (gamma=1). Leaving the
    # identity multiply in the eager reverse scan launches one pointwise CUDA
    # kernel per stream column, at the deepest point of the update queue.
    # Skipping multiplication by exactly one is bit-exact for IEEE tensors;
    # keep the general discounted path for callers that choose another gamma.
    if gamma == 1.0:
        discounted_next_values = next_values
        gamma_lambdas = lambdas
    else:
        discounted_next_values = gamma * next_values
        gamma_lambdas = gamma * lambdas
    deltas = rewards + discounted_next_values * next_valids - values
    running_advantage = torch.zeros(
        values.size(0), device=values.device, dtype=values.dtype
    )
    running_return = torch.zeros_like(running_advantage)
    advantage_columns: list[Tensor] = []
    return_columns: list[Tensor] = []
    for t in range(length - 1, -1, -1):
        delta = deltas[:, t]
        next_valid = next_valids[:, t]
        running_advantage = (
            delta + gamma_lambdas * running_advantage * next_valid
        ) * mask[:, t]
        discounted_running_return = (
            running_return if gamma == 1.0 else gamma * running_return
        )
        running_return = (
            delta + discounted_running_return * next_valid
        ) * mask[:, t]
        advantage_columns.append(running_advantage)
        return_columns.append(running_return)
    # Columns were accumulated from the last position backwards; one stacking
    # pass replaces two indexed stores per column. The stores also cast each
    # column back to ``values.dtype`` -- a mask or lambdas in a wider dtype
    # promotes the accumulators -- so the cast has to be restated here to keep
    # the result bit-identical.
    advantages = torch.stack(advantage_columns[::-1], dim=1).to(values.dtype)
    return_advantages = torch.stack(return_columns[::-1], dim=1).to(
        values.dtype
    )
    return advantages, return_advantages + values


def clipped_policy_loss(
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    epsilon_low: float = 0.20,
    epsilon_high: float = 0.28,
    denominator: Tensor | None = None,
    estimate_kl: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """VAPO's token-level clipped surrogate for one action per position.

    ``new_logprobs`` and ``old_logprobs`` are the log probabilities of the
    complete action.  A factorized action must therefore be combined in log
    space before calling this function.  ``denominator`` may cover a larger
    optimizer minibatch than this memory shard; summing shard losses then
    reproduces VAPO Eq. 7 exactly.
    """
    log_ratio = torch.where(
        mask.bool(), new_logprobs - old_logprobs, torch.zeros_like(new_logprobs)
    )
    # ``new_full`` rather than ``new_tensor``: both materialize the same
    # constant in the same dtype on the same device, but new_tensor builds it
    # on the host and copies, and a copy from pageable memory blocks until
    # the stream drains -- twice per replay shard, at the top of the eager
    # tail. new_full is a fill kernel and takes the scalar as an argument.
    log_lower = torch.log(log_ratio.new_full((), 1.0 - epsilon_low))
    log_upper = torch.log(log_ratio.new_full((), 1.0 + epsilon_high))
    # This is algebraically the standard min(r*A, clip(r)*A), expressed in
    # log space so a favorable but extremely large joint ratio is clipped
    # before exp. PPO deliberately leaves the harmful direction unclipped.
    effective_log_ratio = torch.where(
        advantages >= 0,
        torch.minimum(log_ratio, log_upper),
        torch.maximum(log_ratio, log_lower),
    )
    objective = effective_log_ratio.exp() * advantages
    denom = (
        mask.sum() if denominator is None else denominator.to(mask.device)
    ).clamp_min(1)
    loss = -(objective * mask).sum() / denom
    clip_fraction = (
        ((log_ratio < log_lower) | (log_ratio > log_upper)) * mask.bool()
    ).sum() / denom
    # Schulman's non-negative k3 estimator. Actions come from the frozen
    # behavior policy, so its expectation is KL(old || new). Reusing the PPO
    # ratio makes the diagnostic effectively free compared with actor replay.
    with torch.no_grad():
        approximate_kl = (
            ((torch.expm1(log_ratio) - log_ratio) * mask).sum() / denom
            if estimate_kl
            else loss.detach().new_zeros(())
        )
    return loss, clip_fraction, approximate_kl


def delightful_policy_loss(
    logprobs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    temperature: float = 1.0,
    denominator: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Delightful Policy Gradient for discrete token actions.

    Implements Algorithm 1 of Osband (2026): every score term is weighted by
    ``sigmoid(advantage * surprisal / temperature)``, where surprisal is the
    negative log probability of the sampled token under the *current* policy.
    The gate is deliberately detached. DG specifies a gradient estimator, not
    the gradient of the scalar expression used to construct its gate; allowing
    autograd through the sigmoid would add an unprescribed second-order term.

    No behavior-policy importance ratio or PPO clipping appears in this
    objective. ``denominator`` may span multiple replay shards, just as in
    :func:`clipped_policy_loss`.
    """
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("delight temperature must be finite and positive")
    active = mask.bool()
    detached_advantages = advantages.detach()
    surprisal = -logprobs.detach()
    delight = detached_advantages * surprisal
    gate = torch.sigmoid(delight / temperature).detach()
    denom = (
        mask.sum() if denominator is None else denominator.to(mask.device)
    ).clamp_min(1)
    loss_terms = -gate * detached_advantages * logprobs * mask
    loss = loss_terms.sum() / denom

    positive = active & (detached_advantages > 0)
    negative = active & (detached_advantages < 0)
    diagnostics = {
        "gate_sum": gate.masked_fill(~active, 0.0).sum(),
        "positive_gate_sum": gate.masked_fill(~positive, 0.0).sum(),
        "positive_count": positive.sum(),
        "negative_gate_sum": gate.masked_fill(~negative, 0.0).sum(),
        "negative_count": negative.sum(),
        "delight_sum": delight.masked_fill(~active, 0.0).sum(),
        "surprisal_sum": surprisal.masked_fill(~active, 0.0).sum(),
        # These detached scalar contributions explain the sign of the
        # autograd surrogate. They are not exploration/exploitation metrics:
        # positive-advantage terms reinforce sampled actions, while negative-
        # advantage terms suppress them.
        "positive_loss_sum": loss_terms.detach().masked_fill(
            ~positive, 0.0
        ).sum(),
        "negative_loss_sum": loss_terms.detach().masked_fill(
            ~negative, 0.0
        ).sum(),
    }
    return loss, diagnostics


def target_policy_loss(
    new_log_odds: Tensor,
    old_log_odds: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    eta: float = 2.0,
    denominator: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Fit an intra-trajectory target on executed-token-versus-rest odds.

    At each visited prefix, raw detached GAE shifts the behavior policy's
    executed-token log odds by ``A / eta``.  This defines a normalized and
    feasible local Bernoulli target over the partition {executed token, every
    other token}.  Binary cross entropy fits the current executed-token
    probability to that target, and its isolated gradient extinguishes there.
    Targets from repeated identical prefixes can conflict across sampled
    tokens, in which case the shared categorical policy fits their compromise.
    This is the discrete counterpart of the old-policy-anchored intra-
    trajectory target used by ``ppo_continuous_action_tpo_intra_beta_v2.py``.

    Advantages are deliberately neither centered nor normalized here.  The
    verifier return and critic targets both live on [0, 1], and with gamma=1
    their lambda-GAE stays on that meaningful native scale (up to the critic
    support's narrow margin bins).  ``eta`` is the target-space trust control:
    it limits the requested odds move relative to the frozen rollout policy,
    though it does not impose a hard bound on the optimizer's realized KL.
    There is no policy-gradient auxiliary,
    counterfactual action value, sampled comparison action, or PPO clip.

    Every active token is one transition, matching the CleanRL intra-TPO
    reference and the existing VAPO actor reduction. ``denominator`` may span
    complete replay shards and prompt groups, so memory partitioning does not
    alter the global token mean.
    """
    if new_log_odds.shape != old_log_odds.shape:
        raise ValueError("old and current token log odds must align")
    if advantages.shape != new_log_odds.shape or mask.shape != new_log_odds.shape:
        raise ValueError("TPO log odds, advantages, and mask must align")
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("TPO eta must be finite and positive")

    active = mask.bool()
    token_mask = mask.float()
    current_log_odds = torch.where(
        active,
        new_log_odds.float(),
        torch.zeros_like(new_log_odds, dtype=torch.float32),
    )
    behavior_log_odds = torch.where(
        active,
        old_log_odds.detach().float(),
        torch.zeros_like(old_log_odds, dtype=torch.float32),
    )
    raw_advantages = advantages.detach().float()
    target_log_odds_shift = torch.where(
        active,
        raw_advantages / eta,
        torch.zeros_like(raw_advantages),
    )

    target_log_odds = behavior_log_odds + target_log_odds_shift
    target_probability = torch.sigmoid(target_log_odds).detach()

    def endpoint_safe_binary_cross_entropy(
        logits: Tensor, targets: Tensor
    ) -> Tensor:
        positive_loss = F.softplus(-logits)
        negative_loss = F.softplus(logits)
        interior_loss = (
            targets * positive_loss + (1.0 - targets) * negative_loss
        )
        return torch.where(
            targets == 1.0,
            positive_loss,
            torch.where(targets == 0.0, negative_loss, interior_loss),
        )

    loss_terms = endpoint_safe_binary_cross_entropy(
        current_log_odds, target_probability
    )
    denom = (
        token_mask.sum()
        if denominator is None
        else denominator.to(mask.device)
    ).clamp_min(1)
    loss = (loss_terms * token_mask).sum() / denom

    current_probability = torch.sigmoid(current_log_odds)
    old_probability = torch.sigmoid(behavior_log_odds)
    residual = current_probability - target_probability
    target_entropy = (
        torch.special.entr(target_probability)
        + torch.special.entr(1.0 - target_probability)
    )
    fit_kl = loss_terms - target_entropy
    target_behavior_kl = (
        endpoint_safe_binary_cross_entropy(
            behavior_log_odds, target_probability
        )
        - target_entropy
    )
    positive = (raw_advantages > 0) & active
    negative = (raw_advantages < 0) & active
    neutral = (raw_advantages == 0) & active
    positive_mask = token_mask.masked_fill(~positive, 0.0)
    negative_mask = token_mask.masked_fill(~negative, 0.0)
    neutral_mask = token_mask.masked_fill(~neutral, 0.0)
    diagnostics = {
        "active_count": token_mask.sum(),
        "loss_sum": (loss_terms.detach() * token_mask).sum(),
        "fit_kl_sum": (fit_kl.detach() * token_mask).sum(),
        "target_behavior_kl_sum": (target_behavior_kl * token_mask).sum(),
        "old_probability_sum": (old_probability * token_mask).sum(),
        "current_probability_sum": (
            current_probability.detach() * token_mask
        ).sum(),
        "target_probability_sum": (target_probability * token_mask).sum(),
        "target_move_abs_sum": (
            (target_probability - old_probability).abs()
            * token_mask
        ).sum(),
        "target_log_odds_shift_abs_sum": (
            target_log_odds_shift.abs() * token_mask
        ).sum(),
        "target_log_odds_shift_square_sum": (
            target_log_odds_shift.square() * token_mask
        ).sum(),
        "residual_sum": (residual.detach() * token_mask).sum(),
        "residual_abs_sum": (residual.detach().abs() * token_mask).sum(),
        "residual_square_sum": (
            residual.detach().square() * token_mask
        ).sum(),
        "positive_count": positive_mask.sum(),
        "negative_count": negative_mask.sum(),
        "neutral_count": neutral_mask.sum(),
        "positive_target_probability_sum": (
            target_probability * positive_mask
        ).sum(),
        "negative_target_probability_sum": (
            target_probability * negative_mask
        ).sum(),
    }
    return loss, diagnostics


def masked_token_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def top_p_sample(
    logits: Tensor,
    temperature: float,
    top_p: float,
    *,
    generator: torch.Generator | None = None,
    top_k: int | None = None,
    num_samples: int = 1,
) -> Tensor:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    candidate_support = (
        top_k
        if top_k is not None and 0 < top_k < logits.size(-1)
        else logits.size(-1)
    )
    if num_samples > candidate_support:
        raise ValueError(
            f"cannot draw {num_samples} unique samples from "
            f"support of size {candidate_support}"
        )
    if temperature < 0.0:
        raise ValueError(f"temperature must be nonnegative, got {temperature}")
    if not 0.0 <= top_p <= 1.0:
        # Out of range, or NaN, which fails both comparisons. A negative top_p
        # masks every rank and surfaces as the same opaque multinomial error a
        # corrupted forward gives, so it is worth separating here.
        raise ValueError(f"top_p must lie in [0, 1], got {top_p}")
    logits = logits.float()
    if temperature == 0.0:
        # Greedy decoding, by the usual convention. Dividing by zero would
        # give +inf for every positive logit and NaN for a zero one, and
        # multinomial refuses that outright -- so a caller asking for the
        # argmax path has to be answered here rather than arithmetically.
        #
        # Expressed as a one-hot distribution rather than an argmax shortcut
        # so the draw order stays one multinomial per step: that order is part
        # of the rollout's execution schema, and a caller that skipped a draw
        # would desynchronize a shared generator from a sampled rollout's.
        if num_samples > 1:
            raise ValueError(
                "temperature 0 is greedy decoding and has a single outcome; "
                f"cannot draw {num_samples} distinct samples"
            )
        # multinomial refusing a NaN, +inf, or wholly masked row is this
        # codebase's de-facto detector for a corrupted forward pass or an
        # over-aggressive mask, and argmax has no such reflex: it ranks NaN
        # above every real logit, picks a lone +inf outright, and returns
        # index 0 for an all -inf row. Greedy decoding would otherwise turn a
        # broken forward into a complete, plausible-looking set of predictions
        # that every downstream contract accepts.
        #
        # +inf is refused for parity rather than because argmax is ambiguous
        # about it: a stable softmax subtracts the row max, so `exp(inf - inf)`
        # is NaN and the sampled path raises on the same logits. -inf stays
        # legal -- it is the mask value, and a row masked down to one candidate
        # is a request, not a corruption.
        if (
            logits.isnan().any()
            or logits.eq(torch.inf).any()
            or not logits.isfinite().any(dim=-1).all()
        ):
            raise ValueError(
                "temperature-0 decoding needs at least one finite logit per "
                "row and no NaN or +inf; argmax would silently rank those "
                "above every real logit where sampling refuses them"
            )
        logits = torch.full_like(logits, -torch.inf).scatter(
            -1, logits.argmax(dim=-1, keepdim=True), 0.0
        )
    elif temperature != 1.0:
        # Dividing by exactly 1.0 is the identity in IEEE arithmetic, so the
        # skip is bit-exact -- but the kernel is not free: it reads and writes
        # a (rows, 50304) fp32 tensor at every rollout step of the training
        # configuration, which samples at temperature 1.0.
        logits = logits / temperature
    top_indices = None
    if top_k is not None and 0 < top_k < logits.size(-1):
        logits, top_indices = logits.topk(top_k, dim=-1)
    if top_p >= 1.0:
        # Nucleus truncation is a no-op at top_p >= 1 (cumsum - probs never
        # exceeds 1), and the same distribution needs no full-vocab sort —
        # which otherwise runs at EVERY rollout step of the top-p-1 training
        # configuration.
        sampled = torch.multinomial(
            logits.softmax(dim=-1), num_samples,
            replacement=False,
            generator=generator,
        )
        if top_indices is not None:
            sampled = top_indices.gather(-1, sampled)
        return sampled.squeeze(-1) if num_samples == 1 else sampled
    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    remove = probs.cumsum(dim=-1) - probs > top_p
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    sampled = torch.multinomial(
        sorted_logits.softmax(dim=-1), num_samples,
        replacement=False,
        generator=generator,
    )
    sampled = sorted_indices.gather(-1, sampled).squeeze(-1)
    if top_indices is not None:
        sampled = top_indices.gather(
            -1, sampled[..., None] if num_samples == 1 else sampled
        )
        if num_samples == 1:
            sampled = sampled.squeeze(-1)
    return sampled


@dataclass
class TrajectoryBatch:
    input_ids: Tensor
    target_ids: Tensor
    response_mask: Tensor
    old_logprobs: Tensor
    old_values: Tensor
    rewards: Tensor
    correct: Tensor
    texts: list[str]

    def to(self, device: torch.device) -> "TrajectoryBatch":
        return TrajectoryBatch(
            **{
                name: value.to(device) if torch.is_tensor(value) else value
                for name, value in self.__dict__.items()
            }
        )


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **values) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, sort_keys=True) + "\n")

    def purge_value_warmup_after(self, warmup_step: int) -> int:
        """Atomically drop warmup telemetry newer than a resume checkpoint."""
        if not self.path.exists():
            return 0
        retained: list[str] = []
        removed = 0
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                stale = (
                    record.get("type") == "value_warmup"
                    and int(record["step"]) > warmup_step
                )
                if stale:
                    removed += 1
                else:
                    retained.append(line)
        if removed:
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                handle.writelines(retained)
            temporary.replace(self.path)
        return removed

    def purge_after(self, step: int, warmup_step: int) -> int:
        """Drop telemetry newer than a resumable checkpoint atomically.

        Rollout rows are indexed by the final optimizer step their behavior
        pool feeds, so all actor/eval records newer than ``step`` are stale.
        Warmup has its own independent counter.
        """
        if not self.path.exists():
            return 0
        retained: list[str] = []
        removed = 0
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                record_type = record.get("type")
                record_step = record.get("step")
                stale = False
                if record_step is not None:
                    numeric_step = int(record_step)
                    if record_type == "value_warmup":
                        stale = numeric_step > warmup_step
                    else:
                        stale = numeric_step > step
                if stale:
                    removed += 1
                else:
                    retained.append(line)
        if removed:
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                handle.writelines(retained)
            temporary.replace(self.path)
        return removed
