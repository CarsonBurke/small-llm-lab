"""Pure VAPO math, verification, data, and rollout helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

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
    solution: str, ground_truth: str, style: str = "minerva"
) -> tuple[bool, str]:
    if style == "minerva":
        extracted = extract_final_answer(solution)
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
        extracted = extract_final_answer(solution)
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


def load_unique_math_rows(path: str | Path) -> list[dict]:
    """Load and deduplicate DAPO's physically repeated parquet rows."""
    columns = ["prompt", "reward_model", "extra_info"]
    unique: dict[str, dict] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        for row in batch.to_pylist():
            info = row.get("extra_info") or {}
            key = str(info.get("index", row["prompt"][0]["content"]))
            unique.setdefault(key, row)
    return list(unique.values())


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


class GPT2BPETokenizer:
    """GPT-2 byte-level BPE behind the SentencePiece surface this stack uses.

    The gpt2vocab nano variant pretrains on modded-nanogpt's GPT-2 shards,
    where the single ``<|endoftext|>`` token (50256) is both the leading
    document-boundary cue and the only stop signal — so it plays the roles
    SentencePiece splits between BOS and EOS. ``decode`` skips special
    tokens so a terminal ``<|endoftext|>`` never leaks into answer parsing.
    """

    EOT_ID = 50256

    def __init__(self):
        from transformers import GPT2TokenizerFast

        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        # Prompts are encoded in full and tail-truncated afterwards; the
        # 1024-token warning threshold is pretraining trivia here.
        self._tokenizer.model_max_length = 1 << 30

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

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


def load_posttraining_tokenizer(architecture: str, sp_model_path: str):
    """The tokenizer family the checkpoint's pretraining data was built with."""
    if "gpt2vocab" in architecture:
        return GPT2BPETokenizer()
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=sp_model_path)


def encode_prompt(tokenizer, text: str, max_tokens: int | None = None) -> list[int]:
    """Encode a prompt the way pretraining framed documents: BOS-first.

    Every pretraining document was sharded as ``[BOS] tokens`` — no EOS is
    ever appended (``APPEND_EOS`` is off in the shard writer) — so BOS is the
    model's only learned document-boundary cue.  Prompting without it frames
    the problem as a mid-document continuation.  Truncation (``max_tokens``)
    keeps the BOS plus the LAST ``max_tokens - 1`` content tokens.
    """
    ids = list(tokenizer.encode(text))
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
    deltas = rewards + gamma * next_values * next_valids - values
    gamma_lambdas = gamma * lambdas
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
        running_return = (
            delta + gamma * running_return * next_valid
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


def masked_token_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def positive_example_lm_loss(
    logprobs: Tensor,
    mask: Tensor,
    correct: Tensor,
    denominator: Tensor | None = None,
) -> Tensor:
    """VAPO Eq. 9: NLL averaged over every token in correct responses."""
    positive_mask = mask * correct[:, None]
    denom = (
        positive_mask.sum()
        if denominator is None
        else denominator.to(mask.device)
    ).clamp_min(1)
    return -(logprobs * positive_mask).sum() / denom


def top_p_sample(logits: Tensor, temperature: float, top_p: float) -> Tensor:
    logits = logits.float()
    if temperature != 1.0:
        # Dividing by exactly 1.0 is the identity in IEEE arithmetic, so the
        # skip is bit-exact -- but the kernel is not free: it reads and writes
        # a (rows, 50304) fp32 tensor at every rollout step of the training
        # configuration, which samples at temperature 1.0.
        logits = logits / temperature
    if top_p >= 1.0:
        # Nucleus truncation is a no-op at top_p >= 1 (cumsum - probs never
        # exceeds 1), and the same distribution needs no full-vocab sort —
        # which otherwise runs at EVERY rollout step of the top-p-1 training
        # configuration.
        return torch.multinomial(logits.softmax(dim=-1), 1).squeeze(-1)
    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    remove = probs.cumsum(dim=-1) - probs > top_p
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    sampled = torch.multinomial(sorted_logits.softmax(dim=-1), 1)
    return sorted_indices.gather(-1, sampled).squeeze(-1)


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
