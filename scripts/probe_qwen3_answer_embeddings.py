#!/usr/bin/env python3
"""Audit Qwen3-Embedding as a dense reward for answer correctness."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import transformers
from scipy.stats import spearmanr
from torch import Tensor

from postraining.answer_encoder.probes import (
    DEFAULT_PROBES,
    FUNCTIONS_WRONG,
    FUNCTIONS_XY,
    FUNCTIONS_YX,
    run_behavioral_probes,
)
from postraining.answer_encoder.model import (
    DEFAULT_REWARD_TEMPERATURE,
    cosine_kernel_reward,
)


DEFAULT_INSTRUCTION = (
    "Given a correct reference answer, retrieve attempted answers that are "
    "semantically related, prioritizing the same conclusion, mathematical "
    "result, or program behavior."
)
OFFICIAL_MRL_MIN_DIMENSION = 32


def disable_optional_vision_imports() -> None:
    """Keep a text-only model independent of an installed torchvision build."""

    # Transformers' generic processing layer conditionally imports torchvision
    # while loading Qwen3Model, even though this checkpoint accepts only text.
    # Treat that optional package as unavailable instead of requiring its native
    # operators to match the locally installed PyTorch.
    import transformers.utils
    import transformers.utils.import_utils

    transformers.utils.is_torchvision_available = lambda: False
    transformers.utils.import_utils.is_torchvision_available = lambda: False


@dataclass(frozen=True)
class GradedCandidate:
    label: str
    text: str
    quality: float


@dataclass(frozen=True)
class GradedCase:
    name: str
    target: str
    candidates: tuple[GradedCandidate, ...]


def detailed_instruction(instruction: str, text: str) -> str:
    """Use the query format recommended by the Qwen3-Embedding model card."""

    return f"Instruct: {instruction}\nQuery:{text}"


def last_token_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the final non-padding token for either left- or right-padded batches."""

    if bool((attention_mask[:, -1] == 1).all()):
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(
        last_hidden_state.shape[0], device=last_hidden_state.device
    )
    return last_hidden_state[batch_indices, sequence_lengths]


def truncate_and_normalize(embedding: Tensor, dimension: int) -> Tensor:
    if dimension <= 0 or dimension > embedding.shape[-1]:
        raise ValueError(
            f"dimension must be in [1, {embedding.shape[-1]}], got {dimension}"
        )
    return F.normalize(embedding[..., :dimension].float(), p=2, dim=-1)


def dimension_metadata(dimension: int, hidden_size: int) -> dict:
    if dimension <= 0 or dimension > hidden_size:
        raise ValueError(f"dimension must be in [1, {hidden_size}], got {dimension}")
    supported = dimension >= OFFICIAL_MRL_MIN_DIMENSION
    return {
        "value": dimension,
        "official_mrl_supported": supported,
        "status": (
            "supported"
            if supported
            else "below_official_minimum_extrapolation"
        ),
    }


def _unique_in_order(texts: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(texts))


def build_graded_cases() -> tuple[GradedCase, ...]:
    """Construct semantic, mathematical, and executable-behavior reward cases."""

    cases = [
        GradedCase(
            name="numeric_10",
            target="10",
            candidates=(
                GradedCandidate("exact", "10", 1.0),
                GradedCandidate("decimal_equivalent", "10.0", 1.0),
                GradedCandidate("word_equivalent", "ten", 1.0),
                GradedCandidate("near_integer", "9", 0.5),
                GradedCandidate("near_decimal", "10.1", 1.0 / 1.1),
                GradedCandidate("far_50", "50", 1.0 / 41.0),
                GradedCandidate("far_100", "100", 1.0 / 91.0),
            ),
        ),
        GradedCase(
            name="negation",
            target="The value is greater than ten.",
            candidates=(
                GradedCandidate("exact", "The value is greater than ten.", 1.0),
                GradedCandidate("paraphrase", "The value exceeds 10.", 1.0),
                GradedCandidate(
                    "negated", "The value is not greater than ten.", 0.0
                ),
                GradedCandidate("reversed", "The value is less than ten.", 0.0),
            ),
        ),
        GradedCase(
            name="independent_functions",
            target=FUNCTIONS_XY,
            candidates=(
                GradedCandidate("exact", FUNCTIONS_XY, 1.0),
                GradedCandidate("reordered", FUNCTIONS_YX, 1.0),
                GradedCandidate("one_function_bugged", FUNCTIONS_WRONG, 0.5),
                GradedCandidate(
                    "both_bugged",
                    "def x(value):\n    return value - 1\n\n"
                    "def y(value):\n    return value / 2\n",
                    0.0,
                ),
                GradedCandidate("unrelated", "print('hello world')\n", 0.0),
            ),
        ),
        GradedCase(
            name="linear_equation",
            target="Solve 2x + 3 = 13. Subtracting 3 gives 2x = 10, so x = 5.",
            candidates=(
                GradedCandidate(
                    "exact",
                    "Solve 2x + 3 = 13. Subtracting 3 gives 2x = 10, so x = 5.",
                    1.0,
                ),
                GradedCandidate(
                    "equivalent_reasoning",
                    "From 2x + 3 = 13, we get 2x = 10 and therefore x = 5.",
                    1.0,
                ),
                GradedCandidate("answer_only", "5", 1.0),
                GradedCandidate(
                    "arithmetic_slip",
                    "From 2x + 3 = 13, we get 2x = 10 and therefore x = 4.",
                    0.0,
                ),
                GradedCandidate(
                    "wrong_operation",
                    "Add 3 to both sides to get 2x = 16, so x = 8.",
                    0.0,
                ),
                GradedCandidate("unrelated", "The capital of France is Paris.", 0.0),
            ),
        ),
        GradedCase(
            name="factorial_program",
            target=(
                "def factorial(n):\n"
                "    result = 1\n"
                "    for value in range(2, n + 1):\n"
                "        result *= value\n"
                "    return result\n"
            ),
            candidates=(
                GradedCandidate(
                    "exact",
                    "def factorial(n):\n"
                    "    result = 1\n"
                    "    for value in range(2, n + 1):\n"
                    "        result *= value\n"
                    "    return result\n",
                    1.0,
                ),
                GradedCandidate(
                    "equivalent_library",
                    "import math\n\ndef factorial(n):\n"
                    "    return math.prod(range(1, n + 1))\n",
                    1.0,
                ),
                GradedCandidate(
                    "off_by_one",
                    "def factorial(n):\n"
                    "    result = 1\n"
                    "    for value in range(2, n):\n"
                    "        result *= value\n"
                    "    return result\n",
                    0.25,
                ),
                GradedCandidate(
                    "constant_one", "def factorial(n):\n    return 1\n", 0.1
                ),
                GradedCandidate(
                    "plus_one",
                    "def factorial(n):\n"
                    "    result = 1\n"
                    "    for value in range(2, n + 1):\n"
                    "        result *= value\n"
                    "    return result + 1\n",
                    0.0,
                ),
                GradedCandidate("unrelated", "def greet(name):\n    return f'Hi {name}'\n", 0.0),
            ),
        ),
    ]

    for target in range(-20, 21):
        candidates = [
            GradedCandidate("exact", str(target), 1.0),
            GradedCandidate("decimal_equivalent", f"{target}.0", 1.0),
        ]
        for offset in (-50, -10, -1, 1, 10, 50):
            error = abs(offset)
            candidates.append(
                GradedCandidate(
                    f"offset_{offset:+d}",
                    str(target + offset),
                    1.0 / (1.0 + error),
                )
            )
        cases.append(
            GradedCase(
                name=f"numeric_sweep_{target:+d}",
                target=str(target),
                candidates=tuple(candidates),
            )
        )
    return tuple(cases)


class CachedQwenScorer:
    """Adapt cached Qwen embeddings to the project's behavioral probe protocol."""

    def __init__(
        self,
        candidate_embeddings: Mapping[str, Tensor],
        target_embeddings: Mapping[str, Tensor],
        dimension: int,
        mode: str,
        reward_temperature: float = DEFAULT_REWARD_TEMPERATURE,
    ) -> None:
        self._candidate_embeddings = candidate_embeddings
        self._target_embeddings = target_embeddings
        self.dimension = dimension
        self.space = f"qwen3_embedding/{mode}/{dimension}d"
        cosine_kernel_reward(torch.ones(1), temperature=reward_temperature)
        self.reward_temperature = reward_temperature

    def preencode_target(self, target: str) -> Tensor:
        return truncate_and_normalize(self._target_embeddings[target], self.dimension)

    def score(self, answers: Sequence[str], target_embedding: Tensor) -> Tensor:
        cosine = self.cosine(answers, target_embedding)
        return cosine_kernel_reward(cosine, temperature=self.reward_temperature)

    def cosine(self, answers: Sequence[str], target_embedding: Tensor) -> Tensor:
        candidate_batch = torch.stack(
            [self._candidate_embeddings[answer] for answer in answers]
        )
        candidates = truncate_and_normalize(candidate_batch, self.dimension)
        return (candidates @ target_embedding).clamp(-1.0, 1.0)


def _case_metrics(qualities: Sequence[float], cosine: Sequence[float]) -> dict:
    comparable = 0
    concordant = 0.0
    for left in range(len(qualities)):
        for right in range(left + 1, len(qualities)):
            if math.isclose(qualities[left], qualities[right], abs_tol=1e-12):
                continue
            comparable += 1
            expected = qualities[left] > qualities[right]
            actual_delta = cosine[left] - cosine[right]
            if math.isclose(actual_delta, 0.0, abs_tol=1e-12):
                concordant += 0.5
            elif (actual_delta > 0.0) == expected:
                concordant += 1.0

    correlation = spearmanr(qualities, cosine).statistic
    correct = [score for quality, score in zip(qualities, cosine) if quality == 1.0]
    incorrect = [score for quality, score in zip(qualities, cosine) if quality < 1.0]
    return {
        "spearman": None if math.isnan(float(correlation)) else float(correlation),
        "pairwise_correct": float(concordant / comparable) if comparable else None,
        "pair_count": comparable,
        "correct_min_minus_incorrect_max": (
            float(min(correct) - max(incorrect)) if correct and incorrect else None
        ),
        "cosine_span": float(max(cosine) - min(cosine)),
    }


def run_graded_audit(
    scorer: CachedQwenScorer, cases: Sequence[GradedCase]
) -> dict:
    case_reports: dict[str, dict] = {}
    total_pairs = 0
    weighted_concordance = 0.0
    correlations: list[float] = []
    margins: list[float] = []

    for case in cases:
        target = scorer.preencode_target(case.target)
        rewards = scorer.score([item.text for item in case.candidates], target)
        cosine = scorer.cosine(
            [item.text for item in case.candidates], target
        ).tolist()
        qualities = [item.quality for item in case.candidates]
        metrics = _case_metrics(qualities, cosine)
        case_reports[case.name] = {
            "candidates": {
                item.label: {
                    "quality": item.quality,
                    "cosine": float(cosine[index]),
                    "reward": float(rewards[index]),
                }
                for index, item in enumerate(case.candidates)
            },
            "metrics": metrics,
        }
        if metrics["pairwise_correct"] is not None:
            total_pairs += metrics["pair_count"]
            weighted_concordance += (
                metrics["pairwise_correct"] * metrics["pair_count"]
            )
        if metrics["spearman"] is not None:
            correlations.append(metrics["spearman"])
        if metrics["correct_min_minus_incorrect_max"] is not None:
            margins.append(metrics["correct_min_minus_incorrect_max"])

    semantic_names = {
        "numeric_10",
        "negation",
        "independent_functions",
        "linear_equation",
        "factorial_program",
    }
    semantic = [case_reports[name]["metrics"] for name in semantic_names]
    numeric = [
        report["metrics"]
        for name, report in case_reports.items()
        if name.startswith("numeric_sweep_")
    ]

    def summarize_subset(metrics: Sequence[Mapping[str, float | int | None]]) -> dict:
        valid_correlations = [
            float(item["spearman"])
            for item in metrics
            if item["spearman"] is not None
        ]
        valid_margins = [
            float(item["correct_min_minus_incorrect_max"])
            for item in metrics
            if item["correct_min_minus_incorrect_max"] is not None
        ]
        pair_count = sum(int(item["pair_count"]) for item in metrics)
        pair_hits = sum(
            float(item["pairwise_correct"]) * int(item["pair_count"])
            for item in metrics
            if item["pairwise_correct"] is not None
        )
        return {
            "mean_spearman": sum(valid_correlations) / len(valid_correlations),
            "pairwise_ordering_accuracy": pair_hits / pair_count,
            "pair_count": pair_count,
            "mean_correct_margin": sum(valid_margins) / len(valid_margins),
            "positive_correct_margin_fraction": sum(value > 0.0 for value in valid_margins)
            / len(valid_margins),
        }

    micro_all_cases = {
        "mean_spearman": sum(correlations) / len(correlations),
        "pairwise_ordering_accuracy": weighted_concordance / total_pairs,
        "pair_count": total_pairs,
        "mean_correct_margin": sum(margins) / len(margins),
        "positive_correct_margin_fraction": sum(value > 0.0 for value in margins)
        / len(margins),
    }
    semantic_summary = summarize_subset(semantic)
    numeric_summary = summarize_subset(numeric)
    micro_all_cases["numeric_pair_fraction"] = (
        numeric_summary["pair_count"] / total_pairs
    )
    macro_keys = (
        "mean_spearman",
        "pairwise_ordering_accuracy",
        "mean_correct_margin",
        "positive_correct_margin_fraction",
    )
    family_macro = {
        key: 0.5 * (semantic_summary[key] + numeric_summary[key])
        for key in macro_keys
    }
    family_macro["family_count"] = 2

    return {
        "family_macro": family_macro,
        "semantic_code_math_summary": semantic_summary,
        "numeric_sweep_summary": numeric_summary,
        "micro_all_cases_numeric_dominated": micro_all_cases,
        "cases": case_reports,
    }


@torch.inference_mode()
def encode_texts(
    model: torch.nn.Module,
    tokenizer: object,
    texts: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict[str, Tensor]:
    embeddings: dict[str, Tensor] = {}
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        outputs = model(**inputs)
        pooled = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        for text, embedding in zip(batch, pooled):
            embeddings[text] = embedding.detach().float().cpu()
    return embeddings


def _all_texts(graded_cases: Sequence[GradedCase]) -> list[str]:
    return _unique_in_order(
        text
        for case in (*DEFAULT_PROBES, *graded_cases)
        for text in (
            case.target,
            *(candidate[1] if isinstance(candidate, tuple) else candidate.text for candidate in case.candidates),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 256, 1024, 4096])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--reward-temperature",
        type=float,
        default=DEFAULT_REWARD_TEMPERATURE,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cosine_kernel_reward(torch.ones(1), temperature=args.reward_temperature)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("pretrained-embedder probing is a model workload and requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    disable_optional_vision_imports()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, padding_side="left"
    )
    model = transformers.AutoModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="sdpa",
    )
    model.eval()

    graded_cases = build_graded_cases()
    plain_texts = _all_texts(graded_cases)
    instructed_texts = [
        detailed_instruction(args.instruction, text) for text in plain_texts
    ]
    plain = encode_texts(
        model,
        tokenizer,
        plain_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    instructed_raw = encode_texts(
        model,
        tokenizer,
        instructed_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    instructed = {
        text: instructed_raw[detailed_instruction(args.instruction, text)]
        for text in plain_texts
    }
    hidden_size = next(iter(plain.values())).shape[-1]
    dimensions = sorted({dimension for dimension in args.dimensions if dimension <= hidden_size})
    skipped_dimensions = sorted({dimension for dimension in args.dimensions if dimension > hidden_size})
    if not dimensions:
        raise ValueError(
            f"all requested dimensions exceed model hidden size {hidden_size}: {args.dimensions}"
        )

    modes = {
        "symmetric_plain": (plain, plain),
        "symmetric_instructed": (instructed, instructed),
        "retrieval_target_instructed": (plain, instructed),
    }
    mode_status = {
        "symmetric_plain": "diagnostic_uninstructed_symmetric",
        "symmetric_instructed": "diagnostic_off_contract_instruction_on_candidates",
        "retrieval_target_instructed": "official_retrieval_pattern",
    }
    reports: dict[str, dict[str, dict]] = {}
    for mode, (candidate_embeddings, target_embeddings) in modes.items():
        reports[mode] = {}
        for dimension in dimensions:
            scorer = CachedQwenScorer(
                candidate_embeddings,
                target_embeddings,
                dimension,
                mode,
                args.reward_temperature,
            )
            reports[mode][str(dimension)] = {
                "dimension": dimension_metadata(dimension, hidden_size),
                "behavioral": run_behavioral_probes(scorer),
                "graded": run_graded_audit(scorer, graded_cases),
            }

    result = {
        "schema": "qwen3_answer_reward_probe/v2",
        "model": args.model,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "instruction": args.instruction,
        "pooling": "last_nonpadding_token",
        "dtype": str(next(model.parameters()).dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "reward_mapping": "exp((cosine - 1) / temperature)",
        "reward_temperature": args.reward_temperature,
        "hidden_size": hidden_size,
        "dimensions": dimensions,
        "official_mrl_dimension_range": [OFFICIAL_MRL_MIN_DIMENSION, hidden_size],
        "skipped_dimensions": skipped_dimensions,
        "num_unique_texts": len(plain_texts),
        "mode_status": mode_status,
        "reports": reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
