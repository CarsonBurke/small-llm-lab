"""SFT trace training: packing, splits, optimizer partition, gate metrics."""

import math

import pytest
import torch
import torch.nn.functional as F

import postraining.sft_trace_train as sft_train
from postraining.nano_backbone import NanoGPTBackbone
from postraining.prepare_sft_traces import INSTRUCTION_SUFFIX
from postraining.sft_trace_train import (
    IGNORE_INDEX,
    TokenizedDocument,
    build_optimizers,
    create_fresh_run_dir,
    gate_metrics_from_counts,
    lr_scale_at,
    masked_ce_sum,
    pack_rows,
    split_holdout,
    tokenize_documents,
)


def test_sft_run_directory_is_immutable(tmp_path) -> None:
    run = tmp_path / "new_run"
    create_fresh_run_dir(run)
    assert run.is_dir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_fresh_run_dir(run)


def test_sampling_gate_uses_contract_counts_style_and_think_floor(monkeypatch):
    captured = {}

    class Wrapper:
        def __init__(self, backbone):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_evaluate(wrapper, tokenizer, rows, *args, **kwargs):
        captured["rows"] = rows
        captured["kwargs"] = kwargs
        return {
            "prompt_correct_counts": [8, 8],
            "contract_prompt_correct_counts": [0, 1],
            "accuracy": 1.0,
            "contract_accuracy": 1 / 16,
            "structural_format_fraction": 0.75,
            "ended_fraction": 0.875,
            "emitted_tokens_mean": 10.0,
            "emitted_tokens_p95": 12,
        }

    monkeypatch.setattr(sft_train, "LatentThoughtModel", Wrapper)
    monkeypatch.setattr(sft_train, "evaluate_latent_math", fake_evaluate)
    tokenizer = type(
        "Tokenizer",
        (),
        {
            "think_open_id": 1,
            "think_close_id": 2,
            "answer_open_id": 3,
            "answer_close_id": 4,
        },
    )()
    args = type(
        "Args",
        (),
        {
            "gate_prompts": 2,
            "gate_samples": 8,
            "gate_max_new_tokens": 32,
            "seed": 4,
            "gate_prompt_tokens": 64,
            "gate_think_min_tokens": 33,
        },
    )()
    panel = [
        {
            "problem": "deepmind",
            "final_answer": "x",
            "source": "a1_deepmind",
        },
        {"problem": "gsm", "final_answer": "3", "source": "had653_gold"},
    ]
    gate, _ = sft_train.run_sampling_gate(
        object(), tokenizer, panel, args, torch.device("cpu")
    )
    assert gate["accuracy"] == pytest.approx(1 / 16)
    assert gate["think_min_tokens"] == 33
    assert captured["kwargs"]["min_think_tokens"] == 33
    assert captured["rows"][0]["reward_model"]["style"] == "rule"
    assert (
        captured["rows"][1]["reward_model"]["style"]
        == "rule-lighteval/MATH_v2"
    )


class _Tokenizer:
    def bos_id(self) -> int:
        return 7

    def eos_id(self) -> int:
        return 7

    def encode(self, text: str) -> list[int]:
        return [10 + (byte % 40) for byte in text.encode("utf-8")]


def _document(problem: str, reasoning: str, final: str) -> dict:
    return {
        "problem": problem,
        "document": (
            f"{problem}{INSTRUCTION_SUFFIX}\n{reasoning}\nAnswer: {final}"
        ),
        "final_answer": final,
        "verified": True,
    }


def test_tokenize_documents_matches_rl_framing():
    tokenizer = _Tokenizer()
    doc = _document("What is 2+3?", "2+3 = 5.", "5")
    [tokenized] = tokenize_documents(tokenizer, [doc], seq_len=4096)
    prompt_text = doc["problem"] + INSTRUCTION_SUFFIX
    # Prompt framed exactly like an RL episode: [BOS] + standalone encode.
    expected_prompt = [7] + tokenizer.encode(prompt_text)
    assert list(tokenized.ids[: tokenized.prompt_length]) == expected_prompt
    # Completion encoded SEPARATELY (BPE is not concatenation-stable).
    assert list(tokenized.ids[tokenized.prompt_length:]) == tokenizer.encode(
        doc["document"][len(prompt_text):]
    )

    long_doc = _document("p", "x" * 40, "1")
    kept = tokenize_documents(tokenizer, [doc, long_doc], seq_len=100)
    assert len(kept) == 1  # the long document is dropped, not truncated
    with pytest.raises(ValueError, match="fit --seq-len"):
        tokenize_documents(tokenizer, [long_doc], seq_len=100)

    # The terse-trace filter drops by COMPLETION length, not document length.
    terse = _document("Long problem statement here?", "5", "5")
    verbose = _document("q?", "1+2 = 3 and 3+2 = 5.", "5")
    kept = tokenize_documents(
        tokenizer, [terse, verbose], seq_len=4096, min_completion_tokens=15
    )
    assert len(kept) == 1
    assert kept[0].prompt_length == 1 + len(
        tokenizer.encode("q?" + INSTRUCTION_SUFFIX)
    )


def test_pack_rows_targets_and_boundaries():
    doc_a = TokenizedDocument(2, (7, 1, 2, 3))
    doc_b = TokenizedDocument(1, (7, 4))
    [(tokens, targets)] = pack_rows(
        [doc_a, doc_b], seq_len=12, separator=7, rng=None
    )
    # Whole documents back to back, one terminal separator, separator pad.
    assert tokens == [7, 1, 2, 3, 7, 4, 7, 7, 7, 7, 7, 7]
    expected = [IGNORE_INDEX] * 12
    expected[1] = 2  # A: completion token
    expected[2] = 3  # A: completion token
    expected[3] = 7  # A: stop target = B's BOS
    expected[4] = 4  # B: completion token
    expected[5] = 7  # B: stop target = terminal separator
    assert targets == expected
    # Prompt positions (and padding) supervise nothing.
    assert targets[0] == IGNORE_INDEX
    assert all(target == IGNORE_INDEX for target in targets[6:])


def test_pack_rows_never_splits_documents():
    docs = [TokenizedDocument(1, tuple(range(7, 7 + 5))) for _ in range(3)]
    rows = pack_rows(docs, seq_len=11, separator=7, rng=None)
    # 5+5+1 fits in 11; the third document opens a second row.
    assert len(rows) == 2
    assert rows[0][0][:10] == list(docs[0].ids) + list(docs[1].ids)
    assert rows[1][0][:5] == list(docs[2].ids)
    for tokens, targets in rows:
        assert len(tokens) == 11
        assert len(targets) == 11


def test_split_holdout_groups_normalized_variants():
    variant_a = _document("A  farmer has 3 cows.", "3", "3")
    variant_b = _document("a farmer   has 3 cows.", "three is 3", "3")
    other = [
        _document(f"Problem number {index} text.", f"= {index}", str(index))
        for index in range(6)
    ]
    docs = [variant_a, variant_b, *other]
    train_docs, holdout_docs, panel = split_holdout(docs, holdout_problems=3)
    assert len(train_docs) + len(holdout_docs) == len(docs)
    assert len(panel) == 3
    # Whitespace/case variants of one problem never straddle the split.
    sides = {
        id(doc) in {id(d) for d in holdout_docs}
        for doc in (variant_a, variant_b)
    }
    assert len(sides) == 1
    # Deterministic across calls.
    again = split_holdout(docs, holdout_problems=3)
    assert [d["document"] for d in again[2]] == [
        d["document"] for d in panel
    ]
    with pytest.raises(ValueError, match="holdout-problems"):
        split_holdout(docs, holdout_problems=len(docs))


def test_lr_scale_warmup_and_decay():
    assert lr_scale_at(0, 100, 4) == pytest.approx(0.25)
    assert lr_scale_at(3, 100, 4) == pytest.approx(1.0)
    assert lr_scale_at(4, 100, 4) == pytest.approx(1.0)
    assert lr_scale_at(52, 100, 4) == pytest.approx(0.5)
    assert lr_scale_at(100, 100, 4) == pytest.approx(0.0)
    scales = [lr_scale_at(step, 50, 5) for step in range(50)]
    assert all(scale > 0 for scale in scales)


def test_gate_metrics_from_counts():
    gate = gate_metrics_from_counts([8, 0, 4], samples=8)
    assert gate["accuracy"] == pytest.approx(0.5)
    assert gate["mixed_prompt_fraction"] == pytest.approx(1 / 3)
    assert gate["within_group_reward_std_mean"] == pytest.approx(0.5 / 3)
    assert gate["prompts_all_correct"] == 1
    assert gate["prompts_all_wrong"] == 1
    with pytest.raises(ValueError, match="empty"):
        gate_metrics_from_counts([], samples=8)


def _nano_backbone() -> NanoGPTBackbone:
    torch.manual_seed(11)
    backbone = NanoGPTBackbone(
        vocab_size=64, num_layers=2, model_dim=32, mlp_hidden=64
    )
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    return backbone


def test_build_optimizers_exact_partition():
    from postraining.muon import Muon
    from postraining.vapo.config import POLAR_EXPRESS_STEP_COMPENSATION

    backbone = _nano_backbone()
    optimizers = build_optimizers(backbone, lr_scale=0.1)
    owned = {
        id(parameter)
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert owned == {id(parameter) for parameter in backbone.parameters()}
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            assert group["initial_lr"] == group["lr"]
            assert group["weight_decay"] == 0.0
    [muon] = [opt for opt in optimizers if isinstance(opt, Muon)]
    for group in muon.param_groups:
        # Compensated for the Polar Express step-size shrink, warm-start mu.
        assert group["lr"] == pytest.approx(
            0.025 * 0.1 * POLAR_EXPRESS_STEP_COMPENSATION
        )
        assert group["mu"] == pytest.approx(0.85)


def test_muon_momentum_warmup_matches_pretraining():
    from postraining.sft_trace_train import muon_momentum_at

    assert muon_momentum_at(0) == pytest.approx(0.85)
    assert muon_momentum_at(250) == pytest.approx(0.90)
    assert muon_momentum_at(500) == pytest.approx(0.95)
    assert muon_momentum_at(5000) == pytest.approx(0.95)


def test_think_tokens_land_in_padded_vocab_slack():
    from postraining.core import GPT2BPETokenizer
    from postraining.prepare_sft_traces import compose_document
    from postraining.sft_trace_train import register_special_tokens

    tokenizer = GPT2BPETokenizer(think_tokens=True)
    assert tokenizer.think_open_id == 50257
    assert tokenizer.think_close_id == 50258
    doc = compose_document("What is 1+1?", "1+1 = 2.", "2", think_tags=True)
    ids = tokenizer.encode(doc)
    # Each fence side is exactly ONE token, open before close.
    assert ids.count(tokenizer.think_open_id) == 1
    assert ids.count(tokenizer.think_close_id) == 1
    assert ids.index(tokenizer.think_open_id) < ids.index(
        tokenizer.think_close_id
    )
    # decode skips the fence, so answer parsing sees plain text.
    assert "<think>" not in tokenizer.decode(ids)
    # Untagged composition is byte-identical to the v1 corpus format.
    assert compose_document("p?", "r", "3", think_tags=False) == (
        "p?" + INSTRUCTION_SUFFIX + "\nr\nAnswer: 3"
    )

    padded = NanoGPTBackbone(
        vocab_size=50304, num_layers=1, model_dim=8, mlp_hidden=16
    )
    with torch.no_grad():
        padded.proj.weight.normal_(std=0.1)
    register_special_tokens(padded, tokenizer)
    assert padded.proj.weight[50257].abs().sum() == 0
    assert padded.proj.weight[50258].abs().sum() == 0
    assert padded.proj.weight[50256].abs().sum() > 0  # neighbors untouched
    assert padded.proj.weight[50259].abs().sum() > 0  # unregistered answer ids

    with_answers = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    register_special_tokens(padded, with_answers)
    assert padded.proj.weight[50259].abs().sum() == 0
    assert padded.proj.weight[50260].abs().sum() == 0

    small = _nano_backbone()  # vocab 64: think ids cannot fit
    with pytest.raises(ValueError, match="exceed the checkpoint vocab"):
        register_special_tokens(small, tokenizer)


def test_answer_fence_corpus_shape_is_measured_not_asserted():
    from postraining.core import GPT2BPETokenizer
    from postraining.prepare_sft_traces import (
        INSTRUCTION_SUFFIX,
        INSTRUCTION_SUFFIX_ANSWER,
        compose_document,
    )
    from postraining.sft_trace_train import (
        answer_fence_document_fraction,
        tokenize_documents,
    )

    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    problems = ["What is 1+1?", "What is 2+2?"]
    fenced_docs = [
        {
            "problem": problem,
            "document": compose_document(
                problem, "add them.", final, think_tags=True, answer_tags=True
            ),
        }
        for problem, final in zip(problems, ["2", "4"])
    ]
    fenced = tokenize_documents(
        tokenizer, fenced_docs, 512,
        instruction_suffix=INSTRUCTION_SUFFIX_ANSWER,
    )
    # The composed completion carries the exact anchored shape the RL
    # gate requires: <think> opens it, </answer> ends it.
    assert answer_fence_document_fraction(fenced, tokenizer) == 1.0
    for document in fenced:
        completion = document.ids[document.prompt_length:]
        assert completion[0] == tokenizer.think_open_id
        assert completion[-1] == tokenizer.answer_close_id

    # A think-only corpus measures 0.0 — the fraction is what stops an
    # --answer-fence run over a corpus that never trained the fence.
    think_docs = [
        {
            "problem": problem,
            "document": compose_document(
                problem, "add them.", final, think_tags=True
            ),
        }
        for problem, final in zip(problems, ["2", "4"])
    ]
    think_only = tokenize_documents(
        tokenizer, think_docs, 512, instruction_suffix=INSTRUCTION_SUFFIX
    )
    assert answer_fence_document_fraction(think_only, tokenizer) == 0.0

    with pytest.raises(ValueError, match="answer_tags requires think_tags"):
        compose_document("p", "r", "3", think_tags=False, answer_tags=True)


def test_instruction_suffix_byte_parity_with_rl_rewrite():
    """SFT and RL import one authoritative prompt contract."""
    from postraining.prepare_sft_traces import (
        INSTRUCTION_SUFFIX,
        INSTRUCTION_SUFFIX_ANSWER,
    )
    from postraining.math_prompt import (
        ANSWER_FIELD_INSTRUCTIONS,
        ANSWER_FENCE_SUFFIX,
    )

    assert INSTRUCTION_SUFFIX_ANSWER == ANSWER_FENCE_SUFFIX
    assert INSTRUCTION_SUFFIX == "\n\n" + ANSWER_FIELD_INSTRUCTIONS[1]


def test_think_span_percentiles_expose_floor_reachability():
    """A shape-perfect but terse corpus must be visible to the RL floor.

    The fence fraction validates shape only (finding D): these two
    corpora both measure 1.0, but their span lengths differ, and the
    recorded percentiles are what lets the RL trainer refuse a
    --think-min-tokens floor the corpus never taught.
    """
    from postraining.core import GPT2BPETokenizer
    from postraining.prepare_sft_traces import (
        INSTRUCTION_SUFFIX_ANSWER,
        compose_document,
    )
    from postraining.sft_trace_train import (
        answer_fence_document_fraction,
        think_span_token_percentiles,
        tokenize_documents,
    )

    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    terse_docs = [
        {
            "problem": f"What is {n}+{n}?",
            "document": compose_document(
                f"What is {n}+{n}?", "add.", str(2 * n),
                think_tags=True, answer_tags=True,
            ),
        }
        for n in range(4)
    ]
    terse = tokenize_documents(
        tokenizer, terse_docs, 512,
        instruction_suffix=INSTRUCTION_SUFFIX_ANSWER,
    )
    assert answer_fence_document_fraction(terse, tokenizer) == 1.0
    percentiles = think_span_token_percentiles(terse, tokenizer)
    # compose wraps the reasoning in newlines; the fence specials break
    # BPE merging at the span boundary, so the inner span tokenizes
    # exactly like the bare inner text.
    inner = len(tokenizer.encode("\nadd.\n"))
    assert percentiles == {"min": inner, "p1": inner, "p50": inner}

    long_docs = [
        {
            "problem": "What is 3+3?",
            "document": compose_document(
                "What is 3+3?", "add. " * 40, "6",
                think_tags=True, answer_tags=True,
            ),
        }
    ]
    longer = tokenize_documents(
        tokenizer, long_docs, 512,
        instruction_suffix=INSTRUCTION_SUFFIX_ANSWER,
    )
    long_percentiles = think_span_token_percentiles(longer, tokenizer)
    assert long_percentiles is not None
    assert long_percentiles["p50"] > percentiles["p50"]

    # No well-formed think span anywhere -> None, not a crash.
    class _NoSpan:
        prompt_length = 0
        ids = (1, 2, 3)

    assert think_span_token_percentiles([_NoSpan()], tokenizer) is None


def test_masked_ce_sum_ignores_masked_positions():
    backbone = _nano_backbone()
    inputs = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    targets[:, :3] = IGNORE_INDEX
    targets[1, 6:] = IGNORE_INDEX
    loss, supervised = masked_ce_sum(backbone, inputs, targets)
    assert supervised == int((targets != IGNORE_INDEX).sum())
    logits = backbone.policy_logits(inputs)
    expected = F.cross_entropy(
        logits.view(-1, 64),
        targets.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    assert math.isclose(float(loss), float(expected), rel_tol=1e-4)
