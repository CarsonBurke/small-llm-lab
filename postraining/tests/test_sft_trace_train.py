"""SFT trace training: packing, splits, optimizer partition, gate metrics."""

import random
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import postraining.sft_trace_train as sft_train
from postraining.nano_backbone import NanoGPTBackbone
from postraining.prepare_sft_traces import INSTRUCTION_SUFFIX
from postraining.sft_trace_train import (
    IGNORE_INDEX,
    SupervisedCE,
    TokenizedDocument,
    accumulate_step_gradients,
    build_optimizers,
    create_fresh_run_dir,
    gate_metrics_from_counts,
    device_batch,
    load_resume_contract,
    load_resume_state,
    lr_scale_at,
    pack_rows,
    packed_schedule_sha256,
    resume_contract,
    resume_contract_mismatch,
    resume_source_sha256,
    save_resume_state,
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
            # The gate now measures panel prompt lengths before sampling, so
            # the stub has to tokenize like the real one.
            "bos_id": lambda self: 7,
            "eos_id": lambda self: 7,
            "encode": lambda self, text: [
                10 + (byte % 40) for byte in text.encode("utf-8")
            ],
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
            # The stub tokenizer emits one token per byte, so the instruction
            # suffix alone costs ~70; this is about style and the think floor,
            # not about prompt budgets.
            "gate_prompt_tokens": 256,
            "gate_batch_trajectories": 128,
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


class _BatchTokenizer(_Tokenizer):
    def encode(self, text: str) -> list[int]:
        raise AssertionError("the batch path must not encode one text")

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [_Tokenizer.encode(self, text) for text in texts]


def test_tokenize_documents_batched_encoding_is_identical():
    documents = [
        _document(f"What is {n}+{n}?", f"{n}+{n} = {2 * n}.", str(2 * n))
        for n in range(9000)  # spans more than one encode chunk
    ]
    serial = tokenize_documents(_Tokenizer(), documents, seq_len=4096)
    batched = tokenize_documents(_BatchTokenizer(), documents, seq_len=4096)
    assert len(serial) == len(batched) == len(documents)
    for left, right in zip(serial, batched, strict=True):
        assert left.prompt_length == right.prompt_length
        assert np.array_equal(left.ids, right.ids)


def test_gpt2_encode_batch_matches_encode():
    from postraining.core import GPT2BPETokenizer

    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    texts = [
        "What is 12 x 7?",
        "<think>\n12 x 7 = 84.\n</think><answer>84</answer>",
        " leading space and  double  spaces\n\n",
        "",
    ]
    assert tokenizer.encode_batch(texts) == [
        tokenizer.encode(text) for text in texts
    ]


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


def _doc(prompt_length, ids):
    """TokenizedDocument.ids is an int32 array, not a tuple of Python ints."""
    return TokenizedDocument(prompt_length, np.asarray(ids, dtype=np.int32))


def test_pack_rows_targets_and_boundaries():
    doc_a = _doc(2, (7, 1, 2, 3))
    doc_b = _doc(1, (7, 4))
    [(tokens, targets)] = pack_rows(
        [doc_a, doc_b], seq_len=12, separator=7, rng=None
    )
    # Whole documents back to back, one terminal separator, separator pad.
    assert tokens.tolist() == [7, 1, 2, 3, 7, 4, 7, 7, 7, 7, 7, 7]
    expected = [IGNORE_INDEX] * 12
    expected[1] = 2  # A: completion token
    expected[2] = 3  # A: completion token
    expected[3] = 7  # A: stop target = B's BOS
    expected[4] = 4  # B: completion token
    expected[5] = 7  # B: stop target = terminal separator
    assert targets.tolist() == expected
    # Prompt positions (and padding) supervise nothing.
    assert targets[0] == IGNORE_INDEX
    assert all(target == IGNORE_INDEX for target in targets[6:])


def test_pack_rows_never_splits_documents():
    docs = [_doc(1, range(7, 7 + 5)) for _ in range(3)]
    rows = pack_rows(docs, seq_len=11, separator=7, rng=None)
    # 5+5+1 fits in 11; the third document opens a second row.
    assert len(rows) == 2
    assert rows[0][0][:10].tolist() == (
        docs[0].ids.tolist() + docs[1].ids.tolist()
    )
    assert rows[1][0][:5].tolist() == docs[2].ids.tolist()
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


def test_answer_fenced_sft_manifest_binds_current_schema_and_bytes(
    tmp_path,
) -> None:
    import hashlib
    import json

    from postraining.math_prompt import ANSWER_FENCE_PROMPT_SCHEMA
    from postraining.sft_trace_train import validate_trace_manifest

    traces = tmp_path / "traces.parquet"
    traces.write_bytes(b"immutable corpus")
    manifest = traces.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
                "output_sha256": hashlib.sha256(traces.read_bytes()).hexdigest(),
            }
        )
    )
    assert validate_trace_manifest(traces, answer_fence=True) == manifest
    assert validate_trace_manifest(traces, answer_fence=False) is None

    manifest.write_text(
        json.dumps(
            {
                "answer_fence_prompt_schema": "legacy/v1",
                "output_sha256": hashlib.sha256(traces.read_bytes()).hexdigest(),
            }
        )
    )
    with pytest.raises(ValueError, match="silently change"):
        validate_trace_manifest(traces, answer_fence=True)

    manifest.write_text(
        json.dumps(
            {
                "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
                "output_sha256": "0" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_trace_manifest(traces, answer_fence=True)


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


def _packed_rows(count: int, seq_len: int = 8, vocab: int = 64, seed: int = 0):
    generator = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        tokens = generator.integers(0, vocab, seq_len, dtype=np.int32)
        targets = generator.integers(0, vocab, seq_len, dtype=np.int32)
        targets[generator.random(seq_len) < 0.4] = IGNORE_INDEX
        rows.append((tokens, targets))
    return rows


def test_supervised_ce_matches_full_sequence_masked_ce():
    """Host-selected positions must price exactly the masked full logits."""
    backbone = _nano_backbone()
    rows = _packed_rows(3)
    batch = device_batch(rows, torch.device("cpu"))
    targets = torch.from_numpy(np.stack([t for _, t in rows])).long()
    assert batch.supervised == int((targets != IGNORE_INDEX).sum())
    assert torch.equal(batch.labels, targets.view(-1)[batch.positions])
    loss = SupervisedCE(backbone, compiled=False)(batch)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = backbone.policy_logits(batch.inputs)
    expected = F.cross_entropy(
        logits.float().view(-1, 64),
        targets.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    assert math.isclose(float(loss), float(expected), rel_tol=1e-4)


def test_step_gradients_do_not_depend_on_the_micro_batch_split():
    """The step-wide normalization makes micro-batching a pure memory knob."""
    rows = _packed_rows(4, seed=3)
    results = []
    for micro in (1, 2, 4):
        backbone = _nano_backbone()
        loss = accumulate_step_gradients(
            SupervisedCE(backbone, compiled=False),
            rows,
            micro,
            torch.device("cpu"),
        )
        results.append(
            (
                float(loss),
                [p.grad.clone() for p in backbone.parameters()],
            )
        )
    # Exact in fp32 (measured 1e-7 relative); the forward runs under bf16
    # autocast, where a different row composition changes matmul rounding
    # (measured 2.3e-3), so compare against the gradient norm rather than
    # elementwise.
    reference_loss, reference_grads = results[0]
    reference = torch.cat([grad.flatten() for grad in reference_grads])
    for loss, grads in results[1:]:
        assert math.isclose(loss, reference_loss, rel_tol=1e-3)
        flat = torch.cat([grad.flatten() for grad in grads])
        assert float((flat - reference).norm() / reference.norm()) < 1e-2


def test_save_backbone_checkpoint_records_the_trained_window(tmp_path):
    """The payload must carry both contexts, and must not read a global.

    ``save_backbone_checkpoint`` once referenced ``args.seq_len``, which
    resolves as a module global that does not exist: every run raised
    NameError AFTER the whole training pass and before the checkpoint was
    written. No test exercised this function, which is why that survived.
    ``train_seq_len`` must stay the PRETRAINED bound (downstream loaders and
    the RL budget derivation read it) while ``trained_seq_len`` records the
    window this pass actually trained.
    """

    class Stub:
        architecture = "nanogpt_mini_gpt2vocab_kda_kkkdkkkd_mixers_v3"
        model_config = {"n_layer": 1}
        train_context_tokens = 1024

        def state_dict(self):
            return {"tok_emb.weight": torch.zeros(2, 3)}

    path = tmp_path / "sft_final_model.pt"
    sft_train.save_backbone_checkpoint(
        Stub(), path, {"schema": "test"}, trained_seq_len=5120
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["train_seq_len"] == 1024
    assert payload["trained_seq_len"] == 5120
    assert payload["context_extension"] is True

    # An unextended run must report no extension rather than omitting it.
    same = tmp_path / "same.pt"
    sft_train.save_backbone_checkpoint(
        Stub(), same, {"schema": "test"}, trained_seq_len=1024
    )
    assert torch.load(same, map_location="cpu", weights_only=False)[
        "context_extension"
    ] is False


def _pack_rows_reference(documents, seq_len, separator, rng):
    """The original Python-list packer, kept only as a differential oracle."""
    order = list(range(len(documents)))
    if rng is not None:
        rng.shuffle(order)
    rows, tokens, spans = [], [], []

    def close_row():
        tokens.append(separator)
        targets = [IGNORE_INDEX] * seq_len
        for offset, prompt_length, length in spans:
            for position in range(offset + prompt_length - 1, offset + length):
                targets[position] = tokens[position + 1]
        tokens.extend([separator] * (seq_len - len(tokens)))
        rows.append((list(tokens), targets))
        tokens.clear()
        spans.clear()

    for index in order:
        document = documents[index]
        ids = list(document.ids)
        if len(tokens) + len(ids) + 1 > seq_len:
            close_row()
        spans.append((len(tokens), document.prompt_length, len(ids)))
        tokens.extend(ids)
    if tokens:
        close_row()
    return rows


def test_pack_rows_matches_the_list_implementation_exactly():
    """The int32 packer must be content-identical to the list version.

    ``pack_rows`` was rewritten from Python int lists to int32 arrays purely
    for memory: at --seq-len 5120 a ~900k-document corpus packs into tens of
    thousands of rows, and two 5120-element int lists per row cost ~25 GB of
    interpreter objects against ~2.8 GB as int32. A packing change would
    silently alter what is supervised, so the old implementation is retained
    here as an oracle over randomized shapes, including shuffled orders and
    documents that exactly fill or overflow a row.
    """
    master = random.Random(0)
    for trial in range(200):
        seq_len = master.randint(8, 64)
        documents = []
        for _ in range(master.randint(1, 25)):
            length = master.randint(2, max(2, seq_len - 1))
            documents.append(
                _doc(
                    master.randint(1, length),
                    [7] + [master.randint(0, 50000) for _ in range(length - 1)],
                )
            )
        shuffled = master.random() < 0.5
        actual = pack_rows(
            documents, seq_len, 7, random.Random(trial) if shuffled else None
        )
        expected = _pack_rows_reference(
            documents, seq_len, 7, random.Random(trial) if shuffled else None
        )
        assert len(actual) == len(expected)
        for (tokens, targets), (want_tokens, want_targets) in zip(
            actual, expected
        ):
            assert tokens.tolist() == want_tokens
            assert targets.tolist() == want_targets


def _optimizer_step(backbone, optimizers, rows) -> None:
    accumulate_step_gradients(
        SupervisedCE(backbone, compiled=False), rows, 2, torch.device("cpu")
    )
    for optimizer in optimizers:
        optimizer.step()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def test_resume_state_continues_training_exactly(tmp_path):
    """Two steps, save, reload into fresh objects, two more steps == four.

    Both optimizers carry state (AdamW moments, Muon momentum) that the
    next update depends on, so it must round-trip for the resumed run to be
    the uninterrupted one. The global RNG is restored too, as a guard for
    any future stochastic training.
    """
    steps = [_packed_rows(2, seed=10 + index) for index in range(4)]
    contract = {"resume_schema": "test", "args": {"seed": 1}}

    torch.manual_seed(5)
    straight = _nano_backbone()
    straight_optimizers = build_optimizers(straight, 0.1)
    for rows in steps:
        _optimizer_step(straight, straight_optimizers, rows)
    straight_draw = torch.rand(4)

    torch.manual_seed(5)
    first = _nano_backbone()
    first_optimizers = build_optimizers(first, 0.1)
    for rows in steps[:2]:
        _optimizer_step(first, first_optimizers, rows)
    path = tmp_path / "resume.pt"
    progress = {"elapsed_seconds": 1.5, "resumed_at_steps": []}
    save_resume_state(
        path,
        contract=contract,
        backbone=first,
        optimizers=first_optimizers,
        step=2,
        progress=progress,
    )
    torch.manual_seed(999)  # the resumed process starts from other RNG state

    resumed = _nano_backbone()
    resumed_optimizers = build_optimizers(resumed, 0.1)
    step, restored = load_resume_state(
        path,
        contract=contract,
        backbone=resumed,
        optimizers=resumed_optimizers,
    )
    assert (step, restored) == (2, progress)
    for rows in steps[2:]:
        _optimizer_step(resumed, resumed_optimizers, rows)

    for (name, expected), actual in zip(
        straight.state_dict().items(), resumed.state_dict().values()
    ):
        assert torch.equal(expected, actual), name
    assert torch.equal(torch.rand(4), straight_draw)


def test_resume_refuses_a_changed_contract_and_names_the_change(tmp_path):
    backbone = _nano_backbone()
    optimizers = build_optimizers(backbone, 0.1)
    path = tmp_path / "resume.pt"
    saved = {"traces_sha256": "a", "args": {"lr_scale": 0.1, "seed": 1}}
    save_resume_state(
        path,
        contract=saved,
        backbone=backbone,
        optimizers=optimizers,
        step=1,
        progress={},
    )
    for changed, message in (
        ({**saved, "traces_sha256": "b"}, "traces_sha256"),
        ({**saved, "args": {"lr_scale": 0.2, "seed": 1}}, "args.lr_scale"),
    ):
        with pytest.raises(ValueError, match=message):
            load_resume_state(
                path,
                contract=changed,
                backbone=backbone,
                optimizers=optimizers,
            )


def test_resume_contract_exempts_only_the_checkpoint_cadence():
    import argparse

    base = dict(
        lr_scale=0.1, seed=1, resume=False, checkpoint_interval_seconds=300.0
    )
    first = resume_contract(argparse.Namespace(**base), traces_sha256="a")
    resumed = resume_contract(
        argparse.Namespace(
            **{**base, "resume": True, "checkpoint_interval_seconds": 60.0}
        ),
        traces_sha256="a",
    )
    assert first == resumed
    assert first != resume_contract(
        argparse.Namespace(**{**base, "seed": 2}), traces_sha256="a"
    )


def test_packed_schedule_digest_sees_order_and_targets():
    rows = _packed_rows(3, seed=4)
    digest = packed_schedule_sha256([rows])
    assert digest == packed_schedule_sha256([[(t.copy(), g.copy()) for t, g in rows]])
    assert digest != packed_schedule_sha256([rows[::-1]])
    tokens, targets = rows[0]
    altered = targets.copy()
    altered[np.flatnonzero(altered != IGNORE_INDEX)[0]] = IGNORE_INDEX
    assert digest != packed_schedule_sha256([[(tokens, altered), *rows[1:]]])
    # Epoch boundaries are part of the schedule, not just the row stream.
    assert digest != packed_schedule_sha256([rows[:1], rows[1:]])


def test_resume_contract_reads_without_loading_the_state(tmp_path):
    backbone = _nano_backbone()
    path = tmp_path / "resume.pt"
    contract = {"args": {"seed": 1}, "source_sha256": {"m": "x"}}
    save_resume_state(
        path,
        contract=contract,
        backbone=backbone,
        optimizers=build_optimizers(backbone, 0.1),
        step=3,
        progress={},
    )
    assert load_resume_contract(path) == contract
    assert not path.with_name("resume.pt.tmp").exists()


def test_resume_contract_mismatch_names_nested_fields():
    saved = {"args": {"seed": 1, "lr": 2}, "source_sha256": {"a": "x"}, "t": 1}
    current = {"args": {"seed": 1, "lr": 3}, "source_sha256": {"a": "y"}, "t": 1}
    assert resume_contract_mismatch(saved, current) == [
        "args.lr", "source_sha256.a",
    ]
    assert resume_contract_mismatch(saved, saved) == []


def test_resume_binds_every_training_path_module():
    hashes = resume_source_sha256()
    assert set(hashes) == set(sft_train.RESUME_BOUND_MODULES)
    assert all(len(digest) == 64 for digest in hashes.values())
