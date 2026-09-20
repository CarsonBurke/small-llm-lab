from __future__ import annotations

from collections import Counter
import io
import json
import subprocess
import sys
from types import SimpleNamespace
from urllib.request import Request

import pytest
from postraining.data_acquisition import (
    AcquisitionError,
    BoundedHTTP,
    NoBodyRedirect,
    canonical_json,
)

from postraining.ultradata_data import (
    REVISION,
    Window,
    complete_records,
    plan_windows,
    rejection_reason,
    typed_ultradata_source,
)
from postraining.task_data import make_task


class Response(io.BytesIO):
    def __init__(self, payload, *, status=206, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {}
        self.body_bytes_read = 0

    def read(self, size=-1):
        result = super().read(size)
        self.body_bytes_read += len(result)
        return result


class Tokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        assert tokenize and add_generation_prompt and enable_thinking
        # Distinct chat/thinking overhead makes counting just query tokens fail.
        return list(range(len(messages[0]["content"]) + 11))


def source(domain="Math", query="Compute 6 times 7.", answer="42", identity="example"):
    return {
        "uuid": identity,
        "domain": domain,
        "query": query,
        "ground_truth": answer,
        "source": "UltraData-RL-2609",
    }


def task(item):
    adapted = typed_ultradata_source(item, revision=REVISION, provenance={})
    return make_task(adapted, Tokenizer(), prompt_tokens=4096)


def test_numeric_stem_targets_accept_equivalence_without_folding_entity_case():
    from postraining.verifiable_tasks import score_verifiable_response

    numeric = task(source("Knowledge", "What is the probability?", "0.5"))
    assert score_verifiable_response(numeric, r"Answer: \boxed{\frac{1}{2}}")[0]
    entity = task(source("Knowledge", "What is the chemical symbol?", "Co"))
    assert not score_verifiable_response(entity, r"Answer: \boxed{CO}")[0]


def test_redirect_closes_body_without_urllib_draining_it():
    response = Response(b"x" * 1000, status=302)
    request = Request("https://example.com/source")
    request.timeout = 10
    redirected = []
    handler = NoBodyRedirect()
    handler.add_parent(
        SimpleNamespace(open=lambda req, timeout: redirected.append(req.full_url))
    )
    handler.http_error_302(
        request, response, 302, "Found", {"location": "https://example.com/target"}
    )
    assert response.closed
    assert response.body_bytes_read == 0
    assert redirected == ["https://example.com/target"]


def test_ignored_range_never_reads_unbounded_body_and_budget_survives_retry(tmp_path):
    response = Response(b"x" * 1000, status=200, headers={"Content-Length": "1000"})
    calls = []

    def opener(request, timeout):
        calls.append(request)
        return response

    http = BoundedHTTP(tmp_path, cap=16, opener=opener)
    try:
        with pytest.raises(AcquisitionError, match="Range"):
            http.fetch("https://example.com/shard", start=100, limit=16, total=1000)
        assert response.body_bytes_read == 0
        assert response.closed
    finally:
        http.close()
    reopened = BoundedHTTP(tmp_path, cap=16, opener=opener)
    try:
        with pytest.raises(AcquisitionError, match="cap exhausted"):
            reopened.fetch("https://example.com/shard", start=100, limit=16, total=1000)
        assert len(calls) == 1
    finally:
        reopened.close()


def test_metadata_without_length_cannot_read_beyond_global_cap(tmp_path):
    response = Response(b"x" * 1000, status=200)
    http = BoundedHTTP(tmp_path, cap=17, opener=lambda request, timeout: response)
    try:
        with pytest.raises(AcquisitionError, match="metadata.*cap"):
            http.fetch("https://example.com/tree", limit=17)
        assert response.body_bytes_read == 17
        assert response.closed
        assert http.ledger["charged_bytes"] == 17
    finally:
        http.close()


def test_range_receipt_cache_reuse_and_mismatch_rejection(tmp_path):
    responses = [Response(b"abc\n", headers={"Content-Range": "bytes 4-7/12"})]
    calls = []

    def opener(request, timeout):
        calls.append(request.headers["Range"])
        return responses.pop()

    http = BoundedHTTP(tmp_path, cap=20, opener=opener)
    try:
        first, receipt = http.fetch(
            "https://example.com/shard", start=4, limit=4, total=12
        )
        again, cached = http.fetch(
            "https://example.com/shard", start=4, limit=4, total=12
        )
        assert first == again == b"abc\n"
        assert receipt["sha256"] == cached["sha256"]
        assert cached["cache_hit"]
        assert calls == ["bytes=4-7"]
        assert http.ledger["charged_bytes"] == 4
        responses.append(Response(b"abcd", headers={"Content-Range": "bytes 0-3/12"}))
        with pytest.raises(AcquisitionError, match="Range"):
            http.fetch("https://example.com/other", start=4, limit=4, total=12)
    finally:
        http.close()


def test_complete_records_drop_both_fragments_and_oversized_lines():
    payload = b'partial\n{"ok":1}\n' + b'"' + b"x" * 50 + b'"\n' + b'{"cut":'
    window = Window("shard", "Code", 1000, 20, len(payload), 0, 1000)
    counts = Counter()
    records = list(complete_records(payload, window, counts, record_cap=20))
    assert [(record, offset) for record, offset, _ in records] == [({"ok": 1}, 28)]
    assert counts["oversized_records"] == 1
    assert counts["leading_boundary_bytes"] == 8
    assert counts["trailing_boundary_bytes"] == 7
    final = b'{"last":2}'
    assert list(
        complete_records(
            final,
            Window("s", "Math", len(final), 0, len(final), 0, len(final)),
            Counter(),
        )
    )[0][0] == {"last": 2}


def test_seeded_window_plan_covers_every_large_shard_without_overlap():
    shards = [
        {"path": f"data/Code/{index}.jsonl", "domain": "Code", "size": 100 * 1024**2}
        for index in range(12)
    ]
    planned = plan_windows(shards, seed=19, byte_budget=100 * 1024**2)
    assert planned == plan_windows(shards, seed=19, byte_budget=100 * 1024**2)
    assert planned != plan_windows(shards, seed=20, byte_budget=100 * 1024**2)
    assert sum(window.length for window in planned) <= 100 * 1024**2
    for shard in shards:
        windows = [window for window in planned if window.path == shard["path"]]
        assert len(windows) >= 2
        assert all(a.start + a.length <= b.start for a, b in zip(windows, windows[1:]))
        assert windows[0].stratum_start == 0
        assert windows[-1].stratum_end == shard["size"]


def test_extended_acquisition_reuses_random_windows_within_lifetime_cap():
    shards = [
        {"path": f"data/{domain}/{i}.jsonl", "domain": domain, "size": 10 * 1024**3}
        for domain, count in (("Code", 12), ("Long_Context", 2))
        for i in range(count)
    ]
    initial = plan_windows(shards, seed=1337, byte_budget=254 * 1024**2)
    expanded = plan_windows(
        shards, seed=1337, byte_budget=1022 * 1024**2, extended_prefixes=True
    )
    assert set(initial) <= set(expanded)
    assert sum(window.length for window in expanded) < 1022 * 1024**2
    for domain, end in (("Code", 16 * 1024**2), ("Long_Context", 64 * 1024**2)):
        extensions = [w for w in expanded if w.domain == domain and w not in initial]
        assert extensions
        assert all(w.start + w.length == end for w in extensions)


def test_code_preserves_every_test_and_rejects_instead_of_cutting():
    tests = {"inputs": ["", "1\n", "2\n"], "outputs": ["zero\n", "one\n", "two\n"]}
    item = source("Code", "Read stdin and print its English name.", tests)
    assert (
        rejection_reason(item, "Code", [], test_cap=len(canonical_json(tests))) is None
    )
    assert (
        rejection_reason(item, "Code", [], test_cap=len(canonical_json(tests)) - 1)
        == "oversized_code_tests"
    )
    from postraining.verifiable_tasks import score_verifiable_response

    retained = task(item)
    program = 'import sys\nprint({"": "zero", "1": "one", "2": "two"}[sys.stdin.read().strip()])'
    assert score_verifiable_response(retained, f"```python\n{program}\n```")[0]
    wrong_last_case = program.replace('"two"', '"wrong"')
    assert not score_verifiable_response(
        retained, f"```python\n{wrong_last_case}\n```"
    )[0]
    assert "zero" not in retained["prompt"][0]["content"]
    invalid = source("Code", item["query"], {"inputs": ["1"], "outputs": []})
    assert rejection_reason(invalid, "Code", []) == "invalid_code_tests"


def test_source_quality_excludes_dubious_mc_proof_and_case_preserving_reviews():
    assert (
        rejection_reason(
            source(
                "Knowledge", "Question?\na: first\nb: second", "0", "Knowledge_00001"
            ),
            "Knowledge",
            [],
        )
        == "unverifiable_question_structure"
    )
    assert (
        rejection_reason(
            source("Knowledge", "Which answer?\nA. first\nB. second", "first"),
            "Knowledge",
            [],
        )
        == "unverifiable_question_structure"
    )
    assert (
        rejection_reason(source("Math", "Prove that n is positive."), "Math", [])
        == "unverifiable_question_structure"
    )
    reviews = [("Compute N + n.", {"action": "correct", "corrected_target": "7"})]
    assert (
        rejection_reason(
            source("Math", "New framing. Compute N + n. End framing.", "8"),
            "Math",
            reviews,
        )
        == "reviewed_math_quarantine"
    )
    assert (
        rejection_reason(source("Math", "Compute N + n.", "7"), "Math", reviews) is None
    )
    assert (
        rejection_reason(source("Math", "Compute N + N.", "8"), "Math", reviews) is None
    )


def test_qa_document_structure_is_not_mistaken_for_multiple_choice_question():
    query = "Article:\nA. first section\nB. second section\nPeople prove claims.\n\nQuestion: Who won the title?"
    assert (
        rejection_reason(
            source("Long_Context", query, "Ada Lovelace"), "Long_Context", []
        )
        is None
    )
    question = query + "\nA. Ada\nB. Grace"
    assert (
        rejection_reason(
            source("Long_Context", question, "Ada Lovelace"), "Long_Context", []
        )
        == "unverifiable_question_structure"
    )


def test_open_ended_knowledge_prose_is_not_exact_match_ground_truth():
    item = source(
        "Knowledge",
        "How should the company achieve sustainability?",
        "Developing a comprehensive sustainability strategy that includes reducing carbon footprint and improving labor conditions across the supply chain",
    )
    assert (
        rejection_reason(item, "Knowledge", [])
        == "free_form_target_requires_semantic_judge"
    )
    assert (
        rejection_reason(
            source("Knowledge", "Which doctrine applies?", "Doctrine of equivalents"),
            "Knowledge",
            [],
        )
        is None
    )


def test_source_structure_filters_handle_large_whitespace_without_hanging():
    # A real QA record contains 10,086 consecutive spaces. Isolate the deadline
    # so reintroducing regex backtracking fails instead of hanging the suite.
    rows = [
        source("Long_Context", "Context." + " " * 10_086 + "Identify its value."),
        source("Long_Context", "Context." + " \n" * 32_000 + "Identify its value."),
        source("Long_Context", "Choose  A. first  B. second"),
        source("Long_Context", "Compute:\n(1) the first term\n (2) the second term"),
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys\n"
            "from postraining.ultradata_data import rejection_reason\n"
            "print(json.dumps([rejection_reason(r,r['domain'],[]) for r in json.load(sys.stdin)]))",
        ],
        input=canonical_json(rows),
        capture_output=True,
        check=True,
        timeout=20,
    )
    assert json.loads(result.stdout) == [
        None,
        None,
        "unverifiable_question_structure",
        "unverifiable_question_structure",
    ]


def test_hyphen_choices_are_rejected_without_banning_units_or_subtraction():
    for label_spacing in ("", " "):
        choices = "\n".join(
            f"{letter}{label_spacing}- proposed method {index}"
            for index, letter in enumerate("ABCDEFGHIJ")
        )
        item = source("Knowledge", "Choose a suitable method.\n" + choices, "J")
        assert (
            rejection_reason(item, "Knowledge", []) == "unverifiable_question_structure"
        )
    assert (
        rejection_reason(
            source("Knowledge", "What is the SI unit symbol for energy?", "J"),
            "Knowledge",
            [],
        )
        is None
    )
    assert (
        rejection_reason(
            source("Math", "Given\na-b=2\nb-c=3\nFind a-c.", "5"), "Math", []
        )
        is None
    )


def test_inline_and_tex_answer_choice_lists_are_rejected():
    questions = [
        "Compute a value.\n(A) 1 (B) 2 (C) 3 (D) 4 (E) 5",
        r"Compute a value. $\mathrm{(A) 1} \mathrm{(B) 2} \mathrm{(C) 3}$",
        r"Compute a value. $\textbf {(A)} 1 \textbf {(B)} 2 \textbf {(C)} 3$",
        "Compute a value.\n- A) 1\n- B) 2\n- C) 3\n- D) 4",
        "Option A costs 1; Option B costs 2. Which option is cheaper?",
    ]
    for question in questions:
        assert (
            rejection_reason(source("Math", question, "3"), "Math", [])
            == "unverifiable_question_structure"
        )


def test_parenthesized_variables_and_named_premises_are_not_choices():
    questions = [
        "Let f(A)=1, f(B)=2, f(C)=3, f(D)=4, f(E)=5. Find their sum.",
        "Count two-digit n satisfying exactly two facts: (A) n is odd; (B) n is not divisible by 3; (C) n is divisible by 5.",
        "Given this diagram code: dot (A); dot (B); dot (C); dot (D); dot (E); determine the count.",
        "Option A costs 1; Option B costs 2. Compute the difference in costs.",
    ]
    for question in questions:
        assert rejection_reason(source("Math", question, "15"), "Math", []) is None


def test_proof_request_variants_cannot_be_scored_by_repeating_the_claim():
    for question in (
        "Verify the identity x-x=0.",
        "Verify that this set is a group.",
        "Prove using the limit definition that the answer is zero.",
        "Demonstrate that the sequence converges.",
    ):
        assert (
            rejection_reason(source("Math", question, "0"), "Math", [])
            == "unverifiable_question_structure"
        )
    assert (
        rejection_reason(
            source(
                "Knowledge", "Which theorem did Wiles prove?", "Fermat's Last Theorem"
            ),
            "Knowledge",
            [],
        )
        is None
    )
