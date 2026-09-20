from collections import Counter

import pytest

from postraining.task_data import make_task, split_tasks, unique_tasks


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(messages[0]["content"].split()) + 4))


def source(index, *, domain="arbitrary", kind="text", target="Co"):
    return {
        "uuid": str(index),
        "query": f"Question {index}?",
        "ground_truth": target,
        "verification_kind": kind,
        "domain": domain,
        "source": "some-dataset",
        "source_revision": "pinned",
        "provenance": {"row": index},
    }


def test_full_prompt_boundary_never_truncates_context():
    original = source(1)
    original["query"] = "First evidence. " + "middle " * 30 + "Crucial last question?"
    task = make_task(original, Tokenizer(), prompt_tokens=100, context_tokens=200)
    size = task["extra_info"]["prompt_token_count"]
    assert (
        make_task(original, Tokenizer(), prompt_tokens=size - 1, context_tokens=200)
        is None
    )
    retained = make_task(original, Tokenizer(), prompt_tokens=size, context_tokens=200)
    assert retained["prompt"][0]["content"].startswith(original["query"] + "\n\n")
    assert retained["extra_info"]["prompt_token_cap"] == size
    with pytest.raises(ValueError):
        make_task(original, Tokenizer(), prompt_tokens=200, context_tokens=200)


def test_kind_is_explicit_and_domains_do_not_choose_grading():
    from postraining.verifiable_tasks import score_verifiable_response

    task = make_task(
        source(1, domain="Math", kind="text", target="Co"),
        Tokenizer(),
        prompt_tokens=100,
    )
    assert not score_verifiable_response(task, "Answer: CO")[0]
    numeric = make_task(
        source(2, domain="anything", kind="math", target="0.5"),
        Tokenizer(),
        prompt_tokens=100,
    )
    assert score_verifiable_response(numeric, r"Answer: \boxed{\frac{1}{2}}")[0]


def test_source_test_payload_cannot_override_the_verifier_kind():
    target = {
        "call_type": "std",
        "fn_name": None,
        "inputs": ["1"],
        "outputs": ["2"],
        "kind": "text",
    }
    with pytest.raises(ValueError, match="executable test contract"):
        make_task(
            source(1, kind="python_stdio", target=target),
            Tokenizer(),
            prompt_tokens=100,
        )


def test_uncapped_splits_retain_every_question_and_quarantine_conflicts():
    rows = [
        make_task(
            source(i, domain="large" if i < 15 else "small"),
            Tokenizer(),
            prompt_tokens=100,
        )
        for i in range(20)
    ]
    rows.append(rows[0].copy())
    conflict = make_task(
        source(3, domain="large", target="Ni"), Tokenizer(), prompt_tokens=100
    )
    counters = {domain: Counter() for domain in ("large", "small")}
    unique = unique_tasks(rows + [conflict], counters)
    train, dev = split_tasks(unique, seed=7, validation_fraction=0.2)
    train_ids = {r["extra_info"]["index"] for r in train}
    dev_ids = {r["extra_info"]["index"] for r in dev}
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids | dev_ids == {str(i) for i in range(20) if i != 3}
    assert counters["large"]["conflicting_prompt_quarantine"] == 2
    assert counters["large"]["duplicate_prompt"] == 1
    assert Counter(r["extra_info"]["domain"] for r in train) == {
        "large": 11,
        "small": 4,
    }
    assert split_tasks(unique, seed=7, validation_fraction=0.2) == (train, dev)
