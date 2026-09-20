from scripts.evaluate_minicpm_tasks import select_rows, summarize


def test_development_selection_accepts_arbitrary_domains_without_replacement():
    rows = [
        {"extra_info": {"domain": domain, "index": f"{domain}:{index}"}}
        for domain, count in (("sql", 5), ("scheduling", 2), ("text", 3))
        for index in range(count)
    ]
    selected = select_rows(rows, limit=3, seed=17)
    assert selected == select_rows(rows, limit=3, seed=17)
    assert len({r["extra_info"]["index"] for r in selected}) == 8
    assert sum(r["extra_info"]["domain"] == "scheduling" for r in selected) == 2
    assert all(row in rows for row in selected)


def test_pending_and_incomplete_groups_do_not_inflate_reward_signal():
    def attempt(domain, question, correct):
        return {
            "domain": domain,
            "question_id": question,
            "correct": correct,
            "tokens": 10,
            "truncated": False,
            "result": "pending"
            if correct is None
            else "pass"
            if correct
            else "mismatch",
        }

    attempts = [
        attempt("sql", "a", True),
        attempt("sql", "a", False),
        attempt("sql", "b", True),
        attempt("sql", "b", None),
        attempt("scheduling", "c", False),
        attempt("scheduling", "c", False),
    ]
    metrics = summarize(attempts, samples=2, domains=["sql", "scheduling", "text"])
    assert metrics["overall"]["accuracy"] == 2 / 5
    assert metrics["overall"]["pending_attempts"] == 1
    assert metrics["overall"]["complete_questions"] == 2
    assert metrics["domains"]["sql"]["mixed_questions"] == 1
    assert metrics["domains"]["sql"]["all_pass_questions"] == 0
    assert metrics["domains"]["scheduling"]["all_fail_questions"] == 1
    assert metrics["domains"]["text"]["accuracy"] is None


def test_repetition_is_reported_overall_and_per_domain_without_changing_correctness():
    from postraining.repetition import repetition_metrics

    attempts = [
        {
            "domain": domain, "question_id": domain, "correct": correct,
            "tokens": 30, "truncated": capped, "result": "pass" if correct else "mismatch",
            **repetition_metrics(text),
        }
        for domain, text, correct, capped in (
            ("sql", "A " * 20, True, True),
            ("text", "one two three", False, False),
        )
    ]
    metrics = summarize(attempts, samples=1, domains=["sql", "text"])
    assert metrics["overall"]["accuracy"] == 0.5
    assert metrics["overall"]["capped_attempts"] == 1
    assert metrics["overall"]["repetition_max_identical_word_run_mean"] == 10.5
    assert metrics["overall"]["repetition_16gram_fraction_max"] == 4 / 5
    assert metrics["domains"]["sql"]["repetition_3gram_fraction_mean"] == 17 / 18
    assert metrics["domains"]["text"]["repetition_3gram_fraction_max"] == 0
