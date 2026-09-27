from scripts.audit_math_source_overlap import cosine_candidates, numeric_skeleton


def test_numeric_skeleton_is_retrieval_only_and_preserves_operations():
    assert numeric_skeleton("Compute x + 100.") == numeric_skeleton("Compute x + 200.")
    assert numeric_skeleton("Compute x - 100.") != numeric_skeleton("Compute x + 100.")


def test_broad_retrieval_returns_exact_neighbor_without_claiming_numeric_identity():
    question = "Compute the greatest prime factor of 731."
    distractors = [f"Unrelated topic number {i * 9871}." for i in range(40)]
    forward = cosine_candidates([question], distractors + [question], threshold=0.99)
    backward = cosine_candidates(distractors + [question], [question], threshold=0.99)
    assert set(forward) == {(0, 40)}
    assert set(backward) == {(40, 0)}
