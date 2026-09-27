from __future__ import annotations

import pytest

from postraining.problem_overlap import (
    ProblemOverlapIndex,
    cross_matches,
    numbers_contained,
    number_multiset,
    overlap_tokens,
    shingles,
    skeleton_key,
    template_shingles,
    whitespace_key,
)

PAIRS = (
    "Compute the number of ordered pairs of integers $(x, y)$ with "
    "$1 \\le x < y \\le 200$ such that $i^x + i^y$ is a real number."
)
FORM_CLAUSE = (
    " The answer is in the form \\frac{m}{n}, where gcd(m, n) = 1. Please "
    "provide the value of m + n."
)


def test_whitespace_key_is_the_mixture_identity():
    assert whitespace_key("  Find  X.\n\nNow ") == "find x. now"


def test_skeleton_folds_latex_markup_but_keeps_signs():
    assert skeleton_key("$\\dfrac{1}{2}$ of \\text{apples}") == skeleton_key(
        "\\(\\frac{1}{2}\\) of apples"
    )
    assert skeleton_key("\\left( a \\right)") == skeleton_key("(a)")
    # Dropping signs made these two different integrals one problem.
    assert skeleton_key("$(x+y)^2 dx - (x^2+y^2) dy$") != skeleton_key(
        "$(x+y)^2 dx + (x^2+y^2) dy$"
    )


def test_skeleton_keeps_non_latin_letters():
    # An ASCII-only skeleton reduced both to their shared digits.
    assert skeleton_key("已知 $x_1 = 2$，求 $x_2$") != skeleton_key(
        "设 $x_1 = 2$，证明 $x_2$"
    )


def test_cjk_characters_are_single_tokens():
    assert overlap_tokens("求 x12 的值") == ["求", "x", "12", "的", "值"]


def test_numbers_contained_is_a_multiset_rule():
    assert numbers_contained(number_multiset("1 2 3"), number_multiset("3 2 1 9"))
    assert not numbers_contained(number_multiset("1 2 200"), number_multiset("1 2 100 7"))
    assert not numbers_contained(number_multiset("2 2 3"), number_multiset("2 3 5"))


def test_shingle_matches_restatement_with_appended_clause():
    index = ProblemOverlapIndex([PAIRS])
    (match,) = index.matches(PAIRS.replace("$", "") + FORM_CLAUSE)
    assert match.matcher == "shingle"
    assert match.containment == pytest.approx(1.0)


def test_sibling_variant_with_different_numbers_is_not_a_duplicate():
    index = ProblemOverlapIndex([PAIRS])
    variant = PAIRS.replace("200", "100")
    assert shingles(variant) & shingles(PAIRS)
    assert index.matches(variant) == []


def test_matcher_precedence_and_threshold_validation():
    fractions = "Find $\\frac{1}{2} + \\frac{1}{3}$ in lowest terms."
    index = ProblemOverlapIndex([PAIRS, fractions])
    (match,) = index.matches(PAIRS.upper())
    assert (match.reference, match.matcher) == (0, "whitespace")
    (match,) = index.matches("Find \\(\\dfrac{1}{2}+\\dfrac{1}{3}\\) in lowest terms.")
    assert (match.reference, match.matcher) == (1, "skeleton")
    with pytest.raises(ValueError):
        ProblemOverlapIndex([], min_containment=0.0)
    with pytest.raises(ValueError):
        ProblemOverlapIndex([], min_shared=0)


def test_template_clause_alone_is_not_evidence():
    stems = [
        f"A bag holds {n} red and {n + 1} blue marbles; two are drawn at random."
        for n in range(3, 20)
    ]
    pool = [stem + FORM_CLAUSE for stem in stems]
    template = template_shingles([pool], max_documents=10)
    assert template
    index = ProblemOverlapIndex(pool[:1], template=template)
    other = "Two fair dice are rolled and the sum is recorded 3 4." + FORM_CLAUSE
    assert index.matches(other) == []
    # Without the template filter the shared clause alone would match.
    assert ProblemOverlapIndex(pool[:1], min_shared=1, min_containment=0.1).matches(
        other
    )


def test_within_pool_query_skips_its_own_row():
    problems = [PAIRS, PAIRS.replace("$", "")]
    index = ProblemOverlapIndex(problems)
    assert [match.reference for match in index.matches(problems[0], exclude=0)] == [1]


def test_skeleton_keeps_powers_decimals_and_division_but_not_grouping():
    assert skeleton_key("$2^{10}$") != skeleton_key("210")
    assert skeleton_key("1.5 kg") != skeleton_key("15 kg")
    assert skeleton_key("a/b") != skeleton_key("ab")
    assert skeleton_key("Find x.") == skeleton_key("find x")
    # Known limitation: grouping is formatting-equivalent to the key, so
    # these share one; screen_math_pool quarantines a same-text group whose
    # targets disagree instead of merging it.
    assert skeleton_key("\\frac{1}{2}+3") == skeleton_key("\\frac{1}{2+3}")


def test_cross_matches_reports_strongest_match_per_candidate():
    index = ProblemOverlapIndex([PAIRS.replace("$", "") + FORM_CLAUSE, PAIRS])
    (match,) = cross_matches(["unrelated short text", PAIRS], index)
    assert (match.candidate, match.reference, match.matcher) == (1, 1, "whitespace")
