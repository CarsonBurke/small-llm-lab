from __future__ import annotations

import math
import random
import re
from fractions import Fraction

import pytest

from postraining.arithmetic_probe import (
    PROBE_CURRICULUM,
    PROBE_MIN_DIGITS,
    PROBE_SEED,
    TRAINING_SEED,
    assert_disjoint,
    build_panel,
    canonical_answer,
    graded,
    read_panel,
    score,
    training_keys,
    write_panel,
)
from postraining.math_drills import (
    BUILDERS,
    DEFAULT_CURRICULUM,
    DIGIT_ORDERS,
    FAMILIES,
    FAMILY_ORDERS,
    FamilySpec,
    column_addition,
    column_subtraction,
    decimal_places,
    decimal_text,
    drill_statistics,
    expanded_addition,
    expanded_subtraction,
    generate,
    long_division,
    partial_products,
    strip_decimal,
)

# One sample large enough to reach every family, shared across tests because
# generating it is the expensive part.
SAMPLE = None


def sample():
    global SAMPLE
    if SAMPLE is None:
        SAMPLE = list(generate(seed=7, count=4000))
    return SAMPLE


# -- number formatting ---------------------------------------------------


def test_decimal_places_counts_what_exact_rendering_needs():
    assert decimal_places(Fraction(1, 2)) == 1
    assert decimal_places(Fraction(1, 8)) == 3
    assert decimal_places(Fraction(1, 20)) == 2
    assert decimal_places(Fraction(3, 1)) == 0


def test_non_terminating_values_raise_rather_than_truncate():
    with pytest.raises(ValueError, match="no exact decimal expansion"):
        decimal_places(Fraction(1, 3))


def test_decimal_text_refuses_to_lose_information():
    assert decimal_text(Fraction(1, 8), 3) == "0.125"
    with pytest.raises(ValueError, match="not exact"):
        decimal_text(Fraction(1, 8), 2)


def test_strip_decimal_leaves_a_readable_number():
    assert strip_decimal("2.500") == "2.5"
    assert strip_decimal("7.000") == "7"
    assert strip_decimal("0.0") == "0"
    assert strip_decimal("120") == "120"


# -- working is correct and causal ---------------------------------------


def test_column_addition_digits_reconstruct_the_sum():
    for left, right in [(4728, 3596), (999, 1), (5, 5), (100000, 999999)]:
        lines = column_addition(left, right)
        digits = lines[-1].split(": ")[1].split()
        value = int("".join(reversed(digits)))
        assert value == left + right, (left, right, lines)


def test_column_subtraction_digits_reconstruct_the_difference():
    for left, right in [(76175, 7428), (1000, 1), (55, 55), (900, 899)]:
        lines = column_subtraction(left, right)
        digits = lines[-1].split(": ")[1].split()
        assert int("".join(reversed(digits))) == left - right


def test_expanded_forms_end_on_the_answer():
    for left, right in [(606643, 47746), (452, 58), (10, 9)]:
        assert expanded_addition(left, right)[-1].endswith(str(left + right))
        assert expanded_subtraction(left, right)[-1].endswith(str(left - right))


def test_expanded_subtraction_states_negative_places_rather_than_hiding_them():
    lines = expanded_subtraction(452, 58)
    assert any("= -6" in line for line in lines)
    assert lines[-1].endswith("394")


def test_partial_products_end_on_the_product_in_both_orders():
    for order in DIGIT_ORDERS:
        lines = partial_products(5039, 4589, order)
        assert lines[-1].endswith(str(5039 * 4589))


def test_partial_products_start_at_the_declared_end():
    assert "(ones)" in partial_products(123, 456, "reversed")[1]
    assert "(hundreds)" in partial_products(123, 456, "forward")[1]


def test_long_division_quotient_digits_reconstruct_the_quotient():
    lines = long_division(904583, 41, 0)
    quotient = lines[-1].split(": ")[1]
    assert int(quotient) == 904583 // 41
    lines = long_division(125, 8, 3)
    assert Fraction(lines[-1].split(": ")[1]) == Fraction(125, 8)


def test_long_division_refuses_a_non_positive_divisor():
    with pytest.raises(ValueError, match="divisor must be positive"):
        long_division(10, 0, 0)


def test_every_working_line_is_produced_before_it_is_used():
    # A reversed column trace printed forwards would mention a carry above the
    # line that creates it. This asserts the property that failure would break:
    # in reversed order the first working line is the ones place.
    for drill in sample():
        if drill.digit_order != "reversed":
            continue
        body = drill.solution.splitlines()
        places = [line for line in body if re.match(r"^(ones|tens|hundreds)", line)]
        if places:
            assert places[0].startswith("ones"), drill.solution


# -- generation contract -------------------------------------------------


def test_every_family_is_reachable_and_declares_its_orders():
    families = drill_statistics(sample())["families"]
    assert set(families) == set(FAMILIES)
    assert set(FAMILY_ORDERS) == set(BUILDERS)
    for drill in sample():
        assert drill.digit_order in FAMILY_ORDERS[drill.family]


def test_division_families_only_ever_run_forward():
    # Long division consumes the remainder the line above produced, so it has
    # no causal reversed rendering and must not claim one.
    assert FAMILY_ORDERS["div_integer"] == ("forward",)
    assert FAMILY_ORDERS["div_decimal"] == ("forward",)


def test_generation_is_a_function_of_the_seed():
    first = [d.document() for d in generate(seed=5, count=200)]
    assert first == [d.document() for d in generate(seed=5, count=200)]
    assert first != [d.document() for d in generate(seed=6, count=200)]


def test_generated_problems_are_distinct():
    drills = sample()
    assert len({d.key for d in drills}) == len(drills)


def test_excluded_keys_are_never_emitted():
    first = list(generate(seed=5, count=50))
    excluded = frozenset(d.key for d in first)
    for drill in generate(seed=5, count=50, excluded_keys=excluded):
        assert drill.key not in excluded


def test_an_impossible_count_fails_loudly():
    # One-digit addition has a hundred problems; asking for ten thousand
    # distinct ones must fail rather than loop or repeat.
    tiny = (FamilySpec("add_integer", 1.0, 1, 1),)
    with pytest.raises(RuntimeError, match="problem space is too small"):
        list(generate(seed=1, count=10_000, curriculum=tiny))


def test_family_spec_rejects_nonsense():
    with pytest.raises(ValueError, match="unknown drill family"):
        FamilySpec("add_octal", 1.0)
    with pytest.raises(ValueError, match="positive weight"):
        FamilySpec("add_integer", 0.0)
    with pytest.raises(ValueError, match="not a valid ascending range"):
        FamilySpec("add_integer", 1.0, 4, 2)


def test_documents_end_with_the_answer():
    for drill in sample():
        assert drill.document().endswith(f"\nAnswer: {drill.answer}")


# -- answers are arithmetically right ------------------------------------


def solve(problem: str) -> str | None:
    """Re-solve a generated problem independently of how it was built."""
    if m := re.fullmatch(r"What is ([\d.]+) \+ ([\d.]+)\?", problem):
        # Fraction, not float: 954.134 + 17.94 is 972.074, and binary floating
        # point would report 972.0740000000001 and fail a correct drill.
        return strip_decimal(decimal_text(Fraction(m[1]) + Fraction(m[2])))
    if m := re.fullmatch(r"What is (\d+) divided by (\d+)\? Give the quotient and remainder\.", problem):
        quotient, remainder = divmod(int(m[1]), int(m[2]))
        return f"{quotient} remainder {remainder}"
    return None


def test_answers_match_an_independent_solver():
    checked = 0
    for drill in sample():
        expected = solve(drill.problem)
        if expected is None:
            continue
        checked += 1
        assert graded(drill.answer, expected), (drill.problem, drill.answer)
    assert checked > 100


def test_exact_arithmetic_across_every_numeric_family():
    # Fractions, not floats: a float check would pass on answers that are only
    # nearly right, which is the failure mode that matters here.
    patterns = {
        r"What is ([\d.]+) \+ ([\d.]+)\?": lambda a, b: a + b,
        r"What is ([\d.]+) - ([\d.]+)\?": lambda a, b: a - b,
        r"What is ([\d.]+) \* ([\d.]+)\?": lambda a, b: a * b,
        r"What is ([\d.]+) divided by ([\d.]+)\?": lambda a, b: a / b,
    }
    checked = 0
    for drill in sample():
        for pattern, operation in patterns.items():
            if match := re.fullmatch(pattern, drill.problem):
                left, right = (Fraction(value) for value in match.groups())
                assert operation(left, right) == Fraction(drill.answer), drill.problem
                checked += 1
    assert checked > 500


def test_rounding_uses_one_stated_rule():
    checked = 0
    for drill in sample():
        match = re.fullmatch(
            r"Round ([\d.]+) to (the nearest whole number|\d+ decimal places?)\.",
            drill.problem,
        )
        if not match:
            continue
        checked += 1
        places = 0 if match[2].startswith("the") else int(match[2].split()[0])
        scaled = Fraction(match[1]) * 10**places
        floor = math.floor(scaled)
        expected = Fraction(
            floor + (1 if scaled - floor >= Fraction(1, 2) else 0), 10**places
        )
        assert Fraction(drill.answer) == expected
    assert checked > 20


def test_unit_conversions_are_exact():
    for drill in sample():
        if drill.family != "unit_convert":
            continue
        # An inexact conversion would have raised during generation; this
        # asserts the answer is still a terminating decimal on the way out.
        assert Fraction(drill.answer).denominator in {
            1,
            2,
            4,
            5,
            8,
            10,
            16,
            20,
            25,
            40,
            50,
            100,
            125,
            200,
            250,
            500,
            1000,
        }


# -- probe ---------------------------------------------------------------


def test_probe_answers_canonicalize_one_to_one():
    assert graded("2.50", "2.5")
    assert graded("+7", "7")
    assert graded("4/8", "1/2")
    assert graded("  <  ", "<")
    assert not graded("6", "6 remainder 4")
    assert not graded(">", "<")
    assert canonical_answer("") is None
    assert canonical_answer("about five") is None
    assert canonical_answer("1/0") is None


def test_an_ungradable_prediction_is_wrong_not_skipped():
    panel = build_panel(per_family=2)
    result = score(panel, ["I am not sure"] * len(panel))
    assert result["accuracy"] == 0.0
    assert result["ungradable_predictions"] == len(panel)


def test_a_reference_answer_must_itself_be_canonical():
    with pytest.raises(ValueError, match="not canonical"):
        graded("5", "roughly five")


def test_panel_is_balanced_and_stratified():
    panel = build_panel(per_family=8)
    counts: dict[str, int] = {}
    for item in panel:
        counts[item.family] = counts.get(item.family, 0) + 1
    assert set(counts) == set(FAMILIES)
    assert set(counts.values()) == {8}
    # PROBE_MIN_DIGITS is a floor, capped by what each family can reach:
    # fractions top out at two digits, so their probe items do too.
    ceilings = {spec.name: spec.max_digits for spec in DEFAULT_CURRICULUM}
    for item in panel:
        assert item.digits >= min(PROBE_MIN_DIGITS, ceilings[item.family])


def test_panel_is_disjoint_from_the_training_stream():
    keys = training_keys(count=20_000)
    panel = build_panel(per_family=8, excluded_keys=keys)
    assert_disjoint(panel, training_count=20_000)


def test_disjointness_check_actually_fires():
    # Drawing the panel from the training seed must be caught, or the check
    # would be decoration.
    panel = build_panel(per_family=4, seed=TRAINING_SEED)
    with pytest.raises(AssertionError, match="also appear in"):
        assert_disjoint(panel, training_seed=TRAINING_SEED, training_count=20_000)


def test_probe_and_training_seeds_differ():
    assert PROBE_SEED != TRAINING_SEED


def test_probe_curriculum_never_widens_a_family_range():
    by_name = {spec.name: spec for spec in DEFAULT_CURRICULUM}
    for spec in PROBE_CURRICULUM:
        original = by_name[spec.name]
        assert spec.min_digits >= original.min_digits
        assert spec.max_digits == original.max_digits
        assert spec.min_digits <= spec.max_digits


def test_panel_round_trips_through_disk(tmp_path):
    panel = build_panel(per_family=4)
    write_panel(panel, tmp_path / "probe.jsonl", {"seed": PROBE_SEED})
    restored, provenance = read_panel(tmp_path / "probe.jsonl")
    assert restored == panel
    assert provenance["seed"] == PROBE_SEED


def test_score_refuses_a_mismatched_prediction_count():
    panel = build_panel(per_family=2)
    with pytest.raises(ValueError, match="predictions for"):
        score(panel, ["1"] * (len(panel) - 1))


def test_score_reports_per_family_and_per_digit():
    panel = build_panel(per_family=4)
    predictions = [
        item.answer if item.family == "add_integer" else "wrong"
        for item in panel
    ]
    result = score(panel, predictions)
    assert result["by_family"]["add_integer"]["accuracy"] == 1.0
    assert result["by_family"]["mul_integer"]["accuracy"] == 0.0
    digits = result["by_family_digits"]["add_integer"]
    assert digits and all(bucket["accuracy"] == 1.0 for bucket in digits.values())


# -- the sample itself is not degenerate ---------------------------------


def test_generation_covers_a_range_of_digit_counts():
    statistics = drill_statistics(sample())
    for family in ("add_integer", "mul_integer", "div_integer"):
        assert len(statistics["digits_by_family"][family]) >= 3


def test_both_digit_orders_appear():
    orders = drill_statistics(sample())["digit_orders"]
    assert orders["forward"] > 0
    assert orders["reversed"] > 0


def test_builders_accept_the_generic_signature():
    rng = random.Random(0)
    for name, builder in BUILDERS.items():
        drill = builder(rng, FAMILY_ORDERS[name][0], 3)
        assert drill.family == name
        assert drill.answer
        assert drill.solution


def test_only_contract_valid_completions_are_credited():
    """A scraped tail answer is not an answer.

    `evaluate_latent_math` falls back to the last number in the emitted text
    when there is no closing fence, which is the right leniency for a
    training-time signal and the wrong one for a capability measurement.
    """
    from postraining.run_arithmetic_probe import contract_predictions

    captured = [
        {"terminated": True, "structural_format_ok": True},
        {"terminated": False, "structural_format_ok": True},
        {"terminated": True, "structural_format_ok": False},
        {"terminated": False, "structural_format_ok": False},
    ]
    parsed = ["12", "34", "56", "78"]
    assert contract_predictions(captured, parsed) == ["12", "", "", ""]


def test_contract_filter_refuses_a_length_mismatch():
    from postraining.run_arithmetic_probe import contract_predictions

    with pytest.raises(ValueError):
        contract_predictions([{"terminated": True, "structural_format_ok": True}], [])


def test_a_trained_combiner_is_detected_from_the_parameters():
    """The probe refuses a latent checkpoint on what it holds, not what it says.

    `train_latent_vapo.py` reads its own `reasoning_mode` with a `latent`
    default, so a payload can carry a trained combiner while recording no mode
    at all -- and a check on `args` would let exactly the checkpoint it exists
    to refuse through, to be measured with its hidden carry pinned off.
    """
    from postraining.run_arithmetic_probe import carries_trained_combiner

    assert carries_trained_combiner(
        {"model": {"combiner.carry.weight": None, "backbone.wte.weight": None}}
    )
    # No recorded mode, and no `args` key at all: still refused.
    assert carries_trained_combiner({"model": {"combiner.type_bias": None}})
    # Recorded mode with parameters stored some other way: the second line.
    assert carries_trained_combiner({"model": {}, "reasoning_mode": "latent"})

    assert not carries_trained_combiner(
        {"model": {"wte.weight": None}, "args": {"reasoning_mode": "cot"}}
    )
    # A bare backbone checkpoint, which is what SFT writes.
    assert not carries_trained_combiner({"model": {"blocks.0.attn.qkv.weight": None}})
    assert not carries_trained_combiner({})
