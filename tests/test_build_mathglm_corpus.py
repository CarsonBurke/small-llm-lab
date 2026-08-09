from __future__ import annotations

from collections import Counter

import pytest

from scripts.build_k3_pretrain_dataset import quality_reason
from scripts.build_mathglm_corpus import (
    ANSWER_PREFIX,
    MAX_RESULT_DIGITS,
    MIN_CHARS,
    MIN_NOTED_STEPS,
    STEP_NOTES,
    ChainError,
    choose,
    documents,
    evaluate,
    parse_chain,
    render_item,
)

# Verbatim rows from jonathanasdf/MathGLM-dataset-5M. Keeping the real text
# means these tests fail if the dialect is ever misread, rather than passing
# against a tidied-up invention of what the corpus looks like.
GOOD = "5+4/2*1=5+2*1=5+2=7"
GOOD_PERCENT = "7.8239358053/1%=7.8239358053/0.01=782.3935805299999"
GOOD_BRACKET = "9*9+[2*(3+3)]=9*9+[2*6]=9*9+12=81+12=93"
GOOD_POWER = "7428^0=1"

# MathGLM's step generator reads the `-` inside `1.92...e-06` as an operator
# and collapses the term to `-06`. The stated answer is off by two orders of
# magnitude and the rewrite rule it demonstrates does not exist.
MANGLED_SCI = (
    "(83/(95+58)/29)-50/89/66/75/59=(83/153/29)-50/89/66/75/59"
    "=(0.5424836601307189/29)-50/89/66/75/59"
    "=0.018706333107955823-50/89/66/75/59"
    "=0.018706333107955823-0.5617977528089888/66/75/59"
    "=0.018706333107955823-0.008512087163772558/75/59"
    "=0.018706333107955823-0.00011349449551696743/59"
    "=0.018706333107955823-1.923635517236736e-06"
    "=0.018706333107955823-06"
    "=-5.981293666892044"
)

# The fraction family steps aside into scratch work -- `(5/2)+(1*(7/3))` is
# not the running value -- and then resumes the main chain. Every segment is
# joined by `=`, so the aside states an equality that is false.
FRACTION_ASIDE = (
    "(5/2)+((5/4)+(8/4)-((5/9)+(4/9))/(3/7))"
    "=(5/2)+((5/4)+(8/4)-(9/9)/(3/7))"
    "=(5/2)+((5/4)+(8/4)-(1)/(3/7))"
    "=(5/2)+(1*(7/3))"
    "=(5/2)+((5/4)+(8/4)-(7/3))"
    "=(5/2)+(13/4-7/3)"
    "=(5/2)+(39/12-28/12)"
    "=(5/2)+(11/12)"
    "=30/12+11/12"
)


class TestEvaluate:
    def test_percent_is_hundredths(self):
        assert evaluate("4%") == pytest.approx(0.04)

    def test_percent_takes_the_whole_literal_not_the_last_digit(self):
        # Reading `9598%` as `959` followed by `8%` turns a number into a
        # function call, which is how a validator quietly stops validating.
        assert evaluate("9598%") == pytest.approx(95.98)

    def test_percent_binds_tighter_than_the_division_around_it(self):
        # `4.0/1%` is 4.0/0.01, not 4.0/1/100.
        assert evaluate("4.0/1%") == pytest.approx(400.0)

    def test_brackets_group_like_parentheses(self):
        assert evaluate("9*9+[2*(3+3)]") == pytest.approx(93)

    def test_caret_raises_to_a_power(self):
        assert evaluate("2^10") == pytest.approx(1024)

    def test_precedence_follows_the_usual_rules(self):
        assert evaluate("5+4/2*1") == pytest.approx(7)

    def test_double_negation_is_addition(self):
        assert evaluate("-6.0--4.5") == pytest.approx(-1.5)

    @pytest.mark.parametrize(
        "segment", ["print(1)", "__import__", "1 if 1 else 2", "[1,2]"]
    )
    def test_anything_that_is_not_arithmetic_is_refused(self, segment):
        with pytest.raises(ChainError):
            evaluate(segment)

    def test_a_runaway_power_is_refused_rather_than_computed(self):
        with pytest.raises(ChainError, match="power_out_of_range"):
            evaluate(f"9999^{MAX_RESULT_DIGITS * 10}")

    def test_a_large_negative_exponent_is_computed_not_refused(self):
        # `46^-94` is a float, not a giant integer. Refusing it on exponent
        # magnitude discarded 64,275 rows and, worse, hid which of them were
        # actually wrong: MathGLM states this one's reciprocal with a
        # 125-digit denominator where the true one has 157.
        assert evaluate("15^-95") == pytest.approx(15.0**-95)
        assert evaluate("46^-94") == pytest.approx(46.0**-94)

    def test_a_several_hundred_digit_literal_does_not_escape_as_overflow(self):
        # Comparing segments must never raise out of parse_chain, so an
        # integer too large for a float is a rejection, not a crash.
        huge = "9" * 400
        with pytest.raises(ChainError, match="uncomputable_segment"):
            parse_chain(f"{huge}=1.5")
        # Compared against itself it needs no float at all, so it stands.
        assert parse_chain(f"{huge}={huge}")

    def test_a_wrong_reciprocal_is_caught_by_the_comparison(self):
        with pytest.raises(ChainError, match="segments_disagree"):
            parse_chain(
                "46^-94=1/199175620157317447901793579050643063502632025297"
                "20942980093969223149150811891544805563191778636291133402"
                "75736398800349932"
            )

    @pytest.mark.parametrize("base", ["1", "0"])
    def test_an_identity_base_is_admitted_at_any_exponent(self, base):
        # `1^2864=1` is arithmetic the corpus should teach. Gating on the
        # exponent alone rejected 37,204 rows of exactly this shape.
        assert evaluate(f"{base}^2864") == pytest.approx(float(base) ** 2864)

    def test_a_power_binds_tighter_than_the_sign_in_front_of_it(self):
        # `-1^2864` is -(1^2864), the standard reading. Getting this backwards
        # would not corrupt the corpus -- the chain would simply disagree with
        # itself and be dropped -- but it would drop a family silently.
        assert evaluate("-1^2864") == pytest.approx(-1.0)
        assert evaluate("(-1)^2864") == pytest.approx(1.0)

    def test_a_negative_exponent_is_a_reciprocal(self):
        assert evaluate("2^-2") == pytest.approx(0.25)

    def test_division_by_zero_is_a_chain_error(self):
        with pytest.raises(ChainError):
            evaluate("1/0")


class TestParseChain:
    def test_a_true_chain_splits_into_problem_rewrites_and_answer(self):
        problem, rewrites, answer = parse_chain(GOOD)
        assert problem == "5+4/2*1"
        assert rewrites == ["5+2*1", "5+2"]
        assert answer == "7"

    @pytest.mark.parametrize(
        "line", [GOOD, GOOD_PERCENT, GOOD_BRACKET, GOOD_POWER]
    )
    def test_the_dialect_is_read_rather_than_rejected(self, line):
        assert parse_chain(line)

    def test_the_mangled_scientific_notation_row_is_rejected(self):
        # The mangling leaves `...-06` behind, which is not a number Python
        # will read, so this row is caught at the parse rather than the
        # comparison. Either way it must not reach the corpus.
        with pytest.raises(ChainError, match="unparsed_segment"):
            parse_chain(MANGLED_SCI)

    def test_the_fraction_scratch_work_row_is_rejected(self):
        with pytest.raises(ChainError, match="segments_disagree"):
            parse_chain(FRACTION_ASIDE)

    def test_a_wide_product_is_compared_exactly_not_in_floating_point(self):
        # The true product ends 031080. A float carries sixteen significant
        # digits, so this row and 7,570 others read as correct unless whole
        # integers are compared as whole integers.
        with pytest.raises(ChainError, match="segments_disagree"):
            parse_chain("385924542305736*3405=1314073066551031040")
        assert parse_chain("385924542305736*3405=1314073066551031080")

    def test_a_wrong_answer_is_rejected_even_when_every_step_parses(self):
        with pytest.raises(ChainError, match="segments_disagree"):
            parse_chain("2+2=5")

    @pytest.mark.parametrize("line", ["", "7", "  ", "5+2=", "=7"])
    def test_anything_that_is_not_a_chain_is_rejected(self, line):
        with pytest.raises(ChainError, match="not_a_chain"):
            parse_chain(line)

    def test_a_segment_that_will_not_parse_is_rejected(self):
        # `-06` is the fingerprint of the mangling, and a Python syntax error.
        with pytest.raises(ChainError):
            parse_chain("0.5-06=0.5-06")


class TestRenderItem:
    def test_the_answer_line_carries_the_shared_prefix(self):
        text = render_item(*parse_chain(GOOD))
        assert text.splitlines()[-1] == f"{ANSWER_PREFIX}7"

    def test_every_rewrite_appears_in_order(self):
        text = render_item(*parse_chain(GOOD))
        assert [line for line in text.splitlines() if line.startswith("= ")] == [
            "= 5+2*1",
            "= 5+2",
            "= 7",
        ]

    def test_a_single_step_chain_gets_no_plural_step_note(self):
        # "1 steps, one operation each" is broken agreement in the training
        # data of a model whose downstream job is reading English.
        text = render_item("2+2", [], "4")
        assert "1 steps" not in text
        assert "This takes 1 steps" not in text

    def test_no_single_step_chain_ever_carries_a_note(self):
        # A note, when present, sits between the prompt and the expression,
        # so the expression landing on line two means no note was attached.
        for n in range(500):
            lines = render_item(f"{n}+2", [], str(n + 2)).splitlines()
            assert lines[1] == f"{n}+2"

    def test_a_note_states_the_true_step_count_when_it_appears(self):
        seen = 0
        for n in range(500):
            steps = MIN_NOTED_STEPS + n % 4
            rewrites = [f"r{i}" for i in range(steps - 1)]
            lines = render_item(f"{n}+2", rewrites, str(n + 2)).splitlines()
            if lines[1] == f"{n}+2":
                continue
            assert lines[1] in {
                note.format(steps=steps) for note in STEP_NOTES if note
            }
            seen += 1
        # Notes are hash-selected and half the pool is empty, so the loop has
        # to confirm they occur at all -- otherwise the assert above is
        # vacuous.
        assert seen > 0


class TestChoose:
    def test_selection_is_stable_across_calls(self):
        pool = ("a", "b", "c", "d")
        assert choose(pool, "x", "y") == choose(pool, "x", "y")

    def test_different_content_spreads_across_the_pool(self):
        pool = tuple("abcdefghij")
        picked = {choose(pool, "k", str(n)) for n in range(400)}
        assert len(picked) == len(pool)


class TestDocuments:
    def write(self, tmp_path, lines):
        path = tmp_path / "chains.txt"
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_every_emitted_document_clears_the_quality_gate(self, tmp_path):
        source = self.write(tmp_path, [GOOD] * 400)
        counts: Counter = Counter()
        built = list(documents(source, frozenset(), None, counts))
        assert built
        assert all(quality_reason(text, "math") is None for text in built)

    def test_documents_reach_the_packing_floor(self, tmp_path):
        source = self.write(tmp_path, [GOOD] * 400)
        counts: Counter = Counter()
        built = list(documents(source, frozenset(), None, counts))
        # The last document is whatever was left in the buffer, so it alone is
        # allowed to fall short of the target.
        assert all(len(text) >= MIN_CHARS for text in built[:-1])

    def test_corrupted_rows_never_reach_a_document(self, tmp_path):
        source = self.write(tmp_path, [MANGLED_SCI, FRACTION_ASIDE] * 200)
        counts: Counter = Counter()
        built = list(documents(source, frozenset(), None, counts))
        assert built == []
        rejected = sum(
            v for k, v in counts.items() if k.startswith("rows_rejected:")
        )
        assert rejected == 400
        assert counts["rows_used"] == 0

    def test_rows_used_counts_rows_that_survived_into_the_corpus(
        self, tmp_path
    ):
        source = self.write(tmp_path, [GOOD, MANGLED_SCI] * 200)
        counts: Counter = Counter()
        built = list(documents(source, frozenset(), None, counts))
        accounted = (
            counts["rows_used"]
            + counts["rows_dropped_with_document"]
            + counts["rows_registry_excluded"]
            + sum(v for k, v in counts.items() if k.startswith("rows_rejected:"))
        )
        assert accounted == counts["rows_read"]
        assert counts["documents"] == len(built)

    def test_registry_claimed_problems_are_excluded(self, tmp_path):
        from postraining.problem_registry import problem_key

        source = self.write(tmp_path, [GOOD] * 400)
        counts: Counter = Counter()
        excluded = frozenset({problem_key("5+4/2*1")})
        assert list(documents(source, excluded, None, counts)) == []
        assert counts["rows_registry_excluded"] == 400

    def test_max_rows_stops_the_intake(self, tmp_path):
        source = self.write(tmp_path, [GOOD] * 400)
        counts: Counter = Counter()
        list(documents(source, frozenset(), 25, counts))
        assert counts["rows_read"] == 25
