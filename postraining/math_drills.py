"""Deterministic worked-step arithmetic drills.

The 2026-08-06 corpus audit found that 0.115% of the pretraining mix was core
arithmetic drill and that none of it showed any working: the DeepMind and
OpenMathInstruct QA documents are rendered as ``{question}\\nAnswer: {answer}``,
so the model only ever sees a problem next to its result. A model cannot learn
carrying, borrowing, place value, or decimal alignment from input-output pairs
alone -- there is no supervision on the mechanism, only on the outcome.

This module generates that missing supervision. Every drill carries the
working a person would write, so the corpus teaches the procedure and not just
the answer.

Three properties matter for the drills to be usable as pretraining data.

*Determinism.* A drill set is a function of its seed and its curriculum. The
same seed rebuilds the same bytes, so a manifest hash identifies the data.

*Disjointness.* Generated problems are hashed with the same key function the
global problem registry uses, so a drill can be checked against evaluation and
post-training splits, and a probe set can be carved out that the training
stream provably never contains.

*Causality.* Every line follows from lines above it. This constrains the
design more than it sounds. Autoregressive models find least-significant-digit
first arithmetic far easier, because that is the order carries actually
resolve -- but a column-method trace cannot simply be printed in reverse to
make a "most significant first" variant, because the reversed text would state
each carry before the line that produces it, teaching a procedure that does
not work. The two digit orders are therefore two genuinely different methods:

* ``reversed`` -- the column algorithm, ones place first, carrying or
  borrowing leftwards;
* ``forward`` -- expanded form, largest place first, where each place's
  contribution is independent and the parts are combined at the end.

Both are real algorithms a person uses, both read correctly top to bottom, and
a family that has only one causal order (long division is always most
significant first) declares only that one. The final answer line is always
canonical, most significant digit first, whichever method produced it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from postraining.problem_registry import problem_key

DRILL_SCHEMA = "worked_arithmetic_drills/v2"

DIGIT_ORDERS = ("reversed", "forward")


@dataclass(frozen=True)
class Drill:
    """One generated problem with its worked solution."""

    family: str
    problem: str
    solution: str
    answer: str
    digit_order: str
    difficulty: dict = field(default_factory=dict)

    @property
    def key(self) -> bytes:
        return problem_key(self.problem)

    def document(self) -> str:
        """The pretraining rendering: problem, working, then the answer."""
        return f"{self.problem}\n{self.solution}\nAnswer: {self.answer}"


# -- number formatting ---------------------------------------------------


def digits_of(value: int) -> list[int]:
    return [int(character) for character in str(abs(value))]


def place_name(index: int) -> str:
    names = (
        "ones",
        "tens",
        "hundreds",
        "thousands",
        "ten thousands",
        "hundred thousands",
        "millions",
        "ten millions",
    )
    return names[index] if index < len(names) else f"10^{index}"


def strip_decimal(text: str) -> str:
    """Canonical decimal text: no trailing zeros, no trailing point."""
    if "." not in text:
        return text
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def decimal_places(value: Fraction) -> int:
    """Places needed to write ``value`` exactly.

    A reduced fraction terminates exactly when its denominator is 2^a * 5^b,
    and then needs max(a, b) places. Anything else raises: a drill that
    silently truncated its own answer would teach the wrong one.
    """
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError(f"{value} has no exact decimal expansion")
    return max(twos, fives)


def decimal_text(value: Fraction, places: int | None = None) -> str:
    """Exact decimal rendering; raises if ``places`` would lose information."""
    if places is None:
        places = decimal_places(value)
    scaled = value * 10**places
    if scaled.denominator != 1:
        raise ValueError(f"{value} is not exact to {places} decimal places")
    sign = "-" if scaled < 0 else ""
    magnitude = abs(int(scaled))
    if places == 0:
        return f"{sign}{magnitude}"
    text = str(magnitude).rjust(places + 1, "0")
    return f"{sign}{text[:-places]}.{text[-places:]}"


def exact(value: Fraction) -> str:
    """Shortest exact decimal text for a terminating value."""
    return strip_decimal(decimal_text(value))


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


# -- sampling ------------------------------------------------------------


def sample_integer(rng: random.Random, digits: int) -> int:
    low = 10 ** (digits - 1) if digits > 1 else 0
    return rng.randint(low, 10**digits - 1)


def sample_decimal(rng: random.Random, digits: int, places: int) -> Fraction:
    """An exact decimal with ``digits`` integer digits and ``places`` after."""
    whole = sample_integer(rng, digits)
    fractional = rng.randint(0, 10**places - 1) if places else 0
    return Fraction(whole * 10**places + fractional, 10**places)


# -- shared working ------------------------------------------------------


def column_addition(left: int, right: int) -> list[str]:
    """Ones-place-first column addition with explicit carries."""
    left_digits = list(reversed(digits_of(left)))
    right_digits = list(reversed(digits_of(right)))
    width = max(len(left_digits), len(right_digits))
    lines = ["Add column by column, starting at the ones place."]
    written: list[int] = []
    carry = 0
    for index in range(width):
        a = left_digits[index] if index < len(left_digits) else 0
        b = right_digits[index] if index < len(right_digits) else 0
        total = a + b + carry
        sum_text = f"{a} + {b}" + (f" + {carry} carried" if carry else "")
        lines.append(
            f"{place_name(index)}: {sum_text} = {total}, write {total % 10}, "
            f"carry {total // 10}"
        )
        written.append(total % 10)
        carry = total // 10
    if carry:
        lines.append(f"{place_name(width)}: nothing left but the carry {carry}")
        written.append(carry)
    lines.append(
        "Digits from the ones place: " + " ".join(str(d) for d in written)
    )
    return lines


def expanded_addition(left: int, right: int) -> list[str]:
    """Largest-place-first addition by expanded form.

    Each place's contribution stands alone, so this reads correctly from the
    top even though it starts at the most significant digit.
    """
    left_digits = digits_of(left)
    right_digits = digits_of(right)
    width = max(len(left_digits), len(right_digits))
    lines = ["Split both numbers by place value and add the places."]
    parts: list[int] = []
    for index in range(width - 1, -1, -1):
        a = left_digits[len(left_digits) - 1 - index] if index < len(left_digits) else 0
        b = (
            right_digits[len(right_digits) - 1 - index]
            if index < len(right_digits)
            else 0
        )
        part = (a + b) * 10**index
        lines.append(
            f"{place_name(index)}: {a * 10**index} + {b * 10**index} = {part}"
        )
        parts.append(part)
    running = parts[0]
    for part in parts[1:]:
        lines.append(f"{running} + {part} = {running + part}")
        running += part
    return lines


def column_subtraction(left: int, right: int) -> list[str]:
    """Ones-place-first column subtraction with explicit borrows."""
    left_digits = list(reversed(digits_of(left)))
    right_digits = list(reversed(digits_of(right)))
    lines = ["Subtract column by column, starting at the ones place."]
    written: list[int] = []
    borrow = 0
    for index in range(len(left_digits)):
        top = left_digits[index] - borrow
        bottom = right_digits[index] if index < len(right_digits) else 0
        borrowed = "" if borrow == 0 else " after the borrow"
        if top < bottom:
            lines.append(
                f"{place_name(index)}: {top}{borrowed} is less than {bottom}, "
                f"borrow ten: {top + 10} - {bottom} = {top + 10 - bottom}"
            )
            written.append(top + 10 - bottom)
            borrow = 1
        else:
            lines.append(
                f"{place_name(index)}: {top}{borrowed} - {bottom} = {top - bottom}"
            )
            written.append(top - bottom)
            borrow = 0
    lines.append(
        "Digits from the ones place: " + " ".join(str(d) for d in written)
    )
    return lines


def expanded_subtraction(left: int, right: int) -> list[str]:
    """Largest-place-first subtraction by expanded form.

    A place can come out negative -- 452 - 58 gives ones 2 - 8 = -6 -- and
    that is not a mistake to hide. Combining signed place values left to right
    is a correct method, and it keeps every line independent of the ones
    below it.
    """
    left_digits = digits_of(left)
    right_digits = digits_of(right)
    width = max(len(left_digits), len(right_digits))
    lines = [
        "Split both numbers by place value and subtract place by place; a "
        "place may come out negative."
    ]
    parts: list[int] = []
    for index in range(width - 1, -1, -1):
        a = left_digits[len(left_digits) - 1 - index] if index < len(left_digits) else 0
        b = (
            right_digits[len(right_digits) - 1 - index]
            if index < len(right_digits)
            else 0
        )
        part = (a - b) * 10**index
        lines.append(
            f"{place_name(index)}: {a * 10**index} - {b * 10**index} = {part}"
        )
        parts.append(part)
    running = parts[0]
    for part in parts[1:]:
        operator = "-" if part < 0 else "+"
        lines.append(f"{running} {operator} {abs(part)} = {running + part}")
        running += part
    return lines


def partial_products(left: int, right: int, digit_order: str) -> list[str]:
    """Multiplication by one partial product per digit of ``right``.

    Partial products do not depend on each other, so either order reads
    correctly; only the running total has to follow the order chosen.
    """
    indexed = list(enumerate(reversed(digits_of(right))))
    if digit_order == "forward":
        indexed = list(reversed(indexed))
    lines = [
        f"Multiply {left} by each digit of {right}, then add the partial "
        "products."
    ]
    parts: list[int] = []
    for index, digit in indexed:
        part = left * digit * 10**index
        shift = "" if index == 0 else f", shifted {plural(index, 'place')}"
        lines.append(f"{left} x {digit} ({place_name(index)}){shift} = {part}")
        parts.append(part)
    running = parts[0]
    for part in parts[1:]:
        lines.append(f"{running} + {part} = {running + part}")
        running += part
    return lines


def long_division(dividend: int, divisor: int, places: int) -> list[str]:
    """Most significant digit first long division, carried ``places`` places.

    Long division has exactly one causal order: each step consumes the
    remainder the step above produced. There is no reversed variant of it,
    which is why the division families declare only the forward order.
    """
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    lines = [f"Divide {dividend} by {divisor}, one digit at a time."]
    remainder = 0
    quotient: list[str] = []
    for digit in digits_of(dividend):
        current = remainder * 10 + digit
        lines.append(
            f"bring down {digit} -> {current}; {current} / {divisor} = "
            f"{current // divisor} remainder {current % divisor}"
        )
        quotient.append(str(current // divisor))
        remainder = current % divisor
    if places:
        lines.append(
            f"The remainder is {remainder}; continue past the decimal point."
        )
        quotient.append(".")
        for _ in range(places):
            current = remainder * 10
            lines.append(
                f"bring down 0 -> {current}; {current} / {divisor} = "
                f"{current // divisor} remainder {current % divisor}"
            )
            quotient.append(str(current // divisor))
            remainder = current % divisor
    lines.append("Quotient digits: " + "".join(quotient))
    return lines


# -- integer families ----------------------------------------------------


def add_integer(rng: random.Random, digit_order: str, digits: int) -> Drill:
    left = sample_integer(rng, digits)
    right = sample_integer(rng, rng.randint(max(1, digits - 1), digits))
    working = (
        column_addition(left, right)
        if digit_order == "reversed"
        else expanded_addition(left, right)
    )
    return Drill(
        family="add_integer",
        problem=f"What is {left} + {right}?",
        solution="\n".join(working),
        answer=str(left + right),
        digit_order=digit_order,
        difficulty={"digits": digits},
    )


def sub_integer(rng: random.Random, digit_order: str, digits: int) -> Drill:
    left = sample_integer(rng, digits)
    right = rng.randint(0, left)
    working = (
        column_subtraction(left, right)
        if digit_order == "reversed"
        else expanded_subtraction(left, right)
    )
    return Drill(
        family="sub_integer",
        problem=f"What is {left} - {right}?",
        solution="\n".join(working),
        answer=str(left - right),
        digit_order=digit_order,
        difficulty={"digits": digits},
    )


def mul_integer(rng: random.Random, digit_order: str, digits: int) -> Drill:
    left = sample_integer(rng, digits)
    right = sample_integer(rng, rng.randint(1, max(1, digits - 1)))
    return Drill(
        family="mul_integer",
        problem=f"What is {left} * {right}?",
        solution="\n".join(partial_products(left, right, digit_order)),
        answer=str(left * right),
        digit_order=digit_order,
        difficulty={"digits": digits},
    )


def div_integer(rng: random.Random, digit_order: str, digits: int) -> Drill:
    divisor = max(2, sample_integer(rng, rng.randint(1, max(1, digits - 1))))
    dividend = sample_integer(rng, digits)
    quotient, remainder = divmod(dividend, divisor)
    return Drill(
        family="div_integer",
        problem=(
            f"What is {dividend} divided by {divisor}? Give the quotient and "
            "remainder."
        ),
        solution="\n".join(long_division(dividend, divisor, 0)),
        answer=f"{quotient} remainder {remainder}",
        digit_order=digit_order,
        difficulty={"digits": digits},
    )


# -- decimal families ----------------------------------------------------


def _aligned(rng: random.Random, digits: int) -> tuple[Fraction, int, Fraction, int, int]:
    left_places = rng.randint(1, 3)
    right_places = rng.randint(1, 3)
    left = sample_decimal(rng, digits, left_places)
    right = sample_decimal(rng, max(1, digits - 1), right_places)
    return left, left_places, right, right_places, max(left_places, right_places)


def add_decimal(rng: random.Random, digit_order: str, digits: int) -> Drill:
    left, left_places, right, right_places, width = _aligned(rng, digits)
    scaled_left = int(left * 10**width)
    scaled_right = int(right * 10**width)
    working = [
        f"Line the decimal points up by padding both to "
        f"{plural(width, 'decimal place')}: {decimal_text(left, width)} and "
        f"{decimal_text(right, width)}.",
        f"Drop the points and add whole numbers: {scaled_left} + {scaled_right}.",
        *(
            column_addition(scaled_left, scaled_right)
            if digit_order == "reversed"
            else expanded_addition(scaled_left, scaled_right)
        ),
        f"That whole-number sum is {scaled_left + scaled_right}; put the point "
        f"back {plural(width, 'place')} from the right.",
    ]
    return Drill(
        family="add_decimal",
        problem=(
            f"What is {decimal_text(left, left_places)} + "
            f"{decimal_text(right, right_places)}?"
        ),
        solution="\n".join(working),
        answer=exact(left + right),
        digit_order=digit_order,
        difficulty={"digits": digits, "places": width},
    )


def sub_decimal(rng: random.Random, digit_order: str, digits: int) -> Drill:
    left, left_places, right, right_places, width = _aligned(rng, digits)
    if right > left:
        left, right = right, left
        left_places, right_places = right_places, left_places
    scaled_left = int(left * 10**width)
    scaled_right = int(right * 10**width)
    working = [
        f"Line the decimal points up by padding both to "
        f"{plural(width, 'decimal place')}: {decimal_text(left, width)} and "
        f"{decimal_text(right, width)}.",
        f"Drop the points and subtract whole numbers: {scaled_left} - "
        f"{scaled_right}.",
        *(
            column_subtraction(scaled_left, scaled_right)
            if digit_order == "reversed"
            else expanded_subtraction(scaled_left, scaled_right)
        ),
        f"That whole-number difference is {scaled_left - scaled_right}; put "
        f"the point back {plural(width, 'place')} from the right.",
    ]
    return Drill(
        family="sub_decimal",
        problem=(
            f"What is {decimal_text(left, left_places)} - "
            f"{decimal_text(right, right_places)}?"
        ),
        solution="\n".join(working),
        answer=exact(left - right),
        digit_order=digit_order,
        difficulty={"digits": digits, "places": width},
    )


def mul_decimal(rng: random.Random, digit_order: str, digits: int) -> Drill:
    left_places = rng.randint(1, 2)
    right_places = rng.randint(1, 2)
    left = sample_decimal(rng, digits, left_places)
    right = sample_decimal(rng, max(1, digits - 1), right_places)
    scaled_left = int(left * 10**left_places)
    scaled_right = int(right * 10**right_places)
    total_places = left_places + right_places
    working = [
        f"Ignore the points and multiply {scaled_left} by {scaled_right}.",
        *partial_products(scaled_left, scaled_right, digit_order),
        f"{decimal_text(left, left_places)} has "
        f"{plural(left_places, 'decimal place')} and "
        f"{decimal_text(right, right_places)} has {right_places}, so the "
        f"product has {total_places}.",
        f"Place the point {plural(total_places, 'digit')} from the right of "
        f"{scaled_left * scaled_right}.",
    ]
    return Drill(
        family="mul_decimal",
        problem=(
            f"What is {decimal_text(left, left_places)} * "
            f"{decimal_text(right, right_places)}?"
        ),
        solution="\n".join(working),
        answer=exact(left * right),
        digit_order=digit_order,
        difficulty={"digits": digits, "places": total_places},
    )


def div_decimal(rng: random.Random, digit_order: str, digits: int) -> Drill:
    divisor_places = rng.randint(1, 2)
    divisor = sample_decimal(rng, 1, divisor_places)
    while divisor == 0:
        divisor = sample_decimal(rng, 1, divisor_places)
    # Building the dividend from a chosen quotient keeps the division exact,
    # so the drill never teaches a truncated answer.
    quotient = sample_decimal(rng, digits, rng.randint(0, 2))
    dividend = divisor * quotient
    scale = 10**divisor_places
    whole_divisor = int(divisor * scale)
    scaled_dividend = dividend * scale
    dividend_places = decimal_places(dividend)
    quotient_places = decimal_places(quotient)
    # Long division runs on whole numbers, so shift the scaled dividend too
    # and take the same number of places back out of the quotient.
    shift = decimal_places(scaled_dividend)
    whole_dividend = int(scaled_dividend * 10**shift)
    working = [
        f"Multiply both numbers by {scale} so the divisor becomes a whole "
        f"number: {decimal_text(divisor, divisor_places)} x {scale} = "
        f"{whole_divisor}.",
        f"{decimal_text(dividend, dividend_places)} x {scale} = "
        f"{exact(scaled_dividend)}. Scaling both leaves the quotient unchanged.",
    ]
    if shift:
        working.append(
            f"Shift the dividend {plural(shift, 'more place')} to "
            f"{whole_dividend} and take those places back out at the end."
        )
    # Shifting the dividend already produced `shift` of the quotient's
    # decimal places as whole-number digits, so only the rest are carried
    # past the point.
    working.extend(
        long_division(whole_dividend, whole_divisor, max(quotient_places - shift, 0))
    )
    if shift:
        working.append(
            f"Move the point {plural(shift, 'place')} left to undo the shift."
        )
    return Drill(
        family="div_decimal",
        problem=(
            f"What is {decimal_text(dividend, dividend_places)} divided by "
            f"{decimal_text(divisor, divisor_places)}?"
        ),
        solution="\n".join(working),
        answer=exact(quotient),
        digit_order=digit_order,
        difficulty={"digits": digits, "places": divisor_places},
    )


def compare_decimal(rng: random.Random, digit_order: str, digits: int) -> Drill:
    left_places = rng.randint(1, 3)
    right_places = rng.randint(1, 3)
    left = sample_decimal(rng, digits, left_places)
    right = sample_decimal(rng, digits, right_places)
    width = max(left_places, right_places)
    symbol = "<" if left < right else (">" if left > right else "=")
    working = [
        f"Pad both to {plural(width, 'decimal place')}: "
        f"{decimal_text(left, width)} and {decimal_text(right, width)}.",
        "With the same number of places, comparing the digits left to right "
        "is the same as comparing whole numbers.",
        f"{int(left * 10**width)} against {int(right * 10**width)}.",
    ]
    return Drill(
        family="compare_decimal",
        problem=(
            f"Is {decimal_text(left, left_places)} less than, greater than, "
            f"or equal to {decimal_text(right, right_places)}? Answer with "
            "<, > or =."
        ),
        solution="\n".join(working),
        answer=symbol,
        digit_order=digit_order,
        difficulty={"digits": digits, "places": width},
    )


def round_decimal(rng: random.Random, digit_order: str, digits: int) -> Drill:
    places = rng.randint(2, 4)
    target = rng.randint(0, places - 1)
    value = sample_decimal(rng, digits, places)
    scaled = value * 10**target
    floor = math.floor(scaled)
    # Round half up, stated explicitly so the drill teaches exactly one rule.
    rounded = floor + 1 if scaled - floor >= Fraction(1, 2) else floor
    deciding = int(decimal_text(value, places).split(".")[1][target])
    target_text = (
        "the nearest whole number"
        if target == 0
        else plural(target, "decimal place")
    )
    working = [
        f"Keep {target_text}.",
        f"The deciding digit is the next one along: {deciding}.",
        f"{deciding} is "
        + (
            "5 or more, so round up."
            if deciding >= 5
            else "less than 5, so round down."
        ),
    ]
    return Drill(
        family="round_decimal",
        problem=f"Round {decimal_text(value, places)} to {target_text}.",
        solution="\n".join(working),
        answer=strip_decimal(decimal_text(Fraction(rounded, 10**target), target)),
        digit_order=digit_order,
        difficulty={"digits": digits, "places": target},
    )


# -- proportional families ------------------------------------------------


def percent_of(rng: random.Random, digit_order: str, digits: int) -> Drill:
    percent = rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90])
    amount = sample_integer(rng, digits) * 10 ** rng.randint(0, 1)
    value = Fraction(amount * percent, 100)
    working = [
        f"{percent}% means {percent} out of every 100, so multiply by "
        f"{percent} and divide by 100.",
        *partial_products(amount, percent, digit_order),
        f"{amount * percent} / 100 = {exact(value)}, moving the point two "
        "places left.",
    ]
    return Drill(
        family="percent_of",
        problem=f"What is {percent}% of {amount}?",
        solution="\n".join(working),
        answer=exact(value),
        digit_order=digit_order,
        difficulty={"digits": digits},
    )


def percent_change(rng: random.Random, digit_order: str, digits: int) -> Drill:
    percent = rng.choice([5, 10, 15, 20, 25, 40, 50])
    amount = sample_integer(rng, digits) * 10 ** rng.randint(0, 1)
    increase = rng.random() < 0.5
    delta = Fraction(amount * percent, 100)
    value = amount + delta if increase else amount - delta
    working = [
        f"First find {percent}% of {amount}: {amount} x {percent} = "
        f"{amount * percent}, and {amount * percent} / 100 = {exact(delta)}.",
        f"{'An increase adds' if increase else 'A decrease subtracts'} that "
        "much.",
        f"{amount} {'+' if increase else '-'} {exact(delta)} = {exact(value)}",
    ]
    return Drill(
        family="percent_change",
        problem=(
            f"A price of {amount} is {'increased' if increase else 'decreased'} "
            f"by {percent}%. What is the new price?"
        ),
        solution="\n".join(working),
        answer=exact(value),
        digit_order=digit_order,
        difficulty={"digits": digits},
    )


def fraction_add(rng: random.Random, digit_order: str, digits: int) -> Drill:
    bound = 10**digits
    left = Fraction(rng.randint(1, bound), rng.randint(2, max(3, bound)))
    right = Fraction(rng.randint(1, bound), rng.randint(2, max(3, bound)))
    common = math.lcm(left.denominator, right.denominator)
    left_top = left.numerator * (common // left.denominator)
    right_top = right.numerator * (common // right.denominator)
    working = [
        f"The lowest common denominator of {left.denominator} and "
        f"{right.denominator} is {common}.",
        f"{format_fraction(left)} = {left_top}/{common} and "
        f"{format_fraction(right)} = {right_top}/{common}.",
        f"{left_top}/{common} + {right_top}/{common} = "
        f"{left_top + right_top}/{common}.",
    ]
    divisor = math.gcd(left_top + right_top, common)
    if divisor > 1:
        working.append(
            f"Divide top and bottom by {divisor}: "
            f"{(left_top + right_top) // divisor}/{common // divisor}."
        )
    return Drill(
        family="fraction_add",
        problem=f"What is {format_fraction(left)} + {format_fraction(right)}?",
        solution="\n".join(working),
        answer=format_fraction(left + right),
        digit_order=digit_order,
        difficulty={"digits": digits},
    )


def fraction_mul(rng: random.Random, digit_order: str, digits: int) -> Drill:
    bound = 10**digits
    left = Fraction(rng.randint(1, bound), rng.randint(2, max(3, bound)))
    right = Fraction(rng.randint(1, bound), rng.randint(2, max(3, bound)))
    numerator = left.numerator * right.numerator
    denominator = left.denominator * right.denominator
    working = [
        f"Multiply the numerators: {left.numerator} x {right.numerator} = "
        f"{numerator}.",
        f"Multiply the denominators: {left.denominator} x "
        f"{right.denominator} = {denominator}.",
    ]
    divisor = math.gcd(numerator, denominator)
    if divisor > 1:
        working.append(
            f"{numerator}/{denominator} divides by {divisor} top and bottom: "
            f"{numerator // divisor}/{denominator // divisor}."
        )
    return Drill(
        family="fraction_mul",
        problem=f"What is {format_fraction(left)} * {format_fraction(right)}?",
        solution="\n".join(working),
        answer=format_fraction(left * right),
        digit_order=digit_order,
        difficulty={"digits": digits},
    )


# Conversions are exact by construction: each entry gives how many of the
# smaller unit make one of the larger, so no rounding enters the drill.
UNIT_SCALES = (
    ("millimetres", "metres", "metre", 1000),
    ("centimetres", "metres", "metre", 100),
    ("metres", "kilometres", "kilometre", 1000),
    ("grams", "kilograms", "kilogram", 1000),
    ("millilitres", "litres", "litre", 1000),
    ("seconds", "minutes", "minute", 60),
    ("minutes", "hours", "hour", 60),
    ("hours", "days", "day", 24),
    ("cents", "dollars", "dollar", 100),
)


def unit_convert(rng: random.Random, digit_order: str, digits: int) -> Drill:
    small, large, large_singular, scale = rng.choice(UNIT_SCALES)
    definition = f"There are {scale} {small} in one {large_singular}."
    if rng.random() < 0.5:
        amount = sample_integer(rng, digits + 1)
        # 125 seconds is 2.08333... minutes, and a drill must never teach a
        # truncated answer. Snapping the amount to a multiple of the scale's
        # non-decimal core -- 3 for 60 and 24, 1 for every power of ten here
        # -- makes every quotient terminate exactly.
        core = scale
        for factor in (2, 5):
            while core % factor == 0:
                core //= factor
        amount = max(amount - amount % core, core)
        value = Fraction(amount, scale)
        working = [
            definition,
            f"Going from {small} to {large} divides by {scale}.",
            f"{amount} / {scale} = {exact(value)}",
        ]
        problem = f"Convert {amount} {small} to {large}."
        answer = exact(value)
    else:
        amount = sample_integer(rng, digits)
        zeros = len(str(scale)) - 1
        working = [
            definition,
            f"Going from {large} to {small} multiplies by {scale}.",
            # Multiplying by a power of ten is a place-value shift, not a
            # long multiplication; saying so is the lesson.
            *(
                [
                    f"Multiplying by {scale} shifts every digit "
                    f"{plural(zeros, 'place')} left, which writes "
                    f"{plural(zeros, 'zero')} on the end.",
                    f"{amount} -> {amount * scale}",
                ]
                if scale == 10**zeros
                else partial_products(amount, scale, digit_order)
            ),
        ]
        problem = f"Convert {amount} {large} to {small}."
        answer = str(amount * scale)
    return Drill(
        family="unit_convert",
        problem=problem,
        solution="\n".join(working),
        answer=answer,
        digit_order=digit_order,
        difficulty={"digits": digits, "scale": scale},
    )


BUILDERS = {
    "add_integer": add_integer,
    "sub_integer": sub_integer,
    "mul_integer": mul_integer,
    "div_integer": div_integer,
    "add_decimal": add_decimal,
    "sub_decimal": sub_decimal,
    "mul_decimal": mul_decimal,
    "div_decimal": div_decimal,
    "compare_decimal": compare_decimal,
    "round_decimal": round_decimal,
    "percent_of": percent_of,
    "percent_change": percent_change,
    "fraction_add": fraction_add,
    "fraction_mul": fraction_mul,
    "unit_convert": unit_convert,
}

# Which digit orders each family has a genuinely causal method for. Long
# division and the rules-based families read one way only; declaring that
# here is what stops the generator from emitting text whose lines depend on
# lines below them.
FAMILY_ORDERS = {
    "add_integer": DIGIT_ORDERS,
    "sub_integer": DIGIT_ORDERS,
    "mul_integer": DIGIT_ORDERS,
    "div_integer": ("forward",),
    "add_decimal": DIGIT_ORDERS,
    "sub_decimal": DIGIT_ORDERS,
    "mul_decimal": DIGIT_ORDERS,
    "div_decimal": ("forward",),
    "compare_decimal": ("forward",),
    "round_decimal": ("forward",),
    "percent_of": DIGIT_ORDERS,
    "percent_change": ("forward",),
    "fraction_add": ("forward",),
    "fraction_mul": ("forward",),
    "unit_convert": DIGIT_ORDERS,
}

FAMILIES = tuple(BUILDERS)

assert set(FAMILY_ORDERS) == set(BUILDERS), "FAMILY_ORDERS and BUILDERS disagree"


# -- curriculum ----------------------------------------------------------


@dataclass(frozen=True)
class FamilySpec:
    """How often a family appears and over what digit range."""

    name: str
    weight: float
    min_digits: int = 1
    max_digits: int = 5

    def __post_init__(self) -> None:
        if self.name not in BUILDERS:
            raise ValueError(f"unknown drill family {self.name!r}")
        if self.weight <= 0:
            raise ValueError(f"{self.name} needs a positive weight")
        if not 1 <= self.min_digits <= self.max_digits:
            raise ValueError(
                f"{self.name} digit range {self.min_digits}-{self.max_digits} "
                "is not a valid ascending range starting at 1"
            )


# The default mix. Integer MADS carries the most weight because it is the
# capability everything else composes from, and because it is what the audited
# corpus was most short of. Digit ranges stay inside what fits fully worked in
# the model's context.
DEFAULT_CURRICULUM = (
    FamilySpec("add_integer", 12.0, 1, 6),
    FamilySpec("sub_integer", 12.0, 1, 6),
    FamilySpec("mul_integer", 12.0, 1, 4),
    FamilySpec("div_integer", 12.0, 2, 5),
    FamilySpec("add_decimal", 7.0, 1, 4),
    FamilySpec("sub_decimal", 7.0, 1, 4),
    FamilySpec("mul_decimal", 6.0, 1, 3),
    FamilySpec("div_decimal", 6.0, 1, 3),
    FamilySpec("compare_decimal", 4.0, 1, 4),
    FamilySpec("round_decimal", 4.0, 1, 4),
    FamilySpec("percent_of", 5.0, 1, 4),
    FamilySpec("percent_change", 4.0, 1, 4),
    FamilySpec("fraction_add", 4.0, 1, 2),
    FamilySpec("fraction_mul", 3.0, 1, 2),
    FamilySpec("unit_convert", 5.0, 1, 4),
)


def generate(
    *,
    seed: int,
    count: int,
    curriculum: Sequence[FamilySpec] = DEFAULT_CURRICULUM,
    excluded_keys: frozenset[bytes] = frozenset(),
    max_attempts_per_drill: int = 64,
) -> Iterator[Drill]:
    """Yield ``count`` distinct drills, deterministic in ``seed``.

    Duplicates and registry-excluded problems are skipped rather than replaced
    with a different family, so the realized family mix drifts slightly from
    the requested weights when a family's problem space is small -- one-digit
    addition has only a hundred problems. `drill_statistics` reports the mix
    actually produced; do not assume the requested weights.
    """
    rng = random.Random(seed)
    names = [spec.name for spec in curriculum]
    weights = [spec.weight for spec in curriculum]
    specs = {spec.name: spec for spec in curriculum}
    seen: set[bytes] = set()
    produced = 0
    while produced < count:
        for _ in range(max_attempts_per_drill):
            spec = specs[rng.choices(names, weights)[0]]
            drill = BUILDERS[spec.name](
                rng,
                rng.choice(FAMILY_ORDERS[spec.name]),
                rng.randint(spec.min_digits, spec.max_digits),
            )
            key = drill.key
            if key in seen or key in excluded_keys:
                continue
            seen.add(key)
            yield drill
            produced += 1
            break
        else:
            raise RuntimeError(
                f"could not find a new drill after {max_attempts_per_drill} "
                f"attempts at {produced:,} of {count:,}; the curriculum's "
                "problem space is too small for this count"
            )


def drill_statistics(drills: Sequence[Drill]) -> dict:
    """The realized mix, which is what a manifest should record."""
    families: dict[str, int] = {}
    orders: dict[str, int] = {}
    digits: dict[str, dict[int, int]] = {}
    characters = 0
    for drill in drills:
        families[drill.family] = families.get(drill.family, 0) + 1
        orders[drill.digit_order] = orders.get(drill.digit_order, 0) + 1
        bucket = digits.setdefault(drill.family, {})
        size = int(drill.difficulty.get("digits", 0))
        bucket[size] = bucket.get(size, 0) + 1
        characters += len(drill.document())
    return {
        "schema": DRILL_SCHEMA,
        "drills": len(drills),
        "families": dict(sorted(families.items())),
        "digit_orders": dict(sorted(orders.items())),
        "digits_by_family": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(digits.items())
        },
        "characters": characters,
        "mean_document_characters": characters / max(len(drills), 1),
    }
