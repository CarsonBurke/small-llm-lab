"""Triadic Suffix Tokenization: digits carry explicit magnitude.

Reference: `papers/triadic_suffix_tokenization_2604.11582v3.pdf`. Digits are
partitioned into fixed-size groups and every group is annotated with its order
of magnitude, so a model reads place value directly instead of inferring it
from sequence position. The paper's own experimental validation is deferred to
future work, so every claim it makes is a hypothesis this repository must
measure; `postraining/arithmetic_probe.py` is the measurement.

Two deliberate deviations from the paper, both required for corpus use:

* **Variable-length boundary groups instead of zero padding.** The paper pads
  the leading integer group from the left and the trailing fractional group
  from the right, which makes `0.1`, `0.10`, and `0.100` share one token
  sequence. That is lossy, and a lossy tokenizer silently rewrites the corpus
  it is measuring. Here a group's magnitude is `(depth, length)` rather than
  `depth` alone, which keeps every distinct surface form distinct and makes
  encode/decode exactly invertible. The paper's canonical padding remains
  available as `leading_zero_padding=True` for faithful reproduction, and is
  rejected by the corpus builders.
* **No sign in the numeric span.** The paper's `ExtractPrefix` consumes a sign.
  Deciding whether the `-` in `a-5` is a sign requires context, and getting it
  wrong breaks round-tripping, so signs stay on the text path where they are a
  single stable token either way.

A numeric token is the pair `(digits, power)` where `power` is the exponent of
the group's least significant digit, so the group's value is exactly
`int(digits) * 10**power`. This one family covers both the integer suffixes
(`k`, `m`, `b`, `t`, `q`) and the replicated fractional markers (`p`, `pp`,
...) that the paper defines separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NUMERIC_SPAN_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")

# Human-readable magnitude names, used only for debugging and test messages.
# The model sees token identities, never these strings.
_INTEGER_SUFFIX_NAMES = {0: "", 3: "k", 6: "m", 9: "b", 12: "t", 15: "q"}


class NumericOverflowError(ValueError):
    """A span has more digits than the scheme's magnitude range covers."""


@dataclass(frozen=True)
class NumericScheme:
    """Deterministic mapping between digit strings and magnitude-aware groups.

    ``group_size`` is the paper's ``N``. ``N=3`` is the triadic scheme of the
    title; ``N=1`` is the per-digit variant the paper hypothesises is optimal
    for small language models, and is also the only setting whose groups are
    all length one, which makes the padding question moot.

    ``compound`` selects the paper's Option B (one token per digit-group and
    magnitude pair) over Option A (a digit-group token followed by a separate
    magnitude marker token). Option B is shorter; Option A is far smaller in
    vocabulary. For ``N=1`` Option B costs only
    ``10 * (max_int_digits + max_frac_digits)`` tokens.
    """

    group_size: int = 1
    compound: bool = True
    max_int_digits: int = 19
    max_frac_digits: int = 15
    leading_zero_padding: bool = False

    def __post_init__(self) -> None:
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if self.max_int_digits < 1:
            raise ValueError(
                f"max_int_digits must be >= 1, got {self.max_int_digits}"
            )
        if self.max_frac_digits < 1:
            raise ValueError(
                f"max_frac_digits must be >= 1, got {self.max_frac_digits}"
            )

    # -- magnitude ranges ------------------------------------------------

    @property
    def integer_powers(self) -> tuple[int, ...]:
        """Powers of the least significant digit of each integer group."""
        highest = self.group_size * ((self.max_int_digits - 1) // self.group_size)
        return tuple(range(0, highest + 1, self.group_size))

    @property
    def fraction_powers(self) -> tuple[int, ...]:
        return tuple(range(-1, -self.max_frac_digits - 1, -1))

    def accepts(self, integer_digits: str, fraction_digits: str) -> bool:
        """Whether a span fits the magnitude range.

        Spans that do not fit are left on the text path by the pre-tokenizer
        rather than escaped, so encoding stays deterministic and lossless
        without an escape-token mechanism.
        """
        return (
            len(integer_digits) <= self.max_int_digits
            and len(fraction_digits) <= self.max_frac_digits
        )

    # -- group enumeration -----------------------------------------------

    def group_vocabulary(self) -> tuple[tuple[str, int], ...]:
        """Every ``(digits, power)`` group the scheme can emit, in a stable order.

        Interior groups always have exactly ``group_size`` digits. Only the
        most significant integer group and the least significant fractional
        group may be shorter, which is what keeps the mapping invertible.
        """
        size = self.group_size
        lengths = (size,) if self.leading_zero_padding else tuple(range(1, size + 1))
        groups: list[tuple[str, int]] = []
        for power in self.integer_powers:
            for length in lengths:
                # A group of `length` digits sits at `power` only when a number
                # of exactly `power + length` integer digits exists.
                if power + length > self.max_int_digits:
                    continue
                for value in range(10**length):
                    groups.append((str(value).zfill(length), power))
        for depth in range(self._max_fraction_depth() + 1):
            for length in lengths:
                if depth * size + length > self.max_frac_digits:
                    continue
                groups.extend(
                    (str(value).zfill(length), -(depth * size + length))
                    for value in range(10**length)
                )
        if len(set(groups)) != len(groups):
            raise AssertionError("group vocabulary enumerated a duplicate pair")
        return tuple(groups)

    def marker_vocabulary(self) -> tuple[int, ...]:
        """Magnitude markers for Option A, one per representable power."""
        return tuple(self.integer_powers) + tuple(self.fraction_powers)

    def _max_fraction_depth(self) -> int:
        return (self.max_frac_digits - 1) // self.group_size

    # -- encoding --------------------------------------------------------

    def split_span(self, span: str) -> tuple[str, str]:
        if "." in span:
            integer, fraction = span.split(".", 1)
        else:
            integer, fraction = span, ""
        return integer, fraction

    def groups_for_span(self, span: str) -> list[tuple[str, int]]:
        """Decompose a numeric span into ``(digits, power)`` groups.

        Integer digits are grouped right to left and fractional digits left to
        right, exactly as Algorithm 1 specifies. The decimal point is not a
        group; the caller emits it between the two runs.
        """
        integer, fraction = self.split_span(span)
        if not integer or not integer.isdigit():
            raise ValueError(f"span has no integer digits: {span!r}")
        if fraction and not fraction.isdigit():
            raise ValueError(f"span has a non-digit fraction: {span!r}")
        if not self.accepts(integer, fraction):
            raise NumericOverflowError(
                f"span {span!r} exceeds the scheme range "
                f"({self.max_int_digits} integer, {self.max_frac_digits} "
                f"fractional digits)"
            )

        size = self.group_size
        groups: list[tuple[str, int]] = []

        if self.leading_zero_padding:
            padded = integer.rjust(
                size * -(-len(integer) // size), "0"
            )
        else:
            padded = integer
        # Right to left, so the boundary (short) group lands at the front.
        cuts: list[str] = []
        index = len(padded)
        while index > 0:
            start = max(0, index - size)
            cuts.append(padded[start:index])
            index = start
        for offset, digits in enumerate(cuts):
            groups.append((digits, offset * size))
        groups.reverse()

        if fraction:
            if self.leading_zero_padding:
                fraction_padded = fraction.ljust(
                    size * -(-len(fraction) // size), "0"
                )
            else:
                fraction_padded = fraction
            for depth, start in enumerate(range(0, len(fraction_padded), size)):
                digits = fraction_padded[start : start + size]
                power = -(depth * size + len(digits))
                groups.append((digits, power))
        return groups

    # -- decoding --------------------------------------------------------

    def span_for_groups(self, groups: list[tuple[str, int]]) -> str:
        """Invert :meth:`groups_for_span`.

        Raises on any group sequence the encoder could not have produced, so a
        corrupted stream fails loudly rather than decoding to a plausible but
        different number.
        """
        if not groups:
            raise ValueError("cannot decode an empty group sequence")
        integer_groups = [group for group in groups if group[1] >= 0]
        fraction_groups = [group for group in groups if group[1] < 0]
        if not integer_groups:
            raise ValueError("group sequence has no integer part")
        if integer_groups != groups[: len(integer_groups)]:
            raise ValueError("integer groups must precede fractional groups")

        size = self.group_size
        expected = integer_groups[0][1]
        for position, (digits, power) in enumerate(integer_groups):
            if power != expected:
                raise ValueError(
                    f"integer group {position} has power {power}, expected "
                    f"{expected}"
                )
            if position > 0 and len(digits) != size:
                raise ValueError(
                    f"interior integer group {digits!r} must have {size} digits"
                )
            if len(digits) > size:
                raise ValueError(f"integer group {digits!r} exceeds {size} digits")
            expected -= size
        if expected != -size:
            raise ValueError("integer groups do not reach power zero")
        integer = "".join(digits for digits, _ in integer_groups)

        fraction = ""
        if fraction_groups:
            expected_depth = 0
            for position, (digits, power) in enumerate(fraction_groups):
                if power != -(expected_depth * size + len(digits)):
                    raise ValueError(
                        f"fractional group {digits!r} has inconsistent power "
                        f"{power} at depth {expected_depth}"
                    )
                is_last = position == len(fraction_groups) - 1
                if not is_last and len(digits) != size:
                    raise ValueError(
                        f"interior fractional group {digits!r} must have "
                        f"{size} digits"
                    )
                expected_depth += 1
            fraction = "".join(digits for digits, _ in fraction_groups)
        return f"{integer}.{fraction}" if fraction_groups else integer

    # -- presentation ----------------------------------------------------

    def render_group(self, digits: str, power: int) -> str:
        """Debug rendering, e.g. ``123k``. Never used for token identity."""
        if self.group_size == 3:
            if power in _INTEGER_SUFFIX_NAMES:
                return f"{digits}{_INTEGER_SUFFIX_NAMES[power]}"
            if power < 0:
                depth = (-power - len(digits)) // 3 + 1
                return f"{digits}{'p' * depth}"
        return f"{digits}e{power}" if power else digits
