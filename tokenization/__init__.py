"""Combined ToaST split-tree text tokenization and TST numeric tokenization.

`toast` implements Tokenization with Split Trees (Schmidt et al., 2026,
`papers/tokenization_with_split_trees_2605.22705v1.pdf`): byte n-gram counts
build a vocabulary-independent binary split tree per pretoken, and an integer
program selects the vocabulary that minimizes total token count under recursive
split-tree inference.

`tst` implements Triadic Suffix Tokenization (Chetverina, 2026,
`papers/triadic_suffix_tokenization_2604.11582v3.pdf`): digits are grouped and
each group carries an explicit magnitude, so place value is a token property
rather than something the model must infer from position.

The two compose at the pre-tokenizer. Numeric spans are routed to TST and never
enter the n-gram statistics, so the split-tree vocabulary budget is spent
entirely on text. See `tokenization/README.md`.
"""

from __future__ import annotations

from tokenization.spec import TokenizerSpec
from tokenization.tokenizer import SplitTreeNumericTokenizer

__all__ = ["SplitTreeNumericTokenizer", "TokenizerSpec"]
