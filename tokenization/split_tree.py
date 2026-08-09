"""Split-tree construction and inference (ToaST).

Reference: `papers/tokenization_with_split_trees_2605.22705v1.pdf`, sections 2
and 3, and the reference implementation in its Listing 1.

A split tree is built from byte n-gram counts alone, before any vocabulary
exists. Each pretoken is recursively cut into the two nonempty parts
maximising ``min(count(left), count(right))``, ties broken leftmost, down to
single bytes. Inference then descends the tree and emits the first node found
in the vocabulary on each root-to-leaf path.

The consequence the integer program in `vocab_lp` depends on: because trees are
independent of the vocabulary, changing the vocabulary reshapes no tree and
cascades through no merge list. Removing a token simply lets the descent
continue past that node.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# The listing scores a missing n-gram as a large negative, so any split with a
# part outside the count dictionary loses to any split with both parts inside.
MISSING_COUNT = -10


@dataclass(frozen=True, slots=True)
class SplitNode:
    """One node of a split tree, as a span of the pretoken it belongs to."""

    start: int
    end: int
    left: int
    right: int
    parent: int

    @property
    def is_leaf(self) -> bool:
        return self.left < 0


@dataclass(frozen=True, slots=True)
class SplitTree:
    """A full binary tree over ``pretoken``; node 0 is the root.

    ``2 * len(pretoken) - 1`` nodes, one leaf per byte, matching the model size
    accounting in the paper's Appendix B.2.
    """

    pretoken: bytes
    nodes: tuple[SplitNode, ...]

    def segment(self, index: int) -> bytes:
        node = self.nodes[index]
        return self.pretoken[node.start : node.end]

    def leaves(self) -> tuple[int, ...]:
        return tuple(i for i, node in enumerate(self.nodes) if node.is_leaf)

    def ancestors(self, index: int) -> tuple[int, ...]:
        chain: list[int] = []
        parent = self.nodes[index].parent
        while parent >= 0:
            chain.append(parent)
            parent = self.nodes[parent].parent
        return tuple(chain)


def most_known_split(pretoken: bytes, counts: Mapping[bytes, int]) -> int:
    """Fallback split: after the longest prefix present in the counts.

    The listing returns ``i - 1`` for the first absent prefix, which can be
    zero (an empty left part) or ``None`` (every prefix present). Both would
    recurse forever on a span that never shrinks, so the result is clamped into
    the range of splits that actually make progress.
    """
    longest = 0
    for index in range(1, len(pretoken)):
        if pretoken[:index] not in counts:
            break
        longest = index
    return min(max(longest, 1), len(pretoken) - 1)


def best_split(pretoken: bytes, counts: Mapping[bytes, int]) -> int:
    """The split maximising ``min(count(left), count(right))``, ties leftmost.

    A strict ``>`` comparison keeps the leftmost of equal scores, as specified.
    When no split has both parts in the dictionary every score is
    ``MISSING_COUNT`` and the fallback applies.
    """
    if len(pretoken) < 2:
        raise ValueError(f"cannot split {pretoken!r}: fewer than two bytes")
    best_score = -1
    best_index = -1
    for index in range(1, len(pretoken)):
        left = counts.get(pretoken[:index], MISSING_COUNT)
        right = counts.get(pretoken[index:], MISSING_COUNT)
        score = left if left < right else right
        if score > best_score:
            best_score = score
            best_index = index
    if best_index < 0:
        return most_known_split(pretoken, counts)
    return best_index


def build_split_tree(pretoken: bytes, counts: Mapping[bytes, int]) -> SplitTree:
    """Recursively split ``pretoken`` into a full binary tree of byte spans."""
    if not pretoken:
        raise ValueError("cannot build a split tree for an empty pretoken")
    nodes: list[SplitNode] = []

    def emit(start: int, end: int, parent: int) -> int:
        index = len(nodes)
        nodes.append(SplitNode(start=start, end=end, left=-1, right=-1, parent=parent))
        if end - start > 1:
            cut = start + best_split(pretoken[start:end], counts)
            left = emit(start, cut, index)
            right = emit(cut, end, index)
            nodes[index] = SplitNode(
                start=start, end=end, left=left, right=right, parent=parent
            )
        return index

    emit(0, len(pretoken), -1)
    return SplitTree(pretoken=pretoken, nodes=tuple(nodes))


def tree_tokens(tree: SplitTree, vocabulary: frozenset[bytes]) -> list[bytes]:
    """Split-tree inference: first in-vocabulary node on each path.

    Single bytes are always emitted, matching the listing's ``len(s) == 1``
    guard, so the procedure is total for any vocabulary containing the byte
    alphabet.
    """
    tokens: list[bytes] = []
    stack = [0]
    while stack:
        index = stack.pop()
        node = tree.nodes[index]
        segment = tree.pretoken[node.start : node.end]
        if node.is_leaf or segment in vocabulary:
            tokens.append(segment)
            continue
        stack.append(node.right)
        stack.append(node.left)
    return tokens


def candidate_tokens(tree: SplitTree) -> set[bytes]:
    """Every span appearing as a node, i.e. every token the tree could emit."""
    return {tree.segment(index) for index in range(len(tree.nodes))}


def count_ngrams(
    pretoken_counts: Iterable[tuple[bytes, int]],
    *,
    min_count: int,
    max_length: int,
) -> dict[bytes, int]:
    """Byte n-gram counts computed within pretoken boundaries.

    Counting inside pretokens is what stops split trees from proposing tokens
    that straddle a word boundary. ``min_count`` is the paper's ``c_min``.
    """
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}")
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}")
    totals: dict[bytes, int] = {}
    for pretoken, count in pretoken_counts:
        length = len(pretoken)
        limit = min(length, max_length)
        for size in range(1, limit + 1):
            for start in range(length - size + 1):
                gram = pretoken[start : start + size]
                totals[gram] = totals.get(gram, 0) + count
    return {gram: total for gram, total in totals.items() if total >= min_count}
