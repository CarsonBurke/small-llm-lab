"""Vocabulary selection as an integer program over split-tree nodes.

Reference: `papers/tokenization_with_split_trees_2605.22705v1.pdf`, section 4.

Given split trees that are already fixed, choosing a vocabulary of size ``m``
is exactly: pick ``m`` candidate tokens so that recursive descent emits as few
tokens as possible in total. The paper writes that as an integer program and
observes the LP relaxation is nearly integral in practice, so the relaxation
plus a cheap rounding pass lands within a part per million of optimal.

Variables, following the paper's notation:

* ``x_i`` -- candidate token ``i`` is in the vocabulary.
* ``z_jk`` -- node ``k`` of split tree ``j`` is emitted under that vocabulary.

Constraints:

* Eq 6, the budget: ``sum_i x_i == m``.
* Eq 7, the byte alphabet is always present: ``x_i == 1`` for single bytes.
* Eq 8, coverage: on every root-to-leaf path exactly one node is emitted.
* Eq 9, linkage: a node can only be emitted if its token was selected.

Objective, Eq 5: ``min sum_j c_j sum_k z_jk``.

Link rows for single-byte tokens are omitted. Their ``x_i`` is fixed at 1, so
``z_jk <= x_i`` reduces to ``z_jk <= 1``, which the variable bound already
enforces. Because leaves are exactly the single bytes, that halves the row
count for free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from tokenization.split_tree import SplitTree, tree_tokens

INTEGRALITY_TOLERANCE = 1e-5


@dataclass(frozen=True)
class VocabularySolution:
    """Result of one vocabulary selection."""

    vocabulary: tuple[bytes, ...]
    lp_objective: float
    rounded_objective: float
    fractional_variables: int
    relaxation_gap: float
    status: str

    @property
    def relative_gap(self) -> float:
        if self.rounded_objective <= 0:
            return 0.0
        return self.relaxation_gap / self.rounded_objective


def _candidate_index(
    trees: Sequence[SplitTree], forced: Sequence[bytes]
) -> tuple[list[bytes], dict[bytes, int]]:
    """The set ``T`` of Eq 1: every tree node token, plus the byte alphabet."""
    order: list[bytes] = list(forced)
    index: dict[bytes, int] = {token: i for i, token in enumerate(order)}
    for tree in trees:
        for node_index in range(len(tree.nodes)):
            token = tree.segment(node_index)
            if token not in index:
                index[token] = len(order)
                order.append(token)
    return order, index


def select_vocabulary(
    trees: Sequence[SplitTree],
    counts: Sequence[int],
    *,
    size: int,
    forced: Sequence[bytes],
    solver_options: dict | None = None,
) -> VocabularySolution:
    """Solve the LP relaxation and round it to a vocabulary of exactly ``size``.

    ``counts[j]`` is the aggregated corpus count ``c_j`` of tree ``j``.
    """
    if len(trees) != len(counts):
        raise ValueError(
            f"{len(trees)} trees but {len(counts)} counts; they must correspond"
        )
    forced_unique = tuple(dict.fromkeys(forced))
    if size < len(forced_unique):
        raise ValueError(
            f"vocabulary size {size} cannot hold {len(forced_unique)} forced tokens"
        )

    tokens, token_index = _candidate_index(trees, forced_unique)
    n_tokens = len(tokens)
    if size > n_tokens:
        raise ValueError(
            f"vocabulary size {size} exceeds {n_tokens} candidate tokens; "
            "increase the number of split trees or lower the size"
        )

    # Flatten every node of every tree into one z index space.
    node_offset: list[int] = []
    total_nodes = 0
    for tree in trees:
        node_offset.append(total_nodes)
        total_nodes += len(tree.nodes)

    objective = np.zeros(n_tokens + total_nodes, dtype=np.float64)
    node_token = np.empty(total_nodes, dtype=np.int64)
    for tree_i, tree in enumerate(trees):
        base = node_offset[tree_i]
        weight = float(counts[tree_i])
        for node_i in range(len(tree.nodes)):
            flat = base + node_i
            objective[n_tokens + flat] = weight
            node_token[flat] = token_index[tree.segment(node_i)]

    # Eq 8: one emitted node per root-to-leaf path.
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    row = 0
    for tree_i, tree in enumerate(trees):
        base = node_offset[tree_i]
        for leaf in tree.leaves():
            eq_rows.append(row)
            eq_cols.append(n_tokens + base + leaf)
            for ancestor in tree.ancestors(leaf):
                eq_rows.append(row)
                eq_cols.append(n_tokens + base + ancestor)
            row += 1
    coverage_rows = row

    # Eq 6: the budget.
    for token_i in range(n_tokens):
        eq_rows.append(row)
        eq_cols.append(token_i)
    row += 1

    a_eq = sparse.csr_array(
        (np.ones(len(eq_rows), dtype=np.float64), (eq_rows, eq_cols)),
        shape=(row, n_tokens + total_nodes),
    )
    b_eq = np.ones(row, dtype=np.float64)
    b_eq[coverage_rows] = float(size)

    # Eq 9: z_jk - x_i <= 0, skipped where x_i is pinned to one.
    forced_ids = {token_index[token] for token in forced_unique}
    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []
    link_row = 0
    for flat in range(total_nodes):
        token_i = int(node_token[flat])
        if token_i in forced_ids:
            continue
        ub_rows.extend((link_row, link_row))
        ub_cols.extend((n_tokens + flat, token_i))
        ub_data.extend((1.0, -1.0))
        link_row += 1
    a_ub = sparse.csr_array(
        (np.asarray(ub_data), (ub_rows, ub_cols)),
        shape=(link_row, n_tokens + total_nodes),
    )
    b_ub = np.zeros(link_row, dtype=np.float64)

    lower = np.zeros(n_tokens + total_nodes, dtype=np.float64)
    upper = np.ones(n_tokens + total_nodes, dtype=np.float64)
    for token_i in forced_ids:
        lower[token_i] = 1.0

    result = linprog(
        objective,
        A_ub=a_ub if link_row else None,
        b_ub=b_ub if link_row else None,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=np.stack([lower, upper], axis=1),
        method="highs",
        options=solver_options or {},
    )
    if not result.success:
        raise RuntimeError(f"vocabulary LP did not solve: {result.message}")

    x_star = result.x[:n_tokens]
    z_star = result.x[n_tokens:]
    vocabulary = _round_vocabulary(
        x_star=x_star,
        z_star=z_star,
        node_token=node_token,
        objective_weights=objective[n_tokens:],
        tokens=tokens,
        size=size,
        forced_ids=forced_ids,
    )

    rounded = _inference_token_total(trees, counts, frozenset(vocabulary))
    fractional = int(
        np.count_nonzero(
            (x_star > INTEGRALITY_TOLERANCE)
            & (x_star < 1.0 - INTEGRALITY_TOLERANCE)
        )
    )
    return VocabularySolution(
        vocabulary=vocabulary,
        lp_objective=float(result.fun),
        rounded_objective=float(rounded),
        fractional_variables=fractional,
        relaxation_gap=float(rounded - result.fun),
        status=str(result.message),
    )


def _round_vocabulary(
    *,
    x_star: np.ndarray,
    z_star: np.ndarray,
    node_token: np.ndarray,
    objective_weights: np.ndarray,
    tokens: Sequence[bytes],
    size: int,
    forced_ids: set[int],
) -> tuple[bytes, ...]:
    """The paper's section 4.3 heuristic.

    Tokens the relaxation already decided are kept. The remaining budget goes
    to the fractional tokens with the largest ``C_i*`` of Eq 15, the objective
    mass their nodes carry, because rounding one of those up saves at least
    ``c_j`` per node against splitting it into two tokens.
    """
    selected = set(np.flatnonzero(x_star >= 1.0 - INTEGRALITY_TOLERANCE).tolist())
    selected |= forced_ids
    if len(selected) > size:
        raise RuntimeError(
            f"LP pinned {len(selected)} tokens to one, above the budget {size}"
        )
    fractional = [
        i
        for i in np.flatnonzero(x_star > INTEGRALITY_TOLERANCE).tolist()
        if i not in selected
    ]
    if fractional:
        contribution = np.zeros(len(tokens), dtype=np.float64)
        np.add.at(contribution, node_token, objective_weights * z_star)
        fractional.sort(key=lambda i: (-contribution[i], tokens[i]))
    remaining = size - len(selected)
    for token_i in fractional[:remaining]:
        selected.add(token_i)
    if len(selected) < size:
        # The relaxation left fewer nonzero tokens than the budget. Fill
        # deterministically so the vocabulary is exactly the requested size.
        for token_i in range(len(tokens)):
            if len(selected) == size:
                break
            selected.add(token_i)
    return tuple(tokens[i] for i in sorted(selected))


def _inference_token_total(
    trees: Sequence[SplitTree],
    counts: Sequence[int],
    vocabulary: frozenset[bytes],
) -> int:
    """Token count actually produced by inference, not the LP's estimate."""
    total = 0
    for tree, count in zip(trees, counts, strict=True):
        total += len(tree_tokens(tree, vocabulary)) * count
    return total
