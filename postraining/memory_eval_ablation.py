"""Evaluation-only ablations of a trained memory carry's read.

They ask whether a memory policy's bench accuracy runs through its memory,
and whether through the trajectory's own content or only through the read's
extra input (capacity and bias):

- ``zero_read``: the actor's read-out ``W_o`` (``combiner.reader.out``) is
  zeroed for the evaluation. Every decode input is then the token embedding
  alone, so the policy is the token-only policy on the RL-trained trunk.
- ``foreign_state``: every decode read sees another trajectory's memory
  state -- a row of a different prompt in the same decode batch, at the same
  tick -- through the row's own query (its own token and producing belief).
  Each row still writes its own state, which in turn only its reader sees.
  The read keeps its statistics and magnitude and loses its
  trajectory-specific content.
- ``foreign_sample``: the same, but the donor is another sample of the
  SAME prompt. Against ``foreign_state`` it separates dependence on the
  prompt's content (shared with the donor here) from dependence on this
  trajectory's own content.

The foreign donor is fixed per rollout on ORIGINAL rows (prompt-major,
``prompt_repeats`` samples per prompt). Under ``foreign_state`` row ``i``
reads row ``(i + prompt_repeats) mod rows``, the same sample index of the
next prompt; under ``foreign_sample`` it reads the next sample of its own
prompt, cyclically within the group. Both are bijections; the first always
crosses prompts when a rollout holds at least two, the second always stays
in the prompt and changes sample when it has at least two samples. Decode compacts rows and keeps stepping finished ones, so each
rollout keeps a side buffer of every original row's state indexed by
original row: an active row's write lands there after its step, and a
finished row's entry stays at the state it ended with. A row whose donor has
finished therefore reads the donor's final state.

Neither ablation can reach training: the flag is legal only with
``--bench-only``, the zeroed read-out is restored bitwise when the
evaluation ends, and the foreign reads exist only inside an evaluation
decode.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch import Tensor, nn

MEMORY_EVAL_ABLATIONS = ("none", "zero_read", "foreign_state", "foreign_sample")
# The ablations a decode applies by reading another row's memory.
FOREIGN_READ_ABLATIONS = ("foreign_state", "foreign_sample")
MEMORY_EVAL_ABLATION_SCHEMA = "memory_eval_ablation/v2"


def foreign_donor_rows(
    prompts: int,
    repeats: int,
    ablation: str = "foreign_state",
    device: torch.device | None = None,
) -> Tensor:
    """Each original row's donor under ``ablation``.

    Rows are prompt-major, ``repeats`` per prompt. ``foreign_state``: the
    same sample of the next prompt (a cyclic shift by one prompt).
    ``foreign_sample``: the next sample of the same prompt (a cyclic shift
    inside each group). Both are bijections of ``prompts * repeats`` rows.
    """
    if repeats < 1 or prompts < 1:
        raise ValueError("prompt and repeat counts must be positive")
    rows = torch.arange(prompts * repeats, device=device)
    if ablation == "foreign_state":
        if prompts < 2:
            raise ValueError(
                "foreign_state reads need at least two prompts in one decode "
                "batch: a donor must belong to a different prompt"
            )
        return (rows + repeats) % (prompts * repeats)
    if ablation == "foreign_sample":
        if repeats < 2:
            raise ValueError(
                "foreign_sample reads need at least two samples per prompt: "
                "a donor must be a different sample of the same prompt"
            )
        return rows - rows % repeats + (rows + 1) % repeats
    raise ValueError(f"{ablation!r} is not a foreign-read ablation")


def foreign_read_state(
    states: Tensor, donors: Tensor, live_rows: Tensor
) -> Tensor:
    """The donor's state for each live row.

    ``states`` and ``donors`` are indexed by original row; ``live_rows``
    maps the decode's current (compacted) rows to original rows.
    """
    return states.index_select(0, donors.index_select(0, live_rows))


def record_foreign_state(
    states: Tensor, live_rows: Tensor, state: Tensor, active: Tensor
) -> None:
    """Store the live rows' post-write states at their original rows.

    Only ``active`` rows record: a finished row keeps stepping on filler
    input, and its entry must stay at the state it ended with.
    """
    kept = states.index_select(0, live_rows)
    states.index_put_(
        (live_rows,), torch.where(active[:, None, None, None], state, kept)
    )


@contextlib.contextmanager
def zeroed_memory_read(wrapper: nn.Module) -> Iterator[None]:
    """Zero the actor's memory read-out; restore it bitwise on exit.

    Refuses a module in training mode: this is an evaluation ablation.
    """
    if not getattr(wrapper, "memory_carry", False):
        raise ValueError("zero_read ablates a memory-carry policy's read")
    if wrapper.training:
        raise RuntimeError("zero_read is an evaluation ablation; the wrapper is training")
    weight = wrapper.combiner.reader.out.weight
    saved = weight.detach().clone()
    with torch.no_grad():
        weight.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            weight.copy_(saved)


def memory_eval_ablation(wrapper: nn.Module, mode: str):
    """The weight-level context for ``mode``.

    ``foreign_state`` is applied inside the decode (the arena or the eager
    rollout), so its weight context is empty.
    """
    if mode not in MEMORY_EVAL_ABLATIONS:
        raise ValueError(f"unknown memory evaluation ablation {mode!r}")
    if mode == "zero_read":
        return zeroed_memory_read(wrapper)
    return contextlib.nullcontext()
