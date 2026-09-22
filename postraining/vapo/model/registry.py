"""Model-family registry for post-training RL.

A family key is a checkpoint invariant: it is written into every derived
checkpoint and compared on resume, so keys are never renamed in place. Add a
new key instead, the same way prompt schemas are versioned.
"""

from __future__ import annotations

from collections.abc import Callable

from postraining.vapo.model.protocols import ModelFamily

_FAMILIES: dict[str, Callable[[], ModelFamily]] = {}


def register_family(key: str, factory: Callable[[], ModelFamily]) -> None:
    """Bind ``key`` to a zero-argument family factory.

    Registration is deliberately strict: a duplicate key means two builders
    disagree about what a checkpoint's recorded family means.
    """
    if not key or key != key.strip() or key.lower() != key:
        raise ValueError(f"family keys are lowercase and unpadded: {key!r}")
    if key in _FAMILIES:
        raise ValueError(f"model family already registered: {key}")
    _FAMILIES[key] = factory


def get_family(key: str) -> ModelFamily:
    """Instantiate the family registered under ``key``."""
    _load_builtin_families()
    try:
        factory = _FAMILIES[key]
    except KeyError:
        known = ", ".join(sorted(_FAMILIES)) or "none"
        raise KeyError(f"unknown model family {key!r}; registered: {known}") from None
    family = factory()
    if family.key != key:
        raise RuntimeError(f"family {key!r} reports mismatched key {family.key!r}")
    return family


def list_families() -> tuple[str, ...]:
    _load_builtin_families()
    return tuple(sorted(_FAMILIES))


_LOADED = False


def _load_builtin_families() -> None:
    """Import the shipped family modules exactly once.

    Importing on demand keeps ``transformers`` off the import path of a
    nano-only process, and vice versa.
    """
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from postraining.vapo.model import hf as _hf  # noqa: F401
    from postraining.vapo.model import nano as _nano  # noqa: F401


__all__ = ["get_family", "list_families", "register_family"]
