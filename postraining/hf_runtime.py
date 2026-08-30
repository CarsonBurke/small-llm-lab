"""Shared Hugging Face runtime guards for text-only model paths."""

from __future__ import annotations


def prepare_text_only_transformers_runtime() -> None:
    """Prevent an unrelated, ABI-incompatible torchvision import."""
    import transformers.utils
    import transformers.utils.import_utils as import_utils

    setattr(import_utils, "is_torchvision_available", lambda: False)
    setattr(transformers.utils, "is_torchvision_available", lambda: False)
