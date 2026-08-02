"""On-policy self-distillation (OPSD) post-training."""

from postraining.opsd.schemas import (
    OPSD_CHECKPOINT_SCHEMA,
    OPSD_OBJECTIVE_SCHEMA,
)

__all__ = ["OPSD_CHECKPOINT_SCHEMA", "OPSD_OBJECTIVE_SCHEMA"]
