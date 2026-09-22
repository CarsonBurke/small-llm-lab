"""Model-agnostic model bindings for post-training RL.

``protocols`` declares what the algorithm needs from a model, ``registry``
maps a checkpoint-bound family key to a builder, and ``hf``/``nano`` supply
the two shipped families. ``lora`` and ``readout`` hold the model-side
primitives both families reuse.
"""

from postraining.vapo.model.protocols import (
    Capability,
    ModelFamily,
    Readout,
    RolloutEngine,
    TrunkAdapter,
    TrunkGeometry,
)
from postraining.vapo.model.registry import get_family, list_families, register_family

__all__ = [
    "Capability",
    "ModelFamily",
    "Readout",
    "RolloutEngine",
    "TrunkAdapter",
    "TrunkGeometry",
    "get_family",
    "list_families",
    "register_family",
]
