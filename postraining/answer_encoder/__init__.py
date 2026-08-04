"""LeJEPA-style answer embeddings and similarity rewards."""

from postraining.answer_encoder.model import (
    DEFAULT_REWARD_TEMPERATURE,
    AnswerEncoderConfig,
    AnswerSimilarityReward,
    GlobalLatentObjective,
    GlobalLatentPredictor,
    LeJEPAObjective,
    LeJEPAObjectiveConfig,
    TextAnswerEncoder,
    cosine_kernel_reward,
)

__all__ = [
    "DEFAULT_REWARD_TEMPERATURE",
    "AnswerEncoderConfig",
    "AnswerSimilarityReward",
    "GlobalLatentObjective",
    "GlobalLatentPredictor",
    "LeJEPAObjective",
    "LeJEPAObjectiveConfig",
    "TextAnswerEncoder",
    "cosine_kernel_reward",
]
