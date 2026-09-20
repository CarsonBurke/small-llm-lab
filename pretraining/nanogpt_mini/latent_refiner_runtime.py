"""Identical bounded head execution for all latent-refiner source controls."""
from pretraining.nanogpt_mini.recurrent_slots_runtime import RecurrentLoss


class LatentRefinerLoss(RecurrentLoss):
    """Checkpoint the vocabulary head with one common source-arm schedule.

    All three source arms share the same split, reduction, and recomputation
    schedule. The full B256 head exceeded GPU capacity without refiner
    checkpoints; bounded head storage allows selective refiner recomputation.
    """

    def __call__(self, inputs, targets, *, diagnostics=False):
        loss = super().__call__(inputs, targets, diagnostics=False)
        return (loss,) if diagnostics else loss
