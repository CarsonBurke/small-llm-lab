"""Exact full-BPTT loss; scheduling segments never delay or detach writes."""
from pretraining.nanogpt_mini.recurrent_slots_runtime import RecurrentLoss


class LatentCarryLoss(RecurrentLoss):
    def __call__(self, inputs, targets, *, diagnostics=False):
        loss = super().__call__(inputs, targets, diagnostics=False)
        # Slot-spread diagnostics have no meaning for an associative matrix.
        return (loss,) if diagnostics else loss


from pretraining.nanogpt_mini.chunk_memory_runtime import CompiledFullLoss


class CausalControlLoss(CompiledFullLoss):
    def _loss(self, inputs, targets, diagnostics=False):
        loss = super()._loss(inputs, targets, diagnostics=False)
        return (loss,) if diagnostics else loss
