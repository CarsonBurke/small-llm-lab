"""Inference wiring for the standalone revisable DiffusionGemma sampler."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .diffusion_gemma import (
    EntropyBudgetSamplerConfig,
    RevisableEntropySamplerState,
    initialize_revisable_entropy_sampler,
    revisable_entropy_sampler_step,
)
from .diffusion_gemma_model import DiffusionGemmaModel


@dataclass(frozen=True)
class DiffusionGemmaGeneration:
    ids: Tensor
    valid: Tensor
    state: RevisableEntropySamplerState
    denoising_forwards: int
    clean_commit_forwards: int

    @property
    def model_forwards(self) -> int:
        return self.denoising_forwards + self.clean_commit_forwards

    @property
    def requires_clean_commit(self) -> bool:
        """The returned canvas is final but is not appended to a prefix cache."""

        return self.clean_commit_forwards == 0


@dataclass(frozen=True)
class DiffusionGemmaContinuation:
    """End-to-end clean-prompt continuation with explicit work accounting."""

    ids: Tensor
    valid: Tensor
    prompt_lengths: Tensor
    lengths: Tensor
    finished: Tensor
    canvases: int
    denoising_forwards: int
    clean_prefill_forwards: int
    clean_commit_forwards: int
    cache_backed: bool = False

    @property
    def model_forwards(self) -> int:
        return (
            self.denoising_forwards
            + self.clean_prefill_forwards
            + self.clean_commit_forwards
        )

    @property
    def generated_valid(self) -> Tensor:
        columns = torch.arange(self.ids.shape[1], device=self.ids.device)
        return self.valid & (columns[None] >= self.prompt_lengths[:, None])


@torch.no_grad()
def sample_diffusion_gemma_canvas(
    model: DiffusionGemmaModel,
    clean_ids: Tensor,
    clean_valid: Tensor,
    document_ids: Tensor,
    positions: Tensor,
    branch_starts: Tensor,
    branch_valid: Tensor,
    config: EntropyBudgetSamplerConfig = EntropyBudgetSamplerConfig(),
    *,
    generator: torch.Generator | None = None,
    initial_ids: Tensor | None = None,
    denoise_active: Tensor | None = None,
    allow_synthetic_branch_suffix: bool = False,
    trace_steps: list[list[dict[str, object]]] | None = None,
) -> DiffusionGemmaGeneration:
    """Generate one revisable fixed canvas per row.

    The clean prefix remains a shared causal bank throughout sampling.  EOT is
    ordinary revisable noise until a row stops; only the final deterministic
    canvas is truncated, exactly as in the reference sampler.
    """

    if branch_starts.shape != (clean_ids.shape[0], 1):
        raise ValueError("inference currently supports exactly one canvas per row")
    if branch_valid.ndim != 3 or branch_valid.shape[:2] != branch_starts.shape:
        raise ValueError("branch validity must be [B,1,C]")
    active = branch_valid[:, 0] if denoise_active is None else denoise_active
    if active.shape != branch_valid[:, 0].shape or active.dtype != torch.bool:
        raise ValueError("denoising activity must be one boolean canvas per row")
    if bool((active & ~branch_valid[:, 0]).any()):
        raise ValueError("denoising positions must be model-visible")
    if trace_steps is not None and len(trace_steps) != clean_ids.shape[0]:
        raise ValueError("DiffusionGemma trace rows must align with the batch")
    state = initialize_revisable_entropy_sampler(
        clean_ids.shape[0],
        branch_valid.shape[-1],
        output_size=model.config.vocab.output_size,
        device=clean_ids.device,
        generator=generator,
        active=active,
    )
    if initial_ids is not None:
        if initial_ids.shape != state.ids.shape or initial_ids.dtype != torch.long:
            raise ValueError("initial canvas ids must be aligned int64")
        state_ids = torch.where(~active & branch_valid[:, 0], initial_ids, state.ids)
    else:
        state_ids = state.ids
    state = RevisableEntropySamplerState(
        ids=torch.where(
            branch_valid[:, 0], state_ids, model.config.vocab.pad_id
        ),
        active=state.active,
        previous_argmax=state.previous_argmax,
        finished=state.finished,
        final_ids=state.final_ids,
        final_valid=state.final_valid,
        stop_reason=state.stop_reason,
        step=state.step,
    )
    attention_metadata = model.prepare_attention_metadata(clean_valid, document_ids)
    forwards = 0
    prior_probabilities: Tensor | None = None
    while not bool(state.finished.all()):
        trace_input = state.ids.detach().cpu() if trace_steps is not None else None
        noisy = torch.where(
            branch_valid[:, 0], state.ids, model.config.vocab.pad_id
        )[:, None]
        output = model(
            clean_ids,
            clean_valid,
            document_ids,
            positions,
            noisy,
            branch_valid,
            branch_starts,
            self_condition=False,
            # Zero on the initial terminal-random canvas, then reuse the
            # preceding detached distribution without an extra model forward.
            prior_probabilities=prior_probabilities,
            generator=generator,
            attention_metadata=attention_metadata,
            allow_synthetic_branch_suffix=allow_synthetic_branch_suffix,
        )
        forwards += 1
        step = revisable_entropy_sampler_step(
            state,
            output.branch_logits[:, 0],
            config,
            eot_id=model.config.vocab.eot_id,
            generator=generator,
        )
        if trace_steps is not None:
            assert trace_input is not None
            trace_output = step.state.ids.detach().cpu()
            trace_entropy = step.entropy.detach().cpu()
            trace_selected = step.selected.detach().cpu()
            trace_revised = step.revised.detach().cpu()
            trace_stopped = step.stopped.detach().cpu()
            for row, records in enumerate(trace_steps):
                records.append(
                    {
                        "step": step.state.step,
                        "temperature": step.temperature,
                        "input_ids": trace_input[row].tolist(),
                        "output_ids": trace_output[row].tolist(),
                        "entropy_nats": trace_entropy[row].tolist(),
                        "selected_positions": trace_selected[row]
                        .nonzero(as_tuple=False)
                        .flatten()
                        .tolist(),
                        "revised_positions": trace_revised[row]
                        .nonzero(as_tuple=False)
                        .flatten()
                        .tolist(),
                        "stopped": bool(trace_stopped[row]),
                    }
                )
        prior_probabilities = step.probabilities[:, None]
        state = step.state
    return DiffusionGemmaGeneration(
        ids=state.final_ids,
        valid=state.final_valid & branch_valid[:, 0],
        state=state,
        denoising_forwards=forwards,
        clean_commit_forwards=0,
    )


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


@torch.no_grad()
def _replay_clean_bank(
    model: DiffusionGemmaModel,
    clean_ids: Tensor,
    clean_valid: Tensor,
    document_ids: Tensor,
    positions: Tensor,
    lengths: Tensor,
) -> None:
    """Execute the clean prefill/append work required by a future KV backend.

    The current model exposes no persistent KV cache, so this is an honest
    full-bank replay with one exact-clean tail patch. It is counted as such by
    :func:`generate_diffusion_gemma_continuation` and never described as a
    cache hit.
    """

    stride = model.config.patch_stride
    starts = torch.div(lengths - 1, stride, rounding_mode="floor") * stride
    offsets = torch.arange(stride, device=clean_ids.device)
    indices = starts[:, None] + offsets
    safe = indices.clamp_max(clean_ids.shape[1] - 1)
    tail_ids = torch.gather(clean_ids, 1, safe)
    tail_valid = torch.gather(clean_valid, 1, safe) & indices.lt(clean_ids.shape[1])
    tail_ids = torch.where(tail_valid, tail_ids, model.config.vocab.pad_id)
    model(
        clean_ids,
        clean_valid,
        document_ids,
        positions,
        tail_ids[:, None],
        tail_valid[:, None],
        starts[:, None],
        self_condition=False,
        attention_metadata=model.prepare_attention_metadata(clean_valid, document_ids),
    )


@torch.no_grad()
def generate_diffusion_gemma_continuation(
    model: DiffusionGemmaModel,
    prompt_ids: Tensor,
    prompt_valid: Tensor,
    *,
    canvas_length: int,
    max_new_tokens: int,
    sampler_config: EntropyBudgetSamplerConfig = EntropyBudgetSamplerConfig(),
    max_canvases: int | None = None,
    generator: torch.Generator | None = None,
    trace_records: list[dict[str, object]] | None = None,
) -> DiffusionGemmaContinuation:
    """Continue clean ragged prompts using fresh revisable output canvases.

    Prompt atoms are never corrupted. At a ragged patch phase, the already
    committed atoms in that patch are replayed as fixed clean canvas inputs;
    only the previously unseen suffix is initialized from noise and revised.
    Final canvases are appended to the clean bank and paid for with an explicit
    clean replay because this standalone model does not yet expose persistent
    KV storage.
    """

    if (
        prompt_ids.dtype != torch.long
        or prompt_ids.ndim != 2
        or prompt_ids.shape != prompt_valid.shape
        or prompt_valid.dtype != torch.bool
    ):
        raise ValueError("prompt ids/validity must be aligned rank-2 tensors")
    stride = model.config.patch_stride
    if canvas_length <= 0 or canvas_length % stride:
        raise ValueError("canvas length must be a positive patch multiple")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompt_lengths = prompt_valid.sum(1)
    if bool((prompt_lengths <= 0).any()):
        raise ValueError("every prompt needs at least one clean atom")
    columns = torch.arange(prompt_ids.shape[1], device=prompt_ids.device)
    if not torch.equal(prompt_valid, columns[None] < prompt_lengths[:, None]):
        raise ValueError("prompts must be contiguous clean prefixes")
    if bool(prompt_ids.masked_select(prompt_valid).eq(model.config.vocab.eot_id).any()):
        raise ValueError("a continuation prompt cannot already contain EOT")

    batch = prompt_ids.shape[0]
    if trace_records is not None and len(trace_records) != batch:
        raise ValueError("DiffusionGemma continuation traces must align with prompts")
    capacity = _round_up(
        max(prompt_ids.shape[1], int(prompt_lengths.max()) + max_new_tokens), stride
    )
    ids = torch.full(
        (batch, capacity),
        model.config.vocab.pad_id,
        dtype=torch.long,
        device=prompt_ids.device,
    )
    ids[:, : prompt_ids.shape[1]] = torch.where(
        prompt_valid,
        prompt_ids,
        torch.as_tensor(model.config.vocab.pad_id, device=prompt_ids.device),
    )
    lengths = prompt_lengths.clone()
    limits = prompt_lengths + max_new_tokens
    valid = torch.arange(capacity, device=ids.device)[None] < lengths[:, None]
    document_ids = torch.arange(batch, device=ids.device)[:, None].expand_as(ids)
    positions = torch.arange(capacity, device=ids.device)[None].expand_as(ids)

    # This is a real model forward, not metadata-only bookkeeping. The lack of
    # persistent KV is surfaced in the result rather than hidden in accounting.
    _replay_clean_bank(model, ids, valid, document_ids, positions, lengths)
    clean_prefill_forwards = 1
    clean_commit_forwards = 0
    denoising_forwards = 0
    canvases = 0
    finished = torch.zeros(batch, dtype=torch.bool, device=ids.device)
    canvas_limit = (
        max_new_tokens if max_canvases is None else max_canvases
    )
    if canvas_limit <= 0:
        raise ValueError("max_canvases must be positive when provided")

    while bool((~finished & (lengths < limits)).any()) and canvases < canvas_limit:
        live = ~finished & (lengths < limits)
        rows = live.nonzero(as_tuple=False).flatten()
        row_ids = ids.index_select(0, rows)
        row_valid = valid.index_select(0, rows)
        row_documents = document_ids.index_select(0, rows)
        row_positions = positions.index_select(0, rows)
        row_lengths = lengths.index_select(0, rows)
        row_limits = limits.index_select(0, rows)
        starts = torch.div(row_lengths, stride, rounding_mode="floor") * stride
        phase = row_lengths - starts
        offsets = torch.arange(canvas_length, device=ids.device)[None]
        fresh = (offsets >= phase[:, None]) & (
            offsets < phase[:, None] + (row_limits - row_lengths)[:, None]
        )
        visible = (offsets < phase[:, None]) | fresh
        absolute = starts[:, None] + offsets
        safe = absolute.clamp_max(capacity - 1)
        initial = torch.gather(row_ids, 1, safe)
        initial = torch.where(visible, initial, model.config.vocab.pad_id)
        canvas_traces = (
            None if trace_records is None else [[] for _ in range(rows.numel())]
        )

        canvas = sample_diffusion_gemma_canvas(
            model,
            row_ids,
            row_valid,
            row_documents,
            row_positions,
            starts[:, None],
            visible[:, None],
            sampler_config,
            generator=generator,
            initial_ids=initial,
            denoise_active=fresh,
            allow_synthetic_branch_suffix=True,
            trace_steps=canvas_traces,
        )
        if trace_records is not None:
            assert canvas_traces is not None
            for local_row, output_row in enumerate(rows.cpu().tolist()):
                blocks = trace_records[output_row].get("diffusion_blocks")
                if not isinstance(blocks, list):
                    raise TypeError("DiffusionGemma trace needs diffusion_blocks list")
                blocks.append(
                    {
                        "block": canvases,
                        "absolute_atom_start": int(starts[local_row]),
                        "prompt_atoms_noised": False,
                        "revisable_positions": fresh[local_row]
                        .nonzero(as_tuple=False)
                        .flatten()
                        .cpu()
                        .tolist(),
                        "initial_ids": initial[local_row].cpu().tolist(),
                        "steps": canvas_traces[local_row],
                    }
                )
        denoising_forwards += canvas.denoising_forwards
        write = fresh & canvas.valid
        target_rows = rows[:, None].expand_as(absolute)
        ids[target_rows[write], absolute[write]] = canvas.ids[write]
        written = write.sum(1)
        lengths.index_add_(0, rows, written)
        wrote_eot = (write & canvas.ids.eq(model.config.vocab.eot_id)).any(1)
        finished[rows] |= wrote_eot
        valid = torch.arange(capacity, device=ids.device)[None] < lengths[:, None]

        # Pay the clean append for exactly the rows that produced a canvas.
        _replay_clean_bank(
            model,
            ids.index_select(0, rows),
            valid.index_select(0, rows),
            document_ids.index_select(0, rows),
            positions.index_select(0, rows),
            lengths.index_select(0, rows),
        )
        clean_commit_forwards += 1
        canvases += 1

    return DiffusionGemmaContinuation(
        ids=ids,
        valid=valid,
        prompt_lengths=prompt_lengths,
        lengths=lengths,
        finished=finished,
        canvases=canvases,
        denoising_forwards=denoising_forwards,
        clean_prefill_forwards=clean_prefill_forwards,
        clean_commit_forwards=clean_commit_forwards,
    )


__all__ = (
    "DiffusionGemmaContinuation",
    "DiffusionGemmaGeneration",
    "generate_diffusion_gemma_continuation",
    "sample_diffusion_gemma_canvas",
)
