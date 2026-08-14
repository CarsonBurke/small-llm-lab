"""Exact revisable ancestral sampling for Byte-Duo."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .duo import DuoSchedule
from .duo_kernels import DuoPosteriorBackend, duo_posterior_sample
from .duo_model import DuoCanvasCache, DuoModel


@dataclass(frozen=True)
class DuoGeneration:
    ids: Tensor
    valid: Tensor
    trajectory: Tensor | None
    denoising_forwards: int
    clean_bank_forwards: int
    revision_count: Tensor

    @property
    def model_forwards(self) -> int:
        return self.denoising_forwards + self.clean_bank_forwards


@dataclass(frozen=True)
class DuoContinuation:
    ids: Tensor
    valid: Tensor
    prompt_lengths: Tensor
    lengths: Tensor
    finished: Tensor
    text_stopped: Tensor
    byte_capped: Tensor
    canvases: int
    diffusion_steps: int
    denoising_forwards: int
    denoising_actions: Tensor
    clean_bank_actions: Tensor
    clean_prefill_forwards: int
    clean_commit_forwards: int
    trajectories: tuple[Tensor, ...] | None = None
    trajectory_rows: tuple[Tensor, ...] | None = None
    trajectory_starts: tuple[Tensor, ...] | None = None
    trajectory_active: tuple[Tensor, ...] | None = None
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
def sample_duo_canvas(
    model: DuoModel,
    clean_ids: Tensor,
    clean_valid: Tensor,
    document_ids: Tensor,
    positions: Tensor,
    branch_starts: Tensor,
    branch_valid: Tensor,
    *,
    steps: int,
    schedule: DuoSchedule = DuoSchedule(),
    terminal_eps: float = 1e-5,
    generator: torch.Generator | None = None,
    use_float64: bool = False,
    initial_ids: Tensor | None = None,
    denoise_active: Tensor | None = None,
    return_trajectory: bool = True,
    posterior_backend: DuoPosteriorBackend = "auto",
    prepared_cache: DuoCanvasCache | None = None,
) -> DuoGeneration:
    """Sample one canvas per row from a uniform clean-atom prior.

    All positions remain revisable at every transition.  ``steps`` is chosen at
    call time and counts reverse-grid transitions; one final denoiser/posterior
    evaluation removes the residual schedule noise exactly.
    """

    if steps <= 0:
        raise ValueError("Duo inference steps must be positive")
    if not 0.0 < terminal_eps < 1.0:
        raise ValueError("Duo sampling terminal eps must lie in (0, 1)")
    if model.schedule_eps != schedule.eps:
        raise ValueError("Duo model and inference schedules disagree")
    batch = clean_ids.shape[0]
    if branch_starts.shape != (batch, 1):
        raise ValueError("Duo inference currently supports one canvas per row")
    if branch_valid.ndim != 3 or branch_valid.shape[:2] != (batch, 1):
        raise ValueError("Duo branch validity must be [B,1,C]")
    canvas = branch_valid.shape[-1]
    clean_atoms = model.config.duo_diffusion_atoms
    visible = branch_valid[:, 0]
    active = visible if denoise_active is None else denoise_active
    if active.shape != visible.shape or active.dtype != torch.bool:
        raise ValueError("Duo denoising activity must align with the canvas")
    if active.device.type == "cpu" and bool((active & ~visible).any()):
        raise ValueError("Duo denoising positions must be model-visible")
    prior = torch.randint(
        clean_atoms,
        (batch, canvas),
        dtype=torch.long,
        device=clean_ids.device,
        generator=generator,
    )
    if initial_ids is None:
        fixed = torch.full_like(prior, model.config.vocab.pad_id)
    else:
        if initial_ids.shape != prior.shape or initial_ids.dtype != torch.long:
            raise ValueError("initial Duo canvas must be aligned int64")
        fixed = initial_ids
    ids = torch.where(active, prior, torch.where(visible, fixed, model.config.vocab.pad_id))

    trajectory = (
        torch.empty(
            (steps + 2, batch, canvas), dtype=torch.long, device=clean_ids.device
        )
        if return_trajectory
        else None
    )
    if trajectory is not None:
        trajectory[0] = ids
    # Host endpoints let the fused sampler validate the schedule without a
    # device synchronization on every transition. Model conditioning remains
    # an explicitly materialized FP32 device tensor below.
    # The reference sampler integrates to an independent 1e-5 terminal time.
    # ``schedule.eps`` belongs to alpha(t) and training-time truncation; using
    # it as the sampling endpoint skipped the final 0.001 -> 0.00001 interval.
    grid = tuple(
        1.0 + (terminal_eps - 1.0) * index / steps
        for index in range(steps + 1)
    )
    conditioning_grid = torch.tensor(
        grid, dtype=torch.float32, device=clean_ids.device
    )[:, None].expand(-1, batch)
    cache_backed = isinstance(model, DuoModel)
    clean_bank_forwards = 0
    if prepared_cache is not None:
        if not cache_backed:
            raise ValueError("a prepared Duo cache requires a production DuoModel")
        if prepared_cache.branch_valid.shape != branch_valid.shape:
            raise ValueError("prepared Duo cache does not match the canvas")
        if clean_ids.device.type == "cpu" and (
            not torch.equal(prepared_cache.branch_valid, branch_valid)
            or not torch.equal(prepared_cache.branch_starts, branch_starts)
            or not torch.equal(prepared_cache.clean.clean_ids, clean_ids)
            or not torch.equal(prepared_cache.clean.clean_valid, clean_valid)
        ):
            raise ValueError("prepared Duo cache contents do not match the inputs")
        canvas_cache = prepared_cache
        metadata = None
    elif cache_backed:
        with torch.autocast(
            device_type=clean_ids.device.type,
            dtype=torch.bfloat16,
            enabled=clean_ids.device.type == "cuda",
        ):
            clean_bank = model.prepare_clean_bank(
                clean_ids,
                clean_valid,
                document_ids,
                positions,
                attention_metadata=model.prepare_attention_metadata(
                    clean_valid, document_ids
                ),
            )
            canvas_cache = model.prepare_canvas_cache(
                clean_bank,
                branch_valid,
                branch_starts,
                allow_synthetic_branch_suffix=True,
            )
        clean_bank_forwards = 1
        metadata = None
    else:
        canvas_cache = None
        metadata = model.prepare_attention_metadata(clean_valid, document_ids)
    revisions = torch.zeros((), dtype=torch.long, device=clean_ids.device)
    for index in range(steps):
        t = grid[index]
        s = grid[index + 1]
        row_t = conditioning_grid[index, :, None]
        with torch.autocast(
            device_type=clean_ids.device.type,
            dtype=torch.bfloat16,
            enabled=clean_ids.device.type == "cuda",
        ):
            if canvas_cache is None:
                output = model(
                    clean_ids,
                    clean_valid,
                    document_ids,
                    positions,
                    ids[:, None],
                    branch_valid,
                    branch_starts,
                    row_t,
                    attention_metadata=metadata,
                    allow_synthetic_branch_suffix=True,
                )
            else:
                output = model.forward_prepared(
                    canvas_cache, ids[:, None], row_t
                )
            logits = output.branch_logits[:, 0, :, :clean_atoms]
        alpha_s = 1.0 - (1.0 - schedule.eps) * s
        alpha_t = 1.0 - (1.0 - schedule.eps) * t
        uniforms = torch.rand(
            ids.shape,
            dtype=torch.float32,
            device=ids.device,
            generator=generator,
        )
        sampled = duo_posterior_sample(
            logits,
            ids,
            alpha_s,
            alpha_t,
            uniforms,
            active,
            backend=posterior_backend,
            use_float64=use_float64,
        )
        sampled = torch.where(visible, sampled, model.config.vocab.pad_id)
        revisions += (active & sampled.ne(ids)).sum()
        ids = sampled
        if trajectory is not None:
            trajectory[index + 1] = ids

    # The configured schedule is only nearly clean at the lower time endpoint.
    # Conditioning the last exact posterior on alpha_s=1 is the reference ancestral noise
    # removal operation, not a greedy overwrite.
    row_t = conditioning_grid[-1, :, None]
    with torch.autocast(
        device_type=clean_ids.device.type,
        dtype=torch.bfloat16,
        enabled=clean_ids.device.type == "cuda",
    ):
        if canvas_cache is None:
            output = model(
                clean_ids,
                clean_valid,
                document_ids,
                positions,
                ids[:, None],
                branch_valid,
                branch_starts,
                row_t,
                attention_metadata=metadata,
                allow_synthetic_branch_suffix=True,
            )
        else:
            output = model.forward_prepared(canvas_cache, ids[:, None], row_t)
        logits = output.branch_logits[:, 0, :, :clean_atoms]
    alpha_t = 1.0 - (1.0 - schedule.eps) * terminal_eps
    uniforms = torch.rand(
        ids.shape,
        dtype=torch.float32,
        device=ids.device,
        generator=generator,
    )
    sampled = duo_posterior_sample(
        logits,
        ids,
        1.0,
        alpha_t,
        uniforms,
        active,
        backend=posterior_backend,
        use_float64=use_float64,
    )
    sampled = torch.where(visible, sampled, model.config.vocab.pad_id)
    revisions += (active & sampled.ne(ids)).sum()
    ids = sampled
    if trajectory is not None:
        trajectory[-1] = ids

    offsets = torch.arange(canvas, device=ids.device)[None]
    eot_positions = torch.where(
        active & ids.eq(model.config.vocab.eot_id), offsets, canvas
    )
    first_eot = eot_positions.min(1).values
    valid = visible & offsets.lt(first_eot[:, None])
    return DuoGeneration(
        ids=ids,
        valid=valid,
        trajectory=trajectory,
        denoising_forwards=steps + 1,
        clean_bank_forwards=clean_bank_forwards,
        revision_count=revisions,
    )


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _compact_literal_bytes(ids: Tensor, selected: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Compact selected literal-byte atoms without a Python row loop."""

    width = ids.shape[1]
    literal = selected & ids.lt(256)
    ranks = literal.cumsum(1) - 1
    safe_ranks = ranks.clamp_min(0)
    compact = torch.full_like(ids, 256)
    compact.scatter_reduce_(
        1,
        safe_ranks,
        torch.where(literal, ids, 256),
        reduce="amin",
        include_self=True,
    )
    source_offsets = torch.full_like(ids, width)
    offsets = torch.arange(width, device=ids.device)[None].expand_as(ids)
    source_offsets.scatter_reduce_(
        1,
        safe_ranks,
        torch.where(literal, offsets, width),
        reduce="amin",
        include_self=True,
    )
    return compact, source_offsets, literal.sum(1)


def _append_compact_bytes(
    prefix: Tensor,
    prefix_lengths: Tensor,
    suffix: Tensor,
    suffix_lengths: Tensor,
) -> tuple[Tensor, Tensor]:
    """Append two left-aligned ragged byte tensors into fixed storage."""

    return _append_compact_values(
        prefix,
        prefix_lengths,
        suffix,
        suffix_lengths,
        fill_value=256,
    )


def _append_compact_values(
    prefix: Tensor,
    prefix_lengths: Tensor,
    suffix: Tensor,
    suffix_lengths: Tensor,
    *,
    fill_value: int,
) -> tuple[Tensor, Tensor]:
    """Append two left-aligned ragged tensors using an out-of-domain fill."""

    batch = prefix.shape[0]
    width = prefix.shape[1] + suffix.shape[1]
    combined = torch.full(
        (batch, width), fill_value, dtype=torch.long, device=prefix.device
    )
    prefix_columns = torch.arange(prefix.shape[1], device=prefix.device)[None]
    prefix_valid = prefix_columns < prefix_lengths[:, None]
    combined[:, : prefix.shape[1]] = torch.where(
        prefix_valid, prefix, fill_value
    )
    suffix_columns = torch.arange(suffix.shape[1], device=suffix.device)[None]
    suffix_valid = suffix_columns < suffix_lengths[:, None]
    suffix_targets = prefix_lengths[:, None] + suffix_columns
    combined.scatter_reduce_(
        1,
        suffix_targets.clamp_max(width - 1),
        torch.where(suffix_valid, suffix, fill_value),
        reduce="amin",
        include_self=True,
    )
    return combined, prefix_lengths + suffix_lengths


def _first_stop_span(
    combined: Tensor,
    combined_lengths: Tensor,
    prefix_lengths: Tensor,
    stop_sequences: tuple[bytes, ...],
) -> tuple[Tensor, Tensor]:
    """Return start and exclusive end for the first newly completed stop."""

    sentinel = combined.shape[1] + 1
    best_start = torch.full_like(combined_lengths, sentinel)
    best_end = torch.full_like(combined_lengths, sentinel)
    starts = torch.arange(combined.shape[1], device=combined.device)
    for sequence in stop_sequences:
        pattern = torch.tensor(tuple(sequence), dtype=torch.long, device=combined.device)
        length = pattern.numel()
        if length == 0 or length > combined.shape[1]:
            continue
        windows = combined.unfold(1, length, 1)
        window_starts = starts[: windows.shape[1]]
        ends = window_starts + length
        matched = windows.eq(pattern).all(-1)
        matched &= ends[None] <= combined_lengths[:, None]
        matched &= ends[None] > prefix_lengths[:, None]
        candidate_end = torch.where(matched, ends[None], sentinel).min(1).values
        candidate_start = candidate_end - length
        replace = candidate_end.lt(sentinel) & (
            candidate_end.lt(best_end)
            | (candidate_end.eq(best_end) & candidate_start.lt(best_start))
        )
        best_start = torch.where(replace, candidate_start, best_start)
        best_end = torch.where(replace, candidate_end, best_end)
    return best_start, best_end


def _last_values(
    combined: Tensor,
    lengths: Tensor,
    width: int,
    *,
    fill_value: int,
) -> tuple[Tensor, Tensor]:
    """Retain a left-aligned suffix from compact ragged values."""

    if width == 0:
        return combined[:, :0], torch.zeros_like(lengths)
    retained = lengths.clamp_max(width)
    starts = lengths - retained
    offsets = torch.arange(width, device=combined.device)[None]
    indices = starts[:, None] + offsets
    safe = indices.clamp_max(combined.shape[1] - 1)
    values = torch.gather(combined, 1, safe)
    return torch.where(offsets < retained[:, None], values, fill_value), retained


@torch.no_grad()
def generate_duo_continuation(
    model: DuoModel,
    prompt_ids: Tensor,
    prompt_valid: Tensor,
    *,
    canvas_length: int,
    max_new_atoms: int,
    steps: int,
    visible_width: int | None = None,
    commit_width: int | None = None,
    schedule: DuoSchedule = DuoSchedule(),
    terminal_eps: float = 1e-5,
    max_canvases: int | None = None,
    generator: torch.Generator | None = None,
    use_float64: bool = False,
    return_trajectories: bool = False,
    posterior_backend: DuoPosteriorBackend = "auto",
    stop_sequences: tuple[bytes, ...] = (),
    max_new_bytes: int | None = None,
) -> DuoContinuation:
    """Generate ragged multi-canvas continuations without corrupting prompts."""

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
    if max_new_atoms <= 0 or steps <= 0:
        raise ValueError("generation length and diffusion steps must be positive")
    if commit_width is not None and not 0 < commit_width <= canvas_length:
        raise ValueError("Duo commit width must lie in [1, canvas_length]")
    if visible_width is not None and not 0 < visible_width <= canvas_length:
        raise ValueError("Duo visible width must lie in [1, canvas_length]")
    if commit_width is not None and visible_width is not None and commit_width > visible_width:
        raise ValueError("Duo commit width cannot exceed visible width")
    if max_new_bytes is not None and max_new_bytes <= 0:
        raise ValueError("literal byte budget must be positive when provided")
    prompt_lengths = prompt_valid.sum(1)
    if bool(prompt_lengths.le(0).any()):
        raise ValueError("every Duo prompt needs at least one clean atom")
    columns = torch.arange(prompt_ids.shape[1], device=prompt_ids.device)
    if not torch.equal(prompt_valid, columns[None] < prompt_lengths[:, None]):
        raise ValueError("Duo prompts must be contiguous clean prefixes")
    prompt_atoms = prompt_ids.masked_select(prompt_valid)
    if bool(prompt_atoms.eq(model.config.vocab.eot_id).any()):
        raise ValueError("a Duo continuation prompt cannot already contain EOT")
    if bool(prompt_atoms.eq(model.config.vocab.mask_id).any()):
        raise ValueError("Byte-Duo prompts cannot contain MASK")
    if bool(prompt_atoms.eq(model.config.vocab.pad_id).any()):
        raise ValueError("Byte-Duo prompts cannot contain PAD")
    if bool(
        (prompt_atoms.lt(0) | prompt_atoms.ge(model.config.vocab.output_size)).any()
    ):
        raise ValueError("Byte-Duo prompt atoms must lie in the clean output vocabulary")

    batch = prompt_ids.shape[0]
    capacity = _round_up(
        max(prompt_ids.shape[1], int(prompt_lengths.max()) + max_new_atoms), stride
    )
    ids = torch.full(
        (batch, capacity),
        model.config.vocab.pad_id,
        dtype=torch.long,
        device=prompt_ids.device,
    )
    ids[:, : prompt_ids.shape[1]] = torch.where(
        prompt_valid, prompt_ids, model.config.vocab.pad_id
    )
    lengths = prompt_lengths.clone()
    limits = prompt_lengths + max_new_atoms
    valid = torch.arange(capacity, device=ids.device)[None] < lengths[:, None]
    document_ids = torch.arange(batch, device=ids.device)[:, None].expand_as(ids)
    positions = torch.arange(capacity, device=ids.device)[None].expand_as(ids)

    cache_backed = isinstance(model, DuoModel)
    clean_prefill_forwards = 0
    clean_commit_forwards = 0
    denoising_forwards = 0
    denoising_actions = torch.zeros_like(prompt_lengths)
    clean_bank_actions = torch.zeros_like(prompt_lengths)
    canvases = 0
    finished = torch.zeros(batch, dtype=torch.bool, device=ids.device)
    text_stopped = torch.zeros_like(finished)
    byte_capped = torch.zeros_like(finished)
    literal_counts = torch.zeros_like(prompt_lengths)
    if any(not sequence for sequence in stop_sequences):
        raise ValueError("Duo stop sequences must be nonempty")
    stop_tail_width = max((len(sequence) for sequence in stop_sequences), default=1) - 1
    if stop_tail_width:
        prompt_bytes, prompt_byte_offsets, prompt_byte_counts = _compact_literal_bytes(
            ids, valid
        )
        stop_tail, tail_lengths = _last_values(
            prompt_bytes,
            prompt_byte_counts,
            stop_tail_width,
            fill_value=256,
        )
        stop_tail_positions, _ = _last_values(
            prompt_byte_offsets,
            prompt_byte_counts,
            stop_tail_width,
            fill_value=capacity,
        )
    else:
        tail_lengths = torch.zeros_like(prompt_lengths)
        stop_tail = ids[:, :0]
        stop_tail_positions = ids[:, :0]
    trajectories: tuple[Tensor, ...] | None = () if return_trajectories else None
    trajectory_rows: tuple[Tensor, ...] | None = () if return_trajectories else None
    trajectory_starts: tuple[Tensor, ...] | None = () if return_trajectories else None
    trajectory_active: tuple[Tensor, ...] | None = () if return_trajectories else None
    canvas_limit = max_new_atoms if max_canvases is None else max_canvases
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
        if visible_width is not None:
            fresh &= fresh.cumsum(1).le(visible_width)
        commit_eligible = (
            fresh
            if commit_width is None
            else fresh & fresh.cumsum(1).le(commit_width)
        )
        visible = offsets.lt(phase[:, None]) | fresh
        absolute = starts[:, None] + offsets
        safe = absolute.clamp_max(capacity - 1)
        initial = torch.gather(row_ids, 1, safe)
        initial = torch.where(visible, initial, model.config.vocab.pad_id)
        prepared_cache = None
        if cache_backed:
            with torch.autocast(
                device_type=ids.device.type,
                dtype=torch.bfloat16,
                enabled=ids.device.type == "cuda",
            ):
                clean_bank = model.prepare_clean_bank(
                    row_ids,
                    row_valid,
                    row_documents,
                    row_positions,
                    attention_metadata=model.prepare_attention_metadata(
                        row_valid, row_documents
                    ),
                )
            prepared_cache = model.prepare_canvas_cache(
                    clean_bank,
                    visible[:, None],
                    starts[:, None],
                allow_synthetic_branch_suffix=True,
                validate_inputs=False,
            )
            if canvases == 0:
                clean_prefill_forwards += 1
            else:
                clean_commit_forwards += 1
            clean_bank_actions.index_add_(0, rows, torch.ones_like(rows))
        canvas = sample_duo_canvas(
            model,
            row_ids,
            row_valid,
            row_documents,
            row_positions,
            starts[:, None],
            visible[:, None],
            steps=steps,
            schedule=schedule,
            terminal_eps=terminal_eps,
            generator=generator,
            use_float64=use_float64,
            initial_ids=initial,
            denoise_active=fresh,
            return_trajectory=return_trajectories,
            posterior_backend=posterior_backend,
            prepared_cache=prepared_cache,
        )
        if trajectories is not None:
            if canvas.trajectory is None:
                raise AssertionError("requested Duo trajectory was not retained")
            trajectories += (canvas.trajectory,)
            if trajectory_rows is None:
                raise AssertionError("Duo trajectory row ledger is missing")
            trajectory_rows += (rows.clone(),)
            if trajectory_starts is None:
                raise AssertionError("Duo trajectory start ledger is missing")
            trajectory_starts += (starts.clone(),)
            if trajectory_active is None:
                raise AssertionError("Duo trajectory activity ledger is missing")
            trajectory_active += (fresh.clone(),)
        denoising_forwards += canvas.denoising_forwards
        denoising_actions.index_add_(
            0,
            rows,
            torch.full_like(rows, canvas.denoising_forwards),
        )
        # Denoise the full trained-width canvas but optionally commit only its
        # earliest atoms. This is the faithful semi-autoregressive block-stride
        # control: discarded lookahead is regenerated after becoming closer to
        # the clean prefix, without changing the model's training topology.
        write = commit_eligible & canvas.valid
        eot_candidates = torch.where(
            commit_eligible & canvas.ids.eq(model.config.vocab.eot_id),
            offsets,
            canvas_length,
        )
        first_eot = eot_candidates.min(1).values
        wrote_eot = (
            commit_eligible
            & canvas.ids.eq(model.config.vocab.eot_id)
            & offsets.eq(first_eot[:, None])
        )
        stop_found = torch.zeros(rows.shape[0], dtype=torch.bool, device=ids.device)
        terminal_sentinel = canvas_length + 1
        stop_start_offsets = torch.full_like(first_eot, terminal_sentinel)
        stop_completion_offsets = torch.full_like(first_eot, terminal_sentinel)
        new_bytes, byte_offsets, new_byte_counts = _compact_literal_bytes(
            canvas.ids, write
        )
        if stop_sequences:
            row_tail = stop_tail.index_select(0, rows)
            row_tail_lengths = tail_lengths.index_select(0, rows)
            combined, combined_lengths = _append_compact_bytes(
                row_tail, row_tail_lengths, new_bytes, new_byte_counts
            )
            combined_positions, _ = _append_compact_values(
                stop_tail_positions.index_select(0, rows),
                row_tail_lengths,
                starts[:, None] + byte_offsets,
                new_byte_counts,
                fill_value=capacity,
            )
            stop_starts, stop_ends = _first_stop_span(
                combined, combined_lengths, row_tail_lengths, stop_sequences
            )
            stop_found = stop_ends <= combined_lengths
            stop_start_positions = torch.gather(
                combined_positions,
                1,
                stop_starts.clamp_max(combined_positions.shape[1] - 1)[:, None],
            ).squeeze(1)
            stop_keep_lengths = torch.maximum(
                stop_start_positions, prompt_lengths.index_select(0, rows)
            )
            first_generated_stop_byte = (stop_starts - row_tail_lengths).clamp(
                min=0, max=canvas_length - 1
            )
            matched_start_offsets = torch.gather(
                byte_offsets, 1, first_generated_stop_byte[:, None]
            ).squeeze(1)
            stop_start_offsets = torch.where(
                stop_found, matched_start_offsets, stop_start_offsets
            )
            last_generated_stop_byte = (stop_ends - row_tail_lengths - 1).clamp(
                min=0, max=canvas_length - 1
            )
            matched_completion_offsets = torch.gather(
                byte_offsets, 1, last_generated_stop_byte[:, None]
            ).squeeze(1) + 1
            stop_completion_offsets = torch.where(
                stop_found, matched_completion_offsets, stop_completion_offsets
            )
        else:
            stop_keep_lengths = row_lengths
        byte_cap_found = torch.zeros_like(stop_found)
        byte_cap_exclusive_offsets = torch.full_like(first_eot, terminal_sentinel)
        if max_new_bytes is not None:
            remaining_bytes = max_new_bytes - literal_counts.index_select(0, rows)
            byte_cap_found = new_byte_counts >= remaining_bytes
            cap_byte_rank = (remaining_bytes - 1).clamp(
                min=0, max=canvas_length - 1
            )
            byte_cap_end_offsets = torch.gather(
                byte_offsets, 1, cap_byte_rank[:, None]
            ).squeeze(1)
            byte_cap_exclusive_offsets = byte_cap_end_offsets + 1
        stop_wins = stop_found & stop_completion_offsets.le(
            byte_cap_exclusive_offsets
        )
        byte_cap_wins = byte_cap_found & byte_cap_exclusive_offsets.lt(
            stop_completion_offsets
        )
        terminal_boundary = torch.where(
            stop_wins,
            stop_start_offsets,
            torch.where(
                byte_cap_wins, byte_cap_exclusive_offsets, terminal_sentinel
            ),
        )
        canvas_terminal = stop_found | byte_cap_found
        commit = torch.where(
            canvas_terminal[:, None],
            commit_eligible & offsets.lt(terminal_boundary[:, None]),
            write | wrote_eot,
        )
        target_rows = rows[:, None].expand_as(absolute)
        ids[target_rows[commit], absolute[commit]] = canvas.ids[commit]
        next_lengths = row_lengths + commit.sum(1)
        lengths[rows] = torch.where(stop_wins, stop_keep_lengths, next_lengths)
        finished[rows] |= wrote_eot.any(1) | canvas_terminal
        text_stopped[rows] |= stop_wins
        byte_capped[rows] |= byte_cap_wins
        literal_counts.index_add_(
            0, rows, (commit & canvas.ids.lt(256)).sum(1)
        )
        if stop_sequences:
            (
                committed_bytes,
                committed_offsets,
                committed_byte_counts,
            ) = _compact_literal_bytes(canvas.ids, commit)
            combined, combined_lengths = _append_compact_bytes(
                stop_tail.index_select(0, rows),
                tail_lengths.index_select(0, rows),
                committed_bytes,
                committed_byte_counts,
            )
            combined_positions, _ = _append_compact_values(
                stop_tail_positions.index_select(0, rows),
                tail_lengths.index_select(0, rows),
                starts[:, None] + committed_offsets,
                committed_byte_counts,
                fill_value=capacity,
            )
            next_tail, next_tail_lengths = _last_values(
                combined,
                combined_lengths,
                stop_tail_width,
                fill_value=256,
            )
            next_tail_positions, _ = _last_values(
                combined_positions,
                combined_lengths,
                stop_tail_width,
                fill_value=capacity,
            )
            stop_tail[rows] = next_tail
            stop_tail_positions[rows] = next_tail_positions
            tail_lengths[rows] = next_tail_lengths
        valid = torch.arange(capacity, device=ids.device)[None] < lengths[:, None]
        ids = torch.where(valid, ids, model.config.vocab.pad_id)
        canvases += 1

    return DuoContinuation(
        ids=ids,
        valid=valid,
        prompt_lengths=prompt_lengths,
        lengths=lengths,
        finished=finished,
        text_stopped=text_stopped,
        byte_capped=byte_capped,
        canvases=canvases,
        diffusion_steps=steps,
        denoising_forwards=denoising_forwards,
        denoising_actions=denoising_actions,
        clean_bank_actions=clean_bank_actions,
        clean_prefill_forwards=clean_prefill_forwards,
        clean_commit_forwards=clean_commit_forwards,
        trajectories=trajectories,
        trajectory_rows=trajectory_rows,
        trajectory_starts=trajectory_starts,
        trajectory_active=trajectory_active,
        cache_backed=cache_backed,
    )


__all__ = (
    "DuoContinuation",
    "DuoGeneration",
    "generate_duo_continuation",
    "sample_duo_canvas",
)
