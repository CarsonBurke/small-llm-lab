"""Per-bucket timing of the graph-decode arena on a real checkpoint.

Answers where a rollout's decode time goes without a full trainer run: each
live-row bucket's graph is replayed on the production arena shape at a few
write-head positions (attention reads each row's live key range, so a tick's
cost grows with the position) and timed with CUDA events, and one bucket's
tick is profiled kernel by kernel. Every row's range starts at slot 0 -- the
no-padding worst case -- and replays run on whatever state the capture left,
which is fine for timing: every row is kept active, so each replay does full
work. It also reports what a rollout's workspace costs: the time to re-open
it (memset plus one capture per bucket, paid by every rollout after the
first) and the memory it holds, all of which must be released on exit.

Run through mlq; takes about a minute.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from postraining.graph_decode import PinnedDecodeArena
from postraining.latent_thought import LatentThoughtModel
from postraining.model_io import load_model


def time_buckets(
    arena: PinnedDecodeArena, positions: list[int], replays: int
) -> dict[str, float]:
    """ms per replay of every bucket's graph at each write-head position."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = {}
    arena.ended.zero_()
    arena.emitted.zero_()
    arena.starts.zero_()
    print(f"positions {positions}", flush=True)
    for rows in sorted(arena._graphs, reverse=True):
        graph = arena._graphs[rows]
        row = []
        for position in positions:
            arena.position.fill_(position)
            graph.replay()
            torch.cuda.synchronize()
            start.record()
            for _ in range(replays):
                arena.position.fill_(position)
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            timings[f"{rows}@{position}"] = start.elapsed_time(end) / replays
            row.append(f"{timings[f'{rows}@{position}']:.3f}")
        print(f"rows {rows:5d} ms/tick: {' '.join(row)}", flush=True)
    return timings


def profile_tick(arena: PinnedDecodeArena, rows: int, position: int) -> None:
    """Kernel table of the compiled tick at one bucket and position."""
    arena.position.fill_(position)
    arena._call_tick(rows)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        for _ in range(5):
            arena.position.fill_(position)
            arena._call_tick(rows)
        torch.cuda.synchronize()
    print(
        profile.key_averages().table(
            sort_by="cuda_time_total", row_limit=30, max_name_column_width=70
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--kv-width", type=int, default=1024)
    parser.add_argument("--prompt-width", type=int, default=160)
    parser.add_argument("--carry", action="store_true")
    parser.add_argument(
        "--no-cache-linear-weights", dest="cache_linear_weights",
        action="store_false",
    )
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--positions", type=int, nargs="+", default=[256, 512, 1022])
    parser.add_argument("--profile-rows", type=int, default=1024)
    parser.add_argument("--profile-position", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.fx.experimental._config.use_duck_shape = False

    wrapper = LatentThoughtModel(
        load_model(args.checkpoint, device), hidden_carry=args.carry
    ).to(device).eval()
    arena = PinnedDecodeArena(
        wrapper,
        rows=args.rows,
        kv_width=args.kv_width,
        temperature=1.0,
        stop_ids=(),
        device=device,
        hidden_carry=args.carry,
        cache_linear_weights=args.cache_linear_weights,
    )
    prompts = args.rows // 16
    prompt_ids = torch.randint(
        1, 50000, (prompts, args.prompt_width), device=device
    )
    lengths = torch.full((prompts,), args.prompt_width)
    seeds = torch.arange(prompts * 16, device=device, dtype=torch.int64)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # One real rollout captures every graph (and times the first pool).
    start.record()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        arena.rollout(
            prompt_ids, lengths, prompt_repeats=16,
            max_new_tokens=args.kv_width - args.prompt_width,
            max_stream_steps=args.kv_width - args.prompt_width,
            token_seeds=seeds,
        )
    end.record()
    torch.cuda.synchronize()
    first_ms = start.elapsed_time(end)
    start.record()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        arena.rollout(
            prompt_ids, lengths, prompt_repeats=16,
            max_new_tokens=args.kv_width - args.prompt_width,
            max_stream_steps=args.kv_width - args.prompt_width,
            token_seeds=seeds,
        )
    end.record()
    torch.cuda.synchronize()
    steady_ms = start.elapsed_time(end)
    print(
        f"full-budget rollout (no stops, {arena.ticks} ticks total): "
        f"first {first_ms:.0f} ms, steady {steady_ms:.0f} ms", flush=True
    )

    # Between rollouts the workspace is closed; re-open it to time what every
    # rollout after the first pays, and check that closing frees it again.
    idle_bytes = torch.cuda.memory_allocated()
    torch.cuda.synchronize()
    reopen_start = time.perf_counter()
    with arena.workspace(arena.belief_dtype):
        torch.cuda.synchronize()
        reopen_ms = 1e3 * (time.perf_counter() - reopen_start)
        workspace_bytes = torch.cuda.memory_allocated() - idle_bytes
        print(
            f"workspace: {workspace_bytes / 2**30:.2f} GiB, re-opened "
            f"(memset + {len(arena.row_buckets)} captures) in {reopen_ms:.0f} ms",
            flush=True,
        )
        timings = time_buckets(arena, args.positions, args.replays)
        profile_tick(arena, args.profile_rows, args.profile_position)
    retained_bytes = torch.cuda.memory_allocated() - idle_bytes
    print(
        f"allocated after closing it: {retained_bytes / 2**20:+.1f} MiB",
        flush=True,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "first_rollout_ms": first_ms,
                    "steady_rollout_ms": steady_ms,
                    "workspace_reopen_ms": reopen_ms,
                    "workspace_bytes": workspace_bytes,
                    "retained_bytes_after_close": retained_bytes,
                    "ms_per_tick": timings,
                    "cache_linear_weights": args.cache_linear_weights,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
