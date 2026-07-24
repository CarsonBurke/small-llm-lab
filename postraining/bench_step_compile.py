"""Microbenchmark the stepwise decode: eager narrow vs compiled vs manual graph.

Answers, with a profiler instead of end-to-end wall time, why the compiled
reduce-overhead rollout measured slower than eager (jobs 136/138): per-step
cost of each path, whether compiled steps actually replay as one
``cudaGraphLaunch``, and what a dynamo-free manual ``torch.cuda.CUDAGraph``
capture of the same static-shape step costs.  The manual variant is the
decode analog of pretraining's compile recipe with the python dispatch
machinery removed from the hot loop entirely.

    mlq submit --name bench_step --cwd "$PWD" --max-parallel-runs 1 -- \
        python3 -m postraining.bench_step_compile --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import time

import torch

import train_gpt as baseline  # noqa: F401  (import order: patches must load first)
from postraining.latent_thought import LatentThoughtModel
from postraining.model_io import load_model


def synchronized_seconds(function, steps: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for position in range(steps):
        function(position)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def profile_launches(function, steps: int, label: str) -> None:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        for position in range(steps):
            function(position)
        torch.cuda.synchronize()
    events = trace.key_averages()
    graph_launches = sum(
        event.count for event in events if "cudaGraphLaunch" in event.key
    )
    kernel_launches = sum(
        event.count for event in events if event.key == "cudaLaunchKernel"
    )
    print(
        f"[{label}] {steps} steps: cudaGraphLaunch={graph_launches} "
        f"cudaLaunchKernel={kernel_launches}"
    )
    print(
        events.table(sort_by="self_cpu_time_total", row_limit=12)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 512])
    parser.add_argument("--cache-length", type=int, default=896)
    parser.add_argument("--start-position", type=int, default=384)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--measure-steps", type=int, default=256)
    parser.add_argument("--profile-steps", type=int, default=24)
    args = parser.parse_args()

    device = torch.device("cuda")
    backbone = load_model(args.checkpoint, device)
    backbone.eval()
    wrapper = LatentThoughtModel(backbone).to(device)
    wrapper.eval()
    model_dim = backbone.tok_emb.embedding_dim
    length = args.cache_length

    for batch in args.batch_sizes:
        print(f"\n===== batch {batch}, cache_length {length} =====")
        latent = torch.randn(
            (batch, model_dim), device=device, dtype=torch.bfloat16
        )
        key_masks = torch.ones((length, length), dtype=torch.bool, device=device).tril_()

        def bench(label: str, function, profile_label: str | None = None) -> None:
            span = args.measure_steps
            base = args.start_position

            def at(position: int) -> None:
                function(base + position % (length - base - 1))

            synchronized_seconds(at, args.warmup_steps)
            seconds = synchronized_seconds(at, span)
            print(f"{label}: {seconds / span * 1e3:.3f} ms/step")
            if profile_label:
                profile_launches(at, args.profile_steps, profile_label)

        # --- eager narrow (the current production path) ---
        caches = wrapper.make_generation_cache(
            batch, length, device, dtype=torch.bfloat16
        )

        def eager_step(position: int) -> None:
            wrapper.step_core(latent[:, None], caches, position)

        with torch.no_grad():
            bench("eager narrow", eager_step)

        # --- manual CUDA graph over the EAGER static-shape step ---
        # Captured BEFORE any reduce-overhead compile so dynamo's cudagraph
        # trees cannot interfere with the capture.
        manual_caches = wrapper.make_static_generation_cache(
            batch, length, device, dtype=torch.bfloat16
        )
        manual_position = torch.full(
            (), args.start_position, dtype=torch.long, device=device
        )
        manual_mask = key_masks[args.start_position].clone()
        manual_latent = latent[:, None].clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                wrapper.step_core(
                    manual_latent, manual_caches, manual_position, manual_mask
                )
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            wrapper.step_core(
                manual_latent, manual_caches, manual_position, manual_mask
            )

        def manual_step(position: int) -> None:
            manual_position.fill_(position)
            manual_mask.copy_(key_masks[position])
            manual_latent.copy_(latent[:, None])
            graph.replay()

        bench("manual cuda graph", manual_step, "manual cuda graph")

        # --- compiled static (reduce-overhead), the jobs-136/138 path ---
        torch._dynamo.reset()
        compiled_core = torch.compile(
            wrapper.step_core, mode="reduce-overhead", fullgraph=True, dynamic=False
        )
        static_caches = wrapper.make_static_generation_cache(
            batch, length, device, dtype=torch.bfloat16
        )
        position_index = torch.zeros((), dtype=torch.long, device=device)

        def compiled_step(position: int) -> None:
            position_index.fill_(position)
            compiled_core(
                latent[:, None], static_caches, position_index, key_masks[position]
            )

        with torch.no_grad():
            bench("compiled static", compiled_step, "compiled static")

        # --- compiled dynamic row-mask (the production TRAINING tail) ---
        # Tensor position + growing (batch, position+1) mask is exactly the
        # left-padded tail after compaction: narrow shapes, dynamic compile.
        torch._dynamo.reset()
        compiled_dynamic = torch.compile(
            wrapper.step_core,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
            dynamic=True,
        )
        dynamic_caches = wrapper.make_generation_cache(
            batch, length, device, dtype=torch.bfloat16
        )
        row_valid = torch.ones((batch, length), dtype=torch.bool, device=device)

        def dynamic_row_step(position: int) -> None:
            position_index.fill_(position)
            compiled_dynamic(
                latent[:, None],
                dynamic_caches,
                position_index,
                row_valid[:, : position + 1],
            )

        with torch.no_grad():
            bench(
                "compiled dynamic row-mask (production tail)",
                dynamic_row_step,
                "compiled dynamic row-mask",
            )

        # --- compiled static row-mask (the --rollout-tail-graph path) ---
        # Fixed-width (batch, cache_length) mask extended one column per
        # step, as rollout_continuations' static-tail switch does.
        torch._dynamo.reset()
        compiled_row_core = torch.compile(
            wrapper.step_core, mode="reduce-overhead", fullgraph=True, dynamic=False
        )
        static_row_caches = wrapper.make_static_generation_cache(
            batch, length, device, dtype=torch.bfloat16
        )
        row_mask = (
            key_masks[args.start_position][None].expand(batch, length).clone()
        )

        def compiled_row_step(position: int) -> None:
            position_index.fill_(position)
            row_mask[:, position] = True
            compiled_row_core(
                latent[:, None], static_row_caches, position_index, row_mask
            )

        with torch.no_grad():
            bench(
                "compiled static row-mask (tail graph)",
                compiled_row_step,
                "compiled static row-mask",
            )


if __name__ == "__main__":
    main()
