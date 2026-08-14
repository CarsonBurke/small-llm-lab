"""Bounded CPU-collation and CUDA-transfer overlap for model batches."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Callable, Generator, Iterable, Iterator, Protocol, TypeVar

import torch


class DeviceTransferable(Protocol):
    def pin_memory(self) -> "DeviceTransferable": ...

    def to(
        self, device: torch.device, *, non_blocking: bool = False
    ) -> "DeviceTransferable": ...

    def record_stream(self, stream: torch.cuda.Stream) -> None: ...


Key = TypeVar("Key")
Batch = TypeVar("Batch", bound=DeviceTransferable)


class DeviceBatchPrefetcher:
    """Reusable single-worker CPU producer and CUDA transfer stream.

    A prefetcher owns the expensive lifecycle resources for a complete training
    or benchmark lifetime. At each yield, the next device transfer is already
    enqueued and one following host batch may be preparing. Individual
    ``batches`` iterators therefore remain bounded to one host batch and one
    device batch ahead of the caller. Iterators must be consumed sequentially.
    """

    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        self._closed = False
        self._active = False
        self._active_iterator: Generator[DeviceTransferable, None, None] | None = None
        if device.type == "cuda":
            self._transfer_stream: torch.cuda.Stream | None = torch.cuda.Stream(
                device=device
            )
            self._producer: ThreadPoolExecutor | None = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="byte-device-prefetch"
            )
        else:
            self._transfer_stream = None
            self._producer = None

    def __enter__(self) -> "DeviceBatchPrefetcher":
        if self._closed:
            raise RuntimeError("cannot re-enter a closed device batch prefetcher")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the persistent producer after all iterators are exhausted."""

        if self._closed:
            return
        if self._active:
            if self._active_iterator is None:
                raise AssertionError("active prefetch iterator is unavailable")
            self._active_iterator.close()
            if self._active:
                raise RuntimeError("could not close the active prefetch iterator")
        self._closed = True
        if self._producer is not None:
            self._producer.shutdown(wait=True, cancel_futures=True)
            self._producer = None
        self._transfer_stream = None

    def batches(
        self,
        keys: Iterable[Key],
        prepare_cpu: Callable[[Key], Batch],
    ) -> Iterator[Batch]:
        """Yield batches with one-batch CPU/copy lookahead."""

        if self._closed:
            raise RuntimeError("device batch prefetcher is closed")

        def iterate() -> Generator[Batch, None, None]:
            if self._active:
                raise RuntimeError(
                    "device batch prefetcher iterators must be consumed sequentially"
                )
            if self._closed:
                raise RuntimeError("device batch prefetcher is closed")
            self._active = True
            self._active_iterator = iterator
            pending: Future[Batch] | None = None
            try:
                source = iter(keys)
                try:
                    first_key = next(source)
                except StopIteration:
                    return

                if self.device.type != "cuda":
                    yield prepare_cpu(first_key)
                    for key in source:
                        yield prepare_cpu(key)
                    return

                transfer_stream = self._transfer_stream
                producer = self._producer
                if transfer_stream is None or producer is None:
                    raise AssertionError("CUDA prefetch resources are unavailable")
                execution_stream = torch.cuda.current_stream(self.device)

                def prepare_pinned(key: Key) -> Batch:
                    return prepare_cpu(key).pin_memory()  # type: ignore[return-value]

                def transfer(cpu_batch: Batch) -> tuple[Batch, torch.cuda.Event]:
                    with torch.cuda.stream(transfer_stream):
                        batch = cpu_batch.to(self.device, non_blocking=True)
                        ready = torch.cuda.Event()
                        ready.record(transfer_stream)
                    return batch, ready  # type: ignore[return-value]

                pending = producer.submit(prepare_pinned, first_key)
                try:
                    next_key = next(source)
                except StopIteration:
                    next_key = None
                current, current_ready = transfer(pending.result())
                pending = None
                following_device: tuple[Batch, torch.cuda.Event] | None = None
                if next_key is not None:
                    following_cpu = producer.submit(
                        prepare_pinned, next_key
                    ).result()
                    try:
                        next_key = next(source)
                    except StopIteration:
                        next_key = None
                    pending = (
                        None
                        if next_key is None
                        else producer.submit(prepare_pinned, next_key)
                    )
                    following_device = transfer(following_cpu)

                while True:
                    execution_stream.wait_event(current_ready)
                    current.record_stream(execution_stream)
                    yield current
                    if following_device is None:
                        break
                    current, current_ready = following_device
                    following_device = None
                    if pending is not None:
                        following_cpu = pending.result()
                        try:
                            next_key = next(source)
                        except StopIteration:
                            next_key = None
                        pending = (
                            None
                            if next_key is None
                            else producer.submit(prepare_pinned, next_key)
                        )
                        following_device = transfer(following_cpu)
            finally:
                if pending is not None and not pending.done():
                    pending.cancel()
                    wait((pending,))
                self._active_iterator = None
                self._active = False

        iterator = iterate()
        return iterator


def prefetched_device_batches(
    keys: Iterable[Key],
    prepare_cpu: Callable[[Key], Batch],
    *,
    device: torch.device,
) -> Iterator[Batch]:
    """Compatibility wrapper for one-off callers.

    Hot loops should instead retain one :class:`DeviceBatchPrefetcher` for their
    full lifetime so worker and CUDA-stream construction stay off the update
    path.
    """

    with DeviceBatchPrefetcher(device=device) as prefetcher:
        yield from prefetcher.batches(keys, prepare_cpu)


__all__ = (
    "DeviceBatchPrefetcher",
    "DeviceTransferable",
    "prefetched_device_batches",
)
