"""Compiled exact recurrent loss and CUDA-graph microbatch execution.

Graph preparation executes full forward/backward passes without optimizer
updates. It is runtime preparation, not a reduced training experiment. All
model execution, including construction of CUDAGraphMicrobatch, belongs in mlq.
"""

from __future__ import annotations

import gc
import time

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


class RecurrentLoss:
    """Sum next-token CE with full recurrent BPTT and bounded head storage."""

    def __init__(self, model, segment_size: int = 16):
        if segment_size < 1:
            raise ValueError("segment_size must be positive")
        self.model = model
        self.segment_size = segment_size
        if model._compiled_segment is None:
            model.compile_segments()
        self.head_loss = torch.compile(self._head_loss, fullgraph=True, dynamic=False)

    def _head_loss(self, hidden: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(self.model.logits(hidden).float(), targets, reduction="sum")

    def __call__(self, inputs: Tensor, targets: Tensor, *, diagnostics: bool = False):
        if inputs.device.type != "cuda" or targets.device != inputs.device:
            raise ValueError("Recurrent loss requires inputs and targets on one CUDA device")
        if inputs.ndim != 2 or inputs.shape != targets.shape:
            raise ValueError("inputs and targets must have identical [batch,time] shapes")
        hidden, memory = self.model.forward_hidden(inputs, segment_size=self.segment_size)
        hidden = hidden.flatten(0, 1)
        targets = targets.flatten()
        losses = []
        # One split gives backward a single concatenation of disjoint gradients.
        # Independent slices each allocate/zero a full hidden-gradient tensor
        # and then add those tensors, making large recurrent batches expensive.
        for h, y in zip(hidden.split(4096), targets.split(4096)):
            if torch.is_grad_enabled():
                # There are no random operations in the head. Avoid generator
                # get/set calls, which are unnecessary and unsafe in capture.
                losses.append(checkpoint(self.head_loss, h, y, use_reentrant=False,
                                         preserve_rng_state=False))
            else:
                losses.append(self.head_loss(h, y))
        loss = torch.stack(losses).sum()
        if diagnostics:
            memory = memory.float()
            return loss, memory.square().mean(), memory.var(dim=1, correction=0).mean()
        return loss


def loss_sum(model, inputs: Tensor, targets: Tensor, segment_size: int = 16):
    """Convenience API; cache compiled loss wrappers on the owning model."""
    helpers = getattr(model, "_recurrent_loss_helpers", None)
    if helpers is None:
        helpers = {}
        model._recurrent_loss_helpers = helpers
    if segment_size not in helpers:
        helpers[segment_size] = RecurrentLoss(model, segment_size)
    return helpers[segment_size](inputs, targets)


def _copy_batch(inputs: Tensor, targets: Tensor, static_inputs: Tensor, static_targets: Tensor):
    if inputs.shape != static_inputs.shape or targets.shape != static_targets.shape:
        raise ValueError(f"CUDA graph requires inputs and targets of shape {tuple(static_inputs.shape)}")
    if inputs.device != static_inputs.device or targets.device != static_targets.device:
        raise ValueError("Replay inputs and targets must be on the captured CUDA device")
    if inputs.dtype != torch.int32 or targets.dtype != torch.int64:
        raise ValueError("Replay requires int32 inputs and int64 targets")
    static_inputs.copy_(inputs)
    static_targets.copy_(targets)


class CUDAGraphMicrobatch:
    """Replay exact full-sequence forward/backward into stable grad buffers.

    Call zero_grad() once before an optimizer update, replay each microbatch,
    then step the optimizers. Gradient zeroing is deliberately outside capture:
    repeated replays accumulate, just like repeated loss.backward() calls.

    Preparation discards any existing gradients. Parameters remain unchanged.
    Never use model.zero_grad(set_to_none=True), replace parameter storage, or
    change model mode while this executor is active. The returned detached loss
    aliases graph output storage; consume it on the replay stream or clone it
    before storing it across another replay.
    """

    def __init__(self, loss_fn: RecurrentLoss, batch_size: int = 64,
                 seq_len: int = 1024, vocab_size: int = 1024):
        if min(batch_size, seq_len, vocab_size) < 1:
            raise ValueError("batch_size, seq_len and vocab_size must be positive")
        if seq_len % loss_fn.segment_size:
            raise ValueError("seq_len must be divisible by the compiled segment size")
        self.loss_fn = loss_fn
        self.model = loss_fn.model
        if not self.model.training:
            raise ValueError("CUDA training graph must be prepared in model.train() mode")
        self._parameters = tuple(self.model.named_parameters())
        if not self._parameters:
            raise ValueError("Model has no parameters")
        device = self._parameters[0][1].device
        if device.type != "cuda" or any(p.device != device for _, p in self._parameters):
            raise ValueError("CUDA graph requires all model parameters on one CUDA device")
        if vocab_size != self.model.config["vocab_size"]:
            raise ValueError("vocab_size must match the model")
        self.device = device
        self.shape = (batch_size, seq_len)
        preparation_start = time.perf_counter()
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("CUDA graph requires bf16 support")
            # Nonreentrant checkpoint hooks can form unreachable graph cycles.
            # Collect ordinary-stream AccumulateGrad nodes before constructing
            # a new backward graph on the preparation stream.
            gc.collect()
            self.static_inputs = torch.arange(batch_size * seq_len, device=device,
                                              dtype=torch.int32).remainder(vocab_size).view(self.shape)
            self.static_targets = (self.static_inputs.long() + 1).remainder(vocab_size)
            # Allocate before warmup and capture so AccumulateGrad records adds
            # into stable storage instead of first-backward assignments.
            for _, parameter in self._parameters:
                parameter.grad = torch.zeros_like(parameter)
            self.gradient_buffers = tuple(parameter.grad for _, parameter in self._parameters)
            self._gradient_pointers = tuple(gradient.data_ptr() for gradient in self.gradient_buffers)
            self._parameter_pointers = tuple(parameter.data_ptr() for _, parameter in self._parameters)

            current_stream = torch.cuda.current_stream(device)
            preparation_stream = torch.cuda.Stream(device=device)
            self.capture_stream = preparation_stream
            preparation_stream.wait_stream(current_stream)
            with torch.cuda.stream(preparation_stream):
                for _ in range(3):
                    self.zero_grad()
                    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                        loss = self.loss_fn(self.static_inputs, self.static_targets)
                        loss.backward()
                        del loss
                self.zero_grad()
            current_stream.wait_stream(preparation_stream)
            current_stream.synchronize()
            self._check_storage()

            capture_start = time.perf_counter()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=preparation_stream):
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    self.static_loss = self.loss_fn(self.static_inputs, self.static_targets)
                    self.static_loss.backward()
            # Replay needs tensor storage, not a live Python autograd graph.
            # The CUDA graph owns its private allocator pool for the lifetime
            # of self.graph; detaching the output preserves its data pointer.
            self.static_loss = self.static_loss.detach()
            gc.collect()
            current_stream.wait_stream(preparation_stream)
            current_stream.synchronize()
            self.capture_seconds = time.perf_counter() - capture_start
            self._check_storage()
            self.zero_grad()
            current_stream.synchronize()
        self.preparation_seconds = time.perf_counter() - preparation_start

    def _check_storage(self):
        for (name, parameter), gradient, gradient_pointer, parameter_pointer in zip(
                self._parameters, self.gradient_buffers,
                self._gradient_pointers, self._parameter_pointers):
            if parameter.grad is not gradient or gradient.data_ptr() != gradient_pointer:
                raise RuntimeError(f"Gradient storage replaced for {name}; use zero_grad(set_to_none=False)")
            if parameter.data_ptr() != parameter_pointer:
                raise RuntimeError(f"Captured parameter storage replaced for {name}")

    def zero_grad(self):
        """Clear one optimizer update's gradients without changing storage."""
        self._check_storage()
        torch._foreach_zero_(self.gradient_buffers)

    def replay(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if not self.model.training:
            raise RuntimeError("CUDA training replay requires model.train() mode")
        self._check_storage()
        with torch.cuda.device(self.device):
            _copy_batch(inputs, targets, self.static_inputs, self.static_targets)
            self.graph.replay()
        return self.static_loss


class CUDAGraphValidation:
    """Full-sequence inference graph returning loss and memory diagnostics.

    Prepare and replay in model.eval() mode. Preparation and replay never clear,
    replace, or accumulate parameter gradients, so this can coexist with the
    training graph's stable gradient buffers. Consume each returned output
    immediately on the replay stream: the next replay overwrites its storage.
    """

    def __init__(self, loss_fn: RecurrentLoss, batch_size: int = 64,
                 seq_len: int = 1024, vocab_size: int = 1024):
        if min(batch_size, seq_len, vocab_size) < 1:
            raise ValueError("batch_size, seq_len and vocab_size must be positive")
        if seq_len % loss_fn.segment_size:
            raise ValueError("seq_len must be divisible by the compiled segment size")
        self.loss_fn = loss_fn
        self.model = loss_fn.model
        if self.model.training:
            raise ValueError("Validation graph must be prepared in model.eval() mode")
        self._parameters = tuple(self.model.named_parameters())
        if not self._parameters:
            raise ValueError("Model has no parameters")
        device = self._parameters[0][1].device
        if device.type != "cuda" or any(p.device != device for _, p in self._parameters):
            raise ValueError("CUDA graph requires all model parameters on one CUDA device")
        if vocab_size != self.model.config["vocab_size"]:
            raise ValueError("vocab_size must match the model")
        self.device = device
        self.shape = (batch_size, seq_len)
        self._parameter_pointers = tuple(p.data_ptr() for _, p in self._parameters)
        preparation_start = time.perf_counter()
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("CUDA graph requires bf16 support")
            self.static_inputs = torch.arange(batch_size * seq_len, device=device,
                                              dtype=torch.int32).remainder(vocab_size).view(self.shape)
            self.static_targets = (self.static_inputs.long() + 1).remainder(vocab_size)
            current_stream = torch.cuda.current_stream(device)
            preparation_stream = torch.cuda.Stream(device=device)
            self.capture_stream = preparation_stream
            preparation_stream.wait_stream(current_stream)
            with torch.cuda.stream(preparation_stream), torch.no_grad():
                for _ in range(3):
                    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                        self.loss_fn(self.static_inputs, self.static_targets, diagnostics=True)
            current_stream.wait_stream(preparation_stream)
            current_stream.synchronize()

            capture_start = time.perf_counter()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=preparation_stream), torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    self.static_outputs = self.loss_fn(
                        self.static_inputs, self.static_targets, diagnostics=True)
            current_stream.wait_stream(preparation_stream)
            current_stream.synchronize()
            self.capture_seconds = time.perf_counter() - capture_start
        self.preparation_seconds = time.perf_counter() - preparation_start

    def replay(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.model.training:
            raise RuntimeError("Validation replay requires model.eval() mode")
        for (name, parameter), pointer in zip(self._parameters, self._parameter_pointers):
            if parameter.data_ptr() != pointer:
                raise RuntimeError(f"Captured parameter storage replaced for {name}")
        with torch.cuda.device(self.device):
            _copy_batch(inputs, targets, self.static_inputs, self.static_targets)
            self.graph.replay()
        return self.static_outputs


class ConcurrentMicrobatches:
    """Execute independent microbatch graphs, then reduce in original row order.

    Every replica has distinct Parameter objects and gradient buffers, while
    its parameter storage aliases the master's read-only weights. The master
    must be updated only after replay returns, on the stream that called it.
    Stream waits order the update after all worker reads and gradient reduction.

    A replay computes one complete update's summed loss and gradients; it clears
    previous gradients and never steps an optimizer. Microbatch shapes remain
    unchanged. With the defaults, eight B64 graphs replace eight serial B64
    backward calls, without substituting a numerically different B512 kernel.
    Lower concurrency schedules successive waves using the same bounded pool.
    The returned loss aliases storage overwritten by the next replay.
    """

    def __init__(self, model, microbatch: int = 64, rows: int = 512,
                 segment_size: int = 16, concurrency: int = 8):
        from pretraining.nanogpt_mini.recurrent_slots import RecurrentSlots

        if min(microbatch, rows, segment_size, concurrency) < 1:
            raise ValueError("microbatch, rows, segment_size and concurrency must be positive")
        if rows % microbatch:
            raise ValueError("rows must be divisible by microbatch")
        if 1024 % segment_size:
            raise ValueError("segment_size must divide the 1024-token context")
        if not model.training:
            raise ValueError("Concurrent training must be prepared in model.train() mode")
        self.model = model
        self.microbatch = microbatch
        self.rows = rows
        self.shape = (rows, 1024)
        self.concurrency = min(concurrency, rows // microbatch)
        self._parameters = tuple(model.named_parameters())
        if not self._parameters:
            raise ValueError("Model has no parameters")
        self.device = self._parameters[0][1].device
        if self.device.type != "cuda" or any(p.device != self.device for _, p in self._parameters):
            raise ValueError("Concurrent training requires all parameters on one CUDA device")
        if tuple(model.named_buffers()):
            raise ValueError("Shared-storage replicas require the buffer-free recurrent model")
        self._parameter_pointers = tuple(p.data_ptr() for _, p in self._parameters)
        preparation_start = time.perf_counter()
        self.engines = []
        self.streams = []
        with torch.cuda.device(self.device):
            for _, parameter in self._parameters:
                parameter.grad = torch.zeros_like(parameter)
            self.gradient_buffers = tuple(p.grad for _, p in self._parameters)
            self._gradient_pointers = tuple(g.data_ptr() for g in self.gradient_buffers)
            self.total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            for _ in range(self.concurrency):
                # Construction initializes temporary weights. Preserve RNG and
                # replace their storage; do not deepcopy compiled bound methods
                # or reuse the master's Parameter/AccumulateGrad objects.
                with torch.random.fork_rng(devices=[self.device.index]):
                    replica = RecurrentSlots(**model.config).to(self.device).train()
                replica_parameters = tuple(replica.named_parameters())
                if [name for name, _ in replica_parameters] != [name for name, _ in self._parameters]:
                    raise ValueError("Replica parameter names differ from master")
                with torch.no_grad():
                    for (name, parameter), (_, master) in zip(replica_parameters, self._parameters):
                        if parameter.shape != master.shape or parameter.dtype != master.dtype:
                            raise ValueError(f"Replica metadata differs for {name}")
                        parameter.set_(master.detach())
                        if parameter is master or parameter.data_ptr() != master.data_ptr():
                            raise RuntimeError(f"Replica storage sharing failed for {name}")
                engine = CUDAGraphMicrobatch(
                    RecurrentLoss(replica, segment_size), batch_size=microbatch,
                    seq_len=1024, vocab_size=model.config["vocab_size"])
                if any(stream.cuda_stream == engine.capture_stream.cuda_stream
                       for stream in self.streams):
                    raise RuntimeError("Concurrent captures require distinct CUDA stream handles")
                self.engines.append(engine)
                # cuBLAS workspaces can be keyed to the capture stream. Keep
                # that stream alive and replay on it, not another pooled stream.
                self.streams.append(engine.capture_stream)
            self.zero_grad()
            torch.cuda.current_stream(self.device).synchronize()
            self.reserved_mib = torch.cuda.memory_reserved(self.device) / 2**20
            self.peak_allocated_mib = torch.cuda.max_memory_allocated(self.device) / 2**20
            self.free_device_bytes, self.total_device_bytes = torch.cuda.mem_get_info(self.device)
        self.capture_seconds = sum(engine.capture_seconds for engine in self.engines)
        self.preparation_seconds = time.perf_counter() - preparation_start

    def _check_master_storage(self):
        for (name, parameter), gradient, parameter_pointer, gradient_pointer in zip(
                self._parameters, self.gradient_buffers,
                self._parameter_pointers, self._gradient_pointers):
            if self.model.get_parameter(name) is not parameter or parameter.data_ptr() != parameter_pointer:
                raise RuntimeError(f"Shared master parameter storage replaced for {name}")
            if parameter.grad is not gradient or gradient.data_ptr() != gradient_pointer:
                raise RuntimeError(f"Master gradient storage replaced for {name}; preserve gradient buffers")

    def zero_grad(self):
        """Clear master and replica gradients while preserving captured pointers."""
        self._check_master_storage()
        with torch.cuda.device(self.device):
            torch._foreach_zero_(self.gradient_buffers)
            self.total_loss.zero_()
            for engine in self.engines:
                engine.zero_grad()

    def replay(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if not self.model.training:
            raise RuntimeError("Concurrent training requires model.train() mode")
        if tuple(inputs.shape) != self.shape or tuple(targets.shape) != self.shape:
            raise ValueError(f"Concurrent update requires inputs and targets of shape {self.shape}")
        if inputs.device != self.device or targets.device != self.device:
            raise ValueError("Inputs and targets must be on the master's CUDA device")
        if inputs.dtype != torch.int32 or targets.dtype != torch.int64:
            raise ValueError("Concurrent update requires int32 inputs and int64 targets")
        self._check_master_storage()
        with torch.cuda.device(self.device):
            current = torch.cuda.current_stream(self.device)
            torch._foreach_zero_(self.gradient_buffers)
            self.total_loss.zero_()
            for first_row in range(0, self.rows, self.concurrency * self.microbatch):
                active = min(self.concurrency, (self.rows - first_row) // self.microbatch)
                for index in range(active):
                    stream, engine = self.streams[index], self.engines[index]
                    # This also waits for a previous wave's ordered reduction
                    # before reusing any worker's gradient/output buffers.
                    stream.wait_stream(current)
                    row = first_row + index * self.microbatch
                    with torch.cuda.stream(stream):
                        engine.zero_grad()
                        engine.replay(inputs[row:row + self.microbatch],
                                      targets[row:row + self.microbatch])
                for stream in self.streams[:active]:
                    current.wait_stream(stream)
                for engine in self.engines[:active]:
                    # Keep the original microbatch order, including bf16
                    # embedding rounding after each individual contribution.
                    torch._foreach_add_(self.gradient_buffers, engine.gradient_buffers)
                    self.total_loss.add_(engine.static_loss)
        return self.total_loss

    def full_update(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Alias emphasizing that replay computes all rows, without optimizer work."""
        return self.replay(inputs, targets)
