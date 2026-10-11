"""The fused CWM core (``fused.FusedCore``) on CUDA graphs, as an autograd
function of the perception output and the prepared weights.

Two graphs cover training, each one chunk of ``chunk`` tokens:

* ``forward``: advance the bank state through the chunk, append its store
  entries and save its activations (training). The forward sweep replays it
  once per chunk, saving each chunk's entry bank state (the boundary
  checkpoints).
* ``backward``: rebuild the chunk from its boundary and the saved
  activations, propagate the output gradient and the carried state
  gradients back through it token by token (exact BPTT), and add the
  chunk's weight gradients. The backward sweep replays it from the last
  chunk to the first.

Both graphs address the chunk's tokens through the device scalar ``pos0``.

Off CUDA the same procedures run eagerly (CPU tests, with Triton's
interpreter). ``pretraining/tests/test_cwm_fused.py`` checks values and
gradients against the autograd module path (``CWM.core``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from pretraining.cwm.fused import PREPARED, FusedCore, prepare
from pretraining.cwm.model import CWM


class GraphedCore:
    """One model, batch size, sequence length and feedback mode
    (``model.feedback`` is read at construction). ``train=False`` keeps no
    gradient buffers or boundaries and captures only the forward graph."""

    def __init__(self, model: CWM, batch: int, seq_len: int, chunk: int, train: bool, graphs: bool | None = None, warmup: int = 2):
        self.model, self.batch, self.seq_len, self.chunk, self.train = model, batch, seq_len, chunk, train
        self.feedback = model.feedback
        self.chunks = seq_len // chunk
        self.core = FusedCore(model, batch, seq_len, chunk, train, model.embed.weight.dtype)
        device = self.core.device
        self.graphs = device.type == "cuda" if graphs is None else graphs
        self.pending = False
        if self.graphs:
            self._capture(warmup)

    def _capture(self, warmup: int) -> None:
        core = self.core
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            # Warmup compiles every Triton specialization and lets cuBLAS
            # pick its kernels before capture.
            for _ in range(warmup):
                core.forward_chunk()
                if self.train:
                    core.backward_chunk()
        torch.cuda.current_stream().wait_stream(side)
        pool = torch.cuda.graph_pool_handle()
        with torch.no_grad():
            self.forward_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.forward_graph, pool=pool):
                core.forward_chunk()
            if self.train:
                self.backward_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.backward_graph, pool=pool):
                    core.backward_chunk()

    def _check(self, u: Tensor) -> None:
        if u.shape != (self.batch, self.seq_len, self.model.cfg.dim) or u.dtype != self.core.dtype:
            raise ValueError(f"the core was built for {(self.batch, self.seq_len)} {self.core.dtype}, got {tuple(u.shape)} {u.dtype}")
        if self.model.feedback != self.feedback:
            raise RuntimeError("model feedback mode differs from the one this core was built for")

    @torch.no_grad()
    def forward(self, u: Tensor, save: bool) -> Tensor:
        """Core outputs ``[B, T, d]`` with the weights last ``load``ed."""
        self._check(u)
        if save and not self.train:
            raise RuntimeError("an evaluation core keeps no boundaries")
        if save and self.pending:
            raise RuntimeError("the previous training forward was never backpropagated")
        core, C = self.core, self.chunk
        out = torch.empty_like(u)
        core.M[0].zero_()
        for c in range(self.chunks):
            span = slice(c * C, (c + 1) * C)
            if save:
                core.boundaries[c].copy_(core.M[0])
            core.U.copy_(u[:, span].transpose(0, 1))
            core.pos0.fill_(c * C)
            if self.graphs:
                self.forward_graph.replay()
            else:
                core.forward_chunk()
            out[:, span].copy_(core.HC[-1].transpose(0, 1))
        self.pending = save
        return out

    @torch.no_grad()
    def backward(self, d_out: Tensor, u: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Gradients of the input and of every prepared weight (``PREPARED``
        order), for the last saved forward."""
        if not self.pending:
            raise RuntimeError("backward without a saved training forward")
        core, C = self.core, self.chunk
        core.reset_backward()
        d_u = torch.empty_like(u)
        for c in reversed(range(self.chunks)):
            span = slice(c * C, (c + 1) * C)
            core.M[0].copy_(core.boundaries[c])
            core.U.copy_(u[:, span].transpose(0, 1))
            core.dOut.copy_(d_out[:, span].transpose(0, 1))
            core.pos0.fill_(c * C)
            if self.graphs:
                self.backward_graph.replay()
            else:
                core.backward_chunk()
            d_u[:, span].copy_(core.dU.transpose(0, 1))
        self.pending = False
        # Copies: autograd may keep a returned gradient as a parameter's
        # ``.grad``, and these buffers are reused by the next backward.
        return d_u, [core.grads[name].clone() for name in PREPARED]


class _CoreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: Tensor, core: GraphedCore, *prepared: Tensor) -> Tensor:
        core.core.load(dict(zip(PREPARED, prepared)))
        ctx.core = core
        ctx.save_for_backward(u)
        return core.forward(u, save=True)

    @staticmethod
    def backward(ctx, d_out: Tensor):
        (u,) = ctx.saved_tensors
        d_u, grads = ctx.core.backward(d_out.contiguous(), u)
        return (d_u, None, *grads)


def graphed_core(core: GraphedCore, u: Tensor) -> Tensor:
    """Differentiable core output ``[B, T, d]``: gradients reach ``u`` and,
    through ``fused.prepare``, the model's core parameters."""
    prepared = prepare(core.model)
    if not torch.is_grad_enabled():
        core.core.load(prepared)
        return core.forward(u, save=False)
    return _CoreFunction.apply(u, core, *prepared.values())
