"""Experimental query-shift execution of the dense single-writer GDN2 pool.

Each reader gets a replicated writer head. Upper/current readers query the
inclusive state; lower readers shift queries left and outputs right to query
the previous-token state. This removes read-only recurrent events, preserving
the mathematical recurrence but changing BF16 chunk boundaries. Qualification
must therefore measure numerical agreement as well as complete-update speed.
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from pretraining.nanogpt_mini import gated_delta_ops
from pretraining.nanogpt_mini.gated_delta_pool import (
    PoolPort, SharedPoolGatedDeltaGPT, balance_loss, z_loss,
)


class QueryShiftSharedPoolGPT(SharedPoolGatedDeltaGPT):
    """Same parameters/configuration as the dense pool, alternate execution."""

    def __init__(self, **options):
        super().__init__(**options)
        if (self.config["pool_writer"] != "layer"
                or self.config["pool_reads"] != self.config["pool_banks"]):
            raise ValueError("Query shifts require layer-owned banks and all-bank reads")

    def _pool(self, ports: list[PoolPort]) -> tuple[list[list[Tensor]], dict[str, Tensor]]:
        config = self.config
        sources, heads, dim, chunk = (config[k] for k in
                                      ("pool_banks", "pool_heads", "head_dim", "kernel_chunk_size"))
        readers = len(ports)
        batch, time = ports[0].write.shape
        width = heads * dim
        q, k, v, g, b, w = self.pool(torch.stack([port.hidden for port in ports]))

        def adapt_writer(field, matrix):
            return torch.bmm(field.reshape(sources, batch * time, width),
                             matrix.to(field.dtype)).view(sources, batch, time, heads, dim)

        # Full latent adapters act before heads are replicated; a replica is
        # a separate state, never an additional channel in the adapter.
        k, v = adapt_writer(k, self.adapters.key), adapt_writer(v, self.adapters.value)
        queries = torch.matmul(q.reshape(1, readers, batch * time, width),
                               self.adapters.query[:, None].to(q.dtype))
        queries = queries.view(sources, readers, batch, time, heads, dim)
        queries = torch.stack([torch.stack([
            F.pad(queries[source, reader, :, 1:], (0, 0, 0, 0, 0, 1))
            if reader < source else queries[source, reader]
            for reader in range(readers)]) for source in range(sources)])
        queries = queries.permute(0, 2, 3, 1, 4, 5).reshape(1, sources * batch * time, readers * heads, dim)

        def replicate(field):
            # expand/reshape is differentiable and sums every copy's writer
            # gradient back into the original writer projections.
            return field.unsqueeze(3).expand(sources, batch, time, readers, heads, dim).reshape(
                1, sources * batch * time, readers * heads, dim).contiguous()

        segments = sources * batch
        per_segment = time // chunk
        local = torch.arange(segments * per_segment, device=q.device)
        cu_seqlens = (torch.arange(segments + 1, device=q.device) * time).to(torch.int32)
        chunk_indices = torch.stack((local // per_segment, local % per_segment), -1).to(torch.int32)
        output = gated_delta_ops.chunk_gdn2_training(
            queries, *(replicate(field) for field in (k, v, g, b, w)),
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            state_v_first=config["state_v_first"])[0]
        output = output.reshape(sources, batch, time, readers, heads, dim).permute(0, 3, 1, 2, 4, 5)
        output = torch.stack([torch.stack([
            F.pad(output[source, reader, :, :-1], (0, 0, 0, 0, 1, 0))
            if reader < source else output[source, reader]
            for reader in range(readers)]) for source in range(sources)])
        output = torch.matmul(output.reshape(sources, readers, batch * time, width),
                              self.adapters.output[:, None].to(output.dtype))
        output = output.view(sources, readers, batch, time, heads, dim)
        stats = dict(balance=sum(balance_loss(port.probs, port.reads, sources) for port in ports),
                     z=sum(z_loss(port.logits) for port in ports),
                     load_max=ports[0].probs.new_tensor(1 / sources))
        return [[output[source, reader] for source in range(sources)] for reader in range(readers)], stats
