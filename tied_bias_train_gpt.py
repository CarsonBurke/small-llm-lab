"""tied_bias_train_gpt.py

Single-factor fork of the challenge baseline (train_gpt.py, imported
unmodified): the tied readout gains a per-token bias inside the softcap:

    logits = softcap * tanh((x @ tok_emb.T + b) / softcap)

Motivation: the nanogpt-mini decomposition program (energy_readout/IDEA.md,
H16-H21) found the winning mini head was a tied attached dot readout whose
per-token bias absorbed the unigram log-odds (|b| grew to ~3.4); the
challenge baseline's tied head has no bias term.  This adds the 1024-param
term alone, changing nothing else.

The bias is registered under blocks[-1] (family precedent, see
energy_readout/fresh_lejepa_train_energy_readout.py): train_gpt.main's
optimizer split only sees blocks.named_parameters() plus the
embedding/head/skip specials, so ndim < 2 routes it to the Adam scalar
group (SCALAR_LR) and restore_low_dim_params_to_fp32 gives it an fp32
master.  It is registered AFTER parent construction and is zero-init
(consumes no RNG), so every parent parameter initializes bit-identically
to the baseline at equal seed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import train_gpt as baseline


class TiedBiasGPT(baseline.GPT):
    """Challenge baseline with a per-token bias on the tied readout."""

    def __init__(self, *args, **kwargs):
        vocab_size = kwargs.get("vocab_size", args[0] if args else None)
        super().__init__(*args, **kwargs)
        assert self.tie_embeddings, "the bias fork targets the tied head"
        self.blocks[-1].head_bias = nn.Parameter(
            torch.zeros(vocab_size, dtype=torch.float32)
        )

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        # Parent forward verbatim except for the bias add on the tied logits.
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(x, self.tok_emb.weight) + self.blocks[-1].head_bias.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


baseline.GPT = TiedBiasGPT

if __name__ == "__main__":
    baseline.main()
