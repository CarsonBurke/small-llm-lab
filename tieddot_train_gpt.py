"""tieddot_train_gpt.py

Single-factor fork of the challenge baseline (train_gpt.py, imported
unmodified): the tied readout is reparametrized as the mini program's
winning tieddot head (nanogpt_mini_tieddot_train.py, H18/H21):

    logits = softcap * tanh((s * (x @ rms_norm(tok_emb).T) + b) / softcap)

vs the baseline's `softcap * tanh((x @ tok_emb.T) / softcap)`. Changes:
the codebook rows are rms-normalized (row norms no longer carry
frequency), a zero-init scalar gain s recovers the logit scale, and a
zero-init per-token bias b carries the unigram log-odds. Costs 1025
params. Both new params are zero-init, so logits are exactly 0 at init
(the mini fork's zero-init-projection property) and no RNG is consumed:
every parent parameter initializes bit-identically to the baseline.

s and b are registered under blocks[-1] (family precedent): train_gpt's
optimizer split routes blocks params with ndim < 2 to the Adam scalar
group (SCALAR_LR) and restore_low_dim_params_to_fp32 gives them fp32
masters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import train_gpt as baseline


class TiedDotGPT(baseline.GPT):
    """Challenge baseline with the mini program's tieddot readout."""

    def __init__(self, *args, **kwargs):
        vocab_size = kwargs.get("vocab_size", args[0] if args else None)
        super().__init__(*args, **kwargs)
        assert self.tie_embeddings, "the tieddot fork targets the tied head"
        # Shape (1,) rather than (): broadcasts identically, and avoids being
        # the model's first 0-dim param through the fused Adam path.
        self.blocks[-1].head_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.blocks[-1].head_bias = nn.Parameter(
            torch.zeros(vocab_size, dtype=torch.float32)
        )

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        # Parent forward verbatim except for the tieddot readout.
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
        codebook = F.rms_norm(self.tok_emb.weight, (self.tok_emb.weight.size(-1),))
        logits_proj = (
            F.linear(x, codebook.type_as(x)) * self.blocks[-1].head_scale.to(x.dtype)
            + self.blocks[-1].head_bias.to(x.dtype)
        )
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


baseline.GPT = TiedDotGPT

if __name__ == "__main__":
    baseline.main()
