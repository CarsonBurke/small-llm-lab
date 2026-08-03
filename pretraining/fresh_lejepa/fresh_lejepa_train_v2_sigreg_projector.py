"""V2 ablation: intra-token projector used only by SIGReg."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path

import torch.nn.functional as F
from torch import Tensor, nn

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
from pretraining.fresh_lejepa import fresh_lejepa_train_v2 as v2
from train_gpt import CastedLinear


ARCHITECTURE = "fresh_lejepa_v2_sigreg_only_token_projector"


class FP32BatchNorm1d(nn.BatchNorm1d):
    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Match mixed-BF16 training: affine parameters and running statistics
        # remain FP32 while surrounding matmuls use autocast BF16.
        for parameter in self.parameters(recurse=False):
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.float()
        for name, buffer in self.named_buffers(recurse=False):
            if buffer is not None and buffer.is_floating_point():
                setattr(self, name, buffer.float())
        return self


class TokenProjector(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.input = CastedLinear(model_dim, hidden_dim)
        self.norm = FP32BatchNorm1d(hidden_dim)
        self.output = CastedLinear(hidden_dim, model_dim)

    def forward(self, latent: Tensor) -> Tensor:
        shape = latent.shape
        hidden = self.input(latent).reshape(-1, self.norm.num_features)
        projected = self.output(F.gelu(self.norm(hidden)))
        return projected.reshape(*shape[:-1], -1)

    def inference(self, latent: Tensor) -> Tensor:
        shape = latent.shape
        hidden = self.input(latent).reshape(-1, self.norm.num_features)
        normalized = F.batch_norm(
            hidden, self.norm.running_mean, self.norm.running_var,
            self.norm.weight, self.norm.bias, training=False, eps=self.norm.eps,
        )
        return self.output(F.gelu(normalized)).reshape(*shape[:-1], -1)


class FreshLeJEPAV2SIGRegProjector(v2.FreshLeJEPAGPTV2):
    projector_class = TokenProjector

    def __init__(self, *args, **kwargs):
        model_dim = kwargs.get("model_dim", args[2] if len(args) > 2 else None)
        super().__init__(*args, **kwargs)
        if model_dim is None:
            raise ValueError("model_dim is required")
        self.blocks[-1].latent_projector = self.projector_class(model_dim)

    @property
    def latent_projector(self) -> TokenProjector:
        return self.blocks[-1].latent_projector

    def sigreg_features(self, token_latent: Tensor) -> Tensor:
        return self.latent_projector(token_latent)


def main() -> None:
    v1.FreshLeJEPAGPT = FreshLeJEPAV2SIGRegProjector
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    v1.main()


if __name__ == "__main__":
    main()
