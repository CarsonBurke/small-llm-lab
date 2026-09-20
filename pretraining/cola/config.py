"""Parameter-matched BPE and byte Cola controls with explicit architecture choices."""

from __future__ import annotations

from dataclasses import dataclass

ARCHITECTURE = "cola_nanogpt_120m_v3"


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    latent_dim: int = 16
    vae_dim: int = 512
    vae_layers: int = 4
    vae_heads: int = 8
    vae_ffn_dim: int = 2048
    dit_dim: int = 640
    dit_layers: int = 13
    dit_heads: int = 10
    dit_mlp_ratio: int = 4
    dit_rope_dim: int = 48
    block_size: int = 80
    seq_len: int = 2560
    rope_theta: float = 500000.0
    dit_rope_theta: float = 10000.0
    vae_norm_eps: float = 1e-6
    dit_norm_eps: float = 1e-5

    def __post_init__(self):
        if self.vocab_size not in (256, 100278):
            raise ValueError(
                "control vocabulary must be literal bytes or released Cola BPE"
            )
        for name in (
            "latent_dim",
            "vae_dim",
            "vae_layers",
            "vae_heads",
            "vae_ffn_dim",
            "dit_dim",
            "dit_layers",
            "dit_heads",
            "dit_mlp_ratio",
            "dit_rope_dim",
            "block_size",
            "seq_len",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.vae_dim % self.vae_heads or self.dit_dim % self.dit_heads:
            raise ValueError("attention width must be divisible by head count")
        if self.dit_dim // self.dit_heads < 16:
            raise ValueError(
                "compiled FlexAttention requires DiT head width at least 16"
            )
        if self.vae_ffn_dim % 2:
            raise ValueError("VAE SwiGLU projection width must be even")
        if self.dit_rope_dim % 2 or self.dit_rope_dim > self.dit_dim // self.dit_heads:
            raise ValueError("DiT rotary width must be even and fit an attention head")
        if self.vae_dim // self.vae_heads % 2:
            raise ValueError("VAE attention head width must be even")
        if (
            min(self.rope_theta, self.dit_rope_theta) <= 1
            or min(self.vae_norm_eps, self.dit_norm_eps) <= 0
        ):
            raise ValueError("invalid rotary base or normalization epsilon")

    @property
    def mask_token_id(self) -> int:
        return self.vocab_size


def control_config(tokenization: str) -> ModelConfig:
    if tokenization == "byte":
        return ModelConfig()
    if tokenization == "bpe":
        return ModelConfig(
            vocab_size=100278,
            vae_dim=256,
            vae_heads=4,
            vae_ffn_dim=1024,
            dit_layers=8,
            seq_len=512,
            block_size=16,
        )
    raise ValueError("tokenization must be byte or bpe")
