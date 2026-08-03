"""
CALM-style ablation adapted to the parameter-golf workflow.

This version is intentionally closer to ../calm than the earlier direct-CE attempt:
- stage 1: train a chunk autoencoder that maps K tokens <-> one continuous latent
- stage 2: freeze the autoencoder and train a chunk-level autoregressive model
           with an energy-style objective in latent space
- evaluation: decode predicted chunk latents back to token logits and report BPB

This still is not a literal port of the upstream HF trainer stack, but it preserves
the main published training decomposition that the prior attempt violated.
"""

from __future__ import annotations

import copy
import io
import math
import os
import random
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train_gpt as base  # noqa: E402


class Hyperparameters(base.Hyperparameters):
    calm_patch_size = int(os.environ.get("CALM_PATCH_SIZE", "4"))
    calm_latent_dim = int(os.environ.get("CALM_LATENT_DIM", "128"))
    calm_ae_layers = int(os.environ.get("CALM_AE_LAYERS", "2"))
    calm_ae_expansion = int(os.environ.get("CALM_AE_EXPANSION", "2"))
    calm_lm_layers = int(os.environ.get("CALM_LM_LAYERS", os.environ.get("NUM_LAYERS", "9")))
    calm_num_heads = int(os.environ.get("CALM_NUM_HEADS", os.environ.get("NUM_HEADS", "8")))
    calm_num_kv_heads = int(os.environ.get("CALM_NUM_KV_HEADS", os.environ.get("NUM_KV_HEADS", "4")))
    calm_noise_size = int(os.environ.get("CALM_NOISE_SIZE", "64"))
    calm_num_samples = int(os.environ.get("CALM_NUM_SAMPLES", "2"))
    calm_eval_samples = int(os.environ.get("CALM_EVAL_SAMPLES", "1"))
    calm_beta = float(os.environ.get("CALM_BETA", "1.0"))
    calm_ae_steps = int(os.environ.get("CALM_AE_STEPS", "40"))
    calm_lr = float(os.environ.get("CALM_LR", "3e-4"))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", "1.0"))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "0"))
    calm_ae_dropout = float(os.environ.get("CALM_AE_DROPOUT", "0.05"))
    calm_latent_l2 = float(os.environ.get("CALM_LATENT_L2", "1e-4"))


class ChunkMLPBlock(nn.Module):
    def __init__(self, dim: int, expansion: int):
        super().__init__()
        hidden = max(dim, dim * expansion)
        self.norm = base.RMSNorm()
        self.fc = base.CastedLinear(dim, hidden, bias=False)
        self.proj = base.CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj(F.silu(self.fc(self.norm(x))))


class ChunkAutoencoderEncoder(nn.Module):
    def __init__(self, model_dim: int, patch_size: int, latent_dim: int, num_layers: int, expansion: int, dropout: float):
        super().__init__()
        self.patch_size = patch_size
        self.dropout = dropout
        self.blocks = nn.ModuleList([ChunkMLPBlock(model_dim, expansion) for _ in range(num_layers)])
        self.squeeze = base.CastedLinear(patch_size * model_dim, model_dim, bias=False)
        self.norm = base.RMSNorm()
        self.to_latent = base.CastedLinear(model_dim, latent_dim, bias=False)

    def forward(self, tok_emb: nn.Embedding, pos_emb: Tensor, chunk_tokens: Tensor) -> Tensor:
        bsz, num_chunks, patch_size = chunk_tokens.shape
        if patch_size != self.patch_size:
            raise ValueError(f"Expected patch_size={self.patch_size}, got {patch_size}")
        x = tok_emb(chunk_tokens.reshape(bsz * num_chunks, patch_size))
        x = x + pos_emb.to(dtype=x.dtype)[None, :, :]
        if self.training and self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=True)
        for block in self.blocks:
            x = block(x)
        flat = x.reshape(bsz * num_chunks, -1)
        hidden = self.norm(self.squeeze(flat))
        latent = self.to_latent(hidden)
        return latent.reshape(bsz, num_chunks, -1)


class ChunkAutoencoderDecoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        patch_size: int,
        latent_dim: int,
        num_layers: int,
        expansion: int,
        logit_softcap: float,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.logit_softcap = logit_softcap
        self.latent_proj = base.CastedLinear(latent_dim, model_dim, bias=False)
        self.expand = base.CastedLinear(model_dim, patch_size * model_dim, bias=False)
        self.blocks = nn.ModuleList([ChunkMLPBlock(model_dim, expansion) for _ in range(num_layers)])
        self.norm = base.RMSNorm()

    def forward(self, tok_emb: nn.Embedding, pos_emb: Tensor, latent: Tensor) -> Tensor:
        bsz, num_chunks, _ = latent.shape
        hidden = self.latent_proj(latent.reshape(bsz * num_chunks, -1))
        x = self.expand(hidden).reshape(bsz * num_chunks, self.patch_size, -1)
        x = x + pos_emb.to(dtype=x.dtype)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits_proj = F.linear(x.reshape(-1, x.size(-1)), tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits.reshape(bsz, num_chunks, self.patch_size, -1)


class ContextChunkEmbedder(nn.Module):
    def __init__(self, model_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.in_proj = base.CastedLinear(patch_size * model_dim, 2 * model_dim, bias=False)
        self.out_proj = base.CastedLinear(2 * model_dim, model_dim, bias=False)

    def forward(self, tok_emb: nn.Embedding, pos_emb: Tensor, chunk_tokens: Tensor) -> Tensor:
        bsz, num_chunks, patch_size = chunk_tokens.shape
        if patch_size != self.patch_size:
            raise ValueError(f"Expected patch_size={self.patch_size}, got {patch_size}")
        x = tok_emb(chunk_tokens.reshape(bsz * num_chunks, patch_size))
        x = x + pos_emb.to(dtype=x.dtype)[None, :, :]
        flat = x.reshape(bsz * num_chunks, -1)
        hidden = self.out_proj(F.silu(self.in_proj(flat)))
        return hidden.reshape(bsz, num_chunks, -1)


class EnergyMLPBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.linears = nn.Sequential(
            base.CastedLinear(2 * channels, channels, bias=True),
            nn.SiLU(),
            base.CastedLinear(channels, channels, bias=True),
            nn.SiLU(),
            base.CastedLinear(channels, 2 * channels, bias=True),
        )
        self.down_proj = base.CastedLinear(channels, channels, bias=True)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h = self.linears(torch.cat((self.norm(x), y), dim=-1))
        gate_proj, up_proj = torch.chunk(h, 2, dim=-1)
        step = self.down_proj(F.silu(gate_proj) * up_proj)
        return x + step


class EnergyGenerator(nn.Module):
    def __init__(self, model_dim: int, latent_dim: int, noise_size: int, num_layers: int):
        super().__init__()
        self.noise_size = noise_size
        self.noise_embd = base.CastedLinear(noise_size, model_dim, bias=True)
        self.hidden_embd = base.CastedLinear(model_dim, model_dim, bias=True)
        self.norm_hidden = nn.LayerNorm(model_dim, eps=1e-6)
        self.norm_noise = nn.LayerNorm(model_dim, eps=1e-6)
        self.blocks = nn.ModuleList([EnergyMLPBlock(model_dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(model_dim, eps=1e-6)
        self.final = nn.Sequential(
            base.CastedLinear(model_dim, model_dim, bias=True),
            nn.SiLU(),
            base.CastedLinear(model_dim, latent_dim, bias=True),
        )
        # Match upstream CALM: generator starts near zero output.
        final_linear = self.final[-1]
        nn.init.zeros_(final_linear.weight)
        if final_linear.bias is not None:
            nn.init.zeros_(final_linear.bias)

    def sample(self, hidden_states: Tensor, num_samples: int, *, deterministic: bool = False) -> Tensor:
        # hidden_states: [B, N, D]
        bsz, num_chunks, model_dim = hidden_states.shape
        hidden = hidden_states.unsqueeze(0).expand(num_samples, -1, -1, -1)
        if deterministic:
            noise = torch.zeros(
                num_samples,
                bsz,
                num_chunks,
                self.noise_size,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        else:
            noise = torch.rand(
                num_samples,
                bsz,
                num_chunks,
                self.noise_size,
                device=hidden.device,
                dtype=hidden.dtype,
            ) - 0.5
        x = self.norm_noise(self.noise_embd(noise.reshape(-1, self.noise_size))).reshape(num_samples, bsz, num_chunks, model_dim)
        y = self.norm_hidden(self.hidden_embd(hidden.reshape(-1, model_dim))).reshape(num_samples, bsz, num_chunks, model_dim)
        for block in self.blocks:
            x = block(x, y)
        latent = self.final(self.final_norm(x).reshape(-1, model_dim))
        return latent.reshape(num_samples, bsz, num_chunks, -1)


class CALMEnergyGPT(nn.Module):
    training_only_prefixes = ("ae_encoder.",)

    def __init__(self, args: Hyperparameters):
        super().__init__()
        if args.train_seq_len % args.calm_patch_size != 0:
            raise ValueError(
                f"TRAIN_SEQ_LEN={args.train_seq_len} must be divisible by CALM_PATCH_SIZE={args.calm_patch_size}"
            )
        if args.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {args.logit_softcap}")
        self.patch_size = args.calm_patch_size
        self.latent_dim = args.calm_latent_dim
        self.beta = args.calm_beta
        self.num_samples = args.calm_num_samples
        self.eval_samples = args.calm_eval_samples
        self.ae_steps = args.calm_ae_steps
        self.latent_l2 = args.calm_latent_l2
        self.model_dim = args.model_dim
        self.train_stage = "ae"

        self.ae_tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.ae_pos_emb = nn.Parameter(torch.zeros(self.patch_size, args.model_dim, dtype=torch.float32))
        self.ae_encoder = ChunkAutoencoderEncoder(
            model_dim=args.model_dim,
            patch_size=self.patch_size,
            latent_dim=self.latent_dim,
            num_layers=args.calm_ae_layers,
            expansion=args.calm_ae_expansion,
            dropout=args.calm_ae_dropout,
        )
        self.ae_decoder = ChunkAutoencoderDecoder(
            model_dim=args.model_dim,
            patch_size=self.patch_size,
            latent_dim=self.latent_dim,
            num_layers=args.calm_ae_layers,
            expansion=args.calm_ae_expansion,
            logit_softcap=args.logit_softcap,
        )

        self.ctx_tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.ctx_pos_emb = nn.Parameter(torch.zeros(self.patch_size, args.model_dim, dtype=torch.float32))
        self.chunk_init = nn.Parameter(torch.zeros(args.model_dim, dtype=torch.float32))
        self.chunk_embedder = ContextChunkEmbedder(args.model_dim, self.patch_size)
        self.blocks = nn.ModuleList(
            [
                base.Block(
                    args.model_dim,
                    args.calm_num_heads,
                    args.calm_num_kv_heads,
                    args.mlp_mult,
                    args.rope_base,
                    args.qk_gain_init,
                )
                for _ in range(args.calm_lm_layers)
            ]
        )
        self.final_norm = base.RMSNorm()
        self.generator = EnergyGenerator(
            model_dim=args.model_dim,
            latent_dim=self.latent_dim,
            noise_size=args.calm_noise_size,
            num_layers=max(1, args.calm_ae_layers),
        )

        nn.init.normal_(self.ae_tok_emb.weight, mean=0.0, std=args.tied_embed_init_std)
        nn.init.normal_(self.ctx_tok_emb.weight, mean=0.0, std=args.tied_embed_init_std)
        self._set_stage("ae")

    def _set_trainable(self, module: nn.Module | Tensor, requires_grad: bool) -> None:
        if isinstance(module, nn.Parameter):
            module.requires_grad_(requires_grad)
            return
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def _set_stage(self, stage: str) -> None:
        self.train_stage = stage
        ae_trainable = stage == "ae"
        self._set_trainable(self.ae_tok_emb, ae_trainable)
        self._set_trainable(self.ae_encoder, ae_trainable)
        self._set_trainable(self.ae_decoder, ae_trainable)
        self.ae_pos_emb.requires_grad_(ae_trainable)

        lm_trainable = stage == "lm"
        self._set_trainable(self.ctx_tok_emb, lm_trainable)
        self._set_trainable(self.chunk_embedder, lm_trainable)
        self._set_trainable(self.blocks, lm_trainable)
        self._set_trainable(self.final_norm, lm_trainable)
        self._set_trainable(self.generator, lm_trainable)
        self.ctx_pos_emb.requires_grad_(lm_trainable)
        self.chunk_init.requires_grad_(lm_trainable)

    def update_stage(self, step: int) -> None:
        target_stage = "ae" if step < self.ae_steps else "lm"
        if target_stage != self.train_stage:
            self._set_stage(target_stage)
        if self.train_stage == "ae":
            self.ae_encoder.train()
            self.ae_decoder.train()
        else:
            self.ae_encoder.eval()
            self.ae_decoder.eval()

    def chunk_tokens(self, token_ids: Tensor) -> Tensor:
        bsz, seqlen = token_ids.shape
        if seqlen % self.patch_size != 0:
            raise ValueError(f"Sequence length {seqlen} must be divisible by patch_size={self.patch_size}")
        return token_ids.reshape(bsz, seqlen // self.patch_size, self.patch_size)

    def build_chunk_prediction_pairs(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        full_chunks = self.chunk_tokens(input_ids)
        if full_chunks.size(1) < 2:
            raise ValueError("Need at least two chunks for next-chunk prediction")
        context_chunks = full_chunks[:, :-1, :]
        target_chunks = full_chunks[:, 1:, :]
        return context_chunks, target_chunks

    def encode_target_latent(self, target_chunks: Tensor) -> Tensor:
        if self.train_stage == "lm":
            self.ae_encoder.eval()
        return self.ae_encoder(self.ae_tok_emb, self.ae_pos_emb, target_chunks)

    def decode_latent(self, latent: Tensor) -> Tensor:
        return self.ae_decoder(self.ae_tok_emb, self.ae_pos_emb, latent)

    def autoencoder_loss(self, target_chunks: Tensor) -> Tensor:
        latent = self.encode_target_latent(target_chunks)
        logits = self.decode_latent(latent)
        recon = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_chunks.reshape(-1), reduction="mean")
        return recon + self.latent_l2 * latent.float().pow(2).mean()

    def encode_context(self, input_chunks: Tensor) -> Tensor:
        x = self.chunk_embedder(self.ctx_tok_emb, self.ctx_pos_emb, input_chunks)
        x = x + self.chunk_init.to(dtype=x.dtype)[None, None, :]
        x0 = x
        for block in self.blocks:
            x = block(x, x0)
        return self.final_norm(x)

    def latent_distance(self, x1: Tensor, x2: Tensor) -> Tensor:
        return torch.linalg.vector_norm(x1 - x2, ord=2, dim=-1).pow(self.beta)

    def energy_loss(self, input_chunks: Tensor, target_chunks: Tensor) -> Tensor:
        with torch.no_grad():
            target_latent = self.encode_target_latent(target_chunks)
        hidden = self.encode_context(input_chunks)
        pred_latents = self.generator.sample(hidden, self.num_samples, deterministic=False)
        if self.num_samples > 1:
            x_i = pred_latents.unsqueeze(1)
            x_j = pred_latents.unsqueeze(0)
            distance_x = self.latent_distance(x_i, x_j).sum(dim=(0, 1)) / (self.num_samples * (self.num_samples - 1))
        else:
            distance_x = torch.zeros_like(target_latent[..., 0])
        distance_y = self.latent_distance(pred_latents, target_latent.unsqueeze(0)).mean(dim=0)
        score = distance_x - 2.0 * distance_y
        return (-score).mean()

    def eval_ce(self, input_chunks: Tensor, target_chunks: Tensor) -> Tensor:
        hidden = self.encode_context(input_chunks)
        pred_latents = self.generator.sample(hidden, self.eval_samples, deterministic=True)
        sample_logits = []
        for sample_latent in pred_latents:
            sample_logits.append(self.decode_latent(sample_latent))
        logits = torch.stack(sample_logits, dim=0)  # [S, B, N, P, V]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        log_mean_probs = torch.logsumexp(log_probs, dim=0) - math.log(logits.size(0))
        return F.nll_loss(log_mean_probs.reshape(-1, log_mean_probs.size(-1)), target_chunks.reshape(-1), reduction="mean")

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        del target_ids
        input_chunks, target_chunks = self.build_chunk_prediction_pairs(input_ids)
        if self.training:
            if self.train_stage == "ae":
                return self.autoencoder_loss(self.chunk_tokens(input_ids))
            return self.energy_loss(input_chunks, target_chunks)
        return self.eval_ce(input_chunks, target_chunks)

    def artifact_state_dict(self) -> dict[str, Tensor]:
        return {
            name: tensor
            for name, tensor in self.state_dict().items()
            if not any(name.startswith(prefix) for prefix in self.training_only_prefixes)
        }


def unwrap_model(model: nn.Module) -> CALMEnergyGPT:
    return model.module if isinstance(model, DDP) else model


def eval_val_calm(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    core = unwrap_model(model)
    with torch.no_grad():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            _, target_chunks = core.build_chunk_prediction_pairs(x)
            batch_token_count = float(target_chunks.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x[:, core.patch_size - 1 : -1].reshape(-1)
            tgt_ids = x[:, core.patch_size :].reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = base.load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = CALMEnergyGPT(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, base.CastedLinear):
            module.float()
    base.restore_low_dim_params_to_fp32(base_model)
    model: nn.Module = (
        DDP(base_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
        if distributed
        else base_model
    )

    optimizer = torch.optim.Adam(
        [{"params": list(base_model.parameters()), "lr": args.calm_lr, "base_lr": args.calm_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer]

    train_params = sum(p.numel() for p in base_model.parameters())
    artifact_params = sum(t.numel() for t in base_model.artifact_state_dict().values())
    log0(f"model_params_train:{train_params}")
    log0(f"model_params_artifact:{artifact_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.calm_num_heads} num_kv_heads:{args.calm_num_kv_heads}")
    log0(
        f"calm_two_stage:patch_size={args.calm_patch_size} latent_dim={args.calm_latent_dim} "
        f"ae_steps={args.calm_ae_steps} ae_layers={args.calm_ae_layers} lm_layers={args.calm_lm_layers}"
    )
    log0(
        f"calm_energy:num_samples={args.calm_num_samples} eval_samples={args.calm_eval_samples} "
        f"beta={args.calm_beta} lr={args.calm_lr} grad_clip_norm={args.grad_clip_norm}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    train_loader = base.DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            base_model.update_stage(0)
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = base.DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        base_model.update_stage(step)
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val_calm(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            lm_stage = 0.0 if base_model.train_stage == "ae" else 1.0
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} lm_stage:{lm_stage:.0f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    artifact_state = base_model.artifact_state_dict()
    if master_process:
        torch.save(artifact_state, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized artifact model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = base.quantize_state_dict_int8(artifact_state)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized artifact int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    roundtrip_state = base.dequantize_state_dict_int8(quant_state)
    missing, unexpected = base_model.load_state_dict(roundtrip_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading roundtrip artifact: {unexpected}")
    allowed_missing = {name for name, _ in base_model.named_parameters() if name.startswith("ae_encoder.")}
    if set(missing) - allowed_missing:
        raise RuntimeError(f"Unexpected missing keys when loading roundtrip artifact: {missing}")
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val_calm(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
