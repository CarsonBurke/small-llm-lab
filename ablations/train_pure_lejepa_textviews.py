"""
Interleaved pure LeJEPA text-view ablation forked from train_normuon.py.

LeJEPA updates:
- no CE loss, no stop-grad target, no EMA, no negatives
- form two stochastic global token-mask views and two local span views before
  the transformer, encode each view independently, then pool visible tokens
- align all projected views with the per-sample global-view center
- add sample-count-scaled per-view SIGReg

Detached CE probe updates:
- freeze the learned representation by detaching hidden states
- train only a small linear next-token CE probe/head
- run one detached probe update after every LEJEPA_STEPS_PER_PROBE LeJEPA updates
- validation always reports the probe CE/BPB so TensorBoard can show whether
  the representation is becoming linearly useful
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
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_normuon as base  # noqa: E402


class Hyperparameters(base.Hyperparameters):
    lejepa_steps_per_probe = int(os.environ.get("LEJEPA_STEPS_PER_PROBE", "10"))
    lejepa_lr = float(os.environ.get("LEJEPA_LR", "0.0001"))
    lejepa_weight_decay = float(os.environ.get("LEJEPA_WEIGHT_DECAY", "0.05"))
    lejepa_probe_lr = float(os.environ.get("LEJEPA_PROBE_LR", os.environ.get("HEAD_LR", "0.02")))
    pure_lejepa_lambda = float(os.environ.get("PURE_LEJEPA_LAMBDA", "0.05"))
    pure_lejepa_proj_dim = int(os.environ.get("PURE_LEJEPA_PROJ_DIM", os.environ.get("MODEL_DIM", "512")))
    pure_lejepa_proj_hidden = int(
        os.environ.get(
            "PURE_LEJEPA_PROJ_HIDDEN",
            str(2 * int(os.environ.get("PURE_LEJEPA_PROJ_DIM", os.environ.get("MODEL_DIM", "512")))),
        )
    )
    pure_lejepa_proj_dropout = float(os.environ.get("PURE_LEJEPA_PROJ_DROPOUT", "0.0"))
    pure_lejepa_global_keep = float(os.environ.get("PURE_LEJEPA_GLOBAL_KEEP", "0.75"))
    pure_lejepa_local_span_frac = float(os.environ.get("PURE_LEJEPA_LOCAL_SPAN_FRAC", "0.25"))
    pure_lejepa_sigreg_slices = int(os.environ.get("PURE_LEJEPA_SIGREG_SLICES", "1024"))
    pure_lejepa_sigreg_points = int(os.environ.get("PURE_LEJEPA_SIGREG_POINTS", "17"))
    pure_lejepa_sigreg_t_max = float(os.environ.get("PURE_LEJEPA_SIGREG_T_MAX", "5.0"))
    pure_lejepa_sigreg_tokens = int(os.environ.get("PURE_LEJEPA_SIGREG_TOKENS", "4096"))
    lejepa_vectorize_views = bool(int(os.environ.get("LEJEPA_VECTORIZE_VIEWS", "1")))

    if not base.Hyperparameters.tie_embeddings:
        raise ValueError("Pure LeJEPA text-view ablation keeps the baseline tied embedding setup; set TIE_EMBEDDINGS=1")
    if lejepa_steps_per_probe <= 0:
        raise ValueError(f"LEJEPA_STEPS_PER_PROBE must be positive, got {lejepa_steps_per_probe}")
    if lejepa_lr <= 0.0:
        raise ValueError(f"LEJEPA_LR must be positive, got {lejepa_lr}")
    if lejepa_weight_decay < 0.0:
        raise ValueError(f"LEJEPA_WEIGHT_DECAY must be non-negative, got {lejepa_weight_decay}")
    if lejepa_probe_lr <= 0.0:
        raise ValueError(f"LEJEPA_PROBE_LR must be positive, got {lejepa_probe_lr}")
    if pure_lejepa_lambda < 0.0 or pure_lejepa_lambda > 1.0:
        raise ValueError(f"PURE_LEJEPA_LAMBDA must be in [0, 1], got {pure_lejepa_lambda}")
    if pure_lejepa_proj_dim <= 0:
        raise ValueError(f"PURE_LEJEPA_PROJ_DIM must be positive, got {pure_lejepa_proj_dim}")
    if pure_lejepa_proj_hidden <= 0:
        raise ValueError(f"PURE_LEJEPA_PROJ_HIDDEN must be positive, got {pure_lejepa_proj_hidden}")
    if pure_lejepa_proj_dropout < 0.0 or pure_lejepa_proj_dropout >= 1.0:
        raise ValueError(f"PURE_LEJEPA_PROJ_DROPOUT must be in [0, 1), got {pure_lejepa_proj_dropout}")
    if pure_lejepa_global_keep <= 0.0 or pure_lejepa_global_keep > 1.0:
        raise ValueError(f"PURE_LEJEPA_GLOBAL_KEEP must be in (0, 1], got {pure_lejepa_global_keep}")
    if pure_lejepa_local_span_frac <= 0.0 or pure_lejepa_local_span_frac > 1.0:
        raise ValueError(f"PURE_LEJEPA_LOCAL_SPAN_FRAC must be in (0, 1], got {pure_lejepa_local_span_frac}")
    if pure_lejepa_sigreg_slices <= 0:
        raise ValueError(f"PURE_LEJEPA_SIGREG_SLICES must be positive, got {pure_lejepa_sigreg_slices}")
    if pure_lejepa_sigreg_points < 2:
        raise ValueError(f"PURE_LEJEPA_SIGREG_POINTS must be at least 2, got {pure_lejepa_sigreg_points}")
    if pure_lejepa_sigreg_t_max <= 0.0:
        raise ValueError(f"PURE_LEJEPA_SIGREG_T_MAX must be positive, got {pure_lejepa_sigreg_t_max}")
    if pure_lejepa_sigreg_tokens <= 0:
        raise ValueError(f"PURE_LEJEPA_SIGREG_TOKENS must be positive, got {pure_lejepa_sigreg_tokens}")


class LeJEPATextProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = base.CastedLinear(input_dim, hidden_dim, bias=False)
        self.fc2 = base.CastedLinear(hidden_dim, output_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        x = F.rms_norm(x, (x.size(-1),))
        x = F.gelu(self.fc1(x))
        x = F.rms_norm(x, (x.size(-1),))
        return self.fc2(x)


class PureLeJEPATextViewsGPT(base.GPT):
    def __init__(self, args: Hyperparameters):
        super().__init__(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=True,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
        )
        self.lm_head = None
        self.stage = "lejepa"
        self.lejepa_lambda = float(args.pure_lejepa_lambda)
        self.global_keep = float(args.pure_lejepa_global_keep)
        self.local_span_frac = float(args.pure_lejepa_local_span_frac)
        self.sigreg_tokens = int(args.pure_lejepa_sigreg_tokens)
        self.vectorize_views = bool(args.lejepa_vectorize_views)
        self.collect_lejepa_diagnostics = False
        self.projector = LeJEPATextProjector(
            input_dim=args.model_dim,
            hidden_dim=args.pure_lejepa_proj_hidden,
            output_dim=args.pure_lejepa_proj_dim,
            dropout=args.pure_lejepa_proj_dropout,
        )
        self.probe_head = base.CastedLinear(args.model_dim, args.vocab_size, bias=False)
        nn.init.zeros_(self.probe_head.weight)

        dirs = torch.randn(args.pure_lejepa_sigreg_slices, args.pure_lejepa_proj_dim, dtype=torch.float32)
        dirs = F.normalize(dirs, dim=-1)
        if args.pure_lejepa_sigreg_points % 2 == 1:
            t = torch.linspace(
                0.0,
                args.pure_lejepa_sigreg_t_max,
                args.pure_lejepa_sigreg_points // 2 + 1,
                dtype=torch.float32,
            )
            sigreg_integral_factor = 2.0
        else:
            t = torch.linspace(
                -args.pure_lejepa_sigreg_t_max,
                args.pure_lejepa_sigreg_t_max,
                args.pure_lejepa_sigreg_points,
                dtype=torch.float32,
            )
            sigreg_integral_factor = 1.0
        self.register_buffer("sigreg_dirs", dirs, persistent=False)
        self.register_buffer("sigreg_t", t, persistent=False)
        self.register_buffer(
            "sigreg_integral_factor",
            torch.tensor(sigreg_integral_factor, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("last_loss", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_inv_loss", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_sigreg_loss", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_view_norm", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_global_cos", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_local_cos", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_view_spread", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_view_eff_rank", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_global_mask_jaccard", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_local_mask_jaccard", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_global_pooled_cos", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_local_pooled_cos", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_probe_loss", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_probe_acc", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_probe_entropy", torch.zeros((), dtype=torch.float32), persistent=False)

    def set_stage(self, stage: str) -> None:
        if stage not in {"lejepa", "probe"}:
            raise ValueError(f"unknown stage {stage!r}")
        self.stage = stage

    def clear_rotary_caches(self) -> None:
        for module in self.modules():
            if hasattr(module, "_cos_cached"):
                module._cos_cached = None
            if hasattr(module, "_sin_cached"):
                module._sin_cached = None
            if hasattr(module, "_seq_len_cached"):
                module._seq_len_cached = 0

    def encode_hidden(self, input_ids: Tensor, input_mask: Tensor | None = None) -> Tensor:
        x = self.tok_emb(input_ids)
        if input_mask is not None:
            x = x * input_mask.to(dtype=x.dtype).unsqueeze(-1)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        checkpoint_blocks = self.training and self.stage == "lejepa"
        for i in range(self.num_encoder_layers):
            block = self.blocks[i]
            x = checkpoint(block, x, x0, use_reentrant=False) if checkpoint_blocks else block(x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            block = self.blocks[self.num_encoder_layers + i]
            x = checkpoint(block, x, x0, use_reentrant=False) if checkpoint_blocks else block(x, x0)
        return self.final_norm(x)

    def masked_pool(self, hidden: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(dtype=torch.float32)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (hidden.float() * weights.unsqueeze(-1)).sum(dim=1) / denom
        return pooled.to(dtype=hidden.dtype)

    def global_mask(self, input_ids: Tensor) -> Tensor:
        bsz, seqlen = input_ids.shape
        keep = torch.rand(bsz, seqlen, device=input_ids.device) < self.global_keep
        empty = keep.sum(dim=1) == 0
        if empty.any():
            fill_idx = torch.randint(0, seqlen, (int(empty.sum().item()),), device=input_ids.device)
            keep[empty] = False
            keep[empty, fill_idx] = True
        return keep

    def local_mask(self, input_ids: Tensor) -> Tensor:
        bsz, seqlen = input_ids.shape
        span = max(1, min(seqlen, int(math.ceil(self.local_span_frac * seqlen))))
        start_max = seqlen - span
        starts = torch.randint(0, start_max + 1, (bsz,), device=input_ids.device)
        offsets = torch.arange(span, device=input_ids.device)
        idx = starts[:, None] + offsets[None, :]
        mask = torch.zeros(bsz, seqlen, device=input_ids.device, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask

    def view_project(self, input_ids: Tensor, mask: Tensor) -> Tensor:
        hidden = self.encode_hidden(input_ids, mask)
        return self.projector(self.masked_pool(hidden, mask))

    def stochastic_projected_input_views(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        masks = torch.stack(
            [
                self.global_mask(input_ids),
                self.global_mask(input_ids),
                self.local_mask(input_ids),
                self.local_mask(input_ids),
            ],
            dim=0,
        )
        if not self.vectorize_views:
            pooled = []
            projected = []
            for mask in masks:
                hidden = self.encode_hidden(input_ids, mask)
                pooled_view = self.masked_pool(hidden, mask)
                pooled.append(pooled_view)
                projected.append(self.projector(pooled_view))
            return torch.stack(projected, dim=0), masks, torch.stack(pooled, dim=0)

        view_count, bsz, seqlen = masks.shape
        flat_input_ids = (
            input_ids.unsqueeze(0)
            .expand(view_count, -1, -1)
            .reshape(view_count * bsz, seqlen)
        )
        flat_masks = masks.reshape(view_count * bsz, seqlen)
        hidden = self.encode_hidden(flat_input_ids, flat_masks)
        pooled = self.masked_pool(hidden, flat_masks).reshape(view_count, bsz, -1)
        projected = self.projector(pooled.reshape(view_count * bsz, -1)).reshape(view_count, bsz, -1)
        return projected, masks, pooled

    def _sample_views_for_sigreg(self, views: Tensor) -> Tensor:
        view_count, samples_per_view, dim = views.shape
        if samples_per_view <= self.sigreg_tokens:
            return views
        stride = max(samples_per_view // self.sigreg_tokens, 1)
        return views[:, ::stride, :][:, : self.sigreg_tokens, :].reshape(view_count, -1, dim)

    def sigreg_loss(self, views: Tensor) -> Tensor:
        samples = self._sample_views_for_sigreg(views).float()
        dirs = self.sigreg_dirs.to(device=samples.device, dtype=torch.float32)
        t = self.sigreg_t.to(device=samples.device, dtype=torch.float32)
        gaussian_cf = torch.exp(-0.5 * t.square()).view(1, 1, -1)

        proj = torch.einsum("vnd,md->vnm", samples, dirs)
        angles = proj.unsqueeze(-1) * t.view(1, 1, 1, -1)
        real = torch.cos(angles).mean(dim=1)
        imag = torch.sin(angles).mean(dim=1)
        err = (real - gaussian_cf).square() + imag.square()
        integral = self.sigreg_integral_factor.to(device=samples.device) * torch.trapz(
            err * gaussian_cf,
            t,
            dim=-1,
        )
        per_view = samples.size(1) * integral.mean(dim=-1)
        return per_view.mean()

    def effective_rank(self, samples: Tensor) -> Tensor:
        with torch.autocast(device_type=samples.device.type, enabled=False):
            samples = samples.to(dtype=torch.float32)
            centered = samples - samples.mean(dim=0, keepdim=True)
            cov = (centered.T @ centered) / max(samples.size(0) - 1, 1)
            eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
            eig_norm = eig / eig.sum().clamp_min(1e-12)
            return torch.exp(-(eig_norm * eig_norm.clamp_min(1e-12).log()).sum())

    @torch.no_grad()
    def resample_sigreg_dirs(self, step: int) -> None:
        device = self.sigreg_dirs.device
        dtype = self.sigreg_dirs.dtype
        generator = torch.Generator(device=device)
        generator.manual_seed(int(step))
        dirs = torch.randn(self.sigreg_dirs.shape, generator=generator, device=device, dtype=torch.float32)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.sigreg_dirs.copy_(dirs.to(dtype=dtype))

    def training_loss(self, input_ids: Tensor) -> Tensor:
        views, masks, pooled = self.stochastic_projected_input_views(input_ids)
        center = views[:2].mean(dim=0, keepdim=True)
        inv_loss = (views.float() - center.float()).square().mean()
        sigreg_loss = self.sigreg_loss(views)
        loss = (1.0 - self.lejepa_lambda) * inv_loss + self.lejepa_lambda * sigreg_loss

        with torch.no_grad():
            view_norm = views.float().norm(dim=-1).mean()
            global_cos = (
                F.normalize(views[0].float(), dim=-1) * F.normalize(views[1].float(), dim=-1)
            ).sum(dim=-1).mean()
            local_cos = (
                F.normalize(views[2].float(), dim=-1) * F.normalize(views[3].float(), dim=-1)
            ).sum(dim=-1).mean()
            view_spread = (views.float() - center.float()).square().sum(dim=-1).sqrt().mean()

        self.last_loss.copy_(loss.detach())
        self.last_inv_loss.copy_(inv_loss.detach())
        self.last_sigreg_loss.copy_(sigreg_loss.detach())
        self.last_view_norm.copy_(view_norm)
        self.last_global_cos.copy_(global_cos)
        self.last_local_cos.copy_(local_cos)
        self.last_view_spread.copy_(view_spread)
        if self.collect_lejepa_diagnostics:
            with torch.no_grad():
                view_eff_rank = torch.stack([self.effective_rank(view) for view in views]).mean()
                global_mask_jaccard = (masks[0] & masks[1]).sum(dim=1).float() / (
                    masks[0] | masks[1]
                ).sum(dim=1).clamp_min(1).float()
                local_mask_jaccard = (masks[2] & masks[3]).sum(dim=1).float() / (
                    masks[2] | masks[3]
                ).sum(dim=1).clamp_min(1).float()
                global_pooled_cos = (
                    F.normalize(pooled[0].float(), dim=-1) * F.normalize(pooled[1].float(), dim=-1)
                ).sum(dim=-1).mean()
                local_pooled_cos = (
                    F.normalize(pooled[2].float(), dim=-1) * F.normalize(pooled[3].float(), dim=-1)
                ).sum(dim=-1).mean()
            self.last_view_eff_rank.copy_(view_eff_rank)
            self.last_global_mask_jaccard.copy_(global_mask_jaccard.mean())
            self.last_local_mask_jaccard.copy_(local_mask_jaccard.mean())
            self.last_global_pooled_cos.copy_(global_pooled_cos)
            self.last_local_pooled_cos.copy_(local_pooled_cos)
        return loss

    def probe_ce_loss(self, input_ids: Tensor, target_ids: Tensor, detach_hidden: bool) -> Tensor:
        if detach_hidden:
            with torch.no_grad():
                hidden = self.encode_hidden(input_ids)
        else:
            hidden = self.encode_hidden(input_ids)
        logits = self.probe_head(hidden.reshape(-1, hidden.size(-1)))
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        targets = target_ids.reshape(-1)
        loss = F.cross_entropy(logits.float(), targets, reduction="mean")

        with torch.no_grad():
            probs = logits.float().softmax(dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
            acc = (logits.argmax(dim=-1) == targets).float().mean()
        self.last_probe_loss.copy_(loss.detach())
        self.last_probe_acc.copy_(acc)
        self.last_probe_entropy.copy_(entropy)
        return loss

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        if self.training and self.stage == "lejepa":
            return self.training_loss(input_ids)
        return self.probe_ce_loss(input_ids, target_ids, detach_hidden=self.training)

    @torch.no_grad()
    def token_view_diagnostics(self) -> dict[str, float]:
        was_training = self.projector.training
        self.projector.eval()
        tok = self.projector(self.tok_emb.weight).float()
        if was_training:
            self.projector.train()
        tok_centered = tok - tok.mean(dim=0, keepdim=True)
        cov = (tok_centered.T @ tok_centered) / max(tok.size(0) - 1, 1)
        eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
        eig_norm = eig / eig.sum().clamp_min(1e-12)
        eff_rank = torch.exp(-(eig_norm * eig_norm.clamp_min(1e-12).log()).sum())
        tok_unit = F.normalize(tok, dim=-1)
        cos_mat = tok_unit @ tok_unit.T
        vocab = cos_mat.size(0)
        cos_off = (cos_mat.sum() - cos_mat.diag().sum()) / max(cos_mat.numel() - vocab, 1)
        return {
            "token_view_norm_mean": float(tok.norm(dim=-1).mean().item()),
            "token_view_norm_std": float(tok.norm(dim=-1).std(correction=0).item()),
            "token_view_dim_std_mean": float(tok.std(dim=0, correction=0).mean().item()),
            "token_view_eff_rank": float(eff_rank.item()),
            "token_view_cos_off_mean": float(cos_off.item()),
            "dirs_sum": float(self.sigreg_dirs.to(torch.float32).sum().item()),
        }


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    base.zeropower_via_newtonschulz5 = torch.compile(base.zeropower_via_newtonschulz5)

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

    base_model = PureLeJEPATextViewsGPT(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, base.CastedLinear):
            module.float()
    base.restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
        if distributed
        else compiled_model
    )
    eval_model: nn.Module = base_model

    named_params = list(base_model.named_parameters())
    probe_param_names = {"probe_head.weight"}
    lejepa_params = [
        p
        for name, p in named_params
        if name not in probe_param_names
    ]
    optimizer_lejepa = torch.optim.AdamW(
        [
            {
                "params": lejepa_params,
                "lr": args.lejepa_lr,
                "base_lr": args.lejepa_lr,
                "weight_decay": args.lejepa_weight_decay,
            }
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_probe = torch.optim.Adam(
        [{"params": [base_model.probe_head.weight], "lr": args.lejepa_probe_lr, "base_lr": args.lejepa_probe_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    pretrain_optimizers: list[torch.optim.Optimizer] = [optimizer_lejepa]
    probe_optimizers: list[torch.optim.Optimizer] = [optimizer_probe]
    optimizers: list[torch.optim.Optimizer] = pretrain_optimizers + probe_optimizers

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"pure_lejepa_textviews:lambda={args.pure_lejepa_lambda} proj_dim={args.pure_lejepa_proj_dim} "
        f"proj_hidden={args.pure_lejepa_proj_hidden} proj_dropout={args.pure_lejepa_proj_dropout} "
        f"schedule=interleaved lejepa_steps_per_probe={args.lejepa_steps_per_probe} "
        f"lejepa_lr={args.lejepa_lr} "
        f"lejepa_weight_decay={args.lejepa_weight_decay} probe_lr={args.lejepa_probe_lr} "
        f"global_keep={args.pure_lejepa_global_keep} local_span_frac={args.pure_lejepa_local_span_frac} "
        f"sigreg_slices={args.pure_lejepa_sigreg_slices} sigreg_points={args.pure_lejepa_sigreg_points} "
        f"sigreg_t_max={args.pure_lejepa_sigreg_t_max} sigreg_tokens={args.pure_lejepa_sigreg_tokens} "
        f"sigreg_effective_t_points={base_model.sigreg_t.numel()} "
        f"vectorize_views={int(args.lejepa_vectorize_views)} head=detached_linear_probe"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} lejepa_optimizer:adamw lejepa_lr:{args.lejepa_lr} "
        f"probe_lr:{args.lejepa_probe_lr}"
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
        base_model.set_stage("lejepa")
        for warmup_step in range(args.warmup_steps):
            base_model.resample_sigreg_dirs(warmup_step)
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in pretrain_optimizers:
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
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = base.eval_val(
                args,
                eval_model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            base_model.clear_rotary_caches()
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            if args.val_loss_every > 0:
                diag = base_model.token_view_diagnostics()
                log0(
                    "token_view_diag:"
                    f"step:{step}/{args.iterations} "
                    f"token_view_norm_mean:{diag['token_view_norm_mean']:.4f} "
                    f"token_view_norm_std:{diag['token_view_norm_std']:.4f} "
                    f"token_view_dim_std_mean:{diag['token_view_dim_std_mean']:.4f} "
                    f"token_view_eff_rank:{diag['token_view_eff_rank']:.4f} "
                    f"token_view_cos_off_mean:{diag['token_view_cos_off_mean']:.4f} "
                    f"dirs_sum:{diag['dirs_sum']:.4f}"
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
        schedule_pos = step % (args.lejepa_steps_per_probe + 1)
        stage = "lejepa" if schedule_pos < args.lejepa_steps_per_probe else "probe"
        base_model.set_stage(stage)
        next_step = step + 1
        will_log_train = (
            args.train_log_every > 0
            and (next_step <= 10 or next_step % args.train_log_every == 0 or stop_after_step is not None)
        )
        base_model.collect_lejepa_diagnostics = stage == "lejepa" and will_log_train
        active_optimizers = pretrain_optimizers if stage == "lejepa" else probe_optimizers
        if stage == "lejepa":
            base_model.resample_sigreg_dirs(step)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        train_inv_loss = torch.zeros((), device=device)
        train_sigreg_loss = torch.zeros((), device=device)
        train_view_norm = torch.zeros((), device=device)
        train_global_cos = torch.zeros((), device=device)
        train_local_cos = torch.zeros((), device=device)
        train_view_spread = torch.zeros((), device=device)
        train_view_eff_rank = torch.zeros((), device=device)
        train_global_mask_jaccard = torch.zeros((), device=device)
        train_local_mask_jaccard = torch.zeros((), device=device)
        train_global_pooled_cos = torch.zeros((), device=device)
        train_local_pooled_cos = torch.zeros((), device=device)
        train_probe_loss = torch.zeros((), device=device)
        train_probe_acc = torch.zeros((), device=device)
        train_probe_entropy = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            if stage == "lejepa":
                train_inv_loss += base_model.last_inv_loss.detach()
                train_sigreg_loss += base_model.last_sigreg_loss.detach()
                train_view_norm += base_model.last_view_norm.detach()
                train_global_cos += base_model.last_global_cos.detach()
                train_local_cos += base_model.last_local_cos.detach()
                train_view_spread += base_model.last_view_spread.detach()
                if base_model.collect_lejepa_diagnostics:
                    train_view_eff_rank += base_model.last_view_eff_rank.detach()
                    train_global_mask_jaccard += base_model.last_global_mask_jaccard.detach()
                    train_local_mask_jaccard += base_model.last_local_mask_jaccard.detach()
                    train_global_pooled_cos += base_model.last_global_pooled_cos.detach()
                    train_local_pooled_cos += base_model.last_local_pooled_cos.detach()
            else:
                train_probe_loss += base_model.last_probe_loss.detach()
                train_probe_acc += base_model.last_probe_acc.detach()
                train_probe_entropy += base_model.last_probe_entropy.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        if stage == "lejepa":
            train_inv_loss /= grad_accum_steps
            train_sigreg_loss /= grad_accum_steps
            train_view_norm /= grad_accum_steps
            train_global_cos /= grad_accum_steps
            train_local_cos /= grad_accum_steps
            train_view_spread /= grad_accum_steps
            if base_model.collect_lejepa_diagnostics:
                train_view_eff_rank /= grad_accum_steps
                train_global_mask_jaccard /= grad_accum_steps
                train_local_mask_jaccard /= grad_accum_steps
                train_global_pooled_cos /= grad_accum_steps
                train_local_pooled_cos /= grad_accum_steps
        else:
            train_probe_loss /= grad_accum_steps
            train_probe_acc /= grad_accum_steps
            train_probe_entropy /= grad_accum_steps

        for opt in active_optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            active_params = [p for opt in active_optimizers for group in opt.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(active_params, args.grad_clip_norm)
        for opt in active_optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = will_log_train
        if should_log_train:
            if stage == "lejepa":
                diag = base_model.token_view_diagnostics()
                log0(
                    f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                    f"stage_id:0.0 lejepa_loss:{train_loss.item():.4f} "
                    f"inv_loss:{train_inv_loss.item():.4f} sigreg_loss:{train_sigreg_loss.item():.4f} "
                    f"view_norm:{train_view_norm.item():.4f} global_cos:{train_global_cos.item():.4f} "
                    f"local_cos:{train_local_cos.item():.4f} view_spread:{train_view_spread.item():.4f} "
                    f"view_eff_rank:{train_view_eff_rank.item():.4f} "
                    f"global_mask_jaccard:{train_global_mask_jaccard.item():.4f} "
                    f"local_mask_jaccard:{train_local_mask_jaccard.item():.4f} "
                    f"global_pooled_cos:{train_global_pooled_cos.item():.4f} "
                    f"local_pooled_cos:{train_local_pooled_cos.item():.4f} "
                    f"token_view_eff_rank:{diag['token_view_eff_rank']:.4f} "
                    f"token_view_cos_off_mean:{diag['token_view_cos_off_mean']:.4f} "
                    f"token_view_dim_std_mean:{diag['token_view_dim_std_mean']:.4f} "
                    f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
                )
            else:
                probe_norm = base_model.probe_head.weight.detach().float().norm(dim=-1).mean()
                log0(
                    f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                    f"stage_id:1.0 probe_loss:{train_probe_loss.item():.4f} "
                    f"probe_acc:{train_probe_acc.item():.4f} probe_entropy:{train_probe_entropy.item():.4f} "
                    f"probe_head_norm:{probe_norm.item():.4f} "
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

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = base.quantize_state_dict_int8(base_model.state_dict())
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
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(base.dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = base.eval_val(
        args,
        eval_model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    base_model.clear_rotary_caches()
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
