"""
LeJEPA r48: top-K exact hypersphere manifold reranker on top of r44.

This keeps r44's deployed decoder "mostly intact":
- stage 1 scores the full vocab with the current center-only cosine head
- stage 2 reranks only the top-K candidates with an exact `D`-point manifold

Each token `v` owns `D` literal support points on the sphere:

  p_vj = normalize(e_v + a_vj * q_j),  j = 1..D

where:
- `e_v` is the live token embedding row
- `q_j` are shared fixed orthonormal frame directions
- `a_vj` are learned token-specific amplitudes

The final candidate score is the exact log-mean-exp over those support points:

  score_v = logmeanexp_j tau * cos(z, p_vj)

To keep compute viable, the exact manifold score is only evaluated for the
top-K tokens from the center score. All other vocab items keep their original
stage-1 score.

Why this is not a surrogate collapse:
- The support points are literal points on the sphere.
- We only use algebra to evaluate their cosine score efficiently:
    z·p_vj = (z·e_v + a_vj z·q_j) / ||e_v + a_vj q_j||
- No quadratic basin / ellipsoid collapse is introduced.

Training:
- true token is always injected into the rerank set
- loss stays full-vocab CE on the refined logits
- target-side SIGReg and weak pred-side SIGReg are unchanged from r44

Eval:
- same two-stage decoder, except the true token is NOT injected
- BPB is deployment-honest for the reranker
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
from ablations.baseline import train_normuon as base  # noqa: E402


class Hyperparameters(base.Hyperparameters):
    sigreg_weight = float(os.environ.get("LEJEPA_SIGREG_WEIGHT", "-1.0"))
    lejepa_lambda = float(os.environ.get("LEJEPA_LAMBDA", "0.05"))
    sigreg_projections = int(os.environ.get("LEJEPA_SIGREG_PROJECTIONS", "64"))
    sigreg_num_points = int(os.environ.get("LEJEPA_SIGREG_NUM_POINTS", "17"))
    sigreg_t_max = float(os.environ.get("LEJEPA_SIGREG_T_MAX", "4.0"))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", "1.0"))
    temp_init = float(os.environ.get("LEJEPA_TEMP_INIT", "0.07"))
    log_temp_min = float(os.environ.get("LEJEPA_LOG_TEMP_MIN", "0.0"))
    log_temp_max = float(os.environ.get("LEJEPA_LOG_TEMP_MAX", "5.3"))
    temp_delta_max = float(os.environ.get("LEJEPA_TEMP_DELTA_MAX", "0.25"))
    pred_sigreg_weight = float(os.environ.get("LEJEPA_PRED_SIGREG_WEIGHT", "0.01"))
    topk = int(os.environ.get("LEJEPA_TOPK", "16"))
    manifold_lr = float(os.environ.get("LEJEPA_MANIFOLD_LR", "0.01"))
    manifold_chunk = int(os.environ.get("LEJEPA_MANIFOLD_CHUNK", "16"))


class LeJEPAManifoldRerankGPT(base.GPT):
    def __init__(self, args: Hyperparameters):
        super().__init__(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
        )
        self.sigreg_weight = float(args.sigreg_weight)
        self.lejepa_lambda = float(args.lejepa_lambda)
        self.use_additive = self.sigreg_weight >= 0.0

        self.log_temp = nn.Parameter(torch.tensor(math.log(1.0 / args.temp_init), dtype=torch.float32))
        self.log_temp_min = float(args.log_temp_min)
        self.log_temp_max = float(args.log_temp_max)
        self.temp_delta_max = float(args.temp_delta_max)
        self.pred_sigreg_weight = float(args.pred_sigreg_weight)
        self.topk = int(args.topk)
        self.manifold_chunk = int(args.manifold_chunk)

        self.temp_head = nn.Linear(args.model_dim, 1, bias=True)
        nn.init.zeros_(self.temp_head.weight)
        nn.init.zeros_(self.temp_head.bias)

        # Token-specific amplitudes over a shared exact frame.
        self.manifold_amp = nn.Parameter(torch.zeros(args.vocab_size, args.model_dim))
        frame = torch.randn(args.model_dim, args.model_dim, dtype=torch.float32)
        frame = torch.linalg.qr(frame, mode="reduced").Q.T.contiguous()
        self.register_buffer("manifold_frame", frame, persistent=False)
        self.manifold_log_count = float(math.log(args.model_dim))

        sigreg_dim = args.model_dim
        dirs = torch.randn(args.sigreg_projections, sigreg_dim, dtype=torch.float32)
        dirs = F.normalize(dirs, dim=-1)
        self.register_buffer("sigreg_dirs", dirs, persistent=False)
        t = torch.linspace(0.0, args.sigreg_t_max, args.sigreg_num_points, dtype=torch.float32)
        self.register_buffer("sigreg_t", t, persistent=False)

        self.register_buffer("last_ce", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_sigreg_pred", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_sigreg_tgt", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_sigreg_h", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_temp", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_temp_std", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_topk_recall", torch.zeros((), dtype=torch.float32), persistent=False)

    def sigreg_loss(self, points: Tensor) -> Tensor:
        flat = points.reshape(-1, points.size(-1)).float()
        dirs = self.sigreg_dirs.to(device=flat.device, dtype=flat.dtype)
        t = self.sigreg_t.to(device=flat.device, dtype=flat.dtype)
        proj = flat @ dirs.T
        angles = proj.unsqueeze(-1) * t.view(1, 1, -1)
        real = torch.cos(angles).mean(dim=0)
        imag = torch.sin(angles).mean(dim=0)
        gaussian_cf = torch.exp(-0.5 * t.square()).view(1, -1)
        err = (real - gaussian_cf).square() + imag.square()
        weighted_err = err * gaussian_cf
        return torch.trapz(weighted_err, t, dim=-1).mean()

    @torch.no_grad()
    def resample_sigreg_dirs(self, step: int) -> None:
        device = self.sigreg_dirs.device
        dtype = self.sigreg_dirs.dtype
        g = torch.Generator(device=device)
        g.manual_seed(int(step))
        dirs = torch.randn(self.sigreg_dirs.shape, generator=g, device=device, dtype=torch.float32)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.sigreg_dirs.copy_(dirs.to(dtype))

    @torch.no_grad()
    def manifold_diagnostics(self) -> dict[str, float]:
        tok = self.tok_emb.weight.float()
        tok_unit = F.normalize(tok, dim=-1)
        cos_mat = tok_unit @ tok_unit.T
        n = cos_mat.size(0)
        cos_mean_off = (cos_mat.sum() - cos_mat.diag().sum()) / max(cos_mat.numel() - n, 1)

        tok_centered = tok - tok.mean(dim=0, keepdim=True)
        cov = (tok_centered.T @ tok_centered) / max(tok.size(0) - 1, 1)
        eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
        eig_norm = eig / eig.sum().clamp_min(1e-12)
        eff_rank = torch.exp(-(eig_norm * eig_norm.clamp_min(1e-12).log()).sum())

        amp = self.manifold_amp.float()
        return {
            "sigreg_tok": float(self.sigreg_loss(tok).item()),
            "tok_row_norm_mean": float(tok.norm(dim=-1).mean().item()),
            "tok_row_norm_std": float(tok.norm(dim=-1).std(correction=0).item()),
            "tok_dim_std_mean": float(tok.std(dim=0, correction=0).mean().item()),
            "tok_eff_rank": float(eff_rank.item()),
            "tok_cos_off_mean": float(cos_mean_off.item()),
            "manifold_amp_mean": float(amp.abs().mean().item()),
            "manifold_amp_std": float(amp.std(correction=0).item()),
            "manifold_amp_max": float(amp.abs().max().item()),
            "dirs_sum": float(self.sigreg_dirs.to(torch.float32).sum().item()),
        }

    def encode_hidden(self, input_ids: Tensor) -> Tensor:
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
        return self.final_norm(x)

    def cosine_logits(self, flat_hidden: Tensor) -> tuple[Tensor, Tensor]:
        z = F.normalize(flat_hidden.float(), dim=-1)
        codebook = F.normalize(self.tok_emb.weight.float(), dim=-1)
        temp_delta = self.temp_delta_max * torch.tanh(self.temp_head(z).squeeze(-1))
        temp_delta = temp_delta - temp_delta.mean()
        state_log_temp = self.log_temp.float() + temp_delta
        state_log_temp = state_log_temp.clamp(min=self.log_temp_min, max=self.log_temp_max)
        temp = state_log_temp.exp()
        return (z @ codebook.T) * temp.unsqueeze(-1), temp

    def topk_candidates(self, center_logits: Tensor, target_ids: Tensor | None) -> tuple[Tensor, Tensor]:
        k = min(self.topk, center_logits.size(-1))
        cand_ids = center_logits.topk(k, dim=-1).indices
        if target_ids is None:
            return cand_ids, cand_ids.new_zeros((), dtype=torch.float32)

        target_flat = target_ids.reshape(-1)
        present = (cand_ids == target_flat.unsqueeze(1)).any(dim=-1)
        recall = present.float().mean()
        if not present.all():
            rows = (~present).nonzero(as_tuple=False).squeeze(1)
            cand_ids = cand_ids.clone()
            cand_ids[rows, -1] = target_flat[rows]
        return cand_ids, recall

    def refine_logits(
        self,
        flat_hidden: Tensor,
        center_logits: Tensor,
        temp: Tensor,
        target_ids: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        cand_ids, recall = self.topk_candidates(center_logits, target_ids)
        refined = center_logits.clone()

        work_dtype = flat_hidden.dtype
        codebook = F.normalize(self.tok_emb.weight.to(dtype=work_dtype), dim=-1)
        frame = self.manifold_frame.to(device=flat_hidden.device, dtype=work_dtype)
        z = F.normalize(flat_hidden.to(dtype=work_dtype), dim=-1)

        for start in range(0, z.size(0), self.manifold_chunk):
            end = min(start + self.manifold_chunk, z.size(0))
            z_chunk = z[start:end]
            temp_chunk = temp[start:end]
            cand_chunk = cand_ids[start:end]

            center_chunk = codebook[cand_chunk]                 # [B, K, D]
            amp_chunk = self.manifold_amp[cand_chunk].to(dtype=work_dtype)   # [B, K, D]
            base_cos = (z_chunk.unsqueeze(1) * center_chunk).sum(dim=-1)  # [B, K]

            # Stream support-point scoring over the manifold dimension to avoid
            # materializing the full [B, K, D] support tensor in the backward pass.
            support_logsumexp: Tensor | None = None
            frame_chunk = 16
            for j_start in range(0, frame.size(0), frame_chunk):
                j_end = min(j_start + frame_chunk, frame.size(0))
                frame_slice = frame[j_start:j_end]
                z_frame_slice = z_chunk @ frame_slice.T
                center_frame_slice = center_chunk @ frame_slice.T
                amp_slice = amp_chunk[:, :, j_start:j_end]

                # Exact support-point score:
                # p_vj = normalize(e_v + a_vj q_j)
                # z·p_vj = (z·e_v + a_vj z·q_j) / ||e_v + a_vj q_j||
                numerator = base_cos.unsqueeze(-1) + amp_slice * z_frame_slice.unsqueeze(1)
                denom = torch.sqrt((1.0 + 2.0 * amp_slice * center_frame_slice + amp_slice.square()).clamp_min(1e-6))
                support_scores = temp_chunk.view(-1, 1, 1) * (numerator / denom)
                chunk_lse = torch.logsumexp(support_scores, dim=-1)
                support_logsumexp = chunk_lse if support_logsumexp is None else torch.logaddexp(support_logsumexp, chunk_lse)

            exact_logits = support_logsumexp - self.manifold_log_count
            refined[start:end].scatter_(1, cand_chunk, exact_logits)
        return refined, recall

    def eval_ce_loss(self, hidden: Tensor, target_ids: Tensor) -> Tensor:
        flat = hidden.reshape(-1, hidden.size(-1))
        center_logits, temp = self.cosine_logits(flat)
        logits, _ = self.refine_logits(flat, center_logits, temp, target_ids=None)
        return F.cross_entropy(logits, target_ids.reshape(-1), reduction="mean")

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        hidden = self.encode_hidden(input_ids)
        if not self.training:
            return self.eval_ce_loss(hidden, target_ids)

        flat_h = hidden.reshape(-1, hidden.size(-1))
        tgt_flat = target_ids.reshape(-1)
        center_logits, temp = self.cosine_logits(flat_h)
        logits, topk_recall = self.refine_logits(flat_h, center_logits, temp, target_ids=tgt_flat)
        ce = F.cross_entropy(logits, tgt_flat, reduction="mean")

        target_emb = self.tok_emb(tgt_flat)
        sigreg_tgt = self.sigreg_loss(target_emb)
        sigreg_pred = self.sigreg_loss(flat_h)

        with torch.no_grad():
            sigreg_h = sigreg_pred.detach()
            temp_val = temp.mean()
            temp_std = temp.std(correction=0)

        self.last_ce.copy_(ce.detach())
        self.last_sigreg_pred.copy_(sigreg_pred.detach())
        self.last_sigreg_tgt.copy_(sigreg_tgt.detach())
        self.last_sigreg_h.copy_(sigreg_h.detach())
        self.last_temp.copy_(temp_val.detach())
        self.last_temp_std.copy_(temp_std.detach())
        self.last_topk_recall.copy_(topk_recall.detach())
        if self.use_additive:
            return ce + self.sigreg_weight * sigreg_tgt + self.pred_sigreg_weight * sigreg_pred
        return (1.0 - self.lejepa_lambda) * ce + self.lejepa_lambda * sigreg_tgt + self.pred_sigreg_weight * sigreg_pred


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

    base_model = LeJEPAManifoldRerankGPT(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, base.CastedLinear):
            module.float()
    base.restore_low_dim_params_to_fp32(base_model)
    # Keep eager mode: exact manifold reranking uses chunked Python control flow.
    model: nn.Module = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model

    named_params = list(base_model.named_parameters())
    matrix_params = [
        p
        for name, p in named_params
        if name not in {"tok_emb.weight", "lm_head.weight", "manifold_amp"}
        and p.ndim == 2
        and not any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in named_params
        if name not in {"tok_emb.weight", "lm_head.weight", "manifold_amp"}
        and (p.ndim < 2 or any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS))
    ]

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    manifold_lr = args.manifold_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_manifold = torch.optim.Adam(
        [{"params": [base_model.manifold_amp], "lr": manifold_lr, "base_lr": manifold_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = base.Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        beta2=args.normuon_beta2,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_manifold, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(2, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"metric_projected_lejepa_r48:"
        f"use_additive={base_model.use_additive} "
        f"sigreg_weight={base_model.sigreg_weight} "
        f"lejepa_lambda={base_model.lejepa_lambda} "
        f"sigreg_projections={args.sigreg_projections} "
        f"sigreg_num_points={args.sigreg_num_points} sigreg_t_max={args.sigreg_t_max} "
        f"temp_init={args.temp_init} log_temp_min={args.log_temp_min} "
        f"log_temp_max={args.log_temp_max} temp_delta_max={args.temp_delta_max} "
        f"pred_sigreg_weight={args.pred_sigreg_weight} "
        f"topk={args.topk} manifold_lr={manifold_lr} manifold_chunk={args.manifold_chunk} "
        f"head=topk_exact_hypersphere_manifold"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} manifold_lr:{manifold_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
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

    def global_grad_norm() -> float:
        total = torch.zeros((), device=device, dtype=torch.float32)
        for param in base_model.parameters():
            if param.grad is None:
                continue
            total += param.grad.detach().float().square().sum()
        return float(torch.sqrt(total).item())

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
            base_model.resample_sigreg_dirs(warmup_step)
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
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
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = base.eval_val(
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

        base_model.resample_sigreg_dirs(step)

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

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        grad_norm = global_grad_norm()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        torch.cuda.synchronize()
        training_time_ms += 1000.0 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        if step % 10 == 0 or last_step:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"ce:{base_model.last_ce.item():.4f} "
                f"sigreg_pred:{base_model.last_sigreg_pred.item():.4f} "
                f"sigreg_tgt:{base_model.last_sigreg_tgt.item():.4f} "
                f"sigreg_h:{base_model.last_sigreg_h.item():.4f} "
                f"temp_mean:{base_model.last_temp.item():.4f} "
                f"temp_std:{base_model.last_temp_std.item():.4f} "
                f"topk_recall:{base_model.last_topk_recall.item():.4f} "
                f"grad_norm:{grad_norm:.4f} train_time:{training_time_ms:.0f}ms "
                f"step_avg:{training_time_ms / step:.2f}ms"
            )
        if step % args.val_loss_every == 0 or last_step:
            diag = base_model.manifold_diagnostics()
            log0(
                "manifold_diag:"
                f"step:{step}/{args.iterations} "
                f"sigreg_tok:{diag['sigreg_tok']:.4f} "
                f"tok_row_norm_mean:{diag['tok_row_norm_mean']:.4f} "
                f"tok_row_norm_std:{diag['tok_row_norm_std']:.4f} "
                f"tok_dim_std_mean:{diag['tok_dim_std_mean']:.4f} "
                f"tok_eff_rank:{diag['tok_eff_rank']:.4f} "
                f"tok_cos_off_mean:{diag['tok_cos_off_mean']:.4f} "
                f"manifold_amp_mean:{diag['manifold_amp_mean']:.4f} "
                f"manifold_amp_std:{diag['manifold_amp_std']:.4f} "
                f"manifold_amp_max:{diag['manifold_amp_max']:.4f} "
                f"dirs_sum:{diag['dirs_sum']:.4f}"
            )
        if max_wallclock_ms is not None and training_time_ms >= max_wallclock_ms and stop_after_step is None:
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
