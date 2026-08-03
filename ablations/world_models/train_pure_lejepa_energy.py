"""
Pure LeJEPA energy ablation forked from ablations/baseline/train_normuon.py.

Training objective:
- no CE loss, no stop-grad target, no EMA, no negatives, no tied-token-row MSE
- two stochastic context views from transformer hidden states agree with a
  context center and target/span views
- target token rows are a learned view encoder used by the latent energy
  model, not the tied LM decoder table
- SIGReg regularizes the joint view distribution with resampled random
  projections each step

Eval objective:
- compute next-token CE only for validation by energy logits:
  logits_y = -||z_context - z_vocab[y]||^2 / tau
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


def parse_spans(value: str) -> tuple[int, ...]:
    spans: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        horizon = int(raw)
        if horizon <= 0:
            raise ValueError(f"PURE_LEJEPA_SPANS must contain positive integers, got {horizon}")
        if horizon not in spans:
            spans.append(horizon)
    spans = spans or [1]
    if 1 not in spans:
        raise ValueError("PURE_LEJEPA_SPANS must include 1 because eval scores next-token energy")
    return tuple(spans)


class Hyperparameters(base.Hyperparameters):
    pure_lejepa_lambda = float(os.environ.get("PURE_LEJEPA_LAMBDA", "0.05"))
    pure_lejepa_proj_dim = int(os.environ.get("PURE_LEJEPA_PROJ_DIM", os.environ.get("MODEL_DIM", "512")))
    pure_lejepa_proj_hidden = int(
        os.environ.get(
            "PURE_LEJEPA_PROJ_HIDDEN",
            str(2 * int(os.environ.get("PURE_LEJEPA_PROJ_DIM", os.environ.get("MODEL_DIM", "512")))),
        )
    )
    pure_lejepa_context_dropout = float(os.environ.get("PURE_LEJEPA_CONTEXT_DROPOUT", "0.10"))
    pure_lejepa_target_init_std = float(os.environ.get("PURE_LEJEPA_TARGET_INIT_STD", "1.0"))
    pure_lejepa_spans = parse_spans(os.environ.get("PURE_LEJEPA_SPANS", "1"))
    pure_lejepa_tau = float(os.environ.get("PURE_LEJEPA_TAU", "15.0"))
    pure_lejepa_sigreg_slices = int(os.environ.get("PURE_LEJEPA_SIGREG_SLICES", "128"))
    pure_lejepa_sigreg_points = int(os.environ.get("PURE_LEJEPA_SIGREG_POINTS", "17"))
    pure_lejepa_sigreg_t_max = float(os.environ.get("PURE_LEJEPA_SIGREG_T_MAX", "3.0"))
    pure_lejepa_sigreg_tokens = int(os.environ.get("PURE_LEJEPA_SIGREG_TOKENS", "4096"))
    pure_lejepa_target_lr = float(os.environ.get("PURE_LEJEPA_TARGET_LR", os.environ.get("TIED_EMBED_LR", "0.05")))

    if pure_lejepa_lambda < 0.0 or pure_lejepa_lambda > 1.0:
        raise ValueError(f"PURE_LEJEPA_LAMBDA must be in [0, 1], got {pure_lejepa_lambda}")
    if pure_lejepa_proj_dim <= 0:
        raise ValueError(f"PURE_LEJEPA_PROJ_DIM must be positive, got {pure_lejepa_proj_dim}")
    if pure_lejepa_proj_hidden <= 0:
        raise ValueError(f"PURE_LEJEPA_PROJ_HIDDEN must be positive, got {pure_lejepa_proj_hidden}")
    if pure_lejepa_tau <= 0.0:
        raise ValueError(f"PURE_LEJEPA_TAU must be positive, got {pure_lejepa_tau}")
    if pure_lejepa_sigreg_slices <= 0:
        raise ValueError(f"PURE_LEJEPA_SIGREG_SLICES must be positive, got {pure_lejepa_sigreg_slices}")
    if pure_lejepa_sigreg_points < 2:
        raise ValueError(f"PURE_LEJEPA_SIGREG_POINTS must be at least 2, got {pure_lejepa_sigreg_points}")
    if pure_lejepa_sigreg_t_max <= 0.0:
        raise ValueError(f"PURE_LEJEPA_SIGREG_T_MAX must be positive, got {pure_lejepa_sigreg_t_max}")
    if pure_lejepa_sigreg_tokens <= 0:
        raise ValueError(f"PURE_LEJEPA_SIGREG_TOKENS must be positive, got {pure_lejepa_sigreg_tokens}")


class LeJEPAProjector(nn.Module):
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
        x = self.fc2(x)
        return F.rms_norm(x, (x.size(-1),))


class PureLeJEPAEnergyGPT(base.GPT):
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
        # Pure energy eval does not use CE logits or an LM head.
        self.lm_head = None
        self.lejepa_lambda = float(args.pure_lejepa_lambda)
        self.eval_tau = float(args.pure_lejepa_tau)
        self.span_horizons = tuple(args.pure_lejepa_spans)
        self.sigreg_tokens = int(args.pure_lejepa_sigreg_tokens)
        self.context_projector = LeJEPAProjector(
            input_dim=args.model_dim,
            hidden_dim=args.pure_lejepa_proj_hidden,
            output_dim=args.pure_lejepa_proj_dim,
            dropout=args.pure_lejepa_context_dropout,
        )
        self.target_view_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.target_projector = LeJEPAProjector(
            input_dim=args.model_dim,
            hidden_dim=args.pure_lejepa_proj_hidden,
            output_dim=args.pure_lejepa_proj_dim,
            dropout=0.0,
        )
        nn.init.normal_(self.target_view_emb.weight, mean=0.0, std=args.pure_lejepa_target_init_std)

        dirs = torch.randn(args.pure_lejepa_sigreg_slices, args.pure_lejepa_proj_dim, dtype=torch.float32)
        dirs = F.normalize(dirs, dim=-1)
        t = torch.linspace(0.0, args.pure_lejepa_sigreg_t_max, args.pure_lejepa_sigreg_points, dtype=torch.float32)
        self.register_buffer("sigreg_dirs", dirs, persistent=False)
        self.register_buffer("sigreg_t", t, persistent=False)
        self.register_buffer("last_loss", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_inv_loss", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_sigreg_loss", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_view_norm", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_pos_energy", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_pos_cos", torch.zeros((), dtype=torch.float32), persistent=False)

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

    def target_token_views(self, target_ids: Tensor) -> Tensor:
        return self.target_projector(self.target_view_emb(target_ids))

    def target_vocab_views(self) -> Tensor:
        return self.target_projector(self.target_view_emb.weight)

    def span_views(self, token_views: Tensor) -> list[Tensor]:
        views: list[Tensor] = []
        seqlen = token_views.size(1)
        for horizon in self.span_horizons:
            if horizon > seqlen:
                continue
            if horizon == 1:
                views.append(token_views)
                continue
            csum = F.pad(token_views.float().cumsum(dim=1), (0, 0, 1, 0))
            span = (csum[:, horizon:] - csum[:, :-horizon]) / float(horizon)
            views.append(F.rms_norm(span.to(dtype=token_views.dtype), (token_views.size(-1),)))
        return views

    def _flatten_for_sigreg(self, view: Tensor) -> Tensor:
        flat = view.reshape(-1, view.size(-1))
        if flat.size(0) <= self.sigreg_tokens:
            return flat
        stride = max(flat.size(0) // self.sigreg_tokens, 1)
        return flat[::stride][: self.sigreg_tokens]

    def sigreg_loss(self, views: list[Tensor]) -> Tensor:
        dirs = self.sigreg_dirs.to(device=views[0].device, dtype=torch.float32)
        t = self.sigreg_t.to(device=views[0].device, dtype=torch.float32)
        gaussian_cf = torch.exp(-0.5 * t.square()).view(1, -1)
        stats: list[Tensor] = []
        for view in views:
            samples = self._flatten_for_sigreg(view).float()
            proj = samples @ dirs.T
            angles = proj.unsqueeze(-1) * t.view(1, 1, -1)
            real = torch.cos(angles).mean(dim=0)
            imag = torch.sin(angles).mean(dim=0)
            err = (real - gaussian_cf).square() + imag.square()
            # Epps-Pulley is a sample-size-scaled test statistic. Without the
            # n factor, collapsed views pay an O(1) cost and invariance wins.
            stats.append(2.0 * samples.size(0) * torch.trapz(err * gaussian_cf, t, dim=-1).mean())
        return torch.stack(stats).mean()

    @torch.no_grad()
    def resample_sigreg_dirs(self, step: int) -> None:
        device = self.sigreg_dirs.device
        dtype = self.sigreg_dirs.dtype
        generator = torch.Generator(device=device)
        generator.manual_seed(int(step))
        dirs = torch.randn(self.sigreg_dirs.shape, generator=generator, device=device, dtype=torch.float32)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.sigreg_dirs.copy_(dirs.to(dtype=dtype))

    def training_loss(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        hidden = self.encode_hidden(input_ids)
        context_a = self.context_projector(hidden)
        context_b = self.context_projector(hidden)
        center = (context_a + context_b) * 0.5

        token_view = self.target_token_views(target_ids)
        target_views = self.span_views(token_view)
        if not target_views:
            target_views = [token_view]

        inv_terms = [
            (context_a.float() - center.float()).square().mean(),
            (context_b.float() - center.float()).square().mean(),
        ]
        for view in target_views:
            aligned_center = center[:, : view.size(1)]
            inv_terms.append((view.float() - aligned_center.float()).square().mean())
        inv_loss = torch.stack(inv_terms).mean()
        sigreg_loss = self.sigreg_loss([context_a, context_b, *target_views])
        loss = (1.0 - self.lejepa_lambda) * inv_loss + self.lejepa_lambda * sigreg_loss

        with torch.no_grad():
            aligned_target = target_views[0]
            aligned_center = center[:, : aligned_target.size(1)]
            pos_diff = aligned_center.float() - aligned_target.float()
            pos_energy = pos_diff.square().sum(dim=-1).mean()
            pos_cos = (
                F.normalize(aligned_center.float(), dim=-1) * F.normalize(aligned_target.float(), dim=-1)
            ).sum(dim=-1).mean()
            view_norm = torch.cat(
                (
                    context_a.reshape(-1, context_a.size(-1)).float().norm(dim=-1),
                    context_b.reshape(-1, context_b.size(-1)).float().norm(dim=-1),
                    token_view.reshape(-1, token_view.size(-1)).float().norm(dim=-1),
                )
            ).mean()

        self.last_loss.copy_(loss.detach())
        self.last_inv_loss.copy_(inv_loss.detach())
        self.last_sigreg_loss.copy_(sigreg_loss.detach())
        self.last_view_norm.copy_(view_norm)
        self.last_pos_energy.copy_(pos_energy)
        self.last_pos_cos.copy_(pos_cos)
        return loss

    def eval_energy_ce_loss(self, hidden: Tensor, target_ids: Tensor) -> Tensor:
        z_context = self.context_projector(hidden).reshape(-1, self.sigreg_dirs.size(-1)).float()
        z_vocab = self.target_vocab_views().float()
        z_sq = z_context.square().sum(dim=-1, keepdim=True)
        vocab_sq = z_vocab.square().sum(dim=-1).view(1, -1)
        dist_sq = (z_sq + vocab_sq - 2.0 * (z_context @ z_vocab.T)).clamp_min(0.0)
        logits = -dist_sq / self.eval_tau
        return F.cross_entropy(logits, target_ids.reshape(-1), reduction="mean")

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        if self.training:
            return self.training_loss(input_ids, target_ids)
        hidden = self.encode_hidden(input_ids)
        return self.eval_energy_ce_loss(hidden, target_ids)

    @torch.no_grad()
    def target_view_diagnostics(self) -> dict[str, float]:
        tok = self.target_vocab_views().float()
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
            "target_view_norm_mean": float(tok.norm(dim=-1).mean().item()),
            "target_view_norm_std": float(tok.norm(dim=-1).std(correction=0).item()),
            "target_view_dim_std_mean": float(tok.std(dim=0, correction=0).mean().item()),
            "target_view_eff_rank": float(eff_rank.item()),
            "target_view_cos_off_mean": float(cos_off.item()),
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

    base_model = PureLeJEPAEnergyGPT(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, base.CastedLinear):
            module.float()
    base.restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    named_params = list(base_model.named_parameters())
    row_param_names = {"tok_emb.weight", "target_view_emb.weight"}
    matrix_params = [
        p
        for name, p in named_params
        if name not in row_param_names
        and p.ndim == 2
        and not any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in named_params
        if name not in row_param_names
        and (p.ndim < 2 or any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS))
    ]

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [
            {"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr},
            {
                "params": [base_model.target_view_emb.weight],
                "lr": args.pure_lejepa_target_lr,
                "base_lr": args.pure_lejepa_target_lr,
            },
        ],
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
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"pure_lejepa_energy:lambda={args.pure_lejepa_lambda} proj_dim={args.pure_lejepa_proj_dim} "
        f"proj_hidden={args.pure_lejepa_proj_hidden} context_dropout={args.pure_lejepa_context_dropout} "
        f"spans={','.join(str(h) for h in args.pure_lejepa_spans)} tau={args.pure_lejepa_tau} "
        f"sigreg_slices={args.pure_lejepa_sigreg_slices} sigreg_points={args.pure_lejepa_sigreg_points} "
        f"sigreg_t_max={args.pure_lejepa_sigreg_t_max} sigreg_tokens={args.pure_lejepa_sigreg_tokens} "
        f"head=pure_energy_no_ce_no_stopgrad"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} target_lr:{args.pure_lejepa_target_lr} "
        f"head_lr:0.0 matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
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
            if args.val_loss_every > 0:
                diag = base_model.target_view_diagnostics()
                log0(
                    "target_view_diag:"
                    f"step:{step}/{args.iterations} "
                    f"target_view_norm_mean:{diag['target_view_norm_mean']:.4f} "
                    f"target_view_norm_std:{diag['target_view_norm_std']:.4f} "
                    f"target_view_dim_std_mean:{diag['target_view_dim_std_mean']:.4f} "
                    f"target_view_eff_rank:{diag['target_view_eff_rank']:.4f} "
                    f"target_view_cos_off_mean:{diag['target_view_cos_off_mean']:.4f} "
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
        base_model.resample_sigreg_dirs(step)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        train_inv_loss = torch.zeros((), device=device)
        train_sigreg_loss = torch.zeros((), device=device)
        train_view_norm = torch.zeros((), device=device)
        train_pos_energy = torch.zeros((), device=device)
        train_pos_cos = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            train_inv_loss += base_model.last_inv_loss.detach()
            train_sigreg_loss += base_model.last_sigreg_loss.detach()
            train_view_norm += base_model.last_view_norm.detach()
            train_pos_energy += base_model.last_pos_energy.detach()
            train_pos_cos += base_model.last_pos_cos.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        train_inv_loss /= grad_accum_steps
        train_sigreg_loss /= grad_accum_steps
        train_view_norm /= grad_accum_steps
        train_pos_energy /= grad_accum_steps
        train_pos_cos /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
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
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"inv_loss:{train_inv_loss.item():.4f} sigreg_loss:{train_sigreg_loss.item():.4f} "
                f"view_norm:{train_view_norm.item():.4f} pos_energy:{train_pos_energy.item():.4f} "
                f"pos_cos:{train_pos_cos.item():.4f} "
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
