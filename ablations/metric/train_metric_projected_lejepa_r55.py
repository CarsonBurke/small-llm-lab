"""
LeJEPA r55: CALM ES with jittered target + pairwise ES on codebook, no detach.

Design premise: "codebook ideally shaped by pred loss to the extent it
doesn't collapse". Prior variants with detach (r50/r53/r54) plateaued
because the codebook only received input-embedding gradient — no
organizational signal from prediction error. Removing detach restores
codebook learning through attract, but r48/r49 showed that SIGReg
(per-row grad ~1/V) cannot resist attract's collapse pull.

Fix: borrow CALM's diversity term form and apply it to the CODEBOOK.

  L_repulse = −(1/(K(K−1))) · Σ_{i≠j sampled} ‖e_i − e_j‖

Same mathematical form as the ES diversity term on model samples — but
on codebook rows. Per-pair gradient O(1); per-row force accumulates
over K pairs, giving ~K× stronger anti-collapse than SIGReg at the same
weight. At K=256 that's ~256× the pressure r48/r49 had.

Loss:
  e_y     = tok_emb(y)                    # grad flows (no detach)
  z_m     = e_y + σ_tgt · η_m             # M jittered target samples
  z̃_n    = noise_head(h, ε_n)            # N model samples
  es_pred = (2/(NM)) Σ_{n,m} ‖z̃_n − z_m‖
          − (1/(N(N−1))) Σ_{n≠k} ‖z̃_n − z̃_k‖
  C       = tok_emb.weight[random K indices]
  cb_rep  = −(1/(K(K−1))) Σ_{i≠j} ‖C_i − C_j‖
  loss    = es_pred + β · cb_rep + η · sigreg_pred

Balance knob β: shape vs freeze.
  β too small → attract dominates → codebook collapse (r48/r49 mode)
  β too large → cb_rep dominates → uniform codebook can't specialize
Start β=1.0 (same order as es_pred's attract term). Diagnostics:
tok_eff_rank should hold near init (~340); pos_cos should climb faster
than r50/r54 if the codebook is co-organizing with the predictor.

Target jitter (M=4, σ_tgt=2.0) retained for proper ES on predictions.

Hyperparameters (env): r54's plus
  LEJEPA_CB_REPULSE_WEIGHT  — β, default 1.0
  LEJEPA_CB_REPULSE_SUBSET  — K, default 256
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
    sigreg_projections = int(os.environ.get("LEJEPA_SIGREG_PROJECTIONS", "64"))
    sigreg_num_points = int(os.environ.get("LEJEPA_SIGREG_NUM_POINTS", "17"))
    sigreg_t_max = float(os.environ.get("LEJEPA_SIGREG_T_MAX", "4.0"))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", "1.0"))
    pred_sigreg_weight = float(os.environ.get("LEJEPA_PRED_SIGREG_WEIGHT", "0.01"))
    eval_temp = float(os.environ.get("LEJEPA_EVAL_TEMP", "15.0"))
    # CALM Energy Score config:
    num_es_samples = int(os.environ.get("LEJEPA_NUM_ES_SAMPLES", "4"))
    num_target_samples = int(os.environ.get("LEJEPA_NUM_TARGET_SAMPLES", "4"))
    target_jitter_std = float(os.environ.get("LEJEPA_TARGET_JITTER_STD", "2.0"))
    noise_dim = int(os.environ.get("LEJEPA_NOISE_DIM", "64"))
    es_noise_std = float(os.environ.get("LEJEPA_ES_NOISE_STD", "1.0"))
    es_eps = float(os.environ.get("LEJEPA_ES_EPS", "1e-6"))
    # Codebook pairwise-ES anti-collapse:
    cb_repulse_weight = float(os.environ.get("LEJEPA_CB_REPULSE_WEIGHT", "1.0"))
    cb_repulse_subset = int(os.environ.get("LEJEPA_CB_REPULSE_SUBSET", "256"))


class LeJEPAMetricGramGPT(base.GPT):
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
        self.pred_sigreg_weight = float(args.pred_sigreg_weight)
        self.eval_temp = float(args.eval_temp)
        self.num_es_samples = int(args.num_es_samples)
        self.num_target_samples = int(args.num_target_samples)
        self.target_jitter_std = float(args.target_jitter_std)
        self.noise_dim = int(args.noise_dim)
        self.es_noise_std = float(args.es_noise_std)
        self.es_eps = float(args.es_eps)
        self.cb_repulse_weight = float(args.cb_repulse_weight)
        self.cb_repulse_subset = int(args.cb_repulse_subset)

        d = args.model_dim
        # Noise-conditioned generator head: small residual MLP that maps
        # (hidden, eps) -> sample offset. Initialized to produce near-zero
        # output so initial samples cluster around hidden and attract has a
        # clean starting signal.
        self.noise_proj = nn.Linear(self.noise_dim, d, bias=False)
        self.noise_fc = nn.Linear(d, d, bias=False)
        self.noise_out = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.noise_proj.weight, std=0.02)
        nn.init.normal_(self.noise_fc.weight, std=0.02)
        nn.init.zeros_(self.noise_out.weight)

        dirs = torch.randn(args.sigreg_projections, d, dtype=torch.float32)
        dirs = F.normalize(dirs, dim=-1)
        self.register_buffer("sigreg_dirs", dirs, persistent=False)
        t = torch.linspace(0.0, args.sigreg_t_max, args.sigreg_num_points, dtype=torch.float32)
        self.register_buffer("sigreg_t", t, persistent=False)
        self.register_buffer("last_ce", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_attract", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_diversity", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_es", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_cb_repulse", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_pos_cos", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_sigreg_pred", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_sigreg_h", torch.zeros((), dtype=torch.float32), persistent=False)

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

        return {
            "sigreg_tok": float(self.sigreg_loss(tok).item()),
            "tok_row_norm_mean": float(tok.norm(dim=-1).mean().item()),
            "tok_row_norm_std": float(tok.norm(dim=-1).std(correction=0).item()),
            "tok_dim_std_mean": float(tok.std(dim=0, correction=0).mean().item()),
            "tok_eff_rank": float(eff_rank.item()),
            "tok_cos_off_mean": float(cos_mean_off.item()),
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

    def sample_generator(self, hidden: Tensor, n_samples: int, noise_std: float) -> Tensor:
        """Noise-conditioned generator head. Produces n_samples per position.

        hidden: [..., D]. Returns [..., n_samples, D].
        Deterministic when noise_std == 0 (samples coincide with hidden).
        """
        d = hidden.size(-1)
        shape = hidden.shape[:-1] + (n_samples, d)
        h_exp = hidden.unsqueeze(-2).expand(shape).contiguous()
        if noise_std > 0.0:
            eps = torch.randn(hidden.shape[:-1] + (n_samples, self.noise_dim), device=hidden.device, dtype=hidden.dtype)
            eps = eps * noise_std
            noise_h = self.noise_proj(eps)  # [..., n, D]
            mixed = h_exp + noise_h
            delta = self.noise_out(F.silu(self.noise_fc(mixed)))
            return h_exp + delta
        # Deterministic path: use learned residual at eps=0.
        delta = self.noise_out(F.silu(self.noise_fc(h_exp)))
        return h_exp + delta

    def eval_ce_loss(self, hidden: Tensor, target_ids: Tensor) -> Tensor:
        flat = hidden.reshape(-1, hidden.size(-1))
        # Eval uses the deterministic (eps=0) sample to decode.
        z_det = self.sample_generator(flat.float(), 1, noise_std=0.0).squeeze(-2)
        z = F.normalize(z_det, dim=-1)
        codebook = F.normalize(self.tok_emb.weight.float(), dim=-1)
        logits = (z @ codebook.T) * self.eval_temp
        return F.cross_entropy(logits, target_ids.reshape(-1), reduction="mean")

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        hidden = self.encode_hidden(input_ids)
        if not self.training:
            return self.eval_ce_loss(hidden, target_ids)

        flat_h = hidden.reshape(-1, hidden.size(-1))
        tgt_flat = target_ids.reshape(-1)
        # NO detach: attract gradient flows into the codebook so it learns
        # semantic neighborhoods from prediction error. Collapse is
        # prevented by the pairwise ES repulsion term below, not by
        # stopping the gradient.
        tgt_emb = self.tok_emb(tgt_flat)

        # Multi-sample prediction via the noise-conditioned generator head.
        # Shape: [B*T, N, D]
        n = self.num_es_samples
        m = self.num_target_samples
        z_samples = self.sample_generator(flat_h.float(), n, noise_std=self.es_noise_std)

        # Build M jittered target samples by adding isotropic Gaussian noise
        # to the (detached) target embedding. This gives ES a real
        # distributional target instead of a delta — the proper optimum now
        # matches a Gaussian ball of radius σ/√3 in predictive spread.
        tgt_base = tgt_emb.unsqueeze(-2).to(z_samples.dtype)  # [N_batch, 1, D]
        if m >= 1 and self.target_jitter_std > 0.0:
            eta = torch.randn(tgt_emb.shape[:-1] + (m, tgt_emb.size(-1)), device=tgt_emb.device, dtype=z_samples.dtype)
            tgt_samples = tgt_base + self.target_jitter_std * eta  # [N_batch, M, D]
        else:
            tgt_samples = tgt_base  # [N_batch, 1, D]
            m = 1

        # Energy Score (α = 1, L2 distance, strictly proper):
        #   ES = (2/(N·M)) Σ_{n,m} ||z̃_n − z_m||  −  (1/(N(N-1))) Σ_{n≠k} ||z̃_n − z̃_k||
        eps = self.es_eps
        # attract term: pairwise between model samples and target samples
        diff_target = z_samples.unsqueeze(-2) - tgt_samples.unsqueeze(-3)  # [N_batch, N, M, D]
        attract_dist = torch.sqrt(diff_target.pow(2).sum(dim=-1) + eps)  # [N_batch, N, M]
        attract = 2.0 * attract_dist.mean()

        # diversity term: pairwise distance between model samples at same pos
        if n >= 2:
            diff_samples = z_samples.unsqueeze(-2) - z_samples.unsqueeze(-3)  # [N_batch, N, N, D]
            sample_dist = torch.sqrt(diff_samples.pow(2).sum(dim=-1) + eps)  # [N_batch, N, N]
            mask = 1.0 - torch.eye(n, device=z_samples.device, dtype=z_samples.dtype)
            diversity = (sample_dist * mask).sum(dim=(-1, -2)) / (n * (n - 1))  # [N_batch]
            diversity_mean = diversity.mean()
        else:
            diversity_mean = torch.zeros((), device=z_samples.device, dtype=z_samples.dtype)

        es_loss = attract - diversity_mean

        # Diagnostic: cosine alignment of the deterministic sample with the
        # unjittered (clean) target embedding.
        with torch.no_grad():
            z_det = self.sample_generator(flat_h.float(), 1, noise_std=0.0).squeeze(-2)
            pos_cos = F.cosine_similarity(z_det, tgt_emb.to(z_det.dtype), dim=-1).mean()

        # Pairwise ES repulsion on sampled codebook rows — CALM's diversity
        # term form, applied to the codebook. Rows are normalized to the
        # unit sphere before the pairwise distance so the loss rewards
        # ANGULAR separation (what the eval `z @ codebook.T` consumes),
        # not growing row norms (which would satisfy raw L2 repulsion by
        # simple norm inflation and does not help decoding).
        # Equivalent form: ||x_hat - y_hat|| = sqrt(2(1 - cos(x, y))), so
        # minimizing -mean pairwise is equivalent to minimizing mean cos.
        k = self.cb_repulse_subset
        v = self.tok_emb.weight.size(0)
        if k > v:
            k = v
        idx = torch.randint(0, v, (k,), device=flat_h.device)
        cb_rows = self.tok_emb.weight[idx]  # [K, D], grad flows
        cb_unit = F.normalize(cb_rows, dim=-1)
        cb_diff = cb_unit.unsqueeze(-2) - cb_unit.unsqueeze(-3)  # [K, K, D]
        cb_dist = torch.sqrt(cb_diff.pow(2).sum(dim=-1) + eps)  # [K, K], in [0, 2]
        cb_mask = 1.0 - torch.eye(k, device=cb_rows.device, dtype=cb_rows.dtype)
        # Negative mean pairwise angular distance on sphere. Value in [-2, 0].
        cb_repulse = -(cb_dist * cb_mask).sum() / (k * (k - 1))

        # Predictor SIGReg guardrail on the hidden state (NOT on samples).
        sigreg_pred = self.sigreg_loss(flat_h)
        with torch.no_grad():
            sigreg_h = sigreg_pred.detach()

        self.last_ce.copy_(es_loss.detach())
        self.last_attract.copy_(attract.detach())
        self.last_diversity.copy_(diversity_mean.detach())
        self.last_es.copy_(es_loss.detach())
        self.last_cb_repulse.copy_(cb_repulse.detach())
        self.last_pos_cos.copy_(pos_cos.detach())
        self.last_sigreg_pred.copy_(sigreg_pred.detach())
        self.last_sigreg_h.copy_(sigreg_h.detach())

        return (
            es_loss
            + self.cb_repulse_weight * cb_repulse
            + self.pred_sigreg_weight * sigreg_pred
        )


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

    base_model = LeJEPAMetricGramGPT(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, base.CastedLinear):
            module.float()
    base.restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    named_params = list(base_model.named_parameters())
    matrix_params = [
        p
        for name, p in named_params
        if name not in {"tok_emb.weight", "lm_head.weight"}
        and p.ndim == 2
        and not any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in named_params
        if name not in {"tok_emb.weight", "lm_head.weight"}
        and (p.ndim < 2 or any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS))
    ]

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
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
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"metric_projected_lejepa_r55:"
        f"num_es_samples={base_model.num_es_samples} "
        f"num_target_samples={base_model.num_target_samples} "
        f"target_jitter_std={base_model.target_jitter_std} "
        f"noise_dim={base_model.noise_dim} "
        f"es_noise_std={base_model.es_noise_std} "
        f"es_eps={base_model.es_eps} "
        f"cb_repulse_weight={base_model.cb_repulse_weight} "
        f"cb_repulse_subset={base_model.cb_repulse_subset} "
        f"sigreg_projections={args.sigreg_projections} "
        f"sigreg_num_points={args.sigreg_num_points} sigreg_t_max={args.sigreg_t_max} "
        f"pred_sigreg_weight={args.pred_sigreg_weight} "
        f"eval_temp={args.eval_temp} "
        f"head=calm_es_jittered_cb_pairwise_repulse"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
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
                f"es:{base_model.last_es.item():.4f} "
                f"attract:{base_model.last_attract.item():.4f} "
                f"diversity:{base_model.last_diversity.item():.4f} "
                f"cb_repulse:{base_model.last_cb_repulse.item():.4f} "
                f"pos_cos:{base_model.last_pos_cos.item():.4f} "
                f"sigreg_pred:{base_model.last_sigreg_pred.item():.4f} "
                f"sigreg_h:{base_model.last_sigreg_h.item():.4f} "
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
