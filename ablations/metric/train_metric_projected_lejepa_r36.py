"""
LeJEPA r36: paper-faithful two-view LeJEPA + CE decoder. LeJEPA pretrains
the representation; tied-head CE trains the decoder; both run every step.

  x_clean  = input_ids                                     # view 1
  x_aug    = token_mask(input_ids, p=aug_p, mask=<unk>)    # view 2
  x_both   = cat([x_clean, x_aug], dim=0)                  # one forward
  h_both   = encode_hidden(x_both)                         # [2B, T, D]
  h_clean, h_aug = h_both.split(B, 0)                      # [B, T, D] each

  # LeJEPA in projector space (paper-exact)
  z        = projector(h_both).view(2, B, T, D_proj)       # [V=2, B, T, D_p]
  z_mean   = z.mean(0, keepdim=True)                       # per (b, t) mean
  inv      = (z - z_mean).square().mean()                  # view invariance
  sigreg   = Epps-Pulley(z.reshape(-1, D_proj))            # N(0, I) marginal
  lejepa   = (1-λ) * inv + λ * sigreg                      # λ=0.05

  # CE decoder on clean hidden (what BPB reads)
  flat_h   = h_clean.reshape(-1, D)
  logits   = softcap · tanh(flat_h @ tok_emb.T / softcap)
  ce       = cross_entropy(logits, targets)

  loss     = ce + μ · lejepa                               # μ=1.0 default

Why this is structurally right (where r22–r35 were not):
- Two views of the SAME content via token-level augmentation. The encoder
  sees corrupted input; representations must be content-stable under
  surface perturbation. This is the LM analog of DINO's RandomResizedCrop
  + ColorJitter: preserve semantics, perturb the surface.
- Dedicated training-only projector. Paper puts SIGReg + inv on projector
  output, not on backbone. We match exactly: 1-layer Linear(D → D_proj),
  applied to both views, dropped from the eval path (CE reads backbone
  hidden directly, same as baseline).
- inv_loss uses paper form: (z - z.mean(0)).square().mean(). For V=2 this
  reduces to symmetric MSE/2, but the multi-view form is ready if we
  bump V later.
- SIGReg on {z_v for v, b, t} treats every (view, position, sequence) as
  a sample — N(0, I) marginal over projected hidden.
- CE stays intact on the clean view. Its two-sided tok_emb gradient
  (input-side via embedding, output-side via tied head) is the decoder
  training signal, identical to baseline. LeJEPA only adds a term.

Cost: ONE forward pass on [2B, T, D]. Doubles per-step FLOPs and memory
vs baseline (~21GB peak on 5090, fits 32GB; ~1.1 s/step vs 588 ms). For
fair 2000-step comparison this takes ~40 min.

Hyperparameters (env):
  LEJEPA_LAMBDA          — inv-vs-sigreg convex weight, default 0.05.
  LEJEPA_MU              — weight of lejepa term vs ce, default 1.0.
  LEJEPA_AUG_P           — token-mask probability for view 2, default 0.15.
  LEJEPA_AUG_TOKEN_ID    — mask token id, default 3 (<unk>).
  LEJEPA_PROJ_DIM        — projector output dim, default 64. Paper: 64–512.
  LEJEPA_SIGREG_*        — SIGReg quadrature knobs (unchanged).

Why this can actually help (where r35 can't):
- r35: SIGReg on hidden as aux on CE. Can't create a new objective — CE
  is already doing next-token discrimination; SIGReg is a mild regularizer.
- r36: LeJEPA is a SECOND training signal that rewards augmentation
  invariance. The trunk has to do two things well: (a) discriminate the
  next token (CE) and (b) be stable under input-token masking (LeJEPA).
  Goal (b) gives the trunk a pressure CE alone doesn't provide.

Known risks:
- 2× wallclock per step. In the 10-min final, this halves step count. The
  representation improvement has to more than compensate.
- Augmentation strength (aug_p): too low → inv is trivially zero; too
  high → context destroyed, CE can't learn. 0.15 is BERT-style starting
  point; may need 0.10 or 0.20 sweep.
- D_proj choice. Paper's vision ViT uses 64–512; small D_proj = tight
  information bottleneck in the projector, can over-regularize trunk.
  Start 64.
- projector.weight goes into the Muon group automatically (it's a 2D
  non-embedding param), so no optimizer plumbing needed.

Eval: tied-head CE on the clean hidden, identical to baseline.
"""

from __future__ import annotations

import copy
import io
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
    # Paper-faithful two-view LeJEPA knobs.
    #   lejepa = (1 - λ) * inv + λ * sigreg     # inner convex
    #   loss   = ce + μ * lejepa                # outer additive
    # λ=0.05 is the paper's vision default; μ=1.0 is a neutral starting weight
    # that we'll sweep if needed (ce ≈ log V = 9.0 at init falls to ~1.5;
    # lejepa ≈ O(0.1–1.0) — μ=1.0 keeps lejepa a meaningful signal without
    # dominating ce).
    lejepa_lambda = float(os.environ.get("LEJEPA_LAMBDA", "0.05"))
    lejepa_mu = float(os.environ.get("LEJEPA_MU", "1.0"))
    lejepa_aug_p = float(os.environ.get("LEJEPA_AUG_P", "0.15"))
    lejepa_aug_token_id = int(os.environ.get("LEJEPA_AUG_TOKEN_ID", "3"))  # <unk>
    lejepa_proj_dim = int(os.environ.get("LEJEPA_PROJ_DIM", "64"))
    sigreg_projections = int(os.environ.get("LEJEPA_SIGREG_PROJECTIONS", "64"))
    sigreg_num_points = int(os.environ.get("LEJEPA_SIGREG_NUM_POINTS", "17"))
    sigreg_t_max = float(os.environ.get("LEJEPA_SIGREG_T_MAX", "4.0"))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", "1.0"))


class LeJEPAMetricGPT(base.GPT):
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
        self.lejepa_lambda = float(args.lejepa_lambda)
        self.lejepa_mu = float(args.lejepa_mu)
        self.lejepa_aug_p = float(args.lejepa_aug_p)
        self.lejepa_aug_token_id = int(args.lejepa_aug_token_id)
        self.proj_dim = int(args.lejepa_proj_dim)

        # Training-only projector: Linear(D → D_proj). Applied to both views
        # and dropped from the eval path. Paper uses MLP with BN but a single
        # Linear is the minimum-parameter version and avoids BN's batch-
        # statistics complications in a compiled + DDP setup. base.CastedLinear
        # wouldn't add anything here (same fp32 path as Muon cares about).
        self.projector = nn.Linear(args.model_dim, self.proj_dim, bias=False)
        # SIGReg projection directions live in projector space (D_proj), NOT
        # backbone hidden space. Re-allocate sigreg_dirs at proj_dim.
        dirs = torch.randn(args.sigreg_projections, self.proj_dim, dtype=torch.float32)
        dirs = F.normalize(dirs, dim=-1)
        self.register_buffer("sigreg_dirs", dirs, persistent=False)
        t = torch.linspace(0.0, args.sigreg_t_max, args.sigreg_num_points, dtype=torch.float32)
        self.register_buffer("sigreg_t", t, persistent=False)
        # Per-step loss-component scalars (picked up by scripts/ablation.py's regex
        # parser from the train log line and forwarded to tensorboard).
        self.register_buffer("last_ce", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_inv", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("last_sigreg", torch.zeros((), dtype=torch.float32), persistent=False)

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
        # SIGReg now lives in D_proj space; sigreg_dirs cannot be applied to
        # tok_emb rows directly, so we drop sigreg_tok and keep the purely
        # geometric stats that don't depend on projection dim.
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
        # Apply final_norm so flat_h matches the base-model convention
        # (post-norm hidden → tied head). Also makes flat_h the ViT-style
        # "encoder output z" analog for SIGReg.
        return self.final_norm(x)

    def eval_ce_loss(self, hidden: Tensor, target_ids: Tensor) -> Tensor:
        flat = hidden.reshape(-1, hidden.size(-1))
        logits = F.linear(flat, self.tok_emb.weight)
        softcap = self.logit_softcap
        logits = softcap * torch.tanh(logits / softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")

    def _augment(self, input_ids: Tensor) -> Tensor:
        # Token-level corruption: replace a fraction of positions with the
        # mask token id. We use torch.rand (not rand_like on an int tensor —
        # that's not supported) and broadcast-compare to aug_p. The mask id
        # should be a single token that the encoder already has an embedding
        # for — default <unk>=3. Its embedding effectively becomes "corrupted
        # position" under this training signal.
        noise = torch.rand(input_ids.shape, device=input_ids.device, dtype=torch.float32)
        mask = noise < self.lejepa_aug_p
        aug_id = torch.full_like(input_ids, self.lejepa_aug_token_id)
        return torch.where(mask, aug_id, input_ids)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        if not self.training:
            hidden = self.encode_hidden(input_ids)
            return self.eval_ce_loss(hidden, target_ids)

        # Build the two views and run ONE forward pass on the concatenated
        # batch. This keeps the compiled graph single-shot; the cost is 2x
        # the baseline per-step compute and memory, which the 5090 + the
        # doubled batch fits in (peak expected ~21GB of 32GB).
        B, T = input_ids.shape
        x_aug = self._augment(input_ids)
        x_both = torch.cat([input_ids, x_aug], dim=0)      # [2B, T]
        h_both = self.encode_hidden(x_both)                 # [2B, T, D]
        h_clean = h_both[:B]

        # Paper-exact LeJEPA in projector space.
        # z shape: [V=2, B, T, D_proj]. Mean over V yields per-(b, t) target
        # that each view is pulled toward. For V=2 this reduces to symmetric
        # MSE/2, but the formulation generalizes if we ever add more views.
        #
        # RMSNorm on the projector output is a compile-friendly stand-in for
        # the paper's BatchNorm-inside-projector. Without it, Linear(D, D_proj)
        # produces z with std ≈ 1/sqrt(D_in) ≈ 0.04 at init — so |z1 - z2|²
        # starts trivially close to zero and inv has no gradient to work with
        # (observed in v1: inv plateaued at ~0.002 from step 10 onward). With
        # unit-RMS z, inv starts ~2 and can actually decrease as the encoder
        # learns augmentation-invariance.
        z_both = self.projector(h_both)                     # [2B, T, D_proj]
        z_both = F.rms_norm(z_both, (z_both.size(-1),))
        z = z_both.view(2, B, T, self.proj_dim)
        z_mean = z.mean(dim=0, keepdim=True)                # [1, B, T, D_proj]
        inv = (z - z_mean).square().mean()

        # SIGReg marginal-Gaussianity across every projected vector. Treats
        # every (view, sample, position) as an independent draw from the
        # target N(0, I) distribution. flat shape: [2 * B * T, D_proj].
        sigreg = self.sigreg_loss(z_both)

        lejepa = (1.0 - self.lejepa_lambda) * inv + self.lejepa_lambda * sigreg

        # Tied-head CE on clean hidden — the decoder training signal. CE
        # provides gradient to both flat_h AND tok_emb.weight (two-sided,
        # unchanged from baseline). target_ids are un-augmented, so the CE
        # task is identical to what the 1.30-BPB baseline trains against.
        flat_h = h_clean.reshape(-1, h_clean.size(-1))
        logits_proj = F.linear(flat_h, self.tok_emb.weight)
        softcap = self.logit_softcap
        logits = softcap * torch.tanh(logits_proj / softcap)
        ce = F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")

        self.last_ce.copy_(ce.detach())
        self.last_inv.copy_(inv.detach())
        self.last_sigreg.copy_(sigreg.detach())
        return ce + self.lejepa_mu * lejepa


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

    base_model = LeJEPAMetricGPT(args).to(device).bfloat16()
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
        f"metric_projected_lejepa_r36:"
        f"lejepa_lambda={base_model.lejepa_lambda} "
        f"lejepa_mu={base_model.lejepa_mu} "
        f"lejepa_aug_p={base_model.lejepa_aug_p} "
        f"lejepa_aug_token_id={base_model.lejepa_aug_token_id} "
        f"proj_dim={base_model.proj_dim} "
        f"sigreg_projections={args.sigreg_projections} "
        f"sigreg_num_points={args.sigreg_num_points} sigreg_t_max={args.sigreg_t_max} "
        f"logit_softcap={args.logit_softcap}"
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
                f"ce:{base_model.last_ce.item():.4f} "
                f"inv:{base_model.last_inv.item():.4f} "
                f"sigreg:{base_model.last_sigreg.item():.4f} "
                f"grad_norm:{grad_norm:.4f} train_time:{training_time_ms:.0f}ms "
                f"step_avg:{training_time_ms / step:.2f}ms"
            )
        if step % args.val_loss_every == 0 or last_step:
            diag = base_model.manifold_diagnostics()
            log0(
                "manifold_diag:"
                f"step:{step}/{args.iterations} "
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
