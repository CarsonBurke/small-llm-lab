"""Sampled GDN2 learning observables; CUDA workload, submit exclusively via mlq.

These descriptive statistics are not a full validation score or a causal
intervention. Duplicate diagnostic projections are outside any timed benchmark.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sentencepiece as spm
import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini.gated_delta_model import build_gated_delta_model
from pretraining.nanogpt_mini.gated_delta_runtime import (
    CompiledGatedDeltaLoss, gdn2_dependency_provenance,
)
from scripts.train_recurrent_slots import PackedBatches, atomic_json, build_sentencepiece_luts


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class ObservedForward:
    def __init__(self, model):
        self.model = model
        self.compiled = torch.compile(self.forward, fullgraph=False, dynamic=False)

    def forward(self, tokens):
        model = self.model
        x = model.norm1(model.embed(tokens))
        observed = []
        for block in model.blocks:
            a = block.attn
            normalized = block.norm1(x)
            _, raw_k, _, raw_b, raw_w, f_input, _ = a._project_inputs(normalized)
            convolved_k, _ = a.k_conv1d(raw_k, output_final_state=False)
            k = convolved_k.view(*tokens.shape, a.num_heads, a.head_k_dim).float()
            # FLA's L2 normalization uses sqrt(sum(x*x)+1e-6), then BF16.
            k = (k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).to(convolved_k.dtype).float()
            log_decay = (-a.A_log.float().exp().repeat_interleave(a.head_k_dim)
                         * F.softplus(a.f_proj[1](f_input).float() + a.dt_bias))
            b = raw_b.sigmoid().view(*tokens.shape, a.num_heads, a.head_k_dim).float()
            w = raw_w.sigmoid().view(*tokens.shape, a.num_heads, a.head_v_dim).float()
            mixed, _, _ = a(normalized)
            after_memory = x + mixed
            mlp = block.mlp(block.norm2(after_memory))
            branch = torch.stack((x.float().square().mean(), mixed.float().square().mean(),
                                  after_memory.float().square().mean(), mlp.float().square().mean()))
            x = after_memory + mlp
            observed.append((b, w, log_decay.view_as(b), k, branch))
        logits = model.logits(model.norm2(x))
        return logits, tuple(observed)


class CanonicalLogits:
    """Uninstrumented model path, retaining logits for numerical attribution."""
    def __init__(self, model):
        self.model = model
        self.compiled = torch.compile(self.forward, fullgraph=False, dynamic=False)

    def forward(self, tokens):
        hidden, _ = self.model.forward_hidden(tokens, segment_size=64)
        return self.model.logits(hidden)


def numerical_comparison(observed, reference, targets, canonical_loss):
    # Returning intermediate statistics can change Inductor fusion and BF16
    # rounding. Measure logits and token losses, not only cancellation in sum CE.
    delta = observed.float() - reference.float()
    observed_ce = F.cross_entropy(observed.flatten(0, 1), targets.flatten(), reduction="none")
    reference_ce = F.cross_entropy(reference.flatten(0, 1), targets.flatten(), reduction="none")
    ce_delta = observed_ce - reference_ce
    reference_rms = reference.float().square().mean().sqrt()
    relative_rms = delta.square().mean().sqrt() / reference_rms.clamp_min(1e-12)
    # CE is 2-Lipschitz in the infinity norm; this also detects inconsistent
    # target/position alignment independently of the aggregate CE criterion.
    token_bound = 2 * delta.abs().amax(-1).flatten()
    metrics = dict(logit_reference_rms=float(reference_rms),
                   logit_relative_rms_error=float(relative_rms),
                   logit_absolute_error=distribution(delta.abs()),
                   logit_max_absolute_error=float(delta.abs().max()),
                   token_ce_absolute_error=distribution(ce_delta.abs()),
                   observed_ce_sum=float(observed_ce.sum()),
                   reference_logits_ce_sum=float(reference_ce.sum()),
                   canonical_compiled_ce_sum=float(canonical_loss),
                   observed_ce_relative_error=float((observed_ce.sum() - canonical_loss).abs()
                                                    / canonical_loss.abs().clamp_min(1e-12)),
                   reference_ce_relative_error=float((reference_ce.sum() - canonical_loss).abs()
                                                     / canonical_loss.abs().clamp_min(1e-12)),
                   ce_lipschitz_violation=float((ce_delta.abs() - token_bound).clamp_min(0).max()),
                   bounds=dict(logit_relative_rms_error=.01, aggregate_ce_relative_error=.001,
                               ce_lipschitz_roundoff=1e-5),
                   interpretation="Diagnostic BF16 equivalence bounds, not bitwise parity or a quality acceptance threshold")
    finite = bool(torch.isfinite(observed).all() & torch.isfinite(reference).all()
                  & torch.isfinite(canonical_loss))
    metrics["passed"] = (finite and metrics["logit_relative_rms_error"] <= .01
                         and metrics["observed_ce_relative_error"] <= .001
                         and metrics["reference_ce_relative_error"] <= .001
                         and metrics["ce_lipschitz_violation"] <= 1e-5)
    return metrics, reference_ce.view_as(targets)


def distribution(x):
    x = x.detach().float().flatten()
    finite = torch.isfinite(x)
    valid = x[finite]
    if valid.numel() == 0:
        return dict(count=x.numel(), nonfinite=x.numel(), mean=None, quantiles=None)
    return dict(count=x.numel(), nonfinite=int((~finite).sum()), mean=float(valid.mean()),
                std=float(valid.std(correction=0)),
                quantiles=dict(zip(("p01", "p10", "p50", "p90", "p99"),
                                   torch.quantile(valid, torch.tensor([.01, .1, .5, .9, .99], device=x.device)).tolist())))


def layer_statistics(values):
    b, w, log_decay, keys, branch = values
    heads = keys.shape[2]
    report = dict(erase_gate=distribution(b), write_gate=distribution(w), log_decay=distribution(log_decay),
                  erase_saturation_fraction=float(((b < .01) | (b > .99)).float().mean()),
                  write_saturation_fraction=float(((w < .01) | (w > .99)).float().mean()),
                  memory_branch_rms_over_input=float((branch[1] / branch[0]).sqrt()),
                  mlp_branch_rms_over_input=float((branch[3] / branch[2]).sqrt()), heads=[])
    for h in range(heads):
        k = keys[:, :, h]
        flat = k.flatten(0, 1)
        moment = flat.T @ flat / flat.shape[0]
        eigen = torch.linalg.eigvalsh(moment).clamp_min(0)
        mass = eigen / eigen.sum().clamp_min(1e-30)
        entropy_rank = (-(mass * mass.clamp_min(1e-30).log()).sum()).exp()
        lag_stats = {}
        for lag in (1, 16, 64, 256):
            cosine = F.cosine_similarity(k[:, lag:], k[:, :-lag], dim=-1, eps=1e-8)
            lag_stats[str(lag)] = dict(mean=float(cosine.mean()), mean_square=float(cosine.square().mean()))
        # Stationary approximation per key channel; ignores erase/write and
        # input-conditioned changes. It is NOT an observed information lifetime.
        channel_half_lives = -math.log(2) / log_decay[:, :, h].mean((0, 1))
        report["heads"].append(dict(head=h, erase_gate=distribution(b[:, :, h]),
                                     write_gate=distribution(w[:, :, h]),
                                     decay_only_stationary_channel_half_life=distribution(channel_half_lives),
                                     uncentered_key_effective_rank=float(entropy_rank),
                                     key_dimension=flat.shape[-1], lagged_key_cosine=lag_stats))
    return report


def parameter_group(name, p):
    if name == "embed.weight":
        return "embedding_adam"
    if name == "proj.weight":
        return "head_adam"
    if getattr(p, "_no_weight_decay", False):
        return "decay_adam_no_weight_decay"
    if "_conv1d." in name:
        return "convolution_adam"
    return "dense_muon" if p.ndim == 2 else "scalars_adam"


def gradients(model, inputs, targets):
    model.train()
    loss = CompiledGatedDeltaLoss(model, 64)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        mean_loss = loss(inputs, targets) / targets.numel()
    mean_loss.backward()
    graph_breaks = loss.audit_graph_breaks()
    parameters, groups, families = {}, {}, {}
    for name, p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise RuntimeError(f"Missing/nonfinite gradient: {name}")
        norm, weight = float(p.grad.float().norm()), float(p.detach().float().norm())
        group = parameter_group(name, p)
        parameters[name] = dict(group=group, gradient_l2=norm, parameter_l2=weight,
                               gradient_rms=norm / math.sqrt(p.numel()),
                               parameter_rms=weight / math.sqrt(p.numel()),
                               gradient_rms_over_parameter_rms=norm / weight if weight > 0 else None)
        if p.ndim == 2:
            family = name.split(".", 2)[2] if name.startswith("blocks.") else name
            entry = families.setdefault(family, dict(gradient_square=0., parameter_square=0., elements=0))
            entry["gradient_square"] += norm**2
            entry["parameter_square"] += weight**2
            entry["elements"] += p.numel()
        entry = groups.setdefault(group, dict(gradient_square=0., parameter_square=0., parameters=0, zero_gradient_tensors=0))
        entry["gradient_square"] += norm**2
        entry["parameter_square"] += weight**2
        entry["parameters"] += p.numel()
        entry["zero_gradient_tensors"] += norm == 0
    for entry in groups.values():
        entry["gradient_l2"] = math.sqrt(entry.pop("gradient_square"))
        entry["parameter_l2"] = math.sqrt(entry.pop("parameter_square"))
        entry["gradient_rms"] = entry["gradient_l2"] / math.sqrt(entry["parameters"])
        entry["parameter_rms"] = entry["parameter_l2"] / math.sqrt(entry["parameters"])
        entry["gradient_rms_over_parameter_rms"] = (entry["gradient_l2"] / entry["parameter_l2"]
                                                       if entry["parameter_l2"] > 0 else None)
    for entry in families.values():
        gs, ws = entry.pop("gradient_square"), entry.pop("parameter_square")
        entry["gradient_rms"] = math.sqrt(gs / entry["elements"])
        entry["parameter_rms"] = math.sqrt(ws / entry["elements"])
        entry["gradient_rms_over_parameter_rms"] = math.sqrt(gs / ws) if ws > 0 else None
    model.zero_grad(set_to_none=True)
    return dict(loss_reduction="mean CE over sampled tokens (not optimizer update)",
                mean_loss=float(mean_loss.detach()), groups=groups, matrix_families=families, parameters=parameters,
                graph_breaks=graph_breaks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "ablation_results/gdn2_fullwidth_fla_kfirst_saved_1k")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.rows <= 64:
        parser.error("rows must be between 1 and64 full1024-token sequences")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required; submit through mlq")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    run = args.run_dir.resolve()
    checkpoint_path = run / "model.pt"
    checkpoint_hash = digest(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    training = json.loads((run / "config.json").read_text())
    result = json.loads((run / "result.json").read_text())
    if (result.get("status") != "completed" or result.get("completed_steps") != 1000
            or checkpoint.get("completed_steps") != 1000 or checkpoint.get("train_seq_len") != 1024
            or checkpoint["model_config"] != training["model_config"]):
        raise ValueError("Require the completed matching1000-update checkpoint")
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    loader = PackedBatches(str(ROOT / training["data_path"] / "fineweb_val_*.bin"), 1048576, 1024)
    panel_x, panel_y = loader.next()
    row_ids = torch.linspace(0, 1023, args.rows, device="cuda").round().long()
    inputs, targets = panel_x[row_ids], panel_y[row_ids]
    tokenizer_path = ROOT / training["tokenizer"]
    if digest(tokenizer_path) != training["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed")
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    base, leading, boundary = build_sentencepiece_luts(tokenizer, vocab_size=1024, device="cuda")
    panel_bytes = base[panel_y].long() + (leading[panel_y] & ~boundary[panel_x.long()]).long()
    if int(panel_bytes.sum()) != training["val_bytes"]:
        raise ValueError("Canonical validation panel differs")
    byte_counts = panel_bytes[row_ids]
    sources = [Path(__file__), ROOT / "pretraining/nanogpt_mini/gated_delta_model.py",
               ROOT / "pretraining/nanogpt_mini/gated_delta_runtime.py",
               ROOT / "pretraining/nanogpt_mini/nanogpt_mini_model.py",
               ROOT / "scripts/train_recurrent_slots.py"]
    sources += [p for p in (ROOT / "pretraining/gated_delta/vendor").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    report = dict(status="running", optimizer_updates=0, rows=row_ids.tolist(), seq_len=1024,
                  sample_tokens=targets.numel(), sample_bytes=int(byte_counts.sum()),
                  checkpoint_sha256=checkpoint_hash, checkpoint_path=str(checkpoint_path),
                  tokenizer_sha256=digest(tokenizer_path), validation_shard_sha256=digest(loader.files[loader.shard_index]),
                  model_config=checkpoint["model_config"], torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                  installed_fla=gdn2_dependency_provenance(),
                  source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
                  caveats=["Descriptive sampled observations, not full validation or causal proof.",
                           "Decay-only half-lives ignore content-dependent erase/write and are not information lifetimes.",
                           "Keys reconstructed with FLA normalization epsilon; covariance is uncentered across sampled rows.",
                           "Initialization can have zero upstream gradients because the LM output weight starts at zero.",
                           "Diagnostic projections/convolutions are duplicated; no throughput claims.",
                           "Instrumentation may alter BF16 compiler fusion; numerical errors are recorded against uninstrumented compiled logits and CE.",
                           "Reported sample loss bins use uninstrumented compiled logits; layer observables use the instrumented path.",
                           "Gradient norms use mean CE and are not Muon/Adam update magnitudes."])
    for p in sources:
        destination = args.output / "source" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    path = args.output / "diagnostic.json"
    atomic_json(path, report)
    try:
        for arm in ("initialization", "checkpoint"):
            torch.manual_seed(training["seed"])
            model = build_gated_delta_model(checkpoint["model_config"]).cuda().eval()
            if arm == "checkpoint":
                model.load_state_dict(checkpoint["model"], strict=True)
            wrapper = ObservedForward(model)
            canonical_logits = CanonicalLogits(model)
            audit = CompiledGatedDeltaLoss(model, 64)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                logits, observations = wrapper.compiled(inputs)
                reference_logits = canonical_logits.compiled(inputs)
                canonical = audit(inputs, targets)
                parity, per_token = numerical_comparison(logits, reference_logits, targets, canonical)
            report.setdefault("numerical_comparisons", {})[arm] = parity
            atomic_json(path, report)
            if not parity["passed"]:
                raise RuntimeError(f"Instrumented BF16 comparison failed: {parity}")
            audit.audit_graph_breaks()
            layers = [layer_statistics(value) for value in observations]
            bins = [dict(start=start, end=start + 128,
                         mean_nll=float(per_token[:, start:start+128].mean()),
                         bpb=float(per_token[:, start:start+128].double().sum() /
                                   (math.log(2) * byte_counts[:, start:start+128].sum())))
                    for start in range(0, 1024, 128)]
            arm_result = dict(sample_bpb=float(per_token.double().sum() / (math.log(2) * byte_counts.sum())),
                              position_bins=bins, layers=layers)
            del logits, reference_logits, observations, per_token, canonical, wrapper, canonical_logits, audit
            gc.collect()
            arm_result["gradients"] = gradients(model, inputs, targets)
            report[arm] = arm_result
            del model
            gc.collect()
            torch.cuda.empty_cache()
            atomic_json(path, report)
        if digest(checkpoint_path) != checkpoint_hash:
            raise RuntimeError("Checkpoint changed during diagnostic")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(path, report)


if __name__ == "__main__":
    main()
