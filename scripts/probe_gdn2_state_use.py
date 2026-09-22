"""Paired cached-state erasure diagnostic; submit CUDA execution through mlq.

Erases only recurrent matrices at a single prefix boundary. Preserves short-conv
history and all suffix writes. This is a checkpoint intervention, not training
or an estimate of the benefit of learning a different architecture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini.gated_delta_model import build_gated_delta_model
from pretraining.nanogpt_mini.gated_delta_runtime import gdn2_dependency_provenance
from scripts.diagnose_gdn2_checkpoint import digest, distribution
from scripts.train_recurrent_slots import PackedBatches, atomic_json


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_tree(x) for x in value)
    if isinstance(value, list):
        return [clone_tree(x) for x in value]
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if value is None:
        return None
    raise TypeError(f"Unexpected cache value: {type(value)}")


def clone_cache(model, source, prefix):
    """Recreate metadata and independently clone every tensor, including convs."""
    result = model.new_cache()
    for layer in range(len(source)):
        state = clone_tree(source[layer])
        result.update(**state, layer_idx=layer, offset=prefix)
        if result.get_seq_length(layer) != source.get_seq_length(layer):
            raise RuntimeError("Cache sequence metadata changed during cloning")
        for name in ("recurrent_state", "conv_state"):
            old, new = source[layer][name], result[layer][name]
            old = old if isinstance(old, tuple) else (old,)
            new = new if isinstance(new, tuple) else (new,)
            for a, b in zip(old, new, strict=True):
                if a is not None and (a.data_ptr() == b.data_ptr() or not torch.equal(a, b)):
                    raise RuntimeError("Cache cloning aliased or changed a tensor")
    return result


class CachedForward:
    def __init__(self, model):
        self.model = model
        self.full = torch.compile(self.full_forward, fullgraph=False, dynamic=False)
        self.prefill = torch.compile(self.prefill_forward, fullgraph=False, dynamic=False)
        self.suffix = torch.compile(self.suffix_forward, fullgraph=False, dynamic=False)

    def full_forward(self, tokens):
        hidden, _ = self.model.forward_hidden(tokens)
        return self.model.logits(hidden)

    def prefill_forward(self, tokens, cache):
        _, cache = self.model.forward_hidden(tokens, state=cache, use_cache=True)
        return cache

    def suffix_forward(self, tokens, cache):
        hidden, _ = self.model.forward_hidden(tokens, state=cache, use_cache=True)
        return self.model.logits(hidden)


def parity(reference, observed, targets):
    delta = observed - reference
    ce_ref = F.cross_entropy(reference.flatten(0, 1), targets.flatten(), reduction="none")
    ce_obs = F.cross_entropy(observed.flatten(0, 1), targets.flatten(), reduction="none")
    relative_rms = float(delta.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-12))
    relative_ce = float((ce_obs.sum() - ce_ref.sum()).abs() / ce_ref.sum().abs().clamp_min(1e-12))
    return dict(logit_relative_rms_error=relative_rms, logit_absolute_error=distribution(delta.abs()),
                logit_max_absolute_error=float(delta.abs().max()), ce_relative_error=relative_ce,
                mean_token_ce_absolute_error=float((ce_obs - ce_ref).abs().mean()),
                bounds=dict(logit_relative_rms_error=.01, ce_relative_error=.001),
                passed=bool(torch.isfinite(observed).all()) and relative_rms <= .01 and relative_ce <= .001)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "ablation_results/gdn2_fullwidth_fla_kfirst_saved_1k")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.rows <= 64:
        parser.error("rows must be between 1 and 64")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required; submit through mlq")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    run = args.run_dir.resolve()
    checkpoint_path = run / "model.pt"
    checkpoint_hash = digest(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = json.loads((run / "config.json").read_text())
    training_result = json.loads((run / "result.json").read_text())
    if (training_result.get("status") != "completed" or saved.get("completed_steps") != 1000
            or training_result.get("completed_steps") != 1000 or saved.get("train_seq_len") != 1024
            or config["model_config"] != saved["model_config"]):
        raise ValueError("Require the completed matching 1000-update checkpoint")
    if saved["model_config"]["head_dim"] != 128 or saved["model_config"]["mixer_dim"] != 512:
        raise ValueError("This predefined head intervention requires the four-head full-width model")
    loader = PackedBatches(str(ROOT / config["data_path"] / "fineweb_val_*.bin"), 1048576, 1024)
    panel_x, panel_y = loader.next()
    row_ids = torch.linspace(0, 1023, args.rows, device="cuda").round().long()
    inputs, targets = panel_x[row_ids], panel_y[row_ids]
    prefix = 768
    suffix_inputs, suffix_targets = inputs[:, prefix:].contiguous(), targets[:, prefix:].contiguous()
    sources = [Path(__file__), ROOT / "scripts/diagnose_gdn2_checkpoint.py",
               ROOT / "scripts/train_recurrent_slots.py", ROOT / "pretraining/nanogpt_mini/gated_delta_model.py",
               ROOT / "pretraining/nanogpt_mini/gated_delta_runtime.py",
               ROOT / "pretraining/nanogpt_mini/nanogpt_mini_model.py"]
    sources += [p for p in (ROOT / "pretraining/gated_delta/vendor").rglob("*")
                if p.is_file() and "__pycache__" not in p.parts]
    report = dict(status="running", optimizer_updates=0, checkpoint_path=str(checkpoint_path),
                  checkpoint_sha256=checkpoint_hash, model_config=saved["model_config"],
                  training_config_sha256=digest(run / "config.json"),
                  training_result_sha256=digest(run / "result.json"),
                  rows=row_ids.tolist(), prefix_tokens=prefix, suffix_tokens=256,
                  sampled_inputs_sha256=hashlib.sha256(inputs.cpu().contiguous().numpy().tobytes()).hexdigest(),
                  sampled_targets_sha256=hashlib.sha256(targets.cpu().contiguous().numpy().tobytes()).hexdigest(),
                  validation_shard_sha256=digest(loader.files[loader.shard_index]),
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                  installed_fla=gdn2_dependency_provenance(),
                  source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
                  caveats=["Paired checkpoint intervention on sampled rows; not full-validation BPB.",
                           "Short-convolution history and suffix writes are preserved in every arm.",
                           "Slow/fast labels refer to previously observed decay-only top-layer gate statistics, not proven information horizons.",
                           "Head groups are fixed a priori: top layer heads 0,2 versus 1,3.",
                           "A harmful erasure establishes state use at this boundary, not optimal memory or a training remedy."])
    for source in sources:
        destination = args.output / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    output = args.output / "probe.json"
    atomic_json(output, report)
    try:
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.fail_on_recompile_limit_hit = True
        model = build_gated_delta_model(saved["model_config"]).cuda().eval()
        model.load_state_dict(saved["model"], strict=True)
        compiled = CachedForward(model)
        from torch._dynamo.utils import counters
        before = dict(counters["graph_break"])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            full = compiled.full(inputs)[:, prefix:].contiguous()
            prefix_cache = compiled.prefill(inputs[:, :prefix].contiguous(), model.new_cache())
            if len(prefix_cache) != len(model.blocks):
                raise RuntimeError("Incomplete prefix cache")
            for layer in range(len(prefix_cache)):
                state = prefix_cache[layer]["recurrent_state"]
                if state.ndim != 4 or state.shape[:2] != (args.rows, 4):
                    raise RuntimeError(f"Unexpected state shape: {state.shape}")
            baseline = compiled.suffix(suffix_inputs, clone_cache(model, prefix_cache, prefix))
            report["cached_vs_full_parity"] = parity(full, baseline, suffix_targets)
            atomic_json(output, report)
            if not report["cached_vs_full_parity"]["passed"]:
                raise RuntimeError("Cached/full BF16 equivalence check failed")
            baseline_ce = F.cross_entropy(baseline.flatten(0, 1), suffix_targets.flatten(), reduction="none").view_as(suffix_targets)
            report["baseline_mean_nll"] = float(baseline_ce.mean())
            report["arms"] = {}
            for name in ("all_recurrent_zero", "top_heads_0_2_zero", "top_heads_1_3_zero"):
                cache = clone_cache(model, prefix_cache, prefix)
                erased = []
                if name == "all_recurrent_zero":
                    for layer in range(len(cache)):
                        cache[layer]["recurrent_state"].zero_()
                        erased.append(dict(layer=layer, heads=[0, 1, 2, 3]))
                else:
                    heads = [0, 2] if name == "top_heads_0_2_zero" else [1, 3]
                    cache[len(cache)-1]["recurrent_state"][:, heads] = 0
                    erased.append(dict(layer=len(cache)-1, heads=heads))
                logits = compiled.suffix(suffix_inputs, cache)
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Nonfinite logits: {name}")
                ce = F.cross_entropy(logits.flatten(0, 1), suffix_targets.flatten(), reduction="none").view_as(suffix_targets)
                bins = []
                for start, end in ((0, 8), (8, 32), (32, 128), (128, 256), (0, 256)):
                    change = ce[:, start:end] - baseline_ce[:, start:end]
                    bins.append(dict(start=start, end=end, tokens=change.numel(),
                                     mean_delta_nll=float(change.mean()),
                                     mean_delta_bits_per_token=float(change.mean()) / math.log(2),
                                     per_row_mean_delta_nll=change.mean(-1).tolist(),
                                     mean_absolute_logit_delta=float((logits[:, start:end] - baseline[:, start:end]).abs().mean())))
                report["arms"][name] = dict(erased=erased, bins=bins)
                atomic_json(output, report)
        report["graph_breaks"] = {str(k): v - before.get(k, 0) for k, v in counters["graph_break"].items()
                                  if v > before.get(k, 0)}
        if digest(checkpoint_path) != checkpoint_hash:
            raise RuntimeError("Checkpoint changed during probe")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(output, report)


if __name__ == "__main__":
    main()
