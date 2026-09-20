"""Queued parity and timing check for the exact replay-path changes (2026-09-18).

Compares the legacy replay paths (unfolded LoRA scaling, head-major packed
attention concatenation, advanced-indexing action gathers) against the
current exact paths on one production-sized optimizer minibatch: the 16
longest real 10k-token responses, actor and critic with auxiliary losses and
saved optimizer state. Also qualifies the logit chunk size bitwise and probes
which fused SDPA backends serve GQA causal replay on this device.
Performance and exactness evidence only; no learning-quality claim.
"""
import gc
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "ablation_results/minicpm_exact_replay_paths_20260918"
OUT.mkdir(parents=True, exist_ok=True)
START = time.perf_counter()
import torch
import torch.nn.functional as F

from postraining import train_minicpm_vapo as trainer
from postraining.minicpm_vapo import (
    LoRAConfig,
    LoRALinear,
    MiniCPMVAPOCritic,
    MiniCPMVAPOPolicy,
    TrajectoryRecord,
    _packed_replay_attention,
    enable_packed_replay_attention,
    load_adapter_state_dict,
)
from postraining.train_minicpm_vapo import (
    configure_replay_checkpointing,
    refresh_behavior_statistics,
    update_step,
)

CHECKPOINT = ROOT / "postraining/runs/minicpm5_vapo_balanced_lr_cont200_v15/vapo_adapter_checkpoint.pt"
ROLLOUTS = ROOT / "ablation_results/minicpm_ar_production_20260908/optimized-ar.json"
REPLAY = dict(replay_token_budget=11024, replay_max_trajectories=16)
journal = (OUT / "metrics.jsonl").open("w", buffering=1)


def report(**values):
    journal.write(json.dumps(values) + "\n")
    print(json.dumps(values), flush=True)


def probe_sdpa_backends():
    """Which fused SDPA backends serve packed GQA causal replay on this GPU."""
    from torch.backends import cuda as cuda_backends
    from torch.nn.attention import SDPBackend, sdpa_kernel

    query = torch.randn(1, 16, 4096, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 2, 4096, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    params = cuda_backends.SDPAParams(query, key, value, None, 0.0, True, True)
    available = {
        "flash": bool(cuda_backends.can_use_flash_attention(params, True)),
        "efficient": bool(cuda_backends.can_use_efficient_attention(params, True)),
        "cudnn": bool(cuda_backends.can_use_cudnn_attention(params, True)),
    }
    timings = {}
    backends = {
        "flash": [SDPBackend.FLASH_ATTENTION],
        "efficient": [SDPBackend.EFFICIENT_ATTENTION],
        "cudnn": [SDPBackend.CUDNN_ATTENTION],
        "default": [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.MATH,
        ],
    }
    for name, selection in backends.items():
        grad_query = query.clone().requires_grad_()
        try:
            with sdpa_kernel(selection):
                for repeat in range(4):
                    if repeat == 1:
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                    output = F.scaled_dot_product_attention(
                        grad_query, key, value, is_causal=True, enable_gqa=True
                    )
                    output.float().sum().backward()
                torch.cuda.synchronize()
            timings[name] = (time.perf_counter() - started) / 3
        except RuntimeError as error:
            timings[name] = f"unavailable: {str(error).splitlines()[0][:160]}"
    return {"available": available, "forward_backward_seconds_4096_tokens": timings}


@torch.compiler.disable
def legacy_packed_replay_attention(
    module, query, key, value, attention_mask, dropout=0.0, scaling=None, **_
):
    if query.shape[0] != 1 or attention_mask is not None or dropout:
        raise ValueError("packed replay attention requires one unpadded sequence batch")
    boundaries = module._packed_sequence_boundaries
    if boundaries is None:
        boundaries = tuple(int(item) for item in module._packed_cu_seqlens.tolist())
    outputs = []
    for start, stop in zip(boundaries, boundaries[1:]):
        query_slice = query[:, :, start:stop]
        key_slice = key[:, :, start:stop]
        value_slice = value[:, :, start:stop]
        outputs.append(
            F.scaled_dot_product_attention(
                query_slice,
                key_slice,
                value_slice,
                dropout_p=0.0,
                is_causal=True,
                scale=scaling,
                enable_gqa=query_slice.shape[1] != key_slice.shape[1],
            )
        )
    return torch.cat(outputs, dim=2).transpose(1, 2), None


def legacy_action_rows(hidden, batch):
    return hidden[batch.action_batch_indices, batch.action_positions]


EXACT_ACTION_ROWS = trainer._packed_action_rows


def select_paths(sides, *, legacy: bool):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    for side in sides:
        implementation = side.causal_lm._packed_replay_attention_implementation
        ALL_ATTENTION_FUNCTIONS.register(
            implementation,
            legacy_packed_replay_attention if legacy else _packed_replay_attention,
        )
        for module in side.causal_lm.modules():
            if isinstance(module, LoRALinear):
                module._fold_scaling = (not legacy) and module.scaling == 2.0
    trainer._packed_action_rows = legacy_action_rows if legacy else EXACT_ACTION_ROWS


def build_models():
    with torch.serialization.safe_globals([TrajectoryRecord]):
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    actor = saved["policy"]["actor"]
    config = LoRAConfig(**actor["lora_config"])
    policy, _ = MiniCPMVAPOPolicy.from_pretrained(
        model_id=actor["model_id"], revision=actor["revision"], device=torch.device("cuda"),
        lora_config=config, nextlat_projection_factor=float(actor["nextlat_projection_factor"]),
        gradient_checkpointing=True,
    )
    critic = MiniCPMVAPOCritic.from_pretrained(
        model_id=actor["model_id"], revision=actor["revision"], device=torch.device("cuda"),
        lora_config=config, critic_width=int(saved["args"]["critic_width"]),
        nextlat_projection_factor=float(actor["nextlat_projection_factor"]),
        gradient_checkpointing=True, shared_frozen_source=policy.causal_lm,
    )
    load_adapter_state_dict(policy.causal_lm, actor["adapter"])
    policy.nextlat_head.load_state_dict(actor["nextlat"])
    critic_payload = saved["policy"]["critic"]
    load_adapter_state_dict(critic.causal_lm, critic_payload["adapter"])
    critic.value_head.load_state_dict(critic_payload["value_head"])
    critic.nextlat_head.load_state_dict(critic_payload["nextlat"])
    for side in (policy, critic):
        enable_packed_replay_attention(side.causal_lm, backend="sdpa")
        configure_replay_checkpointing(side.causal_lm, 0)
    return saved, policy, critic


def build_records():
    source = json.loads(ROLLOUTS.read_text())
    responses = sorted(
        source["repetitions"][0]["math_outcomes"]["responses"],
        key=lambda row: row["response_tokens"],
        reverse=True,
    )[:16]
    records = []
    for response in responses:
        prompt = source["prompt_token_ids"][response["prompt_index"]]
        tokens = prompt + response["token_ids"]
        count = len(response["token_ids"])
        records.append(
            TrajectoryRecord.from_device(
                token_ids=torch.tensor(tokens, dtype=torch.int32), prompt_length=len(prompt),
                old_logprobs=torch.zeros(count), old_values=torch.zeros(count),
                correct=response["correct"], text=response["text"],
            )
        )
    return records


def timed_refresh(policy, critic, records, *, chunk, label):
    torch.cuda.synchronize()
    started = time.perf_counter()
    refreshed, metrics = refresh_behavior_statistics(
        policy, critic, records, logit_chunk_tokens=chunk, **REPLAY
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    report(check="behavior_refresh", label=label, logit_chunk_tokens=chunk, seconds=seconds,
           tokens=sum(record.response_length for record in refreshed))
    return refreshed, seconds


def maximum_difference(first, second):
    """Largest behavior difference: log-probs and the GAE advantages built from values."""
    return max(
        max((a.old_logprobs.float() - b.old_logprobs.float()).abs().max().item(),
            (a.advantages.float() - b.advantages.float()).abs().max().item())
        for a, b in zip(first, second, strict=True)
    )


def main():
    report(check="sdpa_backends", **probe_sdpa_backends())
    saved, policy, critic = build_models()
    records = build_records()
    sides = (policy, critic)
    select_paths(sides, legacy=False)
    records_128, refresh_128 = timed_refresh(policy, critic, records, chunk=128, label="exact")
    records_1024, refresh_1024 = timed_refresh(policy, critic, records, chunk=1024, label="exact")
    chunk_difference = maximum_difference(records_128, records_1024)
    report(check="logit_chunk_parity", chunk_128_seconds=refresh_128, chunk_1024_seconds=refresh_1024,
           maximum_behavior_difference=chunk_difference)
    records = records_128

    parameters = [
        parameter for side in sides for _, parameter in side.named_parameters() if parameter.requires_grad
    ]
    initial = [parameter.detach().cpu().clone() for parameter in parameters]
    trials = {}
    schedule = (("legacy", 0), ("exact", 0), ("exact", 1), ("legacy", 1))
    for label, repeat in schedule:
        torch._functorch.config.donated_buffer = False
        with torch.no_grad():
            for parameter, value in zip(parameters, initial, strict=True):
                parameter.copy_(value)
        for side in sides:
            side.zero_grad(set_to_none=True)
            configure_replay_checkpointing(side.causal_lm, 0)
        select_paths(sides, legacy=label == "legacy")
        actor_optimizer = torch.optim.AdamW(
            [
                {"params": list(policy.actor_parameters()), "lr": 1e-6},
                {"params": list(policy.nextlat_head.parameters()), "lr": 1e-6, "weight_decay": 0.1, "betas": (0.9, 0.95)},
            ],
            fused=True, weight_decay=0,
        )
        critic_optimizer = torch.optim.AdamW(
            [
                {"params": list(critic.backbone_parameters()) + list(critic.value_head.parameters()), "lr": 1e-5},
                {"params": list(critic.nextlat_head.parameters()), "lr": 1e-5, "weight_decay": 0.1, "betas": (0.9, 0.95)},
            ],
            fused=True, weight_decay=0,
        )
        actor_optimizer.load_state_dict(saved["actor_optimizer"])
        critic_optimizer.load_state_dict(saved["critic_optimizer"])
        random.seed(177)
        torch.manual_seed(177)
        torch.cuda.manual_seed_all(177)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        metrics = update_step(
            policy, critic, records, actor_optimizer, critic_optimizer,
            optimizer_minibatches=1, logit_chunk_tokens=128, clip_low=0.2, clip_high=0.28,
            value_coefficient=1.0, nextlat_horizon=2, nextlat_samples=64, nextlat_mse_coefficient=1.0,
            nextlat_kl_coefficient=1.0, nextlat_kl_chunk_tokens=16, train_nextlat=True,
            grad_clip_norm=1.0, nextlat_trunk_balance="parameter", **REPLAY,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        trials[(label, repeat)] = {
            "seconds": seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "parameters": [parameter.detach().cpu().clone() for parameter in parameters],
            "policy_loss": metrics.get("policy_loss"),
            "value_loss": metrics.get("value_loss"),
        }
        report(check="update_complete", label=label, repeat=repeat, seconds=seconds,
               peak_allocated_bytes=trials[(label, repeat)]["peak_allocated_bytes"])
        del actor_optimizer, critic_optimizer
        gc.collect()

    def error(a, b):
        return max((x - y).abs().max().item() for x, y in zip(trials[a]["parameters"], trials[b]["parameters"], strict=True))

    summary = {
        "schema": "minicpm_exact_replay_paths/v1",
        "candidate_parameter_maximum_error": error(("legacy", 0), ("exact", 0)),
        "legacy_repeat_parameter_maximum_error": error(("legacy", 0), ("legacy", 1)),
        "exact_repeat_parameter_maximum_error": error(("exact", 0), ("exact", 1)),
        "update_seconds": {f"{label}_{repeat}": trial["seconds"] for (label, repeat), trial in trials.items()},
        "update_speedup": (
            (trials[("legacy", 0)]["seconds"] + trials[("legacy", 1)]["seconds"])
            / (trials[("exact", 0)]["seconds"] + trials[("exact", 1)]["seconds"])
        ),
        "peak_allocated_bytes": {f"{label}_{repeat}": trial["peak_allocated_bytes"] for (label, repeat), trial in trials.items()},
        "behavior_refresh_seconds": {"chunk_128": refresh_128, "chunk_1024": refresh_1024},
        "logit_chunk_maximum_behavior_difference": chunk_difference,
        "experiment_seconds": time.perf_counter() - START,
        "scope": (
            "Exact replay paths (power-of-two LoRA scale fold with in-place add, token-major packed "
            "attention concatenation, index_select action gathers) versus the legacy paths on one "
            "production-sized optimizer minibatch with saved optimizer state; chunk-size parity of "
            "behavior refresh; SDPA backend availability for GQA causal replay. No learning claim."
        ),
        "reproduction_source": Path(__file__).read_text(),
    }
    (OUT / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report(check="complete", **{key: value for key, value in summary.items() if key not in {"reproduction_source", "scope"}})


if __name__ == "__main__":
    main()
